import datetime
import os
import yaml
import time
import torch
import shutil
import random
import argparse
import subprocess
import numpy as np
import importlib



import torch.multiprocessing as mp
from pathlib import Path
from torch.utils import data
import torch.distributed as distrib
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DistributedSampler

# 引用数据加载器
from dataloader.cross_view_dataloader import DatasetLoader


# 引用基本工具
from mmengine.config import Config
from utils.semmap_loss import SemmapLoss
from metric import averageMeter
from metric.iou import IoU
from utils.logger import get_logger


def cosine_annealing_weight(epoch, total_epochs, w_max=1.0, w_min=0):
    return w_min + 0.5 * (w_max - w_min) * (1 + np.cos(np.pi * epoch / total_epochs))

def load_model_class(model_type: str, model_name: str):
    """
    根据模型名字动态加载对应类
    模型都放在 models/network
    """
    module_path = f"models.network.{model_type}"   # 动态拼接路径
    module = importlib.import_module(module_path)  # 动态导入模块
    model_class = getattr(module, model_name)      # 获取类
    return model_class

def train(rank, world_size, cfg):
    # 设置种子随机数
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # 初始化分布式计算
    master_port = int(os.environ.get("MASTER_PORT", 8750))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    tcp_store = torch.distributed.TCPStore(
        master_addr, master_port, world_size, rank == 0
    )
    torch.distributed.init_process_group(
        'nccl', store=tcp_store, rank=rank, world_size=world_size
    )

    ################################################## Setup device #####################################################
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
    else:
        assert world_size == 1
        device = torch.device("cpu")

    if rank == 0:
        writer = SummaryWriter(log_dir=cfg["logdir"])
        logger = get_logger(cfg["logdir"])

        print('**log_dir:', cfg["logdir"])
        logger.info("Let training begin !!")

    t_loader = DatasetLoader(cfg["data"], split=cfg['data']['train_split'])
    v_loader = DatasetLoader(cfg['data'], split=cfg["data"]["val_split"])

    t_sampler = DistributedSampler(t_loader)
    v_sampler = DistributedSampler(v_loader, shuffle=False)

    if rank == 0:
        print('#Envs in train: %d' % (len(t_loader.files_sate)))
        print('#Envs in val: %d' % (len(v_loader.files_sate)))

    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training_setting"]["batch_size"] // world_size,
        num_workers=cfg["training_setting"]["n_workers"],
        drop_last=True,
        pin_memory=True,
        sampler=t_sampler,
        multiprocessing_context='fork',
    )

    valloader = data.DataLoader(
        v_loader,
        batch_size=cfg["training_setting"]["batch_size"] // world_size,
        num_workers=cfg["training_setting"]["n_workers"],
        pin_memory=True,
        drop_last=True,
        sampler=v_sampler,
        multiprocessing_context='fork',
    )

    ################################################### Setup Model ######################################################

    model_type = cfg['model_type']
    model_name = cfg['model_name']
    Model = load_model_class(model_type, model_name)
    model = Model(cfg['model'], device)  # 动态实例化

    model = model.to(device)

    if device.type == 'cuda':
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])


    # Setup optimizer, lr_scheduler and loss function
    optimizer_params = {k: v for k, v in cfg["training_setting"]["optimizer"].items() if k != "name"}
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_params)

    if rank == 0:
        logger.info("Using optimizer {}".format(optimizer))

    lr_decay_lambda = lambda epoch: cfg['training_setting']['scheduler']['lr_decay_rate'] ** (
                epoch // cfg["training_setting"]['scheduler']['lr_epoch_per_decay'])
    scheduler = LambdaLR(optimizer, lr_lambda=lr_decay_lambda)

    # Setup Metrics
    # 初始化运行中指标
    obj_running_metrics = IoU(cfg['model']['n_obj_classes'])
    obj_running_metrics_val = IoU(cfg['model']['n_obj_classes'])
    obj_running_metrics.reset()
    obj_running_metrics_val.reset()
    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    # setup Loss
    loss_fn = SemmapLoss()
    loss_fn = loss_fn.to(device=device)

    if rank == 0:
        logger.info("Using loss {}".format(loss_fn))

    start_iter = 0
    start_epoch = 0
    best_iou = -100.0

    if cfg["training_setting"]["resume"] is not None:
        if os.path.isfile(cfg["training_setting"]["resume"]):
            if rank == 0:
                logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(cfg["training_setting"]["resume"])
                )
            checkpoint = torch.load(cfg["training_setting"]["resume"], map_location="cpu")
            model_state = checkpoint["model_state"]
            model.load_state_dict(model_state)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"]
            start_iter = checkpoint["iter"]
            best_iou = checkpoint['best_iou']
        else:
            if rank == 0:
                logger.info("No checkpoint found at '{}'".format(cfg["training_setting"]["resume"]))

    elif cfg['training_setting']['load_model'] is not None:
        checkpoint = torch.load(cfg["training_setting"]["load_model"], map_location="cpu")
        model_state = checkpoint['model_state']
        model.load_state_dict(model_state)
        if rank == 0:
            logger.info("Loading model and optimizer from checkpoint '{}'".format(cfg["training_setting"]["load_model"]))

    if rank == 0:
        print('# trainable parameters = ', params)
        print("Let Training Begin !!")
        log_dir = cfg["logdir"]
        command = f"tensorboard --logdir={log_dir}"
        # 通过 subprocess 启动 TensorBoard
        subprocess.Popen(command, shell=True)

    ####################################################################################################################
    iter = start_iter
    torch.autograd.set_detect_anomaly(True)

    bev_size = cfg['model']['bev_size']




    for epoch in range(start_epoch, cfg["training_setting"]["epochs"], 1):

        t_sampler.set_epoch(epoch)

        for batch in trainloader:

            iter += 1
            start_ts = time.time()
            img_sate_norm, img_svi_norm, semmap_gt, file_sate = batch

            model.train()
            optimizer.zero_grad()    # 只在第一次迭代时记录计算图


            semmap_pred, observed_masks = model(img_svi_norm, img_sate_norm)

            semmap_gt = semmap_gt.long()

            semmap_gt = semmap_gt.to(device)
            
            if observed_masks.any():
                
                w_cos = cosine_annealing_weight(epoch, cfg["training_setting"]["epochs"])
                
                loss = loss_fn(semmap_gt, semmap_pred, observed_masks)

 
                loss.backward()
                optimizer.step()

                semmap_pred = semmap_pred.permute(0, 2, 3, 1)
                masked_semmap_gt = semmap_gt[observed_masks]
                masked_semmap_pred = semmap_pred[observed_masks]
                
                obj_gt = masked_semmap_gt.detach()
                obj_pred = masked_semmap_pred.data.max(-1)[1].detach()
                obj_running_metrics.add(obj_pred, obj_gt)

                    
            time_meter.update(time.time() - start_ts)

            if (iter % cfg["training_setting"]["print_interval"] == 0):
                conf_metric = obj_running_metrics.conf_metric.conf
                conf_metric = torch.FloatTensor(conf_metric)
                conf_metric = conf_metric.to(device)
                distrib.all_reduce(conf_metric)
                distrib.all_reduce(loss)

                loss /= world_size


                if (rank == 0):
                    conf_metric = conf_metric.cpu().numpy()
                    conf_metric = conf_metric.astype(np.int32)
                    tmp_metrics = IoU(cfg['model']['n_obj_classes'])
                    tmp_metrics.reset()
                    tmp_metrics.conf_metric.conf = conf_metric
                    per_class_IoU, mIoU, acc, _, mRecall, _, mPrecision = tmp_metrics.value()

                    writer.add_scalar("train_metrics/mIoU", mIoU, iter)
                    writer.add_scalar("train_metrics/mRecall", mRecall, iter)
                    writer.add_scalar("train_metrics/mPrecision", mPrecision, iter)
                    writer.add_scalar("train_metrics/Overall_Acc", acc, iter)
                    writer.add_scalar("train_loss/loss", loss, iter)

                            # 新增按类别记录IoU
                    for class_idx, class_iou in enumerate(per_class_IoU):
                        writer.add_scalar(f"train_metrics/Class_{class_idx}_IoU", class_iou, iter)


                    fmt_str = "Iter: {:d} == Epoch [{:d}/{:d}] == Loss: {:.4f} == mIoU: {:.4f} == mRecall:{:.4f} == mPrecision:{:.4f} == Overall_Acc:{:.4f} == lr:{:.4f} == Time: {:.4f}s"
                    logger.info(
                        fmt_str.format(iter, epoch, cfg["training_setting"]["epochs"], loss.item(), mIoU, mRecall, mPrecision, acc,
                                       optimizer.param_groups[0]["lr"], time_meter.avg))


        if (epoch + 1) % cfg["training_setting"]["val_interval"] == 0:
            if rank == 0:
                logger.info("Val on epoch {}".format(epoch + 1))
            val_loss_meter = averageMeter()
            val_metrics = IoU(cfg['model']['n_obj_classes'])
            val_metrics.reset()

            model.eval()
            with torch.no_grad():
                for val_batch in valloader:
                    img_sate_norm, img_svi_norm, semmap_gt, file_sate = val_batch

                    semmap_pred, observed_masks = model(img_svi_norm, img_sate_norm)

                    semmap_gt = semmap_gt.long().to(device)

                    if observed_masks.any():
                        loss = loss_fn(semmap_gt, semmap_pred, observed_masks)
                        val_loss_meter.update(loss.item())

                        semmap_pred = semmap_pred.permute(0, 2, 3, 1)
                        masked_semmap_gt = semmap_gt[observed_masks]
                        masked_semmap_pred = semmap_pred[observed_masks]

                        obj_gt = masked_semmap_gt.detach()
                        obj_pred = masked_semmap_pred.data.max(-1)[1].detach()
                        val_metrics.add(obj_pred, obj_gt)

                # 计算验证集指标
                val_conf_metric = val_metrics.conf_metric.conf
                val_conf_metric = torch.FloatTensor(val_conf_metric).to(device)
                distrib.all_reduce(val_conf_metric)
                val_conf_metric = val_conf_metric.cpu().numpy()
                val_conf_metric = val_conf_metric.astype(np.int32)

                tmp_metrics = IoU(cfg['model']['n_obj_classes'])
                tmp_metrics.reset()
                tmp_metrics.conf_metric.conf = val_conf_metric
                per_class_val_IoU, val_mIoU, val_acc, _, val_mRecall, _, val_mPrecision = tmp_metrics.value()

                # 记录验证集指标
                if rank == 0:
                    writer.add_scalar("val_metrics/mIoU", val_mIoU, iter)
                    writer.add_scalar("val_metrics/mRecall", val_mRecall, iter)
                    writer.add_scalar("val_metrics/mPrecision", val_mPrecision, iter)
                    writer.add_scalar("val_metrics/Overall_Acc", val_acc, iter)
                    # 新增按类别记录IoU
                    for class_idx, class_iou in enumerate(per_class_val_IoU):
                        writer.add_scalar(f"val_metrics/Class_{class_idx}_IoU", class_iou, iter)

                    # 保存最佳模型
                    if val_mIoU > best_iou:
                        best_iou = val_mIoU
                        state = {
                            "epoch": epoch + 1,
                            "iter": iter + 1,
                            "best_iou": best_iou,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                        }
                        save_path = os.path.join(
                            writer.file_writer.get_logdir(),
                            "{}_best_model.pkl".format(cfg["name_experiment"]),
                        )
                        torch.save(state, save_path)

                    state = {
                        "epoch": epoch,
                        "iter": iter,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "latest_iou": val_mIoU,
                    }
                    save_path = os.path.join(cfg['checkpoint_dir'], "ckpt_latest_model.pkl")
                    torch.save(state, save_path)

        val_loss_meter.reset()
        obj_running_metrics_val.reset()
        obj_running_metrics.reset()


        scheduler.step()

if __name__ == "__main__":

    # 输入参数
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/ST.py",
        help="Configuration file to use",
    )


    # 解析参数，读取配置文件
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

        

    name_expe = cfg['name_experiment']

    run_id = random.randint(1, 100000)
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    run_id = nowTime

    log_cfg = cfg['log_dir']
    checkpoint_cfg = cfg['checkpoint_dir']

    logdir = os.path.join(log_cfg, name_expe, str(run_id))
    chkptdir = os.path.join(checkpoint_cfg, name_expe, str(run_id))


    cfg['checkpoint_dir'] = chkptdir
    cfg['logdir'] = logdir
    

    print("RUNDIR: {}".format(logdir))
    Path(logdir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, logdir)

    print("CHECKPOINTDIR: {}".format(chkptdir))
    Path(chkptdir).mkdir(parents=True, exist_ok=True)

    world_size = cfg['training_setting']['world_size']
    mp.spawn(train,
             args=(world_size, cfg),
             nprocs=world_size,
             join=True)