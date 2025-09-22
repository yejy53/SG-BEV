# 实验设置
name_experiment = "ST"

model_type = "ST"

model_name = "ST"

checkpoint_dir = "checkpoints"

log_dir = "runs"

bev_size = 256

model = dict(
    n_obj_classes=7,
    bev_size = bev_size,
    sate_size = 256,
    batch_size_every_processer = 1,
    mem_feature_dim = 128,
    neck_output_dim = 256,
    branch_output_dim = 128,
    decoder_dim = 256,

    backbone = dict(
            type='mscan',
            embed_dims=[64, 128, 320, 512],
            mlp_ratios=[8, 8, 4, 4],
            drop_rate=0.0,
            drop_path_rate=0.1,
            depths=[3, 3, 12, 3],
            norm_cfg=dict(
                type='SyncBN',
                requires_grad=True
            ),
            init_cfg=dict(
                type='Pretrained',
                checkpoint='./checkpoints/mscan_b.pth'
            )
    ),

    ST_config = dict(
        img_height = 256,
        img_weight = 256,
        uv_height = 32,
        uv_weight = 32,
        fov = 170,
        dty = 0,
        dx = 0,
        dy = 0
    )
)

data = dict(
    # 对应城市的路径
    root = '/data/cross-view-datasets/segment/Newyork',
    train_split = 'train',
    val_split = 'val',
    test_split = 'val',
    # 文件夹名称，大部分时候不用改
    sate_folder = 'images/sate',
    svi_folder = 'images/svi',
    gt_folder = 'gt'
    )

training_setting = dict(
    epochs = 40,
    # Please also modify batch_size_every_processer 
    # in the model model config dict.
    batch_size_every_processer = 5,
    # batchsize = word_szie * batch_size_every_processer
    batch_size = 20,
    world_size = 4,
    n_workers = 28,
    print_interval = 10,
    val_interval = 1,
    resume = None,
    load_model = None,
    optimizer = dict(
                    lr = 0.00005,
                    betas = (0.9, 0.999),
                    weight_decay = 0.01
    ),
    scheduler = dict(
                        lr_decay_rate= 0.7,
                        lr_epoch_per_decay=10
    )
)

