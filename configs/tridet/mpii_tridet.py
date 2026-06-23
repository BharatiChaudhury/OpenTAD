_base_ = [
    "../_base_/datasets/mpii/features_mpii.py",  # dataset config
    "../_base_/models/tridet.py",  # model config
]

model = dict(
    projection=dict(
        in_channels=768,
        out_channels=256,
        sgp_win_size=[1, 1, 1, 1, 1, 1],
        k=5,
        sgp_mlp_dim=256,
        use_abs_pe=False,
        max_seq_len=256,
        init_conv_vars=0,
        input_noise=0.0,
    ),
    #neck=dict(in_channels=256, out_channels=256),
    neck=dict(
    type="MambaHybridNeck",
    in_channels=256,
    out_channels=256,
    num_levels=6,
    ),
    rpn_head=dict(
        in_channels=256,
        feat_channels=256,
        boundary_kernel_size=3,
        num_classes=19,
        label_smoothing=0.1,
        loss_normalizer=400,
        iou_weight_power=1,
        num_bins=16
    ),
)
solver = dict(
    train=dict(batch_size=4, num_workers=2),
    val=dict(batch_size=2, num_workers=1),
    test=dict(batch_size=2, num_workers=1),
    clip_grad_norm=1,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.03, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=30, eta_min=5e-4)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.75,
        max_seg_num=250,
        iou_threshold=0,  # does not matter when use soft nms
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.95,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=5,
)

work_dir = "exps/mpii/tridet_baseline"
