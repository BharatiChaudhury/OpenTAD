_base_ = [
    "../_base_/models/causaltad.py",
]

dataset_type = "ThumosSlidingDataset"

annotation_path = "/srv/storage/stars@storage3.sophia.grid5000.fr/bchaudhu/projects/OpenTAD/tools/mpii_tridet.json"

class_map = "/srv/storage/stars@storage3.sophia.grid5000.fr/bchaudhu/projects/OpenTAD/tools/category_idx.txt"

data_path = "/srv/storage/stars@storage3.sophia.grid5000.fr/bchaudhu/projects/Introducing-Gating-and-Context-into-Temporal-Action-Detection/features_npy_v2_check"

window_size = 256


dataset = dict(
    train=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="training",
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,

        feature_stride=1,
        sample_stride=1,
        window_size=window_size,
        window_overlap_ratio=0.25,

        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),

            dict(
                type="ConvertToTensor",
                keys=["feats", "gt_segments", "gt_labels"]
            ),

            dict(
                type="SlidingWindowTrunc",
                with_mask=True
            ),

            dict(
                type="Rearrange",
                keys=["feats"],
                ops="t c -> c t"
            ),

            dict(
                type="Collect",
                inputs="feats",
                keys=["masks", "gt_segments", "gt_labels"]
            ),
        ],
    ),

    val=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="validation",
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,

        feature_stride=1,
        sample_stride=1,
        window_size=window_size,
        window_overlap_ratio=0.25,

        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),

            dict(
                type="ConvertToTensor",
                keys=["feats", "gt_segments", "gt_labels"]
            ),

            dict(
                type="SlidingWindowTrunc",
                with_mask=True
            ),

            dict(
                type="Rearrange",
                keys=["feats"],
                ops="t c -> c t"
            ),

            dict(
                type="Collect",
                inputs="feats",
                keys=["masks", "gt_segments", "gt_labels"]
            ),
        ],
    ),

    test=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="validation",
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,

        feature_stride=1,
        sample_stride=1,
        window_size=window_size,
        window_overlap_ratio=0.5,

        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),

            dict(
                type="ConvertToTensor",
                keys=["feats"]
            ),

            dict(
                type="SlidingWindowTrunc",
                with_mask=True
            ),

            dict(
                type="Rearrange",
                keys=["feats"],
                ops="t c -> c t"
            ),

            dict(
                type="Collect",
                inputs="feats",
                keys=["masks"]
            ),
        ],
    ),
)


evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
)


model = dict(
    projection=dict(
        in_channels=768,
        input_pdrop=0.1,
    ),
)


solver = dict(
    train=dict(
        batch_size=2,
        num_workers=2
    ),

    val=dict(
        batch_size=2,
        num_workers=2
    ),

    test=dict(
        batch_size=2,
        num_workers=2
    ),

    clip_grad_norm=1,
    ema=True,
    amp=True,
)


optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
)


scheduler = dict(
    type="LinearWarmupCosineAnnealingLR",
    warmup_epoch=5,
    max_epoch=50,
)


inference = dict(
    load_from_raw_predictions=False,
    save_raw_prediction=False,
)


post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=2000,
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.7,
    ),

    save_dict=False,
)


workflow = dict(
    logging_interval=20,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=5,
)


work_dir = "exps/mpii/causaltad_videomaev2"

