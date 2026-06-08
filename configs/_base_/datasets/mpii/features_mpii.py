dataset_type = "MPII_Group_Interaction"
annotation_path = "/srv/storage/stars@storage3.sophia.grid5000.fr/bchaudhu/projects/OpenTAD/tools/mpii_tridet.json"
class_map = "/srv/storage/stars@storage3.sophia.grid5000.fr/bchaudhu/projects/OpenTAD/tools/category_idx.txt"
data_path = "/srv/storage/stars@storage3.sophia.grid5000.fr/bchaudhu/projects/Introducing-Gating-and-Context-into-Temporal-Action-Detection/features_npy_v2_check"
block_list = None

window_size = 16
dataset = dict(
    train=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="training",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # mpii dataloader setting
        feature_stride=32,
        sample_stride=1,  # 1x4=4
        window_size=window_size,
        window_overlap_ratio=0.25,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # thumos dataloader setting
        feature_stride=1,
        sample_stride=5,  # 1x4=4
        window_size=window_size,
        window_overlap_ratio=0.25,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        # thumos dataloader setting
        feature_stride=1,
        sample_stride=5,  # 1x4=4
        window_size=window_size,
        window_overlap_ratio=0.5,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)


evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
)
