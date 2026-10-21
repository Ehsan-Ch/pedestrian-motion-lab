# Attribution and licenses

## KITTI

Images and annotations: [KITTI Object Tracking Evaluation](https://www.cvlibs.net/datasets/kitti/eval_tracking.php), Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago. Citation: Andreas Geiger, Philip Lenz and Raquel Urtasun, *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite*, CVPR 2012. The dataset's camera streams are nominally 10 Hz; see [KITTI raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php).

Source archives: `https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip` and `https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip`. The committed report records individual original PNG hashes and the label archive hash. Training labels are used for a descriptive sample evaluation; this project does not submit results to KITTI's leaderboard.

KITTI data is [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) and intended for academic/noncommercial use. Dataset-derived material—including `reports/images/`, annotations within `reports/results.json`, and screenshots containing the imagery—retains these terms, separate from the software license. Modifications: JPEG compression, selection of consecutive excerpts, and overlaid track IDs/paths/predictions in the report. No endorsement is implied.

## Detector

YOLOX-S pretrained on COCO, distributed by [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/object_detection_yolox), commit `47534e27c9851bb1128ccc0102f1145e27f23f98`. Model SHA-256: `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`. Model/adapter license: Apache-2.0; see `models/LICENSE-YOLOX.txt`. The adapter and hash-checked setup script are reused from [Streetwise](https://github.com/Ehsan-Ch/pedestrian-perception-lab) by Ehsan Cheraghi. The 35.9 MB weights are downloaded separately, not committed.

Original model paper: Zheng Ge et al., [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430). No model training or fine-tuning was performed here.

## Tracking methods and software

Conceptual reference: Alex Bewley et al., [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763), ICIP 2016. No source from the GPL SORT repository is incorporated. Our state parameterization, matching implementation and evaluation are original and are not the official SORT implementation.

OpenCV is Apache-2.0; NumPy is BSD-3-Clause. Original tracking/evaluation/UI code is MIT. The detector adapter remains Apache-2.0. Data utilities and matching functions were adapted from the author's MIT-licensed Streetwise project. Each dependency retains its upstream notices. Software development and testing were AI-assisted.
