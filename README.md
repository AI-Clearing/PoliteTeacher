# [Polite Teacher: Semi-Supervised Instance Segmentation with Mutual Learning and Pseudo-Label Thresholding]()

We present Polite Teacher, a simple yet effective method for the task of semi-supervised instance segmentation.
The proposed architecture relies on the Teacher-Student mutual learning framework.
To filter out noisy pseudo-labels, we use confidence thresholding for bounding boxes and mask scoring for masks.
The approach has been tested with CenterMask, a single-stage anchor-free detector.
Tested on the COCO 2017 val dataset, our architecture significantly (approx. +8 pp. in mask AP) outperforms the baseline at different supervision regimes.
To the best of our knowledge, this is one of the first works tackling the problem of semi-supervised instance segmentation and the first one devoted to an anchor-free detector. 

# Data
Download COCO 2017...
```
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```
...and organise it as follows:
```
polite_teacher/
└── datasets/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
```

# Environment
To create the environment with `conda`, run the following:
```
conda create -n centermask2 python=3.9
conda activate centermask2
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install setuptools==59.5.0
```

# Running experiments

First, train the baseline Centermask model (for example, `baseline_centermask_R_50_FPN_ms_3x_sup10.yaml` for 10% supervision):
```python
conda activate centermask2
python train_net.py --num-gpus=8 --config configs/polite_teacher/baseline_centermask_R_50_FPN_ms_3x_sup10.yaml
```
Then, include the selected one in the further experiments as `MODEL.WEIGHTS` in `polite_teacher_centermask_R_50_FPN_ms_3x_sup10.yaml`:
```yaml
MODEL:
  WEIGHTS: "output/centermask/baseline_centermask_R_50_FPN_ms_3x_sup10/model_XXX.pth"
```
Now you can run the main experiment:
```python
python train_net.py --num-gpus=8 --config configs/polite_teacher/polite_teacher_centermask_R_50_FPN_ms_3x_sup10.yaml
```
Add ` --resume` to resume the training.
All the configs needed to reproduce the paper results are in the `configs` directory.
You may want to check `predict_script.py` to visualise some predictions.

# Credits
Our code is based on the code of:
- [detectron2](https://github.com/facebookresearch/detectron2),
- [CenterMask2](https://github.com/youngwanLEE/CenterMask2),
- [Unbiased Teacher](https://github.com/facebookresearch/unbiased-teacher).
