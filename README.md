# Inferring Attention Shift Ranks of Objects for Image Saliency [CVPR 2020]

### Authors:
Avishek Siris, Jianbo Jiao, Gary K.L. Tam, Xianghua Xie, Rynson W.H. Lau

+ PDF: [Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Siris_Inferring_Attention_Shift_Ranks_of_Objects_for_Image_Saliency_CVPR_2020_paper.pdf)
+ Supplemental: [Supplementary Material](http://openaccess.thecvf.com/content_CVPR_2020/supplemental/Siris_Inferring_Attention_Shift_CVPR_2020_supplemental.pdf)

##
<p align="center">
<img src="https://github.com/SirisAvishek/Inferring-Attention-Shift-Ranks/blob/master/images/saliency_rank_data_compare.png" width="800"/>
</p>

We propose a new large-scale saliency rank dataset from attention shift, motivated by psychological and behavioural studies, and justified by our user study.

## Abstract
Psychology studies and behavioural observation show that humans shift their attention from one location to another when viewing an image of a complex scene. This is due to the limited capacity of the human visual system in simultaneously processing multiple visual inputs. The sequential shifting of attention on objects in a non-task oriented viewing can be seen as a form of saliency ranking. Although there are methods proposed for predicting saliency rank, they are not able to model this human attention shift well, as they are primarily based on ranking saliency values from binary prediction. Following psychological studies, in this paper, we propose to predict the saliency rank by inferring human attention shift. Due to the lack of such data, we first construct a large-scale salient object ranking dataset. The saliency rank of objects is defined by the order that an observer attends to these objects based on attention shift. The final saliency rank is an average across the saliency ranks of multiple observers. We then propose a learning-based CNN to leverage both bottom-up and top-down attention mechanisms to predict the saliency rank. Experimental results show that the proposed network achieves state-of-the-art performances on salient object rank prediction. 

## Installation
The code is based on the Mask-RCNN implementation by matterport, https://github.com/matterport/Mask_RCNN. Please follow and install the requirements they list. Additionally, install pycocotools.   

## Dataset
Download our dataset from [google drive](https://drive.google.com/file/d/1ueSpf3avLAPiJxoP40v5KL7qxaYtM1us/view?usp=sharing).

## Training 
The current implementation and results are based on pre-computing backbone and object features, then training the rest of the saliency rank model seperately. 

1. Pre-train backbone for salient object detection (binary, no ranking). Download pre-trained COCO weights (mask_rcnn_coco.h5) from matterport, https://github.com/matterport/Mask_RCNN/releases. Put the weight file in the "weights/" folder. Set data paths and run:
```
python obj_sal_seg_branch/train.py
```

2. Pre-compute backbone and object features of GT objects for "train" and "val" datasplits. Set data paths and run twice for "train" and "val":
```
python pre_process/pre_process_obj_feat_GT.py
```

3.  Finally train the saliency rank model. Set data paths and run:
```
python train.py
```

## Testing
Similarly to training, we need to pre-compute features.

1. Perform object detection. Set data paths and run:
```
python pre_process/object_detection.py
```

2. Pre-compute corresponding object features. Set data paths and run:
```
python pre_process/pre_process_obj_feat.py
```

3. Test the saliency rank model. You can download the weights of the trained model from [google drive](https://drive.google.com/file/d/1fXFGvrS7aMd5FagM9n7-VaJPxRrPXbXN/view?usp=sharing).
Set data paths and run:
```
python evaluation/predict.py
```
This will generate predictions and save into files.

4. Generate predicted Saliency Rank Maps (rank based on grayscale value). Set data paths and run:
```
python evaluation/generate_saliency_map.py
```

## Evaluate
To evaluate MAE and Salient Object Ranking (SOR) scores, set data paths and run:
```
python evaluation/evaluate.py
```

For further mAP evaluation, set data paths and run:
```
python evaluation/evaluate_map.py
```

## Download Pre-computed Features
You can download pre-computed features for training and testing the model from [google drive](https://drive.google.com/file/d/1r1d9q4SACIu1oasC7HCiPYs_NJD_2vWP/view?usp=sharing).

## Results
You can download predicted Saliency Rank Maps by our trained model from [google drive](https://drive.google.com/file/d/1Y2fzTRUtrPLWvkFvKe0D0kd38Z2NluXm/view?usp=sharing).

## User Study Data
You can find results from our user study at [google drive](https://drive.google.com/file/d/1MUIJBSWsQfQjx33X3fpi7uL6_CkW_8mz/view?usp=sharing).

# Citation
```
@InProceedings{Siris_2020_CVPR,
author = {Siris, Avishek and Jiao, Jianbo and Tam, Gary K.L. and Xie, Xianghua and Lau, Rynson W.H.},
title = {Inferring Attention Shift Ranks of Objects for Image Saliency},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

