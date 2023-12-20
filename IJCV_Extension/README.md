# Inferring Attention Shifts for Salient Instance Ranking [IJCV 2023]

### Authors:
Avishek Siris, Jianbo Jiao, Gary K.L. Tam, Xianghua Xie, Rynson W.H. Lau

+ PDF: [Paper](https://link.springer.com/content/pdf/10.1007/s11263-023-01906-7.pdf)

## Abstract
The human visual system has limited capacity in simultaneously processing multiple visual inputs. Consequently, humans rely on shifting their attention from one location to another. When viewing an image of complex scenes, psychology studies and behavioural observations show that humans prioritise and sequentially shift attention among multiple visual stimuli. In this paper, we propose to predict the saliency rank of multiple objects by inferring human attention shift. We first construct a new large-scale salient object ranking dataset, with the saliency rank of objects defined by the order that an observer attends to these objects via attention shift. We then propose a new deep learning-based model to leverage both bottom-up and top-down attention mechanisms for saliency rank prediction. Our model includes three novel modules: Spatial Mask Module (SMM), Selective Attention Module (SAM) and Salient Instance Edge Module (SIEM). SMM integrates bottom-up and semantic object properties to enhance contextual object features, from which SAM learns the dependencies between object features and image features for saliency reasoning. SIEM is designed to improve segmentation of salient objects, which helps further improve their rank predictions. Experimental results show that our proposed network achieves state-of-the-art performances on the salient object ranking task across multiple datasets.

## Installation
The code is based on the Mask-RCNN implementation by matterport, https://github.com/matterport/Mask_RCNN. Please follow and install the requirements they list. Additionally, install pycocotools.
NOTE: This implementation is based on Tensorflow 2.4.0.

## Dataset
Download our dataset from [google drive](https://drive.google.com/file/d/1ueSpf3avLAPiJxoP40v5KL7qxaYtM1us/view?usp=sharing). (Same as in [Attention_Shift_Saliency_Rank](https://github.com/SirisAvishek/Attention_Shift_Ranks/tree/master/Attention_Shift_Saliency_Rank)).

## Training 

To train the saliency rank model, download the pre-trained baseline variation weights (Base_Edge.h5) from [google drive](https://drive.google.com/file/d/1MTygUESkT4wrX2rIILupMCgOt7gz3x12/view?usp=drive_link).
Set data paths, comment/uncomment code for pre-training/fine-tuning and run:

```
python train.py
```

## Testing

Generate predicted Saliency Rank Maps (rank based on grayscale value). You can download the weights of the trained model from [google drive](https://drive.google.com/file/d/1s2LkviY6Lepe2Ve-q1E0WfCY26YjExts/view?usp=sharing).
Set data paths and run:
```
python predict_and_generate_saliency_map.py
```

## Evaluate
To evaluate MAE and Salient Object Ranking (SOR) scores, set data paths and run:
```
python evaluate_MAE_SOR.py
```

To evaluate nDCG score, set data paths and run:
```
python evaluate_nDCG.py
```

## Results
You can download predicted Saliency Rank Maps by our trained model from [google drive](https://drive.google.com/file/d/1w6WL8sQ5cARU_rVCOYVCp6_v2ysZDpn0/view?usp=drive_link).

# Citation
```
@article{siris2023inferring,
  title={Inferring Attention Shifts for Salient Instance Ranking},
  author={Siris, Avishek and Jiao, Jianbo and Tam, Gary KL and Xie, Xianghua and Lau, Rynson WH},
  journal={International Journal of Computer Vision},
  pages={1--23},
  year={2023},
  publisher={Springer}
}
```
