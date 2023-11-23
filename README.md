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


---
---


# Inferring Attention Shifts for Salient Instance Ranking [IJCV 2023]

### Authors:
Avishek Siris, Jianbo Jiao, Gary K.L. Tam, Xianghua Xie, Rynson W.H. Lau

+ PDF: [Paper](https://link.springer.com/content/pdf/10.1007/s11263-023-01906-7.pdf)

## Abstract
The human visual system has limited capacity in simultaneously processing multiple visual inputs. Consequently, humans rely on shifting their attention from one location to another. When viewing an image of complex scenes, psychology studies and behavioural observations show that humans prioritise and sequentially shift attention among multiple visual stimuli. In this paper, we propose to predict the saliency rank of multiple objects by inferring human attention shift. We first construct a new large-scale salient object ranking dataset, with the saliency rank of objects defined by the order that an observer attends to these objects via attention shift. We then propose a new deep learning-based model to leverage both bottom-up and top-down attention mechanisms for saliency rank prediction. Our model includes three novel modules: Spatial Mask Module (SMM), Selective Attention Module (SAM) and Salient Instance Edge Module (SIEM). SMM integrates bottom-up and semantic object properties to enhance contextual object features, from which SAM learns the dependencies between object features and image features for saliency reasoning. SIEM is designed to improve segmentation of salient objects, which helps further improve their rank predictions. Experimental results show that our proposed network achieves state-of-the-art performances on the salient object ranking task across multiple datasets.

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











