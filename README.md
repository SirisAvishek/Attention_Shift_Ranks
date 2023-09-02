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


# Inferring Attention Shifts for Salient Instance Ranking [Accepted IJCV]

### Authors:
Avishek Siris, Jianbo Jiao, Gary K.L. Tam, Xianghua Xie, Rynson W.H. Lau

## CODE & DATA - Coming Soon











