# lung_cancer_2017

This is on going work for [https://www.kaggle.com/c/data-science-bowl-2017](https://www.kaggle.com/c/data-science-bowl-2017). The task is to determine if the patent is likely to be diagnosed with lung cancer or not within one year, given his current CT scans.

The plan is not fixed yet. 

![plan](/docs/images/plan.png)

There are two possible systems. The first one is using 3d segmentation. It labels each 3d voxel  belonging to a nodule or not. The second one is based on 3d object detection. Here, 3d scanning windows of different size and aspects are classified to conatain a nodule or not. This is very much like 3d faster-rcnn. 

It is also not decided if we need a "decision tree layer" or the "ordinary dense or convolution layer" will be sufficient.

![plan](/docs/images/plan2.png)

---
## references
**[1] "U-Net: Convolutional Networks for Biomedical Image Segmentation" - Olaf Ronneberger, Philipp Fischer, Thomas Brox, MICCAI 2015**

- [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

code: 
- [https://github.com/jakeret/tf_unet](https://github.com/jakeret/tf_unet)
 
---

**[2] "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation" - Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger, MICCAI 2016**

- [https://arxiv.org/abs/1606.06650](https://arxiv.org/abs/1606.06650)
code:
- [https://gist.github.com/mongoose54/c93c113ae195188394a7b363c24e2ac0#file-gistfile1-txt](https://gist.github.com/mongoose54/c93c113ae195188394a7b363c24e2ac0#file-gistfile1-txt)

- [http://lmb.informatik.uni-freiburg.de/resources/opensource/unet.en.html](http://lmb.informatik.uni-freiburg.de/resources/opensource/unet.en.html)

"V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation" - 
Fausto Milletari, Nassir Navab1, Seyed-Ahmad Ahmadi -

- [http://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf](http://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf)

code:
- [https://github.com/faustomilletari/VNet](https://github.com/faustomilletari/VNet)

---
**[3] "Deep Neural Decision Forests" - Peter Kontschieder, Madalina Fiterau, Antonio Criminisi, Samuel Rota Bulò, ICCV 2015**

- [http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf)

code: 
- [https://github.com/chrischoy/fully-differentiable-deep-ndf-tf](https://github.com/chrischoy/fully-differentiable-deep-ndf-tf)
- [https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/tensor_forest/hybrid/python/layers/decisions_to_data.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/tensor_forest/hybrid/python/layers/decisions_to_data.py)



