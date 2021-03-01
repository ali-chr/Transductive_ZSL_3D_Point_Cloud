# Transductive Zero-Shot Learning for 3D Point Cloud Classification


Created by [Ali Cheraghian](https://scholar.google.com/citations?user=QT0EXIkAAAAJ&hl=en) from Australian National University.


# Introduction
This work is based on our [arXiv tech report](https://arxiv.org/abs/1912.07161), which is going to appear in WACV 2020. We proposed a novel transductive zero-shot learning approach for 3D point clouds. 

Zero-shot learning, the task of learning to recognize new classes  not  seen  during  training,  has  received  considerable attention in the case of 2D image classification. However despite the increasing ubiquity of 3D sensors, the corresponding 3D point cloud classification problem has not been meaningfully explored and introduces new challenges. This  project  extends,  for  the  first  time, transductive  Zero-Shot Learning (ZSL) and Generalized Zero-Shot Learning (GZSL) approaches to the domain of 3D point cloud classification. 

In the "class_name" folder, the class names of seen and unseen sets from all 3D datasets are shown. Also, the "word_vector" folder contains the semantic word vectors of 3D datasets.  






   
# Train & Test codes
Coming soon ...



# Evaluation protocols
The evaluation protocols for ZSL and GZSL in this project were introduced by [1] and [2] respectively. 








# Feature vector
You can download the feature vectors, which are extracted from PointNet, PointAugment, DGCNN, and PointConv architecures, of ModelNet, McGill, and ScanObjectNN datasets from the following link,

[feature vectors of ModelNet, McGill, and ScanObjectNN datasets using PointNet](https://drive.google.com/drive/folders/1XgYRhG6PY5AVLFSWlD0oWCeQbb3JIrSy?usp=sharing)

# Citation
If you find our work useful in your research, please consider citing:

	@article{cheraghian2019transductive,
	  title={Transductive Zero-Shot Learning for 3D Point Cloud Classification},
	  author={Ali Cheraghian, Shafin Rahman, Dylan Campbell, and Lars Petersson},
	  journal={arXiv preprint arXiv:1912.07161},
	  year={2019}
	}

## Reference
[1] A.  Cheraghian,  S.  Rahman,  and  L.  Petersson.    Zero-shot learning  of  3d  point  cloud  objects.   In International  Conference on Machine Vision Applications (MVA), 2019. 

[2] A. Cheraghian, S. Rahman, D. Campbell, and L. Petersson. Mitigating the hubness problem for zero-shot learning of 3D objects.  In British Machine Vision Conference (BMVC), 2019. 

[3] A. Cheraghian, S. Rahman, D. Campbell, and L. Petersson. Transductive Zero-Shot Learning for 3D Point Cloud Classification.  In Winter Conference on Applications of Computer Vision (WACV), 2020. 

