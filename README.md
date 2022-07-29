# Zero-shot learning on 3d point cloud objects and beyond


Created by [Ali Cheraghian](https://scholar.google.com/citations?user=QT0EXIkAAAAJ&hl=en) from Australian National University.


# Introduction
This work is based on our [arXiv tech report]([[https://arxiv.org/abs/1912.07161]), which is published in IJCV 2022. We proposed novel techniques zero-shot learning approach for 3D point clouds. 

Zero-shot learning, the task of learning to recognize new classes not seen during training, has received considerable attention in the case of 2D image classification. However, despite the increasing ubiquity of 3D sensors, the corresponding 3D point cloud classification problem has not been meaningfully explored and introduces new challenges. In this paper, we identify some of the challenges and apply 2D Zero-Shot Learning (ZSL) methods in the 3D domain to analyze the performance of existing models. Then, we propose a novel approach to address the issues specific to 3D ZSL. We first present an inductive ZSL process and then extend it to the transductive ZSL and Generalized ZSL (GZSL) settings for 3D point cloud classification. To this end, a novel loss function is developed that simultaneously aligns seen semantics with point cloud features and takes advantage of unlabeled test data to address some known issues (e.g., the problems of domain adaptation, hubness, and data bias). While designed for the particularities of 3D point cloud classification, the method is shown to also be applicable to the more common use-case of 2D image classification. An extensive set of experiments is carried out, establishing state-of-the-art for ZSL and GZSL on synthetic (ModelNet40, ModelNet10, McGill) and real (ScanObjectNN) 3D point cloud datasets.

In the "class_name" folder, the class names of seen and unseen sets from all 3D datasets are shown. Also, the "word_vector" folder contains the semantic word vectors of 3D datasets.  






   
# Train & Test codes
Coming soon ...



# Evaluation protocols
The evaluation protocols for ZSL and GZSL in this project were introduced by [1] and [2] respectively. 








# Feature vector
You can download the feature vectors, which are extracted from PointNet, PointAugment, DGCNN, and PointConv architecures, of ModelNet, McGill, and ScanObjectNN datasets from the following link,

[feature vectors of ModelNet, McGill, and ScanObjectNN datasets](https://drive.google.com/drive/folders/1XgYRhG6PY5AVLFSWlD0oWCeQbb3JIrSy?usp=sharing)

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

