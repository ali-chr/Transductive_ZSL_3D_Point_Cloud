# Zero-shot learning on 3d point cloud objects and beyond


Created by [Ali Cheraghian](https://scholar.google.com/citations?user=QT0EXIkAAAAJ&hl=en) from Australian National University.


# Introduction
This work is based on our (https://link.springer.com/article/10.1007/s11263-022-01650-4) work, which is published in International Journal of Computer Vision
(IJCV).

Zero-shot learning, the task of learning to recognize new classes not seen during training, has received considerable attention in the case of 2D image classification. However, despite the increasing ubiquity of 3D sensors, the corresponding 3D point cloud classification problem has not been meaningfully explored and introduces new challenges. In this paper, we identify some of the challenges and apply 2D Zero-Shot Learning (ZSL) methods in the 3D domain to analyze the performance of existing models. Then, we propose a novel approach to address the issues specific to 3D ZSL. We first present an inductive ZSL process and then extend it to the transductive ZSL and Generalized ZSL (GZSL) settings for 3D point cloud classification. To this end, a novel loss function is developed that simultaneously aligns seen semantics with point cloud features and takes advantage of unlabeled test data to address some known issues (e.g., the problems of domain adaptation, hubness, and data bias). While designed for the particularities of 3D point cloud classification, the method is shown to also be applicable to the more common use-case of 2D image classification. An extensive set of experiments is carried out, establishing state-of-the-art for ZSL and GZSL on synthetic (ModelNet40, ModelNet10, McGill) and real (ScanObjectNN) 3D point cloud datasets.


![Unseen_fea-1](https://user-images.githubusercontent.com/52984527/181682439-6bcfdeea-955b-4d54-8a33-b260e9ec128d.png)

Visualization of unseen point cloud feature vectors (circles) with four backbone networks. Since
the synthetic data has less noise, ModelNet40 features are clustered better than the real scanned 3D data (with
noise) from ScanObjectNN. Moreover, for both datasets, the models are trained on synthetic instances belonging
to a subset of ModelNet40 classes, and so we expect the ModelNet10 features to be better clustered than the
ScanObjectNN features.




## Requirements
Requirements can be found in [this](requirements.txt) file.
```
numpy==1.20.2
Pillow==8.2.0
PyYAML==5.4.1
scipy==1.6.2
torch==1.8.1+cu111
torchaudio==0.8.1
torchvision==0.9.1+cu111
typing-extensions==3.7.4.3
```

## Training & Evaluation

The [config](config) folder includes configuration files for all datasets.

For inductive training, you can run this command.
```
python train_inductive.py
```

For transductive training, you can run this command.

```
python train_transductive.py
```

The following are the arguments of both scripts:
``` 
-h, --help       
    show this help message and exit
--dataset {ModelNet,ScanObjectNN,McGill}        
    name of dataset i.e. ModelNet, ScanObjectNN, McGill
--backbone {EdgeConv,PointAugment,PointConv,PointNet}       
    name of backbone i.e. EdgeConv, PointAugment, PointConv, PointNet 
--method {ours,baseline}
    name of method i.e. ours, baseline
--config_path CONFIG_PATH
    configuration path 
```





# Data
You can download the feature vectors, which are extracted from PointNet, PointAugment, DGCNN, and PointConv architecures, of ModelNet, McGill, and ScanObjectNN datasets from the following link,

[feature vectors of ModelNet, McGill, and ScanObjectNN datasets](https://drive.google.com/drive/folders/1y8HbxfBWzIzZ4pH-L1wi07pfuhGy8R2m?usp=sharing)

# Model

You can download the trained PointNet, PointAugment, DGCNN, and PointConv architecures, of ModelNet, McGill, and ScanObjectNN datasets from the following link,

[Model](https://drive.google.com/drive/folders/1iYXtVvyNK7FBMwoKDmRdj2NflCmzOXIL?usp=sharing)



# Citation
If you find our work useful in your research, please consider citing:

	@article{cheraghian2022ZSL,
	  title={Zero-Shot Learning on 3D Point Cloud Objects and Beyond},
	  author={Ali Cheraghian, Shafin Rahman,  Townim F. Chowdhury, Dylan Campbell, and Lars Petersson},
	  journal={International Journal of Computer Vision},
	  year={2022}
	}
	

## Reference
[1] A.  Cheraghian,  S.  Rahman,  and  L.  Petersson.    Zero-shot learning  of  3d  point  cloud  objects.   In International  Conference on Machine Vision Applications (MVA), 2019. 

[2] A. Cheraghian, S. Rahman, D. Campbell, and L. Petersson. Mitigating the hubness problem for zero-shot learning of 3D objects.  In British Machine Vision Conference (BMVC), 2019. 

[3] A. Cheraghian, S. Rahman, D. Campbell, and L. Petersson. Transductive Zero-Shot Learning for 3D Point Cloud Classification.  In Winter Conference on Applications of Computer Vision (WACV), 2020. 

[4] A. Cheraghian, S. Rahman,  T. F. Chowdhury, D. Campbell, and L. Petersson. Zero-Shot Learning on 3D Point Cloud Objects and Beyond.  In International Journal of Computer Vision (IJCV), 2022. 

