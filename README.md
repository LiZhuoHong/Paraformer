# Paraformer: Updating Large-scale High-resolution Land Cover Maps from Low-resolution Historical Labels
Large-scale high-resolution (HR) land-cover mapping is a vital task to survey the Earth's surface and resolve many challenges facing humanity. 
However, it is still a non-trivial task hindered by complex ground details, various landforms, and the scarcity of accurate training labels over a wide-span geographic area. 
To address these limitations, we propose an efficient, weakly supervised framework (Paraformer), a.k.a Low-to-High Network (L2HNet) v2, to guide large-scale HR land-cover mapping with easy-access historical land-cover data of low resolution (LR). 

The Paraformer is accepted by CVPR 2024 **:rocket:Highlight:rocket:** with a score of 5/5/4. See you in VALSE (Chongqing, 5 May) and CVPR (Seattle, 17 June)!

Contact me at ashelee@whu.edu.cn
* [**Paper**](https://arxiv.org/abs/2403.02746)
* [**My homepage**](https://lizhuohong.github.io/lzh/)
  
Our previous works:
* [**L2HNet V1**](https://www.sciencedirect.com/science/article/abs/pii/S0924271622002180): The low-to-high network for HR land-cover mapping using LR labels.
* [**SinoLC-1**](https://essd.copernicus.org/articles/15/4749/2023/): The first 1-m resolution national-scale land-cover map of China.

## Training Instructions

* **To train and test the Paraformer on the default Chesapeake Bay dataset, follow these steps:**
1. Download the imagenet21k ViT pre-train model at [**Pre-train ViT**](https://drive.google.com/file/d/10Ao75MEBlZYADkrXE4YLg6VObvR0b2Dr/view?usp=sharing) and put it at *"./networks/pre-train_model/imagenet21k"*
   
2. If you want to run the code with the default Chesapeake dataset, we provide example data for the state of New York. Download the dataset at [**Baidu cloud**](https://pan.baidu.com/s/1FOCRmWenNmJ9mYDKBK_uXQ?pwd=2024) and put them at *"./dataset/Chesapeake_NewYork_dataset"*.
   
3. Run the "Train" command:
   ```bash
   python train.py --dataset Chesapeake --batch_size 10 --max_epochs 100 --savepath *save path of your folder* --gpu 0
4. After training, run the "Test" command:
   ```bash
   python test.py --dataset Chesapeake --model_path *The path of trained .pth file* --save_path *To save the inferred results* --gpu 0
   
* **To train and test the framework on your dataset:**

1. Generate a train and test lists (.csv) of your dataset (an example is in the "dataset" folder).
2. Change the label class and colormap in the "utils.py" file.
3. Add your dataset_config in the "train.py" and "test.py" files.
4. Run the command above by changing the dataset name.

  
The Chesapeake Dataset
-------
<img src="https://github.com/LiZhuoHong/Paraformer/blob/main/The%20Chesapeake%20Dataset.png" width="70%">
The Chesapeake Bay dataset, grouped by Microsoft Research, contains 1-meter resolution images and a 30-meter resolution land-cover product as the training data pairs and also contains a 1-meter resolution ground reference for assessment. The figure illustrates the location, Digital Elevation Model (DEM), numbers of the tiles, and data samples of the Chesapeake Bay dataset. 

* **The HR remote sensing images** with 1-meter resolution were captured by the airborne platform of the U.S. Department of Agriculture’s National Agriculture Imagery Program (NAIP). The images contained four bands of red, green, blue, and near-infrared.

* **The rough historical land-cover products** with 30-meter resolution were collected from the National Land Cover Database of the United States Geological Survey (USGS). The NLCD data contains 16 land-cover types and is utilized as the labels during the training process of the proposed Paraformer framework.

* **The HR ground references** with 1-meter resolution were obtained from the Chesapeake Bay Conservancy Land Cover (CCLC) project. The CCLC data were interpreted based on the 1-meter NAIP imagery and LiDAR data containing six land-cover types. In this paper, the CCLC data were only used as the ground reference for quantitative and qualitative assessment and were not involved in the framework training or optimization process. 

The data can be downloaded at Microsoft's website: [**Chesapeake dataset**](https://lila.science/datasets/chesapeakelandcover)

The Poland Dataset
-------
<img src="https://github.com/LiZhuoHong/Paraformer/blob/main/The%20Poland%20dataset.png" width="70%">
The Poland dataset contained 14 Provinces of Poland, including the Provinces of Pomorskie, Lódzkie, Lubuskie, Dolnoslaskie, etc. The figure demonstrates the location, DEM, numbers of the tiles, and data samples of the Poland dataset. 

* **The HR remote sensing images** with 0.25-meter and 0.5-meter resolution were collected from the LandCover.ai dataset where the image sources are from the public geodetic resource used in the Land Parcel Identification System (LPIS). The images contained three bands of red, green, and blue.

* **The rough historical labeled data** with 10-meter resolution were collected from three types of global land-cover products which were (1) The FROM_GLC10 provided by the Tsinghua University, (2) The ESA_WorldCover v100 provided by the European Space Agency (ESA), and (3) The ESRI 10-meter global land cover (abbreviated as ESRI_GLC10) provided by the ESRI Inc. and IO Inc. The 30-meter resolution labeled data were collected from the 30-meter global land-cover product GLC_FCS30 provided by the Chinese Academy of Sciences (CAS).

* **The HR ground references** were obtained from the OpenEarthMap dataset provided by the University of Tokyo. The ground references were interpreted based on the 0.25-meter and 0.5-meter resolution LPIS imagery and contained five land-cover types. 

The data can be downloaded at [**Poland dataset**](https://drive.google.com/file/d/1qz1r5IQ-bkpUJN52GeGwvCFcbsaRu1Gs/view?usp=sharing)

The SinoLC-1 Dataset
-------
<img src="https://github.com/LiZhuoHong/Paraformer/blob/main/The%20SinoLC-1%20dataset.png" width="70%">
Based on our previous work on SinoLC-1 (i.e., the first 1-m land-cover map of China), we regard the intersected results of three 10-m land-cover products (ESA_GLC10, Esri_GLC10, and FROM_GLC10) as the LR training labels of 1-m Google Earth images. The Paraformer refines a more accurate urban pattern. For the whole of Wuhan City, the reported overall accuracy (OA) of SinoLC-1 is 72.40%. The updated results of the proposed Paraformer reach 74.98% with a 2.58% improvement.

The data can be downloaded at [**SinoLC-1 dataset**](https://doi.org/10.5281/zenodo.7707461)

Citation
-------
   ```bash
@article{li2024learning,
  title={Learning without Exact Guidance: Updating Large-scale High-resolution Land Cover Maps from Low-resolution Historical Labels},
  author={Li, Zhuohong and He, Wei and Li, Jiepan and Lu, Fangxiao and Zhang, Hongyan},
  journal={arXiv preprint arXiv:2403.02746},
  year={2024}
}

