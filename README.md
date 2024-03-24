# Paraformer: Updating Large-scale High-resolution Land Cover Maps from Low-resolution Historical Labels
Large-scale high-resolution (HR) land-cover mapping is a vital task to survey the Earth's surface and resolve many challenges facing humanity. 
However, it is still a non-trivial task hindered by complex ground details, various landforms, and the scarcity of accurate training labels over a wide-span geographic area. 
To address these limitations, we propose an efficient, weakly supervised framework (Paraformer), a.k.a Low-to-High Network (L2HNet) v2, to guide large-scale HR land-cover mapping with easy-access historical land-cover data of low resolution (LR). 

The code and data are regrouping. 
We are preparing the camera-ready version of the CVPR 2024.
Stay tuned!

The Chesapeake Dataset
-------
![image](https://github.com/LiZhuoHong/Paraformer/blob/main/The%20Chesapeake%20Dataset.png)
The Chesapeake Bay dataset, grouped by Microsoft (https://lila.science/datasets/chesapeakelandcover), contains 1-meter resolution images and a 30-meter resolution land-cover product as the training data pairs and also contains a 1-meter resolution ground reference for assessment. Figure 1 illustrates the location, Digital Elevation Model (DEM), numbers of the tiles, and data samples of the Chesapeake Bay dataset. In more detail, the data sources are shown as follows:

**The HR remote sensing images** with 1-meter resolution were captured by the airborne platform of the U.S. Department of Agriculture’s National Agriculture Imagery Program (NAIP). The images contained four bands of red, green, blue, and near-infrared.

**The rough historical land-cover products** with 30-meter resolution were collected from the National Land Cover Database of the United States Geological Survey (USGS). The NLCD data contains 16 land-cover types and is utilized as the labels during the training process of the proposed Paraformer framework.

**The HR ground references** with 1-meter resolution were obtained from the Chesapeake Bay Conservancy Land Cover (CCLC) project. The CCLC data were interpreted based on the 1-meter NAIP imagery and LiDAR data containing six land-cover types. In this paper, the CCLC data were only used as the ground reference for quantitative and qualitative assessment and were not involved in the framework training or optimization process. 


The Poland Dataset
-------

The SinoLC-1 Dataset
-------


