# Converting PCD files to CSV for wheat rust detection using LiDAR point data
 
Implementation of paper - [Monitoring leaf rust and yellow rust in wheat with 3D LiDAR sensing](#)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov7-trainable-bag-of-freebies-sets-new/real-time-object-detection-on-coco)](https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=yolov7-trainable-bag-of-freebies-sets-new)
<a href="https://colab.research.google.com/gist/AlexeyAB/b769f5795e65fdab80086f6cb7940dae/yolov7detection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2207.02696-B31B1B.svg)](https://arxiv.org/abs/2207.02696)

<div align="center">
    <a href="./">
        <img src="src/img.01.jpg" width="30%"/>
    </a>
    <a href="./">
        <img src="src/img.02.jpg" width="30%"/>
    </a>
    <a href="./">
        <img src="src/img.03.jpg" width="30%"/>
    </a>
</div>

## Installation & runnning the code

- The code use libraries well known. But check in your environment if you have all the libraries used in this code. Check in the folde ```inputs``` you will find the pcd files and in the folder ```outputs``` the csv files will be saved. Currently there is a test.pcd files can be used for testing purposed. 
- 
```
git clone https://github.com/eapolo/agrolidarwheatrust.git
cd agrolidarwheatrust
python converter.py
```


## Device

- This code has been created to process images taken by the device FLIR Systems AB and model RGB FLIR E53



## Acknowledgements

- This code has been created to convert pcd files to csv. We took as reference the code published by [pypcd4](https://github.com/MapIV/pypcd4/tree/main)to build this new code which do what we want to do. 















