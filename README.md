# MFRLNet

Official implementation for TODAES paper :
“MFRLNet: A Layout Hotspot Detection Network Based on Multi-feature Representation Learning”
<br><br>

## News & Updates
* **July 25, 2026: Pulic The MFRLNet Code.**
<br><br>

## Requirements
- **Python 3.7**
- **pytorch 1.13.0 or higher**
- **pip install -r requirements.txt**
<br><br>

## Datasets
* **ICCAD 12** &nbsp; [[download]](https://github.com/phdyang007/dlhsd) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/8360060)
* **ICCAD 19** &nbsp; [[download]](https://github.com/gauravr1991/ICCAD2019Benchmarks) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/8942128)

**We used the ICCAD 12 and ICCAD 19 for both training and val. 
Please first download our datasets via [Baidu Drive](https://pan.baidu.com/s/1Z71e_rCS25V3KNcNX-3i2Q?pwd=hjtd) (key:hjtd) and unzip the file to the folder `./MFRLNet/`.** 

**Note: This repository does not provide preprocessed binary data. You can refer to https://github.com/phdyang007/layout-generator for more information.**

* **The datasets supported by this code have the following structure:**
  ```
  ├──./datasets/classify/images/
  │    ├── ICCAD12
  │    │    ├── train
  │    │    │    ├── HS
  │    │    │    ├──    ├── 1HS0.png
  │    │    │    ├──    ├── 1HS1.png
  │    │    │    ├──    ├── ...
  │    │    │    ├── NHS
  │    │    │    ├──    ├── 1NHS0.png
  │    │    │    ├──    ├── 1NHS1.png
  │    │    │    ├──    ├── ...
  │    │    ├── test
  │    │    │    ├── HS
  │    │    │    ├──    ├── 1HS0.png
  │    │    │    ├──    ├── 1HS1.png
  │    │    │    ├──    ├── ...
  │    │    │    ├── NHS
  │    │    │    ├──    ├── 1NHS0.png
  │    │    │    ├──    ├── 1NHS1.png
  │    │    │    ├──    ├── ...
  │    ├── ICCAD19/1
  │    │    ├── train
  │    │    │    ├── HS
  │    │    │    ├──    ├── 1HS0.png
  │    │    │    ├──    ├── 1HS1.png
  │    │    │    ├──    ├── ...
  │    │    │    ├── NHS
  │    │    │    ├──    ├── 1NHS0.png
  │    │    │    ├──    ├── 1NHS1.png
  │    │    │    ├──    ├── ...
  │    │    ├── test
  │    │    │    ├── HS
  │    │    │    ├──    ├── 1HS0.png
  │    │    │    ├──    ├── 1HS1.png
  │    │    │    ├──    ├── ...
  │    │    │    ├── NHS
  │    │    │    ├──    ├── 1NHS0.png
  │    │    │    ├──    ├── 1NHS1.png
  │    │    │    ├──    ├── ...
  │    ├── ICCAD19/2
  │    │    ├── train
  │    │    │    ├── HS
  │    │    │    ├──    ├── 1HS0.png
  │    │    │    ├──    ├── 1HS1.png
  │    │    │    ├──    ├── ...
  │    │    │    ├── NHS
  │    │    │    ├──    ├── 1NHS0.png
  │    │    │    ├──    ├── 1NHS1.png
  │    │    │    ├──    ├── ...
  │    │    ├── test
  │    │    │    ├── HS
  │    │    │    ├──    ├── 1HS0.png
  │    │    │    ├──    ├── 1HS1.png
  │    │    │    ├──    ├── ...
  │    │    │    ├── NHS
  │    │    │    ├──    ├── 1NHS0.png
  │    │    │    ├──    ├── 1NHS1.png
  │    │    │    ├──    ├── ...
  ```
<be>
<br><br>

## Experiments 

The training and val experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with a single GeForce RTX 3090Ti GPU of 24 GB Memory.

The trained model results are in `./result`
<br><br>

## Training
ICCAD 19-1:
```
python train.py 
```
<br><br>

## Contact
**Welcome to raise issues or email to [WB24101001@stu.ahu.edu.cn](WB24101001@stu.ahu.edu.cn) for any question regarding our MFRLNet.**
<br><br>

## Important License Information 
This software is intended solely for **academic research, personal study and non-commercial purposes**. 

It has been patented (please refer to the "PATENTS" file) and the patent owner has been transferred to **Advanced Manufacturing EDA Co., Ltd**. 

This is strictly prohibited for any commercial use. If you need commercial authorization, please contact **Advanced Manufacturing EDA Co., Ltd**.
