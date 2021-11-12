# The International Workshop on Osteoarthritis Imaging Knee MRI Segmentation Challenge: A Multi-Institute Evaluation and Analysis Framework on a Standardized Dataset

## Introduction
This repo contains the implementation of the deep learning-based knee MRI cartilage segmentation model used for osteoarthritis research as described in our paper: [The International Workshop on Osteoarthritis Imaging Knee MRI Segmentation Challenge: A Multi-Institute Evaluation and Analysis Framework on a Standardized Dataset](https://arxiv.org/abs/2004.14003). By using this implementation, you can either train new knee MRI cartilage segmentation models or obtain knee MRI cartilage segmentation by using our pretrained model. 

[![license](https://img.shields.io/badge/license-AGPL--3.0-brightgreen)](https://github.com/denizlab/2019_IWOAI_Challenge/blob/master/LICENSE)

## Instructions
1. Please refer to IBM Power machine's "PowerAI 1.6.0" conda environment with the dependencies provided in `requirements.txt`. For Linux machines, please use the `requirements.yml` file to create a Conda environment:  
```bash
conda env create -f requirements.yml
```
2. `./data` folder contains subfolders named `./train`, `./test` , and `./valid` which needs to include train, test and validation set images and segmentation maps, respectively. To request the dataset used for the IWOAI challenge, please contact [Dr. Chaudhari](mailto:akshaysc@stanford.edu). 
3. Once you have data ready, you can use `train.py` to train a segmentation model with predefined settings that were used on model training for submission to the IWOAI challenge. Trained models will be saved within the folder named `./TrainedModels` 
4. Use `inference.py` file to obtain knee MRI cartilage segmentation by using our pretrained model. MR images located inside `./data/test` folder will be used to produce cartilage segmentation maps. 

## Repo Structure
* `./data`: Folder for knee MR images and cartilage segmentation that will be used for training.
* `./InferenceModel`: Trained model weights used in this study. They can be downloaded from [here](https://drive.google.com/file/d/1x9ET75IJuL-y64kHOa718PDKu3vTxQ_H/view?usp=sharing).
* `./InferenceResults`: will contain the cartilage segmentation maps when `inference.py` file runs. 
* `./TrainedModels`: will contain trained cartilage segmentation models when `train.py` file runs. 

## Training a DL model
You can use this repo to train models for segmenting the knee MRI cartilage. Default parameters are defined within `train.py` file. So, you directly run the following script to train a DL model. 
```bash
python train.py
```
Please see the arguments in the .py file to make input argument changes if you need. 

## Inference
Once you download pretrained model weights from [here](https://drive.google.com/file/d/1x9ET75IJuL-y64kHOa718PDKu3vTxQ_H/view) to the `./InferenceModel` folder, you can use the following script to obtain cartilage segmentation of knee MR images located under `./data/test`:
```bash
python inference.py 
```
## Computer System
We used an HPC system with multiple IBM Witherspoon GPU nodes, each node has multiple IBM Power9 4Ghz Processors and V100 NVLink (Volta) GPU’s, and a 512GB RAM. In this project, we used TensorFlow v1.13.1 with a single Nvidia Tesla V100 32GB RAM GPU. 

## License
This repository is licensed under the terms of the GNU AGPLv3 license.

## Reference
If you found this code useful, please cite our paper (our team is highlighted in bold text):

*The International Workshop on Osteoarthritis Imaging Knee MRI Segmentation Challenge: A Multi-Institute Evaluation and Analysis Framework on a Standardized Dataset*
Arjun D. Desai, Francesco Caliva, Claudia Iriondo, Naji Khosravan, Aliasghar Mortazi, Sachin Jambawalikar, Drew Torigian, Jutta Ellermann, Mehmet Akcakaya, Ulas Bagci, Radhika Tibrewala, Io Flament, Matthew O' Brien, Sharmila Majumdar, Mathias Perslev, Akshay Pai, Christian Igel, Erik B. Dam, Sibaji Gaj, Mingrui Yang, Kunio Nakamura, Xiaojuan Li, **Cem M. Deniz, Vladimir Juras, Ravinder Regatte**, Garry E. Gold, Brian A. Hargreaves, Valentina Pedoia, Akshay S. Chaudhari
Radiology: Artificial Intelligence
2021 3:3
```
@article{desai2021IWOAI,
    title = {The International Workshop on Osteoarthritis Imaging Knee MRI Segmentation Challenge: A Multi-Institute Evaluation and Analysis Framework on a Standardized Dataset},
    author = {Arjun D. Desai, Francesco Caliva, Claudia Iriondo, Naji Khosravan, Aliasghar Mortazi, Sachin Jambawalikar, Drew Torigian, Jutta Ellermann, Mehmet Akcakaya, Ulas Bagci, Radhika Tibrewala, Io Flament, Matthew O' Brien, Sharmila Majumdar, Mathias Perslev, Akshay Pai, Christian Igel, Erik B. Dam, Sibaji Gaj, Mingrui Yang, Kunio Nakamura, Xiaojuan Li, **Cem M. Deniz**, Vladimir Juras, Ravinder Regatte, Garry E. Gold, Brian A. Hargreaves, Valentina Pedoia, Akshay S. Chaudhari}, 
    journal = {Radiology: Artificial Intelligence},
    year = {2021},
    volume = {3},
    number = {3}
    URL = {https://doi.org/10.1148/ryai.2021200078}
}
```
