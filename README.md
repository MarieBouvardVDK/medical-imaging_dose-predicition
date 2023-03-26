# <center>Radiation Dose Prediction for Cancer Patients</center>


## Credit
This project was created by Amandine Allmang and Marie Bouvard as a part of the Deep Learning for Medical Imaging course of the MSc AI at CentraleSupelec. The project is inspired by the [OpenKBP competition](https://www.aapm.org/GrandChallenge/OpenKBP/) on the comparisons of dose prediction methods.

## Dataset
The [dataset](https://github.com/soniamartinot/MVA-Dose-Prediction.git) used in this project corresponds to 2D data which contains structural masks, possible dose masks, ground-truth dose and CT scans.

## Implementation
For this task, we implemented a UNet model and explored the impact of various block architectures, model hyperparameters, data augmentation and training parameters. The UNet architecture is based on the paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) by Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 

![Unet](unet_archi.png)

## Installation
To run this code, you need to have Python 3 and the following Python libraries installed: 
- pandas 
- numpy 
- matplotlib 
- torch
- os
- tqdm 
- time

To install these libraries, you can use pip by running:

`pip install pandas numpy matplotlib torch os tqdm time`

## Usage
You can run the code by opening the Jupyter Notebook `AD_NAME.ipynb` and executing the cells. The notebook contains the training of the best performing model and visualizations of the results. To do so, the notebook uses the different helper and model methods present in the Python (.py) files.

The code consists of the following elements:

- Data Cleaning and Preprocessing: In this step, we load the dataset, and perform some data augmentation to prepare the data for modeling. `helpers/dataset.py` contains the classes to create Datasets for training and testing
- `helpers/constants.py` contains the constant used for the training pipeline of the models
- Model Training: In this step, we trained and compared the performance of different Deep Learning model architectures and parameters to obtain the best performing model. `helpers/model_functions_cuda.py` and `helpers/model_functions_mps.py` contain the functions used for the training and the evaluation of the models. Two versions exists depending of the use of cuda or mps GPU.
- `helpers/predict_and_submit.py` contains the functions used to make predicts and challenge submissions on the test set.
- The `models` folder contains the code for the various model architecture implemented and compared in our project, namely `UNet.py`, `GeneratorUNet.py`, `MultUNet.py`, and `DCNN.py`.



