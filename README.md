# Classification of Pneumonia Cases in Chest X-ray Images
Full reports of our experiemnts are accessible through [here](https://github.com/AghdamAmir/Pneumonia-Classification/blob/main/TransferLearning.pdf).

## Requirements
* Python
* Numpy
* Tensorflow/Keras
* Scikit-learn
* Matplotlib

## Dataset
We have used *COVID-19 CXR* Dataset(accessible via [here](https://github.com/hasibzunair/synthetic-covid-cxr-dataset/releases/tag/v0.1)), which contains synthetic Chest X-ray images of Covid-19 patients.

![Dataset Samples](https://raw.githubusercontent.com/AghdamAmir/Pneumonia-Classification/main/COVID-19_CXR.jpeg)

* The dataset contains two classes: Normal and Pneumonia
* It consists of 21,295 images: 16537 images for normal cases, and 4758 images for pneumonia cases
* Each gray scale image has a size of 256x256 

## Model Architecture
Xception is a deep model built upon InceptionV3 based on the hypothesis that the mapping of cross-channel correlations and spatial correlations in the feature maps of convolutional neural networks can be entirely decoupled. 

Inception and Xception both have a similar number of parameters but with a major distinction in their convolutions. Xception is designed using Depth-wise separable convolution with an expansion rate of 1.

![Xception](https://github.com/AghdamAmir/Pneumonia-Classification/blob/main/Xception.png)

We used the ImageNet pre-trained model and fine-tuned on the COVID-19 CXR Dataset. The details of our method is fully reported in [here](https://github.com/AghdamAmir/Pneumonia-Classification/blob/main/TransferLearning.pdf).

## Loss Function
We used **weighted Binary Cross-Entropy** loss due to the higly imbalance distribution of samples in the Dataset. Class weights were computed using only the training data.
