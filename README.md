# SVHN-CROPPED Dataset Machine Learning Project

This repository contains a machine learning project using the SVHN-CROPPED dataset. The code demonstrates the complete pipeline of a machine learning project, from data loading and preprocessing to model building, training, and evaluation.

## Table of Contents
- [Data Pipeline](#data-pipeline)
- [VGG16 Experiment](#vgg16-experiment)
- [Gaussian Function and Local Contrast Normalization (LCN)](#gaussian-function-and-local-contrast-normalization-lcn)
- [CNN Model with LCN](#cnn-model-with-lcn)
- [CNN Model with LCN, Batch Normalization, and Learning Rate Scheduler](#cnn-model-with-lcn-batch-normalization-and-learning-rate-scheduler)

## Data Pipeline
- The code defines a `DataPipeline` class that handles loading, preprocessing, and providing data for the ConvNet model.
- It performs tasks such as downloading the dataset, splitting the data into images and labels, building validation data, converting images to grayscale, and applying Global Contrast Normalization (GCN).

## VGG16 Experiment
- The code demonstrates an experiment using the VGG16 pre-trained model.
- It creates a custom model by freezing the layers of the pre-trained VGG16 model and adding custom layers on top.
- The model is then compiled, trained, and evaluated on the dataset.
- The training history is plotted to visualize the accuracy and loss curves.

## Gaussian Function and Local Contrast Normalization (LCN)
- The code defines functions for the Gaussian function and Local Contrast Normalization (LCN).
- These functions are used to preprocess the images before feeding them into the CNN model.

## CNN Model with LCN
- The code builds a CNN model using TensorFlow and Keras.
- It incorporates the LCN preprocessing step as a Lambda layer at the beginning of the model.
- The model architecture consists of convolutional layers, max pooling layers, dropout layers, and dense layers.
- The model is compiled, trained, and evaluated on the dataset.
- The training history is plotted to visualize the loss and accuracy curves.

## CNN Model with LCN, Batch Normalization, and Learning Rate Scheduler
- The code presents an improved version of the CNN model that includes additional techniques such as batch normalization, dropout, and a learning rate scheduler.
- The model architecture is similar to the previous one but with added batch normalization layers and a HeUniform initializer.
- The model is compiled, trained, and evaluated on the dataset.
- The training history is plotted to visualize the loss, accuracy, and learning rate curves.

Feel free to explore the code and experiment with different techniques to further improve the model's performance on the SVHN-CROPPED dataset.
  
