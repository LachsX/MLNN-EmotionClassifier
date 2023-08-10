# Emotion Detector


This repository contains code for an Emotion Detector that uses a trained deep learning model to recognize emotions in faces from images or videos. The model is built using TensorFlow and Keras and is trained to classify emotions into several categories.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Examples](#examples)

## Introduction

The Emotion Detector is built using Python and utilizes the OpenCV library for face detection and image processing. The emotion recognition model is trained on a dataset containing various facial expressions.

## Requirements

To run the Emotion Detector, you'll need the following dependencies:

- TensorFlow
- Pandas
- NumPy
- PIL (Python Imaging Library)
- Matplotlib
- scikit-learn

## Usage

Download the dataset from Kaggle and place it in the appropriate directory.
https://www.kaggle.com/competitions/skillbox-computer-vision-project/data

Update the path variable in the code in **ImageEmotionClassifier.ipynb** to specify the path to the data directory.

To train a new model or retrain an existing one, use the **ImageEmotionClassifier.ipynb** notebook. This notebook provides the option to choose the model architecture (simple CNN model, complex CNN model, or ResNet50-based model). You can also customize model parameters (activate dropout layers or regularizers) and training parameters (use data augmentation or class weights). This notebook allows you to compare models and retrain them if needed. During training, logging is performed to keep a training history, and the model is saved at each step to enable further training. The best checkpoint is also saved. Once you achieve satisfactory results, use the saves_as_best() method to save the final model for use in **VideoEmotionDetection.ipyn**.

To detect emotions in videos, use the **VideoEmotionDetection.ipynb** notebook. In this notebook, you need to specify the paths to the model and the "haarcascade_frontalface_default.xml" file, the path to the video on which you want to detect emotions, and the path to save the processed video.

## Examples

https://github.com/VKe-13/Emotion-Detector/assets/82056992/14051556-2fdb-4d86-8e20-748406181fdf




