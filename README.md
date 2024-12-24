# Cats vs Dogs Classification

**Project Overview**  
This project focuses on classifying images of cats and dogs using traditional machine learning techniques with scikit-learn. 
The pipeline involves: feature extraction with HOG (Histogram of Oriented Gradients), dimensionality reduction using PCA (Principal Component Analysis), and classification using SVM (Support Vector Machine). 
The goal of this project is to demonstrate how traditional machine learning methods can effectively handle image classification tasks.

## 1. **Data Preparation**
The dataset consists of 25,000 images, equally split between cats and dogs. 
Images were loaded from directories and resized to 256 x 256 pixels. 
Each image was converted to grayscale to simplify the feature extraction process. 
HOG was used to extract key features from the images, which were then scaled using StandardScaler. 
Principal Component Analysis (PCA) was applied to reduce the dimensionality of the feature set, retaining 80% of the variance.

## 2. **Model Training**
The training process involved splitting the data into training and test sets (80% for training, 20% for testing). 
SVM (Support Vector Machine) Classifier was trained on the preprocessed data. 
The SVM hyperparameters, such as the regularization parameter `C`, kernel `gamma`, and tolerance for stopping `tol`, were manually selected and iteratively fine-tuned. 

## 3. **Model Evaluation** 
The model achieved an accuracy of **82.67%** performing on the test set.

## Pipeline Overview

| Stage                    | Description                                                      |
|--------------------------|------------------------------------------------------------------|
| **Data Preparation**      | Resize images, convert to grayscale, and extract HOG features.    |
| **Feature Scaling**       | Standardize features using StandardScaler.                       |
| **Dimensionality Reduction** | Apply PCA to reduce feature dimensionality (retain 80% variance). |
| **Model Training**        | Train an SVM model with selected hyperparameters.                |
| **Evaluation**            | Evaluate the model on the test set and compute accuracy.         |
