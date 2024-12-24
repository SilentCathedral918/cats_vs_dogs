#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.svm import SVC
from skimage.feature import hog 
from skimage.color import rgb2gray
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

# loads and resize images
def read_label_resize_all_images_from_folder(folder_path, label, target_size=(128, 128)):
    folder_path_ = os.path.abspath(folder_path)
    data_ = []

    if not os.path.exists(folder_path_):
        print(f"folder path {folder_path_} does not exist.")
        return pd.DataFrame(columns=["image", "label"])

    for file_ in os.listdir(folder_path_):
        if not file_.endswith((".jpg", ".jpeg", ".png")):
            continue

        file_path_ = os.path.join(folder_path_, file_)
        image_ = cv2.imread(file_path_)
        if image_ is None:
            continue

        corrected_image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        resized_image_ = cv2.resize(corrected_image_, target_size)
        data_.append((resized_image_, label))

    return pd.DataFrame(data_, columns=["image", "label"])

# preprocess images - feature extraction, scaling, and PCA
def preprocess_images(images, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys", n_components=0.95, n_jobs=-1):
    gray_images_ = np.array([image if len(image.shape) == 2 else rgb2gray(image) for image in images])
    
    # extract features
    def extract_hog(image):
        return hog(
            image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm
        )
    hog_features_ = np.array(Parallel(n_jobs=n_jobs)(delayed(extract_hog)(image) for image in gray_images_))
    
    # scale the features
    scaler_ = StandardScaler()
    hog_features_ = scaler_.fit_transform(hog_features_)
    
    # apply PCA for reducing feature dimensionality
    pca_ = PCA(n_components=n_components)
    hog_features_ = pca_.fit_transform(hog_features_)
    
    return hog_features_, scaler_, pca_

# define and build SVM Classifier model
def build_model():
    return SVC(C=5, tol=0.0001, random_state=42)

# load and prepare the dataset
df_dog = read_label_resize_all_images_from_folder("data/dog", 0, target_size=(256, 256))
df_cat = read_label_resize_all_images_from_folder("data/cat", 1, target_size=(256, 256))
df = pd.concat([df_dog, df_cat])

X = np.array(df["image"])
y = np.array(df["label"])

# split the data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# preprocess and extract features for training
X_train_prepared, scaler, pca = preprocess_images(X_train, orientations=16, pixels_per_cell=(28, 28), n_components=0.8)
X_test_prepared, _, _ = preprocess_images(X_test, orientations=16, pixels_per_cell=(28, 28), scaler=scaler, pca=pca)

# train the model
svc_model = build_model()
svc_model.fit(X_train_prepared, y_train)

# model final evaluation
accuracy = svc_model.score(X_test_prepared, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
