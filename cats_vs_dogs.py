#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[45]:


# standard imports
import os

# external imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import cv2

# import datasets, models, algorithms, etc.
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# In[35]:


# load images from the folder
def load_images(folder, label, resize_target=(64, 64)):
    folder_path_ = os.path.join(os.getcwd(), "data/set_200", folder)

    data_ = []
    
    if not os.path.exists(folder_path_):
        print(f"path '{folder_path_}' does not exist.")
        return

    # go through every single file in the folder
    for file_ in os.listdir(folder_path_):
        # first check if the file is an image
        if not file_.endswith((".jpg", ".jpeg", ".png")):
            print(f"'{file_}' is not an image.")
            continue

        # read the image
        file_path_ = os.path.join(folder_path_, file_)
        image_ = cv2.imread(file_path_)
        if image_ is None:
            print(f"Failed to load image from path '{file_path_}'.")
            continue

        # scale image to specified target size
        resized_image_ = cv2.resize(image_, resize_target)

        # convert from BGR (cv2 default) to gray-scale (since HOG generally works better with gray-scale)
        gray_image_ = cv2.cvtColor(resized_image_, cv2.COLOR_BGR2GRAY)

        # extract HOG features and image
        features_, hog_img_ = hog(
            gray_image_, 
            orientations = 9, 
            pixels_per_cell = (8, 8), 
            cells_per_block = (2, 2), 
            visualize = True
        )

        # append to data list
        data_.append((image_, label, features_, hog_img_))
    
    dataframe_ = pd.DataFrame(data_, columns=["image", "label", "hog_features", "hog_image"])
    return dataframe_


# In[56]:


# load the dog and cat images
# img_dog = load_image("dog.jpg")
# img_cat = load_image("cat.jpg")
df_dog = load_images("dog", 0)
df_cat = load_images("cat", 1)

df_all =  pd.concat([df_cat, df_dog]).reset_index(drop=True)


# In[57]:


# prepare the features and label

# HOG features
X_data = df_all["hog_features"].tolist()

# labels
y_data = df_all["label"].values


# In[58]:


# split the data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.25)


# In[59]:


# train with the classifier model of SVM
model = SVC(kernel="linear", random_state = 42)
model.fit(X_train, y_train)


# In[60]:


# evaludate the classifier on the test set
y_pred = model.predict(X_test)


# In[61]:


# report the model's accuracy

accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# In[ ]:




