#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from time import perf_counter 
import os
import joblib
from keras.models import load_model
 


# In[2]:


batch_size = 20
img_height = 250
img_width = 250


# In[3]:


training_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/home/zehra/Desktop/Deneme/Dataset/data/train',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size

)
validation_ds =  tf.keras.preprocessing.image_dataset_from_directory(
    '/home/zehra/Desktop/Deneme/Dataset/data/val',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size)

testing_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/home/zehra/Desktop/Deneme/Dataset/data/test',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size)


# In[54]:


model=load_model('/home/zehra/Desktop/Deneme/M8.h5')
 


# In[55]:


AccuracyVector = []
plt.figure(figsize=(30, 30))
for images, labels in testing_ds.take(1):
    predictions = model.predict(images)
    predlabel = []
    prdlbl = []


# In[56]:


class_names = training_ds.class_names


# In[57]:


for mem in predictions:
       deneme=predlabel.append(class_names[np.argmax(mem)])
       print(class_names[np.argmax(mem)])
       deneme=prdlbl.append(np.argmax(mem))


# In[58]:


AccuracyVector = np.array(prdlbl) == labels
print("AccuracyVector: ", AccuracyVector)
   
sayac=0
for values in AccuracyVector:
    if(values==True):
        sayac=sayac+1
testAccuarcy=sayac/20
print("Test Accuarcy: ", testAccuarcy)


# In[ ]:





# In[ ]:




