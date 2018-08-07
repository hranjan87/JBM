
# coding: utf-8

# In[40]:


import os 
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model


# In[41]:


img_width, img_height = 150, 150
model_path = '/Users/HimanshuRanjan/MachineLearning/JBM/models/model.h5'
model_weight_path = '/Users/HimanshuRanjan/MachineLearning/JBM/models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weight_path)

def predict(imagepath):
    x = load_img(imagepath, target_size = (img_width, img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis =0)
    array = model.predict(x)
    category = array[0]
    return np.argmax(category)


# In[42]:


predict('/Users/HimanshuRanjan/MachineLearning/JBM/All_61326/test_61326/61326 (Back Side) -OK- (40).jpg')


# In[43]:


predict('/Users/HimanshuRanjan/MachineLearning/JBM/All_61326/test_61326/61326 (Front Side)-OK- (87).jpg')


# In[44]:


predict('/Users/HimanshuRanjan/MachineLearning/JBM/All_61326/test_61326/61326-SC-(20).jpg')


# In[45]:


predict('/Users/HimanshuRanjan/MachineLearning/JBM/All_61326/test_61326/61326-SD- (17).jpg')


# In[46]:


predict('/Users/HimanshuRanjan/MachineLearning/JBM/All_61326/test_61326/61326-TH- (39).jpg')


# In[47]:


predict('/Users/HimanshuRanjan/MachineLearning/JBM/All_61326/test_61326/61326-WR- (20).jpg')

