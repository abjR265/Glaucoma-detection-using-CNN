#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np 
import pandas as pd 
import cv2

import matplotlib.pyplot as plt
import seaborn as sn
import pickle
import csv
import matplotlib.image as mpimg

from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from PIL import Image
from numpy import asarray


# In[2]:


train_label = pd.read_csv("ds/glaucoma.csv")
y_train = train_label['Glaucoma']
train_label.head()


# In[6]:


image = Image.open('ds/Fundus_Train_Val_Data/Fundus_Scanes_Sorted/Validation/Glaucoma_Positive/646.jpg')

# summarize some details about the image
print(image.format)
print(image.mode)
print(image.size)

# show the image
plt.imshow(image)
image.show()
plt.title("Glaucoma_Positive image")
pixels = asarray(image)


# In[7]:


mean = pixels.mean()
print('Mean: %.3f' % mean)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

# global centering of pixels
pixels = pixels - mean

# confirm it had the desired effect
mean = pixels.mean()
print('Mean: %.3f' % mean)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
print(pixels)


# In[8]:


print('Data Type: %s' % pixels.dtype)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

# convert from integers to floats
pixels = pixels.astype('float32')

# normalize to the range 0-1
pixels /= 255.0
mean = pixels.mean()
print('pixel mean = ', mean)

# confirm the normalization
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
print(pixels)


# In[9]:


import matplotlib.pyplot as plt
fig, (ax0, ax1) = plt.subplots(1, 2)
ax0.imshow(image)
ax0.axis('off')
ax0.set_title('image')
ax1.imshow(pixels)
ax1.axis('off')
ax1.set_title('result')
plt.show()


# In[10]:


from skimage import io
def imshow(image_RGB):
  io.imshow(image_RGB)
  io.show()


# In[11]:


from tensorflow.keras.applications.densenet import preprocess_input


# In[12]:


TRAIN_DIR = 'ds/Fundus_Train_Val_Data/Fundus_Scanes_Sorted/Train'

TEST_DIR = 'ds/Fundus_Train_Val_Data/Fundus_Scanes_Sorted/Validation'


# In[13]:


from keras.preprocessing.image import ImageDataGenerator
from keras.applications import densenet
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback,TensorBoard
from tensorflow.keras.optimizers import SGD, Adam
from keras import regularizers
import keras


HEIGHT = 300
WIDTH = 300

BATCH_SIZE = 8
class_list = ["class_1", "class_2"]
FC_LAYERS = [1024, 512, 256]
dropout = 0.5
NUM_EPOCHS = 25
BATCH_SIZE = 8



def build_model(base_model, dropout, fc_layers, num_classes):
    
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        print(fc)
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

base_model_1 = densenet.DenseNet121(weights='imagenet',
                                     include_top=False,
                                     input_shape = (HEIGHT, WIDTH, 3))



train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                   rotation_range = 90,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.1,)

test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                  rotation_range = 90,
                                  horizontal_flip = True,
                                  vertical_flip = False)
train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    target_size = (HEIGHT, WIDTH),
                                                    batch_size = BATCH_SIZE)

test_generator = test_datagen.flow_from_directory(TEST_DIR,
                                                  target_size = (HEIGHT, WIDTH),
                                                  batch_size = BATCH_SIZE)



densenet_model = build_model(base_model_1,
                                      dropout = dropout,
                                      fc_layers = FC_LAYERS,
                                      num_classes = len(class_list))

adam = Adam(lr = 0.00001)
densenet_model.compile(adam, loss="binary_crossentropy", metrics=["accuracy"])

filepath = "./checkpoints" + "Densenet" + "_model_weights.h5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor = ["acc"], verbose= 1, mode = "max")
cb=TensorBoard(log_dir=("/home/ubuntu/"))
callbacks_list = [checkpoint, cb]

print(train_generator.class_indices)

densenet_model.summary()


# In[14]:


history = densenet_model.fit_generator(generator = train_generator, epochs = NUM_EPOCHS, steps_per_epoch = 25, 
                                       shuffle = True, validation_data = test_generator)


# In[15]:


image_batch,label_batch = train_generator.next()

print(len(image_batch))
for i in range(0,len(image_batch)):
    image = image_batch[i]
    print(label_batch[i])
    imshow(image)


# In[16]:


plt.figure(0)
plt.plot(history.history['accuracy'],'r')
plt.plot(history.history['val_accuracy'],'g')
plt.xticks(np.arange(0, 25, 1.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.legend(['train','validation'])
 
plt.figure(1)
plt.plot(history.history['loss'],'r')
plt.plot(history.history['val_loss'],'g')
plt.xticks(np.arange(0, 25, 1.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel("Num of Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend(['train','validation'])

plt.show()


# In[18]:


densenet_model.evaluate(test_generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False)


# In[20]:


pred = densenet_model.predict(test_generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
predicted = np.argmax(pred, axis=1)


# In[22]:


print('Confusion Matrix')
cm = confusion_matrix(test_generator.classes, np.argmax(pred, axis=1))
plt.figure(figsize = (10,10))
sn.set(font_scale=1.4) #for label size
sn.heatmap(cm, annot=True, annot_kws={"size": 12}) # font size
plt.show()
print()
print('Classification Report')
print(classification_report(test_generator.classes, predicted))


# In[ ]:




