# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 23:52:46 2017

@author: Meteor
"""

%reset -f

### Import package
import os
import pandas as pd
import numpy as np
import keras
from keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
from keras.utils import np_utils
import matplotlib.image as mpimg
from scipy.misc import imread

### System Config
import platform
if platform.system() == 'Windows':
    print('__Windows__')
    path = r'C:\Users\Meteor\Desktop\Stock\captcha_folder'
else:
    print(platform.system())
    path = r'/Users/meteorchang/LocalDesktop/Stock/02.Stock/Stock/captcha_folder'

sys = platform.system()
print(path)    
#/Users/meteorchang/Documents/GitHub/DevelopingFunction/Init_config.py

### Load data from path
def LoadData(path):
    folder = os.listdir(path)
    folder.remove('.DS_Store')
    images = []
    labels = []
    labels_real = []
    folder = ['2','3','4','6','7','8','9','A',
              'C','D','E','F','G','H','J','K','L','N','P','Q',
              'R','T','U','V','X','Y','Z']
    for idx, i in enumerate(folder):
        file_path = path + '//' + i
        for j in os.listdir(file_path):
            img_path = file_path + '//' +j
            #
            img = image.load_img(img_path)
            img = img.resize((40,40))
            img_ary = image.img_to_array(img)
            img_ary_flat = img_ary.flatten()
            if len(images) == 0:
                images = img_ary_flat
            else:
                images = np.vstack((images,img_ary_flat)) # (2,4680)
            labels.append(idx)
            labels_real.append(i)
    data = np.array(images)
    labels = np.array(labels)
    labels_real = np.array(labels_real)
    return data, labels, labels_real

### Call LoadData function
data, labels, labels_real = LoadData(path)
print(data.shape)
print(labels.shape)
print(labels)
print(labels_real)

### Creat mapping table for labels
label_mapping = pd.DataFrame(labels, labels_real)
label_mapping = label_mapping.reset_index()
label_mapping = label_mapping.drop_duplicates()
label_mapping = label_mapping.reset_index()
del label_mapping['level_0']
print(label_mapping.head(5))


labels_OneHot = np_utils.to_categorical(labels)

### Shuffle data
np.random.seed(1234)
shuffle_indices = np.random.permutation(np.arange(len(labels)))
x_shuffled = data[shuffle_indices]
y_shuffled = labels[shuffle_indices]

print(y_shuffled)

### Split Train/Test set
train_rate = 0.8
train_cnt = round(len(labels)*train_rate)
test_cnt = len(labels) - train_cnt
print('Train rate|', train_rate, '| Train data cnt|', train_cnt, '| Test data cnt|', test_cnt)

X_train, X_test = x_shuffled[:train_cnt], x_shuffled[:-train_cnt]
y_train, y_test = y_shuffled[:train_cnt], y_shuffled[:-train_cnt]


### Data normalize
X_train = X_train.reshape(X_train.shape[0], -1) / 255.   
X_test = X_test.reshape(X_test.shape[0], -1) / 255.     
y_trainOneHot = np_utils.to_categorical(y_train)
y_testOneHot = np_utils.to_categorical(y_test)

### Data reshape
X_train = X_train.reshape(X_train.shape[0], 40, 40, 3).astype('float32')  
X_test = X_test.reshape(X_test.shape[0], 40, 40, 3).astype('float32')  

plt.imshow(X_train[1].reshape(40,40,3))
print(y_train[1])


###
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop     

###############################
"""
http://puremonkey2010.blogspot.tw/2017/07/toolkit-keras-mnist-cnn.html
"""
###############################
from keras.models import Sequential  
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D  
  
model = Sequential()  
# Create CN layer 1  
model.add(Conv2D(filters=16,  
                 kernel_size=(5,5),  
                 padding='same',  
                 input_shape=(40,40,3),  
                 activation='relu'))  
# Create Max-Pool 1  
model.add(MaxPooling2D(pool_size=(2,2)))  
  
# Create CN layer 2  
model.add(Conv2D(filters=36,  
                 kernel_size=(5,5),  
                 padding='same',  
                 input_shape=(40,40,3),  
                 activation='relu'))  
  
# Create Max-Pool 2  
model.add(MaxPooling2D(pool_size=(2,2)))  
  
# Add Dropout layer  
model.add(Dropout(0.25))  

model.add(Flatten())  
#
model.add(Dense(128, activation='relu'))  
model.add(Dropout(0.3))  


model.add(Dense(y_trainOneHot.shape[1], activation='softmax'))  

model.summary()  
print("")  


# 定義訓練方式  
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  
  



# 開始訓練  ㄋ
train_history = model.fit(x=X_train,  
                          y=y_trainOneHot, validation_split=0.2,  
                          epochs=100, batch_size=32, verbose=2) 



###
import os  
  
def isDisplayAvl():  
    return 'DISPLAY' in os.environ.keys()  
  
import matplotlib.pyplot as plt  
def plot_image(image):  
    fig = plt.gcf()  
    fig.set_size_inches(2,2)  
    plt.imshow(image, cmap='binary')  
    plt.show()  
  
def plot_images_labels_predict(images, labels, prediction, idx, num=10):  
    fig = plt.gcf()  
    fig.set_size_inches(12, 14)  
    if num > 25: num = 25  
    for i in range(0, num):  
        ax=plt.subplot(5,5, 1+i)  
        ax.imshow(images[idx], cmap='binary')  
        title = "l=" + str(labels[idx])  
        if len(prediction) > 0:  
            title = "l={},p={}".format(str(labels[idx]), str(prediction[idx]))  
        else:  
            title = "l={}".format(str(labels[idx]))  
        ax.set_title(title, fontsize=10)  
        ax.set_xticks([]); ax.set_yticks([])  
        idx+=1  
    plt.show()  
  
def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()  
    
from utils import *  
if isDisplayAvl():  
    show_train_history(train_history, 'acc', 'val_acc')  
    show_train_history(train_history, 'loss', 'val_loss')  



### model accuracy
scores = model.evaluate(X_test, y_testOneHot)  
print()  
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0)) 

### predict
print("\t[Info] Making prediction of X_Test4D_norm")  
prediction = model.predict_classes(X_test)  # Making prediction and save result to prediction  
print()  
print("\t[Info] Show 10 prediction result (From 240):")  
print("%s\n" % (prediction[0:10]))  
if isDisplayAvl():  
    plot_images_labels_predict(X_test, y_test, prediction, idx=0)  

### confusion matrix
import pandas as pd  
print("\t[Info] Display Confusion Matrix:")  
print("%s\n" % pd.crosstab(y_test, prediction, rownames=['label'], colnames=['predict']))
confusion = pd.crosstab(y_test, prediction, rownames=['label'], colnames=['predict'])  



