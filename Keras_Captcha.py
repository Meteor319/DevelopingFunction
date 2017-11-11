# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 23:52:46 2017

@author: Meteor
"""

%reset -f

#
import os
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
from keras.utils import np_utils

import matplotlib.image as mpimg
from scipy.misc import imread
#
path = r'C:\Users\Meteor\Desktop\Stock\captcha_folder'
#
def LoadData(path):
    folder = os.listdir(path)
    images = []
    labels = []
    for idx, i in enumerate(folder):
        file_path = path + '\\' + i
        for j in os.listdir(file_path):
            img_path = file_path + '\\' +j
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
    
    data = np.array(images)
    labels = np.array(labels)
    labels = np_utils.to_categorical(labels, len(folder))
    return data, labels
#
data, labels = LoadData(path)

# Need data shuffle
X_train = data[1:500]
y_train = labels[1:500]

X_test = data[300:]
y_test = labels[300:]
print(data[1].shape)
            
#
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop     


from keras.datasets import mnist
"""
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
y_train = np_utils.to_categorical(y_train, 36)
y_test = np_utils.to_categorical(y_test, 36)
"""



       
model = Sequential([
    Dense(512, input_dim=4800),
    Activation('relu'),
    Dense(36),
    Activation('softmax'),
])
    
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
            
# We add metrics to get more results you want to see
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])



print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, nb_epoch =2000, batch_size=128)


print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)



################################

import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop     


from keras.datasets import mnist

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)




       
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
            
# We add metrics to get more results you want to see
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])



print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, nb_epoch =2, batch_size=32)


print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)






