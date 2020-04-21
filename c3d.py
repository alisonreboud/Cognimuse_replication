

import numpy as np
from my_classes import DataGenerator
from keras import backend as K
K.set_image_dim_ordering('th')
from sklearn.metrics import roc_auc_score
import pandas as pd

from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
import os
import math
import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD

#number of files in test 2498



def step_decay(epoch):
   initial_lrate = 0.001
   drop = 0.1
   epochs_drop = 7
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   print(lrate)
   return lrate


def run_c3d(training_generator,validation_generator,test_generator):
    model = Sequential()
    # 1st layer group
    model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv1',
                            subsample=(1, 1, 1),
                            input_shape=(3, 16, 224, 224)))
              #input_shape=(n_cols,)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv2',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3b',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4b',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5b',
                            subsample=(1, 1, 1)))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(2, activation='softmax', name='fc8'))
    model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
    # Train model on dataset
    lrate = LearningRateScheduler(step_decay)
    #earlystop=EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None,
                             #restore_best_weights=True)
    #callbacks_list = [earlystop,lrate]
    callbacks_list = [lrate]
    hist=model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,workers=2,epochs=15,callbacks = callbacks_list)
    pickle.dump(hist.history, open('history_0002.pkl', "wb"))
    predictions= model.predict_generator(generator=test_generator)
    #predictions = model.predict(validation_data)
    #if summary:
        #print(model.summary())
    return predictions


# Parameters
params = {'dim': (16,224,224),
          'batch_size': 10,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': True}
params_test = {'dim': (16,224,224),
          'batch_size': 2,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': False}
# Datasets
partition = {'train':['train_set/'+ file for file in os.listdir("data/train_set/")],'validation':['test_set/'+file for file in os.listdir("data/test_set/")]}
labels = {}
for file in os.listdir("data/train_set/"):
    loaded_file = pickle.load(open('data/train_set/' + file, "rb"))
    labels.update({'train_set/'+file:loaded_file['label']})
for file in os.listdir("data/test_set/"):
    loaded_file = pickle.load(open('data/test_set/' + file, "rb"))
    labels.update({'test_set/'+file: loaded_file['label']})

a=pickle.load(open('data/train_set/LOR35776.pkl','rb'))
print(a['frames'].shape)

a=pickle.load(open('data/train_set/BMI4272.pkl','rb'))
print(a['frames'].shape)

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)
test_generator= DataGenerator(partition['validation'], labels, **params_test)

predictions=run_c3d(training_generator,validation_generator,test_generator)

pickle.dump(predictions, open('predictions.pkl', "wb"))

Y_test=[labels[ID] for ID in partition['validation']]
auc_score = roc_auc_score(Y_test, np.argmax(predictions, axis=1))
print(auc_score)




