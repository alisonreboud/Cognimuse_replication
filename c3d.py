
Skip to content
Pull requests
Issues
Marketplace
Explore
@alisonreboud
Learn Git and GitHub without any code!

Using the Hello World guide, you’ll start a branch, write comments, and open a pull request.
alisonreboud /
Cognimuse
Private

1
0

    0

Code
Issues 0
Pull requests 0
Actions
Projects 0
Security 0
Insights
Settings
Cognimuse/c3d.py /
@alisonreboud alisonreboud c3d 3cbc59b 17 minutes ago
318 lines (269 sloc) 11.2 KB
Code navigation is available!

Navigate your code with ease. Click on function and method calls to jump to their definitions or references in the same repository. Learn more

from moviepy.editor import VideoFileClip
import numpy as np
from scipy.io import loadmat
from collections import Counter
import tensorflow as tf
from numpy import save
## Kinects-i3D ##
import sys
from my_classes import DataGenerator
from keras import backend as K
K.set_image_dim_ordering('th')
from sklearn.metrics import roc_auc_score
import pandas as pd
from moviepy.editor import VideoFileClip
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
import os
import argparse
import math
import pickle
import json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD
_IMAGE_SIZE = [224,224]
_FRAMES=16
#number of files in test 2498

videos_path=['./Data/2001GLA/GLA.avi',
'./Data/2006CRA/CRA.avi',
'./Data/2007DEP/DEP.avi',
'./Data/2004LOR/LOR.avi',
'./Data/2002BMI/BMI.avi',
'./Data/2003CHI/CHI.avi']
videos=['GLA',
'CRA',
'.DEP',
'LOR',
'BMI',
'CHI']
labels_path=[
'./COGNIMUSEdatabase/SaliencyAnnotation/GLA/SaliencyAnnotation/Visual/Labs_GLA_Visual_IF23.mat',
'./COGNIMUSEdatabase/SaliencyAnnotation/CRA/SaliencyAnnotation/Visual/Labs_CRA_Visual_IF23.mat',
'./COGNIMUSEdatabase/SaliencyAnnotation/DEP/SaliencyAnnotation/Visual/Labs_DEP_Visual_IF23.mat',
'./COGNIMUSEdatabase/SaliencyAnnotation/LOR/SaliencyAnnotation/Visual/Labs_LOR_Visual_IF23.mat',
'./COGNIMUSEdatabase/SaliencyAnnotation/BMI/SaliencyAnnotation/Visual/Labs_BMI_Visual_IF23.mat',
'./COGNIMUSEdatabase/SaliencyAnnotation/CHI/SaliencyAnnotation/Visual/Labs_CHI_Visual_IF23.mat']


test='CRA'
"""
parser = argparse.ArgumentParser("FEATURE EXTRACTION")
# Dataset options
parser.add_argument('-v', '--video', type=str, required=True, help="video name required")
args = parser.parse_args()
video_path=args.video
"""


"""
>> import psutil
>>> mem = psutil.virtual_memory()
>>> mem
svmem(total=10367352832, available=6472179712, percent=37.6, used=8186245120, free=2181107712, active=4748992512, inactive=2758115328, buffers=790724608, cached=3500347392, shared=787554304, slab=199348224)
>>>
>>> THRESHOLD = 100 * 1024 * 1024  # 100MB
>>> if mem.available <= THRESHOLD:
...     print("warning")
# ### Global Variables
"""



def pad_frames(frames):
    frames_qtt = frames.shape[0]
    if (frames_qtt < _FRAMES):  # padding the frame

        pad_left_count = int((_FRAMES - frames_qtt) / 2)
        pad_right_count = _FRAMES - frames_qtt - pad_left_count

        pad_left = np.zeros((pad_left_count, frames.shape[1], frames.shape[2], frames.shape[3]))
        pad_right = np.zeros((pad_right_count, frames.shape[1], frames.shape[2], frames.shape[3]))

        rgb_array = np.concatenate((pad_left, frames, pad_right))

    #         print('Array padded')

    else:
        ##TODO: reduce the array -- CHECK IT!
        rgb_array = np.resize(frames.mean(axis=0).astype(int),
                              (_FRAMES, frames.shape[1], frames.shape[2], frames.shape[3]))
    #         print('Array resized')

    return rgb_array


def step_decay(epoch):
   initial_lrate = 0.001
   drop = 0.1
   epochs_drop = 5
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   print('hoooooooooooooooooooooo')
   print(lrate)
   print('hiiiiiiiiiiiiiiiiiiiiiiiiiiiiii')
   return lrate




def get_video_array(video_path,label_path,set,video_name):
    clip = VideoFileClip(video_path, target_resolution=(225, None))
    clip_width = clip.get_frame(0).shape[1]

    #[1]
    assert (clip_width > 224)
    # "width" should be [2], and should be > 224
    print(clip_width)
    offset = int((clip_width - 224) / 2)
    print(label_path)
    labels_video = loadmat(label_path)
    # x1=loadmat('Labs_AR_LONDON_Visual_IF23.mat')
    df=pd.DataFrame()
    try:
        label_list = (labels_video['IF23'][0])
    except:
        label_list = (labels_video['IF'][0])
    frames_list=[]
    average_labels_list=[]
    start_frame=0
    while True:
        try:
            print(start_frame)
            # frames = np.array([x for x in clip.iter_frames()])
            frames = []
            for frame in clip.iter_frames():
                frames.append(frame)
                if len(frames) == _FRAMES:  # here you have 16 frames in your frames array
                    np_frames = np.array(frames)
                    np_frames = np_frames[:, :224, offset:224+offset, :]
                    np_frames=np_frames.transpose((3, 0, 1, 2))
                    #print(np_frames.shape)
                    labels = label_list[int(start_frame):int(start_frame) + _FRAMES]
                    label = Counter(labels).most_common(1)[0][0]
                    dict={"frames":np_frames,'label':label}
                    file_name=os.path.join(set,video_name+ str(start_frame) + ".pkl")
                    pickle.dump(dict, open(file_name, "wb"))
                    # do whatever you want with it
                    frames = []  # empty to start over
                    start_frame += _FRAMES
            frames_list.append(frames)
            average_labels_list.append(frames)

        except Exception as e:
            print(e)
            for pair in enumerate(clip.iter_frames()):
                if pair[0] in range(start_frame, -1):
                    frames.append(pair[1])
            labels = label_list[int(start_frame):-1]
            if labels!=[]:
                label = Counter(labels).most_common(1)[0][0]
                f = np.array(frames)
                print(len(f))
                f = pad_frames(f)
                print(len(f))
                frames_list.append(frames)
                average_labels_list.append(frames)
            else:
                break


    return print('termine')
"""
for i in range(len(videos)):
    if test not in videos[i]:
        print(labels_path[i])
        train=get_video_array(videos_path[i], labels_path[i],'data/train_set',videos[i])
    else:
        df_test=get_video_array(videos_path[i], labels_path[i],'data/test_set',videos[i])
"""

"""
Y_test=df_test['labels']
#
#Y_train=tf.reshape(Y_train, [Y_train.shape[0],2])
"""


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
    earlystop=EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None,
                             restore_best_weights=True)
    callbacks_list = [earlystop,lrate]
    hist=model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,workers=2,epochs=30,callbacks = callbacks_list)
    pickle.dump(hist.history, open('history_0002', "wb"))
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

pickle.dump(predictions, open('predictions_0002', "wb"))

Y_test=[labels[ID] for ID in partition['validation']]
auc_score = roc_auc_score(Y_test, np.argmax(predictions, axis=1))
print(auc_score)




"""
#Initial input shape: (None, 3, 16, 112, 112)
dic1=pickle.load(open("train_set/GLA0.pkl", "rb"))
print((dic1['frames'].reshape((3, 224,224,16))).shape)
X_train=a
for file in os.listdir("train_set/"):
    loaded_file=pickle.load(open('train_set/'+file, "rb"))
"""



