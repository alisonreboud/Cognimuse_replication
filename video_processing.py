

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


#test='CRA'
parser = argparse.ArgumentParser("FEATURE EXTRACTION")
# Dataset options
parser.add_argument('-t', '--testvideo', type=str, required=True, help="video name required")
args = parser.parse_args()
test=args.testvideo

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
    
    
  def get_video_array(video_path,label_path,set,video_name):
    clip = VideoFileClip(video_path, target_resolution=(225, None))
    clip_width = clip.get_frame(0).shape[1]
    assert (clip_width > 224)
    offset = int((clip_width - 224) / 2)
    labels_video = loadmat(label_path)
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
   return print('got video blocks')

