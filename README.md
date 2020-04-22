# Cognimuse_replication

This paper intends to replicate

Exploring CNN-based architectures for Multimodal Salient Event Detection in Videos Petros Koutras, Athanasia Zlatinsi and Petros Maragos
School of E.C.E., National Technical University of Athens, 15773 Athens, Greece

Using the C3D implementation available at https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2

Data made available byt the Cognimuse team upon request.

1) Video processing 

video_processing.py -t 'CRA'
 
 -t indicates the test video, should be one chosen among videos=['GLA','CRA','.DEP','LOR','BMI','CHI']

Paths to videos and labels hard coded in file.



Saves training set under 
data/train_set/

Saves testing set under 
data/test_set/

2) Running the model 

c3d.py 

runs c3d model and returns auc, saving history and predictions to 'history_000.pkl', 'predictions.pkl' respectively

Hard coded parameters in this file (following the instructions from the paper)

epochs_drop = 7
initial_lrate = 0.001
drop = 0.1
epochs=15

params = {'dim': (16,224,224),
         'batch_size': 30,
         'n_classes': 2,
         'n_channels': 3,
         'shuffle': True}
params_test = {'dim': (16,224,224),
         'batch_size': 2,
         'n_classes': 2,
         'n_channels': 3,
         'shuffle': False}
         
Earlystop also available (needs to be commented out)

