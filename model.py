import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')
import os
import sys
sys.path.insert(0,os.path.dirname(__file__))

from keras.layers import Flatten, Dropout, LeakyReLU, Input, Activation
from keras.models import Model
from keras.layers.convolutional import UpSampling1D
from keras.layers import Conv1D
from keras.optimizers import Adam
from keras.datasets import mnist
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras_adversarial.legacy import Dense, BatchNormalization, Convolution2D
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from os import listdir
from os.path import isfile, join
from PIL import Image
import scipy.spatial as sp
from mfcc_utils import *
import librosa
import scipy
from scipy import dot, linalg, mat
from lstm import *

def leaky_relu(x):
    return K.relu(x, 0.2)


def get_model(input_shape=(100, 1), dropout_rate=0.5):
    #d_input = dim_ordering_input(input_shape, name="input_x")
    nch = 512
    
    d_input = Input(input_shape, name='input_X')
    #print(d_input)
    H = Conv1D(2, 50, activation= 'relu')(d_input)
    #H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Conv1D(2,  50, activation= 'relu')(H)
    #H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Dense(2)(H)
    H = LeakyReLU(0.2)(H)
    H = UpSampling1D(size = 5)(H)
    H = LeakyReLU(0.2)(H)
    H = UpSampling1D(size = 10)(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    d_V = Dense(1, activation='sigmoid')(H)
    return Model(d_input, d_V)


def mnist_process(x):
    x = x.astype(np.float32) / 255.0
    return x

def load_data():
    training_simgs = sorted(listdir("aligned_audio_data/SF1"))
    training_timgs = sorted(listdir("aligned_audio_data/TF1"))
    
    """x = np.empty((len(training_simgs), 100, 100, 3))
    y = np.empty((len(training_timgs), 100, 100, 3))
    a = np.empty((1, 100, 100, 3))"""
    for i in range(0, len(training_simgs)):
        x[i] = Image.open('aligned_audio_data/SF11/'+training_simgs[i])
        y[i] = Image.open('project_data/dummytf1/'+training_timgs[i])
        print("Training sample source..."+training_simgs[i])
        print("Training sample target..."+training_timgs[i])
        
    #return (x[0:21], y[0:21]), (x[20], y[20])
    return (x, y), (x, y)

def get_data():
    (xtrain, ytrain), (xtest, ytest) = load_data()
    return xtrain, ytrain, xtest, ytest

def loss_func(y, ypred):
    print(y.shape, ypred.shape)
    #y1 = mat(y[0])
    #ypred1 = mat(ypred[0])
    y1 = K.l2_normalize(y, axis = 1)
    ypred1 = K.l2_normalize(ypred, axis = 1)
    sim = K.mean(y1 * ypred1, axis=1)
    return (1. - sim)


if __name__ == "__main__":
    
    input_shape = (100, 1)

    conversion_model = build_model([1, 100, 200, 1])
    
    conversion_model.summary()
    #xtrain, ytrain, xtest, ytest = get_data()
    
    """conversion_model.compile(optimizer='adam',
              loss='mse')"""
              #metrics=['accuracy'])
    
    source_path = "aligned_audio_data/SF1"
    target_path = "aligned_audio_data/TF1"
    
    source_data = sorted(listdir(source_path))
    target_data = sorted(listdir(target_path))
    
    xinfo = {}
    yinfo = {}
    xvinfo = {}
    yvinfo = {}
    
    rate = 16000
    
    for i in range(151):
        j = i + 1
        xtrain, mel_filter, mel_inversion_filter, spec_thresh, shorten_factor, rate = mfcc(source_path+'/'+source_data[i])
        xinfo[i] = (mel_filter, mel_inversion_filter, spec_thresh, shorten_factor)
        
        ytrain, mel_filter, mel_inversion_filter, spec_thresh, shorten_factor, rate = mfcc(target_path+'/'+target_data[i])
        yinfo[i] = (mel_filter, mel_inversion_filter, spec_thresh, shorten_factor)
        xtrain = np.reshape(xtrain.T, (xtrain.shape[1], xtrain.shape[0], 1))#np.expand_dims(xtrain.T, axis=2)
        ytrain = np.reshape(ytrain.T, (ytrain.shape[1], ytrain.shape[0], 1))#np.expand_dims(ytrain.T, axis=2)
        print('train')
        print(xtrain.shape, ytrain.shape)
        print(xtrain)
        print('***')
        print(ytrain)
        xval, mel_filter, mel_inversion_filter, spec_thresh, shorten_factor, rate = mfcc(source_path+'/'+source_data[150-j])
        xvinfo[i] = (mel_filter, mel_inversion_filter, spec_thresh, shorten_factor)
        
        yval, mel_filter, mel_inversion_filter, spec_thresh, shorten_factor, rate = mfcc(target_path+'/'+target_data[150-j])
        yvinfo[i] = (mel_filter, mel_inversion_filter, spec_thresh, shorten_factor)
        xval = np.reshape(xval.T, (xval.shape[1], xval.shape[0], 1))#np.expand_dims(xval.T, axis=2)
        yval = np.reshape(yval.T, (yval.shape[1], yval.shape[0], 1))#np.expand_dims(yval.T, axis=2)
        print('val')
        print(xval.shape, yval.shape)
        
        
        
        conversion_model.fit(x=xtrain, y=ytrain, batch_size=1, epochs=100)#, validation_data=(xval, yval))
        
        #break
    
    conversion_model.save('lstm.h5')
    test_mfcc, mel_filter, mel_inversion_filter, spec_thresh, shorten_factor, rate = mfcc(source_path+'/100001.wav')
    predict = conversion_model.predict(np.expand_dims(test_mfcc.T, axis=2))
    print(predict.shape)
    n, m, _ = predict.shape
    predict =predict.reshape(n, m)
    res = imfcc(predict.T, mel_inversion_filter, spec_thresh, shorten_factor)
    print('res',res.shape)
    #scipy.io.wavfile.write('res.wav', rate, res)
    librosa.output.write_wav('res.wav', res, rate)
    #doubt about rate here. should try with librosa because this is resulting in corrupted file
    
    #Must look at the loss function. Seems to be not changing.