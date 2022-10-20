import sys, os, pathlib, time, math
import warnings
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = warning_on_one_line
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import tensorflow as tf
#tensorboard --logdir=... --host localhost --port 6006
#browse: http://localhost:6006/
from tensorflow.keras.activations import *
import tensorflow.keras.backend as K
#https://github.com/keras-team/keras/issues/3945
#https://stackoverflow.com/questions/64118331
K.image_data_format();
#https://towardsdatascience.com/bayesian-hyper-parameter-optimization-neural-networks
#-tensorflow-facies-prediction-example-f9c48d21f795
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from skopt import load as load_gp_minimize
from skopt.space import Integer, Real, Categorical



#model fitter###################################################################################
def learningRateFunc(epoch, initLR, minLR, decayC):
    return initLR*pow(0.1, 1.0*epoch/decayC) + minLR
def schedulerLambda(initLR, minLR, decayC):
    return lambda epoch: learningRateFunc(epoch, initLR, minLR, decayC)
def cloneLayer(layer):
    config = layer.get_config()
    weights = layer.get_weights()
    clonedLayer = type(layer).from_config(config)
    clonedLayer.build(layer.input_shape)
    clonedLayer.set_weights(weights)
    return clonedLayer
def roundedAccuracy(yTrue, yPred):
    return tf.keras.metrics.binary_accuracy(tf.round(yTrue), tf.round(yPred))
#helper funcs###################################################################################
def stand2dArray(array, mean=None, std=None):    #following tf.image.per_image_standardization
    array = np.array(array)
    flatArr = array.flatten()
    if mean is None: mean = np.mean(flatArr)
    if std is None:  std = np.std(flatArr)
    return (array - mean)/max(std, 1/math.sqrt(flatArr.size))
def get2dMean(array): return np.mean(np.array(array).flatten())
def get2dSTD(array):  return np.std( np.array(array).flatten())
def dropNaNY(inputX, inputY):
    inputXOutput = []
    inputYOutput = []
    for i, y in enumerate(inputY):
        if math.isnan(y) == False:
            inputXOutput.append(inputX[i])
            inputYOutput.append(y)
    return np.array(inputXOutput), np.array(inputYOutput)
def zeroPadCenterResize(imgFile, outputImgSize):
    colorN = 1
    if len(imgFile.shape) == 2:   height, width = imgFile.shape #note: np.shape reverses h, w
    elif len(imgFile.shape) == 3: height, width, colorN = imgFile.shape
    else: raise AssertionError("zeroPadCenterResize(): imgFile must have the image format")
    ratioHoW       = 1.0*height/width
    outputRatioHoW = outputImgSize[1]/outputImgSize[0]

    top, bottom, left, right = 0, 0, 0, 0
    if ratioHoW > outputRatioHoW:
        left   = int(np.ceil( ((1.0/outputRatioHoW)*height - width)/2))
        right  = int(np.floor(((1.0/outputRatioHoW)*height - width)/2))
    else:
        top    = int(np.ceil( (outputRatioHoW*width - height)/2))
        bottom = int(np.floor((outputRatioHoW*width - height)/2))
    outputImgFile = cv2.copyMakeBorder(imgFile, top, bottom, left, right,\
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])
    outputImgFile = cv2.resize(outputImgFile, outputImgSize)
    return outputImgFile
def cropCenterResize(imgFile, outputImgSize):
    colorN = None
    if len(imgFile.shape) == 2:   height, width = imgFile.shape #note: np.shape reverses h, w
    elif len(imgFile.shape) == 3: height, width, colorN = imgFile.shape
    else: raise AssertionError("cropCenterResize(): imgFile must have the image format")
    ratioHoW       = 1.0*height/width
    outputRatioHoW = 1.0*outputImgSize[1]/outputImgSize[0]

    outputImgFile = np.array(imgFile)
    if ratioHoW > outputRatioHoW:
        cropSize = int((height - outputRatioHoW*width)/2)
        if cropSize > 0: 
            if colorN is None: outputImgFile = np.array(imgFile[cropSize:-cropSize, :])
            else:              outputImgFile = np.array(imgFile[:, cropSize:-cropSize, :])
    else:
        cropSize = int((width - (1.0/outputRatioHoW)*height)/2)
        if cropSize > 0: 
            if colorN is None: outputImgFile = np.array(imgFile[:, cropSize:-cropSize])
            else:              outputImgFile = np.array(imgFile[:, cropSize:-cropSize, :])
    outputImgFile = cv2.resize(outputImgFile, outputImgSize)
    return outputImgFile

################################################################################################
if __name__ == "__main__": pass





