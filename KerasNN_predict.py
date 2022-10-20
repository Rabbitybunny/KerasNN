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
from _GlobalFuncs import *





################################################################################################
INPUT_LOC  = "./catDogData/zInput/"
OUTPUT_LOC = "./catDogFigPred/"
def main():
    verbosity = 3
    modelName = "catDogConv2D_.model"
    printPreproFigN = -1
    printPredFigN   = -1
################################################################################################












    #loading dataset
    convDimRequired = ("Conv2D" in modelName) or ("RNN" in modelName) or ("ResNet" in modelName)
    with open(modelName + "/trainingInput.pickle", "rb") as handle:
        trainingInputDict = pickle.load(handle)
    nameY          = trainingInputDict["nameY"]
    inputImageSize = trainingInputDict["inputImageSize"]
    testXOrig, testX = [], []

    if verbosity >= 1: print("Loading data:")
    for imgName in os.listdir(INPUT_LOC):
        errorOccured, origImgFile, resizedImgFile = False, None, None
        try:
            origImgFile = cv2.imread(INPUT_LOC+"/"+imgName)
            origImgFile = cv2.cvtColor(origImgFile, cv2.COLOR_BGR2RGB)
        except Exception as e:
            warnings.warn(str(e), Warning)
            errorOccured = True
        if errorOccured == False: 
            resizedImgFile = cv2.cvtColor(origImgFile, cv2.COLOR_RGB2GRAY)
            resizedImgFile = zeroPadCenterResize(resizedImgFile, inputImageSize)
            #resizedImgFile = cropCenterResize(resizedImgFile, inputImageSize) 
            testXOrig.append(origImgFile)
            testX    .append(resizedImgFile)
    pathlib.Path(OUTPUT_LOC).mkdir(parents=True, exist_ok=True)
    if printPreproFigN != 0:
        if verbosity >= 1: print("Saving preprocessed figures:")
        for idx, valX in enumerate(testX[:printPreproFigN]):
            plt.imshow(valX, cmap=plt.cm.binary)
            filenameFig = OUTPUT_LOC + "preprocessed"+str(idx)+"_.png"
            plt.savefig(filenameFig, dpi=100)
            plt.close()
            if verbosity >= 1: print(" "+str(idx)+": "+filenameFig)
#prediction#####################################################################################
    if verbosity >= 1:
        print("###############################################################MODEL PREDICTION")
    #loading trained data
    histDF, optParDict = {}, None
    try:
        model = tf.keras.models.load_model(modelName)
        histDFs, parDicts = {}, {}
        with open(modelName + "/history.pickle", "rb") as handle:
            histDFs = pickle.load(handle) 
        with open(modelName + "/pars.pickle", "rb") as handle:
            parDicts = pickle.load(handle)
        histDF     = histDFs["opt"]
        optParDict = parDicts["opt"]
        if verbosity >= 1: print("\nOptimal parameters:\n   ", optParDict)
        if verbosity >= 2: print(model.summary())
        if verbosity >= 3: 
            for key in parDicts: print(key+":", parDicts[key])
    except OSError or FileNotFoundError:
        print("No trained model is found:\n    ", modelName)
        sys.exit(0)
    except:
        raise
    #data normalization/standardization + data dim requirement
    testXNorm = np.array([stand2dArray(X) for X in testX])
    if convDimRequired: 
        inputShape = [testXNorm.shape[1], testXNorm.shape[2], 1] #note: needed for conv2D
        testXNorm = testXNorm.reshape(testXNorm.shape[0], *inputShape)
    #prediction figures
    if printPredFigN != 0:
        if verbosity >= 1: print("Saving prediction figures:")
        predValY = model.predict(testXNorm)
        predY = np.argmax(predValY, axis=-1)
        for idx, valX in enumerate(testXOrig[:min(len(testXOrig), printPredFigN)]):
            plt.imshow(valX, cmap=plt.cm.Spectral)
            plt.title("Prediction: "+nameY[predY[idx]], fontsize=24)
            filenameFig = OUTPUT_LOC + "predicted"+str(idx)+".png"
            plt.savefig(filenameFig, dpi=100)
            plt.close()
            if verbosity >= 1: print(" "+str(idx)+"=>"+nameY[predY[idx]]+": "+filenameFig)
################################################################################################
if __name__ == "__main__": main()





