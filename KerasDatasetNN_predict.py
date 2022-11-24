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
from tqdm import tqdm
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
    modelName = "catDogStandard.model"
    printPredFigN = -1
################################################################################################












    #loading dataset
    with open(modelName + "/trainingInput.pickle", "rb") as handle:
        trainingInputDict = pickle.load(handle)
    nameY          = trainingInputDict["nameY"]
    inputImageSize = trainingInputDict["inputImageSize"]
    testXOrig, testX = [], []

    if verbosity >= 1: print("Preparing data:")
    prepDataLoc = INPUT_LOC + ""
    while prepDataLoc[-1] == "/": prepDataLoc = prepDataLoc[:-1]
    prepDataLoc = prepDataLoc + "Prep/"
    if verbosity >= 1: print("  saving data under:", prepDataLoc)
    origPath = INPUT_LOC    + "/"
    prepPath = prepDataLoc + "/" + nameY[0] + "/"
    pathlib.Path(prepPath).mkdir(parents=True, exist_ok=True)
    for imgName in tqdm(os.listdir(origPath)):
        outImgName = prepPath + imgName.split("/")[-1]
        if os.path.isfile(outImgName) == False:
            errorOccured, origImgFile, resizedImgFile = False, None, None
            try:
                origImgFile = cv2.imread(origPath+"/"+imgName)
            except Exception as e:
                warnings.warn(str(e), Warning)
                errorOccured = True
            if (errorOccured == False) and (origImgFile is not None): 
                resizedImgFile = zeroPadCenterResize(origImgFile, inputImageSize)
                #resizedImgFile = cv2.cvtColor(resizedImgFile, cv2.COLOR_RGB2GRAY)
                cv2.imwrite(outImgName, resizedImgFile) 
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
    #prediction figures
    pathlib.Path(OUTPUT_LOC).mkdir(parents=True, exist_ok=True)
    if printPredFigN < 0: printPredFigN = np.inf
    dataTest = tf.keras.utils.image_dataset_from_directory(\
        prepDataLoc, image_size=inputImageSize, shuffle=False)
    testPaths = dataTest.file_paths
    testIdx = 0
    if printPredFigN != 0:
        testXs, testYs = next(iter(dataTest))
        testYs = testYs.numpy()
        predYweights = model.predict(testXs)
        predYs = [np.argmax(predYweight) for predYweight in predYweights] 
        for testY, predY in zip(testYs, predYs):
            testXorig = cv2.imread(testPaths[testIdx].replace("Pred", ""))
            testXorig = cv2.cvtColor(testXorig, cv2.COLOR_BGR2RGB)
            plt.imshow(testXorig, cmap=plt.cm.Spectral)
            plt.title("Prediction: "+nameY[predY], fontsize=24)
            filenameFig = OUTPUT_LOC + "predicted" + str(testIdx) + ".png"
            plt.savefig(filenameFig, dpi=100)
            plt.close()
            testIdx += 1
            if verbosity >= 1: print(" ", testIdx, nameY[predY], "\n   ", filenameFig)
            if testIdx > printPredFigN: break



    '''
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
    '''
################################################################################################
if __name__ == "__main__": main()





