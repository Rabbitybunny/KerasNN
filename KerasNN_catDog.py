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
OPTITER, OPTACCU, OPTASTD = 0, 0, 0
################################################################################################
DATA_LOC     = "./catDogData/full/"
TESTDATA_LOC = "./catDogData/zTest/"
FIG_LOC      = "./catDogFig/"
RAND_SEED = 1
def main():
    verbosity = 2

    modelName = "catDogConv2D.model"
    trainOn   = True                #False to test the currently saved model
    printRawFigN  = 10
    printPredFigN = 10
    #dataset 
    testRatio       = 0.1
    dropRatio       = 0.0           #ratio of data to simulate unlabeled Y's
    validationRatio = 0.1           #ratio of data for validation

    #trainings
    trainAutoencoderOn = False
    autoEpochN         = 30
   
    optModelSearchOn  = False
    optimizationCoreN = -1      #-1 to use all CPU cores
    optimizationCallN = 30      #note: increase to a difference >= 10 when reloading
    learningEpochN    = 10      #note: equilibrium needed if # of MC dropout layer varies
    bootstrappingN    = 6
    
    retrainOptModelOn = True
    learningEpochNOpt = 30
    bootstrappingNOpt = 1

    #model setup
    learningRate = Real(   low=1E-6, high=1E-1, prior="log-uniform",     name="learningRate")
    convLayerN   = Integer(low=1,    high=3,                             name="convLayerN")
    convFilterN  = Categorical(categories=[32, 64, 128],                 name="convFilterN")
    denseLayerN  = Integer(low=1,    high=5,                             name="denseLayerN")
    denseNeuronN = Integer(low=10,   high=500,                           name="denseNeuronN")
    actFunc  = Categorical(categories=["relu", "elu", "selu"],           name="actFunc")
    initFunc = Categorical(categories=["he_normal", "he_uniform"],       name="initFunc")

    dims, par0 = [], []
    if "SimpleDense" in modelName:
        optModelSearchOn = False
        bootstrappingNOpt = 1
        denseNeuronN2 = Integer(low=10,   high=500,                      name="denseNeuronN2")
        dims = [learningRate, denseNeuronN, denseNeuronN2]
        par0 = [1E-2,         300,          100]
    elif "Dense" in modelName:
        bootstrappingN    = 1
        bootstrappingNOpt = 1
        actFunc = Categorical(categories=["relu", "sigmoid"],            name="actFunc")
        dims = [learningRate, denseLayerN, denseNeuronN, actFunc]
        par0 = [1E-3,         3,           128,          "relu"]
    elif "Standard" in modelName:
        dims = [learningRate, denseLayerN, denseNeuronN, actFunc, initFunc]
        par0 = [1E-3,         3,           128,           "elu",   "he_normal"]
    elif "Conv2D" in modelName:
        denseLayerN  = Integer(low=1,    high=5,                         name="denseLayerN")
        dims=[learningRate,convLayerN,convFilterN,denseLayerN,denseNeuronN,actFunc,initFunc]
        par0=[1E-3,        2,         64,         3,          128,        "elu",   "he_normal"]
    elif "RNN" in modelName:
        dims = [learningRate]
        par0 = [1E-3]
    elif "ResNet50" in modelName:
        dims = [learningRate]
        par0 = [1E-3]
    convDimRequired = ("Conv2D" in modelName) or ("RNN" in modelName)#or ("ResNet" in modelName)
    if verbosity >= 1: print("\n####################################################RUN STARTS")
#dataset########################################################################################
    nameY = ["dog", "cat"]
    inputImageSize = (128, 128)     #(224, 224) for ResNet
    inputXFull, inputYFull, testX, testY = [], [], [], []

    if verbosity >= 1: print("Loading data:")
    dataTrain, dataTest = [], []
    for yIter, label in enumerate(nameY):
        trainPath = DATA_LOC + "/" + label + "/"
        for imgName in os.listdir(trainPath):
            errorOccured, origImgFile, resizedImgFile = False, None, None
            try:
                #stackoverflow.com/questions/9131992
                #github.com/ImageMagick/ImageMagick/discussions/2754, just remove the ~ files
                origImgFile = cv2.imread(trainPath+"/"+imgName)
            except Exception as e:
                warnings.warn(str(e), Warning)
                errorOccured = True
            if (errorOccured == False) and (origImgFile is not None): 
                resizedImgFile = zeroPadCenterResize(origImgFile, inputImageSize)
                resizedImgFile = cv2.cvtColor(resizedImgFile, cv2.COLOR_RGB2GRAY)
                dataTrain.append([resizedImgFile, yIter])
        testPath = TESTDATA_LOC + "/" + label + "/"
        for imgName in os.listdir(testPath):
            errorOccured, origImgFile, resizedImgFile = False, None, None
            try:
                origImgFile = cv2.imread(testPath+"/"+imgName)
            except Exception as e:
                warnings.warn(str(e), Warning)
                errorOccured = True
            if (errorOccured == False) and (origImgFile is not None):
                resizedImgFile = zeroPadCenterResize(origImgFile, inputImageSize)
                resizedImgFile = cv2.cvtColor(resizedImgFile, cv2.COLOR_RGB2GRAY)
                dataTest.append([resizedImgFile, yIter])
    np.random.shuffle(dataTrain)
    for X, Y in dataTrain:
        inputXFull.append(X)
        inputYFull.append(Y)
    for X, Y in dataTest:
        testX.append(X)
        testY.append(Y)
    


    ############################################################################################
    #simulate unlabeled Y's
    np.random.seed(RAND_SEED)
    inputY = []
    for y in inputYFull:
        if np.random.uniform() < dropRatio: inputY.append(float("NaN"))
        else:                               inputY.append(y)
    #data normalization/standardization
    inputY = np.array(inputY)
    inputXNorm = np.array([stand2dArray(X) for X in inputXFull])
    targetN    = len(nameY)
    inputShape = [inputXNorm.shape[1], inputXNorm.shape[2]]
    #data dim requirement
    if convDimRequired:
        inputShape = [inputXNorm.shape[1], inputXNorm.shape[2], 1]     #note: needed for conv2D
        inputXNorm = inputXNorm.reshape(inputXNorm.shape[0], *inputShape)
    #raw figures
    pathlib.Path(FIG_LOC).mkdir(parents=True, exist_ok=True)
    if printRawFigN > 0:
        if verbosity >= 1: print("Saving preprocessed figures:")
        for idx, valX in enumerate(inputXFull[:printRawFigN]):
            plt.imshow(valX, cmap=plt.cm.binary)
            plt.title(nameY[inputYFull[idx]], fontsize=24)
            filenameFig = FIG_LOC + "preprocessed"+str(idx)+".png"
            plt.savefig(filenameFig, dpi=100)
            plt.close()
            if verbosity >= 1:
                print(" ", idx, nameY[inputYFull[idx]])
                print("   ", filenameFig)
    #saving input values
    trainingInputDict = {}
    trainingInputDict["RAND_SEED"]      = RAND_SEED
    trainingInputDict["nameY"]          = nameY
    trainingInputDict["inputImageSize"] = inputImageSize
    trainingInputDict["modelName"]     = modelName
    trainingInputDict["trainOn"]       = trainOn
    trainingInputDict["printRawFigN"]  = printRawFigN
    trainingInputDict["printPredFigN"] = printPredFigN
    trainingInputDict["testRatio"]       = testRatio
    trainingInputDict["dropRatio"]       = dropRatio
    trainingInputDict["validationRatio"] = validationRatio
    trainingInputDict["trainAutoencoderOn"] = trainAutoencoderOn
    trainingInputDict["autoEpochN"]         = autoEpochN
    trainingInputDict["optModelSearchOn"]  = optModelSearchOn
    trainingInputDict["optimizationCoreN"] = optimizationCoreN
    trainingInputDict["optimizationCallN"] = optimizationCallN
    trainingInputDict["learningEpochN"]    = learningEpochN
    trainingInputDict["bootstrappingN"]    = bootstrappingN
    trainingInputDict["retrainOptModelOn"]  = retrainOptModelOn 
    trainingInputDict["learningEpochNOpt"]  = learningEpochNOpt
    trainingInputDict["bootstrappingNOpt"]  = bootstrappingNOpt

    pathlib.Path(modelName).mkdir(parents=True, exist_ok=True)
    filenameInput = modelName + "/trainingInput.pickle"
    if os.path.isfile(filenameInput) == False:
        with open(filenameInput, "wb") as handle:
            pickle.dump(trainingInputDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if verbosity >= 1: print("Saving input training parameters:\n    ", filenameInput)
    if verbosity >= 1:
        print("Loading dataset parameters:")
        print("  trainOn      :", trainOn)
        print("  printRawFigN :", printRawFigN)
        print("  printPredFigN:", printPredFigN)
        print("  testRatio      :", testRatio)
        print("  dropRatio      :", dropRatio)
        print("  validationRatio:", validationRatio)
        print("Loading training parameters:")
        print("  trainAutoencoderOn:", trainAutoencoderOn)
        print("  autoEpochN        :", autoEpochN)
        print("  optModelSearchOn :", optModelSearchOn)
        print("  optimizationCallN:", optimizationCallN)
        print("  learningEpochN   :", learningEpochN )
        print("  bootstrappingN   : ", bootstrappingN)
        print("  retrainOptModelOn :", retrainOptModelOn)
        print("  learningEpochNOpt :", learningEpochNOpt)
        print("  bootstrappingNOpt :", bootstrappingNOpt)
#autoencoder####################################################################################
    if trainOn == False:
        trainAutoencoderOn = False
        optModelSearchOn   = False
        retrainOptModelOn  = False    
    pretrainedLayers = []
    if trainAutoencoderOn == True:
        if verbosity >= 1:
            print("####################################################AUTOENCODER PRETRAINING")
        encodedXuntrained = None
        trainX, validX, trainY, validY = train_test_split(inputXNorm, inputY, test_size=0.1,\
                                                          shuffle=False)
        #train encoder, decoder, autoencoder
        encoder = buildEncoderConv(inputShape, regularization="dropout")
        encoderOutputShape = list(encoder.layers[-1].output_shape)
        encoderOutputShape = [s for s in encoderOutputShape if s is not None]
        decoder = buildDecoderConv(inputShape=encoderOutputShape)

        autoEncoder = buildAutoEncoder(encoder, decoder)
        encodedXuntrained = encoder.predict(validX)
        tensorboardAutoDir = modelName + "/tensorboardAutoDir/"
        tensorboardAutoDir += str(int(time.time())) + "Conv"
        tensorboardAuto = tf.keras.callbacks.TensorBoard(tensorboardAutoDir)
        history = autoEncoder.fit(trainX, trainX, validation_data=(validX, validX),\
                                  epochs=autoEpochN, callbacks=[tensorboardAuto])
        encoder    .save(modelName+"/zEncoder.model")
        decoder    .save(modelName+"/zDecoder.model")
        autoEncoder.save(modelName+"/zAutoEncoder.model")
        os.rename(tensorboardAutoDir, tensorboardAutoDir.replace("tensorboardAutoDir/",\
                                                                 "tensorboardAutoDir/Fin"))
        #compressed figures   
        encoder     = tf.keras.models.load_model(modelName+"/zEncoder.model",    compile=False)
        decoder     = tf.keras.models.load_model(modelName+"/zDecoder.model",    compile=False)
        autoEncoder = tf.keras.models.load_model(modelName+"/zAutoEncoder.model",compile=False)
        if printFigN > 0:
            print("Saving the following figures:")
            encodedX = encoder.predict(validX)
            cmprsX   = autoEncoder.predict(validX)
            validX = validX.reshape(*validX.shape[:-1])
            cmprsX = cmprsX.reshape(*cmprsX.shape[:-1])
            printTSNE(encodedXuntrained, validY, nameY, "TSNEuntrained", verbosity=verbosity)
            printTSNE(encodedX,          validY, nameY, "TSNE",          verbosity=verbosity)
            for idx, valX in enumerate(validX[:printFigN]):
                fig = plt.figure(figsize=(12, 6))
                gs = gridspec.GridSpec(1, 2)
                ax = []
                for i in range (gs.nrows*gs.ncols): ax.append(fig.add_subplot(gs[i]))
                ax[0].imshow(valX, cmap=plt.cm.binary)
                ax[0].set_title("Original Normalized", fontsize=24)
                ax[1].imshow(cmprsX[idx], cmap=plt.cm.binary)
                ax[1].set_title("Autoencoder Compressed", fontsize=24) 
                filenameFig = FIG_LOC + "compressed"+str(idx)+".png"
                plt.savefig(filenameFig, dpi=100)
                plt.close()
                print("   ", filenameFig)
        #encoder for autoencoder pretraining
        encoder = tf.keras.models.load_model(modelName+"/zEncoder.model", compile=False)
        for i, layer in enumerate(encoder.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                #layer.trainable = False
                pretrainedLayers.append(layer)
#####searching for optimal model################################################################
    inputXNorm, inputY = dropNaNY(inputXNorm, inputY)  #dropping out untagged events
    if optModelSearchOn == True:
        if verbosity >= 1:
            print("################################################SEARCHING FOR OPTIMAL MODEL")
        fitFunc = fitFuncLambda(modelName, dims, targetN, inputShape, inputXNorm, inputY, \
                                validationRatio, learningEpochN, bootstrappingN,\
                                pretrainedLayers=pretrainedLayers, verbosity=verbosity)
        checkpointPath = modelName + "/checkpoint.pkl"
        checkpointSaver = CheckpointSaver(checkpointPath, compress=9, store_objective=False)
        optParDict, eval0 = {}, None
        #restore gp_minimize: remember to delete the .pkl file when changing model
        global OPTITER, OPTACCU, OPTASTD
        try:
            restoredOpt = load_gp_minimize(checkpointPath)
            par0, eval0 = restoredOpt.x_iters, restoredOpt.func_vals
            print("Reading the checkpoint file:\n    ", checkpointPath)
            OPTITER = len(eval0)
            optimizationCallN -= (OPTITER + 1)
            parDicts = {}
            with open(modelName + "/pars.pickle", "rb") as handle:
                parDicts = pickle.load(handle)
            optParDict = parDicts["opt"]
            OPTACCU = optParDict["val_accuracy"]
            OPTASTD = optParDict["val_accu_std"]
            if verbosity >= 2:
                model = tf.keras.models.load_model(modelName)
                print(model.summary())
                print("Parameters:\n   ", optParDict)
            print("Current optimal validation accuracy:")
            print("   ", OPTACCU, "+/-", (OPTASTD if (OPTASTD > 0) else "NA"))
        except FileNotFoundError:
            print("Saving checkpoint file:\n    ", checkpointPath)
        except:
            raise
        #main optimization
        if optimizationCallN > 0:
            result = gp_minimize(func=fitFunc, dimensions=dims, x0=par0,y0=eval0,acq_func="EI",\
                                 n_jobs=optimizationCoreN, n_calls=optimizationCallN,\
                                 callback=[checkpointSaver])
#retrain optimal model##########################################################################
    if retrainOptModelOn == True:
        if verbosity >= 1:
            print("######################################################RETRAIN OPTIMAL MODEL")
        optParDict, parOpt = {}, []
        try:
            parDicts = {}
            with open(modelName + "/pars.pickle", "rb") as handle:
                parDicts = pickle.load(handle)
                optParDict = parDicts["opt"]
            for dim in dims: parOpt.append(optParDict[dim.name])
        except FileNotFoundError:
            parOpt = par0.copy()
            print("The parameter file from model optimization is not found:")
            print("   ", modelName + "/pars.pickle")
            print("Using par0 as the optimized parameters")
        except:
            raise
        fitFuncOpt = fitFuncLambda(modelName, dims, targetN, inputShape, inputXNorm, inputY,\
                                   validationRatio, learningEpochNOpt, bootstrappingNOpt,\
                                   pretrainedLayers=pretrainedLayers, verbosity=verbosity)
        optAccuracy = fitFuncOpt(parOpt)
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
        if verbosity >= 2: print(model.summary())
    except OSError or FileNotFoundError:
        print("No trained model is found:\n    ", modelName)
        sys.exit(0)
    except:
        raise
    #data normalization/standardization + data dim requirement
    testY = np.array(testY)
    testXNorm = np.array([stand2dArray(X) for X in testX])
    if convDimRequired: testXNorm = testXNorm.reshape(testXNorm.shape[0], *inputShape)
    #evaluation
    model.evaluate(x=testXNorm, y=testY)
    histDF.plot(figsize=(8, 5))
    plt.title("Learning Performance History")
    plt.grid("True")
    plt.gca().set_ylim(0.0, 1.0)
    filenameFig = FIG_LOC + "-optModel_learningHistory.png"
    plt.savefig(filenameFig)
    plt.close()
    if verbosity >= 1: print("Saving training result/prediction figures:\n    ", filenameFig)
    #prediction figures
    if printPredFigN != 0:
        predValY = model.predict(testXNorm)
        predY = np.argmax(predValY, axis=-1)
        for idx, valX in enumerate(testX[:min(len(testX), printPredFigN)]):    
            plt.imshow(valX, cmap=plt.cm.binary)
            plt.title("Prediction: "+nameY[predY[idx]], fontsize=24)
            filenameFig = FIG_LOC + "predicted"+str(idx)+".png"
            plt.savefig(filenameFig, dpi=100)
            plt.close()
            if verbosity >= 1:
                print(" ", idx, nameY[predY[idx]], nameY[testY[idx]])
                print("   ", filenameFig)






################################################################################################
################################################################################################
################################################################################################
#model##########################################################################################
def buildModel(modelName, pars, dims, targetN, inputShape, pretrainedLayers=[]):
    if "SimpleDense" in modelName:
        return modelDenseSimple(pars, dims, targetN, inputShape, pretrainedLayers)
    elif "Dense" in modelName:
        return modelDense(pars, dims, targetN, inputShape, pretrainedLayers)
    elif "Standard" in modelName:
        return modelStandard(pars, dims, targetN, inputShape, pretrainedLayers)
    elif "Conv2D" in modelName:
        return modelConv2D(pars, dims, targetN, inputShape, pretrainedLayers)
    elif "RNN" in modelName:
        return modelRNN(pars, dims, targetN, inputShape, pretrainedLayers)
    elif "ResNet50" in modelName:
        return modelResNet50(pars, dims, targetN, inputShape, pretrainedLayers)
    else:
        print("No model found:\n    ", modelName)
        sys.exit(0)
def modelDenseSimple(pars, dims, targetN, inputShape, pretrainedLayers=[]):
    #dims: learningRate, denseNeuronN, denseNeuronN2
    parDict = {}
    for par, dim in zip(pars, dims): parDict[dim.name] = par
    
    model = tf.keras.models.Sequential()
    for layer in pretrainedLayers: model.add(cloneLayer(layer))
    if pretrainedLayers == []: model.add(tf.keras.layers.Flatten(input_shape=inputShape))

    model.add(tf.keras.layers.Dense(parDict["denseNeuronN"],  activation="relu"))
    model.add(tf.keras.layers.Dense(parDict["denseNeuronN2"], activation="relu"))
    model.add(tf.keras.layers.Dense(targetN,                  activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=parDict["learningRate"]),\
                  loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=["accuracy"])
    return model
def modelDense(pars, dims, targetN, inputShape, pretrainedLayers=[]):
    #dims: learningRate, denseLayerN, denseNeuronN, actFunc
    parDict = {}
    for par, dim in zip(pars, dims): parDict[dim.name] = par

    model = tf.keras.models.Sequential()
    for layer in pretrainedLayers: model.add(cloneLayer(layer))
    if pretrainedLayers == []: model.add(tf.keras.layers.Flatten(input_shape=inputShape))

    for i in range(parDict["denseLayerN"]):
        model.add(tf.keras.layers.Dense(parDict["denseNeuronN"], activation=parDict["actFunc"]))
    model.add(tf.keras.layers.Dense(targetN, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=parDict["learningRate"]),\
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])
    return model
def modelStandard(pars, dims, targetN, inputShape, pretrainedLayers=[]):
    #dims: learningRate, denseLayerN, denseNeuronN, actFunc, initFunc
    #############Adjustables#############
    dropoutRate   = 0.2
    momentumRatio = 0.9
    #####################################
    parDict = {}
    for par, dim in zip(pars, dims): parDict[dim.name] = par

    model = tf.keras.models.Sequential()
    for layer in pretrainedLayers: model.add(cloneLayer(layer))
    if pretrainedLayers == []: model.add(tf.keras.layers.Flatten(input_shape=inputShape))

    for i in range(parDict["denseLayerN"]):
        model.add(tf.keras.layers.Dense(parDict["denseNeuronN"],activation=parDict["actFunc"],\
                                        kernel_initializer=parDict["initFunc"]))
        model.add(dropoutMC(rate=dropoutRate))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(parDict["denseNeuronN"],activation=parDict["actFunc"],\
                                        kernel_initializer=parDict["initFunc"]))
    model.add(tf.keras.layers.Dense(targetN, activation="softmax"))
    optimizer = tf.keras.optimizers.SGD(learning_rate=parDict["learningRate"],\
                                        momentum=momentumRatio, nesterov=True)
    model.compile(optimizer=optimizer,\
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])
    return model
#https://stats.stackexchange.com/questions/240305
def modelConv2D(pars, dims, targetN, inputShape, pretrainedLayers=[]):
    #dims: learningRate, convLayerN, convFilterN, denseLayerN, denseNeuronN, actFunc, initFunc
    #############Adjustables#############
    convLayerNinit   = 64
    convFilterNinit  = (8, 8)
    convDropoutRate  = 0.1
    denseDropoutRate = 0.5
    momentumRatio    = 0.9
    #####################################
    parDict = {}
    for par, dim in zip(pars, dims): parDict[dim.name] = par

    model = tf.keras.models.Sequential()
    for layer in pretrainedLayers: model.add(cloneLayer(layer))
    if pretrainedLayers == []:
        model.add(tf.keras.layers.Conv2D(convLayerNinit, convFilterNinit, 
                                         activation=parDict["actFunc"],\
                                         padding="SAME", input_shape=inputShape))
    for i in range(parDict["convLayerN"]):
        if pow(2, i+1) < min(inputShape[0], inputShape[1]): 
            model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        filterN = max(8, parDict["convFilterN"]/pow(2, parDict["convLayerN"]-1-i))
        model.add(tf.keras.layers.Conv2D(filterN, (3, 3), activation=parDict["actFunc"],\
                                         padding="SAME"))
        model.add(tf.keras.layers.Dropout(rate=convDropoutRate))
    model.add(tf.keras.layers.Flatten())
    for i in range(parDict["denseLayerN"]):
        neutronN = max(8, parDict["denseNeuronN"]/pow(2, i))
        model.add(tf.keras.layers.Dropout(rate=denseDropoutRate))
        model.add(tf.keras.layers.Dense(neutronN, activation=parDict["actFunc"],\
                                        kernel_initializer=parDict["initFunc"]))
    model.add(tf.keras.layers.Dense(targetN, activation="softmax"))
    optimizer = tf.keras.optimizers.SGD(learning_rate=parDict["learningRate"],\
                                        momentum=momentumRatio, nesterov=True)
    model.compile(optimizer=optimizer,\
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])
    return model
#https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-
#convolutional-neural-networks/
def modelRNN(pars, dims, targetN, inputShape, pretrainedLayers=[]):
    #dims: learningRate
    #############Adjustables#############
    convLayerNinit  = 64
    convFilterNinit = (8, 8)
    dropoutRate     = 0.5
    momentumRatio   = 0.9
    deepLayers      = [64]*3 + [128]*4 + [256]*6 + [512]*3
    #####################################
    parDict = {}
    for par, dim in zip(pars, dims): parDict[dim.name] = par

    inputZ = tf.keras.layers.Input(inputShape)
    Z = inputZ + 0
    for layer in pretrainedLayers: Z = layer(Z)
    if pretrainedLayers == []:
        Z = tf.keras.layers.Conv2D(convLayerNinit, convFilterNinit, strides=2,\
                                   activation="relu", padding="SAME")(Z)
    Z = tf.keras.layers.BatchNormalization()(Z)
    Z = tf.keras.layers.Activation("relu")(Z)
    Z = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="SAME")(Z)
    
    filterNpre = 64
    for filterN in deepLayers:
        strideN = (1 if filterN == filterNpre else 2);
        Z = residualBlock(Z, filterN, strideN=strideN)
        filterNpre = filterN*1
    
    Z = tf.keras.layers.GlobalAvgPool2D()(Z)
    Z = tf.keras.layers.Flatten()(Z)
    Z = tf.keras.layers.Dropout(rate=dropoutRate)(Z)
    Z = tf.keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal")(Z)
    Z = tf.keras.layers.Dense(targetN, activation="softmax")(Z)
    model = tf.keras.models.Model(inputs=inputZ, outputs=Z)
    optimizer = tf.keras.optimizers.SGD(learning_rate=parDict["learningRate"],\
                                        momentum=momentumRatio, nesterov=True)
    model.compile(optimizer=optimizer,\
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])
    return model
def residualBlock(inputZ, filterN, strideN=1):
    mainZ = inputZ + 0
    mainZ = tf.keras.layers.Conv2D(filterN, [3, 3], strides=strideN, padding="SAME",\
                                   use_bias=False)(mainZ)
    mainZ = tf.keras.layers.BatchNormalization()  (mainZ)
    mainZ = tf.keras.layers.Activation("relu")    (mainZ)
    mainZ = tf.keras.layers.Conv2D(filterN, [3, 3], strides=1, padding="SAME",\
                                   use_bias=False)(mainZ)
    mainZ = tf.keras.layers.BatchNormalization()  (mainZ)

    skipZ = inputZ + 0
    if strideN > 1:             #patch size of [1, 1] is the key
        skipZ = tf.keras.layers.Conv2D(filterN, [1, 1], strides=strideN, padding="SAME",\
                                       use_bias=False)(skipZ)
        skipZ = tf.keras.layers.BatchNormalization()  (skipZ)
    
    mergedZ = tf.keras.layers.add([mainZ, skipZ])
    mergedZ = tf.keras.layers.Activation("relu")(mergedZ)
    return mergedZ
#https://stackoverflow.com/questions/49492255
#also include method on how to inject layers in existing model
def modelResNet50(pars, dims, targetN, inputShape, pretrainedLayers=[]):
    #dims: learningRate
    #############Adjustables#############
    dropoutRate     = 0.5
    momentumRatio   = 0.9
    #####################################
    from tensorflow.keras.applications.resnet50 import ResNet50
    
    parDict = {}
    for par, dim in zip(pars, dims): parDict[dim.name] = par
    #-------------------------------------------------------------------------
    #NOTE: ResNet50 expects inputShape=(None, 224, 224, 3)
    model = ResNet50(weights="imagenet")

    print(model.summary())
    #-------------------------------------------------------------------------
    optimizer = tf.keras.optimizers.SGD(learning_rate=parDict["learningRate"],\
                                        momentum=momentumRatio, nesterov=True)
    model.compile(optimizer=optimizer,\
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])
    return model
class dropoutMC(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True) #to be turned off during .evaluation()
#autoencoder####################################################################################
def buildAutoEncoder(encoder, decoder):
    model = tf.keras.models.Sequential([encoder, decoder])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),\
                  loss="binary_crossentropy", metrics=[roundedAccuracy])
    return model
def buildEncoderConv(inputShape, regularization=False):
    #############Adjustables#############
    convFilterNinit = (8, 8)
    #####################################
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, convFilterNinit, activation="selu", padding="SAME",\
                                     input_shape=inputShape))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    if ("gaus" in regularization) or ("Gaus" in regularization):
        model.add(tf.keras.layers.GaussianNoise(0.1))
    elif ("drop" in regularization) or ("Drop" in regularization):
        model.add(tf.keras.layers.Dropout(0.5))
    if ("l1" in regularization) or ("L1" in regularization):
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="selu", padding="SAME",\
                                         activity_regularizer=tf.keras.regularizers.l1(10e-5)))
    elif ("l2" in regularization) or ("L2" in regularization):
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="selu", padding="SAME",\
                                         activity_regularizer=tf.keras.regularizers.l2(10e-2)))
    else:
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="selu", padding="SAME"));
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    return model
def buildDecoderConv(inputShape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation="selu",\
                                              padding="SAME", input_shape=inputShape))
    model.add(tf.keras.layers.Conv2DTranspose(1,  (3, 3), strides=2, activation="sigmoid",\
                                              padding="SAME"))
    return model
def printTSNE(encodedXInput, knownYInput, nameY, figName, verbosity=1):
    if encodedXInput is None: return
    encodedX, knownY = dropNaNY(encodedXInput, knownYInput)
    tsne = TSNE()
    if len(encodedX.shape) > 2:
        shapeX1D = 1
        for n in encodedX.shape[1:]: shapeX1D *= n
        encodedX = encodedX.reshape(encodedX.shape[0], shapeX1D)
    encodedX2D = tsne.fit_transform(encodedX) #note: different outcome everytime

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 1)
    ax = []
    for i in range (gs.nrows*gs.ncols): ax.append(fig.add_subplot(gs[i]))

    maxY = np.max(knownY)
    minY = np.min(knownY)
    cmap = plt.get_cmap("jet", maxY-minY+1)
    labelFormat = FuncFormatter(lambda x, pos: nameY[int(x)]) #pos required by FuncFormatter
    plot = ax[0].scatter(encodedX2D[:,0], encodedX2D[:,1], c=knownY, s=10, cmap=cmap,\
                         vmax=(maxY+0.5), vmin=(minY-0.5))
    fig.colorbar(plot, ax=ax[0], format=labelFormat, ticks=np.arange(minY, maxY+1))
    ax[0].set_title("Encoder t-SNE Visualization")
    
    filenameFig = FIG_LOC + "-" + figName + ".png"
    gs.tight_layout(fig)
    plt.savefig(filenameFig, dpi=100)
    plt.close()
    if verbosity >= 1: print("   ", filenameFig)
#model fitter###################################################################################
def fitFuncGen(modelName, pars, dims, targetN, inputShape, inputX, inputY,\
               valiR, epochN, bootstrappingN, pretrainedLayers=[], verbosity=1):
    #############Adjustables#############
    minLearningRate       = pow(10, -6)
    scheduleExpDecayConst = 10
    #####################################
    global OPTITER, OPTACCU, OPTASTD
    parStr  = ""
    parDict = {}
    for par, dim in zip(pars, dims):
        parStr += str(par) + "-"
        parDict[dim.name] = par
    parStr = parStr[:-1]
    if verbosity >= 1:
        print("###################################################START MODEL FITTING", OPTITER)
        print("Parameters:", parStr)
    callbacks = []
    tensorboardModelDir = modelName + "/tensorboardModelDir/"
    tensorboardModelDir += str(int(time.time())) + "--" + parStr
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboardModelDir,\
                                                 histogram_freq=0, write_graph=True,\
                                                 write_grads=False, write_images=False)
    callbacks.append(tensorboard)
    if "learningRate" in parDict.keys():
        schedulerFunc = schedulerLambda(parDict["learningRate"], minLearningRate,\
                                        scheduleExpDecayConst)
        scheduler = tf.keras.callbacks.LearningRateScheduler(schedulerFunc)
        callbacks.append(scheduler)
    
    val_accuracies = []
    for bootSeed in range(bootstrappingN):
        if verbosity >= 1: print("\n############################BOOTSTRAPPING:", bootSeed)
        trainX, validX, trainY, validY = train_test_split(inputX, inputY, test_size=valiR,\
                                                          shuffle=True, random_state=bootSeed)
        tf.random.set_seed(bootSeed)    #for dropout Monte Carlo layers
        model = buildModel(modelName, pars, dims, targetN, inputShape,\
                           pretrainedLayers=pretrainedLayers)
        if verbosity >= 3: print(model.summary())
        history = model.fit(trainX, trainY, validation_data=(validX, validY),\
                            epochs=epochN, callbacks=callbacks)
        val_accuracies.append(history.history["val_accuracy"][-1])
        if verbosity >= 1: print("val_accuracy =", val_accuracies[bootSeed])
        if (OPTASTD > 0) and (val_accuracies[bootSeed] < (OPTACCU - 6*OPTASTD)):
            if verbosity >= 1: 
                print("WARNING: 6-sigma smaller than the current optimal validation accuracy:")
                print("   ", OPTACCU, "-", "6*" +str(OPTASTD))
                print("Terminating the bootstrapping...\n")
            break
    val_accuracy = np.mean(val_accuracies)
    val_accu_std = 0.0
    if len(val_accuracies) >= 2: val_accu_std = np.std(val_accuracies, ddof=1)
    if verbosity >= 1:
        print("--------------------------------------------------------------RESULT:")
        print("Parameters:                ", parStr)
        print("Ending Learning Rate      =", K.eval(model.optimizer.learning_rate))
        print("Model Validation Accuracy =", val_accuracy, "+/-", val_accu_std)

    parDict["bootstrappingN"] = bootstrappingN
    parDict["val_accuracy"]   = val_accuracy
    parDict["val_accu_std"]   = val_accu_std
    parDict["isOpt"]          = False
    parDicts, histDFs = {}, {}
    try:
        with open(modelName + "/pars.pickle", "rb") as handle:
            parDicts = pickle.load(handle)
        with open(modelName + "/history.pickle", "rb") as handle:
            histDFs = pickle.load(handle)
    except FileNotFoundError:
        warnings.warn("fitFuncGen(): creating new pars.pickle and history.pickle", Warning) 
    except:
        raise
    parDicts[str(OPTITER)] = parDict
    if val_accuracy > OPTACCU:
        OPTACCU = 1.0*val_accuracy
        OPTASTD = max(1.0*val_accu_std, 0.0)

        histDF = pd.DataFrame(history.history)
        model.save(modelName)
        histDFs[str(OPTITER)] = histDF
        histDFs["opt"] = histDF
        with open(modelName + "/history.pickle", "wb") as handle:
            pickle.dump(histDFs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        parDict["isOpt"] = True
        parDicts["opt"] = parDict 
        with open(modelName + "/pars.pickle", "wb") as handle:
            pickle.dump(parDicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if verbosity >= 1: print("Optimal So Far!")
    if verbosity >= 1:
        print("##########################################################END MODEL FITTING\n\n")
    del model
    os.rename(tensorboardModelDir, \
              tensorboardModelDir.replace("tensorboardModelDir/", "tensorboardModelDir/Fin"))
    OPTITER += 1
    return -val_accuracy
def fitFuncLambda(modelName, dims, targetN, inputShape, trainX, trainY,\
                  valiR, epochN, bootstrappingN, pretrainedLayers=[], verbosity=1):
    return lambda pars: fitFuncGen(modelName, pars, dims, targetN, inputShape,trainX,trainY,\
                                   valiR, epochN, bootstrappingN, pretrainedLayers=[],\
                                   verbosity=verbosity)

################################################################################################
if __name__ == "__main__": main()





