import sys, os, pathlib, time, math
import numpy as np
import pandas as pd
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
OPTITER, OPTACCU, OPTASTD = 0, 0, 0;
EXE_LOC = str(pathlib.Path().absolute());
FIG_LOC = EXE_LOC + "/fashionFig/";
pathlib.Path(FIG_LOC).mkdir(exist_ok=True);



def main():
    modelName = "mnistFashionSimpleDense.model"
    trainOn   = True
    printFigN = 10
    
    #dataset 
    testRatio       = 0.1
    dropRatio       = 0.0          #ratio of data to simulate unlabeled Y's
    validationRatio = 0.1          #ratio of data for validation

    #trainings
    trainAutoencoderOn    = False
    autoEpochN            = 30
    
    optModelSearchOn      = True
    optimizationCallN     = 30
    learningEpochN        = 6
    dropoutMonteCarloN    = 10
    
    retrainOptModelOn     = True
    learningEpochNOpt     = 30
    dropoutMonteCarloNOpt = 30

    #model setup
    learningRate = Real(   low=1E-6, high=1E-1, prior="log-uniform", name="learningRate")
    convLayerN   = Integer(low=1,    high=5,                         name="convLayerN")
    convFilterN  = Categorical(categories=[32, 64, 128, 256, 512],   name="convFilterN")
    denseLayerN  = Integer(low=1,    high=10,                        name="denseLayerN")
    denseNeuronN = Integer(low=10,   high=500,                       name="denseNeuronN")
    actFunc  = Categorical(categories=["relu", "elu", "selu"],       name="actFunc")
    initFunc = Categorical(categories=["he_normal", "he_uniform"],   name="initFunc")

    dims, par0 = [], []
    if "SimpleDense" in modelName:
        optModelSearchOn = False
        dropoutMonteCarloNOpt = 1
        denseNeuronN2 = Integer(low=10,   high=500,                      name="denseNeuronN2")
        dims = [learningRate, denseNeuronN, denseNeuronN2]
        par0 = [1E-2,         300,          100]
    elif "Dense" in modelName:
        dropoutMonteCarloN    = 1
        dropoutMonteCarloNOpt = 1
        actFunc = Categorical(categories=["relu", "sigmoid"],            name="actFunc")
        dims = [learningRate, denseLayerN, denseNeuronN, actFunc]
        par0 = [1E-3,         3,           128,          "relu"]
    elif "Standard" in modelName:
        dims = [learningRate, denseLayerN, denseNeuronN, actFunc, initFunc]
        par0 = [1E-3,         3,           128,          "elu",   "he_normal"]
    elif "Conv2D" in modelName:
        denseLayerN  = Integer(low=1,    high=5,                         name="denseLayerN")
        dims=[learningRate,convLayerN,convFilterN,denseLayerN,denseNeuronN,actFunc,initFunc]
        par0=[1E-3,        3,         256,        3,          128,        "elu",   "he_normal"]
    elif "RNN" in modelName:
        dims = [learningRate]
        par0 = [1E-3]
#####dataset####################################################################################
    fashionData = tf.keras.datasets.fashion_mnist
    [[inputXFull, inputYFull], [testX, testY]] = fashionData.load_data()#no need for testRatio
    nameY = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",\
             "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
    #simulate unlabeled Y's
    np.random.seed(1)
    inputY = []
    for y in inputYFull:
        if np.random.uniform() < dropRatio: inputY.append(float("NaN"))
        else:                               inputY.append(y)
    #data normalization/standardization
    inputXNorm = np.array([stand2dArray(X) for X in inputXFull])
    testXNorm  = np.array([stand2dArray(X) for X in testX])
    targetN    = len(nameY)
    inputShape = [inputXNorm.shape[1], inputXNorm.shape[2]]
    if ("Conv2D" in modelName) or ("RNN" in modelName):
        inputShape = [inputXNorm.shape[1], inputXNorm.shape[2], 1]     #note: needed for conv2D
        inputXNorm = inputXNorm.reshape(inputXNorm.shape[0], *inputShape)
        testXNorm  = testXNorm .reshape(testXNorm .shape[0], *inputShape)
    #raw figures
    if trainOn == True:
        for idx, valX in enumerate(inputXFull[:printFigN]):
            print(idx, nameY[inputYFull[idx]])
            plt.imshow(valX, cmap=plt.cm.binary)
            plt.title(nameY[inputYFull[idx]], fontsize=24)
            filenameFig = FIG_LOC + "raw"+str(idx)+".png"
            plt.savefig(filenameFig, dpi=100)
            plt.close()
            print("   ", filenameFig)
#####autoencoder################################################################################
    if trainOn == False:
        trainAutoencoderOn = False
        optModelSearchOn   = False
        retrainOptModelOn  = False    
    pretrainedLayers = []
    if trainAutoencoderOn == True:
        print("########################################################AUTOENCODER PRETRAINING")
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
        tensorboardAutoDir = EXE_LOC + "/" + modelName + "/tensorboardAutoDir/"
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
        encoder    = tf.keras.models.load_model(modelName+"/zEncoder.model",    compile=False)
        decoder    = tf.keras.models.load_model(modelName+"/zDecoder.model",    compile=False)
        autoEncoder= tf.keras.models.load_model(modelName+"/zAutoEncoder.model",compile=False)
        if printFigN > 0:
            print("Saving the following figures:")
            encodedX = encoder.predict(validX)
            cmprsX   = autoEncoder.predict(validX)
            validX = validX.reshape(*validX.shape[:-1])
            cmprsX = cmprsX.reshape(*cmprsX.shape[:-1])
            printTSNE(encodedXuntrained, validY, nameY, "TSNEuntrained")
            printTSNE(encodedX,          validY, nameY, "TSNE")
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
        print("####################################################SEARCHING FOR OPTIMAL MODEL")
        fitFunc = fitFuncLambda(modelName, dims, targetN, inputShape, inputXNorm, inputY, \
                                validationRatio, learningEpochN, dropoutMonteCarloN,\
                                pretrainedLayers=pretrainedLayers)
        checkpointPath = EXE_LOC + "/" + modelName + "/checkpoint.pkl"
        checkpointSaver = CheckpointSaver(checkpointPath, compress=9, store_objective=False)
        eval0 = None
        #restore gp_minimize: remember to delete the .pkl file when changing model
        global OPTITER, OPTACCU, OPTASTD
        try:
            restoredOpt = load_gp_minimize(checkpointPath)
            par0, eval0 = restoredOpt.x_iters, restoredOpt.func_vals
            print("Reading the checkpoint file:\n    ", checkpointPath)
            OPTITER = len(eval0) - 1
            optimizationCallN -= OPTITER
            optParDict = {}
            with open(modelName + "/pars.pickle", "rb") as handle:
                optParDict = pickle.load(handle)
            OPTACCU = optParDict["accuracy"]
            OPTASTD = optParDict["accuSTD"]
            print("Current optimal accuracy:")
            print("   ", OPTACCU, "+/-", (OPTASTD if (OPTASTD > 0) else "NA"))
        except FileNotFoundError:
            print("Saving checkpoint file:\n    ", checkpointPath)
        except:
            raise
        #main optimization
        result=gp_minimize(func=fitFunc, dimensions=dims,x0=par0,y0=eval0, acq_func="EI",\
                           n_calls=optimizationCallN, callback=[checkpointSaver])
#####retrain optimal model######################################################################
    if retrainOptModelOn == True:
        print("##########################################################RETRAIN OPTIMAL MODEL")
        parOpt = []
        try:
            optParDict = {}
            with open(modelName + "/pars.pickle", "rb") as handle:
                optParDict = pickle.load(handle)
            for dim in dims: parOpt.append(optParDict[dim.name])
        except FileNotFoundError:
            parOpt = par0.copy()
            print("The parameter file from model optimization is not found:")
            print("   ", modelName + "/pars.pickle")
            print("Using par0 as the optimized parameters")
        except:
            raise
        fitFuncOpt = fitFuncLambda(modelName, dims, targetN, inputShape, inputXNorm, inputY,\
                                   validationRatio, learningEpochNOpt, dropoutMonteCarloNOpt,\
                                   pretrainedLayers=pretrainedLayers)
        optAccuracy = fitFuncOpt(parOpt)
####prediction##################################################################################
    #loading trained data
    try:
        model = tf.keras.models.load_model(modelName)
        histDF = pd.read_pickle(modelName + "/history.pickle")
        optParDict = {}
        with open(modelName + "/pars.pickle", "rb") as handle:
            optParDict = pickle.load(handle)
    except OSError or FileNotFoundError:
        print("No trained model is found:\n    ", modelName)
        sys.exit(0)
    except:
        raise
    #evaluation
    model.evaluate(x=testXNorm, y=testY)
    histDF.plot(figsize=(8, 5))
    plt.title("Learning Performance History")
    plt.grid("True")
    plt.gca().set_ylim(0.0, 1.0)
    filenameFig = FIG_LOC + "-optModel_learningHistory.png"
    plt.savefig(filenameFig)
    plt.close()
    print("Saving the following figures:\n    ", filenameFig)
    #prediction figures
    predValY = model.predict(testXNorm)
    predY = np.argmax(predValY, axis=-1)
    for idx, valX in enumerate(testX[:printFigN]):
        print(idx, nameY[predY[idx]], nameY[testY[idx]])
        plt.imshow(valX, cmap=plt.cm.binary)
        plt.title("Prediction: "+nameY[predY[idx]], fontsize=24)
        filenameFig = FIG_LOC + "predicted"+str(idx)+".png"
        plt.savefig(filenameFig, dpi=100)
        plt.close()
        print("   ", filenameFig)






################################################################################################
################################################################################################
################################################################################################
#model##########################################################################################
def buildModel(modelName, pars, dims, targetN, inputShape, dropoutMCSeed=0,pretrainedLayers=[]):
    if "SimpleDense" in modelName:
        return modelDenseSimple(pars, dims, targetN, inputShape, pretrainedLayers)
    elif "Dense" in modelName:
        return modelDense(pars, dims, targetN, inputShape, pretrainedLayers)
    elif "Standard" in modelName:
        return modelStandard(pars, dims, targetN, inputShape, dropoutMCSeed, pretrainedLayers)
    elif "Conv2D" in modelName:
        return modelConv2D(pars, dims, targetN, inputShape, dropoutMCSeed, pretrainedLayers)
    elif "RNN" in modelName:
        return modelRNN(pars, dims, targetN, inputShape, dropoutMCSeed, pretrainedLayers)
    else:
        print("No model found:\n    ", modelName)
        sys.exit(0)
def modelDenseSimple(pars, dims, targetN, inputShape, pretrainedLayers=[]):
    #dims: learningRate, denseNeuronN, denseNeuronN2
    parDict = {}
    for par, dim in zip(pars, dims): parDict[dim.name] = par
    
    model = tf.keras.models.Sequential()
    if not pretrainedLayers: model.add(tf.keras.layers.Flatten(input_shape=inputShape))
    else:
        for layer in pretrainedLayers: model.add(cloneLayer(layer))
    model.add(tf.keras.layers.Dense(parDict["denseNeuronN"],  activation="relu"))
    model.add(tf.keras.layers.Dense(parDict["denseNeuronN2"], activation="relu"))
    model.add(tf.keras.layers.Dense(targetN,       activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=parDict["learningRate"]),\
                  loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=["accuracy"])
    return model
def modelDense(pars, dims, targetN, inputShape, pretrainedLayers=[]):
    #dims: learningRate, denseLayerN, denseNeuronN, actFunc
    parDict = {}
    for par, dim in zip(pars, dims): parDict[dim.name] = par

    model = tf.keras.models.Sequential()
    if not pretrainedLayers: model.add(tf.keras.layers.Flatten(input_shape=inputShape))
    else:
        for layer in pretrainedLayers: model.add(cloneLayer(layer))
    for i in range(parDict["denseLayerN"]):
        model.add(tf.keras.layers.Dense(parDict["denseNeuronN"], activation=parDict["actFunc"]))
    model.add(tf.keras.layers.Dense(targetN, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=parDict["learningRate"]),\
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])
    return model
def modelStandard(pars, dims, targetN, inputShape, dropoutMCSeed=0, pretrainedLayers=[]):
    #dims: learningRate, denseLayerN, denseNeuronN, actFunc, initFunc
    #############Adjustables#############
    dropoutRate   = 0.2
    momentumRatio = 0.9
    #####################################
    parDict = {}
    for par, dim in zip(pars, dims): parDict[dim.name] = par

    model = tf.keras.models.Sequential()
    if not pretrainedLayers: model.add(tf.keras.layers.Flatten(input_shape=inputShape))
    else:
        for layer in pretrainedLayers: model.add(cloneLayer(layer))
    for i in range(parDict["denseLayerN"]):
        model.add(tf.keras.layers.Dense(parDict["denseNeuronN"],activation=parDict["actFunc"],\
                                        kernel_initializer=parDict["initFunc"]))
        model.add(dropoutMC(rate=dropoutRate, seed=dropoutMCSeed))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(parDict["denseNeuronN"],activation=parDict["actFunc"],\
                                        kernel_initializer=parDict["initFunc"]))
    model.add(tf.keras.layers.Dense(targetN, activation="softmax"))
    optimizer = tf.keras.optimizers.SGD(lr=parDict["learningRate"], momentum=momentumRatio,\
                                        nesterov=True)
    model.compile(optimizer=optimizer,\
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])
    return model
def modelConv2D(pars, dims, targetN, inputShape, dropoutMCSeed=0, pretrainedLayers=[]):
    #dims: learningRate, convLayerN, convFilterN, denseLayerN, denseNeuronN, actFunc, initFunc
    #############Adjustables#############
    convLayerNinit  = 64
    convFilterNinit = (8, 8)
    dropoutRate     = 0.5
    momentumRatio   = 0.9
    #####################################
    parDict = {}
    for par, dim in zip(pars, dims): parDict[dim.name] = par

    model = tf.keras.models.Sequential()
    if not pretrainedLayers:
        model.add(tf.keras.layers.Conv2D(convLayerNinit, convFilterNinit, 
                                         activation=parDict["actFunc"],\
                                         padding="SAME", input_shape=inputShape))
    else:
        for layer in pretrainedLayers: model.add(cloneLayer(layer))
    for i in range(parDict["convLayerN"]):
        if pow(2, i+1) < min(inputShape[0], inputShape[1]): 
            model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        filterN = max(8, parDict["convFilterN"]/pow(2, parDict["convLayerN"]-1-i))
        model.add(tf.keras.layers.Conv2D(filterN, (3, 3), activation=parDict["actFunc"],\
                                         padding="SAME"))
    model.add(tf.keras.layers.Flatten())
    for i in range(parDict["denseLayerN"]):
        model.add(tf.keras.layers.Dropout(rate=dropoutRate, seed=dropoutMCSeed))
        neutronN = max(8, parDict["denseNeuronN"]/pow(2, i))
        model.add(tf.keras.layers.Dense(neutronN, activation=parDict["actFunc"],\
                                        kernel_initializer=parDict["initFunc"]))
    model.add(tf.keras.layers.Dense(targetN, activation="softmax"))
    optimizer = tf.keras.optimizers.SGD(lr=parDict["learningRate"], momentum=momentumRatio,\
                                        nesterov=True)
    model.compile(optimizer=optimizer,\
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])
    return model
#https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-
#convolutional-neural-networks/
def modelRNN(pars, dims, targetN, inputShape, dropoutMCSeed=0, pretrainedLayers=[]):
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
    if not pretrainedLayers:
        Z = tf.keras.layers.Conv2D(convLayerNinit, convFilterNinit, strides=2,\
                                   activation="relu", padding="SAME")(Z)
    else:
        for layer in pretrainedLayers: Z = layer(Z)
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
    Z = tf.keras.layers.Dropout(rate=dropoutRate, seed=dropoutMCSeed)(Z)
    Z = tf.keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal")(Z)
    Z = tf.keras.layers.Dense(targetN, activation="softmax")(Z)
    model = tf.keras.models.Model(inputs=inputZ, outputs=Z)
    optimizer = tf.keras.optimizers.SGD(lr=parDict["learningRate"], momentum=momentumRatio,\
                                        nesterov=True)
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
class dropoutMC(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True) #to be turned off during .evaluation()
#model fitter###################################################################################
def fitFuncGen(modelName, pars, dims, targetN, inputShape, inputX, inputY,\
               valiR, epochN, dropMCN, pretrainedLayers=[]):
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
    print("#####################################################################BEGIN", OPTITER)
    print("Parameters:", parStr)
    callbacks = []
    tensorboardModelDir = EXE_LOC + "/" + modelName + "/tensorboardModelDir/"
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

    trainX, validX, trainY, validY = train_test_split(inputX, inputY, test_size=valiR)
    accuracies = []
    for dropSeed in range(dropMCN):
        print("\n###################################MONTE CARLO DROPOUT:", dropSeed)
        model=buildModel(modelName, pars, dims, targetN, inputShape,\
                         dropoutMCSeed=dropSeed, pretrainedLayers=pretrainedLayers)
        history = model.fit(trainX, trainY, validation_data=(validX, validY),\
                            epochs=epochN, callbacks=callbacks)
        accuracies.append(history.history["val_accuracy"][-1])
        print("Drop-Seed", dropSeed, "Accuracy =", accuracies[dropSeed])
        if accuracies[dropSeed] < (OPTACCU - 6*OPTASTD):
            print("The accuracy is 6-sigma smaller than the current optimal accuracy:")
            print("   ", OPTACCU, "-", "6*" +str(OPTASTD))
            print("Terminating the dropout Monte Carlo...\n")
            break
    accuracy = sum(accuracies)/len(accuracies)
    accuSTD = -1
    if len(accuracies) > 1: accuSTD = np.std(np.array(accuracies), ddof=1)
    print("--------------------------------------------------------------RESULT:")
    print("Parameters:           ", parStr)
    print("Ending Learning Rate =", K.eval(model.optimizer.lr))
    print("Model Accuracy       =", accuracy, "+/-", (accuSTD if (accuSTD > 0) else "NA"))
    if accuracy > OPTACCU:
        OPTACCU = 1.0*accuracy
        OPTASTD = max(1.0*accuSTD, 0.0)
        histDF = pd.DataFrame(history.history)
        optParDict = {"accuracy": accuracy,\
                      "accuSTD":  accuSTD}
        for par, dim in zip(pars, dims): optParDict[dim.name] = par
        model.save(modelName)
        histDF.to_pickle(modelName + "/history.pickle")
        with open(modelName + "/pars.pickle", "wb") as handle:
            pickle.dump(optParDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Optimal So Far!")
    print("############################################################################END\n\n")
    del model
    os.rename(tensorboardModelDir, \
              tensorboardModelDir.replace("tensorboardModelDir/", "tensorboardModelDir/Fin"))
    OPTITER += 1
    return -accuracy
def fitFuncLambda(modelName, dims, targetN, inputShape, trainX, trainY,\
                  valiR, epochN, dropMCN, pretrainedLayers=[]):
    return lambda pars: fitFuncGen(modelName, pars, dims, targetN, inputShape,trainX,trainY,\
                                   valiR, epochN, dropMCN, pretrainedLayers=[])
def learningRateFunc(epoch, initLR, minLR, decayC):
    return initLR*pow(0.1, 1.0*epoch/decayC) + minLR
def schedulerLambda(initLR, minLR, decayC):
    return lambda epoch: learningRateFunc(epoch, initLR, minLR, decayC)
#autoencoder####################################################################################
def buildAutoEncoder(encoder, decoder):
    model = tf.keras.models.Sequential([encoder, decoder])
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=1.0), loss="binary_crossentropy",\
                  metrics=[roundedAccuracy])
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
def printTSNE(encodedXInput, knownYInput, nameY, figName):
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
    print("   ", filenameFig)
#helper funcs###################################################################################
def stand2dArray(array):    #following tf.image.per_image_standardization
    array = np.array(array)
    flatArr = array.flatten()
    mean = np.mean(flatArr)
    std = np.std(flatArr)
    return (array - mean)/max(std, 1/math.sqrt(flatArr.size))
def cloneLayer(layer):
    config = layer.get_config()
    weights = layer.get_weights()
    clonedLayer = type(layer).from_config(config)
    clonedLayer.build(layer.input_shape)
    clonedLayer.set_weights(weights)
    return clonedLayer
def dropNaNY(inputX, inputY):
    inputXOutput = []
    inputYOutput = []
    for i, y in enumerate(inputY):
        if math.isnan(y) == False:
            inputXOutput.append(inputX[i])
            inputYOutput.append(y)
    return np.array(inputXOutput), np.array(inputYOutput)
def roundedAccuracy(yTrue, yPred):
    return tf.keras.metrics.binary_accuracy(tf.round(yTrue), tf.round(yPred))



if __name__ == "__main__": main()





