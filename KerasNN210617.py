import sys, os, time, math
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


MODELNAME = "mnistFashion210617.model";
EXEPATH   = os.path.dirname(os.path.abspath(__file__)); 
OPTITER = 0;
OPTACCU = 0;
OPTASTD = 0;
def main():
    fashionData = tf.keras.datasets.fashion_mnist;
    [[inputXFull, inputYFull], [testX, testY]] = fashionData.load_data();
    nameY = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",\
             "Sandal", "Shirt", "Sneaker", "Bag", "Boot"];
    #dropping some labels to simulate unlabeld conditions
    dropRate = 0.0;
    np.random.seed(1);
    inputY = [];
    for y in inputYFull:
        if np.random.uniform() < dropRate:
            inputY.append(float("NaN"));
        else:
            inputY.append(y);
    #####################################################
    inputXNorm = np.array([stand2dArray(X) for X in inputXFull]);
    testXNorm  = np.array([stand2dArray(X) for X in testX]);
    targetN = len(nameY);
    inputShape = [inputXNorm.shape[1], inputXNorm.shape[2], 1];   #note: needed for conv2D
    inputXNorm = inputXNorm.reshape(inputXNorm.shape[0], *inputShape);
    testXNorm  = testXNorm .reshape(testXNorm .shape[0], *inputShape);
#####autoencoder##############################################################################
    pretrainedLayers = []
    '''
    printFigN  = 20;
    encodedXuntrained = None;
    trainX, validX, trainY, validY = train_test_split(inputXNorm, inputY, test_size=0.1,\
                                                      shuffle=False);
    #############Adjustables#############
    autoEpochN = 100; 
    #####################################
    encoder = buildEncoderConv(inputShape, regularization="dropout");
    encoderOutputShape = list(encoder.layers[-1].output_shape);
    encoderOutputShape = [dim for dim in encoderOutputShape if dim is not None];
    decoder = buildDecoderConv(inputShape=encoderOutputShape);

    autoEncoder = buildAutoEncoder(encoder, decoder);
    encodedXuntrained = encoder.predict(validX);
    tensorboardAutoDir = EXEPATH + "/" + MODELNAME + "/tensorboardAutoDir/";
    tensorboardAutoDir += str(int(time.time())) + "Conv"; 
    tensorboardAuto = tf.keras.callbacks.TensorBoard(tensorboardAutoDir);
    history = autoEncoder.fit(trainX, trainX, validation_data=(validX, validX),\
                              epochs=autoEpochN, callbacks=[tensorboardAuto]);
    encoder    .save(MODELNAME+"/zEncoder.model");
    decoder    .save(MODELNAME+"/zDecoder.model");
    autoEncoder.save(MODELNAME+"/zAutoEncoder.model");
    os.rename(tensorboardAutoDir, \
              tensorboardAutoDir.replace("tensorboardAutoDir/", "tensorboardAutoDir/Fin"));
    #compressed figures   
    encoder     = tf.keras.models.load_model(MODELNAME+"/zEncoder.model", compile=False);
    decoder     = tf.keras.models.load_model(MODELNAME+"/zDecoder.model", compile=False);
    autoEncoder = tf.keras.models.load_model(MODELNAME+"/zAutoEncoder.model", compile=False);
    if printFigN > 0:
        print("Saving the following figures:");
        encodedX = encoder.predict(validX);
        cmprsX   = autoEncoder.predict(validX);
        validX = validX.reshape(*validX.shape[:-1]);
        cmprsX = cmprsX.reshape(*cmprsX.shape[:-1]);
        printTSNE(encodedXuntrained, validY, nameY, "fashionConv"+"TSNEuntrained");
        printTSNE(encodedX,          validY, nameY, "fashionConv"+"TSNE");
        for idx, valX in enumerate(validX[:printFigN]):
            fig = plt.figure(figsize=(12, 6));
            gs = gridspec.GridSpec(1, 2);
            ax = [];
            for i in range (gs.nrows*gs.ncols):
                ax.append(fig.add_subplot(gs[i]));
            ax[0].imshow(valX, cmap=plt.cm.binary);
            ax[0].set_title("Original Normalized", fontsize=24);
            ax[1].imshow(cmprsX[idx], cmap=plt.cm.binary);
            ax[1].set_title("Autoencoder Compressed", fontsize=24); 
            filenameFig = EXEPATH + "/fashionFig/cmprsFashionConv"+str(idx)+".png";
            plt.savefig(filenameFig, dpi=50);
            plt.close();
            print("   ", filenameFig);
    #encoder for autoencoder pretraining
    encoder = tf.keras.models.load_model(MODELNAME+"/zEncoder.model", compile=False);
    for i, layer in enumerate(encoder.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            #layer.trainable = False;
            pretrainedLayers.append(layer);
    '''
    #dropping out untagged events
    inputXNorm, inputY = dropNaNY(inputXNorm, inputY);
#####modeling#################################################################################
    #############Adjustables#############
    validationRatio     = 0.1;
    learnEpochN         = 5; 
    dropoutMonteCarloN  = 5;
    optimizationCallN   = 11; #note: need >= 11
    par0 = [pow(10, -3)];
    #####################################  
    learningRate = Real(low=pow(10, -6), high=pow(10, -1),\
                        prior="log-uniform", name="learningRate");
    dims = [learningRate];
    fitFunc = fitFuncLambda(targetN, inputShape, inputXNorm, inputY, \
                            validationRatio, learnEpochN, dropoutMonteCarloN,\
                            pretrainedLayers=pretrainedLayers);
    checkpointPath = EXEPATH + "/" + MODELNAME + "/checkpoint.pkl";
    checkpointSaver = CheckpointSaver(checkpointPath, compress=9, store_objective=False);
    eval0 = None;
    optParDict = {};
    #restore gp_minimize##################Remember to delete the .pkl file when changing model
    global OPTITER;
    global OPTACCU;
    global OPTASTD;
    try:
        restoredOpt = load_gp_minimize(checkpointPath);
        par0, eval0 = restoredOpt.x_iters, restoredOpt.func_vals;
        print("Reading the checkpoint file:\n    ", checkpointPath);
        OPTITER = len(eval0); 
        optimizationCallN -= OPTITER;
        with open(MODELNAME + "/pars.pickle", "rb") as handle:
            optParDict = pickle.load(handle);
        OPTACCU = optParDict["accuracy"];
        OPTASTD = optParDict["accuSTD"];
        print("Current optimal accuracy:");
        print("   ", OPTACCU, "+/-", (OPTASTD if (OPTASTD > 0) else "NA"));
    except FileNotFoundError:
        print("Saving checkpoint file:\n    ", checkpointPath);
    except:
        raise;
    #main optimization
    result = gp_minimize(func=fitFunc, dimensions=dims, x0=par0, y0=eval0, acq_func="EI",\
                         n_calls=optimizationCallN, callback=[checkpointSaver]);
    #retraining optimal
    print("###########################################################RETRAIN OPTIMAL MODEL");
    model = tf.keras.models.load_model(MODELNAME);
    histDF = pd.read_pickle(MODELNAME + "/history.pickle");
    optParDict = {};
    with open(MODELNAME + "/pars.pickle", "rb") as handle:
        optParDict = pickle.load(handle);
    validationRatio     = 0.1; 
    learnEpochN         = 20;
    dropoutMonteCarloN  = 30;
    parOpt = [optParDict["learningRate"]];
    fitFunc = fitFuncLambda(targetN, inputShape, inputXNorm, inputY,\
                            validationRatio, learnEpochN, dropoutMonteCarloN,\
                            pretrainedLayers=pretrainedLayers);
    optAccuracy = fitFunc(parOpt);
##############################################################################################
    #loading
    model = tf.keras.models.load_model(MODELNAME);
    histDF = pd.read_pickle(MODELNAME + "/history.pickle");
    optParDict = {};
    with open(MODELNAME + "/pars.pickle", "rb") as handle:
        optParDict = pickle.load(handle);
    #evaluating
    model.evaluate(x=testXNorm, y=testY);
    
    print("Saving the following figures:");   
    histDF.plot(figsize=(8, 5));
    plt.title("Learning Performance History");
    plt.grid("True");
    plt.gca().set_ylim(0.0, 1.0);
    filenameFig = EXEPATH + "/fashionFig/-fashionHistory.png";
    plt.savefig(filenameFig);
    plt.close();
    print(filenameFig);
    #predicting
    predValY = model.predict(testXNorm);
    predY = np.argmax(predValY, axis=-1);
    for idx, valX in enumerate(testX):
        if idx > 30:
            break;
        print(idx, nameY[predY[idx]], nameY[testY[idx]]);
        plt.imshow(valX, cmap=plt.cm.binary);
        plt.title("Prediction: "+nameY[predY[idx]], fontsize=24);
        filenameFig = EXEPATH + "/fashionFig/predFashion"+str(idx)+".png";
        plt.savefig(filenameFig, dpi=50);
        plt.close();
        print("   ", filenameFig);





##############################################################################################
##############################################################################################
##############################################################################################
def stand2dArray(array):    #following tf.image.per_image_standardization
    array = np.array(array);
    flatArr = array.flatten();
    mean = np.mean(flatArr);
    std = np.std(flatArr);
    return (array - mean)/max(std, 1/math.sqrt(flatArr.size));
def cloneLayer(layer):
    config = layer.get_config();
    weights = layer.get_weights();
    clonedLayer = type(layer).from_config(config);
    clonedLayer.build(layer.input_shape);
    clonedLayer.set_weights(weights);
    return clonedLayer;
def dropNaNY(inputX, inputY):
    inputXOutput = [];
    inputYOutput = [];
    for i, y in enumerate(inputY):
        if math.isnan(y) == False:
            inputXOutput.append(inputX[i]);
            inputYOutput.append(y);
    return np.array(inputXOutput), np.array(inputYOutput);
def roundedAccuracy(yTrue, yPred):
    return tf.keras.metrics.binary_accuracy(tf.round(yTrue), tf.round(yPred))
def buildAutoEncoder(encoder, decoder):
    model = tf.keras.models.Sequential([encoder, decoder]);
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=1.0), loss="binary_crossentropy",\
                  metrics=[roundedAccuracy]);
    return model;
def buildEncoderConv(inputShape, regularization=False):
    #############Adjustables#############
    convFilterNinit = (8, 8);
    #####################################
    model = tf.keras.models.Sequential();
    model.add(tf.keras.layers.Conv2D(32, convFilterNinit, activation="selu", padding="SAME",\
                                     input_shape=inputShape));
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)));
    if ("gaus" in regularization) or ("Gaus" in regularization):
        model.add(tf.keras.layers.GaussianNoise(0.1));
    elif ("drop" in regularization) or ("Drop" in regularization):    
        model.add(tf.keras.layers.Dropout(0.5));
    if ("l1" in regularization) or ("L1" in regularization):
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="selu", padding="SAME",\
                      activity_regularizer=tf.keras.regularizers.l1(10e-5)));
    elif ("l2" in regularization) or ("L2" in regularization):
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="selu", padding="SAME",\
                      activity_regularizer=tf.keras.regularizers.l2(10e-2)));
    else:    
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="selu", padding="SAME"));
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)));
    return model;
def buildDecoderConv(inputShape):
    model = tf.keras.models.Sequential();
    model.add(tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, activation="selu",\
                                              padding="SAME", input_shape=inputShape));
    model.add(tf.keras.layers.Conv2DTranspose(1,  (3, 3), strides=2, activation="sigmoid",\
                                              padding="SAME"));
    return model;
def printTSNE(encodedXInput, knownYInput, nameY, figName):
    if encodedXInput is None:
        return;
    encodedX, knownY = dropNaNY(encodedXInput, knownYInput);
    tsne = TSNE();
    if len(encodedX.shape) > 2:
        shapeX1D = 1;
        for n in encodedX.shape[1:]:
            shapeX1D *= n;
        encodedX = encodedX.reshape(encodedX.shape[0], shapeX1D);
    encodedX2D = tsne.fit_transform(encodedX); #note: different outcome everytime

    fig = plt.figure(figsize=(8, 6));
    gs = gridspec.GridSpec(1, 1);
    ax = [];
    for i in range (gs.nrows*gs.ncols):
        ax.append(fig.add_subplot(gs[i]));

    maxY = np.max(knownY);
    minY = np.min(knownY);
    cmap = plt.get_cmap("jet", maxY-minY+1);
    labelFormat = FuncFormatter(lambda x, pos: nameY[int(x)]); #pos required by FuncFormatter
    plot = ax[0].scatter(encodedX2D[:,0], encodedX2D[:,1], c=knownY, s=10, cmap=cmap,\
                         vmax=(maxY+0.5), vmin=(minY-0.5));
    fig.colorbar(plot, ax=ax[0], format=labelFormat, ticks=np.arange(minY, maxY+1));
    ax[0].set_title("Encoder t-SNE Visualization");
    
    filenameFig = EXEPATH + "/fashionFig/-" + figName + ".png";
    gs.tight_layout(fig);
    plt.savefig(filenameFig, dpi=200);
    plt.close();
    print("   ", filenameFig);

#model########################################################################################
class dropoutMC(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True); #to be turned off during .evaluation()
#https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-
#convolutional-neural-networks/
def residualBlock(inputZ, filterN, strideN=1):
    mainZ = inputZ + 0;
    mainZ = tf.keras.layers.Conv2D(filterN, [3, 3], strides=strideN, padding="SAME",\
                                   use_bias=False)(mainZ);
    mainZ = tf.keras.layers.BatchNormalization()  (mainZ);
    mainZ = tf.keras.layers.Activation("relu")   (mainZ);
    mainZ = tf.keras.layers.Conv2D(filterN, [3, 3], strides=1, padding="SAME",\
                                   use_bias=False)(mainZ);
    mainZ = tf.keras.layers.BatchNormalization()  (mainZ);

    skipZ = inputZ + 0;
    if strideN > 1:             #patch size of [1, 1] is the key
        skipZ = tf.keras.layers.Conv2D(filterN, [1, 1], strides=strideN, padding="SAME",\
                                       use_bias=False)(skipZ);
        skipZ = tf.keras.layers.BatchNormalization()  (skipZ);
    
    mergedZ = tf.keras.layers.add([mainZ, skipZ]);
    mergedZ = tf.keras.layers.Activation("relu")(mergedZ);
    return mergedZ;
def buildModel(learningRate, targetN, inputShape, dropoutMCSeed=0, pretrainedLayers=[]):
    #############Adjustables#############
    convLayerNinit  = 64;
    convFilterNinit = (8, 8);
    dropoutRate     = 0.5;
    momentumRatio   = 0.9;
    #####################################
    inputZ = tf.keras.layers.Input(inputShape);
    Z = inputZ + 0;
    if not pretrainedLayers:
        Z = tf.keras.layers.Conv2D(convLayerNinit, convFilterNinit, strides=2,\
                                   activation="relu", padding="SAME")(Z);
    else:
        for layer in pretrainedLayers:
            Z = layer(Z);
    Z = tf.keras.layers.BatchNormalization()(Z);
    Z = tf.keras.layers.Activation("relu")(Z);
    Z = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="SAME")(Z);
    
    filterNpre = 64
    for filterN in [64]*3 + [128]*4 + [256]*6 + [512]*3:
        strideN = (1 if filterN == filterNpre else 2);
        Z = residualBlock(Z, filterN, strideN=strideN);
        filterNpre = filterN*1;
    
    Z = tf.keras.layers.GlobalAvgPool2D()(Z);
    Z = tf.keras.layers.Flatten()(Z);
    Z = tf.keras.layers.Dropout(rate=dropoutRate, seed=dropoutMCSeed)(Z);
    Z = tf.keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal")(Z);
    Z = tf.keras.layers.Dense(targetN, activation="softmax")(Z);
    model = tf.keras.models.Model(inputs=inputZ, outputs=Z);
    optimizer = tf.keras.optimizers.SGD(lr=learningRate,momentum=momentumRatio,nesterov=True);
    model.compile(optimizer=optimizer, metrics=["accuracy"],\
                  loss=tf.keras.losses.sparse_categorical_crossentropy);  #categorical targets
    return model;

#fit function#################################################################################
def learningRateFunc(epoch, initLR, minLR, decayC):
    return initLR*pow(0.1, 1.0*epoch/decayC) + minLR;
def schedulerLambda(initLR, minLR, decayC):
    return lambda epoch : learningRateFunc(epoch, initLR, minLR, decayC);
def fitFuncGen(learningRate, targetN, inputShape, inputX, inputY,\
               valiR, epochN, dropMCN, pretrainedLayers=[]):
    #############Adjustables#############
    minLearningRate       = pow(10, -6);
    scheduleExpDecayConst = 10;
    #####################################
    global OPTITER;
    global OPTACCU;
    global OPTASTD;
    parStr = "learningRate_" + str(learningRate);
    print("##################################################################BEGIN", OPTITER);
    print("Parameters:", parStr);
    tensorboardModelDir = EXEPATH + "/" + MODELNAME + "/tensorboardModelDir/";
    tensorboardModelDir += str(int(time.time())) + "--" + parStr;
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboardModelDir,\
                                                 histogram_freq=0,\
                                                 write_graph=True,\
                                                 write_grads=False,\
                                                 write_images=False);
    schedulerFunc = schedulerLambda(learningRate, minLearningRate, scheduleExpDecayConst);
    scheduler = tf.keras.callbacks.LearningRateScheduler(schedulerFunc);
    trainX, validX, trainY, validY = train_test_split(inputX, inputY, test_size=valiR);
    accuracies = [];
    for dropSeed in range(dropMCN):
        print("\n#######################################MONTE CARLO DROPOUT:", dropSeed);
        model=buildModel(learningRate, targetN, inputShape,\
                         dropoutMCSeed=dropSeed, pretrainedLayers=pretrainedLayers);
        if dropSeed == 0:
            print("Model summary:\n", model.summary());
        history = model.fit(trainX, trainY, validation_data=(validX, validY),\
                            epochs=epochN, callbacks=[tensorboard, scheduler]);
        accuracies.append(history.history["val_accuracy"][-1]); #note: a dictionary
        print("    Accuracy =", accuracies[dropSeed]);
        if accuracies[dropSeed] < (OPTACCU - 6*OPTASTD):
            print("The accuracy is 6-sigma smaller than the current optimal accuracy:");
            print("   ", OPTACCU, "-", "6*" +str(OPTASTD));
            print("Terminating the dropout Monte Carlo...\n");
            break;
    accuracy = sum(accuracies)/len(accuracies);
    accuSTD = -1;
    if len(accuracies) > 1:
        accuSTD = np.std(np.array(accuracies), ddof=1);
    print("------------------------------------------------------------RESULT:");
    print("Parameters:           ", parStr);
    print("Ending Learning Rate =", K.eval(model.optimizer.lr));
    print("Model Accuracy       =", accuracy, "+/-", (accuSTD if (accuSTD > 0) else "NA"));
    if accuracy > OPTACCU:
        OPTACCU = 1.0*accuracy;
        OPTASTD = max(1.0*accuSTD, 0.0);
        histDF = pd.DataFrame(history.history);
        optParDict = {"accuracy":           accuracy,\
                      "accuSTD":            accuSTD,\
                      "learningRate":       learningRate};
        model.save(MODELNAME);
        histDF.to_pickle(MODELNAME + "/history.pickle");
        with open(MODELNAME + "/pars.pickle", "wb") as handle:
            pickle.dump(optParDict, handle, protocol=pickle.HIGHEST_PROTOCOL);
        print("Optimal So Far!");
    print("#########################################################################END\n\n");
    del model;
    os.rename(tensorboardModelDir, \
              tensorboardModelDir.replace("tensorboardModelDir/", "tensorboardModelDir/Fin"));
    OPTITER += 1;
    return -accuracy;
def fitFuncLambda(targetN, inputShape, trainX, trainY, valiR, epochN, dropMCN,\
                  pretrainedLayers=[]):
    return lambda x : fitFuncGen(*x, targetN, inputShape, trainX, trainY,\
                                 valiR, epochN, dropMCN,\
                                 pretrainedLayers=[]);



if __name__ == "__main__":
    main();    







 
