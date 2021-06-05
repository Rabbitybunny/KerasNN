import sys, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
#tensorboard --logdir=... --host localhost --port 6006
#browse: http://localhost:6006/
from tensorflow.keras.activations import *
import tensorflow.keras.backend as K
#https://towardsdatascience.com/bayesian-hyper-parameter-optimization-neural-networks
#-tensorflow-facies-prediction-example-f9c48d21f795
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer

MODELNAME="mnistFashion210416.model";
LOWLAYERTHRES=3;
def buildModel(denseLayerN, denseNeuronN, learningRate, actFunc, initFunc,\
               targetN, inputShape, dropoutMCSeed=0, dropMCfin=False):
    dropoutRate   = 0.2;
    momentumRatio = 0.9;
    model = tf.keras.models.Sequential();
    model.add(tf.keras.layers.Flatten(input_shape=inputShape));
    for i in range(denseLayerN):
        model.add(tf.keras.layers.Dense(denseNeuronN, activation=actFunc,\
                                        kernel_initializer=initFunc));
        if dropMCfin == True:
            model.add(DropoutMC(rate=dropoutRate, seed=dropoutMCSeed));
            model.add(tf.keras.layers.BatchNormalization());
        else:
            if denseLayerN <= LOWLAYERTHRES:
                model.add(DropoutMC(rate=dropoutRate, seed=dropoutMCSeed));
            else:
                model.add(tf.keras.layers.BatchNormalization());
        model.add(tf.keras.layers.Dense(denseNeuronN, activation=actFunc,\
                                        kernel_initializer=initFunc));
    model.add(tf.keras.layers.Dense(targetN, activation="softmax"));
    optimizer = tf.keras.optimizers.SGD(lr=learningRate,momentum=momentumRatio,nesterov=True);
    model.compile(optimizer=optimizer,\
                  loss=tf.keras.losses.sparse_categorical_crossentropy,\
                  metrics=["accuracy"]);
    return model;
class DropoutMC(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True);

OPTITER=0;
OPTACCU=0;
def learningRateFunc(epoch, initLR, minLR, decayC):
    return initLR*pow(0.1, 1.0*epoch/decayC) + minLR;
def schedulerLambda(initLR, minLR, decayC):
    return lambda epoch : learningRateFunc(epoch, initLR, minLR, decayC);
def fitFuncGen(denseLayerN, denseNeuronN, learningRate, actFunc, initFunc,\
               targetN, inputShape, inputX, inputY, validationR, epochN, dropMCfin=False):
    dropoutMonteCarloN    = 10;
    minLearningRate       = pow(10, -6);
    scheduleExpDecayConst = 10;
    global OPTITER;
    global OPTACCU;
    OPTITER += 1;
    parStr = str(denseLayerN) + "-" + str(denseNeuronN) + "-" +\
             str(learningRate) + "-" + str(actFunc) + "-" + str(initFunc); 
    print("##################################################################BEGIN", OPTITER);
    print("Parameters:", parStr);
    tensorboardLogDir = exepath + "/" + MODELNAME + "/tensorboardDir/";
    tensorboardLogDir += str(int(time.time())) + "--" + parStr;
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboardLogDir,\
                                                 histogram_freq=0,\
                                                 write_graph=True,\
                                                 write_grads=False,\
                                                 write_images=False);
    schedulerFunc = schedulerLambda(learningRate, minLearningRate, scheduleExpDecayConst);
    scheduler = tf.keras.callbacks.LearningRateScheduler(schedulerFunc);
    trainX, validX, trainY, validY = train_test_split(inputX, inputY, test_size=validationR);
    if denseLayerN <= LOWLAYERTHRES or dropMCfin == True:
        accuracys = [];
        for seed in range(1, dropoutMonteCarloN+1, 1):
            print("\n############################################MONTE CARLO DROPOUT:", seed);
            model=buildModel(denseLayerN, denseNeuronN, learningRate, actFunc, initFunc,\
                             targetN, inputShape, dropoutMCSeed=seed, dropMCfin=dropMCfin);
            history = model.fit(trainX, trainY, validation_data=(validX, validY),\
                                epochs=epochN, callbacks=[scheduler, tensorboard]);
            accuracys.append(history.history["val_accuracy"][-1]);
            print("Seed", seed, "Accuracy =", accuracys[seed]);
        accuracy = sum(accuracys)/len(accuracys);
    else:
        model=buildModel(denseLayerN, denseNeuronN, learningRate, actFunc, initFunc,\
                         targetN, inputShape);
        history = model.fit(trainX, trainY, validation_data=(validX, validY),\
                            epochs=epochN, callbacks=[scheduler, tensorboard]);
        accuracy = history.history["val_accuracy"][-1];
    print("Parameters:           ", parStr);
    print("Ending Learning Rate =", K.eval(model.optimizer.lr));
    print("Model Accuracy       =", accuracy);
    if accuracy > OPTACCU:
        OPTACCU = 1.0*accuracy;
        histDF = pd.DataFrame(history.history);
        optParDict = {"accuracy":           accuracy,\
                      "denseLayerN":        denseLayerN,\
                      "denseNeutronN":      denseNeuronN,\
                      "learningRate":       learningRate,\
                      "activationFunc":     actFunc,\
                      "initializationFunc": initFunc};
        model.save(MODELNAME);
        histDF.to_pickle(MODELNAME + "/history.pickle");
        with open(MODELNAME + "/pars.pickle", "wb") as handle:
            pickle.dump(optParDict, handle, protocol=pickle.HIGHEST_PROTOCOL);
        print("Optimal So Far!");
    print("#########################################################################END\n\n");
    del model;
    os.rename(tensorboardLogDir, \
              tensorboardLogDir.replace("tensorboardDir/", "tensorboardDir/Fin"));
    OPTITER += 1;
    return -accuracy;
def fitFuncLambda(targetN, inputShape, trainX, trainY, validationR, epochN, dropMCfin=False):
    return lambda x : fitFuncGen(x[0], x[1], x[2], x[3], x[4],\
                                 targetN, inputShape, trainX, trainY,\
                                 validationR, epochN, dropMCfin);





if __name__ == "__main__":
    exepath = os.path.dirname(os.path.abspath(__file__));

    fashionData = tf.keras.datasets.fashion_mnist;
    [[inputX, inputY], [testX, testY]] = fashionData.load_data();
    nameY = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",\
             "Sandal", "Shirt", "Sneaker", "Bag", "Boot"];
    inputXNorm = tf.keras.utils.normalize(inputX, axis=1);
    testXNorm  = tf.keras.utils.normalize(testX, axis=1);  
    targetN = 10;
    inputShape = [28, 28];
    '''
#####modeling#################################################################################
    validationR = 0.1;
    learnEpochN = 10; 
    optCallN    = 30; #note: need >= 11
    par0 = [3, 100, pow(10, -3), "elu", "he_normal"];

    denseLayerN  = Integer(low=1, high=10, name="denseLayerNum");
    denseNeuronN = Integer(low=10, high=500, name="denseNeuronN");
    learningRate = Real(low=pow(10, -6), high=pow(10, -1),\
                        prior="log-uniform", name="learningRate");
    actFunc = Categorical(categories=["elu", "selu"], name="activationFunc");
    initFunc = Categorical(categories=["he_normal", "he_uniform"], name="initializationFunc");
    dims = [denseLayerN, denseNeuronN, learningRate, actFunc, initFunc];
    fitFunc = fitFuncLambda(targetN, inputShape, inputXNorm, inputY, validationR,learnEpochN);
    result = gp_minimize(func=fitFunc, dimensions=dims, x0=par0,\
                         acq_func="EI", n_calls=optCallN);
    #retraining optimal
    model = tf.keras.models.load_model(MODELNAME);
    histDF = pd.read_pickle(MODELNAME + "/history.pickle");
    optParDict = {};
    with open(MODELNAME + "/pars.pickle", "rb") as handle:
        optParDict = pickle.load(handle);
    validationR = 0.1; 
    learnEpochN = 30;
    parOpt = [optParDict["denseLayerN"], optParDict["denseNeutronN"],\
              optParDict["learningRate"], optParDict["activationFunc"],\
              optParDict["initializationFunc"]];
    fitFunc = fitFuncLambda(targetN, inputShape, inputXNorm, inputY,\
                            validationR, learnEpochN, dropMCfin=True);
    optAccuracy = fitFunc(parOpt);
##############################################################################################
    '''
    #loading
    model = tf.keras.models.load_model(MODELNAME);
    histDF = pd.read_pickle(MODELNAME + "/history.pickle");
    optParDict = {};
    with open(MODELNAME + "/pars.pickle", "rb") as handle:
        optParDict = pickle.load(handle);
    #evaluating
    model.evaluate(x=testXNorm, y=testY);
    histDF.plot(figsize=(8, 5));
    plt.title("Learning Performance History");
    plt.grid("True");
    plt.gca().set_ylim(0.0, 1.0);
    filenameFig = exepath + "/fashionFig/-fashionHistory.png";
    plt.savefig(filenameFig);
    plt.close();
    print("Saving the following figures:");
    print(filenameFig, "\n");
    #predicting
    predValY = model.predict(testXNorm);
    predY = np.argmax(predValY, axis=-1);
    for i, valX in enumerate(testX):
        print(i, nameY[predY[i]], nameY[testY[i]]);
        plt.imshow(valX, cmap=plt.cm.binary);
        plt.title("Prediction: "+nameY[predY[i]], fontsize=24);
        filenameFig = exepath + "/fashionFig/predFashion"+str(i)+".png";
        plt.savefig(filenameFig, dpi=50);
        plt.close();
        print(filenameFig);








 
