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
#https://github.com/keras-team/keras/issues/3945
#https://stackoverflow.com/questions/64118331
K.image_data_format();
#https://towardsdatascience.com/bayesian-hyper-parameter-optimization-neural-networks
#-tensorflow-facies-prediction-example-f9c48d21f795
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from skopt import load as load_gp_minimize
from skopt.space import Integer, Real, Categorical


MODELNAME="mnistFashion210422.model";

def stand2dArray(array):    #following tf.image.per_image_standardization
    array = np.array(array);
    flatArr = array.flatten();
    mean = np.mean(flatArr);
    std = np.std(flatArr);
    return (array - mean)/max(std, 1/math.sqrt(flatArr.size));
def buildModel(convLayerN, convFilterN, denseLayerN, denseNeuronN,\
               learningRate, actFunc, initFunc,\
               targetN, inputShape, dropoutMCSeed=0):
    #############Adjustables#############
    convLayerNinit  = 64;
    convFilterNinit = 8;
    dropoutRate     = 0.5;
    momentumRatio   = 0.9;
    #####################################
    model = tf.keras.models.Sequential();
    model.add(tf.keras.layers.Conv2D(convLayerNinit, convFilterNinit, activation=actFunc,\
                                     padding="SAME", input_shape=inputShape)); 
    for i in range(convLayerN):
        if pow(2, i+1) < min(inputShape[0], inputShape[1]): 
            model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)));
        filterN = max(8, convFilterN/pow(2, convLayerN-1-i));
        model.add(tf.keras.layers.Conv2D(filterN, 3, activation=actFunc, padding="SAME"));
    model.add(tf.keras.layers.Flatten());
    for i in range(denseLayerN):
        model.add(tf.keras.layers.Dropout(rate=dropoutRate, seed=dropoutMCSeed));
        neutronN = max(8, denseNeuronN/pow(2, i));
        model.add(tf.keras.layers.Dense(neutronN, activation=actFunc,\
                                        kernel_initializer=initFunc));
    model.add(tf.keras.layers.Dense(targetN, activation="softmax"));
    optimizer = tf.keras.optimizers.SGD(lr=learningRate,momentum=momentumRatio,nesterov=True);
    model.compile(optimizer=optimizer,\
                  loss=tf.keras.losses.sparse_categorical_crossentropy,\
                  metrics=["accuracy"]);
    return model;

OPTITER = 0;
OPTACCU = 0;
OPTASTD = 0;
def learningRateFunc(epoch, initLR, minLR, decayC):
    return initLR*pow(0.1, 1.0*epoch/decayC) + minLR;
def schedulerLambda(initLR, minLR, decayC):
    return lambda epoch : learningRateFunc(epoch, initLR, minLR, decayC);
def fitFuncGen(convLayerN, convFilterN, denseLayerN, denseNeuronN,\
               learningRate, actFunc, initFunc,\
               targetN, inputShape, inputX, inputY,\
               valiR, epochN, dropMCN):
    #############Adjustables#############
    minLearningRate       = pow(10, -6);
    scheduleExpDecayConst = 10;
    #####################################
    global OPTITER;
    global OPTACCU;
    global OPTASTD;
    parStr = str(convLayerN) + "-" + str(convFilterN) + "-" +\
             str(denseLayerN) + "-" + str(denseNeuronN) + "-" +\
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
    trainX, validX, trainY, validY = train_test_split(inputX, inputY, test_size=valiR);
    accuracies = [];
    for dropSeed in range(dropMCN):
        print("\n#######################################MONTE CARLO DROPOUT:", dropSeed);
        model=buildModel(convLayerN, convFilterN, denseLayerN, denseNeuronN,\
                         learningRate, actFunc, initFunc,\
                         targetN, inputShape, dropoutMCSeed=dropSeed);
        history = model.fit(trainX, trainY, validation_data=(validX, validY),\
                            epochs=epochN, callbacks=[tensorboard, scheduler]);
        accuracies.append(history.history["val_accuracy"][-1]);
        print("Drop-Seed", dropSeed, "Accuracy =", accuracies[dropSeed]);
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
                      "convLayerN":         convLayerN,\
                      "convFilterN":        convFilterN,\
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
def fitFuncLambda(targetN, inputShape, trainX, trainY, valiR, epochN, dropMCN):
    return lambda x : fitFuncGen(x[0], x[1], x[2], x[3], x[4], x[5], x[6],\
                                 targetN, inputShape, trainX, trainY,\
                                 valiR, epochN, dropMCN);





if __name__ == "__main__":
    exepath = os.path.dirname(os.path.abspath(__file__));

    fashionData = tf.keras.datasets.fashion_mnist;
    [[inputX, inputY], [testX, testY]] = fashionData.load_data();
    nameY = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",\
             "Sandal", "Shirt", "Sneaker", "Bag", "Boot"];
    inputXNorm = np.array([stand2dArray(X) for X in inputX]);
    testXNorm  = np.array([stand2dArray(X) for X in testX]);
    targetN = len(nameY);
    inputShape = [inputXNorm.shape[1], inputXNorm.shape[2], 1];   #note: need for conv2D
    inputXNorm = inputXNorm.reshape(inputXNorm.shape[0], \
                                    inputShape[0], inputShape[1], inputShape[2]);
    testXNorm  = testXNorm.reshape( testXNorm.shape[0], \
                                    inputShape[0], inputShape[1], inputShape[2]);
#####modeling#################################################################################
    '''
    #############Adjustables#############
    validationRatio     = 0.1;
    learnEpochN         = 6;
    dropoutMonteCarloN  = 10;
    optimizationCallN   = 30; #note: need >= 11
    par0 = [3, 256, 3, 128, pow(10, -3), "elu", "he_normal"];
    #####################################  
    convLayerN   = Integer(low=1,  high=5, name="convLayerNum");
    convFilterN  = Categorical(categories=[32, 64, 128, 256, 512], name="convFilterNum");
    denseLayerN  = Integer(low=1,  high=5, name="denseLayerNum");
    denseNeuronN = Integer(low=10, high=500, name="denseNeuronN");
    learningRate = Real(low=pow(10, -6), high=pow(10, -1),\
                        prior="log-uniform", name="learningRate");
    actFunc  = Categorical(categories=["elu", "selu"], name="activationFunc");
    initFunc = Categorical(categories=["he_normal", "he_uniform"], name="initializationFunc");
    dims = [convLayerN,convFilterN,denseLayerN,denseNeuronN,learningRate,actFunc,initFunc];
    fitFunc = fitFuncLambda(targetN, inputShape, inputXNorm, inputY, \
                            validationRatio, learnEpochN, dropoutMonteCarloN);
 
    checkpointPath = exepath+"/"+MODELNAME+"/checkpoint.pkl";
    checkpointSaver = CheckpointSaver(checkpointPath, compress=9, store_objective=False);
    eval0 = None;
    optParDict = {};
    #restore gp_minimize##################Remember to delete the .pkl file when changing model
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
    parOpt = [optParDict["convLayerN"], optParDict["convFilterN"],\
              optParDict["denseLayerN"], optParDict["denseNeutronN"],\
              optParDict["learningRate"], optParDict["activationFunc"],\
              optParDict["initializationFunc"]];
    fitFunc = fitFuncLambda(targetN, inputShape, inputXNorm, inputY,\
                            validationRatio, learnEpochN, dropoutMonteCarloN);
    optAccuracy = fitFunc(parOpt);
    '''
##############################################################################################
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
    print("Saving the following figures:\n    ", filenameFig);
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
        print("   ", filenameFig);








 
