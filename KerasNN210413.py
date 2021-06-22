import sys, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
#tensorboard --logdir=... --host localhost --port 6006
#browse: http://localhost:6006/
import tensorflow as tf
from tensorflow.keras.activations import *
#https://towardsdatascience.com/bayesian-hyper-parameter-optimization-neural-networks
#-tensorflow-facies-prediction-example-f9c48d21f795
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer

def stand2dArray(array):    #following tf.image.per_image_standardization
    array = np.array(array);
    flatArr = array.flatten();
    mean = np.mean(flatArr);
    std = np.std(flatArr);
    return (array - mean)/max(std, 1/math.sqrt(flatArr.size));
def buildModel(denseLayerN, denseNeuronN, learningRate, activationFunc,\
               targetN, inputShape):
    model = tf.keras.models.Sequential();
    model.add(tf.keras.layers.Flatten(input_shape=inputShape));
    for i in range(denseLayerN):
        name = "denseLayer" + str(i);
        model.add(tf.keras.layers.Dense(denseNeuronN, name=name,\
                                        activation=activationFunc));
    model.add(tf.keras.layers.Dense(targetN, activation="softmax"));
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=learningRate),\
                  loss=tf.keras.losses.sparse_categorical_crossentropy,\
                  metrics=["accuracy"]);
    return model;

OPTITER=0;
OPTACCU=0;
def fitFuncGen(denseLayerN, denseNeuronN, learningRate, activationFunc,\
               targetN, inputShape, trainX, trainY, epochN):
    validationR = 0.1;
    global OPTITER;
    global OPTACCU;
    OPTITER += 1;
    parStr = str(denseLayerN) + "_" + str(denseNeuronN) + "_" +\
             str(learningRate) + "_" + str(activationFunc); 
    print("##################################################################BEGIN", OPTITER);
    print("pars:", parStr);
    model=buildModel(denseLayerN,denseNeuronN,learningRate,activationFunc,\
                     targetN, inputShape);
    tensorboardLogDir = exepath + "/mnistFashion210413.model/tensorboardDir/";
    tensorboardLogDir += str(int(time.time())) + "-" + parStr;
    tensorboard=tf.keras.callbacks.TensorBoard(log_dir=tensorboardLogDir,\
                                               histogram_freq=0,\
                                               write_graph=True,\
                                               write_grads=False,\
                                               write_images=False);
    history = model.fit(trainX, trainY, epochs=epochN, validation_split=validationR,\
                        callbacks=[tensorboard]);
    accuracy = history.history["val_accuracy"][-1];
    print("accuracy =", accuracy);
    print("##################################################################END\n\n");
    if accuracy > OPTACCU:
        OPTACCU = accuracy;
        histDF = pd.DataFrame(history.history);
        model.save("mnistFashion210413.model");
        histDF.to_pickle("mnistFashion210413.model/history.pickle");
    del model;
    return -accuracy;
def fitFuncLambda(targetN, inputShape, trainX, trainY, epochN):
    return lambda x : fitFuncGen(x[0], x[1], x[2], x[3],\
                                 targetN, inputShape, trainX, trainY, epochN);
if __name__ == "__main__":
    exepath = os.path.dirname(os.path.abspath(__file__));

    fashionData = tf.keras.datasets.fashion_mnist;
    [[trainX, trainY], [testX, testY]] = fashionData.load_data();
    nameY = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",\
             "Sandal", "Shirt", "Sneaker", "Bag", "Boot"];
    trainXNorm = np.array([stand2dArray(X) for X in trainX]);
    testXNorm  = np.array([stand2dArray(X) for X in testX]);
    targetN = 10;
    inputShape = [28, 28];
    '''
    #modeling
    denseLayerN    = Integer(low=1, high=10, name="denseLayerNum");
    denseNeuronN   = Integer(low=10, high=500, name="denseNeuronN");
    learningRate   = Real(low=pow(10, -6), high=pow(10, -1),\
                          prior="log-uniform", name="learningRate");
    activationFunc = Categorical(categories=["relu", "sigmoid"],\
                                 name="activationFunc")
    dims = [denseLayerN, denseNeuronN, learningRate, activationFunc];
    par0 = [3, 100, pow(10, -3), "relu"];
    learnEpochN = 30;
    optCallN    = 30;
    fitFunc = fitFuncLambda(targetN, inputShape, trainXNorm, trainY, learnEpochN);
    result = gp_minimize(func=fitFunc, dimensions=dims, x0=par0,\
                         acq_func="EI", n_calls=optCallN);
    '''
    #evaluating
    model = tf.keras.models.load_model("mnistFashion210413.model");
    histDF = pd.read_pickle("mnistFashion210413.model/history.pickle");

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




 
