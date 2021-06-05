import sys, os, time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def buildModel(targetN=10, input_shape=[28, 28], \
               hiddenN=2, neuronN=100, learningRate=0.01):
    model = tf.keras.models.Sequential();
    model.add(tf.keras.layers.Flatten(input_shape=input_shape));
    for i in range(hiddenN):
        model.add(tf.keras.layers.Dense(neuronN, activation="relu"));
    model.add(tf.keras.layers.Dense(targetN, activation="softmax"));
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=learningRate),\
                  loss=tf.keras.losses.sparse_categorical_crossentropy,\
                  metrics=["accuracy"]);
    return model;

if __name__ == "__main__":
    exepath = os.path.dirname(os.path.abspath(__file__));

    fashionData = tf.keras.datasets.fashion_mnist;
    [[trainX, trainY], [testX, testY]] = fashionData.load_data();
    nameY = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",\
             "Sandal", "Shirt", "Sneaker", "Bag", "Boot"];
    trainXNorm = tf.keras.utils.normalize(trainX, axis=1);
    testXNorm  = tf.keras.utils.normalize(testX, axis=1);  
    '''
    #modeling
    epochN = 30;
    params = {"hiddenN": [0, 1, 2, 3, 4, 5],\
              "neuronN": [50, 100, 150, 200],\
              "learningRate": [0.01, 0.005, 0.001, 0.0005]};
    kerasReg = tf.keras.wrappers.scikit_learn.KerasRegressor(buildModel);
    #RandomizedSearchCV and tensorboard works badly together
    #https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
    #https://stackoverflow.com/questions/45454905
    #https://medium.com/ml-book/neural-networks-hyperparameter-tuning-in-tensorflow-2-0-a7b4e2b574a1
    modelSearch = RandomizedSearchCV(kerasReg, params, n_iter=30, cv=3);
    tensorboardLogDir = exepath + "/mnistFashion210410.model/tensorboardDir/";
    tensorboard = tf.keras.callbacks.TensorBoard(tensorboardLogDir);
    history = modelSearch.fit(trainXNorm, trainY, epochs=epochN, \
                              validation_split=0.1, callbacks=[tensorboard]);
    print("Best params:", modelSearch.best_params_);
    print("Best score: ", modelSearch.best_score_);
    model = modelSearch.best_estimator_.model;
    history = model.fit(trainXNorm, trainY, epochs=epochN, validation_split=0.1);
    histDF = pd.DataFrame(history.history);
    model.save("mnistFashion210410.model");
    histDF.to_pickle("mnistFashion210410.model/history.pickle");
    '''
    #evaluating
    model = tf.keras.models.load_model("mnistFashion210410.model");
    histDF = pd.read_pickle("mnistFashion210410.model/history.pickle");

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





 
