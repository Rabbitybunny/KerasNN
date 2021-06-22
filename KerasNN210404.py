import sys, os, time, math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import pickle


def stand2dArray(array):    #following tf.image.per_image_standardization
    array = np.array(array);
    flatArr = array.flatten();
    mean = np.mean(flatArr);
    std = np.std(flatArr);
    return (array - mean)/max(std, 1/math.sqrt(flatArr.size));


if __name__ == "__main__":
    printFigN = 20;

    fashionData = tf.keras.datasets.fashion_mnist;
    [[trainX, trainY], [testX, testY]] = fashionData.load_data();
    nameY = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",\
             "Sandal", "Shirt", "Sneaker", "Bag", "Boot"];
    trainXNorm = np.array([stand2dArray(X) for X in trainX]);
    testXNorm  = np.array([stand2dArray(X) for X in testX]);

    exepath = os.path.dirname(os.path.abspath(__file__));
    print("Saving the following figures:");
    ''' 
    #printing training images
    for i, valX in enumerate(trainX):
        plt.imshow(valX, cmap=plt.cm.binary);
        plt.title(nameY[trainY[i]] , fontsize=24);
        filenameFig = exepath + "/fashionFig/trainFashion"+str(i)+".png";
        plt.savefig(filenameFig, dpi=50);
        plt.close();
        print(filenameFig);
    '''
    #modeling
    model = tf.keras.models.Sequential();
    model.add(tf.keras.layers.Flatten(input_shape=[28, 28]));
    model.add(tf.keras.layers.Dense(300, activation="relu"));
    model.add(tf.keras.layers.Dense(100, activation="relu"));
    model.add(tf.keras.layers.Dense(len(nameY), activation="softmax"));
    #print(model.summary(), "\n");
    #print(model.layers, "\n");
    #[weights, biases] = model.layers[1].get_weights();
    #print([weights.shape, biases.shape], "\n");
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01),\
                  loss=tf.keras.losses.sparse_categorical_crossentropy,\
                  metrics=["accuracy"]);
    tensorboardLogDir = exepath + "/mnistFashion.model/tensorboardDir/";
    tensorboard = tf.keras.callbacks.TensorBoard(tensorboardLogDir);
    history = model.fit(trainXNorm, trainY, epochs=30, validation_split=0.1,\
                        callbacks=[tensorboard]);
    histDF = pd.DataFrame(history.history);
    model.save("mnistFashion.model");
    histDF.to_pickle("mnistFashion.model/history.pickle");

    model  = tf.keras.models.load_model("mnistFashion.model");
    histDF = pd.read_pickle("mnistFashion.model/history.pickle");

    model.evaluate(x=testXNorm, y=testY);
    print();
    histDF.plot(figsize=(8, 5));
    plt.title("Learning Performance History");
    plt.grid("True");
    plt.gca().set_ylim(0.0, 1.0);
    filenameFig = exepath + "/fashionFig/-fashionHistory.png";
    plt.savefig(filenameFig);
    plt.close();
    print(filenameFig, "\n");


    predValY = model.predict(testXNorm);
    predY = np.argmax(predValY, axis=-1);
    for i, valX in enumerate(testX[:printFigN]):
        print(i, nameY[predY[i]], nameY[testY[i]]);
        plt.imshow(valX, cmap=plt.cm.binary);
        plt.title("Prediction: "+nameY[predY[i]], fontsize=24);
        filenameFig = exepath + "/fashionFig/predFashion"+str(i)+".png";
        plt.savefig(filenameFig, dpi=50);
        plt.close();
        print(filenameFig);






 
