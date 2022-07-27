# Keras Neural Network on MNIST Fashion Data
These are python3 code more or less following
_Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems 2nd Edition_
from Aurélien Géron.

210410: loss: 0.4219 - accuracy: 0.8789, (pg321)<br/>
&ensp;&ensp; Uses RandomizedSearchCV to randomly search for the best parameters for the model

210413: loss: 0.3332 - accuracy: 0.8852, (skopt, pg322)<br/>
&ensp;&ensp; Uses gp_minimize to gradient-search for the best parameters for the model

210416: loss: 0.3150 - accuracy: 0.8867, (Tab. 11-3, pg371)<br/>
&ensp;&ensp; Applies:<br/>
&ensp;&ensp;&ensp;&ensp; kernel initialization,<br/>
&ensp;&ensp;&ensp;&ensp; non-saturating activation function,<br/>
&ensp;&ensp;&ensp;&ensp; batch normalization,<br/>
&ensp;&ensp;&ensp;&ensp; Monte Carlo dropout for regularization,<br/>
&ensp;&ensp;&ensp;&ensp; improving optimizer,<br/>
&ensp;&ensp;&ensp;&ensp; exponential scheduling<br/>
            
210422: loss: 0.2855 - accuracy: 0.8973, (pg461)<br/>
&ensp;&ensp; Applies the convolutional neural network using conv2D 

210617: loss: 0.3940 - accuracy: 0.9022, (ResNet-34, pg478)<br/>
&ensp;&ensp; Applies the residual neural network

This final result of ResNet is not quite satisfactory; it's likely causes by a mistake in input data normalization. May have to every step all over again.

### With 80% samples set as untagged:

210521: loss: 0.3895 - accuracy: 0.8590, (pg574-582)<br/>
&ensp;&ensp; Applies autoencoder unsupervised pretraining with tagged+untagged samples<br/>
&ensp;&ensp; The remaining tagged ~300sample/10features<br/>
&ensp;&ensp; Without pretraining: loss: 0.4145 - accuracy: 0.8487  

210611: loss: 0.3919 - accuracy: 0.8827 (ResNet-34, pg478)<br/>
&ensp;&ensp; Applies the residual neural network with a residualBlock() function + autoencoder
&ensp;&ensp; Not using the full power, only

            for filterN in [64] + [128] + [256] + [512]:

However, it already requires GPU from colab with only ~300 samples.
