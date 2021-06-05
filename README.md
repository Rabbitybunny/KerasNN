# ML_KerasNN

210410: loss: 0.4219 - accuracy: 0.8789  
&ensp;&ensp; Uses RandomizedSearchCV to randomly search for the best parameters for the model

210413: loss: 0.3332 - accuracy: 0.8852  
&ensp;&ensp; Uses gp_minimize to gradient-search for the best parameters for the model

210416: loss: 0.3150 - accuracy: 0.8867  
&ensp;&ensp; Applies:  
&ensp;&ensp;&ensp;&ensp; kernel initialization,  
&ensp;&ensp;&ensp;&ensp; non-saturating activation function,  
&ensp;&ensp;&ensp;&ensp; batch normalization,  
&ensp;&ensp;&ensp;&ensp; Monte Carlo dropout for regularization,  
&ensp;&ensp;&ensp;&ensp; improving optimizer,  
&ensp;&ensp;&ensp;&ensp; exponential scheduling  
            
210422: loss: 0.2855 - accuracy: 0.8973  
&ensp;&ensp; Applies the convolutional neural network using conv2D  

210521: loss: 0.3895 - accuracy: 0.8590 (80% samples set untagged)  
&ensp;&ensp; Applies autoencoder unsupervised pretraining with tagged+untagged samples  
&ensp;&ensp; The remaining tagged ~300sample/10features  
&ensp;&ensp; Without pretraining: loss: 0.4145 - accuracy: 0.8487  
