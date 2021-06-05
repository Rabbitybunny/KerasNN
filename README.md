# ML_KerasNN

210410: loss: 0.4219 - accuracy: 0.8789

    Uses RandomizedSearchCV to randomly search for the best parameters for the model

210413: loss: 0.3332 - accuracy: 0.8852

    Uses gp_minimize to gradient-search for the best parameters for the model

210416: loss: 0.3150 - accuracy: 0.8867

    Applies kernel initialization, 
    
            non-saturating activation function, 
            
            batch normalization, 
            
            Monte Carlo dropout for regularization, 

            improving optimizer, 
            
            exponential scheduling
            
210422: loss: 0.2855 - accuracy: 0.8973

    Applies the convolutional neural network using conv2D

210521: loss: 0.3895 - accuracy: 0.8590 (80% samples set untagged)

    Applies autoencoder unsupervised pretraining with tagged+untagged samples
    
    The remaining tagged ~300sample/10features
    
    Without pretraining: loss: 0.4145 - accuracy: 0.8487
