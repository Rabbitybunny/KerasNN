import sys, os, time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import pickle

if __name__ == "__main__":
#part 1
    x = tf.Variable(2, name="x");
    y = tf.Variable(5, name="y");
    f = x*y + x + 2;
    print(f.numpy());




 
