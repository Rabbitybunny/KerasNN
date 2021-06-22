import sys, os, time, math
import numpy as np
import pandas as pd
import pickle

def main():
    parFileName = sys.argv[1] + "/pars.pickle"
    optParDict = {};
    with open(parFileName, "rb") as handle:
        optParDict = pickle.load(handle);
    print(optParDict);


if __name__ == "__main__":
    main();    







 
