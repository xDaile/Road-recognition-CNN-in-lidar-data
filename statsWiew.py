#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import sys

#for showing created statistics
#usage ./statsWiew statsFile.csv outFile

if(len(sys.argv)!=3):
    print("Error: Usage ./statsWiew statsFile.csv outFile")
    exit()

data=pd.read_csv(sys.argv[1], index_col =False,header = None).astype('float')


plt.imshow(data)
plt.savefig(sys.argv[2])
