import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import utils

def combine_csv(paths):
    
    csvs = []
    for path in paths:
        csvs.append( pd.read_csv(path, sep=',', names=["Step", "Loss", "Min Loss", "Max Loss"], header=0))

    base = csvs[0]
    for csv in csvs[1:]:
        base = pd.concat((base, csv))
        #base.concat(csv)
    return base
import glob

files = glob.glob("./tsts/csv_combiner/c*.csv")
files.sort()
print(files)
df = combine_csv(files)

import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np

def setup_plot():
    mpl.rcParams['lines.linewidth'] = 1
    #mpl.rcParams['font.family'] = 'Microsoft Sans Serif'
    mpl.rcParams['font.family'] = 'Arial'

    
    #these don't work for some reason
    #mpl.rcParams['axes.titleweight'] = 'bold'
    #mpl.rcParams['axes.titlesize'] = '90'
    
    sns.set_theme(style="white", palette='pastel', font = 'Arial', font_scale=3)

    #sns.set_theme(style="white", palette='pastel', font = 'Microsoft Sans Serif', font_scale=1)
    #myFmt = mdates.DateFormatter('%b #Y')
    
    print("Plot settings applied")
#setup_plot()

utils.utility.get_hydra_config()
plt.plot(df["Loss"])
plt.show()