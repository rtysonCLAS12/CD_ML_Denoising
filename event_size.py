import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

from Classifier import Classifier, LossTracker
from HipoParser import HipoParser
from Plotter import Plotter

from pytorch_lightning import Trainer

import time

# -----------------------------
# Plotting params
# -----------------------------
plt.rcParams.update({
    'font.size': 40,
    'legend.edgecolor': 'white',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size':15,
    'xtick.minor.size':10,
    'ytick.major.size':15,
    'ytick.minor.size':10,
    'xtick.major.width':3,
    'xtick.minor.width':3,
    'ytick.major.width':3,
    'ytick.minor.width':3,
    'axes.linewidth' : 3,
    'figure.max_open_warning':200,
    'lines.linewidth' : 5
})

startT_all = time.time()

# -----------------------------
# Setup paths and plotter
# -----------------------------
endName = '_sector1'
endNamePlotDir = ''
endNamePlot = '_weightInTraining'
printDir = '/w/work/clas12/tyson/plots/CD_ML_Tracking/training'+endNamePlotDir+'/'

plotter = Plotter(printDir=printDir, endName=endName)

doTraining = True
nEpoch = 100

# Select variables for plotting
selected_vars  = ["strip","cweight","sweight","x1","x2","y1","y2","z1","z2","sector","layer"]

# -----------------------------
# Load data
# -----------------------------
print('Loading Data...')
startT_load = time.time()

data  = HipoParser.load_dataset("hits_test"+endName+endNamePlotDir+".pt")

y = data["y"]
mask = data["mask"]

plotter = Plotter(printDir=printDir, endName=endName)
plotter.plotEventSize(y,mask)
