import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import torch

from Classifier import Classifier
from HipoParser import HipoParser
from Plotter import Plotter

from pytorch_lightning import Trainer

import time

#parameters for nice plots
plt.rcParams.update({'font.size': 40,
                    #'font.family':  'Times New Roman',
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
                    'lines.linewidth' : 5})

startT_all = time.time()

#string to avoid plots being overwritten
endName='_sector1'

#avoid plot directories becoming too large
endNamePlotDir='' # _stripLayerOnly

# selected_vars = ["strip","cweight","sweight", "layer"]
selected_vars = ["strip","cweight","sweight","x1","x2","y1","y2","z1","z2","sector","layer"]
#selected_vars = ["cweight","sweight","x1","x2","y1","y2","z1","z2", "layer"]

#where to output plots
printDir='plots/scaledVars'+endNamePlotDir+'/'

reader = HipoParser("", bank_name="CVT::MLHit")

print('Loading Data...')

train_batch=HipoParser.load_batch("hits_train"+endName+endNamePlotDir+".pt")

print('Plotting...')

plotter = Plotter(train_batch, printDir=printDir, endName=endName,col_names=selected_vars)
# plotter.compare("x1")
if "x1" in selected_vars:
  plotter.compare_all_layers("x1",layer_scale=12)
if "y1" in selected_vars:
  plotter.compare_all_layers("y1",layer_scale=12)
if "z1" in selected_vars:
  plotter.compare_all_layers("z1",layer_scale=12)
if "x2" in selected_vars:
  plotter.compare_all_layers("x2",layer_scale=12)
if "y2" in selected_vars:
  plotter.compare_all_layers("y2",layer_scale=12)
if "z2" in selected_vars:
  plotter.compare_all_layers("z2",layer_scale=12)
if "sector" in selected_vars:
  plotter.compare_all_layers("sector",layer_scale=12)
if "strip" in selected_vars:
  plotter.compare_all_layers("strip",layer_scale=12)
if "cweight" in selected_vars:
  plotter.compare_all_layers("cweight",layer_scale=12)
if "sweight" in selected_vars:
  plotter.compare_all_layers("sweight",layer_scale=12)

endT_all = time.time()
T_all=endT_all-startT_all
print('\nEntire script took '+format(T_all,'.2f')+'s \n\n')