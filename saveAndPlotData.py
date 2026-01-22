from HipoParser import HipoParser
from Plotter import Plotter

import numpy as np
import time
from matplotlib import pyplot as plt

# Parameters for nicer plots
plt.rcParams.update({'font.size': 40,
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

# File paths
# fp = '/w/work/clas12/tyson/data_repo/CDMLTracking/MLSample_1_reco.hipo'
fp=[]
for i in range(16):
    fp.append('/work/clas12/ziegler/CVT-AI-SAMPLES/dvcs-rga-fall2018/MCAIbg50na_out_bg50na_'+str(i)+'.hipo')
printDir = 'plots/input/'
printDirScaled = 'plots/scaledVars/'

startT_all = time.time()
reader = HipoParser(fp) #, max_events=1000)

print('Reading...')
startT_read = time.time()
hits_list, orders_list = reader.read_all()
endT_read = time.time()
print(f"Reading data took {endT_read-startT_read:.2f}s")

# Min/max dictionaries
global_min_max = {"layer": 1}
global_max_max = {"layer": 12}

per_layer_min = {
    "strip": {i: 1 for i in range(1, 13)},
    "x1": {1:-7,2:-8,3:-10,4:-10,5:-15,6:-15,7:-15,8:-18,9:-18,10:-23,11:-23,12:-23},
    "x2": {1:-7,2:-8,3:-10,4:-10,5:-15,6:-15,7:-15,8:-18,9:-18,10:-23,11:-23,12:-23},
    "y1": {1:-7,2:-8,3:-10,4:-10,5:-15,6:-15,7:-15,8:-18,9:-18,10:-23,11:-23,12:-23},
    "y2": {1:-7,2:-8,3:-10,4:-10,5:-15,6:-15,7:-15,8:-18,9:-18,10:-23,11:-23,12:-23},
    "z1": {1:-25,2:-25,3:-22,4:-22,5:-18,6:-18,7:-18,8:-21,9:-21,10:-21,11:-21,12:-21},
    "z2": {1:-25,2:-25,3:-22,4:-22,5:-18,6:-18,7:-18,8:-21,9:-21,10:-21,11:-21,12:-21},
    "sector": {i: 1 for i in range(1,13)}
}

per_layer_max = {
    "strip": {1:256,2:256,3:256,4:256,5:256,6:256,7:896,8:640,9:640,10:1024,11:768,12:1152},
    "x1": {1:7,2:8,3:10,4:10,5:15,6:15,7:15,8:18,9:18,10:23,11:23,12:23},
    "x2": {1:7,2:8,3:10,4:10,5:15,6:15,7:15,8:18,9:18,10:23,11:23,12:23},
    "y1": {1:7,2:8,3:10,4:10,5:15,6:15,7:15,8:18,9:18,10:23,11:23,12:23},
    "y2": {1:7,2:8,3:10,4:10,5:15,6:15,7:15,8:18,9:18,10:23,11:23,12:23},
    "z1": {1:25,2:25,3:22,4:22,5:18,6:18,7:21,8:21,9:21,10:25,11:25,12:25},
    "z2": {1:25,2:25,3:22,4:22,5:18,6:18,7:21,8:21,9:21,10:25,11:25,12:25},
    "sector": {1:11,2:11,3:15,4:15,5:19,6:19,7:3,8:3,9:3,10:3,11:3,12:3}
}

min_vals = {**per_layer_min, **global_min_max}
max_vals = {**per_layer_max, **global_max_max}

# Loop over output sectors
for output_file_idx in range(1,4):
    endName = f'_sector{output_file_idx}_noCSWeight_DVCSData'

    print(f'\nSector {output_file_idx}')
    print('Splitting by sector...')
    hits_split, orders_split = reader.split_by_output_file(hits_list, orders_list, output_file_idx)

    print('Plotting...')
    plotter = Plotter(x=hits_split, y=orders_split, printDir=printDir, endName=endName)
    for var in ["x1","y1","z1","x2","y2","z2","sector","strip"]: #,"cweight","sweight"
        plotter.compare_all_layers(var)

    #confusing data representation
    # plotter.plot_event_hits_on_strips(10, per_layer_max["strip"])
    # plotter.plot_event_hits_on_strips(20, per_layer_max["strip"])

    #slightly nicer data representation, overwrites previous
    plotter.plot_event_hits_polar(10)
    plotter.plot_event_hits_polar(20)

    # selected_vars  = ["strip","cweight","sweight","x1","x2","y1","y2","z1","z2","sector","layer"]
    selected_vars  = ["strip","x1","x2","y1","y2","z1","z2","sector","layer"]
    print('Scaling...')
    hits_split, orders_split = reader.trim_and_scale_batch((hits_split, orders_split), selected_vars, min_vals, max_vals)

    print('Plotting scaled...')
    plotter = Plotter(x=hits_split, y=orders_split, printDir=printDirScaled, endName=endName, col_names=selected_vars)
    for var in selected_vars:
        plotter.compare_all_layers(var, layer_scale=12)

    print('Padding...')
    hits_split, orders_split, mask_split = reader.pad_events(hits_split, orders_split)

    print('Splitting (train/test)...')
    hits_train, orders_train, mask_train, hits_test, orders_test, mask_test = reader.split_dataset(hits_split, orders_split, mask_split)

    print('Saving...')
    reader.save_dataset(f"hits_train{endName}.pt", hits_train, orders_train, mask_train)
    reader.save_dataset(f"hits_test{endName}.pt", hits_test, orders_test, mask_test)

    endT_all = time.time()
    print(f"Script took {endT_all - startT_all:.2f} s so far...")

endT_all = time.time()
print(f"Entire script took {endT_all - startT_all:.2f} s")
