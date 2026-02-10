import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch

from Classifier import Classifier
from HipoParser import HipoParser
from Plotter import Plotter

import numpy as np

import time

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Classifier testing")

    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--s", type=int, default=1)
    parser.add_argument("--modelType", type=str, default="gravnet")
    parser.add_argument("--outdir", type=str, default="./",
                        help="Directory to save outputs (models, plots)")

    return parser.parse_args()


def findVar(varName,vars):
    i=0
    for var in vars:
      if varName==var:
         return i
      i=i+1
    return -1

# -----------------------------
# Global min/max dictionaries
# -----------------------------
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

# -----------------------------
# Plot settings
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
    'axes.linewidth':3,
    'figure.max_open_warning':200,
    'lines.linewidth':5
})

startT_all = time.time()

args = parse_args()

endName='_sector1_noCSWeight'
endNameModel='_GarNet'
endNamePlotDir=''
endNamePlot='_weightInTraining'
outDir = args.outdir
printDir=outDir + 'plots/training'+endNamePlotDir+'/'

# -----------------------------
# Load test data
# -----------------------------
reader = HipoParser("", bank_name="CVT::MLHit")
print('Loading Data...')
startT_load = time.time()


test_data  = HipoParser.load_dataset("hits_test"+endName+endNamePlotDir+".pt")

# Select variables for plotting
selected_vars  = ["strip","cweight","sweight","x1","x2","y1","y2","z1","z2","sector","layer"]
# Selected variables
if ('_xyOnly' in endNamePlotDir) or ('_xyOnly' in endName):  
  selected_vars = ["cweight","sweight","x1","x2","y1","y2","z1","z2", "layer"]
elif ('_stripLayerOnly' in endNamePlotDir) or ('_stripLayerOnly' in endName):  
  selected_vars = ["strip","cweight","sweight", "layer"]
elif ('_noCSWeight' in endNamePlotDir) or ('_noCSWeight' in endName):  
  selected_vars  = ["strip","x1","x2","y1","y2","z1","z2","sector","layer"]

layer_var_index=findVar("layer",selected_vars)

x_test = test_data["x"]
y_test = test_data["y"]
mask_test = test_data["mask"]

#cap to 6 first layers
# test_events  = mask_events(test_events, layer_var_index, threshold=0.5)

numworkers=4
if torch.backends.mps.is_available():
    numworkers=0 # MPS doesn't support multi-threaded data loading


n_test = x_test.size(0)
print(f"Test events: {n_test}")

test_dataset = TensorDataset(x_test, y_test, mask_test)

#batch size of 1 to get event by event info
val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=numworkers)

endT_load = time.time()
print(f'\nLoading Data took {endT_load-startT_load:.2f}s\n\n')

model = Classifier.load_from_torchscript(
    outDir + "nets/classifier_torchscript"+endName+endNameModel+endNamePlotDir+endNamePlot+".pt",
    in_features=len(selected_vars),
    hidden_features=args.hidden_features,
    num_layers=args.num_layers,
    lr=args.lr,
    k=args.k,
    s=args.s,
    modelType=args.modelType
)

#check model is well scripted
# print(model.model.code)
# print(model.model.convs[0].code)


print('Testing...')
startT_test = time.time()

all_probs = []
all_preds = []
all_labels = []
all_x = []

model.eval()
example_file = outDir + "nets/example"+endName+endNameModel+endNamePlotDir+endNamePlot+".txt"
example_saved = False  # flag to save only once

with torch.no_grad():
    for batch in val_loader:
        
        #cap nEx
        # if len(all_x)>=50:
        #   break

        x_batch, y_batch, mask_batch = batch
        # Convert to numpy
        x_batch_np = x_batch.cpu().numpy()      # [B, H, F]
        y_batch_np = y_batch.cpu().numpy()      # [B, H]
        mask_batch_np = mask_batch.cpu().numpy()# [B, H]

        # Get model predictions and convert to numpy
        probs = model(x_batch, mask_batch)      # [B, H]
        probs_np = probs.cpu().numpy()          # [B, H]

        # Boolean mask for valid hits
        mask_bool = mask_batch_np != 0          # [B, H]

        # Select only valid hits
        x = x_batch_np[mask_bool]         # [num_valid_hits, F]
        labels = y_batch_np[mask_bool]         # [num_valid_hits]
        mask_valid = mask_batch_np[mask_bool]   # [num_valid_hits]
        probs_valid = probs_np[mask_bool]       # [num_valid_hits]
        preds = (probs_valid >= 0.5).astype(int)

        # print(probs_valid.shape)
        # print(x.shape)
        # print(labels.shape)

        all_probs.append(probs_valid)
        all_preds.append(preds)
        all_labels.append(labels)
        all_x.append(x)

        # Save example once
        if not example_saved:
            x_row = x_batch_np[0]        # [H, F]
            y_row = y_batch_np[0]        # [H]
            mask_row = mask_batch_np[0]  # [H]
            probs_row = probs_np[0]      # [H]

            with open(example_file, "w") as f:
                f.write("x:\n")
                for row in x_row:
                    f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

                f.write("\ny:\n")
                for val in y_row:
                    f.write(f"{val}\n")

                f.write("\nmask:\n")
                for val in mask_row:
                    f.write(f"{val}\n")

                f.write("\nprobs:\n")
                for val in probs_row:
                    f.write(f"{val:.6f}\n")

            example_saved = True

all_x_unscaled=reader.unscale_x(all_x, selected_vars, min_vals, max_vals, layer_scale=12)

#already masked
plotter = Plotter(x=all_x_unscaled, y=all_labels, printDir=printDir, endName=endName+endNameModel+endNamePlot, col_names=selected_vars)

all_preds_list=all_preds
all_probs = np.concatenate(all_probs)
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

endT_test = time.time()
Rate_test = (n_test / (endT_test - startT_test)) / 1000.
print(f'\nTesting took {endT_test-startT_test:.2f}s, Eg rate: {Rate_test:.2f} kHz\n\n')

# -----------------------------
# Plot responses and efficiencies
# -----------------------------
plotter.plotResp(all_probs, all_labels)
plotter.compare_all_layers_resp(all_probs, all_labels)
plotter.plot_efficiencies(all_probs, all_labels)
plotter.plot_event_hits_polar(30)
plotter.plot_event_hits_polar(30, all_preds_list)

# -----------------------------
# Metrics
# -----------------------------

# all_preds, all_labels are 1D NumPy arrays of the same length
signal_mask = all_labels == 1       # boolean array
noise_mask = all_labels == 0        # boolean array

# Fraction of signal hits correctly predicted as signal
if signal_mask.sum() > 0:
    frac_signal_retained = (all_preds[signal_mask] == 1).mean()
else:
    frac_signal_retained = 0.0

# Fraction of noise hits correctly predicted as noise
if noise_mask.sum() > 0:
    frac_noise_removed = (all_preds[noise_mask] == 0).mean()
else:
    frac_noise_removed = 0.0

print(f"Fraction of signal hits retained: {frac_signal_retained*100:.2f}%")
print(f"Fraction of noise hits removed: {frac_noise_removed*100:.2f}%")
