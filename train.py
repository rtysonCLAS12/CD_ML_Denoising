import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

from Classifier import Classifier, LossTracker
from HipoParser import HipoParser
from Plotter import Plotter

import numpy as np

from pytorch_lightning import Trainer

import time
import os

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
printDir = 'plots/training'+endNamePlotDir+'/'

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

train_data = HipoParser.load_dataset("hits_train"+endName+endNamePlotDir+".pt")
test_data  = HipoParser.load_dataset("hits_test"+endName+endNamePlotDir+".pt")

x_train = train_data["x"]
y_train = train_data["y"]
mask_train = train_data["mask"]

x_test = test_data["x"]
y_test = test_data["y"]
mask_test = test_data["mask"]

n_train = x_train.size(0)
n_test = x_test.size(0)
n_features = x_train.size(2)

print(f"Training events: {n_train}, Test events: {n_test}")
print(f"Features per hit: {n_features}")

# Compute signal/background statistics
total_signal = ((y_train == 1) * mask_train).sum().item()
total_background = ((y_train == 0) * mask_train).sum().item()
frac_signal_to_bkg = total_signal / total_background
print(f"Signal hits: {total_signal}, Background hits: {total_background}, ratio={frac_signal_to_bkg:.4f}")

# Create weighted masks
train_weights = mask_train.clone()
train_weights[y_train == 0] *= frac_signal_to_bkg

test_weights = mask_test.clone()
test_weights[y_test == 0] *= frac_signal_to_bkg

print("\n=== Debug Info ===")
sample_event = 0
sample_mask = mask_train[sample_event]
sample_weights = train_weights[sample_event]  
n_valid = sample_mask.sum().item()
print(f"Sample event {sample_event}: {n_valid} valid hits out of {mask_train.size(1)}")
if n_valid > 0:
    sample_y = y_train[sample_event][sample_mask != 0]
    sample_w = sample_weights[sample_mask != 0]  # Get weights for valid hits
    print(f"  Signal: {(sample_y == 1).sum().item()}, Noise: {(sample_y == 0).sum().item()}")
    print(f"  Weight for signal hits: {sample_w[sample_y == 1]}")
    print(f"  Weight for noise hits (first 5): {sample_w[sample_y == 0][:5]}")
    print(f"  Feature range: min={x_train[sample_event][sample_mask == 1].min():.3f}, max={x_train[sample_event][sample_mask == 1].max():.3f}")
print("==================\n")

# Create datasets
train_dataset = TensorDataset(x_train, y_train, train_weights)
test_dataset = TensorDataset(x_test, y_test, test_weights)

endT_load = time.time()
print(f'\nLoading Data took {endT_load-startT_load:.2f}s\n\n')

model = Classifier(
    in_features=n_features,
    hidden_features=64,
    num_layers=16,
    lr=5e-4,
    k=30
)

loss_tracker = LossTracker()

if torch.cuda.is_available():
    accelerator = "gpu"
    devices = 1
    #cuda helping things
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # predictable ordering
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # ensure single device visible
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"        # (for debugging only, optional)

    torch.set_float32_matmul_precision('high')
    torch.cuda.init()
    torch.cuda.empty_cache()
    print("Using device:", torch.cuda.get_device_name(0))
    numworkers=4
elif torch.backends.mps.is_available():
    devices = 1
    accelerator = "mps"
    numworkers=0
    torch.set_float32_matmul_precision('medium')
else:
    print("GPU requested but not available. Falling back to CPU.")
    accelerator = "cpu"
    devices = 1
    numworkers=4

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=numworkers)
val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=numworkers)

trainer = Trainer(
    max_epochs=nEpoch,
    accelerator=accelerator,
    devices=devices,
    strategy="auto",
    enable_progress_bar=True,
    log_every_n_steps=1,
    enable_checkpointing=False,
    check_val_every_n_epoch=1,
    num_sanity_val_steps=0,
    callbacks=[loss_tracker]
)

# -----------------------------
# Training
# -----------------------------
if doTraining:
    print('Training...')
    startT_train = time.time()

    trainer.fit(model, train_loader, val_loader)

    # Plot training loss
    plotter.plotTrainLoss(loss_tracker)

    # Save TorchScript model
    model.export_to_torchscript("nets/classifier_torchscript"+endName+endNamePlotDir+endNamePlot+".pt")

    endT_train = time.time()
    T_train = endT_train - startT_train
    Rate_train = ((n_train*nEpoch)/T_train)/1000.
    print(f'\nTraining took {T_train:.2f}s, Eg rate of {Rate_train:.4f} kHz per epoch\n\n')

# -----------------------------
# Load model for inference
# -----------------------------
model = Classifier.load_from_torchscript(
    "nets/classifier_torchscript"+endName+endNamePlotDir+endNamePlot+".pt",
    in_features=n_features
)

# -----------------------------
# Testing
# -----------------------------
print('Testing...')
startT_test = time.time()

test_dataset = TensorDataset(x_test, y_test, mask_test)

val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=numworkers)

all_probs = []
all_preds = []
all_labels = []
all_x = []

model.eval()
example_file = "nets/example"+endName+endNamePlotDir+endNamePlot+".txt"
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

reader = HipoParser("", bank_name="CVT::MLHit")
all_x_unscaled=reader.unscale_x(all_x, selected_vars, min_vals, max_vals, layer_scale=12)

#already masked
plotter = Plotter(x=all_x_unscaled, y=all_labels, printDir=printDir, endName=endName+endNamePlot, col_names=selected_vars)

all_preds_list=all_preds
all_probs = np.concatenate(all_probs)
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

endT_test = time.time()
Rate_test = (n_test / (endT_test - startT_test)) / 1000.
print(f'\nTesting took {endT_test-startT_test:.2f}s, Eg rate: {Rate_test:.2f} kHz\n\n')

plotter.plotResp(all_probs, all_labels)
plotter.compare_all_layers_resp(all_probs, all_labels)
plotter.plot_efficiencies(all_probs, all_labels)
plotter.plot_event_hits_on_strips(30, per_layer_max["strip"])
plotter.plot_event_hits_on_strips(30, per_layer_max["strip"], all_preds_list)

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

endT_all = time.time()
T_all = endT_all - startT_all
print(f'\nEntire script took {T_all:.2f}s\n\n')