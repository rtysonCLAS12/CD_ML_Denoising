import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

from Classifier import Classifier, LossTracker
from HipoParser import HipoParser
from Plotter import Plotter

from pytorch_lightning import Trainer

import time
import os

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

#cuda helping things
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # predictable ordering
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # ensure single device visible
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"        # (for debugging only, optional)

torch.set_float32_matmul_precision('high')
torch.cuda.init()
torch.cuda.empty_cache()
print("Using device:", torch.cuda.get_device_name(0))

startT_all = time.time()

# -----------------------------
# Setup paths and plotter
# -----------------------------
endName = '_sector1'
endNamePlotDir = ''
endNamePlot = '_weightInTraining'
printDir = '/w/work/clas12/tyson/plots/CD_ML_Denoising/training'+endNamePlotDir+'/'

plotter = Plotter(printDir=printDir, endName=endName)

doTraining = True
nEpoch = 50

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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

endT_load = time.time()
print(f'\nLoading Data took {endT_load-startT_load:.2f}s\n\n')

# -----------------------------
# Define model
# -----------------------------
#okay hf=64,nl=8,lr=5e-4,s=30,k=60
#same and slower hf=64,nl=8,lr=5e-4,s=30,k=120
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
else:
    print("GPU requested but not available. Falling back to CPU.")
    accelerator = "cpu"
    devices = 1

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
    model.export_to_torchscript("classifier_torchscript"+endName+endNamePlotDir+endNamePlot+".pt")

    endT_train = time.time()
    T_train = endT_train - startT_train
    Rate_train = ((n_train*nEpoch)/T_train)/1000.
    print(f'\nTraining took {T_train:.2f}s, Eg rate of {Rate_train:.4f} kHz per epoch\n\n')

# -----------------------------
# Load model for inference
# -----------------------------
model = Classifier.load_from_torchscript(
    "classifier_torchscript"+endName+endNamePlotDir+endNamePlot+".pt",
    in_features=n_features
)

# -----------------------------
# Testing
# -----------------------------
print('Testing...')
startT_test = time.time()

all_probs = []
all_labels = []

example_file = "example"+endName+endNamePlotDir+endNamePlot+".txt"
example_saved = False  # flag to save only once

model.eval()
with torch.no_grad():
    for batch in val_loader:
        x_batch, y_batch, mask_batch = batch
        probs = model(x_batch, mask_batch)
        # Flatten while respecting mask
        valid_probs = probs[mask_batch !=0]
        valid_labels = y_batch[mask_batch !=0]
        all_probs.append(valid_probs)
        all_labels.append(valid_labels)

        # Save example once
        if not example_saved:
            x_row = x_batch.cpu().numpy()[0]    
            y_row = y_batch.cpu().numpy()[0]      
            mask_row = mask_batch.cpu().numpy()[0]
            probs_row = probs.cpu().numpy()[0]

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

all_probs = torch.cat(all_probs)
all_labels = torch.cat(all_labels)

# Apply threshold
threshold = 0.5
all_preds = (all_probs >= threshold).long()

endT_test = time.time()
T_test = endT_test - startT_test
Rate_test = (n_test/T_test)/1000.
print(f'\nTesting took {T_test:.2f}s, Eg rate of {Rate_test:.4f} kHz\n\n')

signal_mask = all_labels == 1
frac_signal_retained = (all_preds[signal_mask] == 1).float().mean().item()
noise_mask = all_labels == 0
frac_noise_removed = (all_preds[noise_mask] == 0).float().mean().item()

print(f"Fraction of signal events retained: {frac_signal_retained*100:.2f}%")
print(f"Fraction of noise events removed: {frac_noise_removed*100:.2f}%")

endT_all = time.time()
T_all = endT_all - startT_all
print(f'\nEntire script took {T_all:.2f}s\n\n')