import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
matplotlib.use('Agg')
import numpy as np
import os
import torch


class Plotter:
    """
    Plotter for CVT hit data stored as lists of arrays.
    Saves plots to files instead of showing them.
    """

    def __init__(self, x=None, y=None, mask=None, printDir="./", endName="", col_names=None):
        """
        Parameters
        ----------
        x : list of numpy arrays
            List of hit feature arrays, one per event
        y : list of numpy arrays
            List of label arrays, one per event
        mask : list of numpy arrays, optional
            List of mask arrays, one per event
        printDir : str
            Directory where plots are saved.
        endName : str
            String appended at the end of filenames.
        col_names : list of str
            Names for hit feature columns.
        """
        self.x = x
        self.y = y
        self.mask = mask

        self.printDir = printDir
        self.endName = endName

        os.makedirs(self.printDir, exist_ok=True)

        if col_names is None:
            self.columns = ["strip", "layer", "sector",
                            "x1", "y1", "z1", "x2", "y2", "z2",
                            "cweight","sweight"]
        else:
            self.columns = col_names
        self.col_index = {name: i for i, name in enumerate(self.columns)}

    def _get_concatenated_data(self):
        """Concatenate all events for plotting - only called when needed"""
        if self.x is None:
            raise ValueError("No data loaded")
        
        hits = np.vstack(self.x)
        orders = np.hstack(self.y)
        event_indices = np.hstack([np.full(len(h), i) for i, h in enumerate(self.x)])
        
        return hits, orders, event_indices

    def _select(self, var, layer=None):
        hits, orders, _ = self._get_concatenated_data()
        
        idx = self.col_index[var]
        vals = hits[:, idx]
        labs = orders

        if layer is not None:
            layer_mask = hits[:, self.col_index["layer"]] == layer
            vals, labs = vals[layer_mask], labs[layer_mask]

        vals0 = vals[labs == 0]
        vals1 = vals[labs == 1]
        return vals0, vals1

    def setPrintDir(self, pd):
        self.printDir = pd
    
    def setData(self, x, y, mask=None):
        self.x = x
        self.y = y
        self.mask = mask

    def setColumnNames(self, col_names):
        self.columns = col_names
        self.col_index = {name: i for i, name in enumerate(self.columns)}

    def compare(self, var, layer=None, bins=50, hist_range=None, density=True):
        """
        Save comparison plot of a variable between labels 0 and 1.
        """
        vals0, vals1 = self._select(var, layer=layer)

        plt.figure(figsize=(20, 20))
        if len(vals0) > 0:
            plt.hist(vals0, bins=bins, range=hist_range, density=density,
                     alpha=0.6, color='firebrick', label="Noise")
        if len(vals1) > 0:
            plt.hist(vals1, bins=bins, range=hist_range, density=density,
                     alpha=0.6, color='royalblue', label="Signal")

        title = f"{var}"
        if layer is not None:
            title += f" (Layer {layer})"
        plt.title(title)
        plt.xlabel(var)
        plt.ylabel("Normalized counts" if density else "Counts")
        plt.legend()
        plt.tight_layout()

        # save instead of show
        fname = os.path.join(self.printDir,
                             f"{var}{'_layer'+str(layer) if layer is not None else ''}{self.endName}.png")
        plt.savefig(fname, dpi=150)
        plt.close()

    def compare_all_layers(self, var, bins=50, hist_range=None, density=True, layer_scale=1):
        """
        Save comparison histograms per layer for a variable.
        Fully vectorized: concatenates data once and precomputes per-layer selections.
        """
        if self.x is None or self.y is None:
            raise ValueError("No data loaded")

        hits = np.vstack(self.x)
        orders = np.hstack(self.y)

        idx = self.col_index[var]
        vals = hits[:, idx]

        layers = np.unique(hits[:, self.col_index["layer"]])
        layer_masks = {layer: hits[:, self.col_index["layer"]] == layer for layer in layers}

        vals_by_layer = {}
        for layer in layers:
            mask = layer_masks[layer]
            vals0 = vals[mask & (orders == 0)]
            vals1 = vals[mask & (orders == 1)]
            vals_by_layer[layer] = (vals0, vals1)

        n_layers = len(layers)
        ncols = 3
        nrows = int(np.ceil(n_layers / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(20*ncols, 20*nrows))
        axes = axes.flatten()

        for i, layer in enumerate(layers):
            layerplot = int(layer * (layer_scale - 1) + 1) if layer_scale != 1 else int(layer)
            vals0, vals1 = vals_by_layer[layer]
            ax = axes[i]

            if len(vals0) > 0:
                ax.hist(vals0, bins=bins, range=hist_range, density=density,
                        color='firebrick', alpha=0.6, label="Noise")
            if len(vals1) > 0:
                ax.hist(vals1, bins=bins, range=hist_range, density=density,
                        color='royalblue', alpha=0.6, label="Signal")
            ax.set_title(f"{var} (Layer {layerplot})")
            ax.legend()

        for j in range(i+1, len(axes)):
            axes[j].axis("off")

        fig.tight_layout()

        fname = os.path.join(self.printDir, f"{var}_allLayers{self.endName}.png")
        plt.savefig(fname, dpi=150)
        plt.close()

    def plot_event_hits_on_strips(self, event_idx, max_strip_dict, preds=None, max_layer=99):
        """
        Plot hits for a single event arranged by strip around concentric circles for each layer.

        Parameters
        ----------
        event_idx : int
            Index of the event to plot.
        max_strip_dict : dict
            The maximum number of strips in each layer
        preds : array-like, optional
            Denoiser prediction, allows to remove hits
        max_layer : int
            Maximum layer to plot
        """
        
        # Get data for this specific event
        hits_event=self.x[event_idx]
        labels_event=self.y[event_idx]

        # print(hits_event.shape)
        # print(hits_event)

        endNamePred = ''
        # If predictions provided, filter hits
        if preds is not None:
            preds_event = preds[event_idx]
            keep_mask = preds_event == 1
            hits_event = hits_event[keep_mask]
            labels_event = labels_event[keep_mask]
            endNamePred = '_pastPred'

        if hits_event.shape[0] == 0:
            print(f"No hits found for event {event_idx} (after filtering with preds)" if preds is not None else f"No hits found for event {event_idx}")
            return

        layers = np.unique(hits_event[:, self.col_index["layer"]]).astype(int)
        layers = layers[layers < max_layer]

        plt.figure(figsize=(11, 10))
        ax = plt.gca()
        ax.set_aspect("equal")
        ax.axis("off")

        colors = {0: "black", 1: "royalblue"}  # Noise=red, Signal=blue
        labels_map = {0: "Noise", 1: "Signal"}
        handles = []

        max_radius = 0

        for layer in range(1, min(max_layer, 12) + 1):
            r = layer * 1.5
            max_radius = max(max_radius, r)

            # always draw layer circle
            circ = plt.Circle((0, 0), r, color="gray", fill=False,
                              linewidth=2, linestyle="--", alpha=0.4)
            ax.add_artist(circ)

            # plot hits only if there are any
            layer_mask = hits_event[:, self.col_index["layer"]] == layer
            strips = hits_event[layer_mask, self.col_index["strip"]]
            labs = labels_event[layer_mask]

            if len(strips) == 0:
                continue
              
            max_strip = max_strip_dict.get(layer, strips.max())
            strip_to_angle = {s: 2 * np.pi * (s - 1) / max_strip for s in strips}

            for strip, lab in zip(strips, labs):
                theta = strip_to_angle[strip]
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                sc = ax.scatter(
                    x, y,
                    color=colors.get(lab, "black"),
                    s=100, edgecolor="k",
                    label=labels_map.get(lab, "Other") if lab not in [h.get_label() for h in handles] else ""
                )
                if sc.get_label() not in [h.get_label() for h in handles] and sc.get_label() != "":
                    handles.append(sc)

        # ensure no cropping: add margin around outer circle + markers
        margin = 2.0
        ax.set_xlim(-max_radius - margin, max_radius + margin)
        ax.set_ylim(-max_radius - margin, max_radius + margin)

        plt.title(f"Event {event_idx}")
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(0.8, 0.95),
            borderaxespad=0.,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fname = os.path.join(self.printDir, f"event{event_idx}_hits_on_strips{endNamePred}{self.endName}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_event_hits_polar(self, event_idx, preds=None, max_layer=99):
        """
        Plot hits for a single event in two polar layouts using (x1, y1, z1):
        - Left: all layers except 7, 10, 12 (plotted by φ)
        - Right: only layers 7, 10, 12 (plotted by θ)

        Parameters
        ----------
        event_idx : int
            Index of the event to plot.
        preds : array-like, optional
            Denoiser prediction mask, allows removing hits.
        max_layer : int
            Maximum layer to plot.
        """

        # --- Get data for this event ---
        hits_event = self.x[event_idx]
        labels_event = self.y[event_idx]

        endNamePred = ''
        if preds is not None:
            preds_event = preds[event_idx]
            keep_mask = preds_event == 1
            hits_event = hits_event[keep_mask]
            labels_event = labels_event[keep_mask]
            endNamePred = '_pastPred'

        if hits_event.shape[0] == 0:
            print(f"No hits found for event {event_idx} (after filtering)" if preds is not None else f"No hits found for event {event_idx}")
            return

        # --- Extract coordinates ---
        X = hits_event[:, self.col_index["x1"]]
        Y = hits_event[:, self.col_index["y1"]]
        Z = hits_event[:, self.col_index["z1"]]
        layers = hits_event[:, self.col_index["layer"]].astype(int)

        # --- Compute spherical coordinates ---
        r3d = np.sqrt(X**2 + Y**2 + Z**2)
        theta = np.arccos(np.clip(Z / r3d, -1, 1))   # radians
        phi = np.arctan2(Y, X)                       # radians, range (-π, π)
        phi_deg = np.degrees(phi)
        phi_deg = (phi_deg + 180) % 360 - 180        # ensure [-180, 180)

        # --- Filter by max_layer ---
        mask = layers < max_layer
        X, Y, Z, phi, phi_deg, theta, layers, labels_event = (
            X[mask], Y[mask], Z[mask], phi[mask], phi_deg[mask],
            theta[mask], layers[mask], labels_event[mask]
        )

        # --- Separate layer groups ---
        special_layers = [7, 10, 12]
        mask_special = np.isin(layers, special_layers)
        mask_regular = ~mask_special

        # --- Setup figure and subplots ---
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        for ax in axes:
            ax.set_aspect("equal")
            ax.axis("off")

        ax_left, ax_right = axes

        # fig.suptitle(f"Event {event_idx}",y=0.93)

        colors = {0: "black", 1: "royalblue"}
        labels_map = {0: "Noise", 1: "Signal"}
        handles = []

        # --- Determine max radius for both plots ---
        max_radius = np.max(layers) * 1.5 if len(layers) else 10
        margin = 2.0

        # --- Function to draw concentric layer circles ---
        def draw_layers(ax, layer_list):
            for layer in layer_list:
                r_layer = layer * 1.5
                circle = plt.Circle((0, 0), r_layer, color="gray", fill=False,
                                    linestyle="--", linewidth=1.5, alpha=0.4)
                ax.add_artist(circle)

        # === LEFT PANEL: φ vs layer (all layers except 7,10,12) ===
        unique_regular_layers = np.unique(layers[mask_regular])
        draw_layers(ax_left, unique_regular_layers)

        for layer, phi_val, lab in zip(layers[mask_regular], phi_deg[mask_regular], labels_event[mask_regular]):
            r = layer * 1.5
            x = r * np.cos(np.radians(phi_val))
            y = r * np.sin(np.radians(phi_val))
            label_name = labels_map.get(lab, str(lab))
            first_time = label_name not in [h.get_label() for h in handles]
            sc = ax_left.scatter(
                x, y,
                s=100,
                color=colors.get(lab, "gray"),
                edgecolor="k",
                alpha=0.8,
                label=label_name if first_time else ""
            )
            if first_time:
                handles.append(sc)

        ax_left.set_xlim(-max_radius - margin, max_radius + margin)
        ax_left.set_ylim(-max_radius - margin, max_radius + margin)
        ax_left.set_title(r"Hits in X & Y ($\phi$)")
        ax_left.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(0.75, 0.9),
            borderaxespad=0.,
        )

        # === RIGHT PANEL: θ vs layer (only layers 7,10,12) ===
        unique_special_layers = np.unique(layers[mask_special])
        draw_layers(ax_right, unique_special_layers)

        for layer, theta_val, lab in zip(layers[mask_special], theta[mask_special], labels_event[mask_special]):
            r = layer * 1.5
            x = r * np.cos(theta_val)
            y = r * np.sin(theta_val)
            label_name = labels_map.get(lab, str(lab))
            first_time = label_name not in [h.get_label() for h in handles]
            sc = ax_right.scatter(
                x, y,
                s=100,
                color=colors.get(lab, "gray"),
                edgecolor="k",
                alpha=0.8,
                label=label_name if first_time else ""
            )
            if first_time:
                handles.append(sc)

        ax_right.set_xlim(-max_radius - margin, max_radius + margin)
        ax_right.set_ylim(-max_radius - margin, max_radius + margin)
        ax_right.set_title(r"Hits in Z ($\theta$)")
        # ax_right.legend(
        #     handles=handles,
        #     loc="center left",
        #     bbox_to_anchor=(0.8, 0.95),
        #     borderaxespad=0.,
        # )

        # Use subplots_adjust to reduce vertical and horizontal spacing
        fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.05, wspace=0.)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fname = os.path.join(self.printDir, f"event{event_idx}_hits_on_strips{endNamePred}{self.endName}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()

    def plotEventSize(self,y,mask):
        bg=[]
        sig=[]
        all=[]
        for sample_event in range(y.size(0)):
            sample_mask = mask[sample_event]
            n_valid = sample_mask.sum().item()
            all.append(n_valid)
            #print(f"Sample event {sample_event}: {n_valid} valid hits out of {mask.size(1)}")
            if n_valid > 0:
                sample_y = y[sample_event][sample_mask != 0]
                print(f"  Signal: {(sample_y == 1).sum().item()}, Noise: {(sample_y == 0).sum().item()}")
                bg.append((sample_y == 0).sum().item())
                sig.append((sample_y == 1).sum().item())

            plt.figure(figsize=(20,20))
            plt.hist(all, bins=100, alpha=0.6, label="All", color='black')
            plt.hist(bg, bins=100, alpha=0.6, label="Noise", color='firebrick')
            plt.hist(sig, bins=100, alpha=0.6, label="Signal", color='royalblue')
            plt.xlabel("# Hits per Event")
            plt.yscale('log')
            plt.title("# Hits per Event")
            plt.legend()
            fname = os.path.join(self.printDir, f"event_size{self.endName}.png")
            plt.savefig(fname, dpi=150)
            plt.close()

    def plotResp(self, all_probs, all_labels):
        if isinstance(all_probs, torch.Tensor):
            all_probs = all_probs.cpu().numpy()
        if isinstance(all_labels, torch.Tensor):
            all_labels = all_labels.cpu().numpy()
            
        signal_probs = all_probs[all_labels == 1]
        background_probs = all_probs[all_labels == 0]

        plt.figure(figsize=(20,20))
        plt.hist(background_probs, bins=100, alpha=0.6, range=(0,1), label="Noise", color='firebrick', density=True)
        plt.hist(signal_probs, bins=100, alpha=0.6, range=(0,1), label="Signal", color='royalblue', density=True)
        plt.xlabel("Response")
        plt.ylabel("Density")
        plt.yscale('log')
        plt.title("Response")
        plt.legend()
        fname = os.path.join(self.printDir, f"response{self.endName}.png")
        plt.savefig(fname, dpi=150)
        plt.close()

    def compare_all_layers_resp(self, all_probs, all_labels, hits=None, layer_scale=1):
        """
        Save comparison histograms per layer for response.
        """
        
        
        if isinstance(all_probs, torch.Tensor):
            all_probs = all_probs.cpu().numpy()
        if isinstance(all_labels, torch.Tensor):
            all_labels = all_labels.cpu().numpy()
        if hits!=None:
            if isinstance(hits, torch.Tensor):
                hits = hits.cpu().numpy()
        else:
            hits, orders, _ = self._get_concatenated_data()
            
        
        layers = np.unique(hits[:, self.col_index["layer"]])

        n_layers = len(layers)
        ncols = 3
        nrows = int(np.ceil(n_layers / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(20*ncols, 20*nrows))
        axes = axes.flatten()

        for i, layer in enumerate(layers):
            
            layerplot = layer
            if layer_scale != 1:
                layerplot = layerplot * (layer_scale - 1) + 1
            layerplot = int(layerplot)

            layer_mask = hits[:, self.col_index["layer"]] == layer
            probs = all_probs[layer_mask]
            labels = all_labels[layer_mask]
            signal_probs = probs[labels == 1]
            background_probs = probs[labels == 0]
            ax = axes[i]
            if len(background_probs) > 0:
                ax.hist(background_probs, bins=100, range=(0,1), density=True, color='firebrick', alpha=0.6, label="Noise")
            if len(signal_probs) > 0:
                ax.hist(signal_probs, bins=100, range=(0,1), density=True, color='royalblue', alpha=0.6, label="Signal")
            ax.set_yscale('log')
            ax.set_title(f"Response (Layer {layerplot})")
            ax.legend()

        # turn off any unused axes
        for j in range(i+1, len(axes)):
            axes[j].axis("off")

        fig.tight_layout()

        # save instead of show
        fname = os.path.join(self.printDir, f"response_allLayers{self.endName}.png")
        plt.savefig(fname, dpi=150)
        plt.close()

    def plot_efficiencies(self, all_probs, all_labels, n_points=100):
        if isinstance(all_probs, torch.Tensor):
            all_probs_t = all_probs
            all_labels_t = all_labels
        else:
            all_probs_t = torch.tensor(all_probs)
            all_labels_t = torch.tensor(all_labels)
            
        thresholds = np.linspace(0, 1, n_points)
        frac_signal_retained = []
        frac_noise_removed = []

        signal_mask = all_labels_t == 1
        noise_mask = all_labels_t == 0

        for thr in thresholds:
            preds = (all_probs_t >= thr).long()

            # Guard for division by zero if no signal/noise in batch
            if signal_mask.sum() > 0:
                frac_signal = (preds[signal_mask] == 1).float().mean().item()
            else:
                frac_signal = 0.0

            if noise_mask.sum() > 0:
                frac_noise = (preds[noise_mask] == 0).float().mean().item()
            else:
                frac_noise = 0.0

            frac_signal_retained.append(frac_signal)
            frac_noise_removed.append(frac_noise)

        # Plot
        plt.figure(figsize=(15,15))
        plt.axhline(0.9, color="grey", linestyle="--")
        plt.axhline(1.0, color="black", linestyle="--")
        plt.axhline(0.0, color="black", linestyle="--")
        plt.scatter(thresholds, frac_signal_retained, label="Signal Efficiency", color="royalblue", s=250)
        plt.scatter(thresholds, frac_noise_removed, label="Noise Rejection", color="firebrick", s=250)
        plt.ylim(0.6,1.1)
        plt.xlabel("Threshold on Response")
        plt.ylabel("Fraction")
        plt.title("Metrics")
        plt.legend()
        fname = os.path.join(self.printDir, f"eff_respTh{self.endName}.png")
        plt.savefig(fname, dpi=150)
        plt.close()