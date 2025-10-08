import hipopy as hippy
import awkward as ak
import numpy as np
import torch
import random
import os


class HipoParser:
    """
    Parser for CLAS12 CVT ML hits HIPO files.
    Supports reading, saving, scaling/unscaling, splitting per sector,
    and padding into fixed-size tensors (x, y, mask).
    """

    # Sector mapping for 3 output files
    sector_mapping = {
        1: {  # File 1
            (1, 2): [1, 2, 3, 4],
            (3, 4): [1, 2, 3, 4, 5, 6],
            (5, 6): [1, 2, 3, 4, 5, 6, 7],
            (7, 12): [1],
        },
        2: {  # File 2
            (1, 2): [4, 5, 6, 7, 8],
            (3, 4): [6, 7, 8, 9, 10],
            (5, 6): [7, 8, 9, 10, 11, 12, 13],
            (7, 12): [2],
        },
        3: {  # File 3
            (1, 2): [8, 9, 10],
            (3, 4): [10, 11, 12, 13, 14],
            (5, 6): [14, 15, 16, 17, 18],
            (7, 12): [3],
        },
    }

    def __init__(self, filename, bank_name="CVT::MLHit", max_events=-1):
        """
        Parameters
        ----------
        filename : str
            Path to the HIPO file.
        bank_name : str
            Which bank to read ("CVTRec::MLHit" or "CVT::MLHit").
        """
        self.filename = filename
        self.bank_name = bank_name
        self.max_events = max_events
        self.file = hippy.open(filename, mode="r")
        self.file.readBank(bank_name)

    def read_all(self):
        """
        Read all events from the HIPO file.
        Returns lists of hits arrays and order arrays.
        """
        hits_list = []
        orders_list = []
        n_evs = 0
        max_hits_p_ev = 0
        if self.max_events == -1:
            n_evs = -2  # Read all events

        for batch in hippy.iterate([self.filename], [self.bank_name], step=1):
            if n_evs >= self.max_events:
                break
            if (len(hits_list) % 1000) == 0 and len(hits_list) != 0:
                print(f"Read {len(hits_list)} events so far...")

            # Extract columns
            strip = np.array(ak.Array(batch[self.bank_name + "_strip"]))
            layer = np.array(ak.Array(batch[self.bank_name + "_layer"]))
            sector = np.array(ak.Array(batch[self.bank_name + "_sector"]))
            order = np.array(ak.Array(batch[self.bank_name + "_order"]))
            x1 = np.array(ak.Array(batch[self.bank_name + "_x1"]))
            y1 = np.array(ak.Array(batch[self.bank_name + "_y1"]))
            z1 = np.array(ak.Array(batch[self.bank_name + "_z1"]))
            x2 = np.array(ak.Array(batch[self.bank_name + "_x2"]))
            y2 = np.array(ak.Array(batch[self.bank_name + "_y2"]))
            z2 = np.array(ak.Array(batch[self.bank_name + "_z2"]))
            cweight = np.array(ak.Array(batch[self.bank_name + "_cweight"]))
            sweight = np.array(ak.Array(batch[self.bank_name + "_sweight"]))

            # Filter orders 0 or 1&11
            mask = np.isin(order, [0, 1, 11])
            if np.sum(mask) == 0:
                continue

            hits = np.vstack(
                [
                    strip[mask],
                    layer[mask],
                    sector[mask],
                    x1[mask],
                    y1[mask],
                    z1[mask],
                    x2[mask],
                    y2[mask],
                    z2[mask],
                    cweight[mask],
                    sweight[mask],
                ]
            ).T

            orders = np.where(order[mask] == 0, 1, 0)  # map 0->1, 1&11->0

            hits_list.append(hits)
            orders_list.append(orders)

            if self.max_events != -1:
                n_evs += 1
            if hits.shape[0] > max_hits_p_ev:
                max_hits_p_ev = hits.shape[0]

        print(f"Max hits in an event: {max_hits_p_ev}")
        return hits_list, orders_list

    def split_by_output_file(self, hits_list, orders_list, output_file_idx):
        """
        Split hits/labels into 3 output files using sector_mapping.
        Returns new lists of (hits, orders).
        """
        mapping = self.sector_mapping[output_file_idx]
        split_hits, split_orders = [], []

        for hits, orders in zip(hits_list, orders_list):
            mask = np.zeros(hits.shape[0], dtype=bool)
            for layers, sectors in mapping.items():
                lmin, lmax = layers
                layer_mask = (hits[:, 1] >= lmin) & (hits[:, 1] <= lmax)
                sector_mask = np.isin(hits[:, 2], sectors)
                mask |= (layer_mask & sector_mask)
            if np.sum(mask) == 0:
                continue
            split_hits.append(hits[mask])
            split_orders.append(orders[mask])

        return split_hits, split_orders

    def trim_and_scale_batch(self, batch, selected_vars, min_vals, max_vals):
        """
        Trim to selected variables and scale to [0,1] based on layer-dependent ranges.
        """
        hits_list, orders_list = batch
        
        # Column name to index mapping (based on read_all order)
        col_map = {
            "strip": 0, "layer": 1, "sector": 2,
            "x1": 3, "y1": 4, "z1": 5,
            "x2": 6, "y2": 7, "z2": 8,
            "cweight": 9, "sweight": 10
        }
        
        # Get indices for selected vars
        selected_indices = [col_map[v] for v in selected_vars]
        
        scaled_hits = []
        for hits in hits_list:
            # Select columns
            trimmed = hits[:, selected_indices]
            scaled = trimmed.copy().astype(np.float32)
            
            # Get layer index in the trimmed array
            layer_idx_in_selected = selected_vars.index("layer") if "layer" in selected_vars else None
            
            # Scale each variable
            for i, var in enumerate(selected_vars):
                if var in ["strip", "x1", "x2", "y1", "y2", "z1", "z2", "sector"]:
                    # Layer-dependent scaling
                    if layer_idx_in_selected is not None:
                        for hit_idx in range(trimmed.shape[0]):
                            layer = int(trimmed[hit_idx, layer_idx_in_selected])
                            min_val = min_vals[var][layer]
                            max_val = max_vals[var][layer]
                            if max_val > min_val:
                                scaled[hit_idx, i] = (trimmed[hit_idx, i] - min_val) / (max_val - min_val)
                elif var == "layer":
                    # Global scaling for layer
                    min_val = min_vals[var]
                    max_val = max_vals[var]
                    if max_val > min_val:
                        scaled[:, i] = (trimmed[:, i] - min_val) / (max_val - min_val)
                # cweight and sweight don't need scaling (already normalized)
            
            scaled_hits.append(scaled)
        
        return scaled_hits, orders_list
    
    def unscale_batch(self, batch, selected_vars, min_vals, max_vals, layer_scale=1):
        """
        Trim to selected variables and scale to [0,1] based on layer-dependent ranges.
        """
        hits_list, orders_list = batch
        hits_list= [t.cpu().numpy() for t in hits_list] 
        
        unscaled_hits = []
        for hits in hits_list:
            unscaled = hits.copy().astype(np.float32)
            
            # Get layer index in the trimmed array
            layer_idx_in_selected = selected_vars.index("layer") if "layer" in selected_vars else None
            
            # Scale each variable
            for i, var in enumerate(selected_vars):
                if var in ["strip", "x1", "x2", "y1", "y2", "z1", "z2", "sector"]:
                    # Layer-dependent scaling
                    if layer_idx_in_selected is not None:
                        for hit_idx in range(unscaled.shape[0]):
                            layer = unscaled[hit_idx, layer_idx_in_selected]
                            layer = int(layer * (layer_scale - 1) + 1) if layer_scale != 1 else int(layer)
                            min_val = min_vals[var][layer]
                            max_val = max_vals[var][layer]
                            if max_val > min_val:
                                unscaled[hit_idx, i] = (unscaled[hit_idx, i] * (max_val - min_val)) + min_val
                elif var == "layer":
                    # Global scaling for layer
                    min_val = min_vals[var]
                    max_val = max_vals[var]
                    if max_val > min_val:
                        unscaled[:, i] = (unscaled[:, i] * (max_val - min_val)) + min_val
                # cweight and sweight don't need scaling (already normalized)
            
            unscaled_hits.append(unscaled)
        
        return unscaled_hits
    
    def unscale_x(self, hits_list, selected_vars, min_vals, max_vals, layer_scale=1):
        
        col_index = {name: i for i, name in enumerate(selected_vars)}
        layer_idx = col_index.get("layer", None)
        
        unscaled_hits = []
        for hits in hits_list:
            unscaled = hits.copy().astype(np.float32)
            
            # Scale each variable
            for i, var in enumerate(selected_vars):
                
                col = unscaled[:, i]
                min_val = min_vals.get(var, None)
                max_val = max_vals.get(var, None)
                layers = unscaled[:, layer_idx]
                if layer_scale != 1:
                  layers = layers * (layer_scale - 1) + 1
                layers=layers.astype(np.int64)

                if min_val is None or max_val is None:
                    continue  # skip if no min/max info
                # Per-layer scaling
                if isinstance(min_val, dict) and isinstance(max_val, dict) and layers is not None:
                    for layer in np.unique(layers):
                        mask = layers == layer
                        min_layer = min_val.get(int(layer.item()), None)
                        max_layer = max_val.get(int(layer.item()), None)
                        if min_layer is None or max_layer is None or max_layer == min_layer:
                            continue  # can't unscale
                        col[mask] = col[mask] * (max_layer - min_layer) + min_layer
                else:
                    if max_val == min_val:
                        continue
                    col[:] = col * (max_val - min_val) + min_val
                unscaled[:, i] = col
            
            unscaled_hits.append(unscaled)
        
        return unscaled_hits

    def pad_events(self, hits_list, orders_list, max_hits=450):
        """
        Pad hits and orders to fixed size [max_hits, n_features].
        Returns tensors (x, y, mask).
        """
        x_list, y_list, mask_list = [], [], []
        n_features = hits_list[0].shape[1] if hits_list else 0

        for hits, orders in zip(hits_list, orders_list):
            n = hits.shape[0]
            # pad hits
            padded_hits = np.zeros((max_hits, n_features), dtype=np.float32)
            padded_hits[: min(n, max_hits)] = hits[:max_hits]

            # pad orders
            padded_orders = np.zeros((max_hits,), dtype=np.int64)
            padded_orders[: min(n, max_hits)] = orders[:max_hits]

            # mask: 1=real hit, 0=padding
            mask = np.zeros((max_hits,), dtype=np.float32)
            mask[: min(n, max_hits)] = 1.0

            x_list.append(torch.tensor(padded_hits, dtype=torch.float32))
            y_list.append(torch.tensor(padded_orders, dtype=torch.long))
            mask_list.append(torch.tensor(mask, dtype=torch.float32))

        return torch.stack(x_list), torch.stack(y_list), torch.stack(mask_list)

    def split_dataset(self, x, y, mask, train_frac=0.7, seed=None):
        """
        Split (x, y, mask) tensors into train and test sets.
        """
        n = x.size(0)
        indices = list(range(n))
        if seed is not None:
            random.seed(seed)
        random.shuffle(indices)
        split_idx = int(n * train_frac)
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        return x[train_idx], y[train_idx], mask[train_idx], x[test_idx], y[test_idx], mask[test_idx]

    def save_dataset(self, out_path, x, y, mask):
        torch.save({"x": x, "y": y, "mask": mask}, out_path)

    @staticmethod
    def load_dataset(path):
        return torch.load(path, weights_only=False)
    
    def save_example_text(out_path, x, y, mask):
        """
        Save the first row of x, y, mask to a text file.
        """
        with open(out_path, "w") as f:
            f.write("x:\n")
            for row in x[0].cpu().numpy():
                f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

            f.write("\ny:\n")
            for val in y[0].cpu().numpy():
                f.write(f"{val}\n")

            f.write("\nmask:\n")
            for val in mask[0].cpu().numpy():
                f.write(f"{val}\n")

    def print_dataset_info(self, x, y, mask):
        n_events = x.size(0)
        n_features = x.size(2)
        print(f"Events: {n_events}")
        print(f"Input features per hit: {n_features}")
        print(f"Output labels per hit: 1")