# src/data/dataset.py
import torch
from torch.utils.data import Dataset


class MicrobiomeDataset(Dataset):
    def __init__(self, abundances, tag, which_dataset=None, device=None):
        self.abundances = abundances.to(device)
        self.tag = tag.to(device)
        self.which_dataset = which_dataset.to(device) if which_dataset is not None else None
        self.device = device
        self.sample_ids = torch.arange(len(self.abundances), device=device)

    def __len__(self):
        return len(self.abundances)

    def __getitem__(self, idx):
        which_dataset = (
            self.which_dataset[idx] if self.which_dataset is not None
            else torch.tensor(-1, device=self.device)
        )
        abundance_vector = self.abundances[idx].float()
        # Create attention mask: True (1) where abundance is 0 (padding), False (0) where non-zero (valid)
        attn_mask = (abundance_vector == 0)  # Shape: (n_microbes,), dtype: bool
        return abundance_vector, self.tag[idx].float(), which_dataset, self.sample_ids[idx], attn_mask


class BacteriaDataset(Dataset):
    def __init__(self, existence_matrix, mask_prob=0.9):
        self.existence_matrix = existence_matrix  # May contain nan
        self.num_samples = existence_matrix.shape[0]
        self.mask_prob = mask_prob

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        existence_vector = self.existence_matrix[idx].clone()
        nan_mask = torch.isnan(existence_vector)  # True where nan
        # Replace nan with padding value (-1)
        existence_vector[nan_mask] = -1

        # Mask only non-nan positions
        indices_0 = (existence_vector == 0).nonzero(as_tuple=True)[0]
        indices_1 = (existence_vector == 1).nonzero(as_tuple=True)[0]
        num_bacteria = len(indices_0) + len(indices_1)
        num_mask = int(self.mask_prob * num_bacteria)
        num_mask_0 = num_mask // 2
        num_mask_1 = num_mask - num_mask_0

        mask_indices_0 = torch.randperm(len(indices_0))[:num_mask_0] if len(indices_0) > 0 else []
        mask_indices_1 = torch.randperm(len(indices_1))[:num_mask_1] if len(indices_1) > 0 else []
        mask_indices = torch.cat([indices_0[mask_indices_0], indices_1[mask_indices_1]])

        mask_vector = torch.zeros_like(existence_vector, dtype=torch.bool)
        mask_vector[mask_indices] = True
        existence_vector[mask_vector] = 2  # Masked positions

        # Attention mask: 0 for valid positions (0, 1, 2), 1 for NaN (-1)
        attn_mask = torch.zeros_like(existence_vector, dtype=torch.bool)
        attn_mask[nan_mask] = 1 # True means ignore (NaN positions)

        return {
            'existence_vector': existence_vector.int(),  # 0, 1, 2, -1 (padded)
            'y_true': self.existence_matrix[idx],  # For loss computation- original values 1, 0, nan
            'mask_vector': mask_vector,  # For loss computation- 1 for masked, 0 for non-masked
            'attn_mask': attn_mask  # For transformer- 1 for padding, 0 for valid
        }

