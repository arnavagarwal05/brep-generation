# SolidGen PyTorch Implementation - Training Pipeline
import os
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for headless environments
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
import json
from PIL import Image # For loading images
from torchvision import transforms # For image preprocessing
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm # For progress bars during training epochs
import time # For timing training epochs
import math
import random
import traceback 
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime ### For creating unique run directories
import traceback
from models import ImageEncoder, VertexModel, EdgeModel, FaceModel # Import the neural network models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR


import matplotlib.pyplot as plt
# Ensure reproducible results
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42) # Set a fixed seed for reproducibility


# Max sequence lengths derived from data curation filtering (max 400 content tokens)
# +1 for SOS token, so total sequence length is max_content_tokens + 1
MAX_SEQ_LEN_V = 401 # Max sequence length for VertexModel (approx. 133 vertices * 3 coords + SOS)
MAX_SEQ_LEN_E = 401 # Max sequence length for EdgeModel
MAX_SEQ_LEN_F = 401 # Max sequence length for FaceModel

# Token values (consistent with data curation script's quantization 0-63)
VERTEX_EOS_TOKEN = 64 # Special token for End-of-Sequence in vertex model
VERTEX_SOS_TOKEN = 65 # Special token for Start-of-Sequence in vertex model (unused coordinate index)

# Global maximums for model capacity. These influence embedding sizes.
# Chosen to be larger than any model found in curation to provide a buffer.
GLOBAL_MAX_VERTICES_MODEL = 150 # Corresponds to ~450 vertex tokens + SOS/EOS (fits MAX_SEQ_LEN_V)
GLOBAL_MAX_EDGES_MODEL = 150    # Corresponds to ~450 edge tokens + SOS/EOS (fits MAX_SEQ_LEN_E)

# =============================================================================
# FIXED TOKEN DEFINITIONS - These are now CONSTANT across all samples/batches
# =============================================================================
# Edge tokens (FIXED positions regardless of actual vertex count)
EDGE_NEW_EDGE_TOKEN = GLOBAL_MAX_VERTICES_MODEL      # 150
EDGE_EOS_TOKEN = GLOBAL_MAX_VERTICES_MODEL + 1       # 151
EDGE_SOS_TOKEN = GLOBAL_MAX_VERTICES_MODEL + 2       # 152

# Face tokens (FIXED positions regardless of actual edge count)
FACE_NEW_FACE_TOKEN = GLOBAL_MAX_EDGES_MODEL         # 150
FACE_EOS_TOKEN = GLOBAL_MAX_EDGES_MODEL + 1          # 151
FACE_SOS_TOKEN = GLOBAL_MAX_EDGES_MODEL + 2          # 152

# Padding tokens. Must be unique and outside the range of any valid token or special token.
# Vertices: valid [0-63], EOS (64), SOS (65). Next safe is 66.
PAD_TOKEN_V = VERTEX_SOS_TOKEN + 1 # 66

# Edges: valid [0-V_actual-1], NEW_EDGE (V_actual), EOS (V_actual+1), SOS (V_actual+2).
# Max token for EdgeModel is SOS (GLOBAL_MAX_VERTICES_MODEL + 2). Next safe is GLOBAL_MAX_VERTICES_MODEL + 3.
PAD_TOKEN_E = GLOBAL_MAX_VERTICES_MODEL + 3

# Faces: valid [0-E_actual-1], NEW_FACE (E_actual), EOS (E_actual+1), SOS (E_actual+2).
# Max token for FaceModel is SOS (GLOBAL_MAX_EDGES_MODEL + 2). Next safe is GLOBAL_MAX_EDGES_MODEL + 3.
PAD_TOKEN_F = GLOBAL_MAX_EDGES_MODEL + 3


# ========================
# DATA PROCESSING / TOKENIZATION
# ========================
def tokenize_vertices(vertices_raw_xyz):
    """
    Converts raw (x,y,z) float vertices to token sequence (paper Section 4.1).
    Args:
        vertices_raw_xyz (list of tuples/np.array): List of (x,y,z) float coordinates.
    Returns:
        v_seq (torch.LongTensor): (SOS, z1, y1, x1, z2, y2, x2, ..., EOS)
        v_q_original_order_xyz (torch.LongTensor): (N, 3) (X,Y,Z) quantized coords, for Edge/Face models, in original order.
        V_min (np.array): Original min (X,Y,Z) floats for bounding box.
        V_max (np.array): Original max (X,Y,Z) floats for bounding box.
    """
    V = np.array(vertices_raw_xyz, dtype=np.float32)
    
    if len(V) == 0: # Handle empty vertex list gracefully
        return torch.tensor([VERTEX_SOS_TOKEN, VERTEX_EOS_TOKEN], dtype=torch.long), \
               torch.zeros((0, 3), dtype=torch.long), \
               np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    # Calculate bounding box (V_min, V_max) using ALL vertices first.
    V_min = V.min(axis=0) # (3,) numpy array (min X, min Y, min Z)
    V_max = V.max(axis=0) # (3,) numpy array (max X, max Y, max Z)
    
    # Handle cases where a dimension has zero range (e.g., flat models), add epsilon for division.
    V_range = V_max - V_min
    V_range[V_range == 0] = 1e-6 

    # Quantize vertices in their *original* order for `v_q_original_order_xyz` (used by Edge/Face models)
    V_original_norm = (V - V_min) / V_range
    V_q_original_order_xyz = np.clip(np.round(V_original_norm * 63), 0, 63).astype(np.int64) # (N, 3) quantized (X, Y, Z)

    # Sort vertices lexicographically (Z, Y, X order) for the `v_seq` token sequence.
    # np.lexsort((col_x, col_y, col_z)) sorts by col_z (primary), then col_y, then col_x.
    # So for ZYX sorting, provide (V[:,0], V[:,1], V[:,2]) as arguments (X is least significant).
    # This means sorting will be primarily by Z, then by Y, then by X.
    sort_idx_for_zyx_seq = np.lexsort((V[:, 0], V[:, 1], V[:, 2]))
    V_sorted_zyx_order = V[sort_idx_for_zyx_seq] # Sorted by Z, then Y, then X.

    # Quantize the sorted vertices for `v_seq`.
    V_sorted_norm = (V_sorted_zyx_order - V_min) / V_range
    V_q_sorted_xyz = np.clip(np.round(V_sorted_norm * 63), 0, 63).astype(np.int64) # (N, 3) quantized (X, Y, Z)

    # Reorder columns to Z, Y, X for the token sequence (as per SolidGen paper A.3)
    V_q_reordered_zyx_tokens = V_q_sorted_xyz[:, [2, 1, 0]] # Now it's [Z, Y, X] order
    
    # Flatten and add SOS and EOS tokens.
    V_seq_list = [VERTEX_SOS_TOKEN] + V_q_reordered_zyx_tokens.flatten().tolist() + [VERTEX_EOS_TOKEN]
    
    return torch.tensor(V_seq_list, dtype=torch.long), \
           torch.from_numpy(V_q_original_order_xyz).to(torch.long), \
           V_min, V_max

def tokenize_edges(edges_orig_tuples, num_vertices_actual):
    """
    Converts original edge tuples to token sequence (paper Section 4.2).
    FIXED VERSION: Uses GLOBAL_MAX_VERTICES_MODEL for special token positions.
    This ensures tokens have consistent meaning across all samples and batches.
    
    Args:
        edges_orig_tuples (list of tuples): List of (u,v) or (u,v,w) vertex index tuples.
        num_vertices_actual (int): Actual number of vertices in the current model.
    Returns:
        e_seq (torch.LongTensor): (SOS, u1, v1, NEW_EDGE, u2, v2, v3, NEW_EDGE, ..., EOS)
        edge_indices_padded_for_face_model (torch.LongTensor): (E, 3) padded edge vertex index tuples for FaceModel.
    """
    NEW_EDGE_VAL = EDGE_NEW_EDGE_TOKEN  # 150 (always)
    EOS_VAL = EDGE_EOS_TOKEN            # 151 (always)
    SOS_VAL = EDGE_SOS_TOKEN            # 152 (always)
    
    # Sort edges for ordering invariance (lexicographical sorting of (u,v) or (u,v,w) tuples)
    sorted_edges_for_sequence = []
    for e_tuple in edges_orig_tuples:
        if len(e_tuple) == 2:
            sorted_edges_for_sequence.append(tuple(sorted(e_tuple))) # (u,v) where u<=v
        elif len(e_tuple) == 3:
            sorted_edges_for_sequence.append(tuple(sorted(e_tuple))) # (u,v,w) where u<=v<=w
        # Else: invalid edge format, skip (should be caught by curation)
    
    # Sort the list of tuples (edges) globally based on their elements.
    # The key handles tuples of different lengths by adding a dummy -1.
    sorted_edges_for_sequence.sort(key=lambda x: (x[0], x[1], x[2] if len(x) == 3 else -1))
    
    # Create the token sequence (`e_seq`)
    tokens_list = [SOS_VAL] # Start with SOS token
    for e_tuple in sorted_edges_for_sequence:
        for idx in e_tuple:
            tokens_list.append(idx) # Append vertex indices
        tokens_list.append(NEW_EDGE_VAL) # Append NEW_EDGE token after each edge
    tokens_list.append(EOS_VAL) # Append EOS token at the end
    
    # `edge_indices_padded_for_face_model`: This is the list of original edge tuples,
    # padded to a fixed size (max 3 vertices/edge) for batching. This is passed to FaceModel.
    max_verts_per_edge = 3 # Edges are either (u,v) or (u,v,w)
    edge_indices_padded_for_face_model = []
    for e_tuple in edges_orig_tuples: # Use original `edges_orig_tuples` order for this list
        padded_e = list(e_tuple) + [-1] * (max_verts_per_edge - len(e_tuple)) # Pad with -1
        edge_indices_padded_for_face_model.append(padded_e)
    
    return torch.tensor(tokens_list, dtype=torch.long), \
           torch.tensor(edge_indices_padded_for_face_model, dtype=torch.long)

def tokenize_faces(faces_orig_lists, num_edges_actual):
    """
    Converts original face lists to token sequence (paper Section 4.3).
    
    FIXED VERSION: Uses GLOBAL_MAX_EDGES_MODEL for special token positions.
    This ensures tokens have consistent meaning across all samples and batches.
    
    Args:
        faces_orig_lists (list of lists): List of lists of edge indices defining faces.
        num_edges_actual (int): Actual number of edges in the current model (for validation only).
    Returns:
        f_seq (torch.LongTensor): (SOS, e1, e2, NEW_FACE, e3, e4, ..., EOS)
    """
    # FIXED: Use global constants instead of dynamic values
    NEW_FACE_VAL = FACE_NEW_FACE_TOKEN  # 150 (always)
    EOS_VAL = FACE_EOS_TOKEN            # 151 (always)
    SOS_VAL = FACE_SOS_TOKEN            # 152 (always)
    
    # Sort faces for ordering invariance (lexicographical sorting of edge index lists)
    sorted_faces_for_sequence = [sorted(f_list) for f_list in faces_orig_lists] # Sort edge indices within each face
    sorted_faces_for_sequence.sort(key=lambda x: tuple(x)) # Sort the list of faces themselves globally
    
    tokens_list = [SOS_VAL] # Start with SOS token
    for face_edge_indices in sorted_faces_for_sequence:
        tokens_list.extend(face_edge_indices) # Append edge indices
        tokens_list.append(NEW_FACE_VAL) # Append NEW_FACE token after each face
    tokens_list.append(EOS_VAL) # Append EOS token at the end
    
    return torch.tensor(tokens_list, dtype=torch.long)

# ========================
# DATASET AND DATALOADER - UNCONDITIONAL VERSION
# ========================
class SolidGenDatasetUnconditional(Dataset):
    """
    Dataset for UNCONDITIONAL training - does NOT load images.
    This saves memory and I/O time since images aren't used.
    """
    def __init__(self, data_root, split):
        self.data_root = data_root
        self.split = split
        self.file_ids = []

        split_file_path = os.path.join(data_root, 'train_val_test_split.json')
        if not os.path.exists(split_file_path):
            raise FileNotFoundError(f"Split file not found at {split_file_path}")
        with open(split_file_path, 'r') as f:
            splits_data = json.load(f)

        curated_split_base_dir = os.path.join(data_root, 'processed_dataset', split)
        indexed_brep_files_dir = os.path.join(curated_split_base_dir, 'indexed_brep')

        if not os.path.isdir(indexed_brep_files_dir):
            raise FileNotFoundError(f"Curated indexed_brep directory not found: {indexed_brep_files_dir}.")

        found_curated_base_ids = set()
        for root, dirs, files in os.walk(indexed_brep_files_dir):
            for file in files:
                if file.endswith('_solidgen.json'):
                    base_id = file.replace('_solidgen.json', '')
                    found_curated_base_ids.add(base_id)

        for file_id_with_subfolder in splits_data.get(split, []):
            base_filename = file_id_with_subfolder.split('/')[-1]
            if base_filename in found_curated_base_ids:
                self.file_ids.append(file_id_with_subfolder)
        
        if not self.file_ids:
            print(f"WARNING: No curated models found for split '{split}'.", file=sys.stderr)

        self.indexed_brep_base_dir = os.path.join(curated_split_base_dir, 'indexed_brep')

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        subfolder = file_id.split('/')[0]
        base_filename = file_id.split('/')[-1]

        try:
            # Load indexed B-rep JSON only (NO IMAGE)
            json_path = os.path.join(self.indexed_brep_base_dir, subfolder, f"{base_filename}_solidgen.json")
            with open(json_path, 'r') as f:
                indexed_brep_data = json.load(f)

            v_seq, v_q_xyz, V_min_orig, V_max_orig = tokenize_vertices(indexed_brep_data['vertices'])
            e_seq, edge_indices_for_face_model = tokenize_edges(indexed_brep_data['edges'], v_q_xyz.shape[0])
            f_seq = tokenize_faces(indexed_brep_data['faces'], len(indexed_brep_data['edges']))

            return {
                'v_seq': v_seq,
                'e_seq': e_seq,
                'f_seq': f_seq,
                'v_q_xyz': v_q_xyz,
                'edge_indices_for_face_model': edge_indices_for_face_model,
                'V_min_orig': torch.tensor(V_min_orig, dtype=torch.float32),
                'V_max_orig': torch.tensor(V_max_orig, dtype=torch.float32),
                'file_id': file_id
            }
        except Exception as e:
            print(f"ERROR(Dataset): Skipping {file_id}: {e}", file=sys.stderr)
            return None


def collate_fn_unconditional(batch):
    """
    Custom collate function for UNCONDITIONAL training - no images.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        print("WARN(Collate): Empty batch after filtering.", file=sys.stderr)
        return None

    max_v_len = max(item['v_seq'].size(0) for item in batch)
    max_e_len = max(item['e_seq'].size(0) for item in batch)
    max_f_len = max(item['f_seq'].size(0) for item in batch)
    
    max_V_batch = max(item['v_q_xyz'].size(0) for item in batch)
    max_E_batch = max(item['edge_indices_for_face_model'].size(0) for item in batch)
    
    padded_v_seq = torch.stack([
        F.pad(item['v_seq'], (0, max_v_len - item['v_seq'].size(0)), value=PAD_TOKEN_V)
        for item in batch
    ])
    padded_e_seq = torch.stack([
        F.pad(item['e_seq'], (0, max_e_len - item['e_seq'].size(0)), value=PAD_TOKEN_E)
        for item in batch
    ])
    padded_f_seq = torch.stack([
        F.pad(item['f_seq'], (0, max_f_len - item['f_seq'].size(0)), value=PAD_TOKEN_F)
        for item in batch
    ])

    padded_v_q_xyz = torch.stack([
         F.pad(item['v_q_xyz'], (0, 0, 0, max_V_batch - item['v_q_xyz'].size(0)), value=0)
         for item in batch
    ])
    
    padded_edge_indices_for_face_model = torch.stack([
        F.pad(item['edge_indices_for_face_model'], (0, 0, 0, max_E_batch - item['edge_indices_for_face_model'].size(0)), value=-1)
        for item in batch
    ])

    return {
        'v_seq': padded_v_seq,
        'e_seq': padded_e_seq,
        'f_seq': padded_f_seq,
        'v_q_xyz': padded_v_q_xyz,
        'edge_indices_for_face_model': padded_edge_indices_for_face_model,
        'V_min_orig': torch.stack([item['V_min_orig'] for item in batch]),
        'V_max_orig': torch.stack([item['V_max_orig'] for item in batch]),
        'file_id': [item['file_id'] for item in batch]
    }

# ========================
# DATASET AND DATALOADER
# ========================
class SolidGenDataset(Dataset):
    def __init__(self, data_root, split, transform=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.file_ids = [] # Stores file_id strings like '0067/00675619'

        # Load the overall train/val/test split file
        split_file_path = os.path.join(data_root, 'train_val_test_split.json')
        if not os.path.exists(split_file_path):
            raise FileNotFoundError(f"Split file not found at {split_file_path}")
        with open(split_file_path, 'r') as f:
            splits_data = json.load(f)

        # Get the base directory for the curated data for this split (e.g., 'data/processed_curated/train')
        curated_split_base_dir = os.path.join(data_root, 'processed_dataset', split)
        indexed_brep_files_dir = os.path.join(curated_split_base_dir, 'indexed_brep')

        if not os.path.isdir(indexed_brep_files_dir):
            raise FileNotFoundError(f"Curated indexed_brep directory not found: {indexed_brep_files_dir}. "
                                    f"Please ensure data curation pipeline completed successfully for split '{split}'.")

        # Recursively find all `_solidgen.json` files in the curated directory and extract their base IDs
        found_curated_base_ids = set()
        for root, dirs, files in os.walk(indexed_brep_files_dir):
            for file in files:
                if file.endswith('_solidgen.json'):
                    base_id = file.replace('_solidgen.json', '') # Extract '00675619' from '00675619_solidgen.json'
                    found_curated_base_ids.add(base_id)

        # Filter the original split file_ids to only include those present in the *curated* output.
        # This ensures we only load models that passed all curation filters.
        for file_id_with_subfolder in splits_data.get(split, []):
            base_filename = file_id_with_subfolder.split('/')[-1] # e.g., '00675619'
            if base_filename in found_curated_base_ids:
                self.file_ids.append(file_id_with_subfolder)
        
        if not self.file_ids:
            print(f"WARNING: No curated models found for split '{split}' in {indexed_brep_files_dir}. "
                  f"This split will be empty. Check your data curation process or paths.", file=sys.stderr)

        self.image_base_dir = os.path.join(curated_split_base_dir, 'image')
        self.indexed_brep_base_dir = os.path.join(curated_split_base_dir, 'indexed_brep')

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx] # e.g., '0067/00675619'
        subfolder = file_id.split('/')[0] # e.g., '0067' (the first 4 chars)
        base_filename = file_id.split('/')[-1] # e.g., '00675619' (the full ID)

        try:
            # Load image (.png)
            img_path = os.path.join(self.image_base_dir, subfolder, f"{base_filename}.png")
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            # Load indexed B-rep JSON (`_solidgen.json`)
            json_path = os.path.join(self.indexed_brep_base_dir, subfolder, f"{base_filename}_solidgen.json")
            with open(json_path, 'r') as f:
                indexed_brep_data = json.load(f)

            # Tokenize ground truth data from JSON for model input and target
            # tokenize_vertices returns: v_seq, v_q_xyz, V_min_orig, V_max_orig
            v_seq, v_q_xyz, V_min_orig, V_max_orig = tokenize_vertices(indexed_brep_data['vertices'])
            
            # tokenize_edges returns: e_seq, edge_indices_for_face_model
            # Pass v_q_xyz.shape[0] for actual number of vertices (V_actual)
            e_seq, edge_indices_for_face_model = tokenize_edges(indexed_brep_data['edges'], v_q_xyz.shape[0])
            
            # tokenize_faces returns: f_seq
            # Pass len(indexed_brep_data['edges']) for actual number of edges (E_actual)
            f_seq = tokenize_faces(indexed_brep_data['faces'], len(indexed_brep_data['edges']))

            return {
                'image': image,
                'v_seq': v_seq, # For VertexModel target
                'e_seq': e_seq, # For EdgeModel target
                'f_seq': f_seq, # For FaceModel target
                'v_q_xyz': v_q_xyz, # Quantized (X,Y,Z) vertices for Edge/Face models input memory
                'edge_indices_for_face_model': edge_indices_for_face_model, # Original edge structures for FaceModel input memory
                'V_min_orig': torch.tensor(V_min_orig, dtype=torch.float32), # For dequantization during eval
                'V_max_orig': torch.tensor(V_max_orig, dtype=torch.float32), # For dequantization during eval
                'file_id': file_id # For potential debugging/logging
            }
        except (FileNotFoundError, json.JSONDecodeError, ValueError, Exception) as e:
            # Catch data loading or initial tokenization errors for individual samples.
            # This allows the DataLoader to continue. This sample will be filtered by collate_fn.
            print(f"ERROR(Dataset): Skipping {file_id} due to data loading/tokenization error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return None # Return None for problematic samples

def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable-length sequences.
    Pads sequences with specific PAD_TOKENs and pads dynamic-sized tensors.
    NOTE: With fixed global tokenization, e_seq and f_seq tokens are already consistent
    across all samples. We just need to pad sequences to the same length within a batch.
    """
    # Filter out any None samples that might have been returned by __getitem__ due to errors.
    batch = [item for item in batch if item is not None]
    if not batch: # If the entire batch became empty after filtering
        print("WARN(Collate): Empty batch after filtering problematic samples.", file=sys.stderr)
        return None # Signal DataLoader to skip this batch

    # Determine max lengths in current batch for padding sequences.
    max_v_len = max(item['v_seq'].size(0) for item in batch)
    max_e_len = max(item['e_seq'].size(0) for item in batch)
    max_f_len = max(item['f_seq'].size(0) for item in batch)
    
    # Determine max number of vertices and edges in current batch for padding dynamic tensors.
    # Check if a sample has 0 vertices/edges (edge case, but possible with very strict curation).
    max_V_batch = max(item['v_q_xyz'].size(0) for item in batch)
    max_E_batch = max(item['edge_indices_for_face_model'].size(0) for item in batch)
    
    # Pad sequences (`v_seq`, `e_seq`, `f_seq`) with their respective PAD_TOKENs.
    padded_v_seq = torch.stack([
        F.pad(item['v_seq'], (0, max_v_len - item['v_seq'].size(0)), value=PAD_TOKEN_V)
        for item in batch
    ])
    padded_e_seq = torch.stack([
        F.pad(item['e_seq'], (0, max_e_len - item['e_seq'].size(0)), value=PAD_TOKEN_E)
        for item in batch
    ])
    padded_f_seq = torch.stack([
        F.pad(item['f_seq'], (0, max_f_len - item['f_seq'].size(0)), value=PAD_TOKEN_F)
        for item in batch
    ])

    # Pad vertex quantized (X,Y,Z) data (`v_q_xyz`) for Edge/Face models input.
    # Pads to (max_V_batch, 3). Value 0 is safe as coords are 0-63.
    padded_v_q_xyz = torch.stack([
         F.pad(item['v_q_xyz'], (0, 0, 0, max_V_batch - item['v_q_xyz'].size(0)), value=0)
         for item in batch
    ])
    
    # Pad edge index tuples (`edge_indices_for_face_model`) for FaceModel input.
    # Pads to (max_E_batch, 3). Value -1 is safe as vertex indices are non-negative.
    padded_edge_indices_for_face_model = torch.stack([
        F.pad(item['edge_indices_for_face_model'], (0, 0, 0, max_E_batch - item['edge_indices_for_face_model'].size(0)), value=-1)
        for item in batch
    ])

    # Stack other batch items directly (they should already be tensors of consistent shape).
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'v_seq': padded_v_seq,
        'e_seq': padded_e_seq,
        'f_seq': padded_f_seq,
        'v_q_xyz': padded_v_q_xyz,
        'edge_indices_for_face_model': padded_edge_indices_for_face_model,
        'V_min_orig': torch.stack([item['V_min_orig'] for item in batch]),
        'V_max_orig': torch.stack([item['V_max_orig'] for item in batch]),
        'file_id': [item['file_id'] for item in batch]
    }

def parse_args():
    """Parses command-line arguments for training."""
    parser = argparse.ArgumentParser(description="SolidGen Training Pipeline")

    # --- Path and Directory Arguments ---
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory of the dataset.')
    parser.add_argument('--output_dir', type=str, default='./training_output', help='Directory to save checkpoints and logs.')
    parser.add_argument('--run_name', type=str, default=f'solidgen_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}', help='A name for this training run, used for logging.')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to a checkpoint file to resume training from.')

    # --- Training Hyperparameters ---
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training. Paper used 512, adjust for your GPU memory.')
    parser.add_argument('--epochs', type=int, default=1000, help='Total number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the AdamW optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for the AdamW optimizer.')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Value for gradient clipping.')


    # --- System and Logging Arguments ---
    parser.add_argument('--num_workers', type=int, default=max(0, os.cpu_count() // 2), help='Number of worker processes for DataLoader.')
    parser.add_argument('--log_interval', type=int, default=50, help='Log training metrics to TensorBoard every N batches.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

    return parser.parse_args()

def save_checkpoint(state, is_best, directory, filename='checkpoint.pth.tar'):
    """Saves model and training state."""
    filepath = os.path.join(directory, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(directory, 'model_best.pth.tar')
        torch.save(state, best_filepath)
        print(f" => Saved new best model to {best_filepath}")

def load_checkpoint(models_tuple, optimizer, filepath, device):
    """Loads model and training state from checkpoint."""
    if not os.path.isfile(filepath):
        print(f" => No checkpoint found at '{filepath}'")
        return None

    print(f" => Loading checkpoint from '{filepath}'")
    checkpoint = torch.load(filepath, map_location=device)

    v_model, e_model, f_model = models_tuple
    v_model.load_state_dict(checkpoint['v_model_state_dict'])
    e_model.load_state_dict(checkpoint['e_model_state_dict'])
    f_model.load_state_dict(checkpoint['f_model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    print(f" => Loaded checkpoint from epoch {start_epoch} with best validation loss {best_val_loss:.4f}")
    return start_epoch, best_val_loss

# def load_checkpoint(models_tuple, optimizer, filepath, device):
    # """Loads model and training state from a checkpoint file."""
    # if not os.path.isfile(filepath):
    #     print(f" => No checkpoint found at '{filepath}'")
    #     return None

    # print(f" => Loading checkpoint from '{filepath}'")
    # checkpoint = torch.load(filepath, map_location=device)

    # img_encoder, v_model, e_model, f_model = models_tuple
    # img_encoder.load_state_dict(checkpoint['img_encoder_state_dict'])
    # v_model.load_state_dict(checkpoint['v_model_state_dict'])
    # e_model.load_state_dict(checkpoint['e_model_state_dict'])
    # f_model.load_state_dict(checkpoint['f_model_state_dict'])

    # if optimizer is not None and 'optimizer_state_dict' in checkpoint:
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # start_epoch = checkpoint.get('epoch', 0)
    # best_val_loss = checkpoint.get('best_val_loss', float('inf'))


    # print(f" => Loaded checkpoint from epoch {start_epoch} with best validation loss {best_val_loss:.4f}")
    # return start_epoch, best_val_loss

# ========================
# TRAINING AND EVALUATION FUNCTIONS - UNCONDITIONAL
# ========================
def train_epoch_unconditional(models_tuple, dataloader, optimizer, criterion, device, epoch, writer, args):
    """
    Performs one UNCONDITIONAL training epoch wit gradient accumulation
    Effective batch size = batch_size * accumulation_steps = 64 * 8 = 512 (matches paper)
    Key difference: image_embed is None for all models.
    """
    v_model, e_model, f_model = models_tuple
    v_model.train(); e_model.train(); f_model.train()

    # Gradient accumulation to simulate batch_size=512
    accumulation_steps = 8  # 64 * 8 = 512 effective batch size

    total_loss, total_v_loss, total_e_loss, total_f_loss = 0, 0, 0, 0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{args.epochs}")

    optimizer.zero_grad()   # Zero gradients at start

    for i, batch in enumerate(pbar):
        if batch is None:
            continue

        global_step = epoch * len(dataloader) + i
        
        # Move batch data to device (NO IMAGE)
        v_seq = batch['v_seq'].to(device)
        e_seq = batch['e_seq'].to(device)
        f_seq = batch['f_seq'].to(device)
        v_q_xyz = batch['v_q_xyz'].to(device)
        edge_indices = batch['edge_indices_for_face_model'].to(device)

        # === UNCONDITIONAL: image_embed = None ===
        image_embed = None

        # --- Vertex Model ---
        v_input, v_target = v_seq[:, :-1], v_seq[:, 1:]
        v_mask = (v_target != PAD_TOKEN_V)
        v_logits = v_model(v_input, image_embed)  # image_embed = None
        v_loss = criterion(v_logits.reshape(-1, v_logits.size(-1)), v_target.reshape(-1))
        v_loss = (v_loss * v_mask.view(-1)).sum() / (v_mask.sum() + 1e-6)

        # --- Edge Model ---
        e_input, e_target = e_seq[:, :-1], e_seq[:, 1:]
        e_mask = (e_target != PAD_TOKEN_E)
        e_logits = e_model(e_input, v_q_xyz, image_embed)  # image_embed = None
        e_target_safe = e_target.clone()
        e_target_safe[~e_mask] = 0
        e_loss = criterion(e_logits.reshape(-1, e_logits.size(-1)), e_target_safe.reshape(-1))
        e_loss = (e_loss * e_mask.view(-1)).sum() / (e_mask.sum() + 1e-6)

        # --- Face Model ---
        f_input, f_target = f_seq[:, :-1], f_seq[:, 1:]
        f_mask = (f_target != PAD_TOKEN_F)
        f_logits = f_model(f_input, v_q_xyz, edge_indices, image_embed)  # image_embed = None
        f_target_safe = f_target.clone()
        f_target_safe[~f_mask] = 0
        f_loss = criterion(f_logits.reshape(-1, f_logits.size(-1)), f_target_safe.reshape(-1))
        f_loss = (f_loss * f_mask.view(-1)).sum() / (f_mask.sum() + 1e-6)

        # --- Total Loss and Optimization ---
        loss = (v_loss + e_loss + f_loss) / accumulation_steps
        loss.backward() # Gradients accumulate
        
        # --- Update weights every accumulation_steps ---
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                list(v_model.parameters()) + list(e_model.parameters()) + list(f_model.parameters()),
                args.clip_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad()

        # --- Accumulate and Log Metrics ---
        actual_loss = loss.item() * accumulation_steps
        total_loss += actual_loss
        total_v_loss += v_loss.item()
        total_e_loss += e_loss.item()
        total_f_loss += f_loss.item()
        num_batches += 1

        if (i + 1) % args.log_interval == 0:
            writer.add_scalar('Loss/train_total', actual_loss, global_step)
            writer.add_scalar('Loss/train_vertex', v_loss.item(), global_step)
            writer.add_scalar('Loss/train_edge', e_loss.item(), global_step)
            writer.add_scalar('Loss/train_face', f_loss.item(), global_step)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], global_step)

        pbar.set_postfix({
            'Loss': f'{actual_loss:.4f}',
            'V': f'{v_loss.item():.4f}',
            'E': f'{e_loss.item():.4f}',
            'F': f'{f_loss.item():.4f}'
        })


    # Handle remaining gradients at epoch end
    if num_batches % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            list(v_model.parameters()) + list(e_model.parameters()) + list(f_model.parameters()),
            args.clip_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def evaluate_model_unconditional(models_tuple, dataloader, criterion, device, epoch, writer):
    """Evaluates models on a given dataloader - UNCONDITIONAL."""
    v_model, e_model, f_model = models_tuple
    v_model.eval(); e_model.eval(); f_model.eval()

    total_loss, total_v_loss, total_e_loss, total_f_loss = 0, 0, 0, 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validating Epoch {epoch+1}"):
            if batch is None:
                continue

            v_seq = batch['v_seq'].to(device)
            e_seq = batch['e_seq'].to(device)
            f_seq = batch['f_seq'].to(device)
            v_q_xyz = batch['v_q_xyz'].to(device)
            edge_indices = batch['edge_indices_for_face_model'].to(device)

            # === UNCONDITIONAL: image_embed = None ===
            image_embed = None

            # --- Vertex Model Loss ---
            v_input, v_target = v_seq[:, :-1], v_seq[:, 1:]
            v_mask = (v_target != PAD_TOKEN_V)
            v_logits = v_model(v_input, image_embed)
            v_loss = (criterion(v_logits.reshape(-1, v_logits.size(-1)), v_target.reshape(-1)) * v_mask.reshape(-1)).sum() / v_mask.sum()

            # --- Edge Model Loss ---
            e_input, e_target = e_seq[:, :-1], e_seq[:, 1:]
            e_mask = (e_target != PAD_TOKEN_E)
            e_logits = e_model(e_input, v_q_xyz, image_embed)
            e_target_safe = e_target.clone()
            e_target_safe[~e_mask] = 0
            e_loss = (criterion(e_logits.reshape(-1, e_logits.size(-1)), e_target_safe.reshape(-1)) * e_mask.reshape(-1)).sum() / e_mask.sum()

            # --- Face Model Loss ---
            f_input, f_target = f_seq[:, :-1], f_seq[:, 1:]
            f_mask = (f_target != PAD_TOKEN_F)
            f_logits = f_model(f_input, v_q_xyz, edge_indices, image_embed)
            f_target_safe = f_target.clone()
            f_target_safe[~f_mask] = 0
            f_loss = (criterion(f_logits.reshape(-1, f_logits.size(-1)), f_target_safe.reshape(-1)) * f_mask.reshape(-1)).sum() / f_mask.sum()
            
            loss = v_loss + e_loss + f_loss
            total_loss += loss.item()
            total_v_loss += v_loss.item()
            total_e_loss += e_loss.item()
            total_f_loss += f_loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_v_loss = total_v_loss / num_batches if num_batches > 0 else 0
    avg_e_loss = total_e_loss / num_batches if num_batches > 0 else 0
    avg_f_loss = total_f_loss / num_batches if num_batches > 0 else 0

    writer.add_scalar('Loss/val_total', avg_loss, epoch)
    writer.add_scalar('Loss/val_vertex', avg_v_loss, epoch)
    writer.add_scalar('Loss/val_edge', avg_e_loss, epoch)
    writer.add_scalar('Loss/val_face', avg_f_loss, epoch)

    return avg_loss


# ========================
# TRAINING AND EVALUATION FUNCTIONS
# ========================
def train_epoch(models_tuple, dataloader, optimizer, criterion, device, epoch, writer, args):
    """Performs one training epoch logs metrics."""
    img_encoder, v_model, e_model, f_model = models_tuple
    # Set all models to training mode
    img_encoder.train(); v_model.train(); e_model.train(); f_model.train()

    total_loss, total_v_loss, total_e_loss, total_f_loss = 0, 0, 0, 0
    # Iterate through batches with tqdm for a progress bar
    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{args.epochs}")
    
    for i, batch in enumerate(pbar):
        if batch is None: # Skip if collate_fn returned None (empty batch after filtering)
            continue

        global_step = epoch * len(dataloader) + i
        optimizer.zero_grad() # Clear gradients from previous step

        # Move batch data to the specified device (GPU/CPU)
        image = batch['image'].to(device)
        v_seq = batch['v_seq'].to(device)
        e_seq = batch['e_seq'].to(device)
        f_seq = batch['f_seq'].to(device)
        v_q_xyz = batch['v_q_xyz'].to(device)
        edge_indices = batch['edge_indices_for_face_model'].to(device)

        # === Image Encoder Forward Pass ===
        image_embed = img_encoder(image) # (B, 256, d_model) - memory for decoders

        # === Vertex Model Loss Calculation ===
        # For teacher forcing: input is sequence up to t-1, target is sequence from t.
        # --- Vertex Model ---
        v_input, v_target = v_seq[:, :-1], v_seq[:, 1:]
        v_mask = (v_target != PAD_TOKEN_V)
        v_logits = v_model(v_input, image_embed)
        v_loss = criterion(v_logits.reshape(-1, v_logits.size(-1)), v_target.reshape(-1))
        v_loss = (v_loss * v_mask.view(-1)).sum() / (v_mask.sum() + 1e-6)

        # --- Edge Model ---
        e_input, e_target = e_seq[:, :-1], e_seq[:, 1:]
        e_mask = (e_target != PAD_TOKEN_E)
        e_logits = e_model(e_input, v_q_xyz, image_embed)
        # Create a safe target tensor for the loss function
        e_target_safe = e_target.clone()
        # Replace the out-of-bounds PAD token with a valid index (e.g., 0)
        e_target_safe[~e_mask] = 0 # Replace PAD_TOKEN_E with 0 for loss calculation
        e_loss = criterion(e_logits.reshape(-1, e_logits.size(-1)), e_target_safe.reshape(-1))
        e_loss = (e_loss * e_mask.view(-1)).sum() / (e_mask.sum() + 1e-6)

        # --- Face Model ---
        f_input, f_target = f_seq[:, :-1], f_seq[:, 1:]
        f_mask = (f_target != PAD_TOKEN_F)
        f_logits = f_model(f_input, v_q_xyz, edge_indices, image_embed)
        f_target_safe = f_target.clone()
        f_target_safe[~f_mask] = 0
        f_loss = criterion(f_logits.reshape(-1, f_logits.size(-1)), f_target_safe.reshape(-1))
        f_loss = (f_loss * f_mask.view(-1)).sum() / (f_mask.sum() + 1e-6)

        # --- Total Loss and Optimization ---
        loss = v_loss + e_loss + f_loss # the models are trained jointly
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            list(img_encoder.parameters()) + list(v_model.parameters()) + 
            list(e_model.parameters()) + list(f_model.parameters()),
            args.clip_grad_norm
        )
        optimizer.step()

        # --- Accumulate and Log Metrics ---
        total_loss += loss.item()
        total_v_loss += v_loss.item()
        total_e_loss += e_loss.item()
        total_f_loss += f_loss.item()

        if (i + 1) % args.log_interval == 0:
            writer.add_scalar('Loss/train_total', loss.item(), global_step)
            writer.add_scalar('Loss/train_vertex', v_loss.item(), global_step)
            writer.add_scalar('Loss/train_edge', e_loss.item(), global_step)
            writer.add_scalar('Loss/train_face', f_loss.item(), global_step)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], global_step)

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'V_Loss': f'{v_loss.item():.4f}',
            'E_Loss': f'{e_loss.item():.4f}',
            'F_Loss': f'{f_loss.item():.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate_model(models_tuple, dataloader, criterion, device, epoch, writer):
    """Evaluates the model on a given dataset, logs metrics."""
    img_encoder, v_model, e_model, f_model = models_tuple
    # Set all models to evaluation mode
    img_encoder.eval(); v_model.eval(); e_model.eval(); f_model.eval()

    total_loss, total_v_loss, total_e_loss, total_f_loss = 0, 0, 0, 0
    num_batches = 0

    with torch.no_grad(): # Disable gradient calculations during evaluation
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            if batch is None:
                continue
            
            image = batch['image'].to(device)
            v_seq = batch['v_seq'].to(device)
            e_seq = batch['e_seq'].to(device)
            f_seq = batch['f_seq'].to(device)
            v_q_xyz = batch['v_q_xyz'].to(device)
            edge_indices = batch['edge_indices_for_face_model'].to(device)

            image_embed = img_encoder(image)

            # --- Vertex Model ---
            v_input, v_target = v_seq[:, :-1], v_seq[:, 1:]
            v_mask = (v_target != PAD_TOKEN_V)
            v_logits = v_model(v_input, image_embed)
            v_loss = criterion(v_logits.reshape(-1, v_logits.size(-1)), v_target.reshape(-1))
            v_loss = (v_loss * v_mask.view(-1)).sum() / (v_mask.sum() + 1e-6)

            # --- Edge Model ---
            e_input, e_target = e_seq[:, :-1], e_seq[:, 1:]
            e_mask = (e_target != PAD_TOKEN_E)
            e_logits = e_model(e_input, v_q_xyz, image_embed)
            e_target_safe = e_target.clone()
            e_target_safe[~e_mask] = 0
            e_loss = criterion(e_logits.reshape(-1, e_logits.size(-1)), e_target_safe.reshape(-1))
            e_loss = (e_loss * e_mask.view(-1)).sum() / (e_mask.sum() + 1e-6)

            # --- Face Model ---
            f_input, f_target = f_seq[:, :-1], f_seq[:, 1:]
            f_mask = (f_target != PAD_TOKEN_F)
            f_logits = f_model(f_input, v_q_xyz, edge_indices, image_embed)
            f_target_safe = f_target.clone()
            f_target_safe[~f_mask] = 0
            f_loss = criterion(f_logits.reshape(-1, f_logits.size(-1)), f_target_safe.reshape(-1))
            f_loss = (f_loss * f_mask.view(-1)).sum() / (f_mask.sum() + 1e-6)

            loss = v_loss + e_loss + f_loss
            total_loss += loss.item()
            total_v_loss += v_loss.item()
            total_e_loss += e_loss.item()
            total_f_loss += f_loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_v_loss = total_v_loss / num_batches if num_batches > 0 else 0
    avg_e_loss = total_e_loss / num_batches if num_batches > 0 else 0
    avg_f_loss = total_f_loss / num_batches if num_batches > 0 else 0

    writer.add_scalar('Loss/val_total', avg_loss, epoch)
    writer.add_scalar('Loss/val_vertex', avg_v_loss, epoch)
    writer.add_scalar('Loss/val_edge', avg_e_loss, epoch)
    writer.add_scalar('Loss/val_face', avg_f_loss, epoch)
    
    return avg_loss

# ========================
# MAIN TRAINING SCRIPT EXECUTION (Unconditional)
# ========================
if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn')
        print("--- Multiprocessing start method set to 'spawn'. ---")
    except RuntimeError:
        pass
    
    from multiprocessing import freeze_support
    freeze_support()

    # --- 1. Setup and Configuration ---
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 60)
    print("UNCONDITIONAL TRAINING MODE")
    print("No image conditioning - learning p(B) = p(F|E,V) * p(E|V) * p(V)")
    print("=" * 60)
    
    # Create output directories
    run_dir = os.path.join(args.output_dir, args.run_name)
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    logs_dir = os.path.join(run_dir, 'logs')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # --- 2. Data Loading (NO IMAGES) ---
    print("Loading datasets (unconditional - no images)...")
    
    train_dataset = SolidGenDatasetUnconditional(args.data_root, 'train')
    val_dataset = SolidGenDatasetUnconditional(args.data_root, 'validation')
    test_dataset = SolidGenDatasetUnconditional(args.data_root, 'test')
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                   collate_fn=collate_fn_unconditional, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                 collate_fn=collate_fn_unconditional, num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                  collate_fn=collate_fn_unconditional, num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    if len(train_dataset) == 0:
        print("ERROR: Training dataset is empty.", file=sys.stderr)
        sys.exit(1)

    # --- 3. Model Initialization (NO ImageEncoder) ---
    print("Initializing models (unconditional - no image encoder)...")
    v_model = VertexModel(vocab_size=65).to(device)
    e_model = EdgeModel().to(device)
    f_model = FaceModel().to(device)

    models = (v_model, e_model, f_model)

    # Optimizer only includes V, E, F model parameters (NO img_encoder)
    optimizer = torch.optim.AdamW(
        list(v_model.parameters()) + list(e_model.parameters()) + list(f_model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True)
    
    criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1) # We'll handle masking manually, so use reduction='none'

    # --- 4. Checkpointing and TensorBoard Setup ---
    writer = SummaryWriter(log_dir=logs_dir)
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    if args.resume_checkpoint:
        checkpoint_data = load_checkpoint(models, optimizer, args.resume_checkpoint, device)
        if checkpoint_data:
            start_epoch, best_val_loss = checkpoint_data

            # Load checkpoint to get scheduler state and loss history
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            
            # Load previous loss history if available
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            print(f" => Loaded {len(train_losses)} epochs of loss history")
            
            # # Reset LR BEFORE creating scheduler
            # new_lr = 3e-5
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = new_lr
            # print(f" => Reset learning rate to {new_lr}")
    
    # CREATE SCHEDULER HERE - after LR is set correctly!
    # scheduler = CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=50,
    #     T_mult=2,
    #     eta_min=1e-6
    # )
    scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-6)
    print(f" => Scheduler created with base_lr: {optimizer.param_groups[0]['lr']:.2e}")   
    
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(" => Scheduler state restored (LR schedule will continue correctly)")
        else:
            print("WARNING: No scheduler_state_dict found in checkpoint")
    
    print("Restored LR:", optimizer.param_groups[0]['lr'])
    print("Scheduler last_epoch:", scheduler.last_epoch)    
    print("Training setup complete.")
    
    # --- 5. Main Training Loop ---
    print("Starting UNCONDITIONAL training...")
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        train_loss = train_epoch_unconditional(models, train_dataloader, optimizer, criterion, device, epoch, writer, args)
        val_loss = evaluate_model_unconditional(models, val_dataloader, criterion, device, epoch, writer)
        scheduler.step()
        
        end_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e} | Time: {end_time - start_time:.2f}s")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

                # --- Save Loss Curve Plot ---
        plt.figure(figsize=(12, 5))

        # Plot 1: Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Val Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Progress (Epoch {epoch+1})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Recent epochs zoomed (last 100)
        plt.subplot(1, 2, 2)
        recent = min(100, len(train_losses))
        plt.plot(range(len(train_losses)-recent, len(train_losses)), train_losses[-recent:], label='Train Loss', color='blue')
        plt.plot(range(len(val_losses)-recent, len(val_losses)), val_losses[-recent:], label='Val Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Last {recent} Epochs (LR: {current_lr:.2e})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'loss_curve.png'), dpi=100)
        plt.close()

        # Also save to text file for easy checking
        with open(os.path.join(run_dir, 'training_log.txt'), 'a') as f:
            f.write(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.2e} | Best: {best_val_loss:.4f}\n")

        # --- Checkpoint Saving (NO img_encoder) ---
        # --- Checkpoint Saving (NO img_encoder) ---
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'v_model_state_dict': v_model.state_dict(),
            'e_model_state_dict': e_model.state_dict(),
            'f_model_state_dict': f_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'mode': 'unconditional'  # Mark as unconditional checkpoint
        }, is_best, directory=checkpoints_dir)
    
    print("Training complete.")
    writer.close()

    # --- 6. Final Evaluation on Test Set ---
    print("\n--- Evaluating on Test Set with Best Model ---")
    best_model_path = os.path.join(checkpoints_dir, 'model_best.pth.tar')
    if os.path.exists(best_model_path):
        load_checkpoint(models, optimizer=None, filepath=best_model_path, device=device)
        test_loss = evaluate_model_unconditional(models, test_dataloader, criterion, device, epoch=args.epochs, 
                                                  writer=SummaryWriter(log_dir=os.path.join(logs_dir, 'test')))
        print(f"Final Test Loss: {test_loss:.4f}")
    else:
        print("No best model checkpoint found.", file=sys.stderr)



# ========================
# MAIN TRAINING SCRIPT EXECUTION
# ========================
# if __name__ == "__main__":
    # import torch.multiprocessing as mp
    # try:
    #     mp.set_start_method('spawn')
    #     print("--- Multiprocessing start method set to 'spawn'. ---")
    # except RuntimeError:
    #     pass
    # # Best practice for multiprocessing (e.g., DataLoader with num_workers > 0) on Windows.
    # # Prevents issues where child processes re-import the main script's code.
    # from multiprocessing import freeze_support
    # freeze_support()

    # # --- 1. Setup and Configuration ---
    # args = parse_args()
    # set_seed(args.seed)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    # print("=" * 60)
    # print("UNCONDITIONAL TRAINING MODE")
    # print("No image conditioning - learning p(B) = p(F|E,V) * p(E|V) * p(V)")
    # print("=" * 60)


    # # Create output directories for this run
    # run_dir = os.path.join(args.output_dir, args.run_name)
    # checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    # logs_dir = os.path.join(run_dir, 'logs')
    # os.makedirs(checkpoints_dir, exist_ok=True)
    # os.makedirs(logs_dir, exist_ok=True)

    # # --- 2. Data Loading ---
    # print("Loading datasets (unconditional - no images)...")
    # # image_transform = transforms.Compose([transforms.ToTensor()]) # trying below transform for reducing overfitting (at epoch 66)
    # # train_dataset = SolidGenDataset(args.data_root, 'train', transform=image_transform)
    # # val_dataset = SolidGenDataset(args.data_root, 'validation', transform=image_transform)
    # # test_dataset = SolidGenDataset(args.data_root, 'test', transform=image_transform)
    
    # # ----------- Putting data augmentation only for training set for solving overfitting-----------
    # train_image_transform = transforms.Compose([
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    # # transforms.RandomHorizontalFlip(p=0.5),
    # # transforms.RandomVerticalFlip(p=0.5),
    # transforms.ToTensor(),
    # # You can also add normalization if you find it helps
    # # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    # ])
    # val_test_image_transform = transforms.Compose([transforms.ToTensor()])
    # # -----------------------------------------------------------------------------------------------
  
    # train_dataset = SolidGenDataset(args.data_root, 'train', transform=train_image_transform)
    # val_dataset = SolidGenDataset(args.data_root, 'validation', transform=val_test_image_transform)
    # test_dataset = SolidGenDataset(args.data_root, 'test', transform=val_test_image_transform)

    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

    # print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    # if len(train_dataset) == 0:
    #     print("ERROR: Training dataset is empty. Check data paths.", file=sys.stderr)
    #     sys.exit(1)

    # # --- 3. Model, Optimizer, and Loss Initialization ---
    # print("Initializing models...")
    # img_encoder = ImageEncoder().to(device)
    # v_model = VertexModel(vocab_size=65).to(device)
    # e_model = EdgeModel().to(device)
    # f_model = FaceModel().to(device)
    # # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    # #     print(f"Using {torch.cuda.device_count()} GPUs for Data Parallelism.")
    # #     # Wrap each model in the DataParallel module
    # #     img_encoder = nn.DataParallel(img_encoder)
    # #     v_model = nn.DataParallel(v_model)
    # #     e_model = nn.DataParallel(e_model)
    # #     f_model = nn.DataParallel(f_model)
    # models = (img_encoder, v_model, e_model, f_model)

    # optimizer = torch.optim.AdamW(
    #     list(img_encoder.parameters()) + list(v_model.parameters()) + 
    #     list(e_model.parameters()) + list(f_model.parameters()),
    #     lr=args.lr, weight_decay=args.weight_decay
    # )
    # # This scheduler will watch the validation loss. If it doesn't improve for 5 epochs,
    # # it will reduce the learning rate by a factor of 0.2 (e.g., 1e-4 -> 2e-5).
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)
    # criterion = nn.CrossEntropyLoss(reduction='none') # Use 'none' for manual masking

    # # --- 4. Checkpointing and TensorBoard Setup ---
    # writer = SummaryWriter(log_dir=logs_dir)
    # start_epoch = 0
    # best_val_loss = float('inf')

    # if args.resume_checkpoint:
    #     checkpoint_data = load_checkpoint(models, optimizer, args.resume_checkpoint, device)
    #     if checkpoint_data:
    #         start_epoch, best_val_loss = checkpoint_data
    # # --- Real-time Plotting Setup ---
    # plt.ion() # Turn on interactive mode
    # fig, ax = plt.subplots(figsize=(10, 6))
    # train_losses = []
    # val_losses = []
    # ax.set_xlabel("Epoch")
    # ax.set_ylabel("Loss")
    # ax.set_title("Live Training Progress")
    # line_train, = ax.plot(train_losses, label="Train Loss")
    # line_val, = ax.plot(val_losses, label="Val Loss")
    # ax.legend()
    # # --- End Setup ---
    # print("Training setup complete.")
    # # --- 5. Main Training Loop ---
    # print("Starting training...")
    # for epoch in range(start_epoch, args.epochs):
    #     start_time = time.time()
        
    #     train_loss = train_epoch(models, train_dataloader, optimizer, criterion, device, epoch, writer, args)
    #     val_loss = evaluate_model(models, val_dataloader, criterion, device, epoch, writer)
    #     scheduler.step(val_loss)
    #     end_time = time.time()
        
    #     print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {end_time - start_time:.2f}s")
    #     # --- SIMPLER PLOTTING BLOCK ---
    #     train_losses.append(train_loss)
    #     val_losses.append(val_loss)

    #     # Re-plot and save the image every epoch
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(train_losses, label="Train Loss")
    #     plt.plot(val_losses, label="Val Loss")
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Loss")
    #     plt.title("Training Progress")
    #     plt.legend()
    #     plt.savefig(os.path.join(run_dir, "live_loss_curve4.png"))
    #     plt.close() # Important to close the figure to save memory
    #     # --- END SIMPLER PLOTTING BLOCK ---
    #     # --- Checkpoint Saving ---
    #     is_best = val_loss < best_val_loss
    #     best_val_loss = min(val_loss, best_val_loss)
        
    #     save_checkpoint({
    #         'epoch': epoch + 1,
    #         'img_encoder_state_dict': img_encoder.state_dict(),
    #         'v_model_state_dict': v_model.state_dict(),
    #         'e_model_state_dict': e_model.state_dict(),
    #         'f_model_state_dict': f_model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'best_val_loss': best_val_loss,
    #         'train_losses': train_losses,
    #         'val_losses': val_losses
    #     }, is_best, directory=checkpoints_dir)
    
    # print("Training complete.")
    # writer.close()

    # # --- 6. Final Evaluation on Test Set (using the best model) ---
    # print("\n--- Evaluating on Test Set with Best Model ---")
    # best_model_path = os.path.join(checkpoints_dir, 'model_best.pth.tar')
    # if os.path.exists(best_model_path):
    #     load_checkpoint(models, optimizer=None, filepath=best_model_path, device=device)
    #     test_loss = evaluate_model(models, test_dataloader, criterion, device, epoch=args.epochs, writer=SummaryWriter(log_dir=os.path.join(logs_dir, 'test')))
    #     print(f"Final Test Loss: {test_loss:.4f}")

    #     # ### PLACEHOLDER for custom evaluation metrics ###
    #     # Here you could add a function to calculate Chamfer Distance, IoU, etc.
    #     # It would iterate through the test_dataloader, generate models, and compare to ground truth.
    #     # E.g., `calculate_geometric_metrics(models, test_dataloader, device)`
    # else:
    #     print("No best model checkpoint found to run final evaluation.", file=sys.stderr)

    
