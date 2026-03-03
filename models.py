import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
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

TRANSFORMER_CONFIG = {
    'd_model': 256,
    'nhead': 8,
    'dim_feedforward': 512,       
    'activation': 'gelu', 
    'batch_first': True,
    'norm_first': True,    # Critical for convergence (paper Section 5 - pre-LayerNorm)
    'dropout': 0.2  # Slightly higher dropout for regularization
}


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
PAD_TOKEN_V = VERTEX_SOS_TOKEN + 1 # 66
PAD_TOKEN_E = GLOBAL_MAX_VERTICES_MODEL + 3  # 153
PAD_TOKEN_F = GLOBAL_MAX_EDGES_MODEL + 3  # 153

num_layers = 8


class ImageEncoder(nn.Module):
    def __init__(self, d_model=TRANSFORMER_CONFIG['d_model'], dropout=0.1):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16)) # Output 16x16 spatial features (as per paper A.7)
        self.final_conv = nn.Conv2d(64, d_model, kernel_size=3, padding=1) # Output d_model channels
        
        # 2D positional embeddings for spatial grid (16x16)
        self.pos_embed = nn.Parameter(torch.empty(1, 16, 16, d_model))
        nn.init.xavier_uniform_(self.pos_embed)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Input: (B, 3, 105, 128) - Assumes images are normalized to [0,1]
        x = (x - 0.5) * 2 # Normalize to [-1, 1] for Conv layers
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = self.final_conv(x)         # (B, d_model, 16, 16)
        
        # Permute to (B, H, W, C) and add spatial embeddings
        x = x.permute(0, 2, 3, 1)     # (B, 16, 16, d_model)
        x = x + self.pos_embed         # Add learned 2D positional embeddings
        x = self.norm(x)
        x = self.dropout(x)
        
        # SpatialFlatten operation: reshape to a sequence for Transformer cross-attention
        x = x.reshape(x.size(0), -1, x.size(-1))  # (B, 16*16=256, d_model)
        return x

# ========================
# GENERATOR MODULES (Refined for SOS/EOS/PAD tokens and max_seq_len handling)
# ========================
class VertexModel(nn.Module):
    def __init__(self, d_model=TRANSFORMER_CONFIG['d_model'], vocab_size=65): # vocab_size=65 for 0-64. 64 is EOS.
        super().__init__()
        self.vocab_size = vocab_size         # Range of coordinate values [0-63]
        self.eos_token = VERTEX_EOS_TOKEN    # 64
        self.sos_token = VERTEX_SOS_TOKEN    # 65
        self.pad_token = PAD_TOKEN_V         # 66 (for padding sequences)

        # Embeddings
        self.coord_emb = nn.Embedding(3, d_model)         # For x, y, z coordinate type (0, 1, 2)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN_V, d_model) # Positional encoding for sequence position
        self.val_emb = nn.Embedding(vocab_size + 2, d_model) # For  0-63, EOS(64), SOS(65). Total vocab_size+2 = 67 if includes PAD.
                                                             # Current vocab_size is 65, so need 65+2=67 for val_emb.
                                                             # Indexing is 0..66.
                                                             # We use 0-63 for coords, 64 for EOS, 65 for SOS, 66 for PAD.

        decoder_layer = nn.TransformerDecoderLayer(**TRANSFORMER_CONFIG)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output = nn.Linear(d_model, vocab_size + 2) # Output logits for 0-63, EOS(64), SOS(65)
                                                         # Size 67 to match val_emb vocab.

    def forward(self, seq, image_embed):
        # seq: (B, L_seq), tokenized input sequence with SOS, coords, EOS, and PAD_TOKEN
        device = seq.device
        B, L_seq = seq.shape

        # `coord_ids`: Assigns a type (0,1,2) to each token for coordinate embedding.
        # It cycles 0,1,2,0,1,2...
        # The first token (SOS) is mapped to 0 (arbitrary, as it's not a coordinate).
        coord_ids = torch.arange(L_seq, device=device) % 3
        coord_ids[0] = 0 # The SOS token is at index 0 in the sequence

        # `pos_ids`: Standard positional encoding for the sequence length (0 to L_seq-1).
        pos_ids = torch.arange(L_seq, device=device)
        
        # `val_emb`: Embeds the actual token values (0-63, EOS, SOS, PAD).
        # We need to create a temporary input ID tensor for `val_emb` that maps PAD_TOKEN
        # to a valid embedding index (e.g., 0) and then zero out its embedding.
        val_emb_input_ids = seq.clone()
        pad_mask = (val_emb_input_ids == self.pad_token) # Identify padding tokens
        val_emb_input_ids[pad_mask] = 0 # Temporarily map padding tokens to 0 for embedding lookup

        val_emb = self.val_emb(val_emb_input_ids) # (B, L_seq, d_model)
        val_emb[pad_mask] = 0.0 # Set embedding for padding tokens to zero (mask their effect)
        coord_emb = self.coord_emb(coord_ids)
        pos_emb = self.pos_emb(pos_ids)
        x = val_emb + coord_emb.unsqueeze(0) + pos_emb.unsqueeze(0) # Combine embeddings (B, L_seq, d_model)

        # `tgt_mask`: Causal mask (upper triangle True) to prevent looking into the future.
        tgt_mask = torch.triu(torch.ones(L_seq, L_seq, device=device, dtype=torch.bool), diagonal=1)
        # `tgt_key_padding_mask`: Mask out padding tokens from attention calculations. True means "mask out".
        tgt_key_padding_mask = (seq == self.pad_token) # (B, L_seq), True where padding

        # Transformer decoder forward pass
        if image_embed is None:
            # If no image conditioning (e.g., unconditional model or during initial debugging).
            # Use zero memory of appropriate shape (B, 1, d_model) to maintain API.
            zero_mem = torch.zeros(B, 1, x.size(-1), device=device)
            out = self.transformer(tgt=x, memory=zero_mem, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        else:
            out = self.transformer(tgt=x, memory=image_embed, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        
        return self.output(out) # (B, L_seq, vocab_size + 2) - logits over all possible tokens


class EdgeModel(nn.Module):
    """
    FIXED VERSION: Uses GLOBAL_MAX_VERTICES_MODEL for fixed-size vocabulary.
    
    h_mp is ALWAYS of shape (B, GLOBAL_MAX_VERTICES + 3, d_model):
    - Indices 0 to GLOBAL_MAX_VERTICES-1: vertex embeddings (padded with zeros for unused)
    - Index GLOBAL_MAX_VERTICES (150): NEW_EDGE token
    - Index GLOBAL_MAX_VERTICES+1 (151): EOS token
    - Index GLOBAL_MAX_VERTICES+2 (152): SOS token
    
    This ensures tokens always have the same meaning across all samples and batches.
    """
    def __init__(self, max_num_vertices=GLOBAL_MAX_VERTICES_MODEL, max_seq_len=MAX_SEQ_LEN_E):
        super().__init__()
        d_model = TRANSFORMER_CONFIG['d_model']
        self.d_model = d_model
        self.max_V_capacity = max_num_vertices # FIXED: Always 150

        # Special token values - FIXED positions in vocabulary
        self.new_edge_token = EDGE_NEW_EDGE_TOKEN  # 150
        self.eos_token = EDGE_EOS_TOKEN            # 151
        self.sos_token = EDGE_SOS_TOKEN            # 152
        self.pad_token = PAD_TOKEN_E               # 153

        # Learned projection from vertex coordinates (quantized 0-63) to embeddings
        self.Wx = nn.Embedding(64, 64) # For X-coord of vertex (64 quantized values -> 64 dim)
        self.Wy = nn.Embedding(64, 64) # For Y-coord of vertex
        self.Wz = nn.Embedding(64, 64) # For Z-coord of vertex
        self.phi = nn.Linear(64 * 3, d_model) # Maps concatenated 64*3 dim to d_model for vertex embeddings

        # Special token embeddings (as learnable Parameters, batch-friendly for concatenation)
        self.h_new_edge = nn.Parameter(torch.randn(1, 1, d_model))
        self.h_eos = nn.Parameter(torch.randn(1, 1, d_model))
        self.h_sos = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embedding for edge sequence
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(**TRANSFORMER_CONFIG)
        self.vertex_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(**TRANSFORMER_CONFIG)
        self.edge_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, edge_seq, vertex_tokens, image_embed):
        """
        Args:
            edge_seq: (B, L_seq) sequence of tokens using FIXED vocabulary
                      - Vertex indices: 0 to V_actual-1
                      - NEW_EDGE: 150 (fixed)
                      - EOS: 151 (fixed)
                      - SOS: 152 (fixed)
                      - PAD: 153 (fixed)
            vertex_tokens: (B, V_actual, 3) batch of quantized vertex coordinates
            image_embed: (B, 256, d_model) image features
            
        Returns:
            logits: (B, L_seq, 153) - pointer distribution over FIXED vocabulary
        """
        B, L_seq = edge_seq.shape
        device = edge_seq.device
        V_actual = vertex_tokens.size(1)  # Actual vertices in batch (may be < max_V_capacity)
        
        # === Encode actual vertices ===
        x_emb = self.Wx(vertex_tokens[..., 0])  # (B, V_actual, 64)
        y_emb = self.Wy(vertex_tokens[..., 1])
        z_emb = self.Wz(vertex_tokens[..., 2])
        h_vert_actual = self.phi(torch.cat([x_emb, y_emb, z_emb], dim=-1))  # (B, V_actual, d_model)
        
        # === Build FIXED-SIZE h_mp ===
        # Pad vertex embeddings to max_V_capacity (150)
        # Unused vertex positions get zero embeddings
        h_vert_padded = torch.zeros(B, self.max_V_capacity, self.d_model, device=device)
        h_vert_padded[:, :V_actual, :] = h_vert_actual  # Fill actual vertices
        
        # Concatenate: [v0, v1, ..., v149, NEW_EDGE(150), EOS(151), SOS(152)]
        h_mp = torch.cat([
            h_vert_padded,                              # indices 0-149
            self.h_new_edge.expand(B, -1, -1),          # index 150
            self.h_eos.expand(B, -1, -1),               # index 151
            self.h_sos.expand(B, -1, -1)                # index 152
        ], dim=1)  # Shape: (B, 153, d_model) - ALWAYS this size
        
        # Encode the memory pool
        h_mp = self.vertex_encoder(h_mp)  # (B, 153, d_model)
        
        # === Build input embeddings for edge_seq ===
        pos_ids = torch.arange(L_seq, device=device).unsqueeze(0).expand(B, L_seq)
        pos_embed = self.pos_emb(pos_ids)  # (B, L_seq, d_model)

        # Map edge_seq tokens to embeddings from h_mp
        inp_embed_idx = edge_seq.clone()
        pad_mask = (inp_embed_idx == self.pad_token)  # Identify PAD tokens (153)
        inp_embed_idx[pad_mask] = 0  # Temporarily map to 0 for gather
        
        inp_embed = torch.gather(h_mp, 1, inp_embed_idx.unsqueeze(-1).expand(-1, -1, self.d_model))
        inp_embed[pad_mask] = 0.0  # Zero out padding embeddings
        inp_embed = inp_embed + pos_embed

        # Causal mask and padding mask
        tgt_mask = torch.triu(torch.ones(L_seq, L_seq, device=device, dtype=torch.bool), diagonal=1)
        tgt_key_padding_mask = (edge_seq == self.pad_token)

        # === Transformer decoder forward pass ===
        if image_embed is None:
            zero_mem = torch.zeros(B, 1, self.d_model, device=device)
            out = self.edge_decoder(tgt=inp_embed, memory=zero_mem, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        else:
            out = self.edge_decoder(tgt=inp_embed, memory=image_embed, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        # === Pointer projection over FIXED vocabulary ===
        logits = torch.einsum('blh,bvh->blv', out, h_mp)  # (B, L_seq, 153)
        return logits


class FaceModel(nn.Module):
    """
    FIXED VERSION: Uses GLOBAL_MAX_EDGES_MODEL for fixed-size vocabulary.
    
    h_edge_enc is ALWAYS of shape (B, GLOBAL_MAX_EDGES + 3, d_model):
    - Indices 0 to GLOBAL_MAX_EDGES-1: edge embeddings (padded with zeros for unused)
    - Index GLOBAL_MAX_EDGES (150): NEW_FACE token
    - Index GLOBAL_MAX_EDGES+1 (151): EOS token
    - Index GLOBAL_MAX_EDGES+2 (152): SOS token
    
    This ensures tokens always have the same meaning across all samples and batches.
    """
    def __init__(self, max_num_vertices=GLOBAL_MAX_VERTICES_MODEL, max_num_edges=GLOBAL_MAX_EDGES_MODEL, max_seq_len=MAX_SEQ_LEN_F):
        super().__init__()
        d_model = TRANSFORMER_CONFIG['d_model']
        self.d_model = d_model
        self.max_E_capacity = max_num_edges    # FIXED: Always 150
        self.max_V_capacity = max_num_vertices # FIXED: Always 150

        # Special token values - FIXED positions in vocabulary
        self.new_face_token = FACE_NEW_FACE_TOKEN  # 150
        self.eos_token = FACE_EOS_TOKEN            # 151
        self.sos_token = FACE_SOS_TOKEN            # 152
        self.pad_token = PAD_TOKEN_F               # 153

        # Vertex embedding (same phi projection as used in EdgeModel, for vertex_tokens)
        self.Wx = nn.Embedding(64, 64)
        self.Wy = nn.Embedding(64, 64)
        self.Wz = nn.Embedding(64, 64)
        self.phi = nn.Linear(64 * 3, d_model)

        # Special token embeddings
        self.h_new_face = nn.Parameter(torch.randn(1, 1, d_model))
        self.h_eos = nn.Parameter(torch.randn(1, 1, d_model))
        self.h_sos = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embedding for face sequence
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(**TRANSFORMER_CONFIG)
        self.vertex_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.edge_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(**TRANSFORMER_CONFIG)
        self.face_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
    def forward(self, face_seq, vertex_tokens, edge_indices, image_embed):
        """
        Args:
            face_seq: (B, L_seq) sequence of tokens using FIXED vocabulary
                      - Edge indices: 0 to E_actual-1
                      - NEW_FACE: 150 (fixed)
                      - EOS: 151 (fixed)
                      - SOS: 152 (fixed)
                      - PAD: 153 (fixed)
            vertex_tokens: (B, V_actual, 3) batch of quantized vertex coordinates
            edge_indices: (B, E_actual, 3) edge vertex indices, padded with -1
            image_embed: (B, 256, d_model) image features
            
        Returns:
            logits: (B, L_seq, 153) - pointer distribution over FIXED vocabulary
        """
        B, L_seq = face_seq.shape
        device = face_seq.device
        
        V_actual = vertex_tokens.size(1)
        E_actual = edge_indices.size(1)

        # === Embed and encode vertices ===
        x_emb = self.Wx(vertex_tokens[..., 0])
        y_emb = self.Wy(vertex_tokens[..., 1])
        z_emb = self.Wz(vertex_tokens[..., 2])
        h_vert = self.phi(torch.cat([x_emb, y_emb, z_emb], dim=-1))  # (B, V_actual, d_model)
        h_vert_enc = self.vertex_encoder(h_vert)  # (B, V_actual, d_model)
        
        # =====================================================================
        # === OPTIMIZED: Vectorized edge embedding computation ===
        # OLD: Nested Python loops over batch and edges (SLOW!)
        # NEW: Single vectorized operation (FAST!)
        # =====================================================================
        
        # edge_indices: (B, E_actual, 3) where each edge has [v0, v1, v2]
        # v2 = -1 for line edges (only 2 vertices), valid for arc edges (3 vertices)
        
        # Step 1: Create mask for valid vertex positions in each edge
        # True where vertex index is valid (not -1)
        valid_mask = (edge_indices != -1)  # (B, E_actual, 3)
        
        # Step 2: Replace -1 with 0 for safe gathering (will be masked out later)
        safe_edge_indices = edge_indices.clone()
        safe_edge_indices[~valid_mask] = 0  # (B, E_actual, 3)
        
        # Step 3: Gather vertex embeddings for all edge positions at once
        # We need to gather from h_vert_enc (B, V_actual, d_model) using indices from safe_edge_indices
        
        # Prepare indices for gathering: (B, E_actual, 3) -> expand to (B, E_actual, 3, d_model)
        # For torch.gather on dim=1, we need indices of shape matching output
        
        # Gather for each vertex position (v0, v1, v2)
        # h_vert_enc: (B, V_actual, d_model)
        # We want to index into dim=1 (V_actual) using safe_edge_indices
        
        idx_v0 = safe_edge_indices[:, :, 0].unsqueeze(-1).expand(-1, -1, self.d_model)  # (B, E_actual, d_model)
        idx_v1 = safe_edge_indices[:, :, 1].unsqueeze(-1).expand(-1, -1, self.d_model)  # (B, E_actual, d_model)
        idx_v2 = safe_edge_indices[:, :, 2].unsqueeze(-1).expand(-1, -1, self.d_model)  # (B, E_actual, d_model)
        
        h_v0 = torch.gather(h_vert_enc, 1, idx_v0)  # (B, E_actual, d_model)
        h_v1 = torch.gather(h_vert_enc, 1, idx_v1)  # (B, E_actual, d_model)
        h_v2 = torch.gather(h_vert_enc, 1, idx_v2)  # (B, E_actual, d_model)
        
        # Step 4: Stack into (B, E_actual, 3, d_model)
        h_edge_verts = torch.stack([h_v0, h_v1, h_v2], dim=2)  # (B, E_actual, 3, d_model)
        
        # Step 5: Zero out invalid positions using the mask
        valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, E_actual, 3, 1)
        h_edge_verts_masked = h_edge_verts * valid_mask_expanded  # (B, E_actual, 3, d_model)
        
        # Step 6: Compute mean over valid vertices only
        # Sum the embeddings and divide by count of valid vertices
        num_valid = valid_mask.sum(dim=-1, keepdim=True).float().clamp(min=1)  # (B, E_actual, 1)
        h_edge_actual = h_edge_verts_masked.sum(dim=2) / num_valid  # (B, E_actual, d_model)
        
        # =====================================================================
        # === End of optimized section ===
        # =====================================================================
        
        # === Build FIXED-SIZE h_edge_enc ===
        # Pad edge embeddings to max_E_capacity (150)
        h_edge_padded = torch.zeros(B, self.max_E_capacity, self.d_model, device=device)
        h_edge_padded[:, :E_actual, :] = h_edge_actual  # Fill actual edges
        
        # Concatenate: [e0, e1, ..., e149, NEW_FACE(150), EOS(151), SOS(152)]
        h_all_memory = torch.cat([
            h_edge_padded,                              # indices 0-149
            self.h_new_face.expand(B, -1, -1),          # index 150
            self.h_eos.expand(B, -1, -1),               # index 151
            self.h_sos.expand(B, -1, -1)                # index 152
        ], dim=1)  # Shape: (B, 153, d_model) - ALWAYS this size
        
        # Encode the memory pool
        h_edge_enc = self.edge_encoder(h_all_memory)  # (B, 153, d_model)
        
        # === Build input embeddings for face_seq ===
        pos_ids = torch.arange(L_seq, device=device).unsqueeze(0).expand(B, L_seq)
        pos_embed = self.pos_emb(pos_ids)
        
        inp_embed_idx = face_seq.clone()
        pad_mask = (inp_embed_idx == self.pad_token)  # Identify PAD tokens (153)
        inp_embed_idx[pad_mask] = 0  # Temporarily map to 0 for gather
        
        inp_embed = torch.gather(h_edge_enc, 1, inp_embed_idx.unsqueeze(-1).expand(-1, -1, self.d_model))
        inp_embed[pad_mask] = 0.0
        inp_embed = inp_embed + pos_embed
        
        # Causal mask and padding mask
        tgt_mask = torch.triu(torch.ones(L_seq, L_seq, device=device, dtype=torch.bool), diagonal=1)
        tgt_key_padding_mask = (face_seq == self.pad_token)
        
        # === Transformer decoder forward pass ===
        if image_embed is None:
            zero_mem = torch.zeros(B, 1, self.d_model, device=device)
            out = self.face_decoder(tgt=inp_embed, memory=zero_mem, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        else:
            out = self.face_decoder(tgt=inp_embed, memory=image_embed, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        
        # === Pointer projection over FIXED vocabulary ===
        logits = torch.einsum('bld,bvd->blv', out, h_edge_enc)  # (B, L_seq, 153)
        return logits
