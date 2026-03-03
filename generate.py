import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import math
import traceback
from torchvision import transforms
import argparse
from PIL import Image
import os
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_Sewing
)
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Face
from OCC.Core.Geom import Geom_Plane
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.gce import gce_MakeCirc # For circle fitting from 3 points
from OCC.Core.GC import GC_MakeCylindricalSurface, GC_MakeConicalSurface # For primitive surface fitting

from OCC.Core.Geom import (
    Geom_Plane, Geom_SphericalSurface, Geom_CylindricalSurface,
    Geom_ConicalSurface, Geom_ToroidalSurface
)
from OCC.Core.BRepCheck import BRepCheck_Analyzer # For validating reconstructed shapes
from OCC.Extend.DataExchange import write_step_file # For saving generated models
from OCC.Display.OCCViewer import Viewer3d # For rendering generated models (used sequentially in main process only)
from OCC.Core.AIS import AIS_Shape # For rendering
from OCC.Core.TopAbs import TopAbs_FACE # For rendering/debugging

from models import ImageEncoder, VertexModel, EdgeModel, FaceModel # Import the neural network models

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

# Fixed vocabulary size for EdgeModel and FaceModel
EDGE_VOCAB_SIZE = GLOBAL_MAX_VERTICES_MODEL + 3  # 153 (0-149 vertices + NEW_EDGE + EOS + SOS)
FACE_VOCAB_SIZE = GLOBAL_MAX_EDGES_MODEL + 3     # 153 (0-149 edges + NEW_FACE + EOS + SOS)

# ========================
# SAMPLING & MASKING (For Inference/Generation, not directly for training loss)
# ========================
def masked_nucleus_sampling(logits, top_p=0.9, mask=None, eos_idx=None):
    """
    Nucleus sampling (Top-P sampling) with additional masking.
    
    FIXED: Uses eos_idx instead of pad_idx for fallback when all tokens are masked.
    
    Args:
        logits: (B, vocab_size) tensor of raw logits.
        top_p: Nucleus sampling threshold (default 0.9)
        mask: (B, vocab_size) boolean tensor. True means forbidden, False means allowed.
        eos_idx: The EOS token index, used as a fallback if all tokens are masked.
    
    Returns: 
        (B, 1) tensor of sampled token indices.

    """
    probs = F.softmax(logits, dim=-1) # Convert logits to probabilities

    # Apply hard mask: set forbidden token probabilities to 0.0
    if mask is not None:
        if mask.dim() == 1: # If mask is (vocab_size), expand to (B, vocab_size)
            mask = mask.unsqueeze(0).expand_as(probs).to(probs.device)
        probs = probs.masked_fill(mask, 0.0)

    # Handle cases where all probabilities become zero after masking
    # This prevents errors in `torch.multinomial` if `num_samples` (1) > number of non-zero probabilities.
    if torch.all(probs.sum(dim=-1) == 0):
        # FIXED: Return EOS token instead of PAD token to gracefully end generation
        # This prevents generating invalid tokens outside the vocabulary
        fallback_token = torch.full((logits.size(0), 1),
                                     fill_value=(eos_idx if eos_idx is not None else 0),
                                     dtype=torch.long, device=logits.device)
        print(f"  [WARN] All tokens masked! Forcing fallback to token {eos_idx} to end generation.")
        return fallback_token

    # Nucleus sampling logic
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Create a mask for tokens to be removed by nucleus sampling (cumulative prob > top_p)
    sorted_remove = cumulative_probs > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone() # Shift mask to the right to retain the last token
    sorted_remove[..., 0] = 0 # Ensure the highest probability token is always kept (even if cumulative > top_p)
    
    # Apply nucleus removal mask: set probabilities to 0.0 for removed tokens
    remove_indices_nucleus = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, sorted_indices, sorted_remove)
    probs_filtered_nucleus = probs.masked_fill(remove_indices_nucleus, 0.0)
    
    # Renormalize probabilities before sampling (optional, but good practice)
    # Add a small epsilon to denominator to prevent division by zero if all probs_filtered_nucleus become zero
    # (though the `if torch.all(probs.sum(dim=-1) == 0)` check should catch this before)
    probs_renormalized = probs_filtered_nucleus / (probs_filtered_nucleus.sum(dim=-1, keepdim=True) + 1e-9)
    
    # Sample from the renormalized probabilities
    sampled_tokens = torch.multinomial(probs_renormalized, num_samples=1)
    return sampled_tokens # (B, 1) tensor of sampled token indices

# ========================
# MASKING FUNCTIONS (For Inference/Generation)
# These generate per-step masks based on `prev_tokens_list` for nucleus sampling.
# ========================
def get_vertex_mask(prev_tokens_list, vocab_size=65):
    """
    Implements vertex masking rules from Appendix A.3, for z,y,x order.
    `prev_tokens_list`: Python list of previously generated tokens (including SOS).
    `vocab_size`: Max quantized coordinate value (63) + EOS (64). So 65.
    Returns: torch.BoolTensor (vocab_size + 2) indicating forbidden tokens (True = forbidden).
    """
    SOS = VERTEX_SOS_TOKEN # 65
    EOS = VERTEX_EOS_TOKEN # 64
    
    # Mask covers all possible token indices for VertexModel's output (0-63, EOS=64, SOS=65, PAD=66)
    mask = torch.zeros(vocab_size + 2, dtype=torch.bool)
    
    # Rule 1: Always forbid generating the <SOS> token after its initial placement.
    mask[SOS] = True
    
    t = len(prev_tokens_list) # Current sequence length including SOS
    if t == 1: # Only [SOS] is present. We are generating the first coordinate (z1).
        mask[EOS] = True # Cannot end sequence yet.
        return mask

    # `coord_to_generate` (0 for z, 1 for y, 2 for x) is based on number of REAL tokens generated.
    num_real_tokens_generated = t - 1
    coord_to_generate = num_real_tokens_generated % 3

    # Rule 2: Lexicographical sorting (z, y, x). New token must be >= previous coordinate in that slot.
    # Applies if we are generating subsequent vertices or subsequent coordinates within a vertex.
    if num_real_tokens_generated >= 3: # Have at least one full vertex (z1, y1, x1)
        if coord_to_generate == 0: # Generating a z-coordinate (z2, z3, etc.)
            prev_z = prev_tokens_list[-3] # The z-coordinate of the *previous* complete vertex
            mask[:prev_z] = True # New z must be >= previous z
            
        elif coord_to_generate == 1: # Generating a y-coordinate (y2, y3, etc.)
            prev_z_current_vertex = prev_tokens_list[-1] # This is the z-coord of the *current* vertex being built
            prev_z_prev_vertex = prev_tokens_list[-4]   # The z-coord of the *previous* complete vertex
            
            if prev_z_current_vertex == prev_z_prev_vertex: # If z is constant (same z-plane)...
                prev_y1 = prev_tokens_list[-3] # The y-coord of the previous complete vertex
                mask[:prev_y1] = True # New y must be >= previous y
        
        elif coord_to_generate == 2: # Generating an x-coordinate (x2, x3, etc.)
            prev_z_current_vertex = prev_tokens_list[-2] # z-coord of current vertex
            prev_z_prev_vertex = prev_tokens_list[-5]   # z-coord of previous vertex
            prev_y_current_vertex = prev_tokens_list[-1] # y-coord of current vertex
            prev_y_prev_vertex = prev_tokens_list[-4]   # y-coord of previous vertex
            
            if prev_z_current_vertex == prev_z_prev_vertex and \
               prev_y_current_vertex == prev_y_prev_vertex: # If z and y are constant...
                prev_x1 = prev_tokens_list[-3] # x-coord of previous vertex
                mask[:prev_x1] = True # New x must be >= previous x

    # Rule 3: EOS token validity
    # EOS is only allowed immediately after an x-coordinate (i.e., at the start of a new z-coordinate slot).
    # This implies that `num_real_tokens_generated` must be a multiple of 3.
    # So, if we are about to generate a y or x coordinate (`coord_to_generate` is 1 or 2), EOS is forbidden.
    if coord_to_generate != 0:
        mask[EOS] = True
        
    return mask

def get_edge_mask(prev_tokens_list, V_actual):
    """
    Implements edge masking rules for generation.
    
    FIXED VERSION: Uses global fixed vocabulary (153 tokens).
    - Indices 0-149: vertex positions (only 0 to V_actual-1 are valid)
    - Index 150: NEW_EDGE
    - Index 151: EOS
    - Index 152: SOS
    
    Args:
        prev_tokens_list: Python list of previously generated tokens (including SOS).
        V_actual: Actual number of vertices for the current model.
    Returns:
        torch.BoolTensor (153) indicating forbidden tokens (True = forbidden).
    """
    # FIXED token positions
    NEW_EDGE = EDGE_NEW_EDGE_TOKEN  # 150
    EOS = EDGE_EOS_TOKEN            # 151
    SOS = EDGE_SOS_TOKEN            # 152
    
    # FIXED vocabulary size
    mask = torch.zeros(EDGE_VOCAB_SIZE, dtype=torch.bool)  # Size 153

    # Rule 0: Forbid all non-existent vertices (V_actual to 149)
    # These vertex positions exist in the vocabulary but don't correspond to real vertices
    if V_actual < GLOBAL_MAX_VERTICES_MODEL:
        mask[V_actual:GLOBAL_MAX_VERTICES_MODEL] = True  # Forbid indices V_actual to 149

    # Rule 1: Always forbid generating the <SOS> token after its initial placement.
    mask[SOS] = True

    t = len(prev_tokens_list) # Current sequence length including SOS
    if t == 1: # Only [SOS] is present. We are generating the first token of the first edge.
        mask[EOS] = True      # Cannot end sequence with just SOS
        mask[NEW_EDGE] = True # Cannot start with NEW_EDGE
        return mask

    prev_token = prev_tokens_list[-1] # The last token that was generated

    # Rule 2: Cannot generate special token immediately after another special token.
    if prev_token == EOS or prev_token == NEW_EDGE:
        mask[EOS] = True      # Cannot repeat EOS
        mask[NEW_EDGE] = True # Cannot repeat NEW_EDGE
        mask[SOS] = True      # SOS already masked from Rule 1
    
    # Count how many vertex indices are currently in the edge being built.
    num_in_current_edge = 0
    for token_in_seq in reversed(prev_tokens_list):
        if token_in_seq in [NEW_EDGE, EOS, SOS]:
            break # Stop counting if we hit a special token
        num_in_current_edge += 1

    # Rule 3: Edge length constraints (2 or 3 vertices per edge).
    if num_in_current_edge < 2: # Need at least 2 vertices to complete an edge
        mask[EOS] = True      # Cannot end sequence
        mask[NEW_EDGE] = True # Cannot start new edge
    elif num_in_current_edge == 3: # If 3 vertices already, must end current edge.
        # Forbid all vertex indices [0, V_actual-1]. Must generate NEW_EDGE or EOS.
        mask[:V_actual] = True 
        mask[SOS] = True # SOS already masked

    # Rule 4: Lexicographical sorting within edges.
    # Tokenization: `sorted_edges = [tuple(sorted(e)) for e in edges]` then `tokens.extend(e)`.
    # This means vertex indices within an edge are sorted (u < v < w).
    if prev_token not in [NEW_EDGE, EOS, SOS]: # If previous token was a vertex index (not a special token)
        if num_in_current_edge > 0: # If we are currently building an edge (and prev_token is a vertex index in it)
            prev_actual_vertex_index = prev_tokens_list[-1]
            # Next vertex index must be > previous vertex index in the current edge
            mask[:prev_actual_vertex_index+1] = True 
    
    # Rule 5: Forbid duplicate vertices within the current edge.
    current_edge_verts = []
    for token in reversed(prev_tokens_list):
        if token in [NEW_EDGE, EOS, SOS]:
            break
        current_edge_verts.append(token)
    
    for vert_idx in current_edge_verts:
        if vert_idx < EDGE_VOCAB_SIZE:  # Safety check
            mask[vert_idx] = True # Forbid any vertex that's already in the edge
    
    # SAFETY CHECK: Ensure at least one valid option exists
    # If all vertices and NEW_EDGE are masked, allow EOS
    all_vertices_masked = mask[:V_actual].all() if V_actual > 0 else True
    new_edge_masked = mask[EDGE_NEW_EDGE_TOKEN]
    
    if all_vertices_masked and new_edge_masked:
        # Last resort: ensure EOS is available to end generation gracefully
        mask[EDGE_EOS_TOKEN] = False
        print("  [WARN] All generation options masked! Allowing EOS as last resort.")
    
    # At the end, check if everything is masked
    # If so, relax constraints to allow generation to continue
    if mask[:EDGE_VOCAB_SIZE].all():  # Everything forbidden
        # Reset mask and only apply basic constraints
        mask = torch.zeros(EDGE_VOCAB_SIZE, dtype=torch.bool)
        mask[V_actual:GLOBAL_MAX_VERTICES_MODEL] = True  # Still forbid non-existent vertices
        mask[EDGE_SOS_TOKEN] = True  # Still forbid SOS
    
    return mask

def get_face_mask(prev_tokens_list, E_actual, edge_counts):
    """
    Implements face masking rules for generation.
    
    FIXED VERSION: Uses global fixed vocabulary (153 tokens).
    - Indices 0-149: edge positions (only 0 to E_actual-1 are valid)
    - Index 150: NEW_FACE
    - Index 151: EOS
    - Index 152: SOS
    
    Args:
        prev_tokens_list: Python list of previously generated tokens (including SOS).
        E_actual: Actual number of edges for the current model.
        edge_counts: Dictionary tracking how many times each edge index has been used.
    Returns:
        torch.BoolTensor (153) indicating forbidden tokens (True = forbidden).
    """
    # FIXED token positions
    NEW_FACE = FACE_NEW_FACE_TOKEN  # 150
    EOS = FACE_EOS_TOKEN            # 151
    SOS = FACE_SOS_TOKEN            # 152
    
    # FIXED vocabulary size
    mask = torch.zeros(FACE_VOCAB_SIZE, dtype=torch.bool)  # Size 153
    
    # Rule 0: Forbid all non-existent edges (E_actual to 149)
    # These edge positions exist in the vocabulary but don't correspond to real edges
    if E_actual < GLOBAL_MAX_EDGES_MODEL:
        mask[E_actual:GLOBAL_MAX_EDGES_MODEL] = True  # Forbid indices E_actual to 149

    # Rule 1: Always forbid generating the <SOS> token.
    mask[SOS] = True

    t = len(prev_tokens_list) # Current sequence length including SOS
    if t == 1: # Only [SOS] is present. We are generating the first token of the first face.
        mask[EOS] = True      # Cannot end sequence with just SOS
        mask[NEW_FACE] = True # Cannot start with NEW_FACE
        return mask

    prev_token = prev_tokens_list[-1]

    # Rule 2: Cannot generate special token immediately after another special token.
    if prev_token == EOS or prev_token == NEW_FACE:
        mask[EOS] = True      # Cannot repeat EOS
        mask[NEW_FACE] = True # Cannot repeat NEW_FACE
        mask[SOS] = True      # SOS already masked

    # Count how many edge indices are currently in the face being built.
    num_in_current_face = 0
    for token_in_seq in reversed(prev_tokens_list):
        if token_in_seq in [NEW_FACE, EOS, SOS]:
            break
        num_in_current_face += 1

    # Rule 3: Face length constraints (at least 2 edges for a valid face).
    if num_in_current_face < 2:
        mask[EOS] = True      # Cannot end sequence
        mask[NEW_FACE] = True # Cannot start new face
    
    # Rule 4: Edge index use limit (an edge can be used at most twice globally across faces).
    # This rule is critical for valid manifold meshes. `edge_counts` comes from `generate_faces`.
    for i in range(E_actual): # Iterate through all possible edge indices
        if edge_counts.get(i, 0) >= 2: # If this edge has already been used twice
            mask[i] = True  # Forbid using it again.

    # Rule 5: Lexicographical sorting for edge indices within a face.
    # Tokenization: `sorted_faces = [sorted(f) for f in faces]`, then `tokens.extend(face)`.
    # This means edge indices within a face are sorted.
    if prev_token not in [NEW_FACE, EOS, SOS]: # If previous token was an edge index
        if num_in_current_face > 0: # If we are currently building a face (and prev_token is an edge index in it)
            prev_actual_edge_idx = prev_tokens_list[-1]
            # Next edge index must be >= previous edge index in the current face
            mask[:prev_actual_edge_idx] = True 
    
    # SAFETY CHECK: Ensure at least one valid option exists
    all_edges_masked = mask[:E_actual].all() if E_actual > 0 else True
    new_face_masked = mask[FACE_NEW_FACE_TOKEN]
    
    if all_edges_masked and new_face_masked:
        # Last resort: ensure EOS is available
        mask[FACE_EOS_TOKEN] = False
        print("  [WARN] All generation options masked! Allowing EOS as last resort.")
    
    return mask

# ========================
# SURFACE TYPE DETECTION (Used in reconstruction for specific surface fitting)
# ========================
def detect_surface_type(face_edge_indices_list, vertices_raw_xyz, edges_orig_tuples):
    """
    Basic heuristic for surface type detection for OCP reconstruction.
    Args:
        face_edge_indices_list (list): List of edge indices for the current face.
        vertices_raw_xyz (np.array): Raw (x,y,z) float coordinates.
        edges_orig_tuples (list): List of (u,v) or (u,v,w) vertex index tuples.
    Returns:
        str: "plane", "sphere", or "cylinder"
    """
    arc_edges_vertex_tuples = []
    for edge_idx in face_edge_indices_list:
        # Ensure edge_idx is valid and it corresponds to a 3-vertex arc (circular arc)
        if 0 <= edge_idx < len(edges_orig_tuples) and len(edges_orig_tuples[edge_idx]) == 3:
            arc_edges_vertex_tuples.append(edges_orig_tuples[edge_idx])
            
    if not arc_edges_vertex_tuples:
        return "plane" # If no arcs, it's likely a planar face
    
    # Sphere test: all 3-point arcs must lie on one common center & radius
    centers, radii = [], []
    try:
        for trip_v_indices in arc_edges_vertex_tuples:
            p1, p2, p3 = (gp_Pnt(float(vertices_raw_xyz[i][0]), float(vertices_raw_xyz[i][1]), float(vertices_raw_xyz[i][2])) for i in trip_v_indices)
            circ_maker = gce_MakeCirc(p1, p2, p3)
            if not circ_maker.IsDone(): raise ValueError("Circle creation failed")
            circ = circ_maker.Value()
            centers.append((circ.Location().X(), circ.Location().Y(), circ.Location().Z()))
            radii.append(circ.Radius())
        
        # Check if radii are consistent and centers are clustered
        if radii and len(radii) > 0 and (max(radii) - min(radii) < 1e-3):
            if len(centers) > 1:
                # Max distance between any two centers (simple clustering check)
                max_center_dist = 0.0
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        max_center_dist = max(max_center_dist, math.dist(centers[i], centers[j]))
                if max_center_dist < 1e-3:
                    return "sphere"
            else: # Only one arc, can be considered spherical if its part of a sphere
                return "sphere"
    except Exception: # gce_MakeCirc can fail if points are collinear or for other geometric reasons
        pass
    
    # Cylinder test: exactly two arcs, identical radius & parallel axes
    if len(arc_edges_vertex_tuples) == 2:
        try:
            circ1_maker = gce_MakeCirc(*(gp_Pnt(float(vertices_raw_xyz[i][0]), float(vertices_raw_xyz[i][1]), float(vertices_raw_xyz[i][2])) for i in arc_edges_vertex_tuples[0]))
            circ2_maker = gce_MakeCirc(*(gp_Pnt(float(vertices_raw_xyz[i][0]), float(vertices_raw_xyz[i][1]), float(vertices_raw_xyz[i][2])) for i in arc_edges_vertex_tuples[1]))
            if circ1_maker.IsDone() and circ2_maker.IsDone():
                circ1 = circ1_maker.Value()
                circ2 = circ2_maker.Value()
                
                if (abs(circ1.Radius() - circ2.Radius()) < 1e-3):
                    dir1 = circ1.Axis().Direction()
                    dir2 = circ2.Axis().Direction()
                    dot = abs(dir1.Dot(dir2)) # Dot product of directions to check parallelism
                    if dot > 0.999: # If dot product close to 1 (or -1), axes are parallel
                        return "cylinder"
        except Exception: # gce_MakeCirc can fail
            pass
            
    return "plane"  # Fallback to plane if no specific curved surface detected

# ========================
# DEQUANTIZATION (No changes needed)
# ========================
def dequantize_vertices(V_q_xyz, V_min_orig, V_max_orig):
    """
    Reverse quantization: maps [0..63] tokens back to real-world coordinates.
    Args:
        V_q_xyz (np.array): (N, 3) numpy array of quantized (int64) XYZ coordinates.
        V_min_orig (np.array): (3,) numpy array of original min (X,Y,Z) floats.
        V_max_orig (np.array): (3,) numpy array of original max (X,Y,Z) floats.
    Returns:
        np.array: Dequantized (N, 3) numpy array of float coordinates.
    """
    V_q = np.array(V_q_xyz, dtype=np.float32) # Convert to float32 for calculations
    
    V_range = V_max_orig - V_min_orig
    V_range[V_range == 0] = 1e-6 # Add epsilon to prevent division by zero
    
    V_deq = (V_q / 63.0) * V_range + V_min_orig
    return V_deq # Returns numpy array of floats

# ========================
# FACE TRIMMING HELPER (OCP-heavy, prone to errors for invalid inputs)
# ========================
def make_trimmed_face(geom_edges_list, face_edge_indices_list, vertices_raw_xyz, edges_orig_tuples):
    """
    Attempts to create a TopoDS_Face from a list of edges, fitting a surface if necessary.
    Args:
        geom_edges_list (list): List of `TopoDS_Edge` objects.
        face_edge_indices_list (list): Indices into `geom_edges_list` for the current face.
        vertices_raw_xyz (np.array): Raw (x,y,z) float coordinates of vertices.
        edges_orig_tuples (list): Original list of (u,v) or (u,v,w) vertex index tuples.
    Returns:
        TopoDS_Face or None if creation fails.
    """
    print(f"\n--- Attempting to build face with {len(face_edge_indices_list)} edges: {face_edge_indices_list} ---")
    
    try:
        # 1) Wire up the boundary from the provided geometric edges
        wb = BRepBuilderAPI_MakeWire()
        for ei in face_edge_indices_list:
            if 0 <= ei < len(geom_edges_list) and not geom_edges_list[ei].IsNull():
                wb.Add(geom_edges_list[ei])
        
        if not wb.IsDone():
            print("  [FAIL] BRepBuilderAPI_MakeWire failed to build the wire.")
            return None
            
        wire = wb.Wire()
        if wire.IsNull():
            print("  [FAIL] Wire object is null after building.")
            return None
        print("  [OK] Successfully created wire.")

        # 2) First, always try to build a simple planar face. This is the most common case.
        #    The `only_plane=True` flag makes it fail quickly if the wire is not planar.
        face_builder = BRepBuilderAPI_MakeFace(wire, True) 
        if face_builder.IsDone():
            print("  [OK] Successfully created a PLANAR face.")
            return face_builder.Face()
        
        print("  [INFO] Wire is not planar. Attempting to fit a curved surface.")

        # 3) If planar fails, determine the specific surface type for curved faces
        surface_type = detect_surface_type(face_edge_indices_list, vertices_raw_xyz, edges_orig_tuples)
        print(f"  [INFO] Detected surface type: {surface_type}")
        
        surf = None
        
        # This whole block is a heuristic and prone to failure.
        try:
            pts = []
            for ei in face_edge_indices_list[:4]:
                if 0 <= ei < len(geom_edges_list) and not geom_edges_list[ei].IsNull():
                    c, umin, umax = BRep_Tool().Curve(geom_edges_list[ei])
                    if c and not c.IsNull():
                        pts.append(c.Value((umin + umax) / 2.0))
            
            if surface_type == "cylinder" and len(pts) >= 3:
                cy_maker = GC_MakeCylindricalSurface(pts[0], pts[1], pts[2])
                if cy_maker.IsDone():
                    surf = cy_maker.Value().Cylinder() # For a cylinder primitive
        except Exception as construction_error:
            print(f"  [WARN] Failed to construct primitive for '{surface_type}'. Error: {construction_error}")
            surf = None # Ensure surf is None if construction fails

        # 4) If we successfully created a curved surface, try to trim it with the wire.
        if surf is not None and not surf.IsNull():
            print("  [INFO] Attempting to trim the curved surface with the wire.")
            face_builder_curved = BRepBuilderAPI_MakeFace(surf, wire)
            if face_builder_curved.IsDone():
                print("  [OK] Successfully created a CURVED face.")
                return face_builder_curved.Face()
            else:
                print("  [FAIL] Failed to trim the curved surface. This often happens if the wire does not lie on the surface.")

        # 5) FINAL FALLBACK: If all else fails, create a face from the wire without checking for planarity.
        #    This is a last resort and may produce a valid but non-standard surface.
        print("  [INFO] All other methods failed. Attempting final fallback: MakeFace from wire directly.")
        final_attempt_builder = BRepBuilderAPI_MakeFace(wire, False)
        if final_attempt_builder.IsDone():
            print("  [OK] Final fallback succeeded.")
            return final_attempt_builder.Face()
        else:
            print("  [FAIL] All face creation methods have failed for this wire.")
            return None

    except Exception:
        # This will catch any unexpected Python error and print the full traceback
        print("\nFATAL ERROR occurred inside make_trimmed_face function:")
        traceback.print_exc()
        return None

# ========================
# B-REP RECONSTRUCTION (OCP-heavy, prone to errors for invalid inputs)
# ========================
def fit_circle(p1, p2, p3):
    """
    Creates a circle from three points using OCP's gce_MakeCirc.
    Args:
        p1, p2, p3 (gp_Pnt): Points on the circle.
    Returns:
        gp_Circ or None if creation fails.
    """
    try:
        circ_maker = gce_MakeCirc(p1, p2, p3)
        return circ_maker.Value() if circ_maker.IsDone() else None
    except Exception as e:
        return None

def indexed_to_brep(vertices_raw_xyz, edges_orig_tuples, faces_orig_lists):
    """
    Converts indexed B-rep data (vertices, edges, faces) into a TopoDS_Shape using OCP.
    Args:
        vertices_raw_xyz (np.array): Dequantized (N, 3) numpy array of float coordinates.
        edges_orig_tuples (list): List of (u,v) or (u,v,w) vertex index tuples.
        faces_orig_lists (list): List of lists of edge indices.
    Returns:
        TopoDS_Shape or None if reconstruction fails at any critical step.
    """
    try:
        # 1) Create geometric vertices (gp_Pnt) from raw float coordinates
        geom_verts = []
        if not vertices_raw_xyz.shape[0] > 0:
            print("[FAIL] No vertices provided to indexed_to_brep.")
            return None # No vertices
        for i, v_coords in enumerate(vertices_raw_xyz):
            # Ensure coordinates are floats as expected by gp_Pnt
            geom_verts.append(gp_Pnt(float(v_coords[0]), float(v_coords[1]), float(v_coords[2])))
            print(f"[OK] Created {len(geom_verts)} geometric vertices.")


        # 2) Build geometric edges (TopoDS_Edge) (lines vs arcs)
        geom_edges = []
        if not edges_orig_tuples: 
            print("[FAIL] No edges provided to indexed_to_brep.")
            return None
        for i, e_tuple in enumerate(edges_orig_tuples):
            edge_obj = None
            if len(e_tuple) == 2: # Line segment (vertex_idx_1, vertex_idx_2)
                if e_tuple[0] < len(geom_verts) and e_tuple[1] < len(geom_verts):
                    edge_obj = BRepBuilderAPI_MakeEdge(geom_verts[e_tuple[0]], geom_verts[e_tuple[1]]).Edge()
            elif len(e_tuple) == 3: # Circular arc (vertex_idx_1, vertex_idx_2, vertex_idx_3)
                if e_tuple[0] < len(geom_verts) and e_tuple[1] < len(geom_verts) and e_tuple[2] < len(geom_verts):
                    circ = fit_circle(geom_verts[e_tuple[0]], geom_verts[e_tuple[1]], geom_verts[e_tuple[2]])
                    if circ is not None:
                        edge_obj = BRepBuilderAPI_MakeEdge(circ).Edge()
            
            if edge_obj and not edge_obj.IsNull():
                geom_edges.append(edge_obj)
            else:
                print(f"  [WARN] Failed to create geometric edge for index {i} with vertices {e_tuple}.")
        print(f"[OK] Created {len(geom_edges)} geometric edges.")
        
        if not geom_edges: # No valid geometric edges were created
            print("[FAIL] No valid geometric edges could be created.")
            return None

        # 3) Build faces (TopoDS_Face) using `make_trimmed_face` helper
        sewer = BRepBuilderAPI_Sewing() # Used to stitch faces into a single solid/shell
        if not faces_orig_lists:
            print("[FAIL] No faces provided to indexed_to_brep.")
            return None

        num_faces_added_to_sewer = 0
        for i, fids_list in enumerate(faces_orig_lists): # fids_list is list of edge indices for one face
            print(f"\nCalling make_trimmed_face for face {i}...")
            # Pass all original data needed by make_trimmed_face for surface detection
            face_obj = make_trimmed_face(geom_edges, fids_list, vertices_raw_xyz, edges_orig_tuples)
            if face_obj and not face_obj.IsNull():
                sewer.Add(face_obj)
                num_faces_added_to_sewer += 1
            else:
                print(f"  [FAIL] make_trimmed_face returned None for face {i}.")
        print(f"\n[INFO] Added {num_faces_added_to_sewer} valid faces to the sewer.")
        if num_faces_added_to_sewer == 0:
            return None

        sewer.Perform() # Attempt to sew the faces into a coherent shape
        
        sewed_shape = sewer.SewedShape() # The result of sewing (can be solid, shell, compound)
        if sewed_shape.IsNull():
            print("[FAIL] Sewed shape is null after sewing operation.")
            return None
        print("[OK] Sewing operation completed.")
        # Final validation of the reconstructed shape using BRepCheck_Analyzer
        analyzer = BRepCheck_Analyzer(sewed_shape)
        if not analyzer.IsValid():
            print("[FAIL] BRepCheck_Analyzer reported the final sewed shape is not valid.")
            return None # Or return the shape anyway for debugging
        print("[OK] Final shape is valid.")
            
        return sewed_shape # Return the final TopoDS_Shape
    except Exception as e:
        print("\nFATAL ERROR occurred inside indexed_to_brep function:")
        traceback.print_exc() # This will print the full, detailed error
        return None

# ========================
# GENERATION PIPELINE (For Inference/Evaluation)
# These functions implement the autoregressive generation process.
# ========================
def generate_vertices(img_embed_tensor, model, device, V_min_norm_range=None, V_max_norm_range=None, max_seq_len=MAX_SEQ_LEN_V, top_p=0.9):
    """
    Autoregressive vertex generation.
    Args:
        img_embed_tensor (torch.Tensor): Image embedding (1, 256, d_model).
        model (VertexModel): The trained VertexModel.
        device (torch.device): Device to run generation on.
        V_min_norm_range (np.array, optional): Original min XYZ for dequantization.
        V_max_norm_range (np.array, optional): Original max XYZ for dequantization.
        max_seq_len (int): Maximum length of sequence to generate.
        top_p (float): Nucleus sampling probability threshold.
    Returns:
        np.array: Quantized (0-63) XYZ vertices as a numpy array, shape (N, 3).
                  These are suitable for feeding to EdgeModel/FaceModel.
    """
    current_seq_list = [model.sos_token] # Start with SOS token (Python list)
    print("Starting vertex generation...")
    print("SOS token: ", model.sos_token)
    
    quantized_vertices_zyx_list = [] # Stores generated [z, y, x] quantized integer tuples
    
    # Loop max_seq_len times (including SOS/EOS and padding for generated sequence)
    with torch.no_grad(): # Generation does not need gradients
        for step in range(1, max_seq_len): # Loop for subsequent tokens (after SOS)
            seq_tensor = torch.tensor([current_seq_list], dtype=torch.long, device=device) # (1, L_current)
            print("Step:", step, "Current Seq:", current_seq_list)
            # Get logits for the next token from the model
            logits = model(seq_tensor, img_embed_tensor)[:, -1, :] # (1, vocab_size + 2)
            print("Logits:", logits)

            # Get the mask for the current generation context (based on current_seq_list)
            mask = get_vertex_mask(current_seq_list, vocab_size=model.vocab_size) # (vocab_size + 2) boolean tensor
            
            # Nucleus sampling to pick the next token, applying the generated mask.
            # Pass model.pad_token as fallback if sampling fails (all masked).
            next_token_tensor = masked_nucleus_sampling(logits, top_p, mask=mask.unsqueeze(0).to(device), eos_idx=model.eos_token)
            next_token = next_token_tensor.item()
            # Validate token is in valid range (0-65 for vertices)
            if next_token > model.sos_token:  # SOS=65 is highest valid token
                print(f"  [ERROR] Generated invalid vertex token {next_token}! Forcing EOS to terminate.")
                next_token = model.eos_token
            print("Next token: ", next_token)

            # Stop generation if EOS token is predicted
            if next_token == model.eos_token:
                break
            
            current_seq_list.append(next_token) # Add the generated token to the sequence
                
            # Check if a full vertex (z, y, x) triplet is complete.
            num_real_tokens_generated = len(current_seq_list) - 1 # Exclude SOS token
            if num_real_tokens_generated > 0 and num_real_tokens_generated % 3 == 0:
                # The last 3 tokens are the [z, y, x] coordinates of the newly generated vertex
                zyx_tokens = current_seq_list[-3:]
                quantized_vertices_zyx_list.append(zyx_tokens)
                print("The triplet is (z,y,x): ", zyx_tokens)

    if not quantized_vertices_zyx_list:
        return np.array([], dtype=np.int64).reshape(0,3) # Return empty numpy array (0,3)
    
    # Convert list of [z, y, x] integer lists to numpy array
    quantized_vertices_zyx_np = np.array(quantized_vertices_zyx_list, dtype=np.int64)

    # Reorder columns from ZYX to XYZ for consistent representation and subsequent models/dequantization.
    quantized_vertices_xyz_np = quantized_vertices_zyx_np[:, [2, 1, 0]]
    print("Final vertices (x,y,z) are: ", quantized_vertices_xyz_np)
    # Return the quantized XYZ array. Dequantization to float coordinates happens at the end for B-Rep.
    return quantized_vertices_xyz_np 

def generate_edges(vertices_quantized_xyz, img_embed_tensor, model, device, max_seq_len=MAX_SEQ_LEN_E, top_p=0.9):
    """
    Autoregressive edge generation.
    
    FIXED VERSION: Uses global fixed vocabulary (153 tokens).
    - Token 150: NEW_EDGE (fixed)
    - Token 151: EOS (fixed)
    - Token 152: SOS (fixed)
    
    Args:
        vertices_quantized_xyz (np.array): Quantized (0-63) XYZ vertices from generate_vertices.
        img_embed_tensor (torch.Tensor): Image embedding.
        model (EdgeModel): The trained EdgeModel.
        device (torch.device): Device to run generation on.
        max_seq_len (int): Maximum length of sequence to generate.
        top_p (float): Nucleus sampling probability threshold.
    Returns:
        list: List of (u,v) or (u,v,w) tuples of vertex indices.
    """
    if vertices_quantized_xyz.shape[0] == 0:
        return [] # Cannot generate edges without vertices

    if vertices_quantized_xyz.min() < 0 or vertices_quantized_xyz.max() > 63:
        print(f"WARN(generate_edges): Received invalid vertex tokens from generate_vertices.")
        print(f"Min value: {vertices_quantized_xyz.min()}, Max value: {vertices_quantized_xyz.max()}")
        # Forcibly clamp the values to the valid range [0, 63] to prevent a crash.
        vertices_quantized_xyz = np.clip(vertices_quantized_xyz, 0, 63)
        print("Values have been clamped to [0, 63].")

    V_actual = vertices_quantized_xyz.shape[0] # Actual number of generated vertices
    print("Generating edges with actual number of generated vertices =", V_actual)
    
    # Convert quantized vertices to tensor for EdgeModel input
    vertex_tokens_for_model = torch.tensor(vertices_quantized_xyz, dtype=torch.long, device=device) # (V_actual, 3)
    
    # FIXED: Pad vertex tokens to GLOBAL_MAX_VERTICES_MODEL (150) as expected by the model
    # vertex_tokens_padded = torch.zeros(GLOBAL_MAX_VERTICES_MODEL, 3, dtype=torch.long, device=device)
    # vertex_tokens_padded[:V_actual, :] = vertex_tokens_for_model
    
    # FIXED: Use global fixed token values
    sos_token = EDGE_SOS_TOKEN      # 152
    eos_token = EDGE_EOS_TOKEN      # 151
    new_edge_token = EDGE_NEW_EDGE_TOKEN  # 150
    
    print(f"FIXED tokens: SOS={sos_token}, EOS={eos_token}, NEW_EDGE={new_edge_token}")
    
    current_seq_list = [sos_token]  # Start with FIXED SOS token (152)
    generated_edges_tuples = []
    current_edge_tuple_building = [] # Temporarily stores vertex indices for the current edge being built

    # FIX 1: Track already-generated edges to prevent duplicates
    generated_edge_set = set()  # Track (min_v, max_v) pairs in canonical form

    with torch.no_grad(): # Generation does not need gradients
        for step in range(1, max_seq_len): # Loop for subsequent tokens (after SOS)
            seq_tensor = torch.tensor([current_seq_list], dtype=torch.long, device=device) # (1, L_current)
            print("Step:", step, "Current Seq:", current_seq_list)
            
            # Pass the sequence, PADDED vertex_tokens, and image_embed to the EdgeModel
            # Model now expects (B, 150, 3) vertex tokens
            # logits = model(seq_tensor, vertex_tokens_padded.unsqueeze(0), img_embed_tensor)[:, -1, :]  # (1, 153)
            # Pass UN-PADDED vertex tokens - model handles padding
            logits = model(seq_tensor, vertex_tokens_for_model.unsqueeze(0), img_embed_tensor)[:, -1, :]
            # print("Logits shape:", logits.shape)
            print("Logits:", logits)
            # Get mask for current context using FIXED vocabulary
            mask = get_edge_mask(current_seq_list, V_actual)  # (153,) mask with non-existent vertices forbidden
            
            # FIX 1: Check if current edge would be a duplicate
            # For 2-vertex edges: check sorted tuple
            # For 3-vertex edges: we need to check after the edge is complete (can't know if duplicate yet)
            if len(current_edge_tuple_building) == 2:
                # Create canonical form for 2-vertex edge (sorted)
                edge_canonical = tuple(sorted(current_edge_tuple_building))
                if edge_canonical in generated_edge_set:
                    print(f"  [WARN] 2-vertex edge {edge_canonical} already exists! Forcing NEW_EDGE or EOS.")
                    # Mask all vertex indices to force NEW_EDGE or EOS
                    # This prevents adding a 3rd vertex that would make this a different edge
                    mask[:V_actual] = True
            elif len(current_edge_tuple_building) == 3:
                # For 3-vertex arcs, check if this exact ordered triple exists
                # Also check reverse order since (a,b,c) and (c,b,a) represent same arc
                edge_tuple = tuple(current_edge_tuple_building)
                edge_reversed = tuple(reversed(current_edge_tuple_building))
                if edge_tuple in generated_edge_set or edge_reversed in generated_edge_set:
                    print(f"  [WARN] 3-vertex arc {edge_tuple} already exists! Forcing NEW_EDGE or EOS.")
                    # Mask all vertex indices to force NEW_EDGE or EOS
                    mask[:V_actual] = True

            # FIX 2: Use GREEDY selection for first vertex of new edge, nucleus sampling otherwise
            # if len(current_edge_tuple_building) == 0:
            #     # First vertex of a new edge - use GREEDY (argmax) selection
            #     print("  [INFO] First vertex of edge - using GREEDY selection")
            #     # Apply mask
            #     logits_masked = logits.clone()
            #     logits_masked[0, mask] = -float('inf')
            #     next_token = torch.argmax(logits_masked, dim=-1).item()
            #     print(f"  [GREEDY] Selected token {next_token} with logit {logits[0, next_token].item():.4f}")
            # else:
            #     # Subsequent tokens - use nucleus sampling
            #     print("  [INFO] Subsequent token - using NUCLEUS sampling")
            #     next_token_tensor = masked_nucleus_sampling(logits, top_p, mask=mask.unsqueeze(0).to(device), eos_idx=EDGE_EOS_TOKEN)
            #     next_token = next_token_tensor.item()
            #     # Validate token is in valid range
            #     if next_token >= EDGE_VOCAB_SIZE - 1:  # Must be 0-152 (not 153)
            #         print(f"  [ERROR] Generated invalid token {next_token} (outside vocab 0-{EDGE_VOCAB_SIZE-2})! Forcing EOS to terminate.")
            #         next_token = EDGE_EOS_TOKEN
            # Remove greedy first-vertex selection - use nucleus sampling consistently
            next_token_tensor = masked_nucleus_sampling(logits, top_p, mask=mask.unsqueeze(0).to(device), eos_idx=PAD_TOKEN_E)
            next_token = next_token_tensor.item()
            print("Next token: ", next_token)
            
            # # Check for EOS or NEW_EDGE tokens to finalize current edge
            # if next_token == eos_token:  # FIXED: 151
            #     print("EOS token generated, finalizing edge generation.")
            #     if 2 <= len(current_edge_tuple_building) <= 3:
            #          generated_edges_tuples.append(tuple(current_edge_tuple_building))
            #          print("Final edge added: ", tuple(current_edge_tuple_building))
            #     break # Stop generation
                
            # if next_token == new_edge_token:  # FIXED: 150
            #     # Finalize the current edge if it's valid (2 or 3 vertices)
            #     if 2 <= len(current_edge_tuple_building) <= 3:
            #         generated_edges_tuples.append(tuple(current_edge_tuple_building))
            #         print("Edge added: ", tuple(current_edge_tuple_building))
            #     current_edge_tuple_building = [] # Reset for next edge
            # else: # It's a vertex index (0 to V_actual-1)
            #     current_edge_tuple_building.append(next_token)
            
            # [CHANGED - Feb 4,2026] Check for EOS or NEW_EDGE tokens to finalize current edge
            if next_token == eos_token:  # FIXED: 151
                print("EOS token generated, finalizing edge generation.")
                if 2 <= len(current_edge_tuple_building) <= 3:
                    # Check if this edge is a duplicate before adding
                    is_duplicate = False
                    if len(current_edge_tuple_building) == 2:
                        edge_canonical = tuple(sorted(current_edge_tuple_building))
                        is_duplicate = edge_canonical in generated_edge_set
                    else:  # len == 3
                        edge_canonical = tuple(sorted(current_edge_tuple_building))
                        edge_reversed = tuple(reversed(edge_canonical))
                        is_duplicate = edge_canonical in generated_edge_set or edge_reversed in generated_edge_set
                    
                    if not is_duplicate:
                        generated_edges_tuples.append(edge_canonical)
                        generated_edge_set.add(edge_canonical)
                        if len(edge_canonical) == 3:
                            generated_edge_set.add(tuple(reversed(edge_canonical)))
                        print("Final edge added: ", edge_canonical)
                    else:
                        print(f"  [SKIP] Duplicate edge {edge_canonical} not added to final list.")
                break # Stop generation
                
            if next_token == new_edge_token:  # FIXED: 150
                # Finalize the current edge if it's valid (2 or 3 vertices)
                if 2 <= len(current_edge_tuple_building) <= 3:
                    # Check if this edge is a duplicate before adding
                    is_duplicate = False
                    if len(current_edge_tuple_building) == 2:
                        edge_canonical = tuple(sorted(current_edge_tuple_building))
                        is_duplicate = edge_canonical in generated_edge_set
                    else:  # len == 3
                        edge_canonical = tuple(current_edge_tuple_building)
                        edge_reversed = tuple(reversed(edge_canonical))
                        is_duplicate = edge_canonical in generated_edge_set or edge_reversed in generated_edge_set
                    
                    if not is_duplicate:
                        generated_edges_tuples.append(edge_canonical)
                        generated_edge_set.add(edge_canonical)
                        if len(edge_canonical) == 3:
                            generated_edge_set.add(tuple(reversed(edge_canonical)))
                        print("Edge added: ", edge_canonical)
                    else:
                        print(f"  [SKIP] Duplicate edge {edge_canonical} not added to final list.")
                        
                current_edge_tuple_building = [] # Reset for next edge
            else: # It's a vertex index (0 to V_actual-1)
                current_edge_tuple_building.append(next_token)
            
            current_seq_list.append(next_token) # Add token to sequence for next step

    return generated_edges_tuples

def generate_faces(vertices_quantized_xyz, edges_orig_tuples, img_embed_tensor, model, device, max_seq_len=MAX_SEQ_LEN_F, top_p=0.9):
    """
    Autoregressive face generation.
    
    FIXED VERSION: Uses global fixed vocabulary (153 tokens).
    - Token 150: NEW_FACE (fixed)
    - Token 151: EOS (fixed)
    - Token 152: SOS (fixed)
    
    Args:
        vertices_quantized_xyz (np.array): Quantized (0-63) XYZ vertices.
        edges_orig_tuples (list): List of (u,v) or (u,v,w) tuples of vertex indices from generate_edges.
        img_embed_tensor (torch.Tensor): Image embedding.
        model (FaceModel): The trained FaceModel.
        device (torch.device): Device to run generation on.
        max_seq_len (int): Maximum length of sequence to generate.
        top_p (float): Nucleus sampling probability threshold.
    Returns:
        list: List of lists of edge indices for generated faces.
    """
    if not edges_orig_tuples:
        return [] # Cannot generate faces without edges
    
    V_actual = vertices_quantized_xyz.shape[0] # Number of generated vertices
    E_actual = len(edges_orig_tuples) # Number of generated edges
    
    # Convert quantized vertices to tensor for FaceModel input
    vertex_tokens_for_model = torch.tensor(vertices_quantized_xyz, dtype=torch.long, device=device) # (V_actual, 3)
    
    # FIXED: Pad vertex tokens to GLOBAL_MAX_VERTICES_MODEL (150)
    vertex_tokens_padded = torch.zeros(GLOBAL_MAX_VERTICES_MODEL, 3, dtype=torch.long, device=device)
    vertex_tokens_padded[:V_actual, :] = vertex_tokens_for_model

    # Edge indices for FaceModel input memory needs to be original format (E, max_verts_per_edge) tensor.
    max_verts_per_edge = 3 # Max 3 vertices per edge (lines or arcs)
    edge_indices_tensor_for_model = torch.full((E_actual, max_verts_per_edge), -1, dtype=torch.long, device=device)
    for i, e_tuple in enumerate(edges_orig_tuples):
        edge_indices_tensor_for_model[i, :len(e_tuple)] = torch.tensor(e_tuple, dtype=torch.long)
    
    # FIXED: Pad edge indices to GLOBAL_MAX_EDGES_MODEL (150)
    edge_indices_padded = torch.full((GLOBAL_MAX_EDGES_MODEL, max_verts_per_edge), -1, dtype=torch.long, device=device)
    edge_indices_padded[:E_actual, :] = edge_indices_tensor_for_model
    
    # FIXED: Use global fixed token values
    sos_token = FACE_SOS_TOKEN      # 152
    eos_token = FACE_EOS_TOKEN      # 151
    new_face_token = FACE_NEW_FACE_TOKEN  # 150
    
    print(f"FIXED tokens: SOS={sos_token}, EOS={eos_token}, NEW_FACE={new_face_token}")
    
    current_seq_list = [sos_token]  # Start with FIXED SOS token (152)
    
    generated_faces_lists = []
    current_face_edge_indices_building = [] # Temporarily stores edge indices for the current face being built
    
    # Keep track of edge usage for dynamic masking (get_face_mask Rule 4) during generation.
    # Initialize counts to zero for all generated edges.
    edge_counts_for_mask = {i: 0 for i in range(E_actual)} 

    with torch.no_grad(): # Generation does not need gradients
        for step in range(1, max_seq_len): # Loop for subsequent tokens (after SOS)
            seq_tensor = torch.tensor([current_seq_list], dtype=torch.long, device=device) # (1, L_current)
            
            # Pass sequence, PADDED vertices, PADDED edges, and image_embed to the FaceModel
            logits = model(seq_tensor,
                           vertex_tokens_padded.unsqueeze(0),     # (1, 150, 3)
                           edge_indices_padded.unsqueeze(0),      # (1, 150, 3)
                           img_embed_tensor)[:, -1, :]            # (1, 153)
            
            # Get mask for current context using FIXED vocabulary
            mask = get_face_mask(current_seq_list, E_actual, edge_counts_for_mask)  # (153,)

            # FIX: Additional masking - edges already used in the CURRENT face should be masked
            for edge_idx in current_face_edge_indices_building:
                if edge_idx < FACE_VOCAB_SIZE:
                    mask[edge_idx] = True

            # DEBUG: Check if mask is working
            if len(current_seq_list) > 2:
                prev_edge = current_seq_list[-1]
                if prev_edge < E_actual and not mask[prev_edge]:
                    print(f"BUG! prev_edge={prev_edge} should be masked but isn't!")

            # Nucleus sampling
            next_token_tensor = masked_nucleus_sampling(logits, top_p, mask=mask.unsqueeze(0).to(device), eos_idx=FACE_EOS_TOKEN)
            next_token = next_token_tensor.item()
            # Validate token is in valid range
            if next_token >= FACE_VOCAB_SIZE - 1:  # Must be 0-152 (not 153)
                print(f"  [ERROR] Generated invalid token {next_token} (outside vocab 0-{FACE_VOCAB_SIZE-2})! Forcing EOS to terminate.")
                next_token = eos_token
            # Check for EOS or NEW_FACE tokens to finalize current face
            if next_token == eos_token:  # FIXED: 151
                # If current_face_edge_indices_building has at least 2 edges, it's a valid face
                if len(current_face_edge_indices_building) >= 2:
                    generated_faces_lists.append(current_face_edge_indices_building)
                break # Stop generation
                
            if next_token == new_face_token:  # FIXED: 150
                # Finalize current face if valid
                if len(current_face_edge_indices_building) >= 2:
                    generated_faces_lists.append(current_face_edge_indices_building)
                    print(f"  [INFO] Face completed with {len(current_face_edge_indices_building)} edges: {current_face_edge_indices_building}")

                current_face_edge_indices_building = [] # Reset for next face
            else: # It's an edge index (0 to E_actual-1)
                current_face_edge_indices_building.append(next_token)
                # Update edge usage count for masking next steps (Rule 4)
                edge_counts_for_mask[next_token] = edge_counts_for_mask.get(next_token, 0) + 1
                print(f"  [INFO] Added edge {next_token} to current face (usage count: {edge_counts_for_mask[next_token]})")

            current_seq_list.append(next_token) # Add token to sequence for next step
    
    return generated_faces_lists

# ========================
# MAIN GENERATION FUNCTION (for Inference/Evaluation) - UNCONDTIONAL
# ========================
def generate_unconditional(models_tuple, device, V_min_orig_for_dequant=None, V_max_orig_for_dequant=None):
    """
    Full B-rep generation pipeline - UNCONDITIONAL (no image input).
    Args:
        models_tuple (tuple): Tuple of (v_model, e_model, f_model).
        device (torch.device): Device for inference.
        V_min_orig_for_dequant (np.array, optional): Original min XYZ for dequantization.
        V_max_orig_for_dequant (np.array, optional): Original max XYZ for dequantization.
    Returns:
        (TopoDS_Shape or None, dict or None): Generated B-rep shape and indexed data (vertices, edges, faces).
    """
    v_model, e_model, f_model = models_tuple
    
    # === UNCONDITIONAL: No image encoding, image_embed = None ===
    image_embed = None

    # === Step 1: Generate Vertices ===
    print("\n--- Generating Vertices ---")
    generated_vertices_quantized_xyz = generate_vertices(image_embed, v_model, device)
    print(f"Generated {generated_vertices_quantized_xyz.shape[0]} vertices.")
    if generated_vertices_quantized_xyz.shape[0] == 0:
        print("WARN(GenMain): No vertices generated. Returning None.", file=sys.stderr)
        return None, None

    # === Step 2: Generate Edges ===
    print("\n--- Generating Edges ---")
    generated_edges_tuples = generate_edges(generated_vertices_quantized_xyz, image_embed, e_model, device)
    print(f"Generated {len(generated_edges_tuples)} edges: {generated_edges_tuples}")
    if not generated_edges_tuples:
        print("WARN(GenMain): No edges generated. Returning None.", file=sys.stderr)
        return None, None

    # === Step 3: Generate Faces ===
    print("\n--- Generating Faces ---")
    generated_faces_lists = generate_faces(generated_vertices_quantized_xyz, generated_edges_tuples, image_embed, f_model, device)
    print(f"Generated {len(generated_faces_lists)} faces: {generated_faces_lists}")
    if not generated_faces_lists:
        print("WARN(GenMain): No faces generated. Returning None.", file=sys.stderr)
        return None, None

    # === Step 4: Dequantize vertices for B-Rep reconstruction ===
    if V_min_orig_for_dequant is not None and V_max_orig_for_dequant is not None:
        vertices_for_brep_dequantized = dequantize_vertices(generated_vertices_quantized_xyz, V_min_orig_for_dequant, V_max_orig_for_dequant)
    else:
        # Fallback: scale 0-63 range to 0-1 range for visualization
        vertices_for_brep_dequantized = generated_vertices_quantized_xyz.astype(np.float32) / 63.0
        print("INFO: Scaling generated vertices to [0,1] range.")

    # === Step 5: B-Rep Reconstruction using OCP ===
    try:
        cad_model_brep_shape = indexed_to_brep(vertices_for_brep_dequantized, generated_edges_tuples, generated_faces_lists)
        
        if cad_model_brep_shape is None:
            return None, None

        return cad_model_brep_shape, {
            'vertices': vertices_for_brep_dequantized.tolist(),
            'edges': generated_edges_tuples,
            'faces': generated_faces_lists
        }
    except Exception as e:
        print(f"ERROR in B-Rep reconstruction: {e}")
        traceback.print_exc()
        return None, None

# ========================
# MAIN GENERATION FUNCTION (for Inference/Evaluation)
# ========================
def generate_from_image(image_tensor, models_tuple, device, V_min_orig_for_dequant=None, V_max_orig_for_dequant=None):
    """
    Full B-rep generation pipeline from an input image.
    Args:
        image_tensor (torch.Tensor): Input image tensor (C, H, W).
        models_tuple (tuple): Tuple of (img_encoder, v_model, e_model, f_model).
        device (torch.device): Device for inference.
        V_min_orig_for_dequant (np.array, optional): Original min XYZ for dequantization.
        V_max_orig_for_dequant (np.array, optional): Original max XYZ for dequantization.
    Returns:
        (TopoDS_Shape or None, dict or None): Generated B-rep shape and indexed data (vertices, edges, faces).
    """
    img_encoder, v_model, e_model, f_model = models_tuple
    
    # === Step 1: Encode the input image ===
    image_batch = image_tensor.unsqueeze(0).to(device) # (1, C, H, W)
    with torch.no_grad():
        image_embed = img_encoder(image_batch) # (1, 256, d_model)

    # === Step 2: Generate Vertices ===
    print("\n--- Generating Vertices ---")
    generated_vertices_quantized_xyz = generate_vertices(image_embed, v_model, device)
    print(f"Generated {generated_vertices_quantized_xyz.shape[0]} vertices.")
    if generated_vertices_quantized_xyz.shape[0] == 0:
        print("WARN(GenMain): No vertices generated. Returning None.", file=sys.stderr)
        return None, None

    # === Step 3: Generate Edges ===
    print("\n--- Generating Edges ---")
    generated_edges_tuples = generate_edges(generated_vertices_quantized_xyz, image_embed, e_model, device)
    print(f"Generated {len(generated_edges_tuples)} edges: {generated_edges_tuples}")
    if not generated_edges_tuples:
        print("WARN(GenMain): No edges generated. Returning None.", file=sys.stderr)
        return None, None

    # === Step 4: Generate Faces ===
    print("\n--- Generating Faces ---")
    generated_faces_lists = generate_faces(generated_vertices_quantized_xyz, generated_edges_tuples, image_embed, f_model, device)
    print(f"Generated {len(generated_faces_lists)} faces: {generated_faces_lists}")
    if not generated_faces_lists:
        print("WARN(GenMain): No faces generated. Returning None.", file=sys.stderr)
        return None, None

    # === Step 4.5: Dequantize vertices for B-Rep reconstruction ===
    if V_min_orig_for_dequant is not None and V_max_orig_for_dequant is not None:
        vertices_for_brep_dequantized = dequantize_vertices(generated_vertices_quantized_xyz, V_min_orig_for_dequant, V_max_orig_for_dequant)
    else:
        # Fallback: if no specific dequantization range, scale 0-63 range to a generic 0-1 range for visualization.
        # This will produce tiny models if original scale was large, but prevents reconstruction errors from huge coords.
        vertices_for_brep_dequantized = generated_vertices_quantized_xyz.astype(np.float32) / 63.0
        print("WARN(GenMain): V_min/V_max for dequantization not provided. Scaling generated vertices to [0,1].", file=sys.stderr)

    # === Step 5: B-Rep Reconstruction using OCP ===
    try:
        # indexed_to_brep handles its own errors and returns None if reconstruction fails.
        cad_model_brep_shape = indexed_to_brep(vertices_for_brep_dequantized, generated_edges_tuples, generated_faces_lists)
        
        if cad_model_brep_shape is None: # indexed_to_brep returns None if internal OCP steps fail
            return None, None

        # Return the TopoDS_Shape and the raw generated indexed B-rep data (lists for JSON)
        return cad_model_brep_shape, {
            'vertices': vertices_for_brep_dequantized.tolist(), # Convert numpy array to list for JSON
            'edges': generated_edges_tuples,
            'faces': generated_faces_lists
        }
    except Exception as e:
        return None, None # Return None if reconstruction fails

def load_checkpoint_unconditional(models_tuple, optimizer, filepath, device):
    """Loads model and training state from a checkpoint file."""
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

def load_checkpoint(models_tuple, optimizer, filepath, device):
    """Loads model and training state from a checkpoint file."""
    if not os.path.isfile(filepath):
        print(f" => No checkpoint found at '{filepath}'")
        return None

    print(f" => Loading checkpoint from '{filepath}'")
    checkpoint = torch.load(filepath, map_location=device)

    img_encoder, v_model, e_model, f_model = models_tuple
    img_encoder.load_state_dict(checkpoint['img_encoder_state_dict'])
    v_model.load_state_dict(checkpoint['v_model_state_dict'])
    e_model.load_state_dict(checkpoint['e_model_state_dict'])
    f_model.load_state_dict(checkpoint['f_model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    print(f" => Loaded checkpoint from epoch {start_epoch} with best validation loss {best_val_loss:.4f}")
    return start_epoch, best_val_loss

# --- 7. Example Inference (as before, but using the best model) ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Make sure models are on the correct device and in evaluation mode
    print("Initializing models (unconditional)...")
    # img_encoder = ImageEncoder().to(device)
    v_model = VertexModel().to(device)
    e_model = EdgeModel().to(device)
    f_model = FaceModel().to(device)
    models = (v_model, e_model, f_model)
    
    # Load the best performing checkpoint
    print(f"Loading best model from: {args.checkpoint_path}")
    # load_checkpoint_unconditional(models, optimizer=None, filepath=args.checkpoint_path, device=device)
    load_checkpoint_unconditional(models, optimizer=None, filepath=args.checkpoint_path, device=device)

    for model in models:
        model.eval()

    # # --- 2. Load and Prepare the Input Image ---
    # print(f"Loading input image from: {args.image_path}")
    # image_transform = transforms.Compose([transforms.ToTensor()])
    # image = Image.open(args.image_path).convert('RGB')
    # input_image_tensor = image_transform(image)
    
    # --- 3. Run Inference ---
    print("Attempting to generate model...")
    # NOTE: For a generic image, we don't have the original V_min/V_max for dequantization.
    # The model will fall back to normalizing the output to a [0,1] cube, which is fine.
    # generated_cad_model, _ = generate_from_image(
    #     input_image_tensor, models, device, V_min_orig_for_dequant=None, V_max_orig_for_dequant=None
    # )

    # os.makedirs(args.output_dir, exist_ok=True)

    # --- 4. Generate Multiple Samples ---
    successful = 0
    failed = 0
    
    for i in range(args.num_samples):
        print(f"\n{'='*60}")
        print(f"Generating sample {i+1}/{args.num_samples}")
        print(f"{'='*60}")
        
        generated_cad_model, indexed_data = generate_unconditional(models, device)

        if generated_cad_model is not None:
            # Save STEP file
            step_path = os.path.join(args.output_dir, f"generated_{i+1}.step")
            write_step_file(generated_cad_model, step_path)
            print(f"✓ Saved STEP file: {step_path}")
            
            # Save JSON with indexed B-rep data
            json_path = os.path.join(args.output_dir, f"generated_{i+1}.json")
            with open(json_path, 'w') as f:
                json.dump(indexed_data, f, indent=2)
            print(f"✓ Saved JSON file: {json_path}")
            
            successful += 1
        else:
            print(f"✗ Failed to generate valid B-rep for sample {i+1}")
            failed += 1
    
    # --- 5. Summary ---
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{args.num_samples}")
    print(f"Failed: {failed}/{args.num_samples}")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SolidGen Unconditional Generation Script")
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help='Path to the trained model checkpoint (model_best.pth.tar).')
    parser.add_argument('--output_dir', type=str, default='./generated_output', 
                        help='Directory to save generated .step and .json files.')
    parser.add_argument('--num_samples', type=int, default=10, 
                        help='Number of samples to generate.')
    
    args = parser.parse_args()
    main(args) 
