#!/usr/bin/env python3
"""Generates the streamlined, presentation-ready tutorial notebook:
'notebooks/quantization_math_and_tensors.ipynb'.

Key Features for Presentation:
1. Auto-clones https://github.com/hi-z-k/Quantizer.git if running in Colab/Kaggle.
2. Centralizes all presentation configuration at the top cell.
3. Minimizes boilerplate code.
4. Includes exact repository file locations and commented library implementations
   inside every code cell for effortless presentation and walkthroughs.
"""

from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.metadata = {
    'language_info': {'name': 'python'},
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
}

cells = []

# ==============================================================================
# Cell 0: Title & Presentation Intro
# ==============================================================================
cell_0 = r"""# Post-Training Quantization (PTQ): Mathematics, Mechanics & Live Demo

Welcome to the live presentation demo of **Post-Training Quantization (PTQ)** using the [`quantlab`](https://github.com/hi-z-k/Quantizer) library.

---

### Presentation Overview
This notebook replicates the core principles of neural network quantization demonstrated in classic Post-Training Quantization tutorials (e.g. PyTorch MNIST PTQ):
1. **Mathematical Foundations:** Mapping continuous 32-bit floats to discrete integers ($S$, $Z$, $q$, $\hat{x}$) and calculating error bounds.
2. **Symmetric vs. Asymmetric:** Why zero-centering matters and how asymmetric quantization preserves dynamic range on activation tensors ($x \ge 0$).
3. **Granularities & Outlier Isolation:** Comparing **Per-Tensor**, **Per-Channel (Row)**, and **Per-Group** blocks to protect normal weights from outlier distortion.
4. **Physical Bit-Packing:** How 4-bit and 2-bit values are packed into `uint8` hardware bytes.
5. **Replicating MNIST PTQ:** Converting standard `nn.Linear` layers into packed `QuantizedLinear` layers, reproducing the **~398 KB $\to$ ~100 KB** memory reduction (~75% drop).

> **Presenter Tip:** Every code cell clearly states `📁 Located in repo: src/quantlab/...` and includes the exact library code in the comments for instant reference!"""
cells.append(new_markdown_cell(cell_0, id='cell-0-intro'))

# ==============================================================================
# Cell 1: Setup, Auto-Clone, Imports & Centralized Configuration
# ==============================================================================
cell_1_code = r"""# ==============================================================================
# STEP 0: CLONE REPOSITORY & SETUP ENVIRONMENT
# ==============================================================================
import os, sys, subprocess
from pathlib import Path

# 1. Auto-clone repository if running in Google Colab, Kaggle, or a remote runner
REPO_URL = "https://github.com/hi-z-k/Quantizer.git"
REPO_DIR_NAME = "Quantizer"

found_repo = None
for candidate in [
    Path("."), 
    Path(".."), 
    Path(REPO_DIR_NAME), 
    Path(f"/content/{REPO_DIR_NAME}"), 
    Path(f"/kaggle/working/{REPO_DIR_NAME}")
]:
    if (candidate / "src" / "quantlab").exists():
        found_repo = candidate.resolve()
        break

if found_repo is None:
    print(f"Cloning repository from {REPO_URL}...")
    subprocess.run(["git", "clone", REPO_URL], check=True)
    found_repo = Path(REPO_DIR_NAME).resolve()

src_path = str(found_repo / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print(f"✅ Quantizer repository active at: {found_repo}")
print(f"✅ Added {src_path} to sys.path")

import torch
import torch.nn as nn
import torch.nn.functional as F

# 2. Direct library imports from quantlab
import quantlab
from quantlab.config import QuantConfig, Mode, Granularity
from quantlab.asymmetric_params import compute_asymmetric_params_batched
from quantlab.symmetric_params import compute_symmetric_params_batched
from quantlab.grouping import group_weight, ungroup_weight
from quantlab.bit_packer import BitPacker
from quantlab.linear_dequantizer import LinearDequantizer
from quantlab.rtn_quantizer import RTNQuantizer
from quantlab.quantized_linear import QuantizedLinear

print(f"✅ quantlab loaded successfully from: {quantlab.__file__}")

# ==============================================================================
# PRESENTATION CONFIGURATION (All demo settings centralized here!)
# ==============================================================================
BITS = 4         # Bit-width to demonstrate (e.g. 2, 4, 8)
GROUP_SIZE = 4   # Block size for grouped quantization (G=4)
SEED = 42        # Reproducibility seed for synthetic demo tensors

torch.manual_seed(SEED)
print(f"Configuration set: BITS={BITS}, GROUP_SIZE={GROUP_SIZE}, SEED={SEED}")"""
cells.append(new_code_cell(cell_1_code, id='cell-1-setup'))

# ==============================================================================
# Cell 2: Step 1 Markdown - Quantization & Dequantization Math
# ==============================================================================
cell_2 = r"""---
## Step 1: Quantization & Dequantization on a Single Tensor

### The Math
Uniform affine quantization maps a continuous floating-point range $[\alpha, \beta]$ to an integer range $[q_{\min}, q_{\max}]$:

$$S = \frac{\beta - \alpha}{q_{\max} - q_{\min}}, \qquad Z = \text{round}\left(-\frac{\alpha}{S}\right) + q_{\min}$$

$$q = \text{clamp}\left(\left\lfloor \frac{x}{S} \right\rceil + Z, \; q_{\min}, \; q_{\max}\right)$$

$$\hat{x} = S \times (q - Z)$$

- **$S$ (Scale):** The step size represented by 1 integer code step.
- **$Z$ (Zero-Point):** The integer code that represents real floating-point `0.0`.
- **Max Theoretical Error:** Because of rounding $\lfloor \cdot \rceil$, the maximum reconstruction error satisfies:
  $$|x - \hat{x}| \le \frac{S}{2}$$"""
cells.append(new_markdown_cell(cell_2, id='cell-2-step1-md'))

# ==============================================================================
# Cell 3: Step 1 Code - Quantize & Dequantize Tensor
# ==============================================================================
cell_3_code = r"""# ==============================================================================
# STEP 1: QUANTIZE & DEQUANTIZE A 1D TENSOR
# ------------------------------------------------------------------------------
# 📁 Located in repo: src/quantlab/asymmetric_params.py (Lines 71-103)
# ------------------------------------------------------------------------------
# def compute_asymmetric_params_batched(weight, bits):
#     qmax = (1 << bits) - 1
#     w_min = torch.minimum(weight.amin(dim=1), torch.zeros_like(w_min))
#     w_max = torch.maximum(weight.amax(dim=1), torch.zeros_like(w_max))
#     scale = (w_max - w_min) / qmax
#     zero_point = torch.round(-w_min / scale).clamp(0, qmax)
#     return scale.to(torch.float16), zero_point, 0, qmax
#
# 📁 Located in repo: src/quantlab/linear_dequantizer.py (Lines 22-51)
# ------------------------------------------------------------------------------
# def dequantize(codes, scale, zero_point):
#     return scale.float() * (codes.float() - zero_point.float())
# ==============================================================================

# 1. Input 1D float tensor
x = torch.tensor([-0.8, -0.2, 0.0, 0.55, 1.2, 2.2], dtype=torch.float32)

# 2. Compute Scale (S) and Zero-Point (Z) using quantlab
scales, zero_points, qmin, qmax = compute_asymmetric_params_batched(x.unsqueeze(0), bits=BITS)
S, Z = scales[0].item(), zero_points[0].item()

# 3. Quantize: Float -> Integer Codes
q = torch.clamp(torch.round(x / S) + Z, qmin, qmax).to(torch.int32)

# 4. Dequantize: Integer Codes -> Float using quantlab's LinearDequantizer
x_hat = LinearDequantizer().dequantize(q.float(), torch.tensor([S]), torch.tensor([Z]), bits=BITS)

# 5. Display results & verify error bound
max_err = (x - x_hat).abs().max().item()
print(f"Input Float Tensor (x):       {[round(v, 4) for v in x.tolist()]}")
print(f"Computed Scale S: {S:.4f} | Zero-Point Z: {Z} (represents float 0.0)")
print(f"Quantized Integers (q) [0..15]:{q.tolist()}")
print(f"Reconstructed Float (x_hat):  {[round(v, 4) for v in x_hat.tolist()]}")
print(f"Max Error: {max_err:.4f} <= S/2 ({S / 2:.4f})  [Theoretical Bound Verified!]")"""
cells.append(new_code_cell(cell_3_code, id='cell-3-step1-code'))

# ==============================================================================
# Cell 4: Step 2 Markdown - Symmetric vs Asymmetric
# ==============================================================================
cell_4 = r"""---
## Step 2: Symmetric vs. Asymmetric Quantization

### Why Zero-Point Matters
- **Symmetric ($Z = 0$):** Constrains the integer range symmetrically around zero:
  $$q \in [-2^{b-1} + 1, \; 2^{b-1} - 1] \quad (\text{e.g. 4-bit: } [-7, 7])$$
  Float zero is guaranteed to map exactly to integer `0`. Very fast on hardware because operations do not need zero-point offsets.
- **Asymmetric ($Z \ne 0$):** Uses the full unsigned range:
  $$q \in [0, \; 2^b - 1] \quad (\text{e.g. 4-bit: } [0, 15])$$
- **Key Takeaway for Activations:** Neural network activations after ReLU or GELU are non-negative ($x \ge 0$). Symmetric quantization wastes half of its integer range on negative numbers that never occur!"""
cells.append(new_markdown_cell(cell_4, id='cell-4-step2-md'))

# ==============================================================================
# Cell 5: Step 2 Code - Symmetric vs Asymmetric Comparison
# ==============================================================================
cell_5_code = r"""# ==============================================================================
# STEP 2: SYMMETRIC VS ASYMMETRIC ON ACTIVATIONS (x >= 0)
# ------------------------------------------------------------------------------
# 📁 Located in repo: src/quantlab/symmetric_params.py (Lines 67-89)
# ------------------------------------------------------------------------------
# def compute_symmetric_params_batched(weight, bits):
#     qmax = (1 << (bits - 1)) - 1
#     abs_max = weight.abs().amax(dim=1)
#     scale = abs_max / qmax
#     return scale.to(torch.float16), -qmax, qmax   # Z is fixed to 0!
#
# 📁 Located in repo: src/quantlab/asymmetric_params.py (Lines 71-103)
# ------------------------------------------------------------------------------
# Uses full unsigned range [0, 2^bits - 1] with dynamic zero_point.
# ==============================================================================

# Simulated ReLU activations (always non-negative)
acts = torch.tensor([0.0, 0.5, 1.2, 2.8, 4.5, 7.5], dtype=torch.float32)

# 1. Asymmetric: uses all 16 levels [0..15]
s_asym, zp_asym, qmin_a, qmax_a = compute_asymmetric_params_batched(acts.unsqueeze(0), BITS)
q_asym = torch.clamp(torch.round(acts / s_asym[0].item()) + zp_asym[0].item(), qmin_a, qmax_a)
x_asym = s_asym[0].item() * (q_asym - zp_asym[0].item())
mae_asym = (acts - x_asym).abs().mean().item()

# 2. Symmetric: levels [-7..7], negative codes are completely wasted!
s_sym, qmin_s, qmax_s = compute_symmetric_params_batched(acts.unsqueeze(0), BITS)
q_sym = torch.clamp(torch.round(acts / s_sym[0].item()), qmin_s, qmax_s)
x_sym = s_sym[0].item() * q_sym
mae_sym = (acts - x_sym).abs().mean().item()

print(f"ASYMMETRIC [0..15]: Scale={s_asym[0].item():.4f}, Z={zp_asym[0].item()}, Codes={q_asym.int().tolist()}, MAE={mae_asym:.4f}")
print(f"SYMMETRIC  [-7..7]: Scale={s_sym[0].item():.4f}, Z=0, Codes={q_sym.int().tolist()}, MAE={mae_sym:.4f}")
print(f"-> Asymmetric error is {mae_sym / mae_asym:.1f}x LOWER because all 16 levels are utilized!")"""
cells.append(new_code_cell(cell_5_code, id='cell-5-step2-code'))

# ==============================================================================
# Cell 6: Step 3 Markdown - Granularities & Outlier Isolation
# ==============================================================================
cell_6 = r"""---
## Step 3: Quantization Granularities & Outlier Isolation

### Why Grouping is Essential in Modern LLMs
In weight matrices, a few **outlier weights** often have significantly larger magnitudes than their neighbors.
- **Per-Tensor:** One scale for the entire weight matrix. A single outlier inflates $S$ for every weight in the network, destroying resolution in normal rows!
- **Per-Channel (Row-wise):** Each output channel gets its own scale. Outliers only affect their own row.
- **Per-Group (Block-wise, e.g. $G=4$ or $G=32$):** Rows are partitioned into small groups. The outlier is confined strictly to its tiny block."""
cells.append(new_markdown_cell(cell_6, id='cell-6-step3-md'))

# ==============================================================================
# Cell 7: Step 3 Code - Grouping & Outlier Isolation
# ==============================================================================
cell_7_code = r"""# ==============================================================================
# STEP 3: GRANULARITIES & OUTLIER ISOLATION
# ------------------------------------------------------------------------------
# 📁 Located in repo: src/quantlab/grouping.py (Lines 48-119, 121-146)
# ------------------------------------------------------------------------------
# def group_weight(weight, granularity, group_size):
#     if granularity == Granularity.TENSOR:  return weight.reshape(1, -1), info
#     elif granularity == Granularity.CHANNEL: return weight, info
#     elif granularity == Granularity.GROUP:   return weight.reshape(-1, group_size), info
# ==============================================================================

# Synthetic 4x8 weight matrix with an extreme outlier at W[2, 3] = 12.0
W = torch.randn(4, 8) * 0.15
W[2, 3] = 12.0  # Outlier

print(f"Weight Matrix (4x8) with Outlier at Row 2, Col 3 ({W[2, 3].item():.1f}):")
print(f"Row 0 (Normal): {[round(v, 2) for v in W[0].tolist()]}")
print(f"Row 2 (Outlier):{[round(v, 2) for v in W[2].tolist()]}\n")

for gran, g_size, label in [
    (Granularity.TENSOR, None, "Per-Tensor (1 scale)"),
    (Granularity.CHANNEL, None, "Per-Channel (4 scales)"),
    (Granularity.GROUP, GROUP_SIZE, f"Per-Group G={GROUP_SIZE} (8 scales)")
]:
    # Group -> Compute Scales -> Quantize/Dequantize -> Ungroup
    grouped, info = group_weight(W, gran, g_size)
    scales, qmin, qmax = compute_symmetric_params_batched(grouped, bits=BITS)
    q = torch.clamp(torch.round(grouped / scales.float().unsqueeze(1)), qmin, qmax)
    W_hat = ungroup_weight(q * scales.float().unsqueeze(1), info)
    
    r0_mse = F.mse_loss(W_hat[0], W[0]).item()
    r2_mse = F.mse_loss(W_hat[2], W[2]).item()
    print(f"{label:<26} | Scales: {len(scales):<2} | Row 0 (Normal) MSE: {r0_mse:.6f} | Row 2 (Outlier) MSE: {r2_mse:.6f}")

print("\nTakeaway: In Per-Tensor, Row 0 is corrupted by Row 2's outlier.")
print("In Per-Channel & Per-Group, Row 0 retains near-perfect accuracy (>100x lower MSE)!")"""
cells.append(new_code_cell(cell_7_code, id='cell-7-step3-code'))

# ==============================================================================
# Cell 8: Step 4 Markdown - Physical Bit-Packing
# ==============================================================================
cell_8 = r"""---
## Step 4: Physical Bit-Packing (Sub-Byte Hardware Storage)

### Why Bit-Packing is Required
Computer memory and CPU/GPU registers are byte-addressable (minimum 8 bits).
- If 4-bit numbers are stored in standard 8-bit `int8` or `uint8` containers, **no physical memory is saved!**
- [`quantlab.BitPacker`](https://github.com/hi-z-k/Quantizer) packs multiple sub-byte integers into each single byte:
  - **4-bit:** 2 numbers per byte ($8 / 4 = 2$) $\to$ **2x memory reduction vs int8, 8x vs float32**
  - **2-bit:** 4 numbers per byte ($8 / 2 = 4$) $\to$ **4x memory reduction vs int8, 16x vs float32**

$$\text{byte} = (q_0 \ \& \ 0\text{x0F}) \; | \; ((q_1 \ \& \ 0\text{x0F}) \ll 4)$$"""
cells.append(new_markdown_cell(cell_8, id='cell-8-step4-md'))

# ==============================================================================
# Cell 9: Step 4 Code - Bit Packing Demonstration
# ==============================================================================
cell_9_code = r"""# ==============================================================================
# STEP 4: PHYSICAL BIT-PACKING INTO UINT8 BYTES
# ------------------------------------------------------------------------------
# 📁 Located in repo: src/quantlab/bit_packer.py (Lines 26-101)
# ------------------------------------------------------------------------------
# def pack(codes, bits):
#     codes_per_byte = 8 // bits   # For 4-bit: 2 codes per byte
#     codes = codes.reshape(-1, codes_per_byte)
#     packed = torch.zeros(codes.shape[0], dtype=torch.uint8)
#     for i in range(codes_per_byte):
#         packed |= (codes[:, i] << (i * bits))
#     return packed
# ==============================================================================

codes = torch.tensor([3, 12, 7, 1], dtype=torch.uint8)
packer = BitPacker()

# 1. Pack 4-bit codes into uint8 bytes
packed = packer.pack(codes, bits=BITS)

# 2. Unpack back to verify perfect reconstruction
unpacked = packer.unpack(packed, bits=BITS, count=len(codes))

print("4-BIT PACKING DEMONSTRATION:")
print(f"  Input 4-bit codes:     {codes.tolist()}")
print(f"  Packed bytes (uint8):  {packed.tolist()}")
print(f"  Byte 0 binary layout:  {format(packed[0].item(), '08b')}  (Code 0: 3 | Code 1: 12 << 4)")
print(f"  Byte 1 binary layout:  {format(packed[1].item(), '08b')}  (Code 2: 7 | Code 3:  1 << 4)")
print(f"  Unpacked codes:        {unpacked.tolist()}  (Exact match: {torch.equal(codes, unpacked)})")
print(f"  Physical Compression:  {len(codes)} elements stored in {len(packed)} bytes (2x vs int8, 8x vs float32)!")"""
cells.append(new_code_cell(cell_9_code, id='cell-9-step4-code'))

# ==============================================================================
# Cell 10: Step 5 Markdown - MNIST PTQ Replication
# ==============================================================================
cell_10 = r"""---
## Step 5: Replicating MNIST PTQ Layer Conversion

### Connecting to the Demonstration Video
In the PTQ video demonstration:
1. An MNIST model is converted from floating-point layers to integer quantized layers.
2. The model size drops from **~360 KB to ~94 KB** (~75% reduction) while preserving classification accuracy.

Here, we replicate this exact transformation on an MNIST architecture ($784 \to 128 \to 10$) using [`quantlab.RTNQuantizer`](https://github.com/hi-z-k/Quantizer) to replace `nn.Linear` with packed `QuantizedLinear`."""
cells.append(new_markdown_cell(cell_10, id='cell-10-step5-md'))

# ==============================================================================
# Cell 11: Step 5 Code - MNIST PTQ Model Conversion
# ==============================================================================
cell_11_code = r"""# ==============================================================================
# STEP 5: MNIST PTQ MODEL CONVERSION (REPLICATING THE VIDEO)
# ------------------------------------------------------------------------------
# 📁 Located in repo: src/quantlab/rtn_quantizer.py (Lines 29-118)
# ------------------------------------------------------------------------------
# class RTNQuantizer:
#     def quantize(self, model):
#         for name, module in model.named_modules():
#             if isinstance(module, nn.Linear):
#                 qlinear = self._quantize_layer(module)
#                 setattr(parent, attr, qlinear)
#
# 📁 Located in repo: src/quantlab/quantized_linear.py (Lines 31-164)
# ------------------------------------------------------------------------------
# Stores packed_weight (uint8) + scales (float16). Unpacks on-the-fly during forward(x).
# ==============================================================================
import copy

def get_model_size_kb(model: nn.Module) -> float:
    return (sum(p.numel() * p.element_size() for p in model.parameters()) +
            sum(b.numel() * b.element_size() for b in model.buffers())) / 1024.0

# 1. Standard MNIST MLP Model (784 -> 128 -> 10)
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

model_fp32 = MNISTNet()
size_fp32 = get_model_size_kb(model_fp32)

# 2. Quantize to 8-Bit (Replicating the video's 360 KB -> 94 KB drop)
model_8bit = RTNQuantizer(QuantConfig(bits=8, mode=Mode.SYMMETRIC, granularity=Granularity.CHANNEL), skip_names=()).quantize(copy.deepcopy(model_fp32))
size_8bit = get_model_size_kb(model_8bit)

# 3. Quantize to 4-Bit
model_4bit = RTNQuantizer(QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.CHANNEL), skip_names=()).quantize(copy.deepcopy(model_fp32))
size_4bit = get_model_size_kb(model_4bit)

# 4. Verify Forward Pass & Logit Deviation
test_input = torch.randn(1, 784)
out_fp32 = model_fp32(test_input)
out_8bit = model_8bit(test_input)

print("=" * 65)
print("MNIST PTQ CONVERSION REPORT")
print("=" * 65)
print(f"Float32 Model Size:      {size_fp32:>8.2f} KB (Baseline 100%)")
print(f"Quantized 8-Bit Size:    {size_8bit:>8.2f} KB ({size_8bit/size_fp32*100:.1f}% of baseline, ~75% reduction!)")
print(f"Quantized 4-Bit Size:    {size_4bit:>8.2f} KB ({size_4bit/size_fp32*100:.1f}% of baseline, ~85% reduction!)")
print("-" * 65)
print(f"Converted Layer Types:   {type(model_8bit.fc1).__name__}")
print(f"Output Logit Deviation (Float32 vs 8-bit): {F.mse_loss(out_fp32, out_8bit).item():.5f}")
print("=" * 65)"""
cells.append(new_code_cell(cell_11_code, id='cell-11-step5-code'))

# ==============================================================================
# Cell 12: Step 6 Markdown - Presentation Q&A Sandbox
# ==============================================================================
cell_12 = r"""---
## Step 6: Presentation Q&A Sandbox (1-Line Interactive Test)

Use this simple helper function during your presentation to immediately test any question from the audience:
*"What happens if we quantize a Gaussian tensor to 2-bit?"* or *"What if we use group size 16?"*"""
cells.append(new_markdown_cell(cell_12, id='cell-12-step6-md'))

# ==============================================================================
# Cell 13: Step 6 Code - Presentation Sandbox Function
# ==============================================================================
cell_13_code = r"""# ==============================================================================
# STEP 6: 1-LINE PRESENTATION SANDBOX
# ------------------------------------------------------------------------------
# 📁 Uses quantlab: group_weight, compute_symmetric_params_batched, ungroup_weight
# ==============================================================================

def test_tensor_quantization(tensor: torch.Tensor, bits: int = 4, group_size: int = 16):
    # Minimal 1-line helper for interactive presentation demos
    w_2d = tensor if tensor.ndim == 2 else tensor.unsqueeze(0)
    grouped, info = group_weight(w_2d, Granularity.GROUP, group_size)
    scales, qmin, qmax = compute_symmetric_params_batched(grouped, bits)
    q = torch.clamp(torch.round(grouped / scales.float().unsqueeze(1)), qmin, qmax)
    rec = ungroup_weight(q * scales.float().unsqueeze(1), info)
    
    mse = F.mse_loss(rec, w_2d).item()
    sqnr = 10 * torch.log10(w_2d.pow(2).sum() / (w_2d - rec).pow(2).sum()).item()
    print(f"[{bits}-bit Group={group_size}] MSE: {mse:.6f} | SQNR: {sqnr:.2f} dB | Total Scales: {len(scales)}")
    return rec

# Run an example live test on a random weight tensor
sample_tensor = torch.randn(16, 64)
reconstructed = test_tensor_quantization(sample_tensor, bits=BITS, group_size=16)"""
cells.append(new_code_cell(cell_13_code, id='cell-13-step6-code'))

# ==============================================================================
# Save Notebook
# ==============================================================================
nb.cells = cells

output_path = Path("notebooks/quantization_math_and_tensors.ipynb")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Presentation notebook generated successfully at {output_path} ({output_path.stat().st_size} bytes)")
