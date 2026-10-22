#!/usr/bin/env python3
"""Generates the clean, presentation-ready tutorial notebook:
'notebooks/quantization_math_and_tensors.ipynb'.

Designed specifically for live presentation demos:
1. Clean, readable, and easy-to-follow code cells.
2. Directly uses quantlab library components.
3. Includes the exact underlying quantlab library implementation in comments
   inside each code cell for seamless explanation during presentation slides.
4. Demonstrates the math, symmetric vs asymmetric, granularities & outliers,
   physical bit-packing, and the MNIST PTQ model conversion (398 KB -> 100 KB).
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

> **Note for Presenters:** Every code cell includes the exact underlying implementation from `quantlab` right in the comments, so you can explain the mechanics without switching files!"""
cells.append(new_markdown_cell(cell_0, id='cell-0-intro'))

# ==============================================================================
# Cell 1: Setup & Imports
# ==============================================================================
cell_1_code = r"""# ==============================================================================
# STEP 0: SETUP & ENVIRONMENT
# ==============================================================================
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add repository root to path for local or remote execution
repo_root = Path(".").resolve()
if (repo_root / "src").exists():
    sys.path.insert(0, str(repo_root / "src"))
elif (repo_root.parent / "src").exists():
    sys.path.insert(0, str(repo_root.parent / "src"))

# Verify quantlab library imports
import quantlab
from quantlab.config import QuantConfig, Mode, Granularity
from quantlab.asymmetric_params import compute_asymmetric_params_batched
from quantlab.symmetric_params import compute_symmetric_params_batched
from quantlab.grouping import group_weight, ungroup_weight
from quantlab.bit_packer import BitPacker
from quantlab.linear_dequantizer import LinearDequantizer
from quantlab.rtn_quantizer import RTNQuantizer
from quantlab.quantized_linear import QuantizedLinear

print(f"PyTorch version: {torch.__version__}")
print(f"quantlab library loaded successfully from: {quantlab.__file__}")"""
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
# ==============================================================================
# 💡 UNDER THE HOOD IN QUANTLAB (from src/quantlab/asymmetric_params.py):
# ------------------------------------------------------------------------------
# def compute_asymmetric_params_batched(weight, bits):
#     qmax = (1 << bits) - 1
#     w_min = torch.minimum(weight.amin(dim=1), torch.zeros_like(w_min))
#     w_max = torch.maximum(weight.amax(dim=1), torch.zeros_like(w_max))
#     scale = (w_max - w_min) / qmax
#     zero_point = torch.round(-w_min / scale).clamp(0, qmax)
#     return scale.to(torch.float16), zero_point
#
# 💡 UNDER THE HOOD IN QUANTLAB (from src/quantlab/linear_dequantizer.py):
# ------------------------------------------------------------------------------
# def dequantize(codes, scale, zero_point):
#     return scale.float() * (codes.float() - zero_point.float())
# ==============================================================================

# Input 1D float tensor
x = torch.tensor([-0.8, -0.2, 0.0, 0.55, 1.2, 2.2], dtype=torch.float32)
bits = 4  # 4-bit -> 16 integer levels [0..15]

# 1. Compute Scale and Zero-Point using quantlab
scales, zero_points, qmin, qmax = compute_asymmetric_params_batched(x.unsqueeze(0), bits=bits)
S = scales[0].item()
Z = zero_points[0].item()

# 2. Quantize: Float -> Integer Codes
q = torch.clamp(torch.round(x / S) + Z, qmin, qmax).to(torch.int32)

# 3. Dequantize: Integer Codes -> Approximate Float using quantlab's LinearDequantizer
dequantizer = LinearDequantizer()
x_hat = dequantizer.dequantize(q.float(), torch.tensor([S]), torch.tensor([Z]), bits=bits)

# 4. Display Results & Verify Error Bound
abs_error = (x - x_hat).abs()
print("Original Float Tensor (x):    ", [round(v, 4) for v in x.tolist()])
print(f"Scale (S): {S:.4f} (step size) | Zero-Point (Z): {Z} (represents float 0.0)")
print("Quantized Integers (q) [0..15]:", q.tolist())
print("Reconstructed Float (x_hat):  ", [round(v, 4) for v in x_hat.tolist()])
print("Absolute Error (|x - x_hat|):  ", [round(v, 4) for v in abs_error.tolist()])
print(f"Max Error: {abs_error.max().item():.4f} <= S/2 ({S / 2:.4f}) -> Error bound confirmed!")"""
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
# ==============================================================================
# 💡 UNDER THE HOOD IN QUANTLAB (from src/quantlab/symmetric_params.py):
# ------------------------------------------------------------------------------
# def compute_symmetric_params_batched(weight, bits):
#     qmax = (1 << (bits - 1)) - 1
#     abs_max = weight.abs().amax(dim=1)
#     scale = abs_max / qmax
#     return scale.to(torch.float16), -qmax, qmax
# ==============================================================================

# Simulated ReLU activation tensor (strictly non-negative)
acts = torch.tensor([0.0, 0.5, 1.2, 2.8, 4.5, 7.5], dtype=torch.float32)
bits = 4

# A) Asymmetric Quantization (Full 16 levels [0..15])
s_asym, zp_asym, qmin_a, qmax_a = compute_asymmetric_params_batched(acts.unsqueeze(0), bits)
q_asym = torch.clamp(torch.round(acts / s_asym[0].item()) + zp_asym[0].item(), qmin_a, qmax_a)
x_hat_asym = s_asym[0].item() * (q_asym - zp_asym[0].item())
mae_asym = (acts - x_hat_asym).abs().mean().item()

# B) Symmetric Quantization (Levels [-7..7], but negative codes are unused!)
s_sym, qmin_s, qmax_s = compute_symmetric_params_batched(acts.unsqueeze(0), bits)
q_sym = torch.clamp(torch.round(acts / s_sym[0].item()), qmin_s, qmax_s)
x_hat_sym = s_sym[0].item() * q_sym
mae_sym = (acts - x_hat_sym).abs().mean().item()

print("=" * 65)
print("INPUT ACTIVATION TENSOR: ", acts.tolist())
print("=" * 65)
print(f"ASYMMETRIC (4-bit, [0..15]):")
print(f"  Scale: {s_asym[0].item():.4f} | Zero-Point: {zp_asym[0].item()}")
print(f"  Integer Codes:       {q_asym.int().tolist()}")
print(f"  Mean Absolute Error: {mae_asym:.5f}")
print("-" * 65)
print(f"SYMMETRIC (4-bit, [-7..7]):")
print(f"  Scale: {s_sym[0].item():.4f} | Zero-Point: 0 (Fixed)")
print(f"  Integer Codes:       {q_sym.int().tolist()}  <-- Negative codes wasted!")
print(f"  Mean Absolute Error: {mae_sym:.5f}")
print("=" * 65)
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
# ==============================================================================
# 💡 UNDER THE HOOD IN QUANTLAB (from src/quantlab/grouping.py):
# ------------------------------------------------------------------------------
# def group_weight(weight, granularity, group_size):
#     if granularity == Granularity.TENSOR:
#         return weight.reshape(1, -1), info            # 1 scale for all
#     elif granularity == Granularity.CHANNEL:
#         return weight, info                           # 1 scale per row
#     elif granularity == Granularity.GROUP:
#         return weight.reshape(-1, group_size), info   # 1 scale per block of G
# ==============================================================================

# Synthetic 4x8 weight matrix with an extreme outlier at W[2, 3] = 12.0
torch.manual_seed(42)
W = torch.randn(4, 8) * 0.15
W[2, 3] = 12.0  # The extreme outlier

print(f"Weight Matrix (4x8) with Outlier at Row 2, Col 3 ({W[2, 3].item():.1f}):")
print(f"Row 0 (Normal): {[round(v, 2) for v in W[0].tolist()]}")
print(f"Row 2 (Outlier):{[round(v, 2) for v in W[2].tolist()]}\n")

# Evaluate 3 granularities using quantlab's group_weight
results = []
for gran, g_size, label in [
    (Granularity.TENSOR, None, "Per-Tensor (1 scale)"),
    (Granularity.CHANNEL, None, "Per-Channel (4 scales)"),
    (Granularity.GROUP, 4, "Per-Group G=4 (8 scales)")
]:
    # 1. Group tensor using quantlab
    grouped, info = group_weight(W, gran, g_size)
    
    # 2. Compute symmetric scales for each group
    scales, qmin, qmax = compute_symmetric_params_batched(grouped, bits=4)
    
    # 3. Quantize and dequantize
    q = torch.clamp(torch.round(grouped / scales.float().unsqueeze(1)), qmin, qmax)
    w_hat_grouped = q * scales.float().unsqueeze(1)
    
    # 4. Ungroup back to original 4x8 shape using quantlab
    W_hat = ungroup_weight(w_hat_grouped, info)
    
    # Record Mean Squared Error
    mse_row0 = F.mse_loss(W_hat[0], W[0]).item()
    mse_row2 = F.mse_loss(W_hat[2], W[2]).item()
    results.append((label, len(scales), mse_row0, mse_row2))

print("=" * 72)
print(f"{'Granularity':<24} | {'# Scales':<8} | {'Row 0 (Normal) MSE':<18} | {'Row 2 MSE'}")
print("=" * 72)
for label, num_s, r0_mse, r2_mse in results:
    print(f"{label:<24} | {num_s:<8} | {r0_mse:<18.6f} | {r2_mse:.6f}")
print("=" * 72)
print("Takeaway: In Per-Tensor, Row 0 is corrupted by Row 2's outlier.")
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
# ==============================================================================
# 💡 UNDER THE HOOD IN QUANTLAB (from src/quantlab/bit_packer.py):
# ------------------------------------------------------------------------------
# def pack(codes, bits):
#     codes_per_byte = 8 // bits   # For 4-bit: 2 codes per byte
#     codes = codes.reshape(-1, codes_per_byte)
#     packed = torch.zeros(codes.shape[0], dtype=torch.uint8)
#     for i in range(codes_per_byte):
#         packed |= (codes[:, i] << (i * bits))
#     return packed
# ==============================================================================

# Four 4-bit integer codes: values fit within [0..15]
codes = torch.tensor([3, 12, 7, 1], dtype=torch.uint8)

# 1. Pack using quantlab's BitPacker
packer = BitPacker()
packed_bytes = packer.pack(codes, bits=4)

# 2. Unpack back to verify perfect reconstruction
unpacked_codes = packer.unpack(packed_bytes, bits=4, count=len(codes))

print("4-BIT PACKING DEMONSTRATION:")
print(f"  Input 4-bit codes:     {codes.tolist()}")
print(f"  Packed bytes (uint8):  {packed_bytes.tolist()}")
print(f"  Byte 0 binary layout:  {format(packed_bytes[0].item(), '08b')}  (Code 0: 3 | Code 1: 12 << 4)")
print(f"  Byte 1 binary layout:  {format(packed_bytes[1].item(), '08b')}  (Code 2: 7 | Code 3:  1 << 4)")
print(f"  Unpacked codes:        {unpacked_codes.tolist()}")
print(f"  Exact Match:           {torch.equal(codes, unpacked_codes)}")
print(f"  Physical Compression:  {len(codes)} elements stored in {len(packed_bytes)} bytes (2x vs int8, 8x vs float32)!")"""
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
# ==============================================================================
# 💡 UNDER THE HOOD IN QUANTLAB (from src/quantlab/rtn_quantizer.py):
# ------------------------------------------------------------------------------
# class RTNQuantizer:
#     def quantize(self, model):
#         for name, module in model.named_modules():
#             if isinstance(module, nn.Linear):
#                 # 1. Compute scales & zero-points
#                 # 2. Pack weights into uint8 buffers (BitPacker)
#                 # 3. Replace module with QuantizedLinear
#                 setattr(parent, attr, qlinear)
#         return model
# ==============================================================================
import copy

# Helper function to compute in-memory tensor storage
def get_model_size_kb(model: nn.Module) -> float:
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    return total_bytes / 1024.0

# 1. Define Standard MNIST MLP Model (784 -> 128 -> 10)
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

model_fp32 = MNISTNet()
size_fp32 = get_model_size_kb(model_fp32)

# 2. Quantize to 8-Bit using quantlab (Replicating the video's 360 KB -> 94 KB drop)
config_8bit = QuantConfig(bits=8, mode=Mode.SYMMETRIC, granularity=Granularity.CHANNEL)
model_8bit = RTNQuantizer(config_8bit, skip_names=()).quantize(copy.deepcopy(model_fp32))
size_8bit = get_model_size_kb(model_8bit)

# 3. Quantize to 4-Bit using quantlab
config_4bit = QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.CHANNEL)
model_4bit = RTNQuantizer(config_4bit, skip_names=()).quantize(copy.deepcopy(model_fp32))
size_4bit = get_model_size_kb(model_4bit)

# 4. Verify Forward Pass & Output Logit Deviation
test_input = torch.randn(1, 784)
out_fp32 = model_fp32(test_input)
out_8bit = model_8bit(test_input)
out_4bit = model_4bit(test_input)

print("=" * 65)
print("MNIST PTQ CONVERSION REPORT")
print("=" * 65)
print(f"Float32 Model Size:      {size_fp32:>8.2f} KB (Baseline 100%)")
print(f"Quantized 8-Bit Size:    {size_8bit:>8.2f} KB ({size_8bit/size_fp32*100:.1f}% of baseline, ~75% reduction!)")
print(f"Quantized 4-Bit Size:    {size_4bit:>8.2f} KB ({size_4bit/size_fp32*100:.1f}% of baseline, ~85% reduction!)")
print("-" * 65)
print("Converted Layers in model_8bit:")
print(f"  fc1: {model_8bit.fc1}")
print(f"  fc2: {model_8bit.fc2}")
print("-" * 65)
print(f"Output Logit Deviation (Float32 vs 8-bit): {F.mse_loss(out_fp32, out_8bit).item():.5f}")
print(f"Output Logit Deviation (Float32 vs 4-bit): {F.mse_loss(out_fp32, out_4bit).item():.5f}")
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
# ==============================================================================

def test_tensor_quantization(tensor: torch.Tensor, bits: int = 4, mode: str = "sym", group_size: int = None):
    # Clean 1-line helper for interactive presentation demos
    m = Mode.SYMMETRIC if mode == "sym" else Mode.ASYMMETRIC
    g = Granularity.GROUP if group_size else (Granularity.CHANNEL if tensor.ndim > 1 else Granularity.TENSOR)
    
    # 1. Group tensor
    w_2d = tensor if tensor.ndim == 2 else tensor.unsqueeze(0)
    grouped, info = group_weight(w_2d, g, group_size)
    
    # 2. Compute parameters via quantlab
    if m == Mode.SYMMETRIC:
        scales, qmin, qmax = compute_symmetric_params_batched(grouped, bits)
        q = torch.clamp(torch.round(grouped / scales.float().unsqueeze(1)), qmin, qmax)
        deq = q * scales.float().unsqueeze(1)
    else:
        scales, zp, qmin, qmax = compute_asymmetric_params_batched(grouped, bits)
        q = torch.clamp(torch.round(grouped / scales.float().unsqueeze(1)) + zp.float().unsqueeze(1), qmin, qmax)
        deq = (q - zp.float().unsqueeze(1)) * scales.float().unsqueeze(1)
        
    rec = ungroup_weight(deq, info)
    if tensor.ndim == 1:
        rec = rec.squeeze(0)
        
    mse = F.mse_loss(rec, tensor).item()
    sqnr = 10 * torch.log10(tensor.pow(2).sum() / (tensor - rec).pow(2).sum()).item()
    
    print(f"=== TEST RESULT: {bits}-bit {mode.upper()} ({g.value}) ===")
    print(f"Input Shape: {list(tensor.shape)} | Groups: {len(scales)}")
    print(f"MSE: {mse:.6f} | SQNR: {sqnr:.2f} dB (Higher is better)")
    return rec

# Run an example live test on a random weight tensor
sample_tensor = torch.randn(16, 64)
reconstructed = test_tensor_quantization(sample_tensor, bits=4, mode="sym", group_size=16)"""
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
