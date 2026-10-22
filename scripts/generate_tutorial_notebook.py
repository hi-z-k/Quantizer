#!/usr/bin/env python3
"""Generates the educational notebook 'ptq_tensor_mechanics_tutorial.ipynb'.

This notebook demystifies Post-Training Quantization (PTQ) from first principles,
breaking down the exact tensor math, granularities (per-tensor, per-channel, per-group),
symmetric vs asymmetric modes, bit-packing, error metrics (SQNR), and replicating
the MNIST PTQ layer conversion demonstrated in standard tutorials.
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
# Cell 0: Introduction & Overview
# ==============================================================================
cell_0 = r"""# Post-Training Quantization (PTQ): Tensor Mechanics & Mathematics

Welcome! This interactive tutorial demystifies **Post-Training Quantization (PTQ)** by breaking down the underlying **tensor mathematics** and code implementations using the [`quantlab`](https://github.com/hi-z-k/Quantizer) library.

---

### Connecting to the Standard PTQ Pipeline (e.g., PyTorch on MNIST)
In typical deep learning tutorials (such as PyTorch's official PTQ demonstration on MNIST), the quantization workflow is presented at a high level:
1. **Model Preparation:** Insert **Quantization Stubs** (`QuantStub`) and **Dequantization Stubs** (`DeQuantStub`) to mark boundaries between floating-point and integer tensors.
2. **Observer Insertion:** Attach observers to intermediate layers to monitor activations during a calibration pass and track data dynamic ranges ($[\min, \max]$).
3. **Calibration:** Run a forward pass over a representative dataset to calculate the optimal **Scale ($S$)** and **Zero-Point ($Z$)** parameters.
4. **Conversion:** Call `convert()` to replace standard floating-point layers (`nn.Linear`) with integer-based **Quantized Linear layers**.
5. **Result:** Weights drop from 32-bit floats to 8-bit integers (e.g., dropping from **~360 KB** to **~94 KB** on MNIST), maintaining accuracy while dramatically accelerating inference and cutting memory usage.

### What is the Goal of this Notebook?
While framework APIs like `torch.ao.quantization` hide the low-level arithmetic behind abstractions, this notebook takes you inside the engine room:
- **What is the exact formula** converting a 32-bit float into an integer?
- **Symmetric vs. Asymmetric:** How does zero-centering affect the quantization grid?
- **Granularities:** Why does **Per-Tensor** break on outliers, why does **Per-Channel** fix it, and why is **Per-Group** the secret behind modern LLM quantization (GGUF, GPTQ, AWQ)?
- **Bit-Packing:** How are 4-bit and 2-bit integers physically packed into `uint8` memory bytes?
- **Error & SQNR:** How do we quantify signal loss using Signal-to-Quantization-Noise Ratio (SQNR in dB)?
- **Live MNIST Model Conversion:** How `quantlab.RTNQuantizer` converts a PyTorch model into a memory-efficient `QuantizedLinear` layer."""
cells.append(new_markdown_cell(cell_0, id='cell-0-intro'))

# ==============================================================================
# Cell 1: Setup & Imports
# ==============================================================================
cell_1_code = r"""# ==============================================================================
# 0. SETUP & ENVIRONMENT
# ==============================================================================
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Ensure quantlab is accessible from repository source
for p in [Path('src').resolve(), Path('../src').resolve(), Path('/kaggle/working/quantlab_repo/src')]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import quantlab
from quantlab import (
    QuantConfig,
    Mode,
    Granularity,
    RTNQuantizer,
    QuantizedLinear,
    group_weight,
    ungroup_weight,
    compute_symmetric_params_batched,
    compute_asymmetric_params_batched,
    BitPacker,
    LinearDequantizer,
)

print(f"PyTorch version: {torch.__version__}")
print(f"quantlab version: {getattr(quantlab, '__version__', '0.1.0')}")
print("CUDA Available:", torch.cuda.is_available())"""
cells.append(new_code_cell(cell_1_code, id='cell-1-setup'))

# ==============================================================================
# Cell 2: Section 1 Markdown - The Core Math
# ==============================================================================
cell_2 = r"""## 1. The Core Mathematical Mapping

Quantization maps continuous real numbers $x \in [\alpha, \beta] \subset \mathbb{R}$ to a discrete grid of $b$-bit integers $q \in [q_{\min}, q_{\max}] \subset \mathbb{Z}$.

### The Forward (Quantization) Formula:
$$q = \text{clamp}\left(\left\lfloor \frac{x}{S} \right\rceil + Z, \; q_{\min}, \; q_{\max}\right)$$

### The Backward (Dequantization / Reconstruction) Formula:
$$\hat{x} = (q - Z) \times S$$

Where:
- **Scale ($S$)**: A positive floating-point step size representing the real-world distance between adjacent integer levels.
- **Zero-Point ($Z$)**: An integer ensuring that real-valued $0.0$ maps **exactly** to an integer in the discrete grid without rounding error (crucial for zero-padding, ReLU activations, and sparsity).
- **Rounding ($\lfloor \cdot \rceil$)**: Round-to-nearest integer.
- **Clamping**: Saturates values exceeding the minimum and maximum representable integer bounds."""
cells.append(new_markdown_cell(cell_2, id='cell-2-math-intro'))

# ==============================================================================
# Cell 3: Raw Python vs Quantlab Implementation of Quant / Dequant
# ==============================================================================
cell_3_code = r"""# ==============================================================================
# 1.1 STEP-BY-STEP PYTHON DEMO: From Real Numbers to Quantized Integers
# ==============================================================================
# Let's define a sample 1D tensor with varied values
x = torch.tensor([-0.8, -0.2, 0.0, 0.55, 1.20, 2.20], dtype=torch.float32)
bits = 4

# For 4-bit unsigned quantization: [qmin, qmax] = [0, 15]
qmin = 0
qmax = (1 << bits) - 1

# Calculate dynamic range ensuring 0.0 is contained
x_min = min(x.min().item(), 0.0)
x_max = max(x.max().item(), 0.0)

# 1. Compute Scale S
S = (x_max - x_min) / (qmax - qmin)

# 2. Compute Zero-point Z
Z = int(round(-x_min / S))
Z = max(qmin, min(qmax, Z))

# 3. Quantize
q = torch.clamp(torch.round(x / S) + Z, qmin, qmax).to(torch.int32)

# 4. Dequantize (Reconstruct)
x_hat = (q.float() - Z) * S

print(f"Original Float Tensor (x):\n  {x.tolist()}")
print(f"Computed Scale S: {S:.5f} (step size)")
print(f"Computed Zero-Point Z: {Z} (maps float 0.0 -> int {Z})")
print(f"Quantized Integers (q) [range {qmin}..{qmax}]:\n  {q.tolist()}")
print(f"Reconstructed Float Tensor (x_hat):\n  {[round(v, 4) for v in x_hat.tolist()]}")
print(f"Quantization Error (|x - x_hat|):\n  {[round(v, 4) for v in (x - x_hat).abs().tolist()]}")
print(f"Max Error: {(x - x_hat).abs().max().item():.5f} <= S/2 ({S/2:.5f})")"""
cells.append(new_code_cell(cell_3_code, id='cell-3-manual-quant'))

# ==============================================================================
# Cell 4: Section 2 Markdown - Symmetric vs Asymmetric
# ==============================================================================
cell_4 = r"""## 2. Symmetric vs. Asymmetric Quantization

| Property | Symmetric Quantization | Asymmetric Quantization |
| :--- | :--- | :--- |
| **Zero-Point ($Z$)** | Fixed to **$Z = 0$** | Dynamically computed integer $Z \in [q_{\min}, q_{\max}]$ |
| **Dynamic Range** | Centered on zero: $[-\max(|x|), \; +\max(|x|)]$ | Bounded by $[\min(x, 0), \; \max(x, 0)]$ |
| **Code Range (Signed/Unsigned)** | Signed: $[-2^{b-1}+1, \; 2^{b-1}-1]$ or $[-2^{b-1}, \; 2^{b-1}-1]$ | Unsigned: $[0, \; 2^b - 1]$ |
| **Hardware Efficiency** | **High** (GEMM is a simple integer dot-product: $W x = S_w S_x \sum q_w q_x$) | **Lower** (Requires zero-point subtraction offsets: $\sum (q_w - Z_w)(q_x - Z_x)$) |
| **Grid Utilization** | Suboptimal if distribution is skewed or strictly positive | Optimal (all $2^b$ buckets utilized) |

Let's test both on a positively skewed tensor (common in post-ReLU activations or biased weights)."""
cells.append(new_markdown_cell(cell_4, id='cell-4-sym-vs-asym-md'))

# ==============================================================================
# Cell 5: Code - Symmetric vs Asymmetric Comparison
# ==============================================================================
cell_5_code = r"""# ==============================================================================
# 2.1 SYMMETRIC VS ASYMMETRIC ON A SKEWED TENSOR
# ==============================================================================
# Positively skewed tensor (e.g. range [0.0, 7.5])
x_skewed = torch.tensor([0.0, 0.5, 1.2, 2.8, 4.5, 7.5], dtype=torch.float32)
bits = 4

# --- ASYMMETRIC (Unsigned [0..15]) ---
cfg_asym = QuantConfig(bits=bits, mode=Mode.ASYMMETRIC, granularity=Granularity.TENSOR)
grouped_asym, info_asym = group_weight(x_skewed.unsqueeze(0), cfg_asym.granularity)
scales_asym, zeros_asym, _, _ = compute_asymmetric_params_batched(grouped_asym, bits)

# Quantize and reconstruct
q_asym = torch.clamp(torch.round(grouped_asym / scales_asym.float().unsqueeze(1)) + zeros_asym.unsqueeze(1), 0, 15)
x_hat_asym = (q_asym - zeros_asym.unsqueeze(1)) * scales_asym.float().unsqueeze(1)
error_asym = (grouped_asym - x_hat_asym).abs()

# --- SYMMETRIC (Signed [-7..7]) ---
cfg_sym = QuantConfig(bits=bits, mode=Mode.SYMMETRIC, granularity=Granularity.TENSOR)
grouped_sym, info_sym = group_weight(x_skewed.unsqueeze(0), cfg_sym.granularity)
scales_sym, qmin_sym, qmax_sym = compute_symmetric_params_batched(grouped_sym, bits)

q_sym = torch.clamp(torch.round(grouped_sym / scales_sym.float().unsqueeze(1)), qmin_sym, qmax_sym)
x_hat_sym = q_sym * scales_sym.float().unsqueeze(1)
error_sym = (grouped_sym - x_hat_sym).abs()

print("=" * 65)
print(f"INPUT TENSOR: {x_skewed.tolist()}")
print("=" * 65)
print("ASYMMETRIC (4-bit):")
print(f"  Scale: {scales_asym.item():.4f} | Zero-Point: {zeros_asym.item()}")
print(f"  Integer Codes (0..15): {q_asym.int().tolist()[0]}")
print(f"  Reconstructed Float:   {[round(v, 3) for v in x_hat_asym.tolist()[0]]}")
print(f"  Mean Absolute Error:   {error_asym.mean().item():.5f}")

print("\nSYMMETRIC (4-bit):")
print(f"  Scale: {scales_sym.item():.4f} | Zero-Point: 0 (Fixed)")
print(f"  Integer Codes (-7..7): {q_sym.int().tolist()[0]}  <-- Note: negative codes unused!")
print(f"  Reconstructed Float:   {[round(v, 3) for v in x_hat_sym.tolist()[0]]}")
print(f"  Mean Absolute Error:   {error_sym.mean().item():.5f}")
print("=" * 65)
print(f"-> Asymmetric error is {error_sym.mean().item() / error_asym.mean().item():.1f}x LOWER because it uses all 16 levels!")"""
cells.append(new_code_cell(cell_5_code, id='cell-5-sym-vs-asym-code'))

# ==============================================================================
# Cell 6: Section 3 Markdown - Granularities
# ==============================================================================
cell_6 = r"""## 3. Granularities: Per-Tensor vs. Per-Channel vs. Per-Group

In a 2D weight matrix $W \in \mathbb{R}^{\text{out} \times \text{in}}$, how often do we compute a new Scale ($S$) and Zero-Point ($Z$)?

```
1. PER-TENSOR:               2. PER-CHANNEL (Row):        3. PER-GROUP (e.g. G=4):
[ .  .  .  .  .  .  .  . ]    [ .  .  .  .  .  .  .  . ] -> S0  [ . . . . | . . . . ] -> S0, S1
[ .  .  .  .  .  .  .  . ]    [ .  .  .  .  .  .  .  . ] -> S1  [ . . . . | . . . . ] -> S2, S3
[ .  .  !  .  .  .  .  . ]    [ .  .  !  .  .  .  .  . ] -> S2  [ . . ! . | . . . . ] -> S4, S5
[ .  .  .  .  .  .  .  . ]    [ .  .  .  .  .  .  .  . ] -> S3  [ . . . . | . . . . ] -> S6, S7
           |
         Single S
```

### The Outlier Problem:
Real-world neural network weights and activations contain **extreme outliers** (e.g., $+15.0$ among normal values around $\pm 0.2$).
- **Per-Tensor:** The single scale $S = \frac{15.0}{q_{\max}}$ becomes huge. Small values round to zero, ruining the entire matrix!
- **Per-Channel:** The outlier row gets a large scale, but all other rows retain small scales and high precision.
- **Per-Group:** Even inside the outlier row, only the specific block containing the outlier gets a larger scale. Other blocks in the same row stay protected!"""
cells.append(new_markdown_cell(cell_6, id='cell-6-granularity-md'))

# ==============================================================================
# Cell 7: Code - Granularity Demonstration
# ==============================================================================
cell_7_code = r"""# ==============================================================================
# 3.1 GRANULARITY EXPERIMENT: An Outlier in Row 2
# ==============================================================================
# Create a 4x8 weight matrix with normal weights ~ [-0.3, 0.3], but ONE outlier in row 2
torch.manual_seed(100)
W = torch.randn(4, 8) * 0.15
W[2, 3] = 12.0  # Extreme outlier!

print("Original Weight Matrix W (4x8) [Notice W[2,3] = 12.00]:")
for i, row in enumerate(W):
    print(f"  Row {i}: [" + ", ".join(f"{v:6.2f}" for v in row.tolist()) + "]")

bits = 4

# Function to run quant/dequant and calculate error
def evaluate_granularity(W_mat, gran, g_size=None):
    cfg = QuantConfig(bits=bits, mode=Mode.SYMMETRIC, granularity=gran, group_size=g_size)
    grouped, info = group_weight(W_mat, cfg.granularity, cfg.group_size)
    scales, qmin, qmax = compute_symmetric_params_batched(grouped, bits)
    
    # Quantize & dequantize
    scales_f = scales.float().unsqueeze(1)
    codes = torch.clamp(torch.round(grouped / scales_f), qmin, qmax)
    w_hat = codes * scales_f
    w_rec = ungroup_weight(w_hat, info)
    
    # Calculate row-by-row MSE
    row_mses = ((W_mat - w_rec) ** 2).mean(dim=1).tolist()
    total_mse = ((W_mat - w_rec) ** 2).mean().item()
    return cfg.name, scales.numel(), row_mses, total_mse, w_rec

results = [
    evaluate_granularity(W, Granularity.TENSOR),
    evaluate_granularity(W, Granularity.CHANNEL),
    evaluate_granularity(W, Granularity.GROUP, g_size=4),
]

print("\n" + "=" * 78)
print(f"{'Granularity':22s} | {'# Scales':8s} | {'Row 0 MSE':10s} | {'Row 2 (Outlier) MSE':20s} | {'Total MSE':10s}")
print("=" * 78)
for name, n_sc, row_mse, tot_mse, _ in results:
    print(f"{name:22s} | {n_sc:8d} | {row_mse[0]:10.6f} | {row_mse[2]:20.6f} | {tot_mse:10.6f}")
print("=" * 78)
print("Observation: In Per-Tensor, Row 0's MSE is wiped out by Row 2's outlier.")
print("In Per-Channel & Per-Group, Row 0 retains near-zero error!")"""
cells.append(new_code_cell(cell_7_code, id='cell-7-granularity-code'))

# ==============================================================================
# Cell 8: Section 4 Markdown - Bit Packing
# ==============================================================================
cell_8 = r"""## 4. Physical Bit-Packing: How Low-Bit Integers Are Stored

When quantizing to **4-bit** or **2-bit**, PyTorch does not have native `torch.int4` or `torch.int2` tensor dtypes. If stored as standard `torch.int8`, memory is **not saved**!

To achieve real memory reduction on mobile or edge devices, [`quantlab.BitPacker`](file:///home/hi-z-k/repos/Quantizer/src/quantlab/bit_packer.py) packs multiple sub-byte integers into a single `uint8` byte:
- **4-bit packing (2 integers per byte):**
  $$\text{byte} = q_0 \; | \; (q_1 \ll 4)$$
- **2-bit packing (4 integers per byte):**
  $$\text{byte} = q_0 \; | \; (q_1 \ll 2) \; | \; (q_2 \ll 4) \; | \; (q_3 \ll 6)$$

Let's see the bit manipulation in action!"""
cells.append(new_markdown_cell(cell_8, id='cell-8-packing-md'))

# ==============================================================================
# Cell 9: Code - Bit Packing
# ==============================================================================
cell_9_code = r"""# ==============================================================================
# 4.1 BIT PACKING DEMO WITH quantlab.BitPacker
# ==============================================================================
packer = BitPacker()

# Four 4-bit unsigned integers (all values in range 0..15)
codes_4bit = torch.tensor([3, 12, 7, 1], dtype=torch.uint8)

# Pack into bytes
packed_bytes = packer.pack(codes_4bit, bits=4)

# Unpack back
unpacked_codes = packer.unpack(packed_bytes, bits=4, count=len(codes_4bit))

print("4-BIT PACKING DEMONSTRATION:")
print(f"  Original 4-bit codes:    {codes_4bit.tolist()}")
print(f"  Binary representation:   {[f'{v:04b}' for v in codes_4bit.tolist()]}")
print(f"  Packed bytes (uint8):    {packed_bytes.tolist()}")
print(f"  Byte 0 binary:           {packed_bytes[0].item():08b}  (3: 0011 | 12<<4: 11000000 -> 11000011 = 195)")
print(f"  Byte 1 binary:           {packed_bytes[1].item():08b}  (7: 0111 |  1<<4: 00010000 -> 00010111 = 23)")
print(f"  Unpacked codes verified: {unpacked_codes.tolist()} (Match: {torch.equal(codes_4bit, unpacked_codes)})")
print(f"  Compression: {codes_4bit.numel()} elements stored in {packed_bytes.numel()} bytes (2x reduction vs int8, 8x vs float32)!")"""
cells.append(new_code_cell(cell_9_code, id='cell-9-packing-code'))

# ==============================================================================
# Cell 10: Section 5 Markdown - SQNR and Error Analysis
# ==============================================================================
cell_10 = r"""## 5. Signal-to-Quantization-Noise Ratio (SQNR)

How do we scientifically quantify the quality loss introduced by quantization? We use **Signal-to-Quantization-Noise Ratio (SQNR)** measured in decibels (dB):

$$\text{SQNR (dB)} = 10 \cdot \log_{10}\left(\frac{\sum_{i} x_i^2}{\sum_{i} (x_i - \hat{x}_i)^2 + \epsilon}\right)$$

### The 6 dB Rule of Thumb:
In signal processing, each additional quantization bit provides approximately **$+6.02\text{ dB}$** of dynamic range:
- **8-bit:** $\sim 45 - 50\text{ dB}$ (Near-perfect signal retention)
- **4-bit:** $\sim 20 - 25\text{ dB}$ (Good quality, sufficient for LLM weights)
- **2-bit:** $\sim 3 - 6\text{ dB}$ (Severe degradation, catastrophic for uncalibrated models)"""
cells.append(new_markdown_cell(cell_10, id='cell-10-sqnr-md'))

# ==============================================================================
# Cell 11: Code - SQNR Comparison Across Bit Widths
# ==============================================================================
cell_11_code = r"""# ==============================================================================
# 5.1 SQNR AND ERROR VS BIT-WIDTH SWEEP
# ==============================================================================
torch.manual_seed(42)
W_test = torch.randn(64, 64)  # 4096 elements

bit_widths = [8, 6, 5, 4, 3, 2]
sqnr_results = []
mse_results = []

for b in bit_widths:
    cfg = QuantConfig(bits=b, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32)
    grouped, info = group_weight(W_test, cfg.granularity, cfg.group_size)
    scales, qmin, qmax = compute_symmetric_params_batched(grouped, b)
    
    scales_f = scales.float().unsqueeze(1)
    codes = torch.clamp(torch.round(grouped / scales_f), qmin, qmax)
    w_rec = ungroup_weight(codes * scales_f, info)
    
    # Calculate MSE & SQNR
    signal_pwr = (W_test ** 2).sum().item()
    noise_pwr = ((W_test - w_rec) ** 2).sum().item()
    mse = ((W_test - w_rec) ** 2).mean().item()
    sqnr_db = 10.0 * np.log10(signal_pwr / (noise_pwr + 1e-12))
    
    sqnr_results.append(sqnr_db)
    mse_results.append(mse)

# Plot SQNR and MSE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), dpi=120)

ax1.plot(bit_widths, sqnr_results, marker='o', color='#0072B2', linewidth=2)
ax1.set_xlabel('Bit Width (b)')
ax1.set_ylabel('SQNR (dB)')
ax1.set_title('Signal Quality (SQNR dB) vs Bit-Width')
ax1.grid(True, linestyle=':', alpha=0.6)
for b, s in zip(bit_widths, sqnr_results):
    ax1.annotate(f"{s:.1f} dB", (b, s), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=8)

ax2.plot(bit_widths, mse_results, marker='s', color='#D55E00', linewidth=2)
ax2.set_yscale('log')
ax2.set_xlabel('Bit Width (b)')
ax2.set_ylabel('Mean Squared Error (log scale)')
ax2.set_title('Quantization Noise (MSE) vs Bit-Width')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()"""
cells.append(new_code_cell(cell_11_code, id='cell-11-sqnr-plot'))

# ==============================================================================
# Cell 12: Section 6 Markdown - Replicating the Video PTQ Layer Conversion
# ==============================================================================
cell_12 = r"""## 6. Replicating the Video: MNIST PTQ Model Conversion

In the tutorial video, an MNIST classifier is transformed via PTQ:
1. An initial floating-point model (e.g. `nn.Linear(784, 128)` followed by `nn.Linear(128, 10)`) stores all weights in **32-bit floats**.
2. **`quantlab.RTNQuantizer`** scans all `nn.Linear` layers, computes scales, and replaces them in-place with **`quantlab.QuantizedLinear`**.
3. The weights are stored as packed low-bit integers in memory.
4. On inference, `QuantizedLinear.forward()` performs dynamic on-the-fly dequantization and matrix multiplication.

Let's build the exact MNIST architecture and observe the size drop from **Float32 &rarr; 8-bit &rarr; 4-bit**!"""
cells.append(new_markdown_cell(cell_12, id='cell-12-mnist-md'))

# ==============================================================================
# Cell 13: Code - MNIST Model Creation, Conversion & Size Drop
# ==============================================================================
cell_13_code = r"""# ==============================================================================
# 6.1 MNIST PTQ ARCHITECTURE CONVERSION
# ==============================================================================
class MNISTClassifier(nn.Module):
    """ + '"""Standard 2-layer MLP for MNIST classification (784 -> 128 -> 10)."""' + r"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def get_model_size_kb(model):
    """ + '"""Calculates model state_dict size in Kilobytes (KB)."""' + r"""
    total_bytes = 0
    for name, param in model.state_dict().items():
        total_bytes += param.numel() * param.element_size()
    return total_bytes / 1024.0

# 1. Instantiate Float32 model
torch.manual_seed(42)
float_model = MNISTClassifier()
size_fp32 = get_model_size_kb(float_model)

# Test input batch (4 images of size 28x28 flattened to 784)
sample_input = torch.randn(4, 784)
y_fp32 = float_model(sample_input)

# 2. Convert to 8-bit Quantized Model
model_int8 = MNISTClassifier()
model_int8.load_state_dict(float_model.state_dict())
cfg_int8 = QuantConfig(bits=8, mode=Mode.SYMMETRIC, granularity=Granularity.CHANNEL)
RTNQuantizer(cfg_int8).quantize(model_int8)
size_int8 = get_model_size_kb(model_int8)
y_int8 = model_int8(sample_input)

# 3. Convert to 4-bit Quantized Model
model_int4 = MNISTClassifier()
model_int4.load_state_dict(float_model.state_dict())
cfg_int4 = QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32)
RTNQuantizer(cfg_int4).quantize(model_int4)
size_int4 = get_model_size_kb(model_int4)
y_int4 = model_int4(sample_input)

print("=" * 65)
print("MNIST MODEL PTQ CONVERSION REPORT")
print("=" * 65)
print(f"Original Float32 Model Size: {size_fp32:8.2f} KB (Baseline 100%)")
print(f"Quantized 8-Bit Model Size:  {size_int8:8.2f} KB (Drop to {size_int8/size_fp32*100:.1f}%)")
print(f"Quantized 4-Bit Model Size:  {size_int4:8.2f} KB (Drop to {size_int4/size_fp32*100:.1f}%)")
print("-" * 65)
print("Converted Layers in model_int4:")
for name, m in model_int4.named_modules():
    if isinstance(m, QuantizedLinear):
        print(f"  {name}: {m}")
print("-" * 65)
print(f"Output Logit Deviation (Float32 vs 8-bit): {(y_fp32 - y_int8).abs().max().item():.5f}")
print(f"Output Logit Deviation (Float32 vs 4-bit): {(y_fp32 - y_int4).abs().max().item():.5f}")
print("=" * 65)"""
cells.append(new_code_cell(cell_13_code, id='cell-13-mnist-code'))

# ==============================================================================
# Cell 14: Section 7 Markdown - Interactive Playground
# ==============================================================================
cell_14 = r"""## 7. Interactive Playground: Quantize Any Tensor in 1 Line

Use the helper function below to experiment with custom weights, bit widths, granularities, and symmetric/asymmetric modes. Try inputting your own matrices!"""
cells.append(new_markdown_cell(cell_14, id='cell-14-playground-md'))

# ==============================================================================
# Cell 15: Code - Interactive Helper Function
# ==============================================================================
cell_15_code = r"""# ==============================================================================
# 7.1 INTERACTIVE TENSOR QUANTIZATION INSPECTOR
# ==============================================================================
def inspect_tensor_quantization(tensor, bits=4, mode='symmetric', granularity='channel', group_size=32):
    """ + '"""Full inspection utility for analyzing quantization mechanics on any tensor."""' + r"""
    mode_enum = Mode.SYMMETRIC if mode.lower() == 'symmetric' else Mode.ASYMMETRIC
    gran_map = {
        'tensor': Granularity.TENSOR,
        'channel': Granularity.CHANNEL,
        'group': Granularity.GROUP
    }
    gran_enum = gran_map[granularity.lower()]
    
    cfg = QuantConfig(bits=bits, mode=mode_enum, granularity=gran_enum, group_size=group_size)
    grouped, info = group_weight(tensor, cfg.granularity, cfg.group_size)
    
    if mode_enum == Mode.SYMMETRIC:
        scales, qmin, qmax = compute_symmetric_params_batched(grouped, bits)
        zeros = None
        scales_f = scales.float().unsqueeze(1)
        codes = torch.clamp(torch.round(grouped / scales_f), qmin, qmax)
        w_rec = ungroup_weight(codes * scales_f, info)
    else:
        scales, zeros, qmin, qmax = compute_asymmetric_params_batched(grouped, bits)
        scales_f = scales.float().unsqueeze(1)
        zeros_f = zeros.float().unsqueeze(1)
        codes = torch.clamp(torch.round(grouped / scales_f) + zeros_f, qmin, qmax)
        w_rec = ungroup_weight((codes - zeros_f) * scales_f, info)
        
    error = (tensor - w_rec).abs()
    mse = ((tensor - w_rec) ** 2).mean().item()
    signal_pwr = (tensor ** 2).sum().item()
    noise_pwr = (error ** 2).sum().item()
    sqnr = 10.0 * np.log10(signal_pwr / (noise_pwr + 1e-12))
    
    print(f"=== QUANTIZATION INSPECTION: {cfg.name} ===")
    print(f"Input Shape: {list(tensor.shape)} | Groups: {scales.numel()} scales computed")
    print(f"Scale Stats: min={scales.min().item():.5f}, max={scales.max().item():.5f}, mean={scales.float().mean().item():.5f}")
    if zeros is not None:
        print(f"Zero-Point Stats: min={zeros.min().item()}, max={zeros.max().item()}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Max Absolute Error:       {error.max().item():.6f}")
    print(f"SQNR (Quality):           {sqnr:.2f} dB")
    return w_rec

# Example: Run on a 16x32 random weight tensor
custom_w = torch.randn(16, 32)
reconstructed = inspect_tensor_quantization(custom_w, bits=4, mode='symmetric', granularity='group', group_size=16)"""
cells.append(new_code_cell(cell_15_code, id='cell-15-playground-code'))

# Add all cells to notebook
nb.cells = cells

# Validate notebook
nbformat.validate(nb)
print("Notebook validation passed successfully!")

# Write to destination
out_path = Path("notebooks/quantization_math_and_tensors.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Successfully generated {out_path} ({out_path.stat().st_size} bytes)")
