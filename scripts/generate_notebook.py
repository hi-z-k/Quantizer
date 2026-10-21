#!/usr/bin/env python3
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.metadata = {
    'language_info': {'name': 'python'},
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
}

# Define cells
cell_0_md = '''# Quantization Experiments with quantlab

This notebook evaluates post-training weight quantization across multiple model architectures and sizes using the [quantlab](https://github.com/hi-z-k/Quantizer) library.

### Experiments Covered:
- **Q1 (E2)**: Bit-width sweep (8, 6, 5, 4, 3, 2 bit symmetric group 32) & diminishing returns.
- **Q2 (E3)**: Granularity comparison (per-tensor, per-channel, group 128, group 32).
- **Q3 (E1)**: Theoretical vs real packed byte measurements and component breakdown.
- **Q5 (E5)**: Cross-architecture & cross-scale resilience (SmolLM2-135M, SmolLM2-360M, TinyLlama-1.1B).
- **Q6 (E6)**: Correlation between weight SQNR (dB) and perplexity degradation.
- **Constraint & Winner Selection**: TinyLlama-1.1B mobile budget (Galaxy M15, 6GB RAM: file $\\le$ 1.2 GB, $\\ge$ 60% reduction vs FP16).
- **Native & GGUF Export**: Exporting winner to safetensors and Q8_0 / Q4_0 GGUF.
- **Hugging Face Hub**: Automated publishing of models and model card.'''

cell_1_code = '''# ==============================================================================
# PARAMETERS CELL - Everything editable here
# ==============================================================================
GITHUB_USER = 'hi-z-k'
REPO_NAME = 'Quantizer'
HF_USER = 'Hizkealhizk'
HF_REPO = 'tinyllama-1.1b-chat-quantlab'
SEED = 0
EVAL_TOKENS = 100_000
WINDOW = 1024
MAX_PPL_INCREASE_PCT = 5.0
MAX_FILE_MB = 1200
MIN_REDUCTION_PCT = 60
EVAL_DEADLINE_S = 3600  # Stop starting new evals after 1 hour

# Model list in order of evaluation
MODELS = [
    ('HuggingFaceTB/SmolLM2-135M-Instruct', '135M'),
    ('TinyLlama/TinyLlama-1.1B-Chat-v1.0', '1.1B'),
    ('HuggingFaceTB/SmolLM2-360M-Instruct', '360M'),
]

print(f'Configured parameters: SEED={SEED}, EVAL_TOKENS={EVAL_TOKENS}, DEADLINE={EVAL_DEADLINE_S}s')'''

cell_2_code = '''# ==============================================================================
# SETUP: Clone library, install, configure outputs directory & record meta.json
# ==============================================================================
import os, sys, time, json, shutil, subprocess, gc
from pathlib import Path
import torch

# Set random seed for reproducibility
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Determine runtime paths (Kaggle vs local)
IS_KAGGLE = os.path.exists('/kaggle/working')
BASE_DIR = Path('/kaggle/working') if IS_KAGGLE else Path('.').resolve()
OUTPUTS_DIR = BASE_DIR / 'outputs'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / 'figures').mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / 'tables').mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / 'native').mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / 'gguf').mkdir(parents=True, exist_ok=True)

# Install or clone quantlab
if IS_KAGGLE:
    repo_path = BASE_DIR / 'quantlab_repo'
    if not repo_path.exists():
        print(f'Cloning quantlab from https://github.com/{GITHUB_USER}/{REPO_NAME}...')
        subprocess.run(['git', 'clone', '--depth', '1', f'https://github.com/{GITHUB_USER}/{REPO_NAME}.git', str(repo_path)], check=True)
    else:
        subprocess.run(['git', '-C', str(repo_path), 'pull'], check=False)

    # Install quantlab in editable mode with --no-deps to avoid overriding Kaggle packages
    print('Installing quantlab (--no-deps)...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '-e', str(repo_path)], check=True)

    # Install optional lightweight export packages (gguf, sentencepiece) without touching core packages
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '-q', 'gguf', 'sentencepiece'], check=False)

# Add repository src directory directly to sys.path to guarantee immediate import
candidates = [
    BASE_DIR / 'quantlab_repo' / 'src',
    BASE_DIR / 'src',
    Path('src').resolve(),
    Path('/kaggle/working/quantlab_repo/src'),
]
for c in candidates:
    if c.exists() and str(c) not in sys.path:
        sys.path.insert(0, str(c))

import importlib
importlib.invalidate_caches()

# Check transformers compatibility with torch:
# On Kaggle, if an old session upgraded transformers to 5.x, it disables torch (< 2.6).
# Fix it by pinning transformers < 5.0.0
import transformers
try:
    t_major = int(transformers.__version__.split('.')[0])
    torch_major = int(torch.__version__.split('.')[0])
    torch_minor = int(torch.__version__.split('.')[1])
    if (torch_major, torch_minor) < (2, 6) and t_major >= 5:
        print(f'Detected torch {torch.__version__} (< 2.6) with incompatible transformers {transformers.__version__}.')
        print('Downgrading transformers to < 4.46.0 for torch compatibility...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers>=4.38.0,<4.46.0'], check=True)
        importlib.reload(transformers)
except Exception as e:
    pass

import quantlab

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
meta_data = {
    'quantlab_version': getattr(quantlab, '__version__', '0.1.0'),
    'torch_version': torch.__version__,
    'transformers_version': transformers.__version__,
    'gpu_name': gpu_name,
    'seed': SEED,
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}
with open(OUTPUTS_DIR / 'meta.json', 'w') as f:
    json.dump(meta_data, f, indent=2)

print(f'Setup complete on {gpu_name}. Meta: {meta_data}')'''

cell_3_code = '''# ==============================================================================
# CONFIGS: Define quantization configurations for each model using QuantConfig
# ==============================================================================
from quantlab import QuantConfig, Mode, Granularity

# 135M gets full bit sweep and granularity sweep (all symmetric)
CONFIGS_135M = [
    # Group 32 bit sweep (bits 8, 6, 5, 4, 3, 2)
    QuantConfig(bits=8, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32),
    QuantConfig(bits=6, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32),
    QuantConfig(bits=5, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32),
    QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32),
    QuantConfig(bits=3, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32),
    QuantConfig(bits=2, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32),
    # Granularity sweep for 8-bit and 4-bit
    QuantConfig(bits=8, mode=Mode.SYMMETRIC, granularity=Granularity.TENSOR),
    QuantConfig(bits=8, mode=Mode.SYMMETRIC, granularity=Granularity.CHANNEL),
    QuantConfig(bits=8, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=128),
    QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.TENSOR),
    QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.CHANNEL),
    QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=128),
]

# 360M and 1.1B configs: group 32 (8, 4, 3, 2) + 4-bit per-channel
CONFIGS_COMMON = [
    QuantConfig(bits=8, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32),
    QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32),
    QuantConfig(bits=3, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32),
    QuantConfig(bits=2, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32),
    QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.CHANNEL),
]

MODEL_CONFIGS = {
    '135M': CONFIGS_135M,
    '1.1B': CONFIGS_COMMON,
    '360M': CONFIGS_COMMON,
}

for tag, cfgs in MODEL_CONFIGS.items():
    print(f'{tag}: {len(cfgs)} quantization configs (+ 1 FP16 baseline)')'''

cell_4_code = '''# ==============================================================================
# EXPERIMENT LOOP: Evaluates FP16 baseline and all quant configs per model
# ==============================================================================
import csv, math
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantlab import (
    WikiText2Dataset,
    PerplexityMetric,
    WeightErrorMetric,
    SimulatedRTNQuantizer,
    WeightSnapshot,
    estimate_size,
)

RESULTS_FILE = OUTPUTS_DIR / 'results.csv'
CSV_COLUMNS = [
    'model', 'config_name', 'bits', 'granularity', 'group_size',
    'effective_bits', 'size_mb', 'reduction_vs_fp16_pct',
    'ppl', 'ppl_fp16', 'delta_ppl_pct', 'mse', 'sqnr_db', 'seconds', 'status'
]

# Initialise results.csv with header if not exists
existing_keys = set()
if not RESULTS_FILE.exists():
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
else:
    with open(RESULTS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            existing_keys.add((r['model'], r['config_name']))

start_overall_time = time.time()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Starting experiment evaluations on {device} (Deadline: {EVAL_DEADLINE_S}s)...')

for model_id, model_tag in MODELS:
    print(f'\\n=== Loading Model: {model_tag} ({model_id}) ===')
    t_model_start = time.time()

    # Free cache before model loading
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = WikiText2Dataset(tokenizer, split='test', max_tokens=EVAL_TOKENS, window=WINDOW)
    ppl_metric = PerplexityMetric(batch_size=8)

    # Load in float16
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    except Exception as e:
        print(f'Failed to load {model_id} in float16: {e}. Falling back to float32.')
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)

    # Compute FP16 baseline
    t0 = time.time()
    ppl_fp16 = ppl_metric.measure(model, dataset)
    if not math.isfinite(ppl_fp16):
        print(f'Warning: FP16 baseline PPL was {ppl_fp16}. Reloading in float32...')
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
        ppl_fp16 = ppl_metric.measure(model, dataset)

    dt_baseline = time.time() - t0
    print(f'[{model_tag}] FP16 baseline PPL: {ppl_fp16:.2f} ({dt_baseline:.1f}s)')

    # Save FP16 baseline row if not present
    if (model_tag, 'fp16') not in existing_keys:
        fp16_row = {
            'model': model_tag, 'config_name': 'fp16', 'bits': 16, 'granularity': 'none',
            'group_size': 0, 'effective_bits': 16.0,
            'size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024*1024),
            'reduction_vs_fp16_pct': 0.0, 'ppl': ppl_fp16, 'ppl_fp16': ppl_fp16,
            'delta_ppl_pct': 0.0, 'mse': 0.0, 'sqnr_db': 100.0, 'seconds': dt_baseline, 'status': 'ok'
        }
        with open(RESULTS_FILE, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(fp16_row)
        existing_keys.add((model_tag, 'fp16'))

    # Take snapshot of original weights
    snapshot = WeightSnapshot().take(model, skip_names=('lm_head',))
    error_metric = WeightErrorMetric(snapshot)

    # Evaluate all configs
    for cfg in MODEL_CONFIGS[model_tag]:
        elapsed_total = time.time() - start_overall_time
        if elapsed_total > EVAL_DEADLINE_S:
            print(f'Time deadline reached ({elapsed_total:.0f}s > {EVAL_DEADLINE_S}s). Skipping remaining evals.')
            break

        if (model_tag, cfg.name) in existing_keys:
            print(f'[{model_tag}] {cfg.name}: already computed, skipping.')
            continue

        t_eval = time.time()
        try:
            # Restore original weights
            snapshot.restore(model)

            # Simulate quantization in place
            SimulatedRTNQuantizer(cfg, skip_names=('lm_head',)).quantize(model)

            # Measure metrics
            err_report = error_metric.measure(model)
            ppl = ppl_metric.measure(model, dataset)
            size_rep = estimate_size(model, cfg, skip_names=('lm_head',))

            delta_ppl = ((ppl - ppl_fp16) / ppl_fp16) * 100.0 if ppl_fp16 > 0 else float('nan')
            dt = time.time() - t_eval

            row = {
                'model': model_tag, 'config_name': cfg.name, 'bits': cfg.bits,
                'granularity': cfg.granularity.value, 'group_size': cfg.group_size or 0,
                'effective_bits': round(size_rep.effective_bits, 2),
                'size_mb': round(size_rep.total_mb, 1),
                'reduction_vs_fp16_pct': round(size_rep.reduction_vs_fp16_pct, 1),
                'ppl': round(ppl, 2), 'ppl_fp16': round(ppl_fp16, 2),
                'delta_ppl_pct': round(delta_ppl, 2),
                'mse': round(err_report['mse'], 6),
                'sqnr_db': round(err_report['sqnr_db'], 2),
                'seconds': round(dt, 1), 'status': 'ok'
            }
            status_msg = f'PPL={ppl:.2f} (Δ={delta_ppl:+.1f}%), SQNR={err_report["sqnr_db"]:.1f}dB, Size={size_rep.total_mb:.1f}MB'
        except Exception as e:
            dt = time.time() - t_eval
            print(f'Error evaluating {model_tag} {cfg.name}: {e}')
            row = {
                'model': model_tag, 'config_name': cfg.name, 'bits': cfg.bits,
                'granularity': cfg.granularity.value, 'group_size': cfg.group_size or 0,
                'effective_bits': 0, 'size_mb': 0, 'reduction_vs_fp16_pct': 0,
                'ppl': float('nan'), 'ppl_fp16': ppl_fp16, 'delta_ppl_pct': float('nan'),
                'mse': float('nan'), 'sqnr_db': float('nan'), 'seconds': round(dt, 1), 'status': f'failed: {e}'
            }
            status_msg = f'FAILED: {e}'

        # Append immediately to CSV
        with open(RESULTS_FILE, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)
        existing_keys.add((model_tag, cfg.name))
        print(f'[{model_tag}] {cfg.name}: {status_msg} ({dt:.1f}s)')

    # Free model and memory
    del model, snapshot
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f'\\nAll evaluations completed! Results saved to {RESULTS_FILE}')'''

cell_5_code = '''# ==============================================================================
# E1 SIZES (135M only): Measure formula vs real packed bytes and components
# ==============================================================================
import csv
from quantlab import RTNQuantizer, measure_size, estimate_size

SIZE_ROWS_FILE = OUTPUTS_DIR / 'size_rows.csv'
size_columns = [
    'model', 'config_name', 'formula_bytes', 'real_bytes',
    'packed_weight_bytes', 'scale_bytes', 'zero_point_bytes', 'fp16_rest_bytes',
    'effective_bits', 'reduction_pct'
]

e1_configs = [
    ('fp16', None),
    ('int8_sym_group32', QuantConfig(bits=8, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32)),
    ('int4_sym_group128', QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=128)),
    ('int4_sym_group32', QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32)),
    ('int4_sym_channel', QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.CHANNEL)),
    ('int2_sym_group32', QuantConfig(bits=2, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32)),
]

print('Measuring E1 real packed sizes on SmolLM2-135M...')
size_rows = []

for name, cfg in e1_configs:
    # Fresh model copy for each config
    m = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct', torch_dtype=torch.float16)
    if cfg is None:
        # FP16 baseline
        rep = measure_size(m)
        formula_rep = rep
    else:
        formula_rep = estimate_size(m, cfg, skip_names=('lm_head',))
        RTNQuantizer(cfg, skip_names=('lm_head',)).quantize(m)
        rep = measure_size(m)

    row = {
        'model': '135M', 'config_name': name,
        'formula_bytes': formula_rep.total_bytes,
        'real_bytes': rep.total_bytes,
        'packed_weight_bytes': rep.packed_weight_bytes,
        'scale_bytes': rep.scale_bytes,
        'zero_point_bytes': rep.zero_point_bytes,
        'fp16_rest_bytes': rep.fp16_rest_bytes,
        'effective_bits': round(rep.effective_bits, 2),
        'reduction_pct': round(rep.reduction_vs_fp16_pct, 1),
    }
    size_rows.append(row)
    print(f'  {name:20s}: real={rep.total_mb:.1f}MB, formula={formula_rep.total_mb:.1f}MB, diff={abs(rep.total_bytes - formula_rep.total_bytes)} bytes')
    del m
    gc.collect()

with open(SIZE_ROWS_FILE, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=size_columns)
    writer.writeheader()
    writer.writerows(size_rows)

print(f'Saved size breakdown to {SIZE_ROWS_FILE}')'''

cell_6_code = '''# ==============================================================================
# FIGURES: Generates all 5 report figures in PDF and 300 dpi PNG
# Styling: 8 pt font, labelled axes with units, Okabe-Ito colors, no titles
# ==============================================================================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configure matplotlib styling
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 8,
    'font.family': 'sans-serif',
})

# Okabe-Ito color palette
COLOR_135M = '#0072B2'  # Blue
COLOR_360M = '#D55E00'  # Vermilion / Orange
COLOR_11B  = '#009E73'  # Bluish Green
MODEL_COLORS = {'135M': COLOR_135M, '360M': COLOR_360M, '1.1B': COLOR_11B}

df_results = pd.read_csv(RESULTS_FILE)
df_sizes = pd.read_csv(SIZE_ROWS_FILE)
fig_dir = OUTPUTS_DIR / 'figures'

# ------------------------------------------------------------------------------
# Fig E1: Size breakdown (stacked bars in MB)
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.3, 2.3))
configs_e1 = df_sizes['config_name'].tolist()
labels_e1 = [c.replace('int', '').replace('_sym_', ' ') for c in configs_e1]
x_idx = np.arange(len(configs_e1))

pw_mb = df_sizes['packed_weight_bytes'] / (1024 * 1024)
sc_mb = (df_sizes['scale_bytes'] + df_sizes['zero_point_bytes']) / (1024 * 1024)
fp_mb = df_sizes['fp16_rest_bytes'] / (1024 * 1024)

ax.bar(x_idx, pw_mb, label='Weights', color='#56B4E9')
ax.bar(x_idx, sc_mb, bottom=pw_mb, label='Scales', color='#E69F00')
ax.bar(x_idx, fp_mb, bottom=pw_mb + sc_mb, label='FP16 Kept', color='#999999')

fp16_total_mb = df_sizes[df_sizes['config_name'] == 'fp16']['real_bytes'].values[0] / (1024 * 1024)
ax.axhline(0.40 * fp16_total_mb, color='red', linestyle='--', linewidth=1, label='40% FP16 limit')
ax.set_xticks(x_idx)
ax.set_xticklabels(labels_e1, rotation=35, ha='right')
ax.set_ylabel('Size (MB)')
ax.legend(loc='upper right', frameon=False)
fig.tight_layout()
fig.savefig(fig_dir / 'fig_e1_size_breakdown.pdf')
fig.savefig(fig_dir / 'fig_e1_size_breakdown.png', dpi=300)
plt.close(fig)
print('Saved fig_e1_size_breakdown')

# ------------------------------------------------------------------------------
# Fig E2: Diminishing returns (two panels: line + step return bars)
# ------------------------------------------------------------------------------
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.8, 2.3))
df_e2 = df_results[(df_results['model'] == '135M') & (df_results['granularity'] == 'group') & (df_results['group_size'] == 32)].copy()
df_e2 = df_e2.sort_values('effective_bits', ascending=False)

# Panel (a): line of delta_ppl against effective bits (high to low)
ax_a.plot(df_e2['effective_bits'], df_e2['delta_ppl_pct'], marker='o', color=COLOR_135M, linewidth=1.5)
ax_a.set_yscale('log')
ax_a.invert_xaxis()
ax_a.set_xlabel('Effective Bits per Weight')
ax_a.set_ylabel('Perplexity Increase (%)')
ax_a.grid(True, linestyle=':', alpha=0.6)

# Panel (b): ΔPPL increase per MB saved at each step
steps = ['FP16→8', '8→6', '6→5', '5→4', '4→3', '3→2']
ppl_dict = {row['bits']: row['ppl'] for _, row in df_e2.iterrows()}
size_dict = {row['bits']: row['size_mb'] for _, row in df_e2.iterrows()}
fp16_row = df_results[(df_results['model'] == '135M') & (df_results['config_name'] == 'fp16')].iloc[0]
ppl_dict[16] = fp16_row['ppl']
size_dict[16] = fp16_row['size_mb']

deltas_per_mb = []
transitions = [(16, 8), (8, 6), (6, 5), (5, 4), (4, 3), (3, 2)]
for b1, b2 in transitions:
    if b1 in ppl_dict and b2 in ppl_dict:
        d_ppl_pct = ((ppl_dict[b2] - ppl_dict[b1]) / ppl_dict[16]) * 100.0
        mb_saved = max(0.1, size_dict[b1] - size_dict[b2])
        deltas_per_mb.append(d_ppl_pct / mb_saved)
    else:
        deltas_per_mb.append(0.0)

ax_b.bar(range(len(steps)), deltas_per_mb, color=COLOR_135M)
ax_b.set_yscale('log')
ax_b.set_ylabel('ΔPPL (%) per MB Saved')
ax_b.set_xticks(range(len(steps)))
ax_b.set_xticklabels(steps, rotation=25)
ax_b.grid(True, linestyle=':', alpha=0.6)
fig.tight_layout()
fig.savefig(fig_dir / 'fig_e2_diminishing.pdf')
fig.savefig(fig_dir / 'fig_e2_diminishing.png', dpi=300)
plt.close(fig)
print('Saved fig_e2_diminishing')

# ------------------------------------------------------------------------------
# Fig E3: Granularity impact (grouped bars: 8-bit vs 4-bit)
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.3, 2.3))
grans = ['tensor', 'channel', 'group128', 'group32']
gran_labels = ['Tensor', 'Channel', 'Group 128', 'Group 32']
x_g = np.arange(len(grans))
w_bar = 0.35

for idx, bits in enumerate([8, 4]):
    vals = []
    eff_b = []
    for g in grans:
        cfg_name = f'int{bits}_sym_{g}'
        match = df_results[(df_results['model'] == '135M') & (df_results['config_name'] == cfg_name)]
        if len(match) > 0:
            vals.append(max(0.01, match['delta_ppl_pct'].values[0]))
            eff_b.append(match['effective_bits'].values[0])
        else:
            vals.append(0.01)
            eff_b.append(bits)
    bars = ax.bar(x_g + (idx - 0.5) * w_bar, vals, width=w_bar, label=f'{bits}-bit', color='#0072B2' if bits==8 else '#D55E00')
    for b, eb in zip(bars, eff_b):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() * 1.15, f'{eb:.1f}b', ha='center', va='bottom', fontsize=5.5)

ax.set_yscale('symlog', linthresh=0.1)
ax.set_xticks(x_g)
ax.set_xticklabels(gran_labels)
ax.set_ylabel('Perplexity Increase (%)')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(fig_dir / 'fig_e3_granularity.pdf')
fig.savefig(fig_dir / 'fig_e3_granularity.png', dpi=300)
plt.close(fig)
print('Saved fig_e3_granularity')

# ------------------------------------------------------------------------------
# Fig E5: Model size vs damage (grouped bars per model)
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.3, 2.3))
bits_sweep = [8, 4, 3, 2]
x_b = np.arange(len(bits_sweep))
w_m = 0.25

for m_idx, (m_tag, color) in enumerate([('135M', COLOR_135M), ('360M', COLOR_360M), ('1.1B', COLOR_11B)]):
    m_vals = []
    for b in bits_sweep:
        match = df_results[(df_results['model'] == m_tag) & (df_results['config_name'] == f'int{b}_sym_group32')]
        m_vals.append(max(0.01, match['delta_ppl_pct'].values[0]) if len(match) > 0 else 0.01)
    ax.bar(x_b + (m_idx - 1) * w_m, m_vals, width=w_m, label=m_tag, color=color)

ax.set_yscale('symlog', linthresh=0.1)
ax.set_xticks(x_b)
ax.set_xticklabels([f'{b}-bit' for b in bits_sweep])
ax.set_ylabel('Perplexity Increase (%)')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(fig_dir / 'fig_e5_model_size.pdf')
fig.savefig(fig_dir / 'fig_e5_model_size.png', dpi=300)
plt.close(fig)
print('Saved fig_e5_model_size')

# ------------------------------------------------------------------------------
# Fig E6: Weight error (SQNR) vs Quality loss (PPL increase)
# ------------------------------------------------------------------------------
from scipy.stats import spearmanr

fig, ax = plt.subplots(figsize=(3.3, 2.3))
df_valid = df_results[(df_results['config_name'] != 'fp16') & (df_results['status'] == 'ok')].copy()

markers = {8: 'o', 6: 's', 5: 'D', 4: '^', 3: 'v', 2: 'x'}
for m_tag, color in MODEL_COLORS.items():
    df_m = df_valid[df_valid['model'] == m_tag]
    for b, marker in markers.items():
        subset = df_m[df_m['bits'] == b]
        if len(subset) > 0:
            ax.scatter(subset['sqnr_db'], subset['delta_ppl_pct'], color=color, marker=marker, s=25, alpha=0.85, label=m_tag if b==4 else None)

ax.set_yscale('symlog', linthresh=0.1)
ax.set_xlabel('Weight SQNR (dB)')
ax.set_ylabel('Perplexity Increase (%)')

if len(df_valid) >= 3:
    corr, _ = spearmanr(df_valid['sqnr_db'], df_valid['delta_ppl_pct'])
    ax.text(0.05, 0.90, f'Spearman ρ = {corr:.2f}', transform=ax.transAxes, fontsize=7)

ax.grid(True, linestyle=':', alpha=0.6)
fig.tight_layout()
fig.savefig(fig_dir / 'fig_e6_error_vs_quality.pdf')
fig.savefig(fig_dir / 'fig_e6_error_vs_quality.png', dpi=300)
plt.close(fig)
print('Saved fig_e6_error_vs_quality')'''

cell_7_code = '''# ==============================================================================
# TABLES: Generate LaTeX booktabs results tables
# ==============================================================================
table_dir = OUTPUTS_DIR / 'tables'

# TinyLlama table
df_tiny = df_results[df_results['model'] == '1.1B'].copy()
df_tiny['meets_constraint'] = df_tiny.apply(
    lambda r: 'Yes' if (r['reduction_vs_fp16_pct'] >= MIN_REDUCTION_PCT and r['size_mb'] <= MAX_FILE_MB) else 'No',
    axis=1
)

# Format table columns
cols_show = ['config_name', 'effective_bits', 'size_mb', 'reduction_vs_fp16_pct', 'ppl', 'delta_ppl_pct', 'meets_constraint']
tex_tiny = df_tiny[cols_show].to_latex(index=False, float_format='%.2f')
with open(table_dir / 'results_tinyllama.tex', 'w') as f:
    f.write(tex_tiny)

# Full results table
tex_all = df_results.to_latex(index=False, float_format='%.2f')
with open(OUTPUTS_DIR / 'results_all.tex', 'w') as f:
    f.write(tex_all)

print(f'Saved LaTeX tables to {table_dir / "results_tinyllama.tex"} and {OUTPUTS_DIR / "results_all.tex"}')'''

cell_8_code = '''# ==============================================================================
# DECISION: Identify winner for TinyLlama according to fixed rule
# ==============================================================================
# Winner rule: among TinyLlama configs meeting both size limits (>=60% smaller, <=1200 MB),
# take smallest file with PPL increase <= 5%. If none, take smallest PPL rise.

df_tiny = df_results[df_results['model'] == '1.1B'].copy()
candidates = df_tiny[
    (df_tiny['config_name'] != 'fp16') &
    (df_tiny['reduction_vs_fp16_pct'] >= MIN_REDUCTION_PCT) &
    (df_tiny['size_mb'] <= MAX_FILE_MB)
].copy()

decision = {
    'rule': 'Smallest file size among <=1200MB & >=60% reduction with ΔPPL <= 5.0%; fallback to min ΔPPL.',
    'candidates_count': len(candidates),
    'candidates': candidates['config_name'].tolist(),
}

if len(candidates) == 0:
    print('No candidate configs met the size constraints.')
    winner_row = None
    decision['winner'] = None
else:
    qualifying = candidates[candidates['delta_ppl_pct'] <= MAX_PPL_INCREASE_PCT]
    if len(qualifying) > 0:
        winner_row = qualifying.sort_values('size_mb').iloc[0]
    else:
        winner_row = candidates.sort_values('delta_ppl_pct').iloc[0]
    decision['winner'] = winner_row['config_name']
    decision['winner_stats'] = winner_row.to_dict()
    print(f'WINNER: {winner_row["config_name"]} | Size: {winner_row["size_mb"]}MB | ΔPPL: {winner_row["delta_ppl_pct"]}%')

with open(OUTPUTS_DIR / 'decision.json', 'w') as f:
    json.dump(decision, f, indent=2)
print(f'Saved decision report to {OUTPUTS_DIR / "decision.json"}')'''

cell_9_code = '''# ==============================================================================
# NATIVE EXPORT: Quantize winner and export to safetensors + quant_config.json
# ==============================================================================
from quantlab import NativeExporter, load_native, RTNQuantizer, QuantConfig, Mode, Granularity

winner_config_name = decision.get('winner')
winner_config = None
for cfg in CONFIGS_COMMON:
    if cfg.name == winner_config_name:
        winner_config = cfg
        break
if winner_config is None:
    winner_config = QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32)
    winner_config_name = winner_config.name

print(f'Exporting TinyLlama with winning config: {winner_config_name}...')
model_tl = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', torch_dtype=torch.float16)
tok_tl = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')

RTNQuantizer(winner_config, skip_names=('lm_head',)).quantize(model_tl)

native_out_dir = OUTPUTS_DIR / 'native'
exporter = NativeExporter()
exporter.export(model_tl, winner_config, native_out_dir, tokenizer=tok_tl)

# Test reloading with load_native
print('Verifying reloaded model forward pass...')
reloaded_m, tok = load_native(str(native_out_dir), device='cpu')
test_input = (tok or tok_tl)('Hello from quantlab mobile model!', return_tensors='pt')
with torch.no_grad():
    test_out = reloaded_m(test_input['input_ids']).logits
print(f'Reload verified! Logits shape: {test_out.shape}')

del model_tl, reloaded_m
gc.collect()'''

cell_10_code = '''# ==============================================================================
# GGUF: Convert TinyLlama to F16 template and export Q8_0 & Q4_0 GGUF files
# ==============================================================================
from quantlab import GGUFExporter, QuantConfig, Mode, Granularity

# Clone llama.cpp for converter script if needed
llama_cpp_dir = BASE_DIR / 'llama_cpp'
if not llama_cpp_dir.exists():
    print('Cloning llama.cpp to generate F16 GGUF template...')
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/ggml-org/llama.cpp', str(llama_cpp_dir)], check=True)

# Find downloaded model cache directory for TinyLlama
from huggingface_hub import snapshot_download
tiny_model_dir = snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0')

template_gguf = BASE_DIR / 'tinyllama-f16-template.gguf'
if not template_gguf.exists():
    print(f'Converting {tiny_model_dir} to F16 GGUF template...')
    conv_script = llama_cpp_dir / 'convert_hf_to_gguf.py'
    subprocess.run([sys.executable, str(conv_script), tiny_model_dir, '--outfile', str(template_gguf), '--outtype', 'f16'], check=True)

# Export Q8_0 and Q4_0 using quantlab GGUFExporter
gguf_exporter = GGUFExporter(template_gguf)
cfg_q8 = QuantConfig(bits=8, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32)
cfg_q4 = QuantConfig(bits=4, mode=Mode.SYMMETRIC, granularity=Granularity.GROUP, group_size=32)

out_q8_path = OUTPUTS_DIR / 'gguf' / 'tinyllama-1.1b-chat-q8_0.gguf'
out_q4_path = OUTPUTS_DIR / 'gguf' / 'tinyllama-1.1b-chat-q4_0.gguf'

print('Exporting Q8_0 GGUF...')
gguf_exporter.export(cfg_q8, out_q8_path)
q8_mb = out_q8_path.stat().st_size / (1024 * 1024)
print(f'  Q8_0 exported: {out_q8_path.name} ({q8_mb:.1f} MB)')

print('Exporting Q4_0 GGUF...')
gguf_exporter.export(cfg_q4, out_q4_path)
q4_mb = out_q4_path.stat().st_size / (1024 * 1024)
print(f'  Q4_0 exported: {out_q4_path.name} ({q4_mb:.1f} MB)')'''

cell_11_code = '''# ==============================================================================
# HUGGING FACE UPLOAD: Upload GGUF and native safetensors with model card
# ==============================================================================
from huggingface_hub import HfApi

hf_token = os.environ.get('HF_TOKEN')
if not hf_token and IS_KAGGLE:
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret('HF_TOKEN')
    except Exception:
        hf_token = None

if not hf_token:
    print('HF_TOKEN not found in environment or Kaggle Secrets. Skipping Hugging Face upload.')
else:
    try:
        api = HfApi(token=hf_token)
        try:
            user_info = api.whoami()
            hf_username = user_info.get('name', HF_USER)
            print(f'Authenticated with Hugging Face as user: {hf_username}')
        except Exception as e:
            print(f'whoami check failed ({e}), using configured HF_USER={HF_USER}')
            hf_username = HF_USER

        repo_id = f'{hf_username}/{HF_REPO}'
        print(f'Creating repo {repo_id} on Hugging Face...')
        api.create_repo(repo_id=repo_id, repo_type='model', exist_ok=True, private=False)

        # Create Model Card README
        model_card = (
            "---\n"
            "license: apache-2.0\n"
            "base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0\n"
            "tags:\n"
            "- quantization\n"
            "- quantlab\n"
            "- gguf\n"
            "---\n\n"
            f"# {HF_REPO}\n\n"
            f"Quantized weights for TinyLlama-1.1B-Chat-v1.0 produced by the [quantlab](https://github.com/{GITHUB_USER}/{REPO_NAME}) post-training quantization library.\n\n"
            "## Constraint & Decision\n"
            "- Target device: Galaxy M15 (6 GB RAM)\n"
            "- Size limit: <= 1200 MB and >= 60% reduction vs FP16\n"
            f"- Selected config: `{winner_config_name}`\n\n"
            "## How to load natively\n"
            "```python\n"
            "from quantlab import load_native\n"
            f'model, tokenizer = load_native("{repo_id}", subfolder="native")\n'
            "```\n"
        )
        with open(OUTPUTS_DIR / 'MODEL_CARD.md', 'w') as f:
            f.write(model_card)

        # Upload README
        api.upload_file(path_or_fileobj=str(OUTPUTS_DIR / 'MODEL_CARD.md'), path_in_repo='README.md', repo_id=repo_id)

        # Upload GGUFs
        for gguf_file in (OUTPUTS_DIR / 'gguf').glob('*.gguf'):
            print(f'Uploading {gguf_file.name}...')
            api.upload_file(path_or_fileobj=str(gguf_file), path_in_repo=gguf_file.name, repo_id=repo_id)

        # Upload native folder
        print('Uploading native safetensors folder...')
        api.upload_folder(folder_path=str(OUTPUTS_DIR / 'native'), path_in_repo='native', repo_id=repo_id)
        print(f'Successfully published to https://huggingface.co/{repo_id}!')
    except Exception as e:
        print(f'Hugging Face upload failed: {e}')
        print('Check your HF_TOKEN permissions (requires Write role) and account username.')
        print('All exported files remain saved and accessible in /kaggle/working/outputs/')'''

cell_12_code = '''# ==============================================================================
# FINAL CELL: List all generated artifacts in outputs with file sizes
# ==============================================================================
total_bytes = 0
print('=== Generated Artifacts Summary ===')
for root, dirs, files in os.walk(OUTPUTS_DIR):
    for name in sorted(files):
        fpath = Path(root) / name
        fsize = fpath.stat().st_size
        total_bytes += fsize
        rel = fpath.relative_to(OUTPUTS_DIR)
        print(f'  {str(rel):35s}: {fsize / (1024*1024):8.2f} MB ({fsize:,} bytes)')

print(f'\\nTotal Artifacts Size: {total_bytes / (1024*1024):.2f} MB ({total_bytes:,} bytes)')
print('Notebook run completed successfully!')'''

nb.cells = [
    new_markdown_cell(cell_0_md, id='cell-0-intro'),
    new_code_cell(cell_1_code, id='cell-1-params'),
    new_code_cell(cell_2_code, id='cell-2-setup'),
    new_code_cell(cell_3_code, id='cell-3-configs'),
    new_code_cell(cell_4_code, id='cell-4-experiments'),
    new_code_cell(cell_5_code, id='cell-5-sizes'),
    new_code_cell(cell_6_code, id='cell-6-figures'),
    new_code_cell(cell_7_code, id='cell-7-tables'),
    new_code_cell(cell_8_code, id='cell-8-decision'),
    new_code_cell(cell_9_code, id='cell-9-native-export'),
    new_code_cell(cell_10_code, id='cell-10-gguf-export'),
    new_code_cell(cell_11_code, id='cell-11-hf-upload'),
    new_code_cell(cell_12_code, id='cell-12-summary'),
]

# Validate schema
nbformat.validate(nb)
print('nbformat validation passed with 0 errors!')

with open('notebooks/quantization_experiments.ipynb', 'w') as f:
    nbformat.write(nb, f)

print('Successfully written notebooks/quantization_experiments.ipynb')
