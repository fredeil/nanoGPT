# Experiments

This directory contains everything related to the INF-3600 assignment: configs, run scripts, output logs, and reports.

## Prerequisites

- nanoGPT cloned and set up in the parent directory
- TinyStories dataset prepared at `data/tinystories/` (run `prepare.py` first)
- `uv` installed for running Python
- Config files placed in `../config/` (see below)

The experiments were run on a MacBook with Apple M-series GPU (MPS backend). If you're using CUDA, change `device=mps` to `device=cuda` in the config files and sample commands.

## Config Files

The config files should be placed in `../config/`:

- `config/exp1_depth.py` — Depth sweep. Fixed: n_head=8, n_embd=256, dropout=0.0
- `config/exp2_width.py` — Width sweep. Fixed: n_layer=6, n_head=8, dropout=0.0
- `config/exp3_heads.py` — Head sweep. Fixed: n_layer=6, n_embd=256, dropout=0.0

## Running the Experiments

From the nanoGPT root directory:

```bash
# Experiment 1 - Depth sweep (n_layer ∈ {2, 4, 8, 12})
bash experiments/run_exp1.sh

# Experiment 2 - Width sweep (n_embd ∈ {128, 192, 256, 384})
bash experiments/run_exp2.sh

# Experiment 3 - Head sweep (n_head ∈ {4, 8, 16, 32})
bash experiments/run_exp3.sh
```

Each script trains 4 models and generates 5 text samples (1000 tokens each) per model. Output goes to `out/out-exp{N}-{param}{value}/` with `log.txt` and `samples.txt` in each.

Total training time on MPS was roughly 20-30 minutes per experiment (4 models x 5000 iterations).

## Directory Structure

```
experiments/
├── README.md
├── assignment.md          # Original assignment spec
├── report.md              # Final combined report
├── takeaways.md           # Exam prep notes
├── run_exp1.sh            # Depth sweep script
├── run_exp2.sh            # Width sweep script
├── run_exp3.sh            # Head sweep script
└── out/
    ├── report_exp1.md     # Individual experiment report
    ├── report_exp2.md
    ├── report_exp3.md
    ├── out-exp1-layer2/   # Training logs + samples
    ├── out-exp1-layer4/
    ├── out-exp1-layer8/
    ├── out-exp1-layer12/
    ├── out-exp2-embd128/
    ├── out-exp2-embd192/
    ├── out-exp2-embd256/
    ├── out-exp2-embd384/
    ├── out-exp3-head4/
    ├── out-exp3-head8/
    ├── out-exp3-head16/
    └── out-exp3-head32/
```

Each model output directory contains:
- `log.txt` — Full training log with loss values at every 10 iterations and validation loss every 250 steps
- `samples.txt` — 5 generated text samples of 1000 tokens each
- `ckpt.pt` — Model checkpoint (not committed to git)
