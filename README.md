<div align="center">

# LMFPPO-NCDP

**Local Mean Field Proximal Policy Optimization with Neighbor-Dependent Cooperative Density Punishment**

## Introduction

This repository is the official implementation of the paper *"Promoting Cooperation in Spatial Public Goods Games via Local Mean Field Proximal Policy Optimization With Neighbor-Dependent Cooperative Density Punishment"* (IEEE Transactions on Computational Social Systems, 2026).

Spatial Public Goods Games (SPGGs) are characterized by **local interactions** and a **high-dimensional state space**, which makes stable cooperation difficult to emerge spontaneously. Conventional approaches — imitation-based Fermi updates, tabular Q-learning, and global mean-field approximations — struggle to simultaneously capture neighborhood-level strategic interactions and to dynamically align individual incentives with collective welfare.

This work proposes the **LMFPPO-NCDP** framework, which consists of two key components:

| Component | Role |
| --- | --- |
| **LMF** (Local Mean Field) | Reformulates the classical mean field into a **local statistical representation** embedded into policy learning, so that each agent perceives the neighborhood-level cooperation density |
| **NCDP** (Neighbor-Dependent Cooperative Density Punishment) | Applies **heterogeneous punishment** to defectors according to the local cooperation density, while cooperators bear no direct punishment cost |

Experiments show that LMFPPO-NCDP achieves fast and stable cooperation emergence even at a **low enhancement factor**, with a lower cooperation threshold and stronger robustness than PPO, Q-learning, and Fermi update rules.

> Code repository released with the paper: <https://github.com/geek12138/LMFPPO-NCDP>

---



## Project Structure

```text
LMFPPO/
├── LMFPPO_NCDP.py          # Core implementation: Mean-Field PPO with LMF + NCDP (punishment reward, snapshot plotting)
├── main_LMFPPO_NCDP.py     # Main experiment entry: sweeps the enhancement factor r, trains / saves / plots
├── PPO.py                  # Baseline: standard PPO (3-dim observation, no LMF, no NCDP)
├── main_PPO.py             # Baseline PPO experiment entry
├── Fermi.py                # Baseline: Fermi imitation update rule (with optional punishment)
├── main_Fermi.py           # Baseline Fermi experiment entry
├── QL_Feimi.py             # Baseline: Q-learning (class SPGG_Qlearning, can be combined with Fermi propagation)
└── scripts/
    ├── run_one_LMFPPO.sh   # One-click script for the LMFPPO experiment
    ├── run_one_PPO.sh      # One-click script for the PPO experiment
    ├── run_one_Fermi.sh    # One-click script for the Fermi experiment
    └── run_QL.sh           # One-click script for the Q-learning experiment
```
---

## Requirements

- Python >= 3.8
- PyTorch >= 1.12 (CPU / CUDA / MPS are all supported)
- NumPy, Matplotlib, tqdm

```bash
pip install torch numpy matplotlib tqdm
```

> Running on a GPU is recommended: with $L=200$, each time step requires one batched forward pass over $L^2 = 40000$ agents.

---

## Quick Start

All experiment entries accept arguments via `argparse`, and the **enhancement factor list `r_values` is hard-coded in each `main_*.py`**; modify that list directly if you need a custom sweep.

### 1. Train LMFPPO-NCDP (main method)

```bash
python main_LMFPPO_NCDP.py \
    -epochs 1000 -runs 1 -L_num 200 \
    -alpha 1e-3 -gamma 0.99 -clip_epsilon 0.2 \
    -question 2 -ppo_epochs 1 -batch_size 1 -gae_lambda 0.95 \
    -delta 0.5 -rho 0.01 -punishment_strength 0.5 \
    -seed 1 -device cuda
```

### 2. Ablation / baseline comparison

```bash
# LMFPPO without NCDP: set the punishment strength to 0
python main_LMFPPO_NCDP.py -punishment_strength 0 -device cuda

# PPO baseline (with optional NCDP)
python main_PPO.py \
    -epochs 10000 -runs 1 -L_num 200 \
    -alpha 1e-3 -gamma 0.99 -clip_epsilon 0.2 \
    -question 2 -ppo_epochs 1 -batch_size 1 -gae_lambda 0.95 \
    -delta 0.5 -rho 0.01 -punishment_strength 0.5 -seed 1

# Fermi baseline
python main_Fermi.py -epochs 10000 -runs 1 -L_num 200 \
    -question 2 -seed 1 -punishment_strength 0
```

### 3. One-click scripts

```bash
cd scripts
bash run_one_LMFPPO.sh   # or run_one_PPO.sh / run_one_Fermi.sh / run_QL.sh
```

### 4. Load a saved model

The `checkpoint/` directory of each experiment stores model weights and optimizer states:

```python
import torch
from LMFPPO_NCDP import LMFPPO_SPGG

ckpt = torch.load("data/<experiment_dir>/checkpoint/model_r4.2_final.pth", map_location="cpu")
model.policy.load_state_dict(ckpt["model_state_dict"])
print(ckpt["r"], ckpt["punishment_strength"])
```

---

## Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `-epochs` | `1000` | Number of training iterations $T$ |
| `-runs` | `1` | Independent repeats per $r$ (50 is recommended for statistical experiments) |
| `-L_num` | `200` | Grid side length $L$ ($L\times L$ with periodic boundaries) |
| `-alpha` | `1e-3` | Learning rate $\alpha$ (Adam + StepLR) |
| `-gamma` | `0.99` | Discount factor $\gamma$ |
| `-clip_epsilon` | `0.2` | PPO clipping threshold $\epsilon$ |
| `-gae_lambda` | `0.95` | GAE parameter $\lambda$ |
| `-delta` | `0.5` | Value-function loss weight $\delta$ |
| `-rho` | `0.01` | Entropy regularization coefficient $\rho$ |
| `-punishment_strength` | `0.5` / `1.5` | NCDP punishment strength $p$ (`LMFPPO_NCDP.py` defaults to 1.5, the main entry defaults to 0.5) |
| `-ppo_epochs` | `1` | Number of PPO update epochs per batch $K$ |
| `-batch_size` | `1` | Number of time steps accumulated before each PPO update |
| `-question` | `2` | Initial state mode (see the table below) |
| `-seed` | `1` | Random seed |
| `-device` | `cuda` | Running device (`cuda` / `cpu` / `mps`) |

**Initial state modes `-question`:**

| Value | Initial configuration |
| --- | --- |
| `1` | Bernoulli random (50% cooperators / 50% defectors) |
| `2` | Defectors on the upper half, cooperators on the lower half (the half-and-half initialization used in the paper) |
| `3` | All defectors |
| `4` | Checkerboard alternation (supported in `LMFPPO_NCDP.py`) |

---

## Outputs

Each run creates an experiment directory named `data/<algorithm>_<timestamp>_q.._e.._L.._.../` containing:

```text
data/LMFPPO_NCDP_Punish_<timestamp>_.../
├── params_<timestamp>.json     # All hyperparameters (including the r list, device, algorithm name)
├── main_LMFPPO_NCDP.py         # Auto-backed-up source code (for reproducibility)
├── LMFPPO_NCDP.py              # Auto-backed-up source code
├── Density_C/r<r>.txt          # Cooperator fraction f_C at every time step
├── Density_D/r<r>.txt          # Defector fraction f_D at every time step
├── Value_C/ Value_D/           # Average payoff of cooperators / defectors
├── Total_Value/r<r>.txt        # Total system payoff
├── Punishment/r<r>.txt         # Evolution of the average punishment
├── strategy_evolution_r<r>_run<n>.{pdf,png}   # Strategy fraction evolution (symlog x-axis)
├── profit_evolution_r<r>_run<n>.{pdf,png}     # Average payoff evolution
├── C_D_r.{pdf,png}             # Steady-state cooperator/defector fraction versus r
├── checkpoint/                 # Periodic checkpoints and the final model_r<r>_final.pth
└── shot_pic/r=<r>/two_type/
    ├── strategy_distribution_t=*.{pdf,png}    # Strategy snapshots (blue = cooperator, red = defector)
    ├── profit_t=*.{pdf,png}                   # Payoff heatmaps
    ├── punishment_t=*.{pdf,png}               # Punishment strength heatmaps
    ├── type_t_matrix/T*.txt                   # Raw strategy matrix data
    ├── profit_matrix/T*.txt                   # Raw payoff matrix data
    └── punishment_matrix/T*.txt               # Raw punishment matrix data
```

Snapshots are saved by default at key time steps such as $t = 0, 1, 10, 100, 1000, 10000, 100000$ (see `run()` and `shot_pic_with_punishment()` in `LMFPPO_NCDP.py`).

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{yang2026lmfppo,
  title   = {Promoting Cooperation in Spatial Public Goods Games via Local Mean
             Field Proximal Policy Optimization With Neighbor-Dependent
             Cooperative Density Punishment},
  author  = {Yang, Jinshuo and Yang, Zhaoqilin and Zhou, Wenjie and Wang, Xin and Tian, Youliang},
  journal = {IEEE Transactions on Computational Social Systems},
  year    = {2026},
  doi     = {10.1109/TCSS.2026.3720953}
}
```

**Related methods:**

- **PPO** — J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, *Proximal Policy Optimization Algorithms*, 2017. [arXiv:1707.06347](http://arxiv.org/abs/1707.06347)
- **Mean Field MARL** — Y. Yang et al., *Mean Field Multi-Agent Reinforcement Learning*, ICML 2018.
- **PPO-ACT** — Z. Yang, C. Li, X. Wang, Y. Tian, *Chaos, Solitons & Fractals*, vol. 199, 2025, Art. no. 116762.
- **TUC-PPO** — Z. Yang, X. Wang, R. Zhang, C. Li, Y. Tian, *Chaos, Solitons & Fractals*, vol. 199, 2025, Art. no. 116928.

---

## Contact

- **Zhaoqilin Yang (corresponding author)** — zqlyang@gzu.edu.cn


State Key Laboratory of Public Big Data, College of Computer Science and Technology, Guizhou University, Guiyang, China.

---

## License

This repository currently does not ship an open-source license file. If you intend to use the code outside of academic purposes, please contact the authors first to confirm authorization.
