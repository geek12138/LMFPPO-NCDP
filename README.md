<div align="center">

# LMFPPO-NCDP

**Local Mean Field Proximal Policy Optimization with Neighbor-Dependent Cooperative Density Punishment**

An official implementation for promoting cooperation in spatial public goods games.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243.svg)](https://numpy.org/)
[![Paper](https://img.shields.io/badge/IEEE%20TCSS-2026-00629B.svg)](#citation)

</div>

---

## Table of Contents

- [Introduction](#introduction)
- [Method Overview](#method-overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Arguments](#arguments)
- [Outputs](#outputs)
- [Experimental Results](#experimental-results)
- [FAQ](#faq)
- [Citation](#citation)
- [Contact](#contact)
- [License](#license)

---

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

## Method Overview

### 1. Spatial Public Goods Game (SPGG)

- Played on an $L \times L$ grid with **periodic boundary conditions**, where each node is an agent.
- Interaction structure is the **von Neumann neighborhood** ($k=4$ neighbors).
- Each agent simultaneously participates in $G=5$ public goods groups (centered on itself and its 4 neighbors).
- Strategy set $S=\{C, D\}$: $C$ contributes 1 unit of resource, $D$ contributes nothing.

The payoff of agent $i$ in a single group $g$ is:

$$
\Pi(s_i^g)=
\begin{cases}
\dfrac{r \cdot N_C^g}{G} - 1, & s_i^g = C \\[6pt]
\dfrac{r \cdot N_C^g}{G}, & s_i^g = D
\end{cases}
$$

where $N_C^g$ is the number of cooperators in group $g$ and $r>1$ is the public goods enhancement factor. The total payoff of an agent is the sum over all groups it participates in, $\Pi_i=\sum_{g\in G_i}\Pi(s_i^g)$.

### 2. Local Mean Field Representation (LMF)

The local region is formed by a focal agent and its 4 von Neumann neighbors, $\mathcal{G}(i)=i\cup\mathcal{N}(i)$, and the local cooperation density is defined as:

$$
\mu_i=\frac{1}{5}\sum_{j\in\mathcal{G}(i)} s_j
$$

At time $t$, the observation of each agent is encoded as a **4-dimensional feature vector**:

$$
x_t^i=\left[\,s_t^i,\; n_t^i,\; g_t,\; \mu_t^i\,\right]\in\mathbb{R}^4
$$

| Feature | Meaning |
| --- | --- |
| $s_t^i\in\{0,1\}$ | Agent's own current strategy |
| $n_t^i\in\{0,\dots,5\}$ | Number of cooperators in the local region (including itself) |
| $g_t\in[0,1]$ | Global cooperation frequency |
| $\mu_t^i\in[0,1]$ | Local mean field (local cooperation density) |

In the implementation, both $n_t^i$ and $\mu_t^i$ are computed in a single parallel pass using **circular padding plus a 3×3 convolution kernel** `[[0,1,0],[1,1,1],[0,1,0]]` (`F.conv2d` with `mode='circular'`), so feature extraction over the whole grid is **fully vectorized**.

### 3. NCDP Punishment Mechanism

Unlike conventional schemes that punish all defectors equally, NCDP allocates heterogeneous punishment strength based on the **local cooperation density**:

$$
R_{\text{punish}}^i=-\,p\cdot\mathbb{I}(s_i=D)\cdot N_C^{\text{neigh}}(i)
$$

where $p>0$ is the punishment strength and $N_C^{\text{neigh}}(i)$ is the number of cooperators among the von Neumann neighbors of agent $i$. The more cooperators in the neighborhood, the stronger the punishment a defector receives, which creates a **local payoff gradient** at the boundaries of cooperative clusters and systematically suppresses defection. **Cooperators bear no punishment cost whatsoever**.

The total reward of an agent is:

$$
R_{\text{total}}^i=\Pi_i+R_{\text{punish}}^i
$$

### 4. Actor–Critic Network and Optimization Objective

A shared feature extractor (two `Linear(64) + ReLU` layers) is followed by an actor branch (softmax over C/D probabilities) and a critic branch (state value):

```python
MeanFieldActorCritic(
    shared = Sequential(Linear(4, 64), ReLU, Linear(64, 64), ReLU),
    actor  = Linear(64, 2),    # policy distribution (cooperate / defect)
    critic = Linear(64, 1),    # state value
)
```

The optimization objective of LMFPPO (i.e. the loss function of `ppo_update()` in `LMFPPO_NCDP.py`) is:

$$
\mathcal{L}_{\text{LMFPPO}}(\theta)=\mathcal{L}_{\text{CLIP}}(\theta)+\delta\,\mathcal{L}_{\text{VF}}(\theta)-\rho\,\mathcal{L}_{\text{ENT}}(\theta)
$$

where $\mathcal{L}_{\text{CLIP}}$ is the PPO clipped surrogate objective, $\mathcal{L}_{\text{VF}}$ is the value-function MSE loss, and $\mathcal{L}_{\text{ENT}}$ is the entropy regularization term. Advantages are estimated with GAE. The corresponding code is:

```python
loss = actor_loss + self.delta * critic_loss - self.rho * entropy
```

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

Shared payoff conventions across all implementations:

- **Base payoff** `calculate_reward()`: circular convolution over the cooperator matrix gives the number of cooperators in each group, another convolution aggregates them per agent, and the cost of joining 5 groups is subtracted (cooperators pay `5.0`, defectors pay `0`).
- **Punishment payoff** `calculate_punishment()`: assigns $-p \times N_C^{\text{neigh}}$ only at defector positions.
- **Total payoff** `calculate_reward_with_punishment()` = base payoff + punishment payoff.

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

---

## Experimental Results

### Hyperparameter sensitivity (paper Figs. 2–4)

- **Entropy coefficient $\rho$**: the effect is non-monotonic. With $\rho=0.001$ the system still falls into all-defection at $r=5.0$, whereas with $\rho=0.01$ it reaches full cooperation at the same point → $\rho=0.01$ is used in the final setting.
- **Learning rate $\alpha$**: too small slows down policy adaptation and delays the emergence of cooperation; too large brings no extra benefit and even delays it slightly → $\alpha=10^{-3}$ is used.
- **Punishment strength $p$**: increasing $p$ shifts the cooperation threshold to the left, but overly strong punishment introduces non-monotonic oscillations near the threshold and reduces evolutionary stability → $p=0.5$ is used.

### Ablation and statistical tests (paper Table I, 50 independent runs, 95% BCa bootstrap CI)

| Algorithm | Cooperation threshold behavior |
| --- | --- |
| **LMFPPO-NCDP** | All-defection for $r\le 3.6$; average cooperation rate 0.82 at $r=4.1$; full cooperation at $r=4.2$ |
| PPO-NCDP | Average cooperation rate 0.66 at $r=4.1$; full cooperation at $r=4.3$ |
| LMFPPO | All-defection for $r\le4.9$; average cooperation rate 0.50 at $r=5.0$; full cooperation at $r=5.1$ |
| PPO | Average cooperation rate 0.46 at $r=5.0$; full cooperation at $r=5.1$ |

Conclusion: NCDP significantly lowers the cooperation threshold, and the LMF representation yields **narrower error bars and confidence intervals** near the critical threshold, indicating improved learning stability.

### Comparison with state-of-the-art algorithms (paper Table II)

| Algorithm | First emergence of cooperation ($r$) | Sustained cooperation ($r$) |
| --- | --- | --- |
| LMFPPO-NCDP ($p=1.5$) | 2.0 | 2.4 |
| LMFPPO-NCDP ($p=1.0$) | 2.7 | 3.2 |
| LMFPPO-NCDP ($p=0.5$) | 4.0 | 4.0 |
| PPO-ACT | 3.6 | 4.0 |
| TUC-PPO | 3.3 | 3.3 |

### Fixed enhancement factor comparison ($r=4.0$, paper Fig. 6)

| Algorithm | Evolutionary outcome |
| --- | --- |
| LMFPPO-NCDP | Reaches global cooperation $f_C=1.0$ within about 20 steps and keeps it |
| LMFPPO | Shows strategy mixing first but eventually converges to all-defection (LMF alone is insufficient to sustain cooperation at $r=4.0$) |
| Fermi update rule | Cooperators slightly dominate in the first 100 steps, defectors dominate between 100–500 steps, then it stabilizes at the mixed state $f_C\approx0.55$ |
| Q-learning | Cooperation declines steadily early on and stabilizes at $f_C\approx0.40$ after about 100 steps |

---

## FAQ

**Q1. `scripts/run_QL.sh` reports that `main_QL.py` cannot be found?**
The `main_QL.py` referenced by that script is not included in this repository. The Q-learning implementation lives in `QL_Feimi.py` (class `SPGG_Qlearning`); you need to write your own entry script following `main_Fermi.py` / `main_PPO.py` (the arguments are `-alpha -gamma -epsilon -is_QL -is_fermi`).

**Q2. How do I change the range of the enhancement factor sweep?**
`r_values` is hard-coded inside `main()` of each `main_*.py`:

- `main_LMFPPO_NCDP.py`: `[round(i*0.1, 1) for i in range(20, 36)]` → $r\in[2.0, 3.5]$
- `main_PPO.py`: `[round(i*0.1, 1) for i in range(30, 61)]` → $r\in[3.0, 6.0]$
- `main_Fermi.py`: an explicit list over $r\in[2.0, 6.0]$

**Q3. How do I reproduce the statistical results in the paper?**
Set `-runs` to `50` and change `-seed` for each run (see the per-seed loop in `scripts/run_one_PPO.sh`), then compute the BCa bootstrap confidence interval over the last row of `Density_C/r<r>.txt`.

**Q4. Why is the x-axis of the plots on a log-like scale?**
All evolution curves use `plt.xscale('symlog', linthresh=1)` so that both the fast early changes and the long-term stable behavior can be displayed at the same time.

**Q5. Out of memory (RAM or VRAM)?**
Reduce `-L_num` (e.g. to 100), or keep `-batch_size 1` and lower `-epochs` accordingly.

---

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

- Jinshuo Yang — gs.jsyang25@gzu.edu.cn
- **Zhaoqilin Yang (corresponding author)** — zqlyang@gzu.edu.cn
- Wenjie Zhou — gs.wjzhou25@gzu.edu.cn
- Xin Wang — xinwang3@bjtu.edu.cn
- Youliang Tian — yltian@gzu.edu.cn

State Key Laboratory of Public Big Data, College of Computer Science and Technology, Guizhou University, Guiyang, China.

---

## License

This repository currently does not ship an open-source license file. If you intend to use the code outside of academic purposes, please contact the authors first to confirm authorization.
