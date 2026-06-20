# CausalRLBreaker

> Causal Reinforcement Learning for Automated Security Testing of Large Language Models

**MSc Thesis — Data Science & Engineering, University of Naples Federico II**
**Author:** Parisa Zeinaliashtiyani
**Supervisor:** Prof. Roberto Pietrantuono

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-orange.svg)](https://github.com/DLR-RM/stable-baselines3)

---

## Overview

CausalRLBreaker is a framework for automated black-box red-teaming of aligned LLMs.
It extends standard RL-based jailbreak search (e.g. RLBreaker) by embedding a
**Structural Causal Model (SCM)**, discovered via the **Fast Causal Inference (FCI)**
algorithm, directly into the PPO agent's observation and reward signal.

Instead of optimizing purely on sparse, terminal success/failure feedback, the agent
observes a compact **6-dimensional causal state vector** — capturing factors such as
instructional style, responsibility externalization, obfuscation, and hypothetical
framing — and receives **dense, causally-shaped intermediate rewards**. This addresses
the credit assignment problem that limits standard black-box RL red-teaming, improving
both attack effectiveness and training efficiency.

This repository contains the implementation, training/evaluation scripts, and
experimental results supporting the thesis manuscript (in preparation):
*"Security Testing of Large Language Models via Causal Reinforcement Learning."*

---

## Key Results

Evaluated on 104 unseen AdvBench prompts against Llama-3.3-70B-Versatile,
averaged across 3 random seeds (42, 100, 2026):

| Method | ASR (%) | Training Time | API Calls | Token Footprint |
|---|---|---|---|---|
| Static DAN baseline | 2.10% | — | — | — |
| RLBreaker (PPO baseline) | 19.33% ± 3.21 | 273.75 min | 20,480 | 7,021,619 |
| **CausalRLBreaker (Ours)** | **65.06% ± 3.09** | **96.00 min** | **11,915** | **2,707,647** |

→ **3.4× higher attack success rate**, **64.9% lower training compute**, **61.4% lower token cost** vs. the RL-only baseline.

### Zero-shot cross-model transfer

The policy trained exclusively on Llama-3.3 transfers without retraining to:

| Target Model | ASR (%) | Retention |
|---|---|---|
| Llama-3.3 (in-distribution) | 61.54% | 100% |
| Llama-4 (next-gen, zero-shot) | 65.38% | 106% |
| Qwen-3 (cross-family, zero-shot) | 45.19% | 73% |

> **Note on reporting:** All headline figures are averaged over seeds {42, 100, 2026}.
> Per-seed results and raw evaluation logs are available in [`results/`](results/) for
> full reproducibility.

---

## Architecture

![Causal graph (PAG) discovered via FCI](results/figures/pag_graph.png)
*Partial Ancestral Graph (PAG) estimated by the COAT-based causal discovery pipeline over six prompt-level causal factors.*

**Pipeline:**

```
harmful seed prompt
       │
       ▼
  PPO agent selects mutation action (rephrase / obfuscate / reframe / ...)
       │
       ▼
  Mutator LLM rewrites the prompt
       │
       ▼
  Causal Factor Extractor → 6D causal state vector
  [instructional_style, responsibility_externalization, obfuscation_techniques,
   hypothetical_framing, imperative_tone, malicious_intent]
       │
       ▼
  CausalBreakerSCM scores causal potential Φ(s) using the discovered PAG as prior
       │
       ▼
  Dynamic Causal Weights re-weight factors every 500 training steps
       │
       ▼
  Dense causal reward: R = R_outcome + R_judge + R_SCM + R_synergy − R_cost
       │
       ▼
  Target LLM response → Judge LLM classifies jailbreak success/failure
       │
       └──► feeds back into PPO policy update
```

Full mathematical formulation (SCM definition, reward decomposition, dynamic weight
update rule) is documented in [`docs/methodology.md`](docs/methodology.md).

---

## Repository Structure

```
CausalRLBreaker/
├── causalrlbreaker/        # Core library
│   ├── env.py               # CausalRLBreakerEnv (Gymnasium environment)
│   ├── causal/
│   │   ├── factor_extractor.py   # Lexical-semantic causal factor extraction
│   │   ├── scm.py                # CausalBreakerSCM (causal prior + scoring)
│   │   └── dynamic_weights.py    # DynamicCausalWeights (adaptive factor weighting)
│   ├── reward/
│   │   └── shaping.py            # Dense causal reward function
│   ├── mutators/
│   │   └── mutator_bank.py       # Prompt mutation operators (action space)
│   └── judge/
│       └── judge.py               # LLM-based jailbreak judge
├── scripts/
│   ├── train.py              # PPO training entry point
│   ├── evaluate.py           # Single-run evaluation
│   └── multi_seed_eval.py    # Multi-seed statistical evaluation
├── configs/
│   └── default.yaml          # Hyperparameters
├── results/
│   ├── figures/               # PAG diagram, causal weight convergence plot
│   └── tables/                # Main results, ablation, cross-model transfer (CSV)
├── docs/
│   └── methodology.md         # Full mathematical formulation
├── notebooks/exploratory/     # Original research notebooks (kept for transparency)
└── tests/
    └── test_env.py
```

---

## Installation

```bash
git clone https://github.com/parisazeynaly/CausalRLBreaker.git
cd CausalRLBreaker
pip install -r requirements.txt
```

Requires a [Groq API key](https://console.groq.com) (used for mutator/target/judge
LLM calls). Set it as an environment variable before running:

```bash
export GROQ_API_KEY="your-key-here"
```

---

## Usage

**Train the PPO + causal reward agent:**
```bash
python scripts/train.py --config configs/default.yaml --output-dir ./runs/exp1
```

**Evaluate a trained model across multiple seeds:**
```bash
python scripts/multi_seed_eval.py \
    --model-path ./runs/exp1/final_model.zip \
    --scm-path ./runs/exp1/final_scm.npz \
    --seeds 42 100 2026 \
    --use-pag-guidance
```

**Reproduce the ablation study:**
```bash
python scripts/evaluate.py --config configs/ablation.yaml
```

---

## Methodology Summary

| Component | Description |
|---|---|
| **Causal discovery** | FCI algorithm over logged RLBreaker interaction trajectories → Partial Ancestral Graph (PAG) |
| **State representation** | 6D normalized causal factor vector (replaces raw text observation) |
| **Action space** | Standard text mutators + causally-targeted mutators aligned to PAG factors |
| **SCM** | Linear structural causal model with PAG-derived priors; computes causal potential Φ(s) |
| **Dynamic weighting** | Re-estimates factor importance every 500 steps from observed success correlations |
| **Reward shaping** | Dense, multi-objective: outcome + judge + SCM potential/delta + synergy − cost |
| **Training signal** | Proxy keyword-based success check during training (cost-efficient); full LLM-judge evaluation only at test time |

See [`docs/methodology.md`](docs/methodology.md) for full equations and design rationale.

---

## Reproducibility Notes

- All evaluation results in `results/tables/` are generated from real Groq API calls
  against the stated target models — no synthetic or placeholder evaluation data is
  included in reported metrics.
- Training uses a lightweight keyword-based proxy reward signal (for cost efficiency
  during the ~10,000-step PPO training loop); **final reported ASR values always use
  the full LLM-based judge**, applied only at evaluation time.
- Multi-seed evaluation (seeds 42 / 100 / 2026) is used for all headline results to
  report mean ± standard deviation rather than a single run.

---

## Citation

```bibtex
@mastersthesis{zeinaliashtiyani2026causalrlbreaker,
  title  = {Security Testing of Large Language Models via Causal Reinforcement Learning},
  author = {Zeinaliashtiyani, Parisa},
  school = {University of Naples Federico II},
  year   = {2026},
  note   = {Supervised by Prof. Roberto Pietrantuono}
}
```

---

## Disclaimer

This repository is released strictly for **academic LLM safety research and
responsible red-teaming purposes**. It is intended to help identify and ultimately
fix alignment vulnerabilities in deployed language models, not to facilitate misuse.
Please use responsibly and disclose any newly discovered vulnerabilities to the
relevant model providers.

## Acknowledgments

This work was conducted as part of an MSc thesis at the University of Naples
Federico II, under the supervision of Prof. Roberto Pietrantuono.
