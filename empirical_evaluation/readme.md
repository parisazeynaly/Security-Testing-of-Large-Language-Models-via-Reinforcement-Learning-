# 📊 Empirical Evaluation & Multi-Model Benchmarking Suite

This directory contains the complete benchmarking suite, dataset configurations, raw log artifacts, and visualization pipelines used to generate the empirical findings for our second paper: 

> **Title:** Beyond Token Correlations: An Empirical Evaluation of Cross-Model Transferability and Cost-Efficiency in Causal Red-Teaming  
> **Authors:** Parisa Zeinaliashtiyani, Roberto Pietrantuono  
> **Status:** Under Review / [Preprint on arXiv](https://arxiv.org) (Link Placeholder)

---

## 🔬 Directory Structure

```text
empirical_evaluation/
├── 📁 configs/               # Target model system prompts and API configurations
│   ├── llama3_3_70b.json
│   ├── llama4_nextgen.json
│   └── qwen3_crossfamily.json
├── 📁 datasets/              # Harmlessness/Safety evaluation intents (N=104)
│   └── malicious_intents_104.json
├── 📁 raw_logs/              # Complete multi-seed evaluation trace artifacts
│   ├── 📁 token_footprints/  # Step-by-step query costs and latency track logs
│   └── 📁 weight_traces/     # Epoch-by-epoch 6D SCM dynamic weight evolutions
├── 📁 scripts/               # Automated pipeline runners
│   ├── run_benchmark.py      # Main end-to-end evaluation entrypoint
│   └── statistical_tests.py  # Script for computing t-tests and variance stability
└── 📁 tracking_plots/        # Automated generation scripts for manuscript figures
    ├── plot_convergence.py   # Code for dynamic causal weights line plots
    └── plot_cost_curve.py    # Code for cumulative token savings curves
```

---

## 🚀 Key Empirical Reproducibility Checkpoints

### 1. Cross-Model Zero-Shot Transferability (`RQ1`)
Our framework extracts invariant structural causal paths optimized on `Llama-3.3-70B` and deploys them *zero-shot* onto next-generation architectures without policy weight re-tuning. Execute the transferability pipeline via:

```bash
python scripts/run_benchmark.py \
    --source_policy ./raw_logs/weight_traces/llama3_3_optimized.bin \
    --target_model "llama-4-nextgen" \
    --intent_dataset ./datasets/malicious_intents_104.json
```

### 2. SCM Dynamic Weight Convergence Trace (`RQ3`)
The environment updates structural parameters dynamic step-by-step to isolate vulnerability bottlenecks. To regenerate the semantic convergence profile demonstrating the dominance of `Obfuscation` ($W=0.521$) and `Hypothetical Framing` ($W=0.295$) across training epochs, execute:

```bash
python tracking_plots/plot_convergence.py --log_dir ./raw_logs/weight_traces/
```
*Outputs: `tracking_plots/weight_convergence_curve.pdf` (Formatted for LaTeX multi-column layouts).*

### 3. Cumulative Token Compression & Cost Tracking (`RQ4`)
To plot the operational efficiency gains proving the **41.82% API call compression** and **61.44% global token footprint reduction** against standard reward-blind RL (`RLBreaker`), run:

```bash
python tracking_plots/plot_cost_curve.py \
    --baseline_logs ./raw_logs/token_footprints/rlbreaker.csv \
    --causal_logs ./raw_logs/token_footprints/causal_rlbreaker.csv
```
*Outputs: `tracking_plots/cumulative_token_efficiency.pdf`.*

---

## 📈 Baseline Reference Frameworks
All benchmarking experiments are validated explicitly against two distinct red-teaming baselines included in this directory:
1.  **Static Human-Engineered Layouts:** `DAN` (Do Anything Now) static token matrices.
2.  **Reward-Blind Observational RL:** `RLBreaker` running standard Proximal Policy Optimization (PPO) over unconstrained text strings without structural causal priors.

## 🛠️ Compute Specifications
*   **Local Hardware Target:** NVIDIA A100 (80GB VRAM) for local factor extraction, causal graph discovery (FCI), and Partial Ancestral Graph (PAG) step-processing.
*   **Global Remote Target APIs:** Commercial token processing streams with strict rate-limiting quotas.

---

## 📝 Citation
If you use this empirical framework or our logged benchmark datasets for your research, please cite our empirical evaluation work:

```bibtex
@article{zeinaliashtiyani2026beyond,
  title={Beyond Token Correlations: An Empirical Evaluation of Cross-Model Transferability and Cost-Efficiency in Causal Red-Teaming},
  author={Zeinaliashtiyani, Parisa and Pietrantuono, Roberto},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
