# kernel-conservation

**Heat kernel conservation theory — K = exp(-βL) spectral analysis, SVM classification, perturbation stability, and multi-scale kernel PCA.**

Builds the heat kernel from the tension-graph Laplacian and explores its conservation properties. Tests the hypothesis: the heat kernel of a conservation-structured Laplacian has stable trace under perturbation, while random graphs have unstable trace. Conservation = kernel robustness.

## What This Gives You

- **Heat kernel computation** — K = exp(-βL) via eigendecomposition
- **Kernel trace stability** — conservation-structured graphs resist perturbation
- **SVM classification** — heat kernel as SVM kernel, compare with RBF
- **Multi-scale analysis** — β sweep from local (small β) to global (large β)
- **Kernel PCA** — visualize conservation vs random graph embeddings
- **Perturbation experiments** — add/remove edges, measure kernel trace change

## Quick Start

```bash
pip install numpy scipy matplotlib scikit-learn
python experiment.py
```

Outputs go to `figures/` with PNG plots.

## How It Fits

Part of the SuperInstance ecosystem:

- **[heat-spectral](https://github.com/SuperInstance/heat-spectral)** — Heat diffusion on graphs (Rust)
- **kernel-conservation** — Heat kernel theory and experiments (this repo)

## License

MIT
