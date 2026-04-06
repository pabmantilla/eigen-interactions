# eigen-interactions

Eigenvector decomposition of cross-cell-type DeepLIFT attributions for dissecting shared and cell-type-specific regulatory grammar in LentiMPRA enhancer sequences.

## Overview

Given attributions (DeepLIFT, SHAP, or ISM) from K cell-type models across N sequences, EigenMap builds a per-position importance matrix and decomposes it to extract eigenvectors capturing regulatory mechanisms. Use these eigenvectors to steer sequences, test motif necessity/sufficiency, and analyze higher-order epistasis via Shapley interactions.

## Key Features

- **Eigen decomposition** of cross-cell-type attributions via covariance decomposition
- **DeepLIFT/SHAP/ISM** attribution computation (single-pass batched inference)
- **Motif annotation** via JASPAR and tangermeme
- **Necessity/sufficiency tests** using dinucleotide-shuffle knockouts and insertions
- **Shapley interaction indices** for pairwise and higher-order regulatory epistasis
- **TF expression filtering** via CCLE proteomics and ENCODE scRNA-seq
- **Logo visualization** for eigenvector-weighted importance

## Quick Start

```python
from eigen_steering import EigenMap

em = EigenMap(cell_types=['HepG2', 'K562'], device='cuda')
em.load_sequences(enhancer_seqs)
em.compute_attributions(method='deeplift', n_shuffles=20)
em.eigendecompose()
em.annotate_motifs()
em.plot_eigen_logos(seq_idx=0)
em.necessity_test(seq_idx=0, n_shuffles=100)
em.shapley_interaction_index(seq_idx=0, pos_pairs=[(50, 100)])
```

## Key Methods

| Method | Description |
|---|---|
| `load_sequences(seqs)` | Load one-hot encoded sequences (N, 4, 281) |
| `compute_attributions(method, n_shuffles)` | Compute DeepLIFT/SHAP/ISM through all cell-type models |
| `eigendecompose()` | PCA on per-position importance covariance matrix |
| `annotate_motifs()` | JASPAR motif hits via tangermeme |
| `plot_eigen_logos(seq_idx)` | Eigenvector-weighted attribution logos |
| `necessity_test(seq_idx, n_shuffles)` | Dinucleotide-shuffle KO test for motif function |
| `sufficiency_test(seq_idx, n_shuffles)` | Dinucleotide-shuffle KI test for motif function |
| `shapley_interaction_index(seq_idx, pos_pairs)` | Pairwise regulatory epistasis via Shapley values |
| `steer(seq_idx, eigvec, direction, top_k)` | Propose sequence edits along an eigenvector |
| `predict(X)` | Batch predictions across cell types |

## Directory Structure

```
eigen_steering.py              EigenMap class & attribution methods
ag_deeplift_patches.py         AlphaGenome DeepLIFT hook patches
fast_logo.py                   Fast logo rendering
motif_db/                      JASPAR2026 vertebrate motif database
encode_expression/             ENCODE/CCLE gene quantification caches
scripts/                       Analysis notebooks (eigen_steering.ipynb, etc)
```

## Installation

```bash
pip install alphagenome-pytorch tangermeme torch numpy pandas matplotlib
```

## Model Weights

Models are loaded from `../weights/model_fold_0.safetensors` (K562, HepG2, WTC11 cell-type models, two-step architecture). See `DEFAULT_MODELS` in eigen_steering.py for available cell types.

## Dependencies

- `alphagenome-pytorch` (AlphaGenome architecture & models)
- `tangermeme` (attribution, motif scanning, logo rendering)
- PyTorch, NumPy, pandas, matplotlib
