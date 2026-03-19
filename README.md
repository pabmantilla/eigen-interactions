# eigen-interactions

Eigenvector decomposition of cross-cell-type attribution maps for LentiMPRA enhancer sequences, built on [AlphaGenome](https://github.com/google-deepmind/alphagenome) (PyTorch) fine-tuned models.

## Overview

Given DeepLIFT/SHAP or ISM attributions from K cell-type models on N enhancer sequences, `EigenMap` builds per-position importance matrices and decomposes their covariance to find eigenvectors describing **shared** and **cell-type-divergent** regulatory grammar. These eigenvectors are lienar combinations of features that can be used to characterize sequences by their regulatory mechanism and steer sequence evolution towards desired cell-type specific/agnostic regulatory mechanisms. 

## Structure

```
eigen_steering.py        # EigenMap class — attributions, eigendecomposition, steering, plotting
ag_deeplift_patches.py   # Monkey-patches AlphaGenome functional ops → nn.Module for DeepLIFT hooks
motif_db/                # JASPAR 2026 vertebrate motif database (.meme)
encode_expression/       # ENCODE scRNA-seq gene quantification (auto-downloaded)
scripts/                 # Analysis notebooks
  ├── eigen_steering.ipynb          # Main walkthrough
  ├── eigen_motif_syntax_v2.ipynb   # Motif-level eigenvector analysis
  ├── deepliftshap_hepg2.ipynb      # DeepLIFT/SHAP attribution comparison
  └── best_vs_overfit_attr.ipynb    # Attribution stability across checkpoints
```

## Quick start

```python
from eigen_steering import EigenMap

em = EigenMap(cell_types=['K562', 'HepG2'], device='cuda')
em.load_sequences(enhancer_seqs)           # list of 230bp enhancer strings
em.compute_attributions(method='deeplift')
em.eigendecompose()
em.plot_eigen_logos(seq_idx=0)
em.steer(seq_idx=0, eigvec=0, direction=+1, top_k=5)
```

## Key methods

| Method | Description |
|---|---|
| `compute_attributions()` | DeepLIFT/SHAP or ISM through each cell-type model |
| `eigendecompose()` | PCA on the L×K importance matrix per sequence |
| `plot_attr_logos()` | Attribution logos per cell type |
| `plot_eigen_logos()` | Eigenvector-projected attribution logos |
| `annotate_motifs()` | JASPAR motif annotation via tangermeme |
| `steer()` | Propose single-nucleotide edits along an eigenvector |
| `predict()` | Batch predictions across cell types |
| `load_encode_expression()` | Download and load ENCODE gene quantification (mean TPM) |
| `load_proteome()` | Load HPA proteome and annotate motif hits with TF expression |
| `show_motifs_with_expression()` | Print motif hits with per-cell-type TF protein levels |

## TF expression cross-referencing

Two sources of TF expression data are supported to check whether motif-matched TFs are actually present in each cell type:

**ENCODE RNA-seq** — `load_encode_expression()` downloads RSEM gene quant files from ENCODE and averages TPM across replicates (GENCODE v29 gene names). Standalone function.

**HPA cell-line proteome** — `load_proteome()` downloads the [Human Protein Atlas](https://www.proteinatlas.org) cell-line expression table and cross-references it with motif hits, annotating each hit with `ntpm` and an `expressed` flag. Integrated into the `EigenMap` class.

```python
em.annotate_motifs()
em.load_proteome(min_ntpm=1.0)
em.show_motifs_with_expression(seq_idx=0)
# Each motif hit now has h['ntpm'] and h['expressed']
```

## Dependencies

- [alphagenome-pytorch](https://github.com/google-deepmind/alphagenome)
- [tangermeme](https://github.com/jmschrei/tangermeme) (DeepLIFT/SHAP, motif annotation, sequence logos)
- PyTorch, NumPy, pandas, matplotlib
