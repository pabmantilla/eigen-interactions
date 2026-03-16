# eigen-interactions

Eigenvector decomposition of cross-cell-type attribution maps for LentiMPRA enhancer sequences, built on [AlphaGenome](https://github.com/google-deepmind/alphagenome) (PyTorch) fine-tuned models.

## Overview

Given DeepLIFT/SHAP or ISM attributions from K cell-type models on N enhancer sequences, `EigenMap` builds per-position importance matrices and decomposes their covariance to find eigenvectors describing **shared** and **cell-type-divergent** regulatory grammar.

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

## ENCODE TF expression cross-referencing

Motif hits from `annotate_motifs()` can be cross-referenced against ENCODE RNA-seq gene quantification to check whether matched TFs are actually expressed in each cell type. `load_encode_expression()` auto-downloads RSEM gene quant files from ENCODE and averages TPM across replicates, with gene names mapped via GENCODE v29. Currently configured for K562 and HepG2.

```python
from eigen_steering import load_encode_expression

expr = load_encode_expression(cell_types=['K562', 'HepG2'])
# expr['K562'] -> DataFrame with columns [gene_id, gene_name, TPM]
```

## Dependencies

- [alphagenome-pytorch](https://github.com/google-deepmind/alphagenome)
- [tangermeme](https://github.com/jmschrei/tangermeme) (DeepLIFT/SHAP, motif annotation, sequence logos)
- PyTorch, NumPy, pandas, matplotlib
