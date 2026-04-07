# eigen-interactions

Eigenvector decomposition of cross-cell-type deep learning attributions for dissecting regulatory grammar in multiplex expression assays (LentiMPRA, STARR-seq, UMI-STARR-seq, etc.). Works with any PyTorch genomic deep learning seq2func model (AlphaGenome, MPRALegNet, DeepSTARR, etc.).

## Quick Start (2 cell types)

```python
from eigen_steering import EigenMap

em = EigenMap(cell_types=['HepG2', 'K562'], device='cuda')
em.load_sequences(enhancer_seqs)
em.compute_attributions(method='deeplift', n_shuffles=20)
em.eigendecompose()
em.plot_eigendecomp()
```

With 2 cell types, the DeepSHAP normalization (zero-mean, unit-variance per cell type) forces EI_1's angle to always be 45 degrees, making the eigenvector direction uninformative. Instead we use **EI_1 var x r** — the variance explained by the first eigenvector multiplied by the Pearson correlation between cell-type importances — yielding a score in [-1, 1]. Negative = cell-type-divergent, near zero = mixed, positive = shared.

## Quick Start (3+ cell types)

```python
em = EigenMap(cell_types=['HepG2', 'K562', 'WTC11'], device='cuda')
em.load_sequences(enhancer_seqs)
em.compute_attributions(method='deeplift', n_shuffles=20)
em.eigendecompose()
em.plot_eigendecomp()
```

With 3+ cell types, both eigenvector angles and variance explained are informative. EI_1's direction in cell-type space reveals which cell types share regulatory grammar.

## Features

### Setup

Initialize an EigenMap with your cell types, device, and model configuration. Any PyTorch seq2func model that takes one-hot encoded sequences and returns scalar predictions can be used.

```python
from eigen_steering import EigenMap

em = EigenMap(cell_types=['HepG2', 'K562'], device='cuda')
```

### Load Sequences

Load enhancer sequences as strings. If your library includes a construct backbone (promoter, barcode, etc.), pass it as `construct`; otherwise only the enhancers are used. Can also load from a DataFrame.

```python
em.load_sequences(enhancer_seqs)  # list of enhancer strings

# or from a DataFrame
em.load_from_dataframe(df, seq_col='sequence', n=500, sort_by='activity')
```

### Predictions

Forward pass through all cell-type models. Optionally set measured activity for validation.

```python
preds = em.predict(constructs)  # returns {cell_type: predictions} for all cell types
em.set_actual(...)  # optional
```

### Attribution Maps

Compute per-position DeepLIFT or ISM attributions through all cell-type models. Attributions can be saved/loaded for reuse.

```python
em.compute_attributions(method='deeplift', n_shuffles=20)

em.save_attributions('attrs.npz')
em.load_attributions('attrs.npz')

em.plot_attr_logos(seq_idx=0)
```

For SLURM-parallelized computation, use the static shard/merge methods:

```python
EigenMap.compute_shard(sequences, 'HepG2', 'model_fold_0', output_dir='shards/', shard_idx=0, shard_size=100)
EigenMap.merge_shards('shards/', cell_types=['HepG2', 'K562'], output_path='attrs.npz')
```

### Eigendecomposition

PCA on the per-position importance covariance matrix extracts eigenvectors capturing shared vs divergent regulatory mechanisms. The EI_1 variance x correlation score quantifies cell-type divergence.

```python
em.eigendecompose()
em.summary(seq_idx=0)

em.plot_eigendecomp()
em.plot_eigen_logos(seq_idx=0)
em.plot_eigentracks(seq_idx=0)
em.plot_importance_scatter(seq_idx=0)
```

### Motif Annotation and TF Assignment

Scan sequences with JASPAR motifs via tangermeme (TFMoDISco seqlets + TOMTOM). Motif hits are used by all downstream functional tests. Re-rank TOMTOM hits using a composite binding score informed by the EigenMap mechanism class (EI_1 var x r) and cell-type RNA-seq expression. Filters to TFs actually expressed in each cell line.

```python
em.annotate_motifs()  # defaults to JASPAR2026 vertebrate motifs
em.show_motifs(seq_idx=0)

em.load_expression()
em.rank_motif_hits(min_tpm=1.0)
em.expression_match(seq_idx=0, top_k=1, min_tpm=1.0)

em.plot_attr_logos_with_motifs(seq_idx=0)
em.plot_expression_match(seq_idx=0)
```

### Steering

Propose and apply sequence edits along eigenvectors to steer expression toward a target cell type or mechanism class.

```python
edits = em.steer(seq_idx=0, eigvec=0, direction=1, top_k=5)
em.apply_edits(seq_idx=0, edits=edits)
```

### Necessity and Sufficiency Tests

KO motifs with dinucleotide-shuffle backgrounds (necessity) or KI motifs into shuffled backgrounds (sufficiency) to score each motif's functional importance per cell type.

```python
nec = em.necessity_test(seq_idx=0, n_rep=20)
em.plot_necessity_summary(nec)

suf = em.sufficiency_test(seq_idx=0, n_rep=20)
em.plot_sufficiency_summary(suf)
```

### n-SHAPIQ Interactions

Higher-order Shapley interaction indices where each motif is an independent player, computed via necessity or sufficiency games up to arbitrary order. Optionally include construct context (background, promoter) as extra players to quantify their contribution.

```python
# motifs only
sii = em.shapley_interaction_index(seq_idx=0, max_order=2, mode='necessity')

# motifs + construct context as players
sii_ctx = em.shapley_interaction_index_context(seq_idx=0, max_order=2, mode='necessity')
em.plot_shapiq_summary(sii_ctx)
```

### Context SHAP

2-player Shapley decomposition quantifying how much expression is driven by motif syntax vs flanking/background sequence.

```python
svb = em.shapley_syntax_vs_background(seq_idx=0, n_rep=20)
em.plot_context_shap_summary(svb)
```

### Motif Context Swap

Swap motif syntax and backgrounds across cell lines, activity bins, or mechanism classes to test whether regulatory grammar is portable across contexts.

```python
mcs = em.motif_context_swap(seq_idx=None, swap='cell_lines', n_rep=20)
em.motif_context_swap(seq_idx=None, swap='activity', n_rep=20)
em.motif_context_swap(seq_idx=None, swap='mechanism', n_rep=20)
```

### Genomic Context (ChIP-Atlas)

Map sequences to hg38 coordinates and query ChIP-Atlas for TF binding at sequence loci. Cross-reference binding data with motif positions.

```python
em.load_genomic_coords(joint_df)
em.query_chipatlas(threshold='05')
em.show_chipatlas(seq_idx=0)
em.plot_chipatlas_heatmap(seq_idx=0, top_k=20)
em.chipatlas_at_motifs(seq_idx=0)
em.plot_chipatlas_at_motifs(seq_idx=0)
```

## Directory Structure

```
eigen_steering.py              EigenMap class (~4300 lines)
ag_deeplift_patches.py         AlphaGenome DeepLIFT hook patches
fast_logo.py                   Fast logo rendering
motif_db/                      JASPAR2026 vertebrate motif database
scripts/                       Analysis notebooks
```

## Installation

```bash
pip install tangermeme torch numpy pandas matplotlib
pip install shapiq  # for n-SHAPIQ methods
```

## Model Weights

By default, models are loaded from `DEFAULT_MODELS` (AlphaGenome fine-tuned weights). To use your own models, pass a dict of `{cell_type: model}` to the relevant methods.
