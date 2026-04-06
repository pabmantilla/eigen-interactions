# eigen-interactions

Eigenvector decomposition of cross-cell-type DeepLIFT attributions for dissecting shared and cell-type-specific regulatory grammar in LentiMPRA enhancer sequences.

## What this does

Given your LentiMPRA library and cell-type-specific AlphaGenome models, EigenMap supports:

1. **Training/validation** -- load sequences, compute attributions (DeepLIFT or ISM) through multiple cell-type models, and validate predictions against measured activity.

2. **Mechanistic interpretation via EigenMaps** -- eigendecomposition of the cross-cell-type importance matrix E (positions x cell types) extracts eigenvectors capturing shared vs divergent regulatory mechanisms. The EI_1 variance x correlation score quantifies whether a sequence is driven by shared (same-same) or cell-type-specific (diff-diff) grammar.

3. **In silico CRISPR KO and KI** -- necessity (KO) and sufficiency (KI) tests use marginalized dinucleotide-shuffle backgrounds to functionally score each motif per cell type. This replaces reliance on sequence conservation alone: motif annotation (TOMTOM + TFMoDISco) is used downstream of mechanistic description, not as the primary signal.

4. **Isolating mechanism classes** -- sequences are classified by their EI_1 var x r score into diff-diff (cell-type-divergent mechanisms) and same-same (shared mechanisms), enabling targeted analysis of regulatory differences and commonalities.

5. **SEAM decomposition** -- external integration with SEAM for decomposing target regulatory spaces via mutagenesis + attribution clustering.

6. **n-SHAPIQ** -- higher-order Shapley interaction indices (k-SII, orders 1 through max_order) where each motif is a player, computed via necessity or sufficiency games. This is NOT pairwise -- each motif is an independent player and interactions are computed up to arbitrary order. Optional context players (background + promoter) can be included.

7. **Context SHAP** -- 2-player Shapley decomposition of motif syntax vs background context via `shapley_syntax_vs_background()`, quantifying how much expression is driven by the motifs themselves vs their flanking/background sequence.

8. **Motif Context Swap** -- `motif_context_swap()` swaps motif syntax and backgrounds across cell lines, activity bins, and mechanism classes to test whether regulatory grammar is portable across contexts.

## Quick Start

```python
from eigen_steering import EigenMap

em = EigenMap(cell_types=['HepG2', 'K562'], device='cuda')
em.load_sequences(enhancer_seqs)           # list of 230bp strings
em.compute_attributions(method='deeplift', n_shuffles=20)
em.eigendecompose()
em.annotate_motifs()

# Visualize
em.plot_eigen_logos(seq_idx=0)
em.plot_importance_scatter(seq_idx=0)

# Functional tests
nec = em.necessity_test(seq_idx=0, n_rep=20)
suf = em.sufficiency_test(seq_idx=0, n_rep=20)

# n-SHAPIQ (each motif = 1 player, up to order 4)
sii = em.shapley_interaction_index(seq_idx=0, max_order=2, mode='necessity')

# Context SHAP (2-player: syntax vs background)
svb = em.shapley_syntax_vs_background(seq_idx=0, n_rep=20)

# Motif context swap across cell lines
mcs = em.motif_context_swap(seq_idx=None, swap='cell_lines')

# Expression-aware motif ranking
em.load_expression()
em.rank_motif_hits(min_tpm=1.0)
```

## Key Methods

### Sequence loading

| Method | Description |
|---|---|
| `load_sequences(enhancers)` | Load 230bp enhancer strings; auto-appends promoter + barcode to make 281bp constructs |
| `load_from_dataframe(df, seq_col, n, sort_by)` | Load sequences from a DataFrame |

### Attributions and eigendecomposition

| Method | Description |
|---|---|
| `compute_attributions(method, n_shuffles)` | DeepLIFT or ISM through all cell-type models |
| `compute_shard(sequences, cell_type, ...)` | Static method for SLURM-parallelized DeepLIFT shards |
| `merge_shards(output_dir, cell_types, ...)` | Static method to merge shards into a single attribution file |
| `save_attributions(path)` / `load_attributions(path)` | Save/load computed attributions (npz) |
| `eigendecompose(enhancer_only)` | PCA on per-position importance covariance matrix; produces eigenvectors, scores, var ratios |

### Visualization

| Method | Description |
|---|---|
| `plot_attr_logos(seq_idx)` | Per-cell-type attribution logos |
| `plot_eigen_logos(seq_idx)` | Eigenvector-weighted attribution logos |
| `plot_eigentracks(seq_idx)` | Per-position eigenvector score tracks |
| `plot_importance_scatter(seq_idx)` | Cell-type importance scatter (EI_1 var x r) |
| `plot_attr_logos_with_motifs(seq_idx)` | Attribution logos with motif hit annotations |
| `plot_proteome_match(seq_idx)` | Proteomics-matched motif visualization |
| `plot_expression_match(seq_idx)` | Expression-matched motif visualization |

### Functional tests

| Method | Description |
|---|---|
| `necessity_test(seq_idx, n_rep, nec_order)` | Dinucleotide-shuffle KO of motif regions; scores necessity per cell type |
| `sufficiency_test(seq_idx, n_rep, suf_order)` | Dinucleotide-shuffle KI of motifs into shuffled background; scores sufficiency |
| `shapley_interaction_index(seq_idx, max_order, mode)` | n-player k-SII via shapiq; each motif is a player, optional context players |
| `shapley_interaction_index_context(seq_idx, max_order, mode)` | Same as above but with background + promoter as extra players |
| `shapley_syntax_vs_background(seq_idx, n_rep)` | 2-player Shapley: motif syntax vs background context |
| `motif_context_swap(seq_idx, swap, groups)` | Swap syntax/backgrounds across cell lines, activity bins, or mechanism classes |

### Motif annotation and TF matching

| Method | Description |
|---|---|
| `annotate_motifs(meme_file, window_size)` | JASPAR motif scanning via tangermeme (TFMoDISco seqlets + TOMTOM) |
| `show_motifs(seq_idx)` | Print motif hits for a sequence |
| `rank_motif_hits(min_tpm)` | Re-rank TOMTOM hits using EI mechanism score + expression compatibility |
| `load_expression(data_dir)` | Load ENCODE scRNA-seq TPM for expression filtering |
| `expression_match(seq_idx, top_k, min_tpm)` | Match motifs to expressed TFs |
| `load_proteome()` | Load CCLE mass-spec proteomics |
| `proteome_match(seq_idx, top_k)` | Match motifs to detected proteins |

### Predictions

| Method | Description |
|---|---|
| `predict(constructs, cell_type, batch_size, models)` | Forward pass on construct strings; returns `{ct: array}` |
| `set_actual(actual)` | Set measured activity values for comparison |
| `steer(seq_idx, eigvec, direction, top_k)` | Propose sequence edits along an eigenvector |
| `apply_edits(seq_idx, edits)` | Apply proposed edits to a sequence |
| `summary(seq_idx)` | Print eigendecomposition summary for a sequence |

### Genomic context (ChIP-Atlas)

| Method | Description |
|---|---|
| `load_genomic_coords(joint_df)` | Map sequences to hg38 genomic coordinates |
| `query_chipatlas(threshold, antigens)` | Query ChIP-Atlas for TF binding at sequence loci |
| `show_chipatlas(seq_idx)` | Print ChIP-Atlas hits |
| `plot_chipatlas_heatmap(seq_idx, top_k)` | Heatmap of ChIP-Atlas binding across sequences |
| `chipatlas_at_motifs(seq_idx)` | Cross-reference ChIP-Atlas with motif positions |
| `plot_chipatlas_at_motifs(seq_idx)` | Visualize ChIP-Atlas support at motif sites |

## Directory Structure

```
eigen_steering.py              EigenMap class (~3900 lines)
ag_deeplift_patches.py         AlphaGenome DeepLIFT hook patches
fast_logo.py                   Fast logo rendering
motif_db/                      JASPAR2026 vertebrate motif database
scripts/                       Analysis notebooks
```

## Installation

```bash
pip install alphagenome-pytorch tangermeme torch numpy pandas matplotlib
pip install shapiq  # for n-SHAPIQ methods
```

## Model Weights

Models are loaded from `../weights/model_fold_0.safetensors` (two-step fine-tuned AlphaGenome). Available cell types are defined in `DEFAULT_MODELS`: K562, HepG2, WTC11.
