"""EigenMap — Eigenvector decomposition of cross-cell-type attribution maps.

Given DeepLIFT/SHAP (or ISM) attributions through K cell-type models on
N LentiMPRA constructs, we build a per-position importance matrix
E in R^{L x K} and decompose its covariance matrix to find eigenvectors
that describe shared and cell-type-divergent regulatory grammar.

Usage:
    from eigen_steering import EigenMap
    em = EigenMap(cell_types=['K562', 'HepG2'], device='cuda')
    em.load_sequences(seqs)
    em.compute_attributions(method='ism')
    em.eigendecompose()
    em.plot_eigen_logos(seq_idx=0)
    em.steer(seq_idx=0, eigvec=0, direction=+1, top_k=5)
"""

import os
import sys
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))

for _p in [REPO_ROOT, _SCRIPT_DIR, os.path.join(_SCRIPT_DIR, 'tangermeme')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.extensions.finetuning.transfer import remove_all_heads
from alphagenome_pytorch.extensions.finetuning.utils import sequence_to_onehot
from tangermeme.plot import plot_logo
from tangermeme.deep_lift_shap import deep_lift_shap, _nonlinear
from tangermeme.seqlet import tfmodisco_seqlets
from tangermeme.annotate import annotate_seqlets
from tangermeme.io import read_meme
from ag_deeplift_patches import patch_alphagenome, AGCustomGELU

patch_alphagenome()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENCODER_DIM = 1536
BASES = 'ACGT'
BASE2IDX = {b: i for i, b in enumerate(BASES)}

PROMOTER_SEQ = 'TCCATTATATACCCTCTAGTGTCGGTTCACGCAATG'  # 36bp, constant
RAND_BARCODE = 'AGAGACTGAGGCCAC'                       # 15bp, constant
ENHANCER_LEN = 230
PROMOTER_START = ENHANCER_LEN                           # 230
BARCODE_START = ENHANCER_LEN + len(PROMOTER_SEQ)        # 266
TOTAL_LEN = ENHANCER_LEN + len(PROMOTER_SEQ) + len(RAND_BARCODE)  # 281

WEIGHTS_PATH = os.path.join(REPO_ROOT, 'weights', 'model_fold_0.safetensors')
RESULTS_DIR = os.path.join(REPO_ROOT, 'training', 'results')

DEFAULT_MODELS = {
    'K562':  'K562_twostep_v4_do03',
    'HepG2': 'HepG2_twostep_v4_do03',
    'WTC11': 'WTC11_twostep_v6_do075',
}

JASPAR_MEME = os.path.join(_SCRIPT_DIR, 'motif_db', 'JASPAR2026_vertebrates.meme')
REGION_COLORS = {'enhancer': '#2196F3', 'promoter': '#E91E63', 'barcode': '#9C27B0'}

# ---------------------------------------------------------------------------
# ENCODE scRNA-seq gene quantification
# ---------------------------------------------------------------------------
ENCODE_BASE = 'https://www.encodeproject.org'
ENCODE_DATA_DIR = os.path.join(_SCRIPT_DIR, 'encode_expression')
GENCODE_GENE_MAP = os.path.join(ENCODE_DATA_DIR, 'gencode_v29_gene_names.tsv')
GENCODE_GTF_URL = ('https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/'
                   'release_29/gencode.v29.annotation.gtf.gz')

# ENCODE experiment -> replicate file accessions (RSEM gene quant TSVs)
ENCODE_EXPERIMENTS = {
    'HepG2': {
        'experiment': 'ENCSR181ZGR',
        'replicates': ['ENCFF103FSL', 'ENCFF692QVJ'],
    },
    'K562': {
        'experiment': 'ENCSR885DVH',
        'replicates': ['ENCFF713BHS', 'ENCFF366SYN'],
    },
}


def _download_encode_file(accession, dest_dir):
    """Download an ENCODE file by accession if not already cached."""
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f'{accession}.tsv')
    if os.path.exists(dest):
        return dest
    url = f'{ENCODE_BASE}/files/{accession}/@@download/{accession}.tsv'
    print(f"  Downloading {accession} from ENCODE...")
    resp = requests.get(url, allow_redirects=True, timeout=120)
    resp.raise_for_status()
    with open(dest, 'wb') as f:
        f.write(resp.content)
    print(f"  Saved to {dest}")
    return dest


def _ensure_gene_name_map():
    """Download GENCODE v29 GTF and extract gene_id -> gene_name mapping.

    Cached as a 2-column TSV (gene_id_base, gene_name) at GENCODE_GENE_MAP.
    """
    if os.path.exists(GENCODE_GENE_MAP):
        df = pd.read_csv(GENCODE_GENE_MAP, sep='\t')
        return dict(zip(df['gene_id'], df['gene_name']))

    import gzip, re
    os.makedirs(os.path.dirname(GENCODE_GENE_MAP), exist_ok=True)
    print("  Downloading GENCODE v29 annotation (one-time)...")
    resp = requests.get(GENCODE_GTF_URL, stream=True, timeout=300)
    resp.raise_for_status()

    gene_map = {}
    pat = re.compile(r'gene_id "([^"]+)".*gene_name "([^"]+)"')
    for line in gzip.open(resp.raw, 'rt'):
        if line.startswith('#'):
            continue
        fields = line.split('\t')
        if len(fields) < 3 or fields[2] != 'gene':
            continue
        m = pat.search(fields[8])
        if m:
            ensg_full = m.group(1)
            ensg_base = ensg_full.split('.')[0]
            gene_map[ensg_base] = m.group(2)

    # Cache
    pd.DataFrame(list(gene_map.items()),
                 columns=['gene_id', 'gene_name']).to_csv(
        GENCODE_GENE_MAP, sep='\t', index=False)
    print(f"  Cached {len(gene_map)} gene name mappings -> {GENCODE_GENE_MAP}")
    return gene_map


def load_encode_expression(cell_types=None, data_dir=None):
    """Download (if needed) and load ENCODE gene quantification as mean TPM.

    Returns dict: {cell_type: DataFrame with columns [gene_id, gene_name, TPM]}.
    TPM is averaged across replicates. Gene names from GENCODE v29.
    """
    cell_types = cell_types or list(ENCODE_EXPERIMENTS.keys())
    data_dir = data_dir or ENCODE_DATA_DIR
    gene_names = _ensure_gene_name_map()

    expression = {}
    for ct in cell_types:
        if ct not in ENCODE_EXPERIMENTS:
            print(f"  Warning: no ENCODE data configured for {ct}, skipping.")
            continue
        info = ENCODE_EXPERIMENTS[ct]
        rep_dfs = []
        for acc in info['replicates']:
            path = _download_encode_file(acc, os.path.join(data_dir, ct))
            df = pd.read_csv(path, sep='\t')
            df = df[['gene_id', 'TPM']].copy()
            rep_dfs.append(df)
        # Average TPM across replicates
        merged = rep_dfs[0].rename(columns={'TPM': 'TPM_1'})
        for i, rdf in enumerate(rep_dfs[1:], 2):
            merged = merged.merge(
                rdf[['gene_id', 'TPM']].rename(columns={'TPM': f'TPM_{i}'}),
                on='gene_id', how='outer',
            )
        tpm_cols = [c for c in merged.columns if c.startswith('TPM_')]
        merged['TPM'] = merged[tpm_cols].mean(axis=1)
        # Map ENSG -> gene_name
        merged['gene_id_base'] = merged['gene_id'].str.split('.').str[0]
        merged['gene_name'] = merged['gene_id_base'].map(gene_names)
        # Keep only ENSG rows with mapped names
        merged = merged.dropna(subset=['gene_name']).reset_index(drop=True)
        expression[ct] = merged[['gene_id', 'gene_id_base', 'gene_name', 'TPM']].copy()
        print(f"  {ct}: {len(merged)} genes, "
              f"median TPM={merged['TPM'].median():.2f}")
    return expression


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------
class MPRAHead(nn.Module):
    def __init__(self, n_positions=3, nl_size=1024, dropout=0.0,
                 activation='relu', pooling_type='flatten', center_bp=256):
        super().__init__()
        self.pooling_type = pooling_type
        self.n_positions = n_positions
        self.norm = nn.LayerNorm(ENCODER_DIM)
        in_dim = n_positions * ENCODER_DIM if pooling_type == 'flatten' else ENCODER_DIM
        hidden_sizes = [nl_size] if isinstance(nl_size, int) else list(nl_size)
        layers = []
        for hs in hidden_sizes:
            layers.append(nn.Linear(in_dim, hs))
            in_dim = hs
        self.hidden_layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output = nn.Linear(in_dim, 1)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()

    def forward(self, encoder_output):
        x = self.norm(encoder_output)
        if self.pooling_type == 'flatten':
            x = x.flatten(1)
        for layer in self.hidden_layers:
            x = self.act(self.dropout(layer(x)))
        return self.output(x).squeeze(-1)


class AlphaGenomeMPRA(nn.Module):
    """One-hot (B, 4, L) -> (B, 1) for tangermeme, or (B,) if squeeze=True."""
    def __init__(self, encoder, head, squeeze=False):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.squeeze = squeeze

    def forward(self, x):
        x = x.transpose(1, 2)
        org_idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        enc_out = self.encoder(
            x, org_idx, encoder_only=True
        )['encoder_output'].transpose(1, 2)
        pred = self.head(enc_out)
        return pred if self.squeeze else pred.unsqueeze(-1)


# ---------------------------------------------------------------------------
# EigenMap
# ---------------------------------------------------------------------------
class EigenMap:
    """Eigenvector decomposition of cross-cell-type MPRA attribution maps.

    The core object: for each sequence, we build an importance matrix
    E of shape (L, K) where L = number of enhancer positions (230) and
    K = number of cell types. We compute C = E^T E (the K x K covariance
    matrix), then find its eigenvectors. These eigenvectors live in
    cell-type space and tell us the directions of maximum positional
    variance across cell types. Each position's projection onto an
    eigenvector is its "eigenmap coordinate" — how strongly that position
    participates in that mode.

    Promoter (36bp) and barcode (15bp) are constant across all LentiMPRA
    constructs, so they are excluded from the eigendecomposition. Only
    the 230bp enhancer region is decomposed.
    """

    def __init__(self, model_names=None, cell_types=None, device='cuda'):
        if model_names:
            self.model_names = model_names
            self.cell_types = list(model_names.keys())
        else:
            self.cell_types = cell_types or ['K562', 'HepG2']
            self.model_names = {ct: DEFAULT_MODELS[ct] for ct in self.cell_types}
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.constructs: list[str] = []
        self.enhancers: list[str] = []
        self.X: Optional[torch.Tensor] = None        # (N, 4, 281)
        self.attr: dict[str, np.ndarray] = {}         # ct -> (N, 4, 281)
        self.importance: dict[str, np.ndarray] = {}   # ct -> (N, 281)
        self.predictions: dict[str, np.ndarray] = {}  # ct -> (N,) predicted values
        self.actual: dict[str, Optional[np.ndarray]] = {}  # ct -> (N,) if provided
        self.eigen_results: list[dict] = []
        print(f"EigenMap: {self.cell_types}, models={self.model_names}")

    # ----- sequence loading -----
    def load_sequences(self, enhancers: list[str],
                       promoter: str = PROMOTER_SEQ,
                       barcode: str = RAND_BARCODE):
        """Load 230bp enhancer strings, auto-append constant promoter+barcode."""
        self.enhancers = enhancers
        self.constructs = [seq + promoter + barcode for seq in enhancers]
        ohe_list = []
        for c in self.constructs:
            assert len(c) == TOTAL_LEN, f"Expected {TOTAL_LEN}bp, got {len(c)}bp"
            ohe = sequence_to_onehot(c).astype(np.float32)
            ohe_list.append(torch.from_numpy(ohe).T)
        self.X = torch.stack(ohe_list)
        print(f"Loaded {len(self.constructs)} sequences, X shape: {self.X.shape}")
        return self

    def load_from_dataframe(self, df, seq_col='sequence', n=None, sort_by=None):
        """Load from DataFrame."""
        if sort_by:
            df = df.nlargest(n or len(df), sort_by)
        elif n:
            df = df.head(n)
        self.load_sequences(df[seq_col].tolist())
        self._source_df = df.reset_index(drop=True)
        return self

    # ----- model loading -----
    def _load_model(self, ct, squeeze=False):
        model_name = self.model_names[ct]
        ckpt_path = os.path.join(RESULTS_DIR, model_name, 'checkpoints', 'best_stage2.pt')
        print(f"  Loading {ct}: {ckpt_path}")
        enc = AlphaGenome.from_pretrained(WEIGHTS_PATH, device='cpu')
        remove_all_heads(enc)
        hd = MPRAHead()
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        enc.load_state_dict(ckpt['model_state_dict'], strict=False)
        hd.load_state_dict(ckpt['head_state_dict'])
        return AlphaGenomeMPRA(enc, hd, squeeze=squeeze).to(self.device).eval()

    # ----- attributions -----
    def compute_attributions(self, method='deeplift', n_shuffles=20,
                             batch_size=20, verbose=True):
        """Compute attributions. method='deeplift' or 'ism'."""
        assert self.X is not None, "Call load_sequences() first"
        if method == 'deeplift':
            self._compute_deeplift(n_shuffles, batch_size, verbose)
        elif method == 'ism':
            self._compute_ism(verbose)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Importance = attribution at the WT base only (dot with one-hot)
        ohe = self.X.numpy()  # (N, 4, 281)
        for ct in self.cell_types:
            self.importance[ct] = (self.attr[ct] * ohe).sum(axis=1)  # (N, 281)
        return self

    def set_actual(self, actual: dict):
        """Set actual measured expression. actual = {'K562': array, 'HepG2': array}."""
        self.actual = actual
        return self

    def _compute_deeplift(self, n_shuffles, batch_size, verbose):
        for ct in self.cell_types:
            print(f"DeepLIFT/SHAP: {ct}...")
            model = self._load_model(ct, squeeze=False)
            # Store predictions
            with torch.no_grad():
                preds = model(self.X.to(self.device)).squeeze(-1).cpu().numpy()
            self.predictions[ct] = preds
            attr = deep_lift_shap(
                model, self.X, target=0,
                n_shuffles=n_shuffles, batch_size=batch_size,
                device=str(self.device),
                additional_nonlinear_ops={AGCustomGELU: _nonlinear},
                warning_threshold=0.01, random_state=42, verbose=verbose,
            )
            self.attr[ct] = attr.cpu().numpy()
            del model
            torch.cuda.empty_cache()

    def _compute_ism(self, verbose):
        for ct in self.cell_types:
            print(f"ISM: {ct}...")
            model = self._load_model(ct, squeeze=True)
            model.eval()
            all_logos = []
            wt_preds = []
            for si in range(len(self.constructs)):
                x_wt = self.X[si]
                with torch.no_grad():
                    wt_pred = model(x_wt.unsqueeze(0).to(self.device)).cpu().item()
                wt_preds.append(wt_pred)
                mutants, _ = self._gen_mutants(x_wt)
                deltas = []
                for i in range(0, len(mutants), 64):
                    batch = mutants[i:i+64].to(self.device)
                    with torch.no_grad():
                        preds = model(batch).cpu().numpy()
                    deltas.append(preds - wt_pred)
                deltas = np.concatenate(deltas)
                imp = -deltas.reshape(TOTAL_LEN, 3).mean(axis=1)
                logo = x_wt.numpy() * imp[np.newaxis, :]
                all_logos.append(logo)
                if verbose:
                    print(f"  seq {si}: wt={wt_pred:.3f}, max|d|={np.abs(deltas).max():.3f}")
            self.attr[ct] = np.stack(all_logos)
            self.predictions[ct] = np.array(wt_preds)
            del model
            torch.cuda.empty_cache()

    @staticmethod
    def _gen_mutants(onehot):
        L = onehot.shape[1]
        mutants, info = [], []
        for p in range(L):
            ref_idx = onehot[:, p].argmax().item()
            for alt_idx in range(4):
                if alt_idx == ref_idx:
                    continue
                mut = onehot.clone()
                mut[:, p] = 0
                mut[alt_idx, p] = 1
                mutants.append(mut)
                info.append((p, BASES[ref_idx], BASES[alt_idx]))
        return torch.stack(mutants), info

    # ----- eigendecomposition -----
    def eigendecompose(self, enhancer_only=True):
        """Eigendecompose the importance matrix E for each sequence.

        For each sequence:
          1. Build E of shape (L, K) — importance per position per cell type
          2. Restrict to enhancer region (230bp) since promoter+barcode are constant
          3. Scale each cell-type column to zero mean, unit variance so that
             magnitude differences between cell types don't bias the decomposition
          4. Compute the covariance matrix C = (1/L) E^T E, shape (K, K)
          5. Eigendecompose C to get K eigenvectors in cell-type space
          6. Project E onto eigenvectors to get per-position coordinates (scores)

        enhancer_only: if True, only decompose the 230bp enhancer.
        """
        assert self.importance, "Call compute_attributions() first"
        self.eigen_results = []
        n_ct = len(self.cell_types)

        for si in range(self.X.shape[0]):
            # Build the full importance matrix
            imp_full = np.column_stack([
                self.importance[ct][si] for ct in self.cell_types
            ])  # (281, K)

            # Restrict to enhancer region
            if enhancer_only:
                E = imp_full[:ENHANCER_LEN].copy()  # (230, K)
            else:
                E = imp_full.copy()

            # Scale: zero-mean, unit-variance per cell type
            E_scaled = np.zeros_like(E)
            col_means = np.zeros(n_ct)
            col_stds = np.zeros(n_ct)
            for ci in range(n_ct):
                col = E[:, ci]
                col_means[ci] = col.mean()
                col_stds[ci] = col.std()
                if col_stds[ci] > 0:
                    E_scaled[:, ci] = (col - col_means[ci]) / col_stds[ci]
                else:
                    E_scaled[:, ci] = col - col_means[ci]

            # Covariance matrix and eigendecomposition
            C = E_scaled.T @ E_scaled / E_scaled.shape[0]  # (K, K)
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            # Sort descending by eigenvalue
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]  # columns are eigenvectors

            # Fraction of variance explained
            total_var = eigenvalues.sum()
            var_ratio = eigenvalues / total_var if total_var > 0 else eigenvalues

            # Project positions onto eigenvectors -> coordinates in eigenmap
            scores = E_scaled @ eigenvectors  # (L, K)

            self.eigen_results.append({
                'E_raw': imp_full,         # (281, K) before restricting
                'E_scaled': E_scaled,      # (L, K) after scaling
                'cov': C,                  # (K, K)
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,  # (K, K), columns
                'var_ratio': var_ratio,
                'scores': scores,          # (L, K) positions in eigenmap
                'col_means': col_means,
                'col_stds': col_stds,
                'enhancer_only': enhancer_only,
            })
        print(f"Eigendecomposed {len(self.eigen_results)} sequences "
              f"({'enhancer only, 230bp' if enhancer_only else 'full 281bp'})")
        return self

    # ----- visualization -----
    def plot_attr_logos(self, seq_idx=0, figsize=(18, 3)):
        """Plot attribution logos for each cell type on a shared y-scale.

        Title shows model name, predicted value, and actual (if set via set_actual).
        """
        n_ct = len(self.cell_types)
        all_vals = np.concatenate([
            self.attr[ct][seq_idx].ravel() for ct in self.cell_types
        ])
        yabs = max(abs(all_vals.min()), abs(all_vals.max())) * 1.05
        ylim = (-yabs, yabs)

        fig, axes = plt.subplots(n_ct, 1, figsize=(figsize[0], figsize[1] * n_ct))
        if n_ct == 1:
            axes = [axes]

        for ci, ct in enumerate(self.cell_types):
            ax = axes[ci]
            plot_logo(self.attr[ct][seq_idx], ax=ax, ylim=yabs)
            ax.set_ylim(*ylim)
            ax.axvspan(PROMOTER_START, BARCODE_START, alpha=0.10, color=REGION_COLORS['promoter'])
            ax.axvspan(BARCODE_START, TOTAL_LEN, alpha=0.10, color=REGION_COLORS['barcode'])

            # Build title: ct | model_name | pred=X.XX | actual=X.XX
            parts = [ct, self.model_names[ct]]
            if ct in self.predictions:
                parts.append(f'pred={self.predictions[ct][seq_idx]:.3f}')
            if ct in self.actual and self.actual[ct] is not None:
                parts.append(f'actual={self.actual[ct][seq_idx]:.3f}')
            ax.set_title(' | '.join(parts), fontsize=10)
            ax.set_ylabel('Attribution')
        axes[-1].set_xlabel('Position (230bp enhancer | 36bp promoter | 15bp barcode)')
        plt.tight_layout()
        return fig, axes

    def plot_eigen_logos(self, seq_idx=0, figsize=(18, 8)):
        """Attribution logos per cell type + eigenvector-1 weighted reconstruction.

        All logos on the same y-scale (raw attribution units). The final row
        shows the eigenvector-1 logo: the linear combination of cell-type
        attributions weighted by the first eigenvector.
        """
        r = self.eigen_results[seq_idx]
        n_ct = len(self.cell_types)

        # Use raw attributions (not L2-normalized) — shared y-scale instead
        attr_raw = {ct: self.attr[ct][seq_idx] for ct in self.cell_types}

        # Eigenvector-1 weighted logo
        ev1 = r['eigenvectors'][:, 0]  # (K,)
        ev1_logo = sum(
            ev1[ci] * attr_raw[ct] for ci, ct in enumerate(self.cell_types)
        )

        # Shared y-limits across all panels
        all_vals = np.concatenate(
            [attr_raw[ct].ravel() for ct in self.cell_types] + [ev1_logo.ravel()]
        )
        yabs = max(abs(all_vals.min()), abs(all_vals.max())) * 1.05
        ylim = (-yabs, yabs)

        n_rows = n_ct + 1
        fig, axes = plt.subplots(n_rows, 1, figsize=figsize)

        for ci, ct in enumerate(self.cell_types):
            ax = axes[ci]
            plot_logo(attr_raw[ct], ax=ax, ylim=yabs)
            ax.set_ylim(*ylim)
            ax.axvspan(PROMOTER_START, BARCODE_START, alpha=0.10, color=REGION_COLORS['promoter'])
            ax.axvspan(BARCODE_START, TOTAL_LEN, alpha=0.10, color=REGION_COLORS['barcode'])
            w = ev1[ci]
            ax.set_title(f'{ct}  (eigenvector 1 weight: {w:+.3f})', fontsize=10)

        ax = axes[-1]
        plot_logo(ev1_logo, ax=ax, ylim=yabs)
        ax.set_ylim(*ylim)
        ax.axvspan(PROMOTER_START, BARCODE_START, alpha=0.10, color=REGION_COLORS['promoter'])
        ax.axvspan(BARCODE_START, TOTAL_LEN, alpha=0.10, color=REGION_COLORS['barcode'])
        weights_str = ', '.join(f'{ct}:{ev1[ci]:+.2f}'
                                for ci, ct in enumerate(self.cell_types))
        v = r['var_ratio'][0] * 100
        ax.set_title(f'Eigenvector 1 = [{weights_str}]  ({v:.0f}% variance)', fontsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor('#FF5722')
            spine.set_linewidth(2)

        axes[-1].set_xlabel('Position (230bp enhancer | 36bp promoter | 15bp barcode)')
        plt.tight_layout()
        return fig, axes

    def plot_eigentracks(self, seq_idx=0, figsize=(16, 3)):
        """Bar plot of per-position eigenmap coordinates (enhancer only)."""
        r = self.eigen_results[seq_idx]
        n_ev = r['scores'].shape[1]
        L = r['scores'].shape[0]
        positions = np.arange(L)
        ev_colors = ['#2196F3', '#FF9800', '#4CAF50']

        fig, axes = plt.subplots(n_ev, 1, figsize=(figsize[0], figsize[1] * n_ev))
        if n_ev == 1:
            axes = [axes]
        for ei in range(n_ev):
            ax = axes[ei]
            ax.bar(positions, r['scores'][:, ei], width=1.0,
                   color=ev_colors[ei % len(ev_colors)], alpha=0.7)
            ax.axhline(0, color='k', linewidth=0.5)
            ev = r['eigenvectors'][:, ei]
            s = ', '.join(f'{ct}:{ev[ci]:+.2f}' for ci, ct in enumerate(self.cell_types))
            v = r['var_ratio'][ei] * 100
            ax.set_title(f'Eigenvector {ei+1} ({v:.1f}% var) — [{s}]', fontsize=10)
            ax.set_ylabel(f'EV{ei+1} coord')
        axes[-1].set_xlabel(f'Position (enhancer, {L}bp)')
        plt.tight_layout()
        return fig, axes

    def plot_importance_scatter(self, seq_idx=0):
        """Scatter: cell type A importance vs cell type B (enhancer only)."""
        assert len(self.cell_types) == 2, "Scatter requires exactly 2 cell types"
        r = self.eigen_results[seq_idx]
        E = r['E_scaled']  # (230, 2), enhancer only, scaled
        L = E.shape[0]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(E[:, 0], E[:, 1], c=REGION_COLORS['enhancer'], s=25, alpha=0.6)

        # Draw eigenvector directions
        ev_colors = ['#FF5722', '#FF9800']
        for ei in range(2):
            ev = r['eigenvectors'][:, ei]
            scale = np.abs(E).max() * 0.8
            dx, dy = ev * scale
            v = r['var_ratio'][ei] * 100
            ax.annotate('', xy=(dx, dy), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=ev_colors[ei], lw=2.5))
            ax.text(dx * 1.15, dy * 1.15, f'EV{ei+1} ({v:.0f}%)',
                    color=ev_colors[ei], fontsize=10, fontweight='bold')

        lims = [E.min() * 1.1, E.max() * 1.1]
        ax.plot(lims, lims, 'k--', linewidth=0.5, alpha=0.4)
        ax.axhline(0, color='k', linewidth=0.3)
        ax.axvline(0, color='k', linewidth=0.3)
        r_val = np.corrcoef(E[:, 0], E[:, 1])[0, 1]
        ax.set_xlabel(f'{self.cell_types[0]} importance (scaled)', fontsize=12)
        ax.set_ylabel(f'{self.cell_types[1]} importance (scaled)', fontsize=12)
        ax.set_title(f'Position importance (r={r_val:.3f})', fontsize=13)
        ax.set_aspect('equal')
        plt.tight_layout()
        return fig, ax

    # ----- steering -----
    def get_top_positions(self, seq_idx=0, eigvec=0, top_k=10):
        """Top positions by |coordinate| on an eigenvector."""
        r = self.eigen_results[seq_idx]
        scores = r['scores'][:, eigvec]
        ranked = np.argsort(np.abs(scores))[::-1][:top_k]
        seq = self.constructs[seq_idx]

        results = []
        for rank, p in enumerate(ranked):
            start = max(0, p - 5)
            end = min(len(seq), p + 6)
            ctx = seq[start:end]
            rel = p - start
            ctx_marked = ctx[:rel] + f'[{ctx[rel]}]' + ctx[rel + 1:]
            results.append({
                'rank': rank, 'pos': p,
                'score': scores[p], 'context': ctx_marked,
                'base': seq[p],
                'imp': {ct: self.importance[ct][seq_idx, p] for ct in self.cell_types},
            })
        return results

    def steer(self, seq_idx=0, eigvec=0, direction=1, top_k=5):
        """Propose single-base edits along an eigenvector direction.

        Finds enhancer positions opposing the desired direction and flips
        them to the base with highest attribution for the dominant cell type.
        """
        r = self.eigen_results[seq_idx]
        scores = r['scores'][:, eigvec]
        seq = list(self.constructs[seq_idx])
        ev = r['eigenvectors'][:, eigvec]

        target = scores * direction
        enh_positions = np.arange(len(scores))  # already enhancer-only
        opposing = enh_positions[np.argsort(target)[:top_k]]

        dominant_ct_idx = np.argmax(np.abs(ev))
        dominant_ct = self.cell_types[dominant_ct_idx]

        edits = []
        for p in opposing:
            ref_base = seq[p]
            attr_at_pos = self.attr[dominant_ct][seq_idx, :, p].copy()
            ref_idx = BASE2IDX[ref_base]
            attr_at_pos[ref_idx] = -np.inf
            best_alt_idx = np.argmax(attr_at_pos)
            edits.append({
                'pos': int(p), 'ref': ref_base, 'alt': BASES[best_alt_idx],
                'ev_score': float(scores[p]), 'dominant_ct': dominant_ct,
            })

        return {
            'eigvec': eigvec, 'direction': direction,
            'eigvec_weights': {ct: float(ev[ci]) for ci, ct in enumerate(self.cell_types)},
            'edits': edits,
        }

    def apply_edits(self, seq_idx, edits):
        """Apply edits, return new construct string."""
        seq = list(self.constructs[seq_idx])
        for e in edits:
            seq[e['pos']] = e['alt']
        return ''.join(seq)

    def predict(self, constructs, cell_type=None, batch_size=64):
        """Forward pass on construct strings. Returns {ct: array}."""
        cts = [cell_type] if cell_type else self.cell_types
        X = []
        for c in constructs:
            ohe = sequence_to_onehot(c).astype(np.float32)
            X.append(torch.from_numpy(ohe).T)
        X = torch.stack(X)
        preds = {}
        for ct in cts:
            model = self._load_model(ct, squeeze=True)
            chunks = []
            with torch.no_grad():
                for i in range(0, len(X), batch_size):
                    batch = X[i:i+batch_size].to(self.device)
                    chunks.append(model(batch).cpu().numpy())
            preds[ct] = np.concatenate(chunks)
            del model
            torch.cuda.empty_cache()
        return preds

    def summary(self, seq_idx=0):
        """Print eigendecomposition summary."""
        r = self.eigen_results[seq_idx]
        print(f"Sequence {seq_idx}: {self.constructs[seq_idx][:30]}...")
        for ei in range(r['eigenvectors'].shape[1]):
            ev = r['eigenvectors'][:, ei]
            weights = ', '.join(f'{ct}:{ev[ci]:+.3f}'
                                for ci, ct in enumerate(self.cell_types))
            v = r['var_ratio'][ei] * 100
            print(f"  Eigenvector {ei+1} ({v:.1f}%): [{weights}]")

        tops = self.get_top_positions(seq_idx, eigvec=0, top_k=5)
        print(f"\n  Top 5 positions on eigenvector 1:")
        for t in tops:
            imp_str = ', '.join(f'{ct}:{t["imp"][ct]:+.3f}' for ct in self.cell_types)
            print(f"    pos={t['pos']:3d}  coord={t['score']:+.4f}"
                  f"  {t['context']}  [{imp_str}]")

    # ----- motif annotation -----
    @staticmethod
    def _tf_name(motif_key):
        """Extract TF name from JASPAR key like 'MA0004.1 Arnt' or 'MA0004.1_Arnt'."""
        for sep in (' ', '_'):
            if sep in motif_key:
                return motif_key.split(sep, 1)[-1]
        return motif_key

    def _annotate_one_ct(self, ct, X_tensor, motifs, motif_names,
                         window_size, flank, pval_thresh, **tomtom_kwargs):
        """Run seqlet calling + TOMTOM for one cell type. Returns per-seq hit lists."""
        attr_tensor = torch.from_numpy(self.attr[ct]).float()
        projected = (attr_tensor * X_tensor).sum(dim=1)
        seqlets = tfmodisco_seqlets(projected, window_size=window_size, flank=flank)
        n_seq = len(self.constructs)
        per_seq = [[] for _ in range(n_seq)]
        if len(seqlets) == 0:
            return per_seq, 0

        idxs, pvals = annotate_seqlets(X_tensor, seqlets, motifs, n_nearest=1,
                                        **tomtom_kwargs)
        for si in range(len(seqlets)):
            midx = int(idxs[si, 0])
            pv = float(pvals[si, 0])
            if midx < 0 or pv >= pval_thresh:
                continue
            seq_i = int(seqlets.iloc[si, 0])
            per_seq[seq_i].append({
                'start': int(seqlets.iloc[si, 1]),
                'end': int(seqlets.iloc[si, 2]),
                'tf': self._tf_name(motif_names[midx]),
                'pval': pv,
                'attribution': float(seqlets.iloc[si, 3]) if seqlets.shape[1] > 3 else 0.0,
            })
        # Deduplicate overlapping hits: keep best p-value per overlapping group
        for i in range(n_seq):
            hits = sorted(per_seq[i], key=lambda x: x['pval'])
            kept = []
            for h in hits:
                if not any(h['start'] < k['end'] and h['end'] > k['start'] for k in kept):
                    kept.append(h)
            per_seq[i] = sorted(kept, key=lambda x: x['start'])
        n_hits = sum(len(h) for h in per_seq)
        return per_seq, n_hits

    def annotate_motifs(self, meme_file=None, window_size=21,
                        flank=10, pval_thresh=0.05, **tomtom_kwargs):
        """Call seqlets and annotate with TOMTOM for all cell types.

        Results stored in self.motif_hits[ct] — a dict mapping cell type
        to a list (per sequence) of hit dicts.
        Returns self for chaining.
        """
        meme_file = meme_file or JASPAR_MEME
        X_tensor = self.X.float()
        motifs = read_meme(meme_file)
        motif_names = list(motifs.keys())

        self.motif_hits = {}
        for ct in self.cell_types:
            assert ct in self.attr, f"No attributions for {ct}. Run compute_attributions() first."
            per_seq, n_hits = self._annotate_one_ct(
                ct, X_tensor, motifs, motif_names,
                window_size, flank, pval_thresh, **tomtom_kwargs)
            self.motif_hits[ct] = per_seq
            print(f"  {ct}: {n_hits} motif hits (p<{pval_thresh})")
        return self

    def show_motifs(self, seq_idx=0):
        """Print motif hits for a sequence, grouped by cell type."""
        assert hasattr(self, 'motif_hits'), "Run annotate_motifs() first."
        for ct in self.cell_types:
            hits = self.motif_hits[ct][seq_idx]
            print(f"  {ct}: {len(hits)} hits")
            for h in hits:
                region = 'enhancer' if h['end'] <= ENHANCER_LEN else 'promoter' if h['start'] < BARCODE_START else 'barcode'
                print(f"    [{h['start']:3d}-{h['end']:3d}] {h['tf']:20s}  p={h['pval']:.1e}  ({region})")

    def plot_attr_logos_with_motifs(self, seq_idx=0, figsize=(18, 4.5)):
        """Plot attribution logos with per-cell-type motif annotations."""
        fig, axes = self.plot_attr_logos(seq_idx=seq_idx, figsize=figsize)
        if not hasattr(self, 'motif_hits'):
            return fig, axes
        for ci, ct in enumerate(self.cell_types):
            ax = axes[ci]
            hits = self.motif_hits[ct][seq_idx]
            if not hits:
                continue
            ylim = ax.get_ylim()
            yrange = ylim[1] - ylim[0]
            ax.set_ylim(ylim[0], ylim[1] + yrange * 0.3)
            for h in hits:
                mid = (h['start'] + h['end']) / 2
                ax.annotate(h['tf'], xy=(mid, ylim[1] + yrange * 0.05),
                            fontsize=7, ha='center', va='bottom',
                            rotation=45, color='#333', fontweight='bold')
                ax.axvspan(h['start'], h['end'], alpha=0.08, color='#FF9800')
        plt.tight_layout()
        return fig, axes
