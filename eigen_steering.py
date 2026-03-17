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
import urllib.request
try:
    import requests
except ImportError:
    requests = None
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
# ChIP-Atlas constants
# ---------------------------------------------------------------------------
CHIPATLAS_DATA = 'https://chip-atlas.dbcls.jp/data'
CHIPATLAS_EXPLIST_URL = f'{CHIPATLAS_DATA}/metadata/experimentList.tab'
CHIPATLAS_CACHE = os.path.join(_SCRIPT_DIR, 'chipatlas_cache')
CHIPATLAS_CELL_MAP = {
    'K562':  ['K-562'],
    'HepG2': ['Hep G2'],
    'WTC11': ['iPS cells'],  # WTC11 listed under "iPS cells"; filtered by title
}
JOINT_LIBRARY_PATH = os.path.join(
    os.path.dirname(REPO_ROOT),  # LentiMPRA_mcs/
    'Cell_line_MoCon', 'Cross-line_analysis', 'pred_first',
    'joint_data', 'joint_library_combined.csv'
)

# ---------------------------------------------------------------------------
# CCLE mass-spec proteomics (Gygi lab, DepMap)
# ---------------------------------------------------------------------------
CCLE_PROTEOMICS_URL = ('https://gygi.hms.harvard.edu/data/ccle/'
                       'protein_quant_current_normalized.csv.gz')
CCLE_PROTEOMICS_CACHE = os.path.join(_SCRIPT_DIR, 'encode_expression',
                                     'ccle_proteomics.csv.gz')
# Column names in the CCLE CSV: CELLLINE_TISSUE_TenPxNN
CCLE_CELL_LINE_MAP = {
    'K562': 'K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',
    'HepG2': 'HEPG2_LIVER',
    'WTC11': None,
}

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


def _download_file(url, dest):
    """Download a URL to dest, using requests if available, else urllib."""
    if requests is not None:
        resp = requests.get(url, allow_redirects=True, timeout=120)
        resp.raise_for_status()
        with open(dest, 'wb') as f:
            f.write(resp.content)
    else:
        urllib.request.urlretrieve(url, dest)


def _download_encode_file(accession, dest_dir):
    """Download an ENCODE file by accession if not already cached."""
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f'{accession}.tsv')
    if os.path.exists(dest):
        return dest
    url = f'{ENCODE_BASE}/files/{accession}/@@download/{accession}.tsv'
    print(f"  Downloading {accession} from ENCODE...")
    _download_file(url, dest)
    print(f"  Saved to {dest}")
    return dest


def _ensure_gene_name_map():
    """Download GENCODE v29 GTF and extract gene_id -> gene_name mapping.

    Cached as a 2-column TSV (gene_id_base, gene_name) at GENCODE_GENE_MAP.
    """
    if os.path.exists(GENCODE_GENE_MAP):
        df = pd.read_csv(GENCODE_GENE_MAP, sep='\t')
        return dict(zip(df['gene_id'], df['gene_name']))

    import gzip, re, tempfile
    os.makedirs(os.path.dirname(GENCODE_GENE_MAP), exist_ok=True)
    print("  Downloading GENCODE v29 annotation (one-time)...")
    tmp_path = os.path.join(tempfile.gettempdir(), 'gencode_v29.gtf.gz')
    _download_file(GENCODE_GTF_URL, tmp_path)

    gene_map = {}
    pat = re.compile(r'gene_id "([^"]+)".*gene_name "([^"]+)"')
    with gzip.open(tmp_path, 'rt') as fh:
        for line in fh:
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
    os.unlink(tmp_path)

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


def load_ccle_proteomics(cell_types=None):
    """Download (if needed) CCLE mass-spec proteomics and return per-cell-type DataFrames.

    Returns dict: {cell_type: DataFrame with columns [gene_name, protein_abundance]}.
    protein_abundance is log2 relative abundance (TMT-normalized, Gygi lab).
    """
    cell_types = cell_types or ['K562', 'HepG2']

    if not os.path.exists(CCLE_PROTEOMICS_CACHE):
        os.makedirs(os.path.dirname(CCLE_PROTEOMICS_CACHE), exist_ok=True)
        print("  Downloading CCLE proteomics (one-time, ~30MB)...")
        _download_file(CCLE_PROTEOMICS_URL, CCLE_PROTEOMICS_CACHE)
        print(f"  Cached -> {CCLE_PROTEOMICS_CACHE}")

    # Find the column for each cell type (CELLLINE_TISSUE_TenPxNN)
    # Read just header to find columns
    df_header = pd.read_csv(CCLE_PROTEOMICS_CACHE, nrows=0)
    all_cols = list(df_header.columns)

    ct_cols = {}
    for ct in cell_types:
        prefix = CCLE_CELL_LINE_MAP.get(ct)
        if prefix is None:
            print(f"  Warning: {ct} not in CCLE map, skipping.")
            continue
        matches = [c for c in all_cols if c.startswith(prefix + '_TenPx')]
        if not matches:
            print(f"  Warning: no CCLE column for {ct} (prefix={prefix}), skipping.")
            continue
        ct_cols[ct] = matches[0]

    if not ct_cols:
        return {}

    # Load only needed columns
    use_cols = ['Gene_Symbol'] + list(ct_cols.values())
    df = pd.read_csv(CCLE_PROTEOMICS_CACHE, usecols=use_cols)

    result = {}
    for ct, col in ct_cols.items():
        sub = df[['Gene_Symbol', col]].dropna(subset=['Gene_Symbol', col]).copy()
        sub = sub.rename(columns={'Gene_Symbol': 'gene_name',
                                  col: 'protein_abundance'})
        sub['gene_name'] = sub['gene_name'].astype(str).str.upper()
        sub = sub.reset_index(drop=True)
        result[ct] = sub
        print(f"  {ct}: {len(sub)} proteins quantified, "
              f"median log2={sub['protein_abundance'].median():.2f}")
    return result


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

        # Gradient correction: mean-center across the 4 nucleotide channels
        # then multiply by one-hot so attr stores visualization-ready maps
        ohe = self.X.numpy()  # (N, 4, L)
        for ct in self.cell_types:
            corrected = (self.attr[ct]
                         - self.attr[ct].mean(axis=1, keepdims=True))
            self.attr[ct] = corrected * ohe  # (N, 4, L) — logo-ready
            self.importance[ct] = self.attr[ct].sum(axis=1)  # (N, L)
        return self

    def save_attributions(self, path):
        """Save attributions, importance, and predictions to a compressed .npz file."""
        data = {}
        for ct in self.cell_types:
            if ct in self.attr:
                data[f'attr_{ct}'] = self.attr[ct]
            if ct in self.importance:
                data[f'importance_{ct}'] = self.importance[ct]
            if ct in self.predictions:
                data[f'predictions_{ct}'] = self.predictions[ct]
        data['cell_types'] = np.array(self.cell_types)
        np.savez_compressed(path, **data)
        import os as _os
        size_mb = _os.path.getsize(path if path.endswith('.npz') else path + '.npz') / 1e6
        print(f"Saved attributions to {path} ({size_mb:.1f} MB)")
        return self

    def load_attributions(self, path):
        """Load attributions, importance, and predictions from a .npz file."""
        data = np.load(path, allow_pickle=False)
        for ct in self.cell_types:
            if f'attr_{ct}' in data:
                self.attr[ct] = data[f'attr_{ct}']
            if f'importance_{ct}' in data:
                self.importance[ct] = data[f'importance_{ct}']
            if f'predictions_{ct}' in data:
                self.predictions[ct] = data[f'predictions_{ct}']
        print(f"Loaded attributions for {list(self.attr.keys())} from {path}")
        return self

    def set_actual(self, actual: dict):
        """Set actual measured expression. actual = {'K562': array, 'HepG2': array}."""
        self.actual = actual
        return self

    def _compute_deeplift(self, n_shuffles, batch_size, verbose):
        for ct in self.cell_types:
            print(f"DeepLIFT/SHAP: {ct}...")
            model = self._load_model(ct, squeeze=False)
            # Store predictions (batched to avoid OOM)
            pred_chunks = []
            with torch.no_grad():
                for i in range(0, len(self.X), batch_size):
                    chunk = self.X[i:i+batch_size].to(self.device)
                    pred_chunks.append(model(chunk).squeeze(-1).cpu())
            self.predictions[ct] = torch.cat(pred_chunks).numpy()
            del pred_chunks
            torch.cuda.empty_cache()
            attr = deep_lift_shap(
                model, self.X, target=0,
                n_shuffles=n_shuffles, batch_size=batch_size,
                device=str(self.device),
                additional_nonlinear_ops={AGCustomGELU: _nonlinear},
                warning_threshold=0.01, random_state=None, verbose=verbose,
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
                         window_size, flank, pval_thresh, n_nearest=3,
                         **tomtom_kwargs):
        """Run seqlet calling + TOMTOM for one cell type. Returns per-seq hit lists.

        Each hit stores 'top_hits': list of up to n_nearest (tf, pval) tuples.
        """
        attr_tensor = torch.from_numpy(self.attr[ct]).float()
        projected = (attr_tensor * X_tensor).sum(dim=1)
        seqlets = tfmodisco_seqlets(projected, window_size=window_size, flank=flank)
        n_seq = len(self.constructs)
        per_seq = [[] for _ in range(n_seq)]
        if len(seqlets) == 0:
            return per_seq, 0

        idxs, pvals = annotate_seqlets(X_tensor, seqlets, motifs,
                                        n_nearest=n_nearest, **tomtom_kwargs)
        for si in range(len(seqlets)):
            midx0 = int(idxs[si, 0])
            pv0 = float(pvals[si, 0])
            if midx0 < 0 or pv0 >= pval_thresh:
                continue
            seq_i = int(seqlets.iloc[si, 0])
            top_hits = []
            for ki in range(n_nearest):
                midx = int(idxs[si, ki])
                pv = float(pvals[si, ki])
                if midx < 0 or pv >= pval_thresh:
                    continue
                top_hits.append({
                    'tf': self._tf_name(motif_names[midx]),
                    'pval': pv,
                })
            per_seq[seq_i].append({
                'start': int(seqlets.iloc[si, 1]),
                'end': int(seqlets.iloc[si, 2]),
                'tf': top_hits[0]['tf'],
                'pval': pv0,
                'top_hits': top_hits,
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
                        flank=10, pval_thresh=0.05, n_nearest=3,
                        **tomtom_kwargs):
        """Call seqlets and annotate with TOMTOM for all cell types.

        Results stored in self.motif_hits[ct] — a dict mapping cell type
        to a list (per sequence) of hit dicts. Each hit has 'top_hits'
        with up to n_nearest candidates.
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
                window_size, flank, pval_thresh, n_nearest=n_nearest,
                **tomtom_kwargs)
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
                top_str = ', '.join(f"{th['tf']}(p={th['pval']:.1e})"
                                    for th in h.get('top_hits', [{'tf': h['tf'], 'pval': h['pval']}]))
                print(f"    [{h['start']:3d}-{h['end']:3d}] {top_str}  ({region})")

    def load_proteome(self):
        """Load CCLE mass-spec proteomics and build TF -> protein abundance lookup.

        Stores self.proteome (dict of DataFrames) and
        self._tf_protein: {gene_name_upper: {ct: log2_abundance}}.
        """
        self.proteome = load_ccle_proteomics(cell_types=self.cell_types)
        self._tf_protein = {}
        for ct, pdf in self.proteome.items():
            for _, row in pdf.iterrows():
                name = row['gene_name']  # already upper
                if name not in self._tf_protein:
                    self._tf_protein[name] = {}
                self._tf_protein[name][ct] = row['protein_abundance']
        print(f"  Protein lookup: {len(self._tf_protein)} genes across "
              f"{list(self.proteome.keys())}")
        return self

    def proteome_match(self, seq_idx=0, top_k=1):
        """Match motif hits to CCLE protein abundance, preferring detected proteins.

        Requires load_proteome() to have been called first.
        Returns list of dicts with 'protein' key holding per-gene per-ct abundance.
        """
        assert hasattr(self, 'motif_hits'), "Run annotate_motifs() first."
        assert hasattr(self, '_tf_protein'), "Run load_proteome() first."

        positions = self._collect_motif_positions(seq_idx)
        results = []
        for pos in positions:
            selected = []
            for tf in pos['tf_names']:
                components = self._parse_tf_components(tf)
                for comp in components:
                    if comp in self._tf_protein:
                        selected.append(tf)
                        break
                if len(selected) >= top_k:
                    break
            if not selected:
                selected = pos['tf_names'][:top_k]

            for tf in selected:
                components = self._parse_tf_components(tf)
                all_genes = set(self._tf_protein.keys())
                paralogs = []
                seen = set(components)
                for comp in components:
                    prefix = self._tf_family_prefix(comp)
                    if len(prefix) < 2:
                        continue
                    for gene in all_genes:
                        if gene in seen:
                            continue
                        if self._tf_family_prefix(gene) == prefix:
                            if gene in self._tf_protein:
                                paralogs.append(gene)
                                seen.add(gene)
                paralogs.sort(
                    key=lambda g: max(self._tf_protein.get(g, {}).get(ct, -99)
                                      for ct in self.cell_types), reverse=True)
                paralogs = paralogs[:5]
                genes = components + paralogs
                protein = {}
                for g in genes:
                    protein[g] = {ct: self._tf_protein.get(g, {}).get(ct, float('nan'))
                                  for ct in self.cell_types}
                results.append({
                    'start': pos['start'], 'end': pos['end'],
                    'mid': pos['mid'],
                    'tf': tf, 'components': components,
                    'paralogs': paralogs, 'genes': genes, 'protein': protein,
                    'all_tfs': pos['tf_names'],
                })
        return results

    def plot_proteome_match(self, seq_idx=0, top_k=1,
                            max_paralogs=4, figsize=(20, None),
                            annotation_style='heatmap'):
        """Attribution logos with CCLE protein abundance heatmaps at motif sites.

        annotation_style: 'heatmap' (inline, aligned to motifs) or 'bars'.
        Values are log2 relative protein abundance (TMT mass-spec, Gygi/DepMap).
        """
        assert hasattr(self, 'motif_hits'), "Run annotate_motifs() first."
        assert hasattr(self, '_tf_protein'), "Run load_proteome() first."

        matches = self.proteome_match(seq_idx, top_k=top_k)
        if not matches:
            print("No motif hits to plot.")
            return None, None

        for m in matches:
            m['paralogs'] = m['paralogs'][:max_paralogs]
            m['genes'] = m['components'] + m['paralogs']
            for g in m['genes']:
                if g not in m['protein']:
                    m['protein'][g] = {
                        ct: self._tf_protein.get(g, {}).get(ct, float('nan'))
                        for ct in self.cell_types}

        if annotation_style == 'heatmap':
            return self._plot_inline_heatmap(
                seq_idx, matches, value_key='protein',
                unit='log2 protein', figsize=figsize)
        else:
            return self._plot_inline_bars(
                seq_idx, matches, value_key='protein',
                unit='log2 protein', min_val=0.0, figsize=figsize)

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

    # ----- ENCODE expression matching -----
    def load_expression(self, data_dir=None):
        """Download ENCODE gene quantification and build TF name -> TPM lookup.

        Populates self.expression: {cell_type: DataFrame}
        and self.tf_tpm: {tf_name_upper: {cell_type: TPM}}.
        """
        expr_dfs = load_encode_expression(self.cell_types, data_dir)
        self.expression = expr_dfs
        self.tf_tpm = {}
        for ct, df in expr_dfs.items():
            for _, row in df.iterrows():
                name = row['gene_name'].upper()
                if name not in self.tf_tpm:
                    self.tf_tpm[name] = {}
                self.tf_tpm[name][ct] = row['TPM']
        print(f"  TF lookup: {len(self.tf_tpm)} gene symbols across "
              f"{list(expr_dfs.keys())}")
        return self

    # ----- TF name helpers -----
    @staticmethod
    def _parse_tf_components(tf_name):
        """Split composite motif names like 'Ahr::Arnt' into individual genes."""
        parts = tf_name.replace('::', ':').split(':')
        return [p.strip().upper() for p in parts if p.strip()]

    @staticmethod
    def _tf_family_prefix(gene):
        """Strip trailing digits to get family prefix: GATA1 -> GATA."""
        import re
        m = re.match(r'^([A-Za-z]+)', gene)
        return m.group(1).upper() if m else gene.upper()

    def _tf_tpm_for(self, tf_name, ct):
        """Look up TPM for a TF in a cell type.

        For composite motifs (Ahr::Arnt), returns max TPM across components.
        """
        if not hasattr(self, 'tf_tpm'):
            return 0.0
        components = self._parse_tf_components(tf_name)
        tpms = [self.tf_tpm.get(g, {}).get(ct, 0.0) for g in components]
        return max(tpms) if tpms else 0.0

    def _tf_is_expressed(self, tf_name, min_tpm=1.0):
        """True if any component is expressed (>= min_tpm) in any cell type."""
        for ct in self.cell_types:
            if self._tf_tpm_for(tf_name, ct) >= min_tpm:
                return True
        return False

    def _find_paralogs(self, tf_name, max_paralogs=5):
        """Find paralogs: genes sharing the same family prefix.

        E.g. GATA1 -> [GATA2, GATA3, ...]. For composites, returns
        paralogs for all components. Only expressed paralogs included.
        """
        if not hasattr(self, 'tf_tpm'):
            return []
        components = self._parse_tf_components(tf_name)
        all_genes = set(self.tf_tpm.keys())
        paralogs = []
        seen = set(components)
        for comp in components:
            prefix = self._tf_family_prefix(comp)
            if len(prefix) < 2:
                continue
            for gene in all_genes:
                if gene in seen:
                    continue
                if self._tf_family_prefix(gene) == prefix:
                    if any(self.tf_tpm[gene].get(ct, 0) >= 1.0
                           for ct in self.cell_types):
                        paralogs.append(gene)
                        seen.add(gene)
        paralogs.sort(key=lambda g: max(self.tf_tpm.get(g, {}).get(ct, 0)
                                         for ct in self.cell_types), reverse=True)
        return paralogs[:max_paralogs]

    def _collect_motif_positions(self, seq_idx):
        """Merge motif hits across cell types into unique positions.

        Overlapping seqlet intervals are merged. TF superset is the union
        of top hits from every cell type, sorted by best p-value.
        """
        raw = []
        for ct in self.cell_types:
            for h in self.motif_hits[ct][seq_idx]:
                raw.append({
                    'start': h['start'], 'end': h['end'], 'ct': ct,
                    'top_hits': h.get('top_hits',
                                      [{'tf': h['tf'], 'pval': h['pval']}]),
                })
        raw.sort(key=lambda x: (x['start'], x['end']))
        merged = []
        for r in raw:
            if merged and r['start'] < merged[-1]['end']:
                merged[-1]['end'] = max(merged[-1]['end'], r['end'])
                merged[-1]['hits'].append(r)
            else:
                merged.append({'start': r['start'], 'end': r['end'],
                               'hits': [r]})
        results = []
        for m in merged:
            tf_best_pval = {}
            for r in m['hits']:
                for th in r['top_hits'][:3]:
                    tf = th['tf']
                    pv = th['pval']
                    if tf not in tf_best_pval or pv < tf_best_pval[tf]:
                        tf_best_pval[tf] = pv
            tf_names = sorted(tf_best_pval, key=tf_best_pval.get)
            results.append({
                'start': m['start'], 'end': m['end'],
                'mid': (m['start'] + m['end']) / 2,
                'tf_names': tf_names,
            })
        return results

    def expression_match(self, seq_idx=0, top_k=1, min_tpm=1.0):
        """Match motif hits to TF expression, preferring expressed TFs.

        For each merged motif position, picks the first top_k TFs with
        detectable expression. Falls back to best-p-value hit.
        """
        assert hasattr(self, 'motif_hits'), "Run annotate_motifs() first."
        assert hasattr(self, 'tf_tpm'), "Run load_expression() first."

        positions = self._collect_motif_positions(seq_idx)
        results = []
        for pos in positions:
            selected = []
            for tf in pos['tf_names']:
                if self._tf_is_expressed(tf, min_tpm):
                    selected.append(tf)
                    if len(selected) >= top_k:
                        break
            if not selected:
                selected = pos['tf_names'][:top_k]

            for tf in selected:
                components = self._parse_tf_components(tf)
                paralogs = self._find_paralogs(tf)
                genes = components + paralogs
                tpm = {}
                for g in genes:
                    tpm[g] = {ct: self.tf_tpm.get(g, {}).get(ct, 0.0)
                              for ct in self.cell_types}
                results.append({
                    'start': pos['start'], 'end': pos['end'],
                    'mid': pos['mid'],
                    'tf': tf, 'components': components,
                    'paralogs': paralogs, 'genes': genes, 'tpm': tpm,
                    'all_tfs': pos['tf_names'],
                })
        return results

    def plot_expression_match(self, seq_idx=0, top_k=1, min_tpm=1.0,
                              max_paralogs=4, figsize=(20, None),
                              annotation_style='bars'):
        """Attribution logos with expression annotations above the top row.

        annotation_style:
            'bars'    — horizontal TPM bars per gene (original).
            'heatmap' — mini heatmaps per motif: left = attribution
                        (n_ct x motif_width), right = expression
                        (n_ct x n_tfs). Shows positional grammar +
                        candidate TF expression side by side.
        """
        assert hasattr(self, 'motif_hits'), "Run annotate_motifs() first."
        assert hasattr(self, 'tf_tpm'), "Run load_expression() first."

        matches = self.expression_match(seq_idx, top_k=top_k, min_tpm=min_tpm)
        if not matches:
            print("No motif hits to plot.")
            return None, None

        for m in matches:
            m['paralogs'] = self._find_paralogs(m['tf'], max_paralogs)
            m['genes'] = m['components'] + m['paralogs']
            for g in m['genes']:
                if g not in m['tpm']:
                    m['tpm'][g] = {ct: self.tf_tpm.get(g, {}).get(ct, 0.0)
                                   for ct in self.cell_types}

        if annotation_style == 'heatmap':
            return self._plot_inline_heatmap(
                seq_idx, matches, value_key='tpm', unit='TPM', figsize=figsize)
        else:
            return self._plot_inline_bars(
                seq_idx, matches, value_key='tpm', unit='TPM',
                min_val=min_tpm, figsize=figsize)

    def _plot_inline_bars(self, seq_idx, matches, value_key='tpm',
                          unit='TPM', min_val=1.0, figsize=(20, None)):
        """Bar-style annotations above the top logo row.

        value_key: 'tpm' for ENCODE expression, 'protein' for CCLE proteomics.
        """
        from matplotlib.patches import Rectangle as Rect

        n_ct = len(self.cell_types)
        ct_colors = {'K562': '#E53935', 'HepG2': '#1E88E5', 'WTC11': '#43A047'}
        bar_order = sorted(self.cell_types,
                           key=lambda c: list(ct_colors.keys()).index(c)
                           if c in ct_colors else 99)

        all_vals = np.concatenate([
            self.attr[ct][seq_idx].ravel() for ct in self.cell_types
        ])
        yabs = max(abs(all_vals.min()), abs(all_vals.max())) * 1.05
        ylim = (-yabs, yabs)
        yrange = ylim[1] - ylim[0]

        all_vals_expr = []
        for m in matches:
            for g in m['genes']:
                for ct in self.cell_types:
                    all_vals_expr.append(m[value_key].get(g, {}).get(ct, 0.0))
        log_max = np.log1p(max(all_vals_expr)) if all_vals_expr else 1.0

        max_genes = max(len(m['genes']) for m in matches) if matches else 1
        BAR_MAX_W = 18
        ANNOT_FRAC = 0.15 + max_genes * 0.14

        top_h = 3.5 * (1 + ANNOT_FRAC)
        other_h = 3.5
        total_h = (figsize[1] if figsize[1] is not None
                   else top_h + other_h * (n_ct - 1))

        fig = plt.figure(figsize=(figsize[0], total_h))
        heights = [top_h] + [other_h] * (n_ct - 1)
        gs = gridspec.GridSpec(n_ct, 1, height_ratios=heights, hspace=0.3)

        axes = []
        for ci, ct in enumerate(self.cell_types):
            ax = fig.add_subplot(gs[ci])
            plot_logo(self.attr[ct][seq_idx], ax=ax, ylim=yabs)
            ax.axvspan(PROMOTER_START, BARCODE_START, alpha=0.10,
                       color=REGION_COLORS['promoter'])
            ax.axvspan(BARCODE_START, TOTAL_LEN, alpha=0.10,
                       color=REGION_COLORS['barcode'])

            parts = [ct, self.model_names[ct]]
            if ct in self.predictions:
                parts.append(f'pred={self.predictions[ct][seq_idx]:.3f}')
            if ct in self.actual and self.actual[ct] is not None:
                parts.append(f'actual={self.actual[ct][seq_idx]:.3f}')
            ax.set_title(' | '.join(parts), fontsize=10)

            for m in matches:
                ax.axvspan(m['start'], m['end'], alpha=0.08, color='#FF9800')

            if ci == 0:
                ax.set_ylim(ylim[0], ylim[1] + yrange * ANNOT_FRAC)
                row_h_data = yrange * ANNOT_FRAC / (max_genes + 1.5)
                bar_h = row_h_data * 0.35

                for m in matches:
                    start = m['start']
                    genes = m['genes']

                    for gi, gene in enumerate(genes):
                        y_base = ylim[1] + yrange * 0.04 + gi * row_h_data
                        is_component = gene in m['components']

                        for bj, ct_bar in enumerate(bar_order):
                            v = m[value_key].get(gene, {}).get(ct_bar, 0.0)
                            bar_w = (BAR_MAX_W * np.log1p(v) / log_max
                                     if log_max > 0 else 0)
                            bar_y = y_base + bj * bar_h
                            bar_color = ct_colors.get(ct_bar, f'C{bj}')
                            alpha = 0.85 if v >= min_val else 0.25
                            rect = Rect(
                                (start, bar_y), max(bar_w, 0.3), bar_h * 0.9,
                                facecolor=bar_color, edgecolor='none',
                                alpha=alpha, zorder=3)
                            ax.add_patch(rect)

                        label_x = start + BAR_MAX_W + 1.5
                        label_y = y_base + len(bar_order) * bar_h * 0.5
                        weight = 'bold' if is_component else 'normal'
                        style = 'normal' if is_component else 'italic'
                        val_strs = []
                        for ct_bar in bar_order:
                            v = m[value_key].get(gene, {}).get(ct_bar, 0.0)
                            val_strs.append(f'{v:.0f}' if v >= 1 else '<1')
                        label = f'{gene} ({"/".join(val_strs)})'
                        ax.text(label_x, label_y, label,
                                fontsize=5.5, va='center', ha='left',
                                fontweight=weight, fontstyle=style,
                                color='#333' if is_component else '#777',
                                zorder=4)

                handles = [Rect((0, 0), 1, 1,
                                fc=ct_colors.get(c, '#999'), alpha=0.85)
                           for c in bar_order]
                labels = [f'{c} {unit}' for c in bar_order]
                ax.legend(handles, labels, loc='upper right', fontsize=7,
                          framealpha=0.8)
            else:
                ax.set_ylim(*ylim)

            axes.append(ax)

        axes[-1].set_xlabel('Position (230bp enhancer | 36bp promoter | 15bp barcode)')
        fig.suptitle(f'Sequence {seq_idx} — Motif TF {unit} + Paralogs',
                     fontsize=12, y=1.01)
        plt.tight_layout()
        return fig, axes

    def _plot_inline_heatmap(self, seq_idx, matches, value_key='tpm',
                             unit='TPM', figsize=(20, None)):
        """Inline expression/protein heatmaps aligned to motif positions above logos.

        Each motif gets a small (n_ct x n_tfs) heatmap placed directly
        above the motif position on the top logo row. TF names above columns,
        cell-type labels as colored row labels on the first heatmap only.
        Single shared colorbar in the upper-right corner.

        value_key: 'tpm' for ENCODE, 'protein' for CCLE proteomics.
        unit: label for colorbar ('TPM' or 'log2 protein').
        """
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm

        n_ct = len(self.cell_types)
        ct_colors = {'K562': '#E53935', 'HepG2': '#1E88E5', 'WTC11': '#43A047'}
        ct_order = sorted(self.cell_types,
                          key=lambda c: list(ct_colors.keys()).index(c)
                          if c in ct_colors else 99)

        # Shared y-limits for logos
        all_vals = np.concatenate([
            self.attr[ct][seq_idx].ravel() for ct in self.cell_types
        ])
        yabs = max(abs(all_vals.min()), abs(all_vals.max())) * 1.05
        ylim = (-yabs, yabs)
        yrange = ylim[1] - ylim[0]

        # Collect all values for colormap scaling
        # For protein data (log2), use raw values; for TPM, use log1p
        is_log2 = (value_key == 'protein')
        all_raw = []
        for m in matches:
            for g in m['genes']:
                for ct in self.cell_types:
                    v = m[value_key].get(g, {}).get(ct, 0.0)
                    if not np.isnan(v):
                        all_raw.append(v)

        if is_log2:
            vmin = min(all_raw) if all_raw else -2
            vmax = max(all_raw) if all_raw else 2
            expr_norm = Normalize(vmin=vmin, vmax=vmax)
            expr_cmap = cm.RdYlBu_r  # diverging: blue=low, red=high
        else:
            expr_max = np.log1p(max(all_raw)) if all_raw else 1.0
            expr_norm = Normalize(vmin=0, vmax=expr_max)
            expr_cmap = cm.YlOrRd

        # Layout: heatmaps inline above first logo row (data coords)
        max_genes = max(len(m['genes']) for m in matches) if matches else 1
        ANNOT_FRAC = 0.15 + max_genes * 0.09 + 0.10

        top_h = 4.0 * (1 + ANNOT_FRAC)
        other_h = 3.5
        total_h = (figsize[1] if figsize[1] is not None
                   else top_h + other_h * (n_ct - 1))

        fig = plt.figure(figsize=(figsize[0], total_h))
        heights = [top_h] + [other_h] * (n_ct - 1)
        gs = gridspec.GridSpec(n_ct, 1, height_ratios=heights, hspace=0.3)

        axes = []
        for ci, ct in enumerate(self.cell_types):
            ax = fig.add_subplot(gs[ci])
            plot_logo(self.attr[ct][seq_idx], ax=ax, ylim=yabs)
            ax.axvspan(PROMOTER_START, BARCODE_START, alpha=0.10,
                       color=REGION_COLORS['promoter'])
            ax.axvspan(BARCODE_START, TOTAL_LEN, alpha=0.10,
                       color=REGION_COLORS['barcode'])

            parts = [ct, self.model_names[ct]]
            if ct in self.predictions:
                parts.append(f'pred={self.predictions[ct][seq_idx]:.3f}')
            if ct in self.actual and self.actual[ct] is not None:
                parts.append(f'actual={self.actual[ct][seq_idx]:.3f}')
            ax.set_title(' | '.join(parts), fontsize=10)

            for m in matches:
                ax.axvspan(m['start'], m['end'], alpha=0.08, color='#FF9800')

            if ci == 0:
                # --- Inline heatmaps above first logo (data coords) ---
                ax.set_ylim(ylim[0], ylim[1] + yrange * ANNOT_FRAC)

                annot_base = ylim[1] + yrange * 0.03
                row_h = yrange * 0.07          # height per TF row
                col_w = 8.0                    # wide so headers never overlap

                # Pre-compute grid x-positions, resolving overlaps
                grid_positions = []
                for m in matches:
                    mid = (m['start'] + m['end']) / 2
                    n_tfs = len(m['genes'])
                    if n_tfs == 0:
                        continue
                    # right edge = grid_x0 + n_ct*col_w + label_margin
                    label_margin = 18  # room for TF name text
                    total_w = n_ct * col_w + label_margin
                    desired_x0 = mid - (n_ct * col_w) / 2
                    # Push right if overlapping previous grid
                    if grid_positions:
                        prev_right = grid_positions[-1]['right']
                        if desired_x0 < prev_right + 2:
                            desired_x0 = prev_right + 2
                    grid_positions.append({
                        'match': m, 'x0': desired_x0,
                        'right': desired_x0 + total_w,
                    })

                for gp in grid_positions:
                    m = gp['match']
                    grid_x0 = gp['x0']
                    tfs = m['genes']
                    n_tfs = len(tfs)

                    # Cell-type column headers above grid — black text
                    header_y = annot_base + n_tfs * row_h + row_h * 0.15
                    for ci_col, ct_c in enumerate(ct_order):
                        ax.text(grid_x0 + (ci_col + 0.5) * col_w, header_y,
                                ct_c, fontsize=7.5, ha='center', va='bottom',
                                fontweight='demibold',
                                color='#111', zorder=4)

                    # TF rows
                    for ti, tf in enumerate(tfs):
                        is_comp = tf in m['components']
                        for ci_col, ct_c in enumerate(ct_order):
                            raw_val = m[value_key].get(tf, {}).get(ct_c, 0.0)
                            if np.isnan(raw_val):
                                color = '#e0e0e0'
                            elif is_log2:
                                color = expr_cmap(expr_norm(raw_val))
                            else:
                                color = expr_cmap(expr_norm(np.log1p(raw_val)))
                            x0 = grid_x0 + ci_col * col_w
                            y0 = annot_base + ti * row_h
                            rect = plt.Rectangle(
                                (x0, y0), col_w * 0.92, row_h * 0.92,
                                facecolor=color, edgecolor='#999',
                                linewidth=0.5, zorder=3)
                            ax.add_patch(rect)

                        # TF name to the right of the row
                        label_x = grid_x0 + n_ct * col_w + 0.8
                        label_y = annot_base + ti * row_h + row_h * 0.46
                        tf_disp = tf if len(tf) <= 12 else tf[:11] + '..'
                        ax.text(label_x, label_y, tf_disp,
                                fontsize=7, ha='left', va='center',
                                fontweight='medium' if is_comp else 'normal',
                                fontstyle='normal' if is_comp else 'italic',
                                color='#222' if is_comp else '#888',
                                clip_on=True, zorder=4)

                # --- Colorbar with unit label (upper-right) ---
                cbar_x = TOTAL_LEN - 18
                cbar_w = 12
                cbar_h = row_h * 0.5
                cbar_y = annot_base + row_h * 0.2
                n_steps = 30
                for si in range(n_steps):
                    frac = si / n_steps
                    clr = expr_cmap(frac)
                    rect = plt.Rectangle(
                        (cbar_x + si * cbar_w / n_steps, cbar_y),
                        cbar_w / n_steps, cbar_h,
                        facecolor=clr, edgecolor='none', zorder=5)
                    ax.add_patch(rect)
                # Tick marks on colorbar (bottom edge)
                if is_log2:
                    tick_vals = [vmin, (vmin + vmax) / 2, vmax]
                    tick_labels = [f'{v:.1f}' for v in tick_vals]
                    tick_fracs = [(v - vmin) / (vmax - vmin) if vmax != vmin else 0.5
                                  for v in tick_vals]
                else:
                    raw_max = np.expm1(expr_max) if expr_max > 0 else 1
                    tick_raw = [0, raw_max / 2, raw_max]
                    tick_labels = [f'{v:.0f}' for v in tick_raw]
                    tick_fracs = [np.log1p(v) / expr_max if expr_max > 0 else 0
                                  for v in tick_raw]
                for frac, label in zip(tick_fracs, tick_labels):
                    tx = cbar_x + frac * cbar_w
                    # Tick line
                    ax.plot([tx, tx], [cbar_y - row_h * 0.06, cbar_y],
                            color='#333', linewidth=0.8, zorder=6)
                    ax.text(tx, cbar_y - row_h * 0.08, label,
                            fontsize=5.5, ha='center', va='top',
                            color='#333', zorder=6)
                # Unit label centered above colorbar
                ax.text(cbar_x + cbar_w / 2, cbar_y + cbar_h + row_h * 0.1,
                        unit, fontsize=7, ha='center', va='bottom',
                        color='#444', zorder=5)
                # Grey = not detected legend
                if is_log2:
                    nd_x = cbar_x + cbar_w + 2
                    rect = plt.Rectangle(
                        (nd_x, cbar_y), cbar_h, cbar_h,
                        facecolor='#e0e0e0', edgecolor='#aaa',
                        linewidth=0.4, zorder=5)
                    ax.add_patch(rect)
                    ax.text(nd_x + cbar_h + 0.8, cbar_y + cbar_h * 0.5,
                            'n.d.', fontsize=6.5, va='center', ha='left',
                            color='#666', zorder=5)
            else:
                ax.set_ylim(*ylim)

            axes.append(ax)

        axes[-1].set_xlabel(
            'Position (230bp enhancer | 36bp promoter | 15bp barcode)')
        fig.suptitle(
            f'Sequence {seq_idx} — Motif {unit} Heatmaps',
            fontsize=12, y=1.01)
        plt.tight_layout()
        return fig, axes

    # ----- ChIP-Atlas annotation -----
    def load_genomic_coords(self, joint_df=None, seq_col='sequence'):
        """Match loaded sequences back to joint_df to get genomic coords.

        joint_df: DataFrame with columns chr_hg38, start_hg38, stop_hg38, sequence.
                  If None, loads from JOINT_LIBRARY_PATH.
        Populates self.coords: DataFrame with chr, start, stop per loaded sequence.
        """
        if joint_df is None:
            print(f"  Loading joint library from {JOINT_LIBRARY_PATH}")
            joint_df = pd.read_csv(JOINT_LIBRARY_PATH)

        # Build lookup: enhancer sequence -> (chr, start, stop)
        # joint_library chr_hg38 may lack 'chr' prefix (e.g. '10' vs 'chr10');
        # ChIP-Atlas BED uses 'chr10' format — normalize here.
        lookup = {}
        for _, row in joint_df.iterrows():
            seq = str(row[seq_col])
            if pd.isna(row['start_hg38']) or pd.isna(row['stop_hg38']):
                continue
            chrom = str(row['chr_hg38']).strip()
            if not chrom.startswith('chr'):
                chrom = 'chr' + chrom
            lookup[seq] = (chrom, int(float(row['start_hg38'])),
                           int(float(row['stop_hg38'])))

        coords = []
        missing = 0
        for i, enh in enumerate(self.enhancers):
            if enh in lookup:
                c, s, e = lookup[enh]
                coords.append({'seq_idx': i, 'chr': c, 'start': int(s), 'stop': int(e)})
            else:
                missing += 1
                coords.append({'seq_idx': i, 'chr': None, 'start': None, 'stop': None})

        self.coords = pd.DataFrame(coords)
        n_mapped = len(self.coords) - missing
        print(f"  Mapped {n_mapped}/{len(self.enhancers)} sequences to genomic coords"
              + (f" ({missing} missing)" if missing else ""))
        return self

    def _download_experiment_list(self):
        """Download and parse ChIP-Atlas experimentList.tab (cached)."""
        os.makedirs(CHIPATLAS_CACHE, exist_ok=True)
        dest = os.path.join(CHIPATLAS_CACHE, 'experimentList.tab')
        if not os.path.exists(dest):
            print("  Downloading ChIP-Atlas experiment list (one-time)...")
            _download_file(CHIPATLAS_EXPLIST_URL, dest)
            print(f"  Cached -> {dest}")

        cols = ['srx_id', 'genome', 'track_class', 'antigen',
                'cell_class', 'cell_type', 'metadata', 'stats', 'title']
        df = pd.read_csv(dest, sep='\t', header=None, names=cols,
                         usecols=range(9), dtype=str, on_bad_lines='skip')
        df = df[df['genome'] == 'hg38'].copy()
        return df

    def _find_experiments(self, explist_df, cell_type,
                          antigens=None, exclude_antigens=None,
                          best_per_antigen=True):
        """Filter experimentList for a cell type. Returns list of (srx_id, antigen).

        best_per_antigen: If True (default), keep only the experiment with the
            most peaks per antigen — avoids downloading redundant replicates
            (e.g. 84 CTCF experiments in K562 -> 1).
        """
        cell_names = CHIPATLAS_CELL_MAP.get(cell_type, [cell_type])
        mask = explist_df['cell_type'].isin(cell_names)

        # WTC11 special case: "iPS cells" is broad, filter by title
        if cell_type == 'WTC11':
            mask = mask & explist_df['title'].str.contains('WTC11', case=False, na=False)

        # Only TFs and others (ChIP-seq for proteins)
        mask = mask & (explist_df['track_class'] == 'TFs and others')

        sub = explist_df[mask].copy()

        if antigens is not None:
            antigens_upper = {a.upper() for a in antigens}
            sub = sub[sub['antigen'].str.upper().isin(antigens_upper)]
        if exclude_antigens is not None:
            exclude_upper = {a.upper() for a in exclude_antigens}
            sub = sub[~sub['antigen'].str.upper().isin(exclude_upper)]

        if best_per_antigen and len(sub) > 0:
            # stats col format: "reads,mapped%,dup%,peak_count"
            def _peak_count(s):
                try:
                    return int(str(s).split(',')[-1])
                except (ValueError, IndexError):
                    return 0
            sub = sub.copy()
            sub['_npeaks'] = sub['stats'].apply(_peak_count)
            sub = (sub.sort_values('_npeaks', ascending=False)
                     .drop_duplicates(subset='antigen', keep='first')
                     .drop(columns='_npeaks'))

        results = list(zip(sub['srx_id'], sub['antigen']))
        return results

    @staticmethod
    def _download_one_peak(args):
        """Download a single peak BED file (for use with ThreadPoolExecutor)."""
        srx_id, threshold, peaks_dir = args
        fname = f'{srx_id}.{threshold}.bed'
        dest = os.path.join(peaks_dir, fname)
        if os.path.exists(dest):
            return srx_id, dest
        url = f'{CHIPATLAS_DATA}/hg38/eachData/bed{threshold}/{fname}'
        try:
            _download_file(url, dest)
            return srx_id, dest
        except Exception:
            if os.path.exists(dest):
                os.unlink(dest)
            return srx_id, None

    def _download_peaks_parallel(self, srx_ids, threshold='05', n_workers=16,
                                  verbose=True):
        """Download peak BEDs in parallel. Returns {srx_id: path_or_None}."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        peaks_dir = os.path.join(CHIPATLAS_CACHE, 'peaks')
        os.makedirs(peaks_dir, exist_ok=True)

        # Split into cached vs need-download
        results = {}
        to_download = []
        for srx_id in srx_ids:
            fname = f'{srx_id}.{threshold}.bed'
            dest = os.path.join(peaks_dir, fname)
            if os.path.exists(dest):
                results[srx_id] = dest
            else:
                to_download.append((srx_id, threshold, peaks_dir))

        if verbose and to_download:
            print(f"    {len(results)} cached, downloading {len(to_download)}...")

        if to_download:
            done = 0
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(self._download_one_peak, args): args[0]
                           for args in to_download}
                for future in as_completed(futures):
                    srx_id, path = future.result()
                    results[srx_id] = path
                    done += 1
                    if verbose and done % 100 == 0:
                        print(f"    Downloaded {done}/{len(to_download)}...")
            if verbose:
                n_ok = sum(1 for v in results.values() if v is not None)
                print(f"    Done: {n_ok} peak files available")

        return results

    @staticmethod
    def _parse_peaks(bed_path):
        """Parse a peak BED file into a DataFrame for fast overlap."""
        try:
            df = pd.read_csv(bed_path, sep='\t', header=None,
                             usecols=[0, 1, 2, 4],
                             dtype={0: str, 1: int, 2: int, 4: int},
                             names=['chr', 'start', 'end', 'score'])
            return df
        except Exception:
            return None

    @staticmethod
    def _overlap_peaks_numpy(peak_chr, peak_start, peak_end, peak_score,
                              coord_chr, coord_start, coord_stop):
        """Vectorized overlap: peaks vs all coords on matching chromosomes.

        Returns list of (seq_idx, best_score) for coords that overlap.
        """
        results = []
        # Group coords by chromosome for fast filtering
        for chrom in set(coord_chr):
            c_mask = np.array([c == chrom for c in coord_chr])
            p_mask = (peak_chr == chrom)
            if not p_mask.any() or not c_mask.any():
                continue
            p_starts = peak_start[p_mask]
            p_ends = peak_end[p_mask]
            p_scores = peak_score[p_mask]
            c_idxs = np.where(c_mask)[0]
            for ci in c_idxs:
                cs, ce = coord_start[ci], coord_stop[ci]
                hit_mask = (p_starts < ce) & (p_ends > cs)
                if hit_mask.any():
                    results.append((ci, int(p_scores[hit_mask].max())))
        return results

    def query_chipatlas(self, threshold='05', antigens=None,
                        exclude_antigens=None, exclude_tomtom=False,
                        max_experiments=None, n_workers=16, verbose=True):
        """Query ChIP-Atlas for all factors with peaks at loaded sequences' loci.

        Args:
            threshold: Peak stringency ('05'=top 5k, '10', '20', '50').
            antigens: Only these antigens (None = all).
            exclude_antigens: Remove these antigens.
            exclude_tomtom: Auto-exclude TFs already found by annotate_motifs().
            max_experiments: Limit experiments per cell type (for speed).
            n_workers: Parallel download threads (default 16).
            verbose: Print progress.

        Populates:
            self.chipatlas_hits[ct]: DataFrame [antigen, srx_id, seq_idx, overlap, peak_score]
            self.chipatlas_summary[ct]: DataFrame [antigen, n_experiments, n_overlaps, frac_seqs, best_score]
        """
        assert hasattr(self, 'coords'), "Call load_genomic_coords() first."

        # Build exclude list from TOMTOM
        if exclude_tomtom and hasattr(self, 'motif_hits'):
            tomtom_tfs = set()
            for ct in self.cell_types:
                for seq_hits in self.motif_hits[ct]:
                    for h in seq_hits:
                        tomtom_tfs.add(h['tf'].upper())
                        for th in h.get('top_hits', []):
                            tomtom_tfs.add(th['tf'].upper())
            if exclude_antigens:
                exclude_antigens = list(set(a.upper() for a in exclude_antigens) | tomtom_tfs)
            else:
                exclude_antigens = list(tomtom_tfs)
            if verbose:
                print(f"  Excluding {len(exclude_antigens)} TOMTOM-found TFs")

        explist = self._download_experiment_list()
        self.chipatlas_hits = {}
        self.chipatlas_summary = {}

        # Pre-compute coord arrays for vectorized overlap
        valid = self.coords[self.coords['chr'].notna()].copy()
        coord_chr = valid['chr'].values.astype(str)
        coord_start = valid['start'].values.astype(int)
        coord_stop = valid['stop'].values.astype(int)
        coord_seq_idx = valid['seq_idx'].values.astype(int)

        for ct in self.cell_types:
            experiments = self._find_experiments(
                explist, ct, antigens=antigens,
                exclude_antigens=exclude_antigens)
            if max_experiments:
                experiments = experiments[:max_experiments]
            if verbose:
                print(f"  {ct}: {len(experiments)} experiments")

            # Group by antigen
            from collections import defaultdict
            antigen_exps = defaultdict(list)
            all_srx = set()
            for srx_id, antigen in experiments:
                antigen_exps[antigen].append(srx_id)
                all_srx.add(srx_id)

            # Parallel download all peak files
            peak_paths = self._download_peaks_parallel(
                list(all_srx), threshold, n_workers=n_workers, verbose=verbose)

            # Compute overlaps
            rows = []
            n_processed = 0
            for antigen, srx_ids in antigen_exps.items():
                for srx_id in srx_ids:
                    bed_path = peak_paths.get(srx_id)
                    if bed_path is None:
                        continue
                    peaks = self._parse_peaks(bed_path)
                    if peaks is None or len(peaks) == 0:
                        continue

                    n_processed += 1
                    if verbose and n_processed % 200 == 0:
                        print(f"    Processed {n_processed} experiments...")

                    p_chr = peaks['chr'].values
                    p_start = peaks['start'].values
                    p_end = peaks['end'].values
                    p_score = peaks['score'].values

                    hits = self._overlap_peaks_numpy(
                        p_chr, p_start, p_end, p_score,
                        coord_chr, coord_start, coord_stop)
                    for ci, score in hits:
                        rows.append({
                            'antigen': antigen,
                            'srx_id': srx_id,
                            'seq_idx': int(coord_seq_idx[ci]),
                            'overlap': True,
                            'peak_score': score,
                        })

            hits_df = pd.DataFrame(rows) if rows else pd.DataFrame(
                columns=['antigen', 'srx_id', 'seq_idx', 'overlap', 'peak_score'])
            self.chipatlas_hits[ct] = hits_df

            # Summary: per antigen
            if len(hits_df) > 0:
                n_seqs = len(valid)
                summary = (hits_df.groupby('antigen')
                           .agg(
                               n_experiments=('srx_id', 'nunique'),
                               n_overlaps=('seq_idx', 'nunique'),
                               best_score=('peak_score', 'max'),
                           )
                           .reset_index())
                summary['frac_seqs'] = summary['n_overlaps'] / max(n_seqs, 1)
                summary = summary.sort_values('n_overlaps', ascending=False)
            else:
                summary = pd.DataFrame(
                    columns=['antigen', 'n_experiments', 'n_overlaps',
                             'frac_seqs', 'best_score'])
            self.chipatlas_summary[ct] = summary
            if verbose:
                print(f"    {ct}: {len(summary)} antigens with overlaps, "
                      f"{len(hits_df)} total hit records")

        return self

    def show_chipatlas(self, seq_idx=None, min_experiments=1,
                       sort_by='n_overlaps', antigens=None,
                       exclude_antigens=None, exclude_tomtom=False):
        """Print ChIP-Atlas factors at loaded loci.

        Args:
            seq_idx: If given, show only factors overlapping this sequence.
                     If None, show summary across all sequences.
            min_experiments: Minimum number of experiments with peaks.
            sort_by: Column to sort by ('n_overlaps', 'best_score', 'n_experiments').
            antigens/exclude_antigens: Filter display.
            exclude_tomtom: Hide TOMTOM-found TFs.
        """
        assert hasattr(self, 'chipatlas_summary'), "Run query_chipatlas() first."

        tomtom_tfs = set()
        if exclude_tomtom and hasattr(self, 'motif_hits'):
            for ct in self.cell_types:
                for seq_hits in self.motif_hits[ct]:
                    for h in seq_hits:
                        tomtom_tfs.add(h['tf'].upper())
                        for th in h.get('top_hits', []):
                            tomtom_tfs.add(th['tf'].upper())

        for ct in self.cell_types:
            if seq_idx is not None:
                # Filter hits for this sequence only
                hits = self.chipatlas_hits[ct]
                hits = hits[hits['seq_idx'] == seq_idx]
                if len(hits) == 0:
                    print(f"  {ct}: no ChIP-Atlas hits for seq {seq_idx}")
                    continue
                summary = (hits.groupby('antigen')
                           .agg(n_experiments=('srx_id', 'nunique'),
                                best_score=('peak_score', 'max'))
                           .reset_index())
                summary['n_overlaps'] = 1
            else:
                summary = self.chipatlas_summary[ct].copy()

            # Apply filters
            if min_experiments > 1:
                summary = summary[summary['n_experiments'] >= min_experiments]
            if antigens:
                ag_upper = {a.upper() for a in antigens}
                summary = summary[summary['antigen'].str.upper().isin(ag_upper)]
            if exclude_antigens:
                ex_upper = {a.upper() for a in exclude_antigens}
                summary = summary[~summary['antigen'].str.upper().isin(ex_upper)]
            if exclude_tomtom and tomtom_tfs:
                summary = summary[~summary['antigen'].str.upper().isin(tomtom_tfs)]

            summary = summary.sort_values(sort_by, ascending=False)

            label = f"seq {seq_idx}" if seq_idx is not None else "all seqs"
            print(f"\n  {ct} ({label}): {len(summary)} factors")
            print(f"  {'Antigen':<20s} {'#Exps':>6s} {'#Seqs':>6s} {'BestScore':>10s}")
            print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*10}")
            for _, row in summary.head(50).iterrows():
                n_ov = row.get('n_overlaps', '-')
                print(f"  {row['antigen']:<20s} {row['n_experiments']:>6d} "
                      f"{str(n_ov):>6s} {row['best_score']:>10d}")

    def plot_chipatlas_heatmap(self, seq_idx=None, top_k=20,
                                exclude_tomtom=False, figsize=(10, 8)):
        """Heatmap: rows=factors, columns=cell types, color=best peak score.

        Args:
            seq_idx: If given, only peaks at this sequence. If None, all sequences.
            top_k: Show top K factors (ranked by total overlaps across cell types).
            exclude_tomtom: Exclude TOMTOM-found DNA-binding TFs.
        """
        assert hasattr(self, 'chipatlas_hits'), "Run query_chipatlas() first."

        tomtom_tfs = set()
        if exclude_tomtom and hasattr(self, 'motif_hits'):
            for ct in self.cell_types:
                for seq_hits in self.motif_hits[ct]:
                    for h in seq_hits:
                        tomtom_tfs.add(h['tf'].upper())
                        for th in h.get('top_hits', []):
                            tomtom_tfs.add(th['tf'].upper())

        # Collect best score per antigen per cell type
        antigen_scores = {}
        antigen_total = {}

        for ct in self.cell_types:
            hits = self.chipatlas_hits[ct]
            if seq_idx is not None:
                hits = hits[hits['seq_idx'] == seq_idx]
            for antigen, grp in hits.groupby('antigen'):
                if exclude_tomtom and antigen.upper() in tomtom_tfs:
                    continue
                if antigen not in antigen_scores:
                    antigen_scores[antigen] = {}
                    antigen_total[antigen] = 0
                antigen_scores[antigen][ct] = int(grp['peak_score'].max())
                antigen_total[antigen] += len(grp)

        if not antigen_scores:
            print("No ChIP-Atlas hits to plot.")
            return None, None

        ranked = sorted(antigen_total, key=antigen_total.get, reverse=True)[:top_k]

        matrix = np.zeros((len(ranked), len(self.cell_types)))
        for ri, antigen in enumerate(ranked):
            for ci, ct in enumerate(self.cell_types):
                matrix[ri, ci] = antigen_scores.get(antigen, {}).get(ct, 0)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd',
                        interpolation='nearest')
        ax.set_xticks(range(len(self.cell_types)))
        ax.set_xticklabels(self.cell_types, fontsize=10)
        ax.set_yticks(range(len(ranked)))
        ax.set_yticklabels(ranked, fontsize=9)

        for ri in range(len(ranked)):
            for ci in range(len(self.cell_types)):
                val = int(matrix[ri, ci])
                if val > 0:
                    ax.text(ci, ri, str(val), ha='center', va='center',
                            fontsize=7,
                            color='black' if val < matrix.max() * 0.7 else 'white')

        plt.colorbar(im, ax=ax, label='Peak score', shrink=0.8)
        title = f'ChIP-Atlas peak scores — top {len(ranked)} factors'
        if seq_idx is not None:
            title += f' (seq {seq_idx})'
        if exclude_tomtom:
            title += ' (excl. TOMTOM TFs)'
        ax.set_title(title, fontsize=12)
        plt.tight_layout()
        return fig, ax

    def chipatlas_at_motifs(self, seq_idx=0, top_coactivators=3,
                            max_paralogs=0):
        """Build per-motif annotation: TOMTOM TFs + top coactivators from ChIP-Atlas.

        For each motif position, returns the TOMTOM-identified DNA-binding TF(s)
        plus the top_coactivators factors (by peak score) from ChIP-Atlas that
        are NOT in the TOMTOM hit list — i.e. likely coactivators/corepressors.

        Composite motifs like AHR::ARNT are split into individual components
        (AHR, ARNT), each shown as a separate row with its own ChIP-Atlas score.

        max_paralogs: number of additional TOMTOM candidate TFs to show per motif
            (from top_hits). Useful when the top TOMTOM hit isn't biologically
            accurate — paralogs give alternative DNA-binding TF candidates.

        Returns list of match dicts compatible with _plot_inline_heatmap:
            {start, end, mid, genes, components, peak_score: {gene: {ct: score}}}
        """
        assert hasattr(self, 'motif_hits'), "Run annotate_motifs() first."
        assert hasattr(self, 'chipatlas_hits'), "Run query_chipatlas() first."

        # Collect all TOMTOM TF names AND their individual components (upper)
        tomtom_tfs = set()
        for ct in self.cell_types:
            for seq_hits in self.motif_hits[ct]:
                for h in seq_hits:
                    tomtom_tfs.add(h['tf'].upper())
                    for comp in self._parse_tf_components(h['tf']):
                        tomtom_tfs.add(comp)
                    for th in h.get('top_hits', []):
                        tomtom_tfs.add(th['tf'].upper())
                        for comp in self._parse_tf_components(th['tf']):
                            tomtom_tfs.add(comp)

        # Get ChIP-Atlas hits for this sequence, per cell type
        # Build: antigen -> {ct: best_peak_score}
        antigen_scores = {}
        for ct in self.cell_types:
            hits = self.chipatlas_hits[ct]
            seq_hits = hits[hits['seq_idx'] == seq_idx]
            for _, row in seq_hits.iterrows():
                ag = row['antigen']
                if ag not in antigen_scores:
                    antigen_scores[ag] = {}
                prev = antigen_scores[ag].get(ct, 0)
                antigen_scores[ag][ct] = max(prev, row['peak_score'])

        # Split into DNA-binding (in TOMTOM) vs coactivators (not in TOMTOM)
        coactivators = {ag: scores for ag, scores in antigen_scores.items()
                        if ag.upper() not in tomtom_tfs}
        # Rank coactivators by max peak score across cell types
        coact_ranked = sorted(coactivators.keys(),
                              key=lambda ag: max(coactivators[ag].values()),
                              reverse=True)

        # Case-insensitive lookup for antigen scores
        ag_upper_map = {}  # UPPER -> original key in antigen_scores
        for ag in antigen_scores:
            ag_upper_map[ag.upper()] = ag

        # Debug: show what ChIP-Atlas factors overlap this sequence
        if antigen_scores:
            print(f"  ChIP-Atlas: {len(antigen_scores)} factors at seq {seq_idx} locus")
            top5 = sorted(antigen_scores.items(),
                          key=lambda x: max(x[1].values()), reverse=True)[:5]
            for ag, sc in top5:
                sc_str = ', '.join(f'{ct}:{s}' for ct, s in sc.items())
                is_tf = '(DNA-binding)' if ag.upper() in tomtom_tfs else ''
                print(f"    {ag}: {sc_str} {is_tf}")
        else:
            print(f"  ChIP-Atlas: NO factors overlap seq {seq_idx} locus")

        # All antigens that have peaks at this locus (upper)
        all_antigens_upper = set(ag_upper_map.keys())

        # Build matches per motif position
        positions = self._collect_motif_positions(seq_idx)
        matches = []
        for pos in positions:
            tf_name = pos['tf_names'][0] if pos['tf_names'] else 'unknown'

            # Split composite motifs: AHR::ARNT -> [AHR, ARNT]
            components = self._parse_tf_components(tf_name)

            # Add paralog TFs from TOMTOM top_hits (alternative candidates)
            seen = set(c.upper() for c in components)
            if max_paralogs > 0:
                for alt_tf in pos['tf_names'][1:]:
                    if len(components) >= len(self._parse_tf_components(tf_name)) + max_paralogs:
                        break
                    for comp in self._parse_tf_components(alt_tf):
                        if comp not in seen:
                            components.append(comp)
                            seen.add(comp)

            # Expand each DNA-binding component to its TF family from ChIP-Atlas
            # e.g. GATA1 -> [GATA1, GATA2, GATA3] if GATA2/3 have peaks here
            family_components = []
            family_seen = set()
            for comp in components:
                family_components.append(comp)
                family_seen.add(comp.upper())
                prefix = self._tf_family_prefix(comp)
                if len(prefix) >= 2:
                    for ag_up in sorted(all_antigens_upper):
                        if ag_up in family_seen:
                            continue
                        if self._tf_family_prefix(ag_up) == prefix:
                            family_components.append(ag_upper_map[ag_up])
                            family_seen.add(ag_up)

            genes = list(family_components)  # all shown bold as DNA-binding
            components = list(family_components)  # all are DNA-binding family

            # Add top coactivators (exclude anything in the DNA-binding family)
            top_coact = [ag for ag in coact_ranked
                         if ag.upper() not in family_seen][:top_coactivators]
            genes.extend(top_coact)

            # Build peak_score dict: {gene: {ct: score}}
            peak_score = {}
            for g in genes:
                peak_score[g] = {}
                real_key = ag_upper_map.get(g.upper())
                for ct in self.cell_types:
                    if real_key and real_key in antigen_scores:
                        peak_score[g][ct] = antigen_scores[real_key].get(ct, 0)
                    else:
                        peak_score[g][ct] = 0

            matches.append({
                'start': pos['start'], 'end': pos['end'],
                'mid': pos['mid'],
                'tf': tf_name,
                'components': components,
                'genes': genes,
                'peak_score': peak_score,
                'all_tfs': pos['tf_names'],
            })

        return matches

    # Well-known coactivators/corepressors to always include in targeted queries
    COMMON_COACTIVATORS = [
        'EP300', 'CREBBP', 'BRD4', 'MED1', 'MED12', 'MED26',
        'NCOA1', 'NCOA2', 'NCOA3', 'HDAC1', 'HDAC2', 'KDM1A',
        'SMARCA4', 'SMARCC1', 'CHD4', 'NSD2', 'KAT2B', 'SETD1A',
    ]

    def plot_chipatlas_at_motifs(self, seq_idx=0, top_coactivators=3,
                                 max_paralogs=0, coactivator_list=None,
                                 threshold='05', figsize=(20, None)):
        """Attribution logos with ChIP-Atlas peak score heatmaps at motif sites.

        Above each motif: TOMTOM TF family (bold) + top coactivators (italic),
        with ChIP-Atlas peak scores per cell type as colored cells.

        Only downloads peak files for relevant antigens (motif TFs + coactivators),
        not the full ~600 per cell type. Auto-calls query_chipatlas() with a
        targeted antigen list if not already cached.

        Args:
            max_paralogs: additional TOMTOM candidate TFs per motif (shown bold).
            coactivator_list: explicit list of coactivators to query. If None,
                uses COMMON_COACTIVATORS.
            threshold: ChIP-Atlas peak stringency ('05', '10', '20', '50').

        Usage:
            em.annotate_motifs()
            em.load_genomic_coords()
            em.plot_chipatlas_at_motifs(seq_idx=0, max_paralogs=2)
        """
        assert hasattr(self, 'motif_hits'), "Run annotate_motifs() first."
        assert hasattr(self, 'coords'), "Run load_genomic_coords() first."

        # Build targeted antigen list from TOMTOM hits + coactivators
        if not hasattr(self, 'chipatlas_hits') or not self.chipatlas_hits:
            motif_tfs = set()
            for ct in self.cell_types:
                for seq_hits in self.motif_hits[ct]:
                    for h in seq_hits:
                        for comp in self._parse_tf_components(h['tf']):
                            motif_tfs.add(comp)
                            # Add family prefix matches
                            prefix = self._tf_family_prefix(comp)
                            if len(prefix) >= 2:
                                motif_tfs.add(comp)
                        for th in h.get('top_hits', []):
                            for comp in self._parse_tf_components(th['tf']):
                                motif_tfs.add(comp)

            # Expand motif TFs to include family members from experimentList
            explist = self._download_experiment_list()
            all_antigens_in_db = set(explist['antigen'].dropna().str.upper().unique())
            family_expanded = set()
            for tf in motif_tfs:
                family_expanded.add(tf)
                prefix = self._tf_family_prefix(tf)
                if len(prefix) >= 2:
                    for ag in all_antigens_in_db:
                        if self._tf_family_prefix(ag) == prefix:
                            family_expanded.add(ag)

            coact = coactivator_list or self.COMMON_COACTIVATORS
            target_antigens = list(family_expanded | set(c.upper() for c in coact))
            print(f"  Targeted query: {len(motif_tfs)} motif TFs -> "
                  f"{len(family_expanded)} with families + "
                  f"{len(coact)} coactivators = {len(target_antigens)} antigens")
            self.query_chipatlas(threshold=threshold, antigens=target_antigens,
                                verbose=True)

        matches = self.chipatlas_at_motifs(seq_idx, top_coactivators,
                                           max_paralogs=max_paralogs)
        if not matches:
            print("No motif hits to annotate with ChIP-Atlas data.")
            return None, None

        return self._plot_inline_heatmap(
            seq_idx, matches, value_key='peak_score',
            unit='ChIP peak score', figsize=figsize)
