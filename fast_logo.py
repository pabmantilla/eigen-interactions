"""Batch-generate DNA attribution logo PNGs from an HDF5 file.

Core primitive
--------------
``fast_logo(values, ax)`` renders one attribution logo onto a Matplotlib axis.
It accepts an ``(L, 4)`` array (columns A/C/G/T), draws each base letter
scaled to its attribution magnitude (positive glyphs stack upward, negative
ones flip and stack downward), and returns immediately.  Glyph geometry is
computed once and cached globally so repeated calls are cheap.

DeepSTARR wrapper
-----------------
The remainder of the script applies ``fast_logo`` to a concrete use-case:
plotting attribution scores for DeepSTARR sequences that were stored in HDF5
after running an attribution method (e.g. tangermeme's ``deep_lift_shap``,
gradient × input, or any other method that produces per-nucleotide scores).
Attribution arrays may be ``(N, L, 4)`` or ``(N, 4, L)``; both layouts are
accepted.  When annotation arrays ``start_dev``/``end_dev``/``start_hk``/
``end_hk`` are present, a highlight band is drawn over the motif region.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath


# ---------------------------------------------------------------------------
# DNA color palette
# ---------------------------------------------------------------------------

_DNA_COLORS: dict[str, tuple[float, float, float]] = {
    "A": (0.0, 0.5, 0.0),
    "C": (0.0, 0.0, 1.0),
    "G": (1.0, 0.65, 0.0),
    "T": (1.0, 0.0, 0.0),
}


# ---------------------------------------------------------------------------
# Glyph geometry cache
# ---------------------------------------------------------------------------

class _GlyphCache:
    """Lazily populated cache of pre-computed glyph geometry for A/C/G/T."""

    def __init__(self) -> None:
        self.verts: dict[str, np.ndarray] = {}
        self.codes: dict[str, np.ndarray] = {}
        self.xmin: dict[str, float] = {}
        self.ymin: dict[str, float] = {}
        self.w: dict[str, float] = {}
        self.h: dict[str, float] = {}
        # Pre-flipped geometry (for negative attributions drawn upside-down)
        self.flip_verts: dict[str, np.ndarray] = {}
        self.flip_ymin: dict[str, float] = {}
        self.flip_h: dict[str, float] = {}
        self.ref_w: float = 0.0
        self.ready: bool = False

    def build(
        self,
        font_name: str = "sans",
        font_weight: str = "bold",
        ref_char: str = "E",
    ) -> None:
        """Compute and cache glyph geometry (called once on first use)."""
        if self.ready:
            return
        fp = fm.FontProperties(family=font_name, weight=font_weight)
        for ch in "ACGT":
            tp = TextPath((0, 0), ch, size=1, prop=fp)
            ext = tp.get_extents()
            v = np.array(tp.vertices, dtype=np.float64)
            self.verts[ch] = v
            self.codes[ch] = np.array(tp.codes, dtype=np.uint8)
            self.xmin[ch] = float(ext.xmin)
            self.ymin[ch] = float(ext.ymin)
            self.w[ch] = float(ext.width)
            self.h[ch] = float(ext.height)
            fv = v.copy()
            fv[:, 1] = -fv[:, 1]
            self.flip_verts[ch] = fv
            self.flip_ymin[ch] = float(fv[:, 1].min())
            self.flip_h[ch] = float(fv[:, 1].max()) - self.flip_ymin[ch]
        self.ref_w = TextPath((0, 0), ref_char, size=1, prop=fp).get_extents().width
        self.ready = True


_CACHE = _GlyphCache()


# ---------------------------------------------------------------------------
# Core: fast_logo
# ---------------------------------------------------------------------------

def fast_logo(
    values: np.ndarray,
    ax: Axes,
    width: float = 0.95,
    height_scale: float = 1.0,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Render a single attribution logo on a Matplotlib axis.

    Parameters
    ----------
    values:
        Attribution matrix of shape ``(L, 4)`` with columns ordered A/C/G/T.
        Positive values stack upward; negative values are drawn flipped
        (upside-down) and stack downward.
    ax:
        Axis to draw onto.
    width:
        Horizontal glyph width as a fraction of one position unit.
    height_scale:
        Scalar multiplier applied to all attribution values before drawing.
    ylim:
        Fixed y-axis limits.  When omitted the limits are inferred from the
        data with a small 5 % padding.
    """
    _CACHE.build()
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Expected values shape (L, 4), got {values.shape}")

    seq_len = values.shape[0]
    chars = list("ACGT")
    patches: list[PathPatch] = []
    facecolors: list[tuple[float, float, float]] = []
    y_min = 0.0
    y_max = 0.0

    for pos in range(seq_len):
        vs = values[pos] * float(height_scale)
        order = np.argsort(vs)
        vs_sorted = vs[order]
        cs = [chars[i] for i in order]

        # Negative attributions accumulate below zero; positive ones above.
        floor = float(np.sum(vs_sorted[vs_sorted < 0]))
        pos_min = floor

        for v, ch in zip(vs_sorted, cs):
            h = abs(float(v))
            if h == 0.0:
                continue
            ceiling = floor + h
            flip = v < 0
            bx = pos - width / 2.0

            if flip:
                vt = _CACHE.flip_verts[ch]
                oy, oh = _CACHE.flip_ymin[ch], _CACHE.flip_h[ch]
            else:
                vt = _CACHE.verts[ch]
                oy, oh = _CACHE.ymin[ch], _CACHE.h[ch]
            ow = _CACHE.w[ch]
            ox = _CACHE.xmin[ch]

            hstretch = min(width / ow, width / _CACHE.ref_w)
            cw = hstretch * ow
            shift = (width - cw) / 2.0
            vstretch = h / oh

            new_verts = vt.copy()
            new_verts[:, 0] = (vt[:, 0] - ox) * hstretch + bx + shift
            new_verts[:, 1] = (vt[:, 1] - oy) * vstretch + floor

            patches.append(PathPatch(MplPath(new_verts, _CACHE.codes[ch])))
            facecolors.append(_DNA_COLORS[ch])
            floor = ceiling

        pos_max = floor
        y_min = min(y_min, pos_min)
        y_max = max(y_max, pos_max)

    pc = PatchCollection(
        patches,
        match_original=False,
        facecolors=facecolors,
        edgecolors="none",
        linewidths=0,
    )
    ax.add_collection(pc)
    ax.set_xlim(-0.5, seq_len - 0.5)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        if y_max == y_min:
            y_max = y_min + 1.0
        pad = 0.05 * (y_max - y_min)
        ax.set_ylim(y_min - pad, y_max + pad)


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _iter_attr_datasets(h5_file: h5py.File, explicit: list[str] | None) -> list[str]:
    """Return the list of attribution dataset names to render."""
    if explicit:
        missing = [k for k in explicit if k not in h5_file]
        if missing:
            raise KeyError(f"Missing requested datasets: {missing}")
        return explicit
    keys = [k for k in h5_file.keys() if k.startswith("attributions")]
    if not keys:
        raise KeyError("No datasets starting with 'attributions' found.")
    return sorted(keys)


def _to_l_by_4(arr: np.ndarray, dset_name: str) -> np.ndarray:
    """Normalize one sample array to shape ``(L, 4)``."""
    if arr.ndim != 2:
        raise ValueError(f"{dset_name}: expected 2-D sample, got {arr.shape}")
    if arr.shape[1] == 4:
        return np.asarray(arr, dtype=np.float32)
    if arr.shape[0] == 4:
        return np.asarray(arr.T, dtype=np.float32)
    raise ValueError(
        f"{dset_name}: nucleotide axis (length 4) not found in dims 1/2; got {arr.shape}"
    )


# ---------------------------------------------------------------------------
# DeepSTARR-specific: motif highlight annotations
# ---------------------------------------------------------------------------
# The following helpers read ``start_dev``/``end_dev``/``start_hk``/``end_hk``
# arrays from the HDF5 file and resolve a single highlight span per sample.
# These are DeepSTARR-specific conventions; remove or replace them when
# adapting this script to other datasets.

def _load_annotation_arrays(h5_file: h5py.File) -> dict[str, np.ndarray]:
    """Load motif-boundary arrays that are present in the file."""
    return {
        key: np.asarray(h5_file[key][:])
        for key in ("start_dev", "end_dev", "start_hk", "end_hk")
        if key in h5_file
    }


def _resolve_highlight_for_sample(
    ann: dict[str, np.ndarray],
    idx: int,
) -> tuple[int, int] | None:
    """Return the inclusive ``(start, end)`` highlight span for one sample.

    Prefers the dev motif span; falls back to hk; returns ``None`` if neither
    is available or the index is out of range.
    """
    required = ("start_dev", "end_dev", "start_hk", "end_hk")
    if any(k not in ann for k in required):
        return None
    if any(idx < 0 or idx >= ann[k].shape[0] for k in required):
        return None

    sd, ed = float(ann["start_dev"][idx]), float(ann["end_dev"][idx])
    sh, eh = float(ann["start_hk"][idx]),  float(ann["end_hk"][idx])
    dev_valid = not (np.isnan(sd) or np.isnan(ed))
    hk_valid  = not (np.isnan(sh) or np.isnan(eh))

    if dev_valid:
        s, e = sd, ed
    elif hk_valid:
        s, e = sh, eh
    else:
        return None

    si, ei = int(round(s)), int(round(e))
    return (ei, si) if ei < si else (si, ei)


def _build_highlight_regions(
    n_samples: int,
    primary_ann: dict[str, np.ndarray],
    fallback_ann: dict[str, np.ndarray] | None,
    sorted_indices: np.ndarray | None,
) -> list[tuple[int, int] | None]:
    """Build per-sample highlight spans, falling back to a secondary H5 file."""
    highlights: list[tuple[int, int] | None] = []
    missing: list[int] = []

    for sample_idx in range(n_samples):
        span = _resolve_highlight_for_sample(primary_ann, sample_idx)

        if span is None and fallback_ann is not None:
            mapped = int(sorted_indices[sample_idx]) if sorted_indices is not None else sample_idx
            if mapped >= 0:
                span = _resolve_highlight_for_sample(fallback_ann, mapped)

        highlights.append(span)
        if span is None:
            missing.append(sample_idx)

    if missing:
        preview = ", ".join(str(i) for i in missing[:20])
        raise ValueError(
            f"Could not resolve highlight for {len(missing)}/{n_samples} samples. "
            f"First missing indices: [{preview}]"
        )
    return highlights


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_and_save(
    values_l4: np.ndarray,
    out_png: Path,
    highlight_region: tuple[int, int] | None = None,
    dpi: int = 350,
    fig_width: float = 8.0,
    fig_height: float = 1.35,
) -> None:
    """Render one attribution logo and write it to ``out_png``."""
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    if highlight_region is not None:
        seq_len = values_l4.shape[0]
        hs = max(0, min(seq_len - 1, highlight_region[0]))
        he = max(0, min(seq_len - 1, highlight_region[1]))
        ax.axvspan(hs - 0.5, he + 0.5, facecolor="#ffd54f", alpha=0.25, edgecolor="none")

    fast_logo(values_l4, ax=ax)

    ax.set_yticks([])
    ax.set_xlabel("position")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------

def render_all_attr_logos(
    input_h5: Path,
    out_dir: Path,
    annotation_h5: Path | None = None,
    datasets: list[str] | None = None,
) -> None:
    """Render and save one logo PNG per sample for each attribution dataset.

    Parameters
    ----------
    input_h5:
        HDF5 file containing attribution arrays (shape ``(N, L, 4)`` or
        ``(N, 4, L)``).  Any top-level dataset whose name starts with
        ``attributions`` is included unless *datasets* is given explicitly.
    out_dir:
        Directory where PNG files are written (created if absent).
    annotation_h5:
        Optional secondary HDF5 file with motif-boundary annotation arrays
        (``start_dev``/``end_dev``/``start_hk``/``end_hk``).  Used only when
        those arrays are absent from *input_h5*.
    datasets:
        Explicit dataset names to render.  Defaults to all ``attributions*``
        datasets in *input_h5*.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    with h5py.File(input_h5, "r") as h5f:
        primary_ann = _load_annotation_arrays(h5f)
        sorted_indices = (
            np.asarray(h5f["sorted_indices"][:]) if "sorted_indices" in h5f else None
        )

        fallback_ann: dict[str, np.ndarray] | None = None
        if annotation_h5 is not None and annotation_h5.exists():
            if annotation_h5.resolve() != input_h5.resolve():
                with h5py.File(annotation_h5, "r") as annf:
                    fallback_ann = _load_annotation_arrays(annf)
            elif not primary_ann:
                fallback_ann = _load_annotation_arrays(h5f)

        target_dsets = _iter_attr_datasets(h5f, datasets)

        # Validate that all requested datasets have the same sample count.
        dset_lengths = {d: int(h5f[d].shape[0]) for d in target_dsets}
        if len(set(dset_lengths.values())) != 1:
            raise ValueError(f"Attribution datasets have mismatched sample counts: {dset_lengths}")
        n_samples = next(iter(dset_lengths.values()))

        highlight_regions = _build_highlight_regions(
            n_samples=n_samples,
            primary_ann=primary_ann,
            fallback_ann=fallback_ann,
            sorted_indices=sorted_indices,
        )

        for dname in target_dsets:
            dset = h5f[dname]
            if dset.ndim != 3:
                raise ValueError(f"{dname}: expected 3-D array (N, L, 4) or (N, 4, L), got {dset.shape}")
            if 4 not in dset.shape[1:]:
                raise ValueError(f"{dname}: nucleotide axis (length 4) missing from dims 1/2; got {dset.shape}")

            safe_name = dname.replace("/", "__")
            for idx in range(n_samples):
                arr = np.asarray(dset[idx], dtype=np.float32)
                out_png = out_dir / f"{safe_name}_sample{idx:06d}.png"
                _plot_and_save(
                    _to_l_by_4(arr, dname),
                    out_png=out_png,
                    highlight_region=highlight_regions[idx],
                )
                total += 1

            print(f"{dname}: wrote {n_samples} logos")

    print(f"Done. Wrote {total} logo PNGs to: {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch-render attribution logo PNGs from an HDF5 file."
    )
    ap.add_argument(
        "--input-h5",
        type=Path,
        default=Path(
            "/grid/koo/home/duran/D3-DNA-Discrete-Diffusion/sampled_seqs/inpainting/attr_all_hits_sorted_gc.h5"
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/grid/koo/home/duran/D3-DNA-Discrete-Diffusion/sampled_seqs/inpainting/all_hits_attr_logos"
        ),
    )
    ap.add_argument(
        "--annotation-h5",
        type=Path,
        default=Path(
            "/grid/koo/home/duran/D3-DNA-Discrete-Diffusion/inpainting_data/all_hits_combined.h5"
        ),
        help=(
            "Optional H5 with start_/end_ motif coordinates. "
            "Used when those keys are absent in --input-h5."
        ),
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Explicit dataset names to render (e.g. attributions_dev attributions_hk).",
    )
    args = ap.parse_args()

    render_all_attr_logos(
        input_h5=args.input_h5,
        out_dir=args.out_dir,
        annotation_h5=args.annotation_h5,
        datasets=args.datasets,
    )


if __name__ == "__main__":
    main()
