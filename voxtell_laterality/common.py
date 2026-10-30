"""Shared helpers: dataset JSON parsing, file resolution, NIfTI/mask IO, side maps, mask packing.

All arrays handled here are in the ORIGINAL nibabel voxel order of the CT file (np.asanyarray(img.dataobj)).
VoxTell predicts in nnU-Net's reoriented reader space; `reader_to_original` maps those predictions back.
"""
import json
import os
import re
from pathlib import Path

import numpy as np

LOBES = {1: "lung_upper_lobe_left", 2: "lung_lower_lobe_left",
         3: "lung_upper_lobe_right", 4: "lung_middle_lobe_right", 5: "lung_lower_lobe_right"}
LEFT_LABELS, RIGHT_LABELS = (1, 2), (3, 4, 5)

# ---------------------------------------------------------------- dataset json

SPLIT_ALIASES = {"val": ("val", "valid", "validation"), "train": ("train", "training"), "test": ("test",)}


def load_entries(json_path, split):
    """Return the list of case dicts for `split` from a ReXGroundingCT json (dict-of-lists layout)."""
    data = json.load(open(json_path))
    if isinstance(data, list):
        raise ValueError(f"{json_path}: top level is a list; expected splits. Keys unknown - inspect the file.")
    keymap = {k.lower(): k for k in data}
    for alias in SPLIT_ALIASES.get(split, (split,)):
        if alias in keymap:
            entries = data[keymap[alias]]
            if isinstance(entries, dict):  # {name: entry}
                entries = [dict(v, name=v.get("name", k)) for k, v in entries.items()]
            return entries
    raise KeyError(f"{json_path}: no split '{split}'. Available: {list(data)}")


def finding_index_map(findings: dict, n_channels=None):
    """Map json finding keys -> 0-based channel index. Accepts 0- or 1-based integer keys."""
    keys = sorted(int(k) for k in findings)
    if n_channels is not None and len(keys) != n_channels:
        raise ValueError(f"{len(keys)} findings in json but mask has {n_channels} channels")
    if keys == list(range(len(keys))):
        off = 0
    elif keys == list(range(1, len(keys) + 1)):
        off = 1
    else:
        raise ValueError(f"finding keys are not contiguous integers: {keys}")
    return {str(k): k - off for k in keys}


def case_stem(name: str) -> str:
    for suf in (".nii.gz", ".nii", ".npz", ".npy"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


# ---------------------------------------------------------------- file resolution

def index_files(root, patterns=(".nii.gz", ".nii", ".npz", ".npy")):
    idx = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(patterns):
                idx.setdefault(case_stem(fn), []).append(os.path.join(dp, fn))
    return idx


def ctrate_relpaths(name: str):
    """Candidate CT-RATE repo paths for e.g. 'train_1741_b_2.nii.gz' (fixed-header folders first)."""
    stem = case_stem(name)
    m = re.match(r"^(train|valid)_(\d+)_([a-z])_(\d+)$", stem)
    if not m:
        return []
    split, pid, rec, _ = m.groups()
    sub = f"{split}_{pid}/{split}_{pid}_{rec}/{stem}.nii.gz"
    return [f"dataset/{split}_fixed/{sub}", f"dataset/{split}/{sub}"]


# ---------------------------------------------------------------- nifti / masks

def load_nifti(path):
    import nibabel as nib
    img = nib.load(path)
    return img, np.asanyarray(img.dataobj)


def voxel_spacing(affine):
    return np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(0))


def load_rex_mask(path, n_findings, spatial_shape):
    """Load a ReXGroundingCT mask -> array (F, *spatial_shape) of entity ids (0 = background).

    Accepts 4D NIfTI or npy/npz with the finding axis first or last. Refuses to guess when the
    spatial axes don't match the CT's original voxel order exactly.
    """
    if path.endswith((".nii", ".nii.gz")):
        _, arr = load_nifti(path)
    elif path.endswith(".npz"):
        z = np.load(path)
        arr = z[z.files[0]]
    else:
        arr = np.load(path)
    spatial_shape = tuple(spatial_shape)
    if arr.ndim == 3:
        if n_findings != 1:
            raise ValueError(f"{path}: 3D mask but {n_findings} findings")
        arr = arr[None]
    elif arr.ndim == 4:
        if tuple(arr.shape[1:]) == spatial_shape and arr.shape[0] == n_findings:
            pass
        elif tuple(arr.shape[:3]) == spatial_shape and arr.shape[3] == n_findings:
            arr = np.moveaxis(arr, 3, 0)
        else:
            raise ValueError(f"{path}: mask shape {arr.shape} does not match CT {spatial_shape} with "
                             f"{n_findings} findings (finding axis first or last). Not guessing a permutation.")
    else:
        raise ValueError(f"{path}: unexpected mask ndim {arr.ndim}")
    if tuple(arr.shape[1:]) != spatial_shape:
        raise ValueError(f"{path}: spatial shape {arr.shape[1:]} != CT {spatial_shape}")
    return arr


def reader_to_original(seg_reader, props):
    """Map an array from NibabelIOWithReorient space (X,Y,Z after RAS + transpose) to original voxel order.

    Equivalent to NibabelIOWithReorient.write_seg followed by reading the file back with nibabel.
    Works for (X,Y,Z) or (N,X,Y,Z).
    """
    from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform
    from_canonical = ornt_transform(axcodes2ornt("RAS"), io_orientation(props["nibabel_stuff"]["original_affine"]))
    if seg_reader.ndim == 4:
        return np.stack([reader_to_original(s, props) for s in seg_reader])
    return apply_orientation(seg_reader.transpose(2, 1, 0), from_canonical)


# ---------------------------------------------------------------- side map

def side_map(lobes, spacing, roi_mm=25.0, step=2):
    """Assign every voxel to the nearer lung. Returns (side, roi) with side 1=left, 2=right.

    roi = within roi_mm of either lung (covers pleura / pleural effusions just outside parenchyma).
    Computed on a strided grid for speed and upsampled by repetition.
    """
    from scipy.ndimage import distance_transform_edt
    sl = tuple(slice(None, None, step) for _ in range(3))
    sub = lobes[sl]
    sp = np.asarray(spacing, float) * step
    left = np.isin(sub, LEFT_LABELS)
    right = np.isin(sub, RIGHT_LABELS)
    if not left.any() or not right.any():
        raise ValueError("lobe mask lacks a left or right lung")
    dl = distance_transform_edt(~left, sampling=sp)
    dr = distance_transform_edt(~right, sampling=sp)
    side_s = np.where(dl <= dr, 1, 2).astype(np.uint8)
    roi_s = np.minimum(dl, dr) <= roi_mm
    side = _upsample(side_s, step, lobes.shape)
    roi = _upsample(roi_s, step, lobes.shape)
    return side, roi


def _upsample(a, step, shape):
    for ax in range(3):
        a = np.repeat(a, step, axis=ax)
    return a[: shape[0], : shape[1], : shape[2]]


def side_fractions(mask, side, roi):
    """Fraction of mask voxels in left / right hemithorax ROI, and fraction inside ROI at all."""
    m = mask.astype(bool)
    n = int(m.sum())
    if n == 0:
        return {"n": 0, "left": np.nan, "right": np.nan, "in_roi": np.nan}
    in_roi = m & roi
    k = int(in_roi.sum())
    left = int((in_roi & (side == 1)).sum())
    return {"n": n, "left": left / k if k else np.nan, "right": (k - left) / k if k else np.nan,
            "in_roi": k / n}


# ---------------------------------------------------------------- packing

def pack_mask(m):
    """Bounding-box crop + bitpack. Small for lesions, a few MB for a lung."""
    m = m.astype(bool)
    if not m.any():
        return {"shape": list(m.shape), "bbox": None}
    idx = np.argwhere(m)
    lo, hi = idx.min(0), idx.max(0) + 1
    crop = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return {"shape": list(m.shape), "bbox": [lo.tolist(), hi.tolist()], "bits": np.packbits(crop.ravel())}


def unpack_mask(p):
    out = np.zeros(p["shape"], bool)
    if p["bbox"] is None:
        return out
    lo, hi = p["bbox"]
    cshape = tuple(h - l for l, h in zip(lo, hi))
    crop = np.unpackbits(p["bits"], count=int(np.prod(cshape))).reshape(cshape).astype(bool)
    out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = crop
    return out


def save_packed(path, masks: dict):
    """masks: {key: bool array}. Stored as one npz of object-free arrays."""
    flat = {}
    for k, m in masks.items():
        p = pack_mask(m)
        flat[f"{k}__shape"] = np.asarray(p["shape"])
        if p["bbox"] is None:
            flat[f"{k}__empty"] = np.asarray(1)
        else:
            flat[f"{k}__bbox"] = np.asarray(p["bbox"])
            flat[f"{k}__bits"] = p["bits"]
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, **flat)
    os.replace(tmp, path)


def load_packed(path):
    z = np.load(path)
    keys = sorted({f.rsplit("__", 1)[0] for f in z.files})
    out = {}
    for k in keys:
        shape = z[f"{k}__shape"].tolist()
        if f"{k}__empty" in z.files:
            out[k] = unpack_mask({"shape": shape, "bbox": None})
        else:
            out[k] = unpack_mask({"shape": shape, "bbox": z[f"{k}__bbox"].tolist(), "bits": z[f"{k}__bits"]})
    return out


def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else np.nan


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
