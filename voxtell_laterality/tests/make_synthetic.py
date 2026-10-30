"""Build a tiny synthetic ReXGroundingCT-like dataset with deliberately awkward NIfTI orientations.

Left/right is defined in world (RAS) coordinates, so the pipeline must handle flipped and permuted
voxel axes to get laterality right.
"""
import json
import os

import nibabel as nib
import numpy as np

SHAPE = (48, 40, 36)
AFFINES = {  # name -> voxel->world affine
    "train_1_a_1": np.diag([-0.8, -0.8, 1.5, 1.0]),                       # LPS-like (typical DICOM->NIfTI)
    "train_2_a_1": np.diag([0.8, 0.8, 1.5, 1.0]),                         # RAS
    "valid_3_a_1": np.array([[0, 0, 1.5, 0], [-0.8, 0, 0, 0], [0, 0.8, 0, 0], [0, 0, 0, 1.0]]),  # permuted + flip
}
# findings: (text, category, lobe id it really sits in (1 LUL,2 LLL,3 RUL,4 RML,5 RLL) or 'effusion_<side>')
FINDINGS = {
    "train_1_a_1": [("Nodule in the right upper lobe.", "1a", 3), ("Nodule in the left lower lobe.", "1a", 2),
                    ("Bilateral pleural effusion.", "2b", None)],
    "train_2_a_1": [("Consolidation in the LUL.", "1b", 1), ("Right pleural effusion.", "2b", "effusion_right")],
    "valid_3_a_1": [("Nodule in the right middle lobe.", "1a", 4), ("Lingular nodule.", "1a", 1),
                    ("Mild cardiomegaly.", "3a", None)],
}


def world_coords(affine, shape):
    ijk = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3)
    xyz = nib.affines.apply_affine(affine, ijk)
    return xyz.reshape(*shape, 3)


def build(root):
    ct_dir, rex_dir = os.path.join(root, "ctrate"), os.path.join(root, "rex")
    os.makedirs(ct_dir, exist_ok=True)
    os.makedirs(os.path.join(rex_dir, "masks"), exist_ok=True)
    lobes_truth = {}
    entries = []
    rng = np.random.default_rng(0)
    for ci, (case, aff) in enumerate(AFFINES.items()):
        xyz = world_coords(aff, SHAPE)
        c = xyz.reshape(-1, 3).mean(0)
        x, y, z = (xyz[..., i] - c[i] for i in range(3))
        img = np.full(SHAPE, -1000.0)
        body = (x / 18) ** 2 + (y / 14) ** 2 < 1
        img[body] = 40 + 10 * ci  # distinct per case (the mock predictor identifies scans by content)
        lungL = ((x - 8) / 6) ** 2 + (y / 8) ** 2 + (z / 22) ** 2 < 1      # patient left = +x (RAS)
        lungR = ((x + 8) / 6) ** 2 + (y / 8) ** 2 + (z / 22) ** 2 < 1
        img[lungL | lungR] = -800
        lobes = np.zeros(SHAPE, np.uint8)
        lobes[lungL & (z > 0)] = 1
        lobes[lungL & (z <= 0)] = 2
        lobes[lungR & (z > 8)] = 3
        lobes[lungR & (z <= 8) & (z > -4)] = 4
        lobes[lungR & (z <= -4)] = 5
        lobes_truth[case] = lobes
        nib.save(nib.Nifti1Image(img.astype(np.int16), aff), os.path.join(ct_dir, f"{case}.nii.gz"))

        fl = FINDINGS[case]
        masks = np.zeros((len(fl), *SHAPE), np.uint8)
        for k, (text, cat, where) in enumerate(fl):
            if isinstance(where, int):
                pts = np.argwhere(lobes == where)
                p = pts[rng.integers(len(pts))]
                ii, jj, kk = np.ogrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]]
                sph = (ii - p[0]) ** 2 + (jj - p[1]) ** 2 + (kk - p[2]) ** 2 <= 4
                masks[k][sph & (lobes == where)] = 1
            elif isinstance(where, str):
                lung = lungR if where.endswith("right") else lungL
                masks[k][lung & (y < -4)] = 1
            elif cat == "2b":  # bilateral effusion, two entities
                masks[k][lungL & (y < -4)] = 1
                masks[k][lungR & (y < -4)] = 2
            else:  # cardiomegaly: midline, outside lungs
                masks[k][(np.abs(x) < 3) & (np.abs(y) < 3) & (np.abs(z) < 5)] = 1
        if case == "train_1_a_1":   # finding axis last, NIfTI
            nib.save(nib.Nifti1Image(np.moveaxis(masks, 0, 3), aff), os.path.join(rex_dir, "masks", f"{case}.nii.gz"))
        else:                       # finding axis first, npy
            np.save(os.path.join(rex_dir, "masks", f"{case}.npy"), masks)
        entries.append({"name": f"{case}.nii.gz", "findings": {str(i): t for i, (t, _, _) in enumerate(fl)},
                        "categories": {str(i): c for i, (_, c, _) in enumerate(fl)},
                        "entity_counts": {str(i): int(masks[i].max()) for i in range(len(fl))},
                        "shape": list(SHAPE), "protocol": "protocol1"})
    json.dump({"train": [], "val": entries, "test": []}, open(os.path.join(rex_dir, "dataset.json"), "w"))
    return ct_dir, rex_dir, lobes_truth


if __name__ == "__main__":
    import sys
    build(sys.argv[1])
