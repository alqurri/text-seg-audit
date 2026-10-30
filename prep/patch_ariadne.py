"""Compatibility patches for training/evaluating LanGuideMedSeg-MICCAI2023 with PyTorch Lightning 1.9.

1. engine/wrapper.py reads hyper-parameters as self.hparams.<key>; under PL 1.9 save_hyperparameters()
   stores the config object under its init-argument name, so self.hparams.lr fails. All
   self.hparams.<key> accesses are routed through a property that falls back to that stored object.
   Model and training logic are unchanged (a backup is written to wrapper.py.orig).
2. train.py uses num_workers = batch size (32); set to 8 (data loading only).
3. evaluate.py hard-codes ./save_model/medseg.ckpt; writes evaluate_ckpt.py taking --ckpt.

  python patch_ariadne.py --repo $EXT/LanGuideMedSeg-MICCAI2023
"""
import argparse, os, re, shutil

PROP = '''
    @property
    def _cfg(self):
        # PL>=1.x stores the config under its init-arg name; fall back to it when keys aren't flat
        h = self.hparams
        if "lr" in h:
            return h
        for v in h.values():
            if hasattr(v, "lr") or (isinstance(v, dict) and "lr" in v):
                return v
        return h
'''


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); a = ap.parse_args()
    w = os.path.join(a.repo, "engine", "wrapper.py")
    s = open(w).read()
    if "_cfg" not in s:
        shutil.copyfile(w, w + ".orig")
        s = re.sub(r"self\.hparams\.(\w+)", r"self._cfg.\1", s)
        s, k = re.subn(r"(class LanGuideMedSegWrapper\([^)]*\):\n)", r"\1" + PROP, s, count=1)
        assert k == 1, "wrapper class not found"
        open(w, "w").write(s)
        print("patched", w, "-", s.count("self._cfg."), "hparams accesses")
    t = os.path.join(a.repo, "train.py")
    s = open(t).read()
    s2 = s.replace("num_workers=args.train_batch_size", "num_workers=8").replace("num_workers=args.valid_batch_size", "num_workers=8")
    if s2 != s:
        open(t, "w").write(s2); print("patched", t, "(num_workers=8)")
    e = os.path.join(a.repo, "evaluate.py")
    s = open(e).read()
    if "'./save_model/medseg.ckpt'" in s:
        s = s.replace("'./save_model/medseg.ckpt'", "_ckpt")
        s = s.replace("import argparse", "import argparse, sys\n_ckpt = sys.argv.pop(sys.argv.index('--ckpt') + 1) "
                      "if '--ckpt' in sys.argv else './save_model/medseg.ckpt'\nif '--ckpt' in sys.argv: sys.argv.remove('--ckpt')", 1)
        open(os.path.join(a.repo, "evaluate_ckpt.py"), "w").write(s); print("wrote evaluate_ckpt.py (--ckpt PATH)")


if __name__ == "__main__":
    main()
