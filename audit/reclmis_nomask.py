"""RecLMIS: is the ground-truth mask passed to the test-time forward really unused? Compares predictions
(probabilities and thresholded masks) with the true mask vs a zero mask, and times one full pass."""
import argparse, importlib, json, os, sys, tempfile, time
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reclmis_shuffle import forward_logits

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
import pandas as pd, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.getcwd())
config = importlib.import_module(a.cfg)
from Load_Dataset import ImageToImage2D, ValGenerator
from nets.RecLMIS import RecLMIS
from utils import read_text
model = RecLMIS(config, config.get_ViT_config(), n_channels=config.n_channels, n_classes=config.n_labels).cuda()
res = model.load_state_dict(torch.load(a.ckpt, map_location="cuda", weights_only=False)["state_dict"], strict=False)
print("missing", len(res.missing_keys), "unexpected", len(res.unexpected_keys))
if res.missing_keys or res.unexpected_keys:
    sys.exit("ERROR: checkpoint mismatch")
model.eval()
df = pd.read_excel(os.path.join(config.test_dataset, "Test_text.xlsx"))
padded = [t + " EOF XXX" * (9 - len(t.split())) if len(t.split()) < 9 else t for t in df.Description]
p = os.path.join(tempfile.mkdtemp(), "m.xlsx")
pd.DataFrame({"Image": list(df.Image), "Description": padded}).to_excel(p, index=False)
ds = ImageToImage2D(config.test_dataset, config.task_name, read_text(p), ValGenerator(output_size=[config.img_size] * 2),
                    image_size=config.img_size, data_name=config.task_name, token_len=config.token_len, config=config, mode="val")
maxd, same_mask, n = 0.0, 0, 0
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad():
    for b, names in DataLoader(ds, batch_size=1, shuffle=False):
        img, lab = b["image"].cuda(), b["label"].cuda(); tok, tm = b["text_token"].cuda(), b["text_mask"].cuda()
        o1 = forward_logits(model, img, lab, tok, tm)
        o0 = forward_logits(model, img, torch.zeros_like(lab), tok, tm)
        maxd = max(maxd, float((o1 - o0).abs().max())); same_mask += int(torch.equal(o1 > 0.5, o0 > 0.5)); n += 1
torch.cuda.synchronize(); el = time.time() - t0
r = {"n": n, "identical_masks": same_mask, "max_abs_output_diff": maxd, "seconds_two_passes": el, "seconds_per_pass": el / 2}
json.dump(r, open(os.path.join(a.out, "nomask.json"), "w"), indent=1); print(r)
