"""v7 Ditto-style cross-encoder on Kaggle GPU (T4 x2). Paste into ONE notebook cell and run.
v7 changes: training data from both entity samples, v6-derived gate (0.01-0.99, graph pairs
included), 7% confident pairs + 60k French pairs per fold, test scored once by the fold-1
model (saves ~40 min), NaN guard, one architecture per notebook (ARCH below).

Ditto (Li et al., "Deep Entity Matching with Pre-Trained Language Models", VLDB 2021):
  1. Serialization : each record -> "[COL] name [VAL] ... [COL] address [VAL] ..."
  2. Domain knowledge injection : numbers (house/unit numbers, PINs) are tagged so the
     model attends to them; legal forms are tagged as low-importance spans.
  3. Data augmentation (on training pairs, 50%): span deletion, span shuffle,
     attribute deletion (drop address), entry swap (swap the two records).
  (Ditto's long-text summarization is not needed: our records are short.)

Input  (attached Kaggle dataset): ce_train.parquet, ce_test.parquet
Output (/kaggle/working):         ce_oof_<arch>.parquet, ce_test_<arch>.parquet, ce_report.txt

For each architecture: 2-fold cross-fitting by S1 entity. Train on fold A pairs
(all uncertain pairs + a sample of confident ones), score fold B's uncertain pairs,
and swap. Test uncertain pairs get the mean of both fold models.
Models (license OK, << 8B params):
  minilm : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  (Apache-2.0, 118M)
  xlmr   : xlm-roberta-base                                              (MIT, 278M)
"""
import glob
import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ARCHS = {  # name: (hub id, lr, train batch)
    "minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 5e-5, 128),
    "xlmr": ("xlm-roberta-base", 2e-5, 64),
    "mdeberta": ("microsoft/mdeberta-v3-base", 2e-5, 64),  # MIT, 278M
}
ARCH = "xlmr"  # this notebook's architecture
RUN = [ARCH]
MAX_LEN = 128
CONFIDENT_SAMPLE = 0.07  # share of confident pairs added to training (hard + easy examples)
FR_SAMPLE = 60_000  # pseudo-labelled French pairs added per fold (optional file ce_train_fr.parquet)
SEED = 42
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/content"  # Kaggle or Colab
DEV = "cuda"
torch.manual_seed(SEED)
np.random.seed(SEED)


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) + glob.glob(f"/content/**/{name}", recursive=True)
    assert hits, f"{name} not found - attach the dataset (Kaggle) or upload it to /content (Colab)"
    return hits[0]


NUM = re.compile(r"\d+(?:[/-]\d+)*")
LEGAL = re.compile(r"\b(pvt|private|ltd|limited|llc|llp|inc|corp|corporation|co|sarl|sas|sasu|eurl|sa|sci)\b\.?",
                   re.I)


def inject(text):
    """Ditto domain-knowledge injection: mark numbers and legal-form spans."""
    text = NUM.sub(lambda m: f"[NUM] {m.group(0)} [/NUM]", text)
    return LEGAL.sub(lambda m: f"[LEGAL] {m.group(0)}", text)


def serialize(name, addr):
    return f"[COL] name [VAL] {inject(name)} [COL] address [VAL] {inject(addr)}"


def augment(na, aa, nb, ab):
    """One Ditto augmentation operator, chosen at random."""
    op = random.choice(["span_del", "span_shuffle", "attr_del", "swap"])

    def span_del(t):
        w = t.split()
        if len(w) > 2:
            i = random.randrange(len(w) - 1)
            del w[i:i + random.randint(1, 2)]
        return " ".join(w)

    def span_shuffle(t):
        w = t.split()
        if len(w) > 3:
            i = random.randrange(len(w) - 2)
            seg = w[i:i + 3]
            random.shuffle(seg)
            w[i:i + 3] = seg
        return " ".join(w)
    side = random.random() < 0.5
    if op == "span_del":
        nb, ab = (span_del(nb), ab) if side else (nb, span_del(ab))
    elif op == "span_shuffle":
        nb, ab = (span_shuffle(nb), ab) if side else (nb, span_shuffle(ab))
    elif op == "attr_del":
        ab = ""
    else:
        na, aa, nb, ab = nb, ab, na, aa
    return na, aa, nb, ab


def batches(df, tok, bs, shuffle, labels=True, aug=False):
    idx = np.random.permutation(len(df)) if shuffle else np.argsort(
        (df.name_a.str.len() + df.addr_a.str.len() + df.name_b.str.len() + df.addr_b.str.len()).values)
    cols = [df[c].values for c in ("name_a", "addr_a", "name_b", "addr_b")]
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        ta, tb = [], []
        for r in j:
            f = [c[r] for c in cols]
            if aug and random.random() < 0.5:
                f = augment(*f)
            ta.append(serialize(f[0], f[1]))
            tb.append(serialize(f[2], f[3]))
        enc = tok(ta, tb, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        y = torch.tensor(df.label.values[j], dtype=torch.float32) if labels else None
        yield j, enc, y


def train_model(hub, lr, bs, df, tok):
    model = AutoModelForSequenceClassification.from_pretrained(hub, num_labels=1)
    model = model.float()  # master weights must be fp32 for AMP (some checkpoints load as fp16)
    model.resize_token_embeddings(len(tok))  # Ditto special tokens
    model = model.to(DEV)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = (len(df) + bs - 1) // bs
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t = time.time()
    n_nan = 0
    for k, (_, enc, y) in enumerate(batches(df, tok, bs, shuffle=True, aug=True)):
        enc = {a: b.to(DEV) for a, b in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(**enc).logits.squeeze(-1)
            loss = lossf(logit.float(), y.to(DEV))
        if not torch.isfinite(loss):
            n_nan += 1
            assert n_nan < 50, "loss is NaN/inf repeatedly - fp16 instability, stop this architecture"
            opt.zero_grad()
            continue
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sch.step()
        if k % 200 == 0:
            el = time.time() - t
            eta = el / max(k, 1) * (steps - k) / 60
            print(f"    step {k}/{steps} loss {loss.item():.4f} scale {scaler.get_scale():.0f} ({el:.0f}s, "
                  f"~{eta:.0f} min left in this fold, skipped non-finite {n_nan})", flush=True)
    return model


@torch.no_grad()
def predict(model, df, tok, bs=512):
    model.eval()
    out = np.zeros(len(df), dtype=np.float32)
    for j, enc, _ in batches(df, tok, bs, shuffle=False, labels=False):
        enc = {a: b.to(DEV) for a, b in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            out[j] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
    assert np.isfinite(out).all(), "non-finite predictions"
    return out


def predict_sharded(model, df, tok, prefix, shard=400_000):
    """Test scoring in resumable shards: each shard is written as soon as it is done."""
    parts = []
    for i in range(0, len(df), shard):
        path = f"{OUT}/{prefix}_part{i // shard:02d}.parquet"
        if os.path.exists(path):
            parts.append(pd.read_parquet(path)["ce"].values)
            continue
        t = time.time()
        p = predict(model, df.iloc[i:i + shard], tok)
        df.iloc[i:i + shard][["s1_id", "cand_id"]].assign(ce=p).to_parquet(path, index=False)
        parts.append(p)
        print(f"    test shard {i // shard}: {len(p):,} pairs in {time.time() - t:.0f}s "
              f"(mean {p.mean():.3f}, std {p.std():.3f})", flush=True)
    return np.concatenate(parts)


def main():
    assert torch.cuda.is_available(), ("NO GPU in this session: set Accelerator = GPU T4 x2 "
                                       "(and 'Run with GPU' in Save Version)")
    tr = pd.read_parquet(find("ce_train.parquet"))
    te = pd.read_parquet(find("ce_test.parquet"))
    fr_hits = (glob.glob("/kaggle/input/**/ce_train_fr.parquet", recursive=True)
               + glob.glob("/content/**/ce_train_fr.parquet", recursive=True))
    fr = pd.read_parquet(fr_hits[0]) if fr_hits else None
    if fr is not None:
        for c in ("name_a", "addr_a", "name_b", "addr_b"):
            fr[c] = fr[c].fillna("").astype(str)
        print(f"French pseudo-labelled pairs available: {len(fr):,}")
    random.seed(SEED)
    for d in (tr, te):
        for c in ("name_a", "addr_a", "name_b", "addr_b"):
            d[c] = d[c].fillna("").astype(str)
    print(f"GPUs: {torch.cuda.device_count()} | train pairs {len(tr):,} (uncertain {tr.uncertain.sum():,}) "
          f"| test uncertain {len(te):,}")
    report = []
    for arch in RUN:
        hub, lr, bs = ARCHS[arch]
        tok = AutoTokenizer.from_pretrained(hub)
        tok.add_special_tokens({"additional_special_tokens": ["[COL]", "[VAL]", "[NUM]", "[/NUM]", "[LEGAL]"]})
        oof = pd.Series(np.nan, index=tr.index, dtype=np.float32)
        test_p = np.zeros(len(te), dtype=np.float32)
        t0 = time.time()
        for fold in (0, 1):
            part = tr[tr.fold != fold]
            fit_df = pd.concat([part[part.uncertain],
                                part[~part.uncertain].sample(frac=CONFIDENT_SAMPLE, random_state=SEED)])
            if fr is not None:  # teach French text patterns (training only, never scored)
                fit_df = pd.concat([fit_df, fr.sample(min(FR_SAMPLE, len(fr)), random_state=SEED + fold)],
                                   ignore_index=True)
            print(f"[{arch}] fold {fold}: training on {len(fit_df):,} pairs", flush=True)
            model = train_model(hub, lr, bs, fit_df, tok)
            score_df = tr[(tr.fold == fold) & tr.uncertain]
            po = predict(model, score_df, tok)
            oof.loc[score_df.index] = po
            score_df[["s1_id", "cand_id"]].assign(ce=po).to_parquet(f"{OUT}/ce_oof_{arch}_fold{fold}.parquet", index=False)
            print(f"[{arch}] fold {fold}: OOF saved ({len(po):,} pairs, mean {po.mean():.3f}, std {po.std():.3f}, "
                  f"AUC {roc_auc_score(score_df.label, po):.4f})", flush=True)
            (model.module if hasattr(model, "module") else model).save_pretrained(f"{OUT}/model_{arch}_fold{fold}")
            tok.save_pretrained(f"{OUT}/model_{arch}_fold{fold}")
            if fold == 1:  # test scored once, by the fold-1 model, in resumable shards
                test_p = predict_sharded(model, te, tok, f"ce_test_{arch}")
            del model
            torch.cuda.empty_cache()
        u = tr[tr.uncertain]
        assert oof[u.index].notna().all(), "OOF coverage incomplete"
        assert len(test_p) == len(te) and np.isfinite(test_p).all(), "test coverage incomplete"
        auc_ce = roc_auc_score(u.label, oof[u.index])
        auc_gbm = roc_auc_score(u.label, u.p)
        auc_avg = roc_auc_score(u.label, (oof[u.index] + u.p) / 2)
        line = (f"{arch}: uncertain-pair AUC  cross-encoder={auc_ce:.4f}  lightgbm={auc_gbm:.4f}  "
                f"average={auc_avg:.4f}  ({time.time() - t0:.0f}s)")
        print(line, flush=True)
        report.append(line)
        tr.loc[u.index, ["s1_id", "cand_id"]].assign(ce=oof[u.index].values).to_parquet(
            f"{OUT}/ce_oof_{arch}.parquet", index=False)
        te[["s1_id", "cand_id"]].assign(ce=test_p).to_parquet(f"{OUT}/ce_test_{arch}.parquet", index=False)
    open(f"{OUT}/ce_report_{ARCH}.txt", "w").write("\n".join(report))
    print("DONE. Download the ce_*.parquet files and ce_report.txt from the Output panel.")


main()
