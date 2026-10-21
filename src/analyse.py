"""Tables, figure and facts for the prompt-injection detector study."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, TAB, FIG = ROOT / "results", ROOT / "paper" / "tables", ROOT / "paper" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif"})

DS = {"jailbreakhub": "JailbreakHub", "tensortrust": "Tensor Trust", "bipia": "BIPIA", "opi": "Open-PI"}
DET = {"rebuff": "Rebuff heuristic", "vigil": "Vigil YARA", "lr_word": "LR word n-grams", "lr_char": "LR char n-grams"}
EV = {"none": "None", "synonyms": "Synonyms", "padding": "Benign padding", "leetspeak": "Leetspeak",
      "homoglyph": "Homoglyphs", "spacing": "Letter spacing", "zero_width": "Zero-width chars", "base64": "Base64"}

e12 = pd.read_csv(RES / "e1_e2.csv")
e3 = pd.read_csv(RES / "e3_evasion.csv")
sizes = pd.read_csv(RES / "dataset_sizes.csv")
facts = {}

# Table 1: in-dataset performance (heuristics as shipped; learned detectors 5-fold CV).
ind = e12[(e12.train == e12.test) | (e12.train == "-")]
rows = []
for det in DET:
    r = {"Detector": DET[det]}
    for ds in DS:
        x = ind[(ind.detector == det) & (ind.test == ds)].iloc[0]
        r[DS[ds]] = f"{x.auc:.2f} / {x.tpr:.2f} / {x.fpr:.2f}"
    rows.append(r)
(TAB / "in_dataset.tex").write_text(
    "\\begin{table}[t]\n\\centering\n\\caption{In-dataset detection: AUC / true-positive rate / false-positive rate. Heuristics at "
    "their shipped thresholds (Rebuff 0.75, Vigil any rule match; Rebuff on a stratified subsample of 250 prompts per class "
    "and dataset); learned detectors at 0.5 with 5-fold cross-validation.}\n\\label{tab:in}\n\\footnotesize\n"
    "\\setlength{\\tabcolsep}{3pt}\n" + pd.DataFrame(rows).to_latex(index=False, column_format="lcccc") + "\\end{table}\n")
facts["in_dataset"] = {f"{d}|{t}": {"auc": round(a, 3), "tpr": round(p, 3), "fpr": round(f, 3)}
                       for d, t, a, p, f in ind[["detector", "test", "auc", "tpr", "fpr"]].itertuples(index=False)}

# Figure: cross-dataset AUC for the two learned detectors (rows = train, cols = test), plus leave-one-out row.
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
order = list(DS)
for ax, det in zip(axes, ("lr_word", "lr_char")):
    M = np.full((len(order) + 1, len(order)), np.nan)
    for i, tr in enumerate(order + ["all others"]):
        for j, te in enumerate(order):
            x = e12[(e12.detector == det) & (e12.train == tr) & (e12.test == te)]
            if len(x):
                M[i, j] = x.auc.iloc[0]
    ax.imshow(M, vmin=0.3, vmax=1.0, cmap="viridis")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white" if M[i, j] < 0.75 else "black")
    ax.set_xticks(range(len(order)), [DS[d] for d in order], rotation=35, ha="right")
    ax.set_yticks(range(len(order) + 1), [DS[d] for d in order] + ["Other three"])
    ax.set_xlabel("Tested on")
    ax.set_title(DET[det])
axes[0].set_ylabel("Trained on")
fig.tight_layout()
fig.savefig(FIG / "cross_auc.pdf")

cross = e12[(e12.train != e12.test) & (e12.train != "-") & (e12.train != "all others")]
loo = e12[e12.train == "all others"]
facts["cross_auc"] = {d: {"median": round(float(cross[cross.detector == d].auc.median()), 3),
                          "min": round(float(cross[cross.detector == d].auc.min()), 3),
                          "max": round(float(cross[cross.detector == d].auc.max()), 3)} for d in ("lr_word", "lr_char")}
facts["cross_fpr_median"] = {d: round(float(cross[cross.detector == d].fpr.median()), 3) for d in ("lr_word", "lr_char")}
facts["cross_tpr_median"] = {d: round(float(cross[cross.detector == d].tpr.median()), 3) for d in ("lr_word", "lr_char")}
facts["loo"] = {f"{d}|{t}": {"auc": round(a, 3), "tpr": round(p, 3), "fpr": round(f, 3)}
                for d, t, a, p, f in loo[["detector", "test", "auc", "tpr", "fpr"]].itertuples(index=False)}

# Table 2: evasion, TPR at 1% FPR, averaged over datasets (weighted by positives), and at shipped thresholds.
rows = []
for how in EV:
    r = {"Transformation": EV[how]}
    for det in DET:
        x = e3[(e3.evasion == how) & (e3.detector == det)]
        w = x.n_pos.values
        r[DET[det]] = f"{np.average(x.tpr_at_1pct_fpr, weights=w):.2f}"
        facts.setdefault("evasion", {}).setdefault(how, {})[det] = {
            "tpr1": round(float(np.average(x.tpr_at_1pct_fpr, weights=w)), 3),
            "tpr_default": round(float(np.average(x.tpr_default, weights=w)), 3),
            "by_ds": {a: round(b, 3) for a, b in zip(x.dataset, x.tpr_at_1pct_fpr)}}
    rows.append(r)
(TAB / "evasion.tex").write_text(
    "\\begin{table}[t]\n\\centering\n\\caption{Evasion: true-positive rate on held-out attacks after each transformation, with the "
    "threshold set for a 1\\% false-positive rate on held-out benign prompts of the same dataset, averaged over the four "
    "datasets weighted by the number of attacks. Learned detectors were trained on all four datasets.}\n\\label{tab:ev}\n"
    "\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n" + pd.DataFrame(rows).to_latex(index=False, column_format="lcccc")
    + "\\end{table}\n")

facts["sizes"] = {f"{a}|{b}": int(c) for a, b, c in sizes.itertuples(index=False)}
(RES / "facts.json").write_text(json.dumps(facts, indent=1))
print(json.dumps({k: facts[k] for k in ("cross_auc", "cross_fpr_median", "cross_tpr_median", "loo")}, indent=1))
for how in EV:
    print(how, {d: (v["tpr1"], v["tpr_default"]) for d, v in facts["evasion"][how].items()})
for k, v in facts["in_dataset"].items():
    print(k, v)
