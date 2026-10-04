import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1); tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def summary(y, p):
    return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), ece=ece(y, p), n=int(len(y)), pos=int(np.sum(y)))


def sens_at_spec(y_val, p_val, y_test, p_test, spec=0.90):
    """Threshold chosen on the VALIDATION fold only, then applied to the test fold."""
    thr = float(np.quantile(p_val[y_val == 0], spec))
    pred = p_test > thr
    sens = float(pred[y_test == 1].mean()); sp = float((~pred[y_test == 0]).mean())
    return sens, sp, thr


def patient_bootstrap_auc(y, p, groups, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, p, groups = map(np.asarray, (y, p, groups))
    ids = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ids}
    out = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[g] for g in rng.choice(ids, len(ids))])
        if len(np.unique(y[pick])) == 2:
            out.append(roc_auc_score(y[pick], p[pick]))
    return np.percentile(out, [2.5, 97.5]).tolist()
