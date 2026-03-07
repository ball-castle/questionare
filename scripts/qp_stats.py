"""本脚本用于提供问卷分析所需的统计计算与聚类基础函数。"""

import math
from collections import Counter

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def cronbach_alpha(x):
    if x.ndim != 2 or x.shape[1] < 2:
        return np.nan, 0
    m = ~np.isnan(x).any(axis=1)
    z = x[m]
    n, k = z.shape
    if n < 3:
        return np.nan, n
    iv = z.var(axis=0, ddof=1)
    tv = z.sum(axis=1).var(ddof=1)
    if tv <= 1e-12:
        return np.nan, n
    a = (k / (k - 1.0)) * (1.0 - iv.sum() / tv)
    return float(a), n


def kmo_bartlett(x):
    m = ~np.isnan(x).any(axis=1)
    z = x[m]
    n, p = z.shape
    if n < 10 or p < 2:
        return {"n_complete": int(n), "kmo": np.nan, "bartlett_chi2": np.nan, "bartlett_df": np.nan, "bartlett_p": np.nan}
    r = np.corrcoef(z, rowvar=False)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0) + np.eye(p) * 1e-8
    invr = np.linalg.pinv(r)
    d = np.sqrt(np.diag(invr))
    pr = -invr / np.outer(d, d)
    np.fill_diagonal(pr, 0.0)
    r2, p2 = r**2, pr**2
    np.fill_diagonal(r2, 0.0)
    np.fill_diagonal(p2, 0.0)
    kmo = float(r2.sum() / (r2.sum() + p2.sum()))
    sgn, ldet = np.linalg.slogdet(r)
    if sgn <= 0:
        chi2, pval = np.nan, np.nan
    else:
        chi2 = -(n - 1 - (2 * p + 5) / 6.0) * ldet
        pval = stats.chi2.sf(chi2, p * (p - 1) / 2.0)
    return {"n_complete": int(n), "kmo": kmo, "bartlett_chi2": float(chi2), "bartlett_df": int(p * (p - 1) / 2), "bartlett_p": float(pval)}


def freq_table(vec):
    v = vec[~np.isnan(vec)]
    cnt = Counter(int(x) if abs(x - round(x)) < 1e-12 else float(x) for x in v)
    tot = sum(cnt.values())
    out = []
    for k in sorted(cnt):
        out.append({"code": k, "count": cnt[k], "pct": 100.0 * cnt[k] / tot})
    return out


def crosstab(a, b):
    m = (~np.isnan(a)) & (~np.isnan(b))
    x, y = a[m], b[m]
    if x.size == 0:
        return None
    xa, ya = sorted(set(int(v) for v in x)), sorted(set(int(v) for v in y))
    mat = np.zeros((len(xa), len(ya)), dtype=int)
    ix, iy = {c: i for i, c in enumerate(xa)}, {c: i for i, c in enumerate(ya)}
    for u, v in zip(x, y):
        mat[ix[int(u)], iy[int(v)]] += 1
    if mat.shape[0] >= 2 and mat.shape[1] >= 2:
        chi2, p, dof, _ = stats.chi2_contingency(mat, correction=False)
    else:
        chi2, p, dof = np.nan, np.nan, np.nan
    return {"n": int(m.sum()), "xa": xa, "ya": ya, "mat": mat, "chi2": float(chi2), "p": float(p), "dof": dof}


def run_mca(num, cols_1b):
    x = num[:, [c - 1 for c in cols_1b]]
    m = ~np.isnan(x).any(axis=1)
    x = x[m]
    if x.shape[0] < 20:
        return None
    levels = [sorted(set(int(v) for v in x[:, j])) for j in range(x.shape[1])]
    d = sum(len(c) for c in levels)
    z = np.zeros((x.shape[0], d))
    labels, t = [], 0
    for j, cats in enumerate(levels):
        for c in cats:
            z[:, t] = (x[:, j] == c).astype(float)
            labels.append(f"Q{cols_1b[j]}={c}")
            t += 1
    p = z / z.sum()
    r, c = p.sum(axis=1), p.sum(axis=0)
    dr = np.diag(1.0 / np.sqrt(r))
    dc = np.diag(1.0 / np.sqrt(c))
    s = dr @ (p - np.outer(r, c)) @ dc
    u, sg, vt = np.linalg.svd(s, full_matrices=False)
    eig = sg**2
    col = (dc @ vt.T[:, :2]) * sg[:2]
    contrib = np.zeros((col.shape[0], 2))
    for k in range(2):
        if eig[k] > 1e-12:
            contrib[:, k] = c * (col[:, k] ** 2) / eig[k]
    return {"eigen": eig, "col": col, "labels": labels, "contrib": contrib}


def logistic_fit(x, y, names):
    m = (~np.isnan(y)) & (~np.isnan(x).any(axis=1))
    x, y = x[m], y[m]
    if x.shape[0] < 30 or len(np.unique(y)) < 2:
        return None
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd < 1e-10] = 1.0
    z = (x - mu) / sd
    n, p = z.shape
    X = np.column_stack([np.ones(n), z])

    def nll(b):
        s = np.clip(X @ b, -30, 30)
        pr = 1.0 / (1.0 + np.exp(-s))
        return -np.sum(y * np.log(pr + 1e-12) + (1 - y) * np.log(1 - pr + 1e-12))

    def grd(b):
        s = np.clip(X @ b, -30, 30)
        pr = 1.0 / (1.0 + np.exp(-s))
        return X.T @ (pr - y)

    res = minimize(nll, np.zeros(p + 1), jac=grd, method="BFGS")
    b = res.x
    cov = np.asarray(res.hess_inv)
    if cov.shape != (p + 1, p + 1):
        return None
    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    zv = b / se
    pv = 2.0 * stats.norm.sf(np.abs(zv))
    s = np.clip(X @ b, -30, 30)
    pr = 1.0 / (1.0 + np.exp(-s))
    pred = (pr >= 0.5).astype(int)
    acc = float((pred == y).mean())
    p0 = y.mean()
    llm = -nll(b)
    lln = np.sum(y * np.log(p0 + 1e-12) + (1 - y) * np.log(1 - p0 + 1e-12))
    r2 = float(1.0 - llm / lln) if abs(lln) > 1e-12 else np.nan
    pos, neg = pr[y == 1], pr[y == 0]
    if len(pos) and len(neg):
        rk = stats.rankdata(np.concatenate([pos, neg]))
        auc = (rk[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    else:
        auc = np.nan
    pbar = 1.0 / (1.0 + np.exp(-b[0]))
    me = b[1:] * pbar * (1 - pbar)
    rows = [{"term": "Intercept", "coef": b[0], "std_err": se[0], "z": zv[0], "p_value": pv[0], "odds_ratio": math.exp(b[0]), "marginal_effect": ""}]
    for i, nm in enumerate(names):
        rows.append(
            {"term": nm, "coef": b[i + 1], "std_err": se[i + 1], "z": zv[i + 1], "p_value": pv[i + 1], "odds_ratio": math.exp(b[i + 1]), "marginal_effect": me[i]}
        )
    return {"n": int(n), "events": int(y.sum()), "accuracy": acc, "auc": float(auc), "pseudo_r2": r2, "rows": rows, "sign": {names[i]: float(np.sign(b[i + 1])) for i in range(len(names))}}


def silhouette_manual(x, labels):
    labs = np.asarray(labels)
    u = np.unique(labs)
    if len(u) < 2:
        return np.nan
    d = cdist(x, x)
    out = []
    for i in range(x.shape[0]):
        same = labs == labs[i]
        same[i] = False
        a = d[i, same].mean() if same.sum() else 0.0
        b = np.inf
        for c in u:
            if c == labs[i]:
                continue
            m = labs == c
            b = min(b, d[i, m].mean())
        den = max(a, b)
        out.append((b - a) / den if den > 0 else 0.0)
    return float(np.mean(out))


def two_stage_cluster(x, ks=(2, 3, 4), seed=42):
    np.random.seed(seed)
    z = linkage(x, method="ward")
    cand, best = [], None
    for k in ks:
        h = fcluster(z, k, criterion="maxclust")
        cen = np.array([x[h == c].mean(axis=0) for c in sorted(np.unique(h))])
        try:
            _, lk = kmeans2(x, cen, minit="matrix", iter=80)
            lb = lk + 1
        except Exception:
            lb = h
        s = silhouette_manual(x, lb)
        rec = {"k": int(k), "silhouette": s, "labels": lb}
        cand.append(rec)
        if best is None or (np.isfinite(s) and s > best["silhouette"]):
            best = rec
    return best, cand


def assign_cluster_names(rows):
    if not rows:
        return {}
    ids = [r["cluster"] for r in rows]
    cog = np.array([r["cognition_mean"] for r in rows])
    mot = np.array([r["motive_count"] for r in rows])
    prm = np.array([r["promo_pref_count"] for r in rows])
    perf = np.array([r["performance_mean"] for r in rows])
    out = {}
    out[ids[int(np.argmax(cog + mot))]] = "文化深度体验型"
    rem = [c for c in ids if c not in out]
    if rem:
        i = [ids.index(c) for c in rem]
        out[rem[int(np.argmax(prm[i]))]] = "价格优惠敏感型"
    rem = [c for c in ids if c not in out]
    if rem:
        i = [ids.index(c) for c in rem]
        out[rem[int(np.argmax(perf[i]))]] = "品质稳定复游型"
    for c in ids:
        out.setdefault(c, "轻度打卡观光型")
    return out
