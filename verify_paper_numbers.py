"""Final check: every numeric claim in main.tex vs the merged CSV."""
import pandas as pd, numpy as np, re, glob
from scipy import stats

d = pd.read_csv("results/experiment_results_merged.csv")
f = d[d["round"] == 50]
m = f[(f.feature_set == "original") & (f.noisy_fraction == 0.2)]
M = ["fedavg", "uniform_mean", "fedprox", "csagg", "krum", "trimmed_mean", "coord_median"]
ok, bad = [], []

def chk(label, claimed, actual, tol=5e-5):
    if isinstance(claimed, bool) or isinstance(actual, (bool, np.bool_)):
        good = bool(claimed) == bool(actual)
    else:
        good = abs(claimed - actual) <= tol
    (ok if good else bad).append((label, claimed, actual))
    return good

# --- condition / row counts
chk("total condition-round rows = 31200", 31200, len(d), 0)
chk("unique final conditions = 624", 624,
    len(f[["method","dataset","feature_set","alpha","noise_rate","noisy_fraction","seed"]].drop_duplicates()), 0)
chk("main grid conditions = 504", 504,
    len(m[["method","dataset","alpha","noise_rate","seed"]].drop_duplicates()), 0)
dup = f[f.duplicated(subset=["method","dataset","feature_set","alpha","noise_rate","noisy_fraction","seed"], keep=False)]
chk("duplicate final rows = 0", 0, len(dup), 0)

# --- tab:seven_method
T1 = {"brfss": {"fedavg":(0.7081,0.1439),"fedprox":(0.7286,0.1079),"csagg":(0.7746,0.0237),
                "krum":(0.7439,0.0987),"trimmed_mean":(0.7691,0.0449),"coord_median":(0.7828,0.0137),
                "uniform_mean":(0.7665,0.0376)},
      "breast_cancer": {"fedavg":(0.9970,0.0012),"fedprox":(0.9975,0.0009),"csagg":(0.9975,0.0007),
                "krum":(0.9248,0.1267),"trimmed_mean":(0.9975,0.0007),"coord_median":(0.9971,0.0007),
                "uniform_mean":(0.9977,0.0005)}}
for ds, row in T1.items():
    s = m[m.dataset == ds]
    for meth, (mu, sd) in row.items():
        g = s[s.method == meth]["auc"]
        chk(f"T1 {ds}/{meth} mean", mu, round(g.mean(), 4))
        chk(f"T1 {ds}/{meth} std", sd, round(g.std(), 4))
        chk(f"T1 {ds}/{meth} n=36", 36, len(g), 0)

# --- tab:decomposition (alpha=0.1 BRFSS)
a = m[(m.dataset == "brfss") & (m.alpha == 0.1)]
mu = a.groupby("method")["auc"].mean()
D = {"fedavg":0.5529,"uniform_mean":0.7249,"fedprox":0.6140,"krum":0.6577,
     "trimmed_mean":0.7290,"csagg":0.7497,"coord_median":0.7695}
for k, v in D.items():
    chk(f"T2 alpha=0.1 {k}", v, round(mu[k], 4))
chk("T2 weighting effect +0.1720", 0.1720, round(mu["uniform_mean"] - mu["fedavg"], 4))
for k, v in {"fedprox":-0.1109,"krum":-0.0672,"trimmed_mean":0.0042,"csagg":0.0248,"coord_median":0.0447}.items():
    chk(f"T2 robustness {k}", v, round(mu[k] - mu["uniform_mean"], 4))
for k, v in {"coord_median":79,"csagg":87,"trimmed_mean":98}.items():
    chk(f"T2 {k} pct weighting", v, round(100*(mu["uniform_mean"]-mu["fedavg"])/(mu[k]-mu["fedavg"])), 0.6)

# --- tab:collapse_all
c = m[(m.dataset=="brfss")&(m.alpha==0.1)&(m.noise_rate==0.0)].pivot_table(index="seed",columns="method",values="auc")
C = {42:{"fedavg":0.4933,"uniform_mean":0.7256,"fedprox":0.6184,"csagg":0.7753,"krum":0.7413,"trimmed_mean":0.7355,"coord_median":0.7854},
     123:{"fedavg":0.7740,"uniform_mean":0.7681,"fedprox":0.7769,"csagg":0.7724,"krum":0.7675,"trimmed_mean":0.7761,"coord_median":0.7789},
     456:{"fedavg":0.4368,"uniform_mean":0.7181,"fedprox":0.4924,"csagg":0.7532,"krum":0.3275,"trimmed_mean":0.6855,"coord_median":0.7697}}
for sd, row in C.items():
    for k, v in row.items():
        chk(f"T3 seed{sd}/{k}", v, round(c.loc[sd, k], 4))
for k, v in {"fedavg":0.1806,"uniform_mean":0.0270,"fedprox":0.1425,"csagg":0.0120,
             "krum":0.2468,"trimmed_mean":0.0454,"coord_median":0.0079}.items():
    chk(f"T3 std {k}", v, round(c[k].std(), 4))

# --- tab:nf_sweep
nf = f[(f.dataset=="brfss")&(f.alpha==0.1)&(f.noise_rate==0.2)&(f.feature_set=="original")]
N = nf.groupby(["noisy_fraction","method"])["auc"].mean().unstack("method")
NF = {0.2:{"fedavg":0.5535,"uniform_mean":0.7123,"fedprox":0.6124,"csagg":0.7425,"krum":0.6450,"trimmed_mean":0.7203,"coord_median":0.7699},
      0.3:{"fedavg":0.5322,"uniform_mean":0.7500,"fedprox":0.6227,"csagg":0.7063,"krum":0.6805,"trimmed_mean":0.7459,"coord_median":0.7520},
      0.4:{"fedavg":0.4932,"uniform_mean":0.7462,"fedprox":0.5832,"csagg":0.6820,"krum":0.6539,"trimmed_mean":0.7405,"coord_median":0.7364},
      0.5:{"fedavg":0.5993,"uniform_mean":0.7553,"fedprox":0.6232,"csagg":0.6979,"krum":0.6693,"trimmed_mean":0.7500,"coord_median":0.7383},
      0.6:{"fedavg":0.5528,"uniform_mean":0.7569,"fedprox":0.5628,"csagg":0.6963,"krum":0.5950,"trimmed_mean":0.7454,"coord_median":0.7267}}
for fr, row in NF.items():
    for k, v in row.items():
        chk(f"T5 nf={fr}/{k}", v, round(N.loc[fr, k], 4))

# --- tab:feature_ablation
fa = f[(f.dataset=="brfss")&(f.method.isin(["fedavg","csagg"]))&(f.noisy_fraction==0.2)&(f.noise_rate.isin([0.0,0.2]))]
FA = fa.groupby(["feature_set","alpha","method"])["auc"].mean()
A = {("original",0.1):(0.5608,0.7547,0.1940),("original",0.5):(0.7843,0.7860,0.0017),
     ("original",1.0):(0.7876,0.7880,0.0004),("expanded",0.1):(0.7570,0.7735,0.0165),
     ("expanded",0.5):(0.7829,0.8020,0.0191),("expanded",1.0):(0.7989,0.8024,0.0035)}
for (fs, al), (fe, cs, gap) in A.items():
    chk(f"T4 {fs}/a={al} fedavg", fe, round(FA[(fs,al,"fedavg")], 4))
    chk(f"T4 {fs}/a={al} csagg", cs, round(FA[(fs,al,"csagg")], 4))
    chk(f"T4 {fs}/a={al} gap", gap, round(FA[(fs,al,"csagg")]-FA[(fs,al,"fedavg")], 4), 1e-4)

# --- ratios
fav = m[(m.dataset=="brfss")&(m.method=="fedavg")]
h = fav.groupby("alpha")["auc"].mean(); n = fav[fav.alpha==0.5].groupby("noise_rate")["auc"].mean()
chk("149x ratio", 149, (h[1.0]-h[0.1])/(n[0.0]-n[0.3]), 0.5)
chk("het effect 0.235", 0.235, round(h[1.0]-h[0.1], 3), 5e-4)
chk("noise effect 0.0016", 0.0016, round(n[0.0]-n[0.3], 4))
chk("fedavg a=1.0 0.7877", 0.7877, round(h[1.0],4)); chk("fedavg a=0.1 0.5529", 0.5529, round(h[0.1],4))
chk("fedavg noise0 0.7850", 0.7850, round(n[0.0],4)); chk("fedavg noise0.3 0.7834", 0.7834, round(n[0.3],4))
h2 = fav[fav.noise_rate.isin([0.0,0.2])].groupby("alpha")["auc"].mean()
n2 = fav[(fav.alpha==0.5)&(fav.noise_rate.isin([0.0,0.2]))].groupby("noise_rate")["auc"].mean()
chk("167x ratio", 167, (h2[1.0]-h2[0.1])/abs(n2[0.0]-n2[0.2]), 1.0)
chk("22.7pp 4-feat", 0.227, round(h2[1.0]-h2[0.1], 3), 5e-4)
ex = f[(f.dataset=="brfss")&(f.feature_set=="expanded")&(f.method=="fedavg")]
he = ex.groupby("alpha")["auc"].mean(); ne = ex[ex.alpha==0.5].groupby("noise_rate")["auc"].mean()
chk("14x ratio", 14, (he[1.0]-he[0.1])/abs(ne[0.0]-ne[0.2]), 0.5)
chk("4.2pp 12-feat", 0.042, round(he[1.0]-he[0.1], 3), 5e-4)
chk("5.4x collapse reduction", 5.4, (h2[1.0]-h2[0.1])/(he[1.0]-he[0.1]), 0.05)
chk("19.4pp csagg gap 4-feat", 0.194, round(FA[("original",0.1,"csagg")]-FA[("original",0.1,"fedavg")],4), 5e-4)
chk("1.65pp csagg gap 12-feat", 0.0165, round(FA[("expanded",0.1,"csagg")]-FA[("expanded",0.1,"fedavg")],4), 5e-4)

# --- BC / BRFSS gaps
for k, v in {"coord_median":21.4,"fedavg":28.9,"krum":18.1}.items():
    g = 100*(m[(m.method==k)&(m.dataset=="breast_cancer")].auc.mean()-m[(m.method==k)&(m.dataset=="brfss")].auc.mean())
    chk(f"gap {k}", v, round(g,1), 0.06)
stable = [x for x in M if x != "krum"]
bcs = m[(m.dataset=="breast_cancer")&(m.method.isin(stable))]
chk("BC stable min >= 0.994", True, bcs.auc.min() >= 0.994, 0)
chk("BC stable all means >= 0.997", True, bcs.groupby("method").auc.mean().min() >= 0.997, 0)
sp = bcs.groupby("method").auc.mean(); chk("BC mean spread 0.0007", 0.0007, round(sp.max()-sp.min(),4))
bs = m[(m.dataset=="brfss")&(m.method.isin(stable))].groupby("method").auc.mean()
chk("BRFSS mean spread 0.075", 0.075, round(bs.max()-bs.min(),3), 5e-4)
chk("krum BC worst 0.3816", 0.3816, round(m[(m.method=="krum")&(m.dataset=="breast_cancer")].auc.min(),4))
chk("others BC std <= 0.0012", True, m[(m.dataset=="breast_cancer")&(m.method!="krum")].groupby("method").auc.std().max() <= 0.0012, 0)
chk("fedavg worst a=0.1 noise=0 is 0.437", 0.4368, round(c["fedavg"].min(),4))
chk("fedavg worst a=0.1 all-noise 0.400", 0.4002,
    round(m[(m.dataset=="brfss")&(m.alpha==0.1)&(m.method=="fedavg")].auc.min(),4))
chk("uniform worst seed 0.718", 0.7181, round(c["uniform_mean"].min(),4))

# --- Holm-corrected p-values quoted in the text
def holm(ps):
    n=len(ps); o=sorted(range(n),key=lambda i:ps[i]); adj=[0]*n; prev=0
    for k,i in enumerate(o):
        v=min(1.0,(n-k)*ps[i]); prev=max(prev,v); adj[i]=prev
    return adj
s = m[m.dataset=="brfss"]; base = s[s.method=="uniform_mean"].set_index(["alpha","noise_rate","seed"])["auc"].sort_index()
ks=[x for x in M if x!="uniform_mean"]; ps=[];ds_=[]
for k in ks:
    v=s[s.method==k].set_index(["alpha","noise_rate","seed"])["auc"].sort_index()
    cm=base.index.intersection(v.index); st,p=stats.wilcoxon(v[cm].values,base[cm].values)
    ps.append(p); ds_.append(v[cm].mean()-base[cm].mean())
adj=dict(zip(ks,holm(ps))); dl=dict(zip(ks,ds_))
chk("coord_median vs control +0.0163", 0.0163, round(dl["coord_median"],4))
chk("coord_median vs control p<0.0001", True, adj["coord_median"]<0.0001, 0)
chk("trimmed vs control +0.0026", 0.0026, round(dl["trimmed_mean"],4))
chk("trimmed vs control p=0.0036", 0.0036, round(adj["trimmed_mean"],4), 5e-5)
chk("csagg vs control +0.0081", 0.0081, round(dl["csagg"],4))
chk("csagg vs control p=0.22 ns", True, 0.20 <= adj["csagg"] <= 0.23, 0)
chk("krum vs control -0.0226", -0.0226, round(dl["krum"],4))
chk("fedavg/fedprox vs control p=0.0036", 0.0036, round(adj["fedavg"],4), 5e-5)
a01 = m[(m.dataset=="brfss")&(m.alpha==0.1)]
b01 = a01[a01.method=="uniform_mean"].set_index(["noise_rate","seed"])["auc"].sort_index()
ps=[];ds2=[]
for k in ks:
    v=a01[a01.method==k].set_index(["noise_rate","seed"])["auc"].sort_index()
    cm=b01.index.intersection(v.index); st,p=stats.wilcoxon(v[cm].values,b01[cm].values)
    ps.append(p); ds2.append(v[cm].mean()-b01[cm].mean())
a2=dict(zip(ks,holm(ps)))
chk("a=0.1 coord_median p=0.0029", 0.0029, round(a2["coord_median"],4), 5e-5)
chk("a=0.1 only coord_median sig", True, all(a2[k]>0.05 for k in ks if k!="coord_median"), 0)

# CS-Agg comparisons (Holm over 12)
rows=[]
for ds2_ in ["brfss","breast_cancer"]:
    s2=m[m.dataset==ds2_]; cs=s2[s2.method=="csagg"].set_index(["alpha","noise_rate","seed"])["auc"].sort_index()
    for o in [x for x in M if x!="csagg"]:
        v=s2[s2.method==o].set_index(["alpha","noise_rate","seed"])["auc"].sort_index()
        cm=cs.index.intersection(v.index); st,p=stats.wilcoxon(cs[cm].values,v[cm].values)
        rows.append((ds2_,o,p))
aj=holm([r[2] for r in rows]); H={(r[0],r[1]):a for r,a in zip(rows,aj)}
chk("csagg vs coord_median p=0.0002", 0.0002, round(H[("brfss","coord_median")],4), 5e-5)
chk("csagg vs trimmed p=0.59", 0.59, round(H[("brfss","trimmed_mean")],2), 5e-3)

print(f"\nPASS {len(ok)}   FAIL {len(bad)}")
for label, cl, ac in bad:
    print(f"  MISMATCH  {label}: paper says {cl}, data says {ac}")
