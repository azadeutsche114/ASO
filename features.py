import re, math, itertools
from collections import Counter
import numpy as np

NN_DG = {
    "AA":-1.0,"AT":-0.88,"AC":-1.44,"AG":-1.28,
    "TA":-0.58,"TT":-1.0,"TC":-1.30,"TG":-1.45,
    "CA":-1.45,"CT":-1.28,"CC":-1.84,"CG":-2.17,
    "GA":-1.30,"GT":-1.44,"GC":-2.24,"GG":-1.84
}

TRANSFECT_MAP = {
    "electroporation":0,"gymnosis":1,
    "lipofection":2,"lipofectamine":2,"rnaimax":2,
    "unknown":3,"other":3
}

def encode_transfection(m):
    m = str(m).lower()
    for k, v in TRANSFECT_MAP.items():
        if k in m: return v
    return 3

def parse_chemistry_arrays(chem_str, seq_len=20):
    sugar    = np.zeros(seq_len, dtype=np.int64)
    backbone = np.ones(seq_len,  dtype=np.int64)
    if chem_str is None: return sugar, backbone
    import pandas as pd
    if pd.isna(chem_str) or str(chem_str).strip() in ["","nan","Unknown","[]"]:
        return sugar, backbone
    s = str(chem_str)
    for mt, mc, ps in re.findall(
            r"Modification\(modification='(\w+)',\s*type='(\w+)',\s*positions=\[([^\]]*)\]\)", s):
        try:
            positions = [int(p.strip())-1 for p in ps.split(',') if p.strip()]
        except:
            continue
        if mc.lower() == 'sugar':
            lbl = 1 if mt.upper()=="MOE" else 2 if mt.upper() in ["CET","CEt","cEt"] else 0
            for p in positions:
                if 0 <= p < seq_len: sugar[p] = lbl
        elif mc.lower() == 'backbone':
            lbl = 1 if mt.upper()=="PS" else 0
            for p in positions:
                if 0 <= p < seq_len: backbone[p] = lbl
    return sugar, backbone

def assign_chemistry(design="5_10_5_MOE"):
    sugar    = np.zeros(20, dtype=np.int64)
    backbone = np.ones(20,  dtype=np.int64)
    if design == "5_10_5_MOE":    sugar[0:5]=1; sugar[15:20]=1
    elif design == "4_12_4_MOE":  sugar[0:4]=1; sugar[16:20]=1
    elif design == "3_14_3_MOE":  sugar[0:3]=1; sugar[17:20]=1
    elif design == "uniform_MOE": sugar[:]=1
    return sugar, backbone

def self_comp_score(seq):
    comp = {"A":"T","T":"A","G":"C","C":"G"}
    rc = "".join(comp.get(b,b) for b in reversed(seq))
    n = len(seq)//2
    return sum(1 for a,b in zip(seq[:n],rc[:n]) if a==b)/max(n,1)

def extract_features(seq, log_dosage=8.3, transfect=3, sugar=None, backbone=None):
    s=seq.upper(); n=len(s); f={}
    f["n"]=n
    for b in "ACGT": f[f"c{b}"]=s.count(b)
    f["gc"]=(f["cG"]+f["cC"])/n; f["at"]=1-f["gc"]
    f["gc_sk"]=(f["cG"]-f["cC"])/max(f["cG"]+f["cC"],1)
    f["at_sk"]=(f["cA"]-f["cT"])/max(f["cA"]+f["cT"],1)
    f["gc5p"]=s[:5].count("G")+s[:5].count("C")
    f["gc3p"]=s[-5:].count("G")+s[-5:].count("C")
    for qi,qs in enumerate([s[0:5],s[5:10],s[10:15],s[15:20]]):
        f[f"gc_q{qi+1}"]=(qs.count("G")+qs.count("C"))/max(len(qs),1)
    f["pur"]=(f["cA"]+f["cG"])/n
    t1,t2=n//3,2*n//3
    for k,seg in [(1,s[:t1]),(2,s[t1:t2]),(3,s[t2:])]:
        f[f"gc{k}"]=(seg.count("G")+seg.count("C"))/max(len(seg),1)
    f["gc_gr"]=f["gc3"]-f["gc1"]
    c=Counter(s)
    f["ent"]=-sum((v/n)*math.log2(v/n) for v in c.values() if v>0)
    runs=re.findall(r'(.)\1+',s)
    f["mhp"]=max((len(r)+1 for r in runs),default=1)
    for b in "ACGT": f[f"p4{b}"]=int(b*4 in s)
    w=5; gap=s[w:n-w]
    f["gT"]=gap.count("T")
    f["gcg"]=(gap.count("G")+gap.count("C"))/max(len(gap),1)
    f["nn"]=sum(NN_DG.get(s[i:i+2],-1.2) for i in range(n-1))
    f["tm"]=2*(s.count("A")+s.count("T"))+4*(s.count("G")+s.count("C"))
    f["e5"]=sum(NN_DG.get(s[i:i+2],-1.2) for i in range(3))
    f["e3"]=sum(NN_DG.get(s[n-4+i:n-4+i+2],-1.2) for i in range(3))
    f["ed"]=f["e5"]-f["e3"]
    f["cpg"]=s.count("CG")
    for qi,(a,b) in enumerate([(0,5),(5,10),(10,15),(15,20)]):
        f[f"nn_q{qi+1}"]=sum(NN_DG.get(s[i:i+2],-1.2) for i in range(a,min(b,n-1)))
    f["self_comp"]=self_comp_score(s)
    for pos in range(1,21):
        for b in "ACGT": f[f"p{pos}{b}"]=int(n>=pos and s[pos-1]==b)
    GOOD=["GCGT","GTCG","CGTA","GTAT","TTGT"]
    BAD =["GGGG","AAAA","TAAA","CTAA","TTTT"]
    f["gm"]=sum(int(m in s) for m in GOOD)
    f["bm"]=sum(int(m in s) for m in BAD)
    f["nm"]=f["gm"]-f["bm"]
    for m in GOOD+BAD: f[f"m{m}"]=int(m in s)
    for k in [2,3]:
        d2=max(n-k+1,1)
        for km in ["".join(p) for p in itertools.product("ACGT",repeat=k)]:
            f[f"k{k}{km}"]=sum(1 for i in range(n-k+1) if s[i:i+k]==km)/d2
    f["log_dosage"]=float(log_dosage)
    f["transfect"]=int(transfect)
    if sugar is not None:
        f["n_MOE"]      =(sugar==1).sum()
        f["n_cEt"]      =(sugar==2).sum()
        f["n_DNA"]      =(sugar==0).sum()
        f["is_510_MOE"] =int((sugar==1).sum()==10 and (sugar==0).sum()==10)
        f["is_3103_cEt"]=int((sugar==2).sum()==6  and (sugar==0).sum()==10)
        f["n_PS"]       =(backbone==1).sum() if backbone is not None else 20
    return f

def reverse_complement(seq):
    comp={'A':'T','T':'A','G':'C','C':'G','N':'N','U':'A'}
    return "".join(comp.get(b,'N') for b in reversed(seq.upper()))

def clean_mrna(seq):
    return re.sub(r'[^ACGTUacgtu]','',str(seq)).upper().replace('U','T')

def calc_tm(aso_seq):
    n=len(aso_seq); gc=aso_seq.count('G')+aso_seq.count('C')
    return round(0.41*(gc/n*100)+67.2-675.0/n, 1)

def calc_gc(seq):
    return round((seq.count('G')+seq.count('C'))/max(len(seq),1)*100, 1)

def calc_nn_dg(seq):
    return round(sum(NN_DG.get(seq[i:i+2],-1.2) for i in range(len(seq)-1)), 2)