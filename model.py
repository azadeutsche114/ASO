import os, re, glob, time, random, zipfile, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection   import GroupShuffleSplit
from sklearn.preprocessing     import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr

from app.config import *
from app.features import (
    extract_features, parse_chemistry_arrays, encode_transfection,
    assign_chemistry, reverse_complement, clean_mrna,
    calc_gc, calc_tm, calc_nn_dg, NN_DG
)

warnings.filterwarnings("ignore")
torch.manual_seed(42); np.random.seed(42); random.seed(42)

try:
    import RNA as _RNA
    VIENNA_AVAILABLE = True
except ImportError:
    VIENNA_AVAILABLE = False

try:
    from Bio import Entrez, SeqIO
    Entrez.email = "your@email.com"
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False


# ── Model architecture ────────────────────────────────────────────────────
class OligoAI_Optuna(nn.Module):
    def __init__(
        self, n_feat, aso_dim=0, ctx_dim=16,
        chem_dim=16, back_dim=8, transfect_dim=4,
        seq_len=20, cnn_dim=32, pool_dim=64,
        fusion_dim1=256, fusion_dim2=128,
        feat_dim1=256, feat_dim2=128,
        activation="gelu", drop=0.30
    ):
        super().__init__()
        self.seq_len = seq_len
        ACT = {"relu":nn.ReLU,"gelu":nn.GELU,"silu":nn.SiLU,"mish":nn.Mish}[activation]
        self.sugar_emb    = nn.Embedding(3, chem_dim)
        self.backbone_emb = nn.Embedding(2, back_dim)
        chem_flat = seq_len*(chem_dim+back_dim)
        self.chem_proj = nn.Sequential(
            nn.Linear(chem_flat,128), nn.LayerNorm(128), ACT(),
            nn.Dropout(drop), nn.Linear(128,pool_dim), ACT())
        self.seq_emb = nn.Embedding(4,16)
        self.seq_cnn = nn.Sequential(
            nn.Conv1d(16,64,3,padding=1), nn.BatchNorm1d(64), ACT(),
            nn.Conv1d(64,cnn_dim,5,padding=2), nn.BatchNorm1d(cnn_dim), ACT(),
            nn.AdaptiveMaxPool1d(1))
        self.aso_proj = (nn.Sequential(
            nn.Linear(aso_dim,512), nn.LayerNorm(512), ACT(),
            nn.Dropout(drop), nn.Linear(512,pool_dim), ACT())
            if aso_dim>0 else None)
        self.ctx_pool = (nn.Sequential(
            nn.Linear(ctx_dim,pool_dim), nn.LayerNorm(pool_dim), ACT())
            if ctx_dim>0 else None)
        self.feat_net = nn.Sequential(
            nn.Linear(n_feat,feat_dim1), nn.LayerNorm(feat_dim1), ACT(),
            nn.Dropout(drop), nn.Linear(feat_dim1,feat_dim2),
            nn.LayerNorm(feat_dim2), ACT())
        self.transfect_emb = nn.Embedding(4, transfect_dim)
        aso_out = pool_dim if aso_dim>0 else 0
        ctx_out = pool_dim if ctx_dim>0 else 0
        fin = pool_dim+cnn_dim+aso_out+ctx_out+feat_dim2+transfect_dim
        self.fusion = nn.Sequential(
            nn.Linear(fin,fusion_dim1), nn.LayerNorm(fusion_dim1), ACT(),
            nn.Dropout(drop), nn.Linear(fusion_dim1,fusion_dim2),
            nn.LayerNorm(fusion_dim2), ACT(), nn.Dropout(drop*0.5))
        self.inh_head  = nn.Linear(fusion_dim2,1)
        self.rank_head = nn.Sequential(nn.Linear(fusion_dim2,1), nn.Sigmoid())

    def forward(self, ids, feat, emb_aso, emb_ctx, conf,
                sugar, backbone, log_dos, transfect):
        parts=[]; B=sugar.shape[0]
        sg_flat = self.sugar_emb(sugar).view(B,-1)
        bk_flat = self.backbone_emb(backbone).view(B,-1)
        parts.append(self.chem_proj(torch.cat([sg_flat,bk_flat],dim=1)))
        parts.append(self.seq_cnn(self.seq_emb(ids).transpose(1,2)).squeeze(-1))
        if self.aso_proj is not None: parts.append(self.aso_proj(emb_aso))
        if self.ctx_pool is not None:
            parts.append(self.ctx_pool(emb_ctx*(0.2+0.8*conf.unsqueeze(1))))
        parts.append(self.feat_net(feat))
        parts.append(self.transfect_emb(transfect)*log_dos.unsqueeze(1))
        fused = self.fusion(torch.cat(parts,dim=1))
        return self.inh_head(fused).squeeze(-1), self.rank_head(fused).squeeze(-1)


BASE_MAP = {"A":0,"C":1,"G":2,"T":3}

def seq_to_ids(seq, length=20):
    ids = torch.zeros(length, dtype=torch.long)
    for i,b in enumerate(seq.upper()[:length]):
        ids[i] = BASE_MAP.get(b,0)
    return ids


# ── Global state ──────────────────────────────────────────────────────────
MODEL           = None
MODEL_KWARGS    = None
TOP_FEATURES    = None
EFF_DIM         = None
CTX_DIM         = None
BEST_SPEARMAN   = None
LGBM_PRESCREEN  = None
feat_scaler     = None
vt              = None
vt_name_to_idx  = None
TRAIN_FEAT_COLS = None
X_tr_top        = None
TRAIN_SEQS      = None
y_train         = None
SUGAR_TR        = None
BACKBONE_TR     = None
LOGDOS_TR       = None
TRANS_TR        = None
EMB_SCALER      = None
CTX_SCALER      = None
EMB_TR_NPY      = None
CTX_TR_NPY      = None
CONF_TR_NPY     = None
seq_to_emb_idx  = None
CALIB_RAW       = None
CALIB_Y         = None


def startup():
    global MODEL, MODEL_KWARGS, TOP_FEATURES, EFF_DIM, CTX_DIM, BEST_SPEARMAN
    global LGBM_PRESCREEN, feat_scaler, vt, vt_name_to_idx, TRAIN_FEAT_COLS
    global X_tr_top, TRAIN_SEQS, y_train
    global SUGAR_TR, BACKBONE_TR, LOGDOS_TR, TRANS_TR
    global EMB_SCALER, CTX_SCALER, EMB_TR_NPY, CTX_TR_NPY, CONF_TR_NPY, seq_to_emb_idx
    global CALIB_RAW, CALIB_Y

    _t0 = time.time()

    # ── 1. Load checkpoint ───────────────────────────────────────────────
    zip_path = MODEL_DIR + ".zip"
    if not os.path.exists(MODEL_DIR) and os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path,"r") as z:
            z.extractall("models/")

    initial_ckpt_path = f"{MODEL_DIR}/best_model.pt"
    ckpt_path = initial_ckpt_path
    if not os.path.exists(ckpt_path):
        nested = os.path.join(MODEL_DIR, os.path.basename(MODEL_DIR), "best_model.pt")
        if os.path.exists(nested):
            ckpt_path = nested
            print(f"[STARTUP] Found checkpoint at nested path: {ckpt_path}")

    assert os.path.exists(ckpt_path), \
        f"Checkpoint not found. Checked: {initial_ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    MODEL_KWARGS  = ckpt["model_kwargs"]
    TOP_FEATURES  = ckpt["top_features"]
    EFF_DIM       = ckpt["eff_dim"]
    CTX_DIM       = ckpt["ctx_dim"]
    BEST_SPEARMAN = ckpt.get("best_spearman", 0.0)

    MODEL = OligoAI_Optuna(**MODEL_KWARGS)
    MODEL.load_state_dict(ckpt["model_state_dict"])
    MODEL.eval()

    n_p = sum(p.numel() for p in MODEL.parameters())
    print(f"[STARTUP] Model loaded: {n_p:,} params | Spearman={BEST_SPEARMAN:.4f}")
    print(f"[STARTUP] EFF_DIM={EFF_DIM}  CTX_DIM={CTX_DIM}")

    # ── 2. Rebuild feat_scaler + LGBM from aso_atlas.csv ────────────────
    print(f"[STARTUP] Rebuilding feat_scaler + LGBM from {DATA_FILE}...")
    assert os.path.exists(DATA_FILE), f"{DATA_FILE} not found"

    df = pd.read_csv(DATA_FILE, engine="python", on_bad_lines="skip")
    df.columns = [re.sub(r'[^A-Za-z0-9_]+','_',str(c)).strip('_') for c in df.columns]
    df = df.dropna(subset=["aso_sequence_5_to_3","inhibition_percent"])
    df["aso_sequence_5_to_3"] = df["aso_sequence_5_to_3"].astype(str).str.upper().str.strip()
    df["inhibition_percent"]  = pd.to_numeric(df["inhibition_percent"],errors="coerce").clip(0,100)
    df = df.dropna(subset=["inhibition_percent"])
    for col in ["target_gene","chemistry","cell_line","transfection_method",
                "target_mrna","custom_id","cell_line_species","steric_blocking"]:
        if col in df.columns: df[col] = df[col].fillna("Unknown")
    if "dosage" in df.columns:
        df["dosage"] = pd.to_numeric(df["dosage"],errors="coerce").fillna(
            df["dosage"].median())
    if "custom_id" in df.columns:
        df["patent_id"] = df["custom_id"].str.extract(
            r'(US\d{7,}|EP\d{6,}|WO\d{8,})', expand=False).fillna("unknown")
    else:
        df["patent_id"] = "unknown"

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    raw_tr, _ = next(gss.split(df, groups=df["patent_id"].values))
    df_raw_tr = df.iloc[raw_tr].copy()
    mode_fn = lambda x: x.mode().iloc[0] if len(x)>0 else "Unknown"
    agg = {"inhibition_percent":"mean","patent_id":"first"}
    for c in ["target_gene","chemistry","cell_line","transfection_method",
              "cell_line_species","steric_blocking","custom_id","target_mrna"]:
        if c in df.columns: agg[c] = mode_fn
    if "dosage" in df.columns: agg["dosage"] = "mean"
    df_tr = df_raw_tr.groupby("aso_sequence_5_to_3").agg(agg).reset_index()

    TRAIN_SEQS = df_tr["aso_sequence_5_to_3"].tolist()
    y_train    = df_tr["inhibition_percent"].values.astype("float32")
    chem_col   = "chemistry" if "chemistry" in df_tr.columns else None

    ft_tr=[]; sugar_tr=[]; backbone_tr=[]; logdos_tr=[]; trans_tr=[]
    for _, row in df_tr.iterrows():
        c  = row.get(chem_col) if chem_col else None
        sg, bk = parse_chemistry_arrays(c)
        d  = float(row.get("dosage",4000) or 4000)
        ld = np.log1p(d)
        tr = encode_transfection(row.get("transfection_method","unknown"))
        ft_tr.append(extract_features(row["aso_sequence_5_to_3"], ld, tr, sg, bk))
        sugar_tr.append(sg); backbone_tr.append(bk)
        logdos_tr.append(ld); trans_tr.append(tr)

    SUGAR_TR    = np.array(sugar_tr,    dtype=np.int64)
    BACKBONE_TR = np.array(backbone_tr, dtype=np.int64)
    LOGDOS_TR   = np.array(logdos_tr,   dtype="float32")
    TRANS_TR    = np.array(trans_tr,    dtype=np.int64)

    fdf_tr = pd.DataFrame(ft_tr)
    TRAIN_FEAT_COLS = fdf_tr.columns.tolist()
    vt  = VarianceThreshold(0.0)
    Xr_tr = vt.fit_transform(fdf_tr.values.astype("float32"))
    cols_vt = [TRAIN_FEAT_COLS[i] for i in range(len(TRAIN_FEAT_COLS))
               if vt.get_support()[i]]
    feat_scaler = StandardScaler()
    Xs_tr = feat_scaler.fit_transform(Xr_tr).astype("float32")
    vt_name_to_idx = {n:i for i,n in enumerate(cols_vt)}

    X_tr_top = np.zeros((len(df_tr), len(TOP_FEATURES)), dtype="float32")
    for j,fn in enumerate(TOP_FEATURES):
        if fn in vt_name_to_idx: X_tr_top[:,j] = Xs_tr[:,vt_name_to_idx[fn]]

    LGBM_PRESCREEN = LGBMRegressor(
        n_estimators=500, num_leaves=63, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, n_jobs=-1, random_state=42, verbose=-1)
    LGBM_PRESCREEN.fit(X_tr_top, y_train)
    print(f"[STARTUP] feat_scaler + LGBM fitted on {len(df_tr):,} training sequences")

    # ── 3. Load precomputed embeddings → fit EMB_SCALER ─────────────────
    EMB_SCALER = None; CTX_SCALER = None
    EMB_TR_NPY = None; CTX_TR_NPY = None; CONF_TR_NPY = None
    seq_to_emb_idx = None

    # search in model dir AND models/ folder — notebook saves .npy alongside .pt
    _emb_p  = EMB_TR_PATH
    _seqs_p = TRAIN_SEQS_PATH
    _ctx_p  = CTX_TR_PATH
    _conf_p = CONF_TR_PATH

    if not os.path.exists(_emb_p):
        candidates = (
            glob.glob(f"{MODEL_DIR}/emb_tr*.npy") +
            glob.glob(f"{MODEL_DIR}/*emb*train*.npy") +
            glob.glob("models/**/emb_tr*.npy") +
            glob.glob("*emb_tr*.npy")
        )
        if candidates: _emb_p = candidates[0]

    if not os.path.exists(_seqs_p):
        candidates = (
            glob.glob(f"{MODEL_DIR}/train_seqs*.npy") +
            glob.glob("models/**/train_seqs*.npy") +
            glob.glob("*train_seqs*.npy")
        )
        if candidates: _seqs_p = candidates[0]

    if not os.path.exists(_ctx_p):
        candidates = (
            glob.glob(f"{MODEL_DIR}/ctx_tr*.npy") +
            glob.glob("models/**/ctx_tr*.npy") +
            glob.glob("*ctx_tr*.npy")
        )
        if candidates: _ctx_p = candidates[0]

    if not os.path.exists(_conf_p):
        candidates = (
            glob.glob(f"{MODEL_DIR}/conf_tr*.npy") +
            glob.glob("models/**/conf_tr*.npy") +
            glob.glob("*conf_tr*.npy")
        )
        if candidates: _conf_p = candidates[0]

    print(f"[STARTUP] emb_tr   : {_emb_p}  exists={os.path.exists(_emb_p)}")
    print(f"[STARTUP] train_seqs: {_seqs_p}  exists={os.path.exists(_seqs_p)}")
    print(f"[STARTUP] ctx_tr   : {_ctx_p}  exists={os.path.exists(_ctx_p)}")
    print(f"[STARTUP] conf_tr  : {_conf_p}  exists={os.path.exists(_conf_p)}")

    if os.path.exists(_emb_p) and os.path.exists(_seqs_p):
        EMB_TR_NPY = np.load(_emb_p,   allow_pickle=True).astype("float32")
        _seqs_npy  = [str(s).upper().strip()
                      for s in np.load(_seqs_p, allow_pickle=True)]
        seq_to_emb_idx = {s:i for i,s in enumerate(_seqs_npy)}
        EMB_SCALER = StandardScaler(); EMB_SCALER.fit(EMB_TR_NPY)
        print(f"[STARTUP] EMB_SCALER fitted — shape={EMB_TR_NPY.shape}  "
              f"unique_seqs={len(seq_to_emb_idx)}")
        if os.path.exists(_ctx_p):
            CTX_TR_NPY = np.load(_ctx_p, allow_pickle=True).astype("float32")
            CTX_SCALER = StandardScaler(); CTX_SCALER.fit(CTX_TR_NPY)
            print(f"[STARTUP] CTX_SCALER fitted — shape={CTX_TR_NPY.shape}")
        if os.path.exists(_conf_p):
            CONF_TR_NPY = np.load(_conf_p, allow_pickle=True).astype("float32")
            print(f"[STARTUP] CONF loaded — shape={CONF_TR_NPY.shape}")
    else:
        print("[STARTUP] WARNING: emb_tr.npy or train_seqs.npy NOT FOUND — "
              "calibration will be DISABLED. "
              "Inhibition values will be raw clipped output (expect flat ~43%).")

    # override y_train from .npy if available (more accurate labels)
    if os.path.exists(Y_TRAIN_PATH):
        _y = np.load(Y_TRAIN_PATH, allow_pickle=True).astype("float32")
        if len(_y) == len(X_tr_top):
            y_train = _y
            print(f"[STARTUP] y_train overridden from {Y_TRAIN_PATH}")

    # ── 4. Build calibration table ───────────────────────────────────────
    CALIB_RAW = None; CALIB_Y = None

    if EMB_TR_NPY is not None and seq_to_emb_idx is not None:
        avail_idx = [i for i,s in enumerate(TRAIN_SEQS) if s in seq_to_emb_idx]
        print(f"[STARTUP] Calibration: {len(avail_idx)} training seqs "
              f"have precomputed embeddings")

        if len(avail_idx) == 0:
            print("[STARTUP] WARNING: TRAIN_SEQS and seq_to_emb_idx have ZERO overlap. "
                  "Ensure train_seqs.npy was generated from the same aso_atlas.csv split.")
        else:
            n_calib   = min(CALIB_SAMPLES, len(avail_idx))
            calib_idx = random.sample(avail_idx, n_calib)
            emb_rows  = [seq_to_emb_idx[TRAIN_SEQS[i]] for i in calib_idx]

            calib_y_arr = y_train[calib_idx]
            X_calib     = X_tr_top[calib_idx]
            calib_emb   = EMB_SCALER.transform(
                EMB_TR_NPY[emb_rows]).astype("float32")

            if CTX_TR_NPY is not None and CTX_SCALER is not None:
                calib_ctx = CTX_SCALER.transform(
                    CTX_TR_NPY[emb_rows]).astype("float32")
            else:
                calib_ctx = np.zeros((n_calib, max(CTX_DIM,1)), dtype="float32")

            calib_conf = (CONF_TR_NPY[emb_rows]
                          if CONF_TR_NPY is not None
                          else np.zeros(n_calib, dtype="float32"))

            sugar_c    = torch.tensor(SUGAR_TR[calib_idx],    dtype=torch.long)
            backbone_c = torch.tensor(BACKBONE_TR[calib_idx], dtype=torch.long)
            logdos_c   = torch.tensor(LOGDOS_TR[calib_idx],   dtype=torch.float32)
            trans_c    = torch.tensor(TRANS_TR[calib_idx],    dtype=torch.long)

            with torch.no_grad():
                ids_c  = torch.stack([seq_to_ids(s)
                                      for s in [TRAIN_SEQS[i] for i in calib_idx]])
                feat_c = torch.tensor(X_calib,    dtype=torch.float32)
                emb_c  = torch.tensor(calib_emb,  dtype=torch.float32)
                ctx_c  = torch.tensor(calib_ctx,  dtype=torch.float32)
                conf_c = torch.tensor(calib_conf, dtype=torch.float32)
                pi_c, _ = MODEL(ids_c, feat_c, emb_c, ctx_c, conf_c,
                                sugar_c, backbone_c, logdos_c, trans_c)

            order     = np.argsort(pi_c.numpy())
            CALIB_RAW = pi_c.numpy()[order]
            CALIB_Y   = calib_y_arr[order]
            sp = spearmanr(CALIB_RAW, CALIB_Y)[0]
            print(f"[STARTUP] Calibration built — n={n_calib} | "
                  f"raw=[{CALIB_RAW.min():.2f}, {CALIB_RAW.max():.2f}] | "
                  f"calib_y=[{CALIB_Y.min():.1f}, {CALIB_Y.max():.1f}] | "
                  f"Spearman={sp:.3f} (target ~{BEST_SPEARMAN:.3f})")
    else:
        print("[STARTUP] WARNING: calibration SKIPPED — "
              "inhibition_pct = raw clipped output (expect flat ~43%)")

    print(f"[STARTUP] Total startup time: {time.time()-_t0:.1f}s")
    print("[STARTUP] READY")


# ── Helpers ───────────────────────────────────────────────────────────────
def _calibrate(raw_vals):
    """Monotonic quantile mapping: raw model output → realistic inhibition %."""
    if CALIB_RAW is None:
        print("[PREDICT] WARNING: CALIB_RAW is None — returning clipped raw output. "
              "Values will be unreliable (flat ~43%).")
        return np.clip(raw_vals, 0, 100)
    pct    = np.searchsorted(CALIB_RAW, raw_vals) / len(CALIB_RAW)
    pct    = np.clip(pct, 0, 1)
    idx    = np.clip((pct*(len(CALIB_Y)-1)).astype(int), 0, len(CALIB_Y)-1)
    result = CALIB_Y[idx]
    print(f"[PREDICT] calibrate: raw=[{raw_vals.min():.2f}, {raw_vals.max():.2f}] "
          f"→ inh=[{result.min():.1f}, {result.max():.1f}]")
    return result


def _compute_vienna_ctx(mrna, start, length, ctx_dim):
    if not VIENNA_AVAILABLE or ctx_dim == 0:
        return np.zeros(max(ctx_dim,1), dtype="float32"), 0.0
    try:
        a = max(0, start-VIENNA_WINDOW)
        b = min(len(mrna), start+length+VIENNA_WINDOW)
        ctx_seq = mrna[a:b].replace("T","U")
        fc = _RNA.fold_compound(ctx_seq)
        struct, mfe = fc.mfe(); fc.pf()
        bpp  = fc.bpp()
        site = start-a; site_len = min(length, len(ctx_seq)-site)
        pu = []
        for k in range(site_len):
            pp = (sum(bpp[site+k+1][j] for j in range(1,len(ctx_seq)+1)
                      if j<len(bpp[site+k+1]))
                  if (site+k+1)<len(bpp) else 0.0)
            pu.append(1.0-pp)
        if not pu: pu=[0.5]
        pu = np.array(pu, dtype=float)
        v  = np.zeros(ctx_dim, dtype="float32")
        vals = [
            float(pu.mean()),
            float(pu.std()) if len(pu)>1 else 0.,
            float(pu.max()), float(pu.min()),
            float(pu.mean()>=0.6), float(pu.mean()<0.4),
            float(pu[:5].mean())  if len(pu)>=5 else float(pu.mean()),
            float(pu[len(pu)//2-2:len(pu)//2+2].mean()) if len(pu)>=4 else float(pu.mean()),
            float(pu[-5:].mean()) if len(pu)>=5 else float(pu.mean()),
            float(pu[-1]-pu[0])   if len(pu)>=2 else 0.,
            float(pu[:3].mean())  if len(pu)>=3 else float(pu.mean()),
            float(pu[-3:].mean()) if len(pu)>=3 else float(pu.mean()),
            float(-mfe/max(len(ctx_seq),1)),
            float(start/max(len(mrna),1)),
            float(1.-pu.mean()), float(pu.mean())
        ]
        for i,val in enumerate(vals):
            if i<ctx_dim: v[i]=val
        return v[:ctx_dim], 1.0
    except Exception:
        return np.zeros(ctx_dim, dtype="float32"), 0.0


def _label_region(pos, cds_start, cds_end, mrna_len):
    """Real CDS coords from GenBank. Fallback heuristic for raw seq input."""
    if cds_start is not None and cds_end is not None:
        if pos < cds_start:  return "5UTR"
        elif pos >= cds_end: return "3UTR"
        else:                return "CDS"
    # fallback for raw sequence (no GenBank record)
    if pos < int(mrna_len*0.05):               return "5UTR"
    elif pos > mrna_len - int(mrna_len*0.10):  return "3UTR"
    else:                                       return "CDS"


def _extract_cds(gb_record):
    """Return (cds_start, cds_end) 0-based from a GenBank SeqRecord."""
    for feat in gb_record.features:
        if feat.type == "CDS":
            return int(feat.location.start), int(feat.location.end)
    return None, None


def _detect_input_type(query):
    q = query.strip()
    if os.path.isfile(q):
        return "fasta", q
    if re.match(r'^(NM_|NR_|XM_|XR_|NP_|XP_|NC_|NG_)\d+(\.\d+)?$', q, re.I):
        return "accession", q
    if re.match(r'^\d{3,9}$', q):
        return "gene_id", q
    if re.match(r'^[ACGTUacgtu\s\n]+$', q) and len(re.sub(r'\s','',q)) > 30:
        return "sequence", q
    return "gene_name", q


def _fetch_ncbi(query):
    """Fetch mRNA sequence + real CDS coords from NCBI GenBank.
    Returns (seq_str, description, cds_start, cds_end)."""
    if not BIOPYTHON_AVAILABLE:
        raise RuntimeError("Biopython not installed.")

    input_type, q = _detect_input_type(query)

    if input_type == "accession":
        h   = Entrez.efetch(db="nucleotide", id=q, rettype="gb", retmode="text")
        rec = SeqIO.read(h, "genbank"); h.close()
        cds_start, cds_end = _extract_cds(rec)
        return str(rec.seq), rec.description, cds_start, cds_end

    if input_type == "gene_id":
        h      = Entrez.elink(dbfrom="gene", db="nuccore", id=q,
                               linkname="gene_refseqmrna")
        result = Entrez.read(h); h.close()
        ids    = ([link["Id"] for link in result[0]["LinkSetDb"][0]["Link"]]
                  if result[0]["LinkSetDb"] else [])
        if not ids:
            raise RuntimeError(f"No RefSeq mRNA linked to Gene ID {q}")
        h   = Entrez.efetch(db="nucleotide", id=ids[0], rettype="gb", retmode="text")
        rec = SeqIO.read(h, "genbank"); h.close()
        cds_start, cds_end = _extract_cds(rec)
        return str(rec.seq), rec.description, cds_start, cds_end

    if input_type == "gene_name":
        term = (f"{q}[Gene Name] AND refseq[Filter] AND mRNA[Filter]"
                " AND Homo sapiens[Organism]")
        h   = Entrez.esearch(db="nucleotide", term=term, retmax=5, sort="relevance")
        ids = Entrez.read(h)["IdList"]; h.close()
        if not ids:
            h   = Entrez.esearch(db="nucleotide",
                                  term=f"{q}[Gene Name] AND refseq[Filter] AND mRNA[Filter]",
                                  retmax=5)
            ids = Entrez.read(h)["IdList"]; h.close()
        if not ids:
            raise RuntimeError(f"No RefSeq mRNA found for '{q}'")
        h   = Entrez.efetch(db="nucleotide", id=ids[0], rettype="gb", retmode="text")
        rec = SeqIO.read(h, "genbank"); h.close()
        cds_start, cds_end = _extract_cds(rec)
        return str(rec.seq), rec.description, cds_start, cds_end

    raise ValueError(f"Unknown input type: {input_type}")


# ── Public prediction function ────────────────────────────────────────────
def run_prediction(target_input: str, chemistry: str, dosage_nm: float, transfection: str):
    raw = target_input.strip()
    cds_start, cds_end = None, None

    # ── Resolve mRNA ─────────────────────────────────────────────────────
    if os.path.isfile(raw):
        if BIOPYTHON_AVAILABLE:
            recs = list(SeqIO.parse(raw, "fasta"))
            mrna_raw, desc = str(recs[0].seq), recs[0].description
        else:
            with open(raw) as f:
                lines = [l.strip() for l in f if not l.startswith('>')]
            mrna_raw, desc = "".join(lines), raw
    elif re.match(r'^[ACGTUacgtu\s\n]+$', raw) and len(re.sub(r'\s','',raw)) > 30:
        mrna_raw, desc = raw, "user_sequence"
    else:
        mrna_raw, desc, cds_start, cds_end = _fetch_ncbi(raw)

    mrna       = clean_mrna(mrna_raw)
    log_dos    = float(np.log1p(dosage_nm))
    trans_code = encode_transfection(transfection)
    sugar, backbone = assign_chemistry(chemistry)

    print(f"[PREDICT] target={desc[:60]}  mrna_len={len(mrna):,}  "
          f"cds={cds_start}-{cds_end}")

    # ── Stage 1: generate all windows ────────────────────────────────────
    records = []
    for i in range(0, len(mrna)-WINDOW_LEN+1, WINDOW_STEP):
        win = mrna[i:i+WINDOW_LEN]
        if 'N' in win: continue
        aso = reverse_complement(win)
        records.append({
            "position"    : i,
            "mrna_target" : win,
            "aso_sequence": aso,
            "gc_pct"      : calc_gc(aso),
            "tm_celsius"  : calc_tm(aso),
            "nn_dg"       : calc_nn_dg(aso),
            "mrna_region" : _label_region(i, cds_start, cds_end, len(mrna)),
        })
    df_all = pd.DataFrame(records)
    print(f"[PREDICT] Windows generated: {len(df_all):,}")

    # ── Stage 1: LGBM prescreen ───────────────────────────────────────────
    ft_all  = [extract_features(r.aso_sequence, log_dos, trans_code, sugar, backbone)
               for r in df_all.itertuples()]
    fdf_all = pd.DataFrame(ft_all).reindex(columns=TRAIN_FEAT_COLS, fill_value=0)
    Xr_all  = vt.transform(fdf_all.values.astype("float32"))
    Xs_all  = feat_scaler.transform(Xr_all).astype("float32")
    X_top   = np.zeros((len(df_all), len(TOP_FEATURES)), dtype="float32")
    for j,fn in enumerate(TOP_FEATURES):
        if fn in vt_name_to_idx: X_top[:,j] = Xs_all[:,vt_name_to_idx[fn]]
    df_all["lgbm_score"] = np.clip(LGBM_PRESCREEN.predict(X_top), 0, 100)

    # dedup by position distance
    df_sorted = df_all.sort_values("lgbm_score", ascending=False).reset_index(drop=True)
    kept=[]; last_pos=[]
    for _, row in df_sorted.iterrows():
        if all(abs(row["position"]-p) >= PRESCREEN_DIST for p in last_pos):
            kept.append(row); last_pos.append(row["position"])
        if len(kept) >= PRESCREEN_N: break
    df_pre = pd.DataFrame(kept).reset_index(drop=True)
    print(f"[PREDICT] After LGBM prescreen: {len(df_pre)} candidates")

    pos_to_row = {p:i for i,p in enumerate(df_all["position"].values)}
    pre_rows   = [pos_to_row[p] for p in df_pre["position"].values]
    X_pre_top  = X_top[pre_rows]

    # ── Stage 2: ViennaRNA context ────────────────────────────────────────
    ctx_pre  = np.zeros((len(df_pre), max(CTX_DIM,1)), dtype="float32")
    conf_pre = np.zeros(len(df_pre), dtype="float32")
    if VIENNA_AVAILABLE and CTX_DIM > 0:
        for i, row in enumerate(df_pre.itertuples()):
            ctx_pre[i], conf_pre[i] = _compute_vienna_ctx(
                mrna, row.position, WINDOW_LEN, CTX_DIM)
    ctx_pre_scaled = (CTX_SCALER.transform(ctx_pre).astype("float32")
                      if CTX_SCALER is not None else ctx_pre)

    # ── Stage 2: DL inference (zeros for RiNALMo — calibration handles bias) ──
    n_pre      = len(df_pre)
    emb_zeros  = np.zeros((n_pre, max(EFF_DIM,1)), dtype="float32")
    sugar_t    = torch.tensor(sugar,    dtype=torch.long).unsqueeze(0).repeat(n_pre,1)
    backbone_t = torch.tensor(backbone, dtype=torch.long).unsqueeze(0).repeat(n_pre,1)
    log_dos_t  = torch.full((n_pre,), log_dos,    dtype=torch.float32)
    trans_t    = torch.full((n_pre,), trans_code, dtype=torch.long)

    MODEL.eval()
    with torch.no_grad():
        ids_b  = torch.stack([seq_to_ids(s) for s in df_pre["aso_sequence"]])
        feat_b = torch.tensor(X_pre_top,      dtype=torch.float32)
        emb_b  = torch.tensor(emb_zeros,      dtype=torch.float32)
        ctx_b  = torch.tensor(ctx_pre_scaled, dtype=torch.float32)
        conf_b = torch.tensor(conf_pre,       dtype=torch.float32)
        pi, pr = MODEL(ids_b, feat_b, emb_b, ctx_b, conf_b,
                       sugar_t, backbone_t, log_dos_t, trans_t)

    df_pre = df_pre.copy()
    df_pre["raw_score"]      = pi.numpy()
    df_pre["inhibition_pct"] = _calibrate(pi.numpy())
    df_pre["rank_score"]     = pr.numpy()

    # ── Filters ───────────────────────────────────────────────────────────
    def no_hp(s, mx=F_MAX_HP):
        runs = re.findall(r'(.)\1+', s)
        return max((len(r)+1 for r in runs), default=1) <= mx

    df_pre["f_gc"]      = df_pre["gc_pct"].between(F_GC_MIN, F_GC_MAX)
    df_pre["f_homopoly"]= df_pre["aso_sequence"].apply(no_hp)
    df_pre["f_toxic"]   = df_pre["aso_sequence"].apply(
        lambda s: not any(m in s for m in TOXIC_MOTIFS))
    df_pre["f_pos"]     = df_pre["position"] >= F_NO_5CAP
    df_pre["f_pred"]    = df_pre["inhibition_pct"] >= F_MIN_PRED

    fcols = ["f_gc","f_homopoly","f_toxic","f_pos","f_pred"]
    df_pre["filters_passed"] = df_pre[fcols].sum(axis=1)
    df_pre["all_pass"]       = df_pre[fcols].all(axis=1)
    df_pre["score"] = (
        df_pre["inhibition_pct"]*0.70 +
        df_pre["rank_score"]*100*0.10 +
        df_pre["filters_passed"]*2.5
    ).round(2)

    # ── Final top-N with position dedup ───────────────────────────────────
    def dedup_final(d, min_dist=5):
        d = d.sort_values("score", ascending=False)
        kept=[]; last=[]
        for _,row in d.iterrows():
            if all(abs(row["position"]-p) >= min_dist for p in last):
                kept.append(row); last.append(row["position"])
        return pd.DataFrame(kept).reset_index(drop=True)

    top = pd.concat([
        dedup_final(df_pre[df_pre["all_pass"]]),
        dedup_final(df_pre[~df_pre["all_pass"]]),
    ], ignore_index=True).head(TOP_N).copy()
    top.insert(0, "rank", range(1, len(top)+1))

    print(f"[PREDICT] Final candidates: {len(top)} | "
          f"inh range=[{top['inhibition_pct'].min():.1f}, "
          f"{top['inhibition_pct'].max():.1f}]")

    cols = ["rank","position","aso_sequence","mrna_region",
            "inhibition_pct","gc_pct","tm_celsius","nn_dg",
            "score","all_pass","filters_passed"]
    return top[cols].to_dict(orient="records"), desc