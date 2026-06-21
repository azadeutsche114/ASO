import os

MODEL_DIR       = os.getenv("MODEL_DIR",   "models/optuna_best_model")
DATA_FILE       = os.getenv("DATA_FILE",   "models/aso_atlas.csv")
EMB_TR_PATH     = os.getenv("EMB_TR_PATH", "models/emb_tr.npy")
CTX_TR_PATH     = os.getenv("CTX_TR_PATH", "models/ctx_tr_emb.npy")
CONF_TR_PATH    = os.getenv("CONF_TR_PATH","models/conf_tr.npy")
TRAIN_SEQS_PATH = os.getenv("TRAIN_SEQS_PATH", "models/train_seqs.npy")
Y_TRAIN_PATH    = os.getenv("Y_TRAIN_PATH","models/y_train.npy")

TOP_N          = 15
WINDOW_LEN     = 20
WINDOW_STEP    = 1
VIENNA_WINDOW  = 150
PRESCREEN_N    = 250
PRESCREEN_DIST = 5
CALIB_SAMPLES  = 500

DEFAULT_CHEMISTRY    = "5_10_5_MOE"
DEFAULT_DOSAGE_NM    = 4000.0
DEFAULT_TRANSFECTION = "lipofection"

F_GC_MIN     = 35.0
F_GC_MAX     = 65.0
F_MAX_HP     = 3
F_MIN_PRED   = 40.0
F_NO_5CAP    = 50
TOXIC_MOTIFS = ["GGGG", "CCCC", "TTTT", "AAAA", "TTTTT"]