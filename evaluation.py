from dice_ml.utils import helpers 
import dice_ml
from sklearn.model_selection import train_test_split
import os, re, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURE_ORDER = ["age", "workclass", "education", "marital_status", "occupation", "race", "gender", "hours_per_week", "income"]
OUTCOME_NAME = "income"
IMMUTABLE = ["age", "gender", "race"]         # feasibility constraints
DIRECTIONAL = {"hours_per_week": "up"}     # 'up' or 'down'
POSITIVE_LABEL = 1                         # model's positive class

rag_cfe_dir = "/rag_cfe_outputs" 
dice_cfe_dir = "/dice_cfe_outputs"         

dataset = helpers.load_adult_income_dataset()
target = dataset["income"] 
train_dataset, test_dataset, _, _ = train_test_split(dataset,
                                                     target,
                                                     test_size=0.2,
                                                     random_state=0,
                                                     stratify=target)

query_instance = test_dataset.drop(columns=OUTCOME_NAME)
batch_index = query_instance.index.to_list()  

d = dice_ml.Data(dataframe=train_dataset,
                 continuous_features=['age', 'hours_per_week'],
                 outcome_name='income')

m = dice_ml.Model(model_path=dice_ml.utils.helpers.get_adult_income_modelpath(),
                  backend='TF2', func="ohe-min-max")

exp = dice_ml.Dice(d,m)

def parse_fid(fn):
    m = re.search(r"(\d+)", fn)
    return int(m.group(1)) if m else None   

def list_to_dict(lst, order):
    return {k: v for k, v in zip(order, lst)}

def load_json_dir(json_dir, feature_order, outcome_name, factual_df, batch_index=None):
    """Load JSON CFs and map filename number to the correct factual row."""
    pairs = []
    pos_df = factual_df.reset_index(drop=True)  

    for fn in sorted(os.listdir(json_dir)):
        if not fn.lower().endswith(".json"):
            continue
        fid_pos = parse_fid(fn)  
        if fid_pos is None:
            raise ValueError(f"Filename '{fn}' must contain an integer id.")

        with open(os.path.join(json_dir, fn), "r") as f:
            obj = json.load(f)

        cf_full = list_to_dict(obj["best_cf"], feature_order)
        cf = {k: v for k, v in cf_full.items() if k != outcome_name}

        if batch_index is not None and 0 <= fid_pos < len(batch_index):
            true_label = batch_index[fid_pos] 
            factual_row = factual_df.loc[true_label, :]
        elif fid_pos in factual_df.index:
            factual_row = factual_df.loc[fid_pos, :]
        elif 0 <= fid_pos < len(pos_df):
            factual_row = pos_df.iloc[fid_pos, :]
        else:
            raise KeyError(
                f"ID {fid_pos} not found as batch position, index label, or positional row."
            )

        factual = {k: factual_row[k] for k in feature_order if k != outcome_name}
        pairs.append({"id": fid_pos, "factual": factual, "cfs": [cf]})
    return pairs

def dice_predict(model_obj, df, data_interface, reference_df=None):

    df2 = df.copy()
    if reference_df is not None:
        for col in df2.columns:
            if col in reference_df.columns:
                try:
                    df2[col] = df2[col].astype(reference_df[col].dtype)
                except Exception:
                    pass

    preds = None
    err_msgs = []
    
    try:
        preds = model_obj.get_output(data_interface, df2)
    except Exception as e:
        err_msgs.append(f"get_output(data_interface, df): {e}")

    if preds is None:
        try:
            preds = model_obj.get_output(df2)
        except Exception as e:
            err_msgs.append(f"get_output(df): {e}")

    if preds is None:
        transformed = None
        for fn in ("prepare_query_instance", "transform_data", "one_hot_encode_data"):
            if hasattr(data_interface, fn):
                try:
                    transformed = getattr(data_interface, fn)(df2)
                    break
                except Exception as e:
                    err_msgs.append(f"{fn}(df) failed: {e}")
        if transformed is not None:
            try:
                preds = model_obj.predict(transformed)
            except Exception as e:
                err_msgs.append(f"model_obj.predict(transformed): {e}")
                if hasattr(model_obj, "model"):
                    try:
                        preds = model_obj.model.predict(transformed)
                    except Exception as e2:
                        err_msgs.append(f"model_obj.model.predict(transformed): {e2}")

    if preds is None:
        raise RuntimeError("dice_predict: unable to get predictions. Trace:\n- " + "\n- ".join(err_msgs))

    preds = np.asarray(preds)
    if preds.ndim == 1:
        prob = preds
    elif preds.ndim == 2 and preds.shape[1] == 1:
        prob = preds[:, 0]
    elif preds.ndim == 2 and preds.shape[1] == 2:
        prob = preds[:, 1]
    else:
        prob = 1.0 / (1.0 + np.exp(-preds[:, -1]))
    labels = (prob >= 0.5).astype(int)
    return labels, prob

def l1_l2(f, c, num_cols, scaler):
    fx = np.array([float(f[k]) for k in num_cols], float).reshape(1,-1)
    cx = np.array([float(c[k]) for k in num_cols], float).reshape(1,-1)
    if scaler is not None:
        fx = scaler.transform(fx); cx = scaler.transform(cx)
    l1 = float(np.abs(fx - cx).sum())
    l2 = float(np.sqrt(((fx - cx) ** 2).sum()))
    return l1, l2

def gower_row(f, c, num_cols, cat_cols, ranges):
    parts = []
    for k in num_cols:
        lo, hi = ranges[k]; denom = (hi - lo) if hi > lo else 1.0
        parts.append(abs(float(f[k]) - float(c[k])) / denom)
    for k in cat_cols:
        parts.append(0.0 if f[k] == c[k] else 1.0)
    return float(np.mean(parts)) if parts else np.nan

def sparsity_count(f, c, all_cols):
    return int(sum(1 for k in all_cols if f.get(k) != c.get(k)))

def feasibility_ok(f, c, immutable, directional):
    for k in immutable:
        if k in f and k in c and f[k] != c[k]:
            return False
    for k, d in directional.items():
        if k in f and k in c:
            try:
                fv, cv = float(f[k]), float(c[k])
            except Exception:
                continue
            if d == "up" and cv < fv: return False
            if d == "down" and cv > fv: return False
    return True

def knn_plausibility(train_df, point_df, numeric_cols, k=10):

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.neighbors import NearestNeighbors
    if not numeric_cols:
        return np.nan
    scaler = MinMaxScaler()
    X = scaler.fit_transform(train_df[numeric_cols])
    x = scaler.transform(point_df[numeric_cols])
    nbrs = NearestNeighbors(n_neighbors=min(k, len(train_df))).fit(X)
    dists, _ = nbrs.kneighbors(x)
    return float(1.0 / (1.0 + dists.mean()))

try:
    num_cols = list(exp.data_interface.continuous_feature_names)
    cat_cols = list(exp.data_interface.categorical_feature_names)
    all_feats = num_cols + cat_cols
except Exception:
    feature_cols = [c for c in FEATURE_ORDER if c != OUTCOME_NAME]

    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_dataset[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]
    all_feats = feature_cols

# sanity check: every feature we’ll use is present
missing = [f for f in all_feats if f not in train_dataset.columns]
if missing:
    raise ValueError(f"These features aren't in train_dataset: {missing}")

# numeric ranges & scaler for distances
from sklearn.preprocessing import MinMaxScaler
ranges = {k: (pd.to_numeric(train_dataset[k], errors="coerce").min(),
              pd.to_numeric(train_dataset[k], errors="coerce").max())
          for k in num_cols}
num_scaler = MinMaxScaler().fit(train_dataset[num_cols])

# ---- load methods' JSONs ----
rag_cfe_pairs = load_json_dir(rag_cfe_dir, FEATURE_ORDER, OUTCOME_NAME, test_dataset, batch_index=batch_index)
dice_cfe_pairs = load_json_dir(dice_cfe_dir, FEATURE_ORDER, OUTCOME_NAME, test_dataset, batch_index=batch_index)

def evaluate_pairs(model_obj, data_interface, pairs, method_name, reference_df):
    out = []
    for item in pairs:
        fid = item["id"]; f = item["factual"]; cfs = item["cfs"]

        f_df  = pd.DataFrame([f])[all_feats]
        cf_df = pd.DataFrame(cfs)[all_feats]

        # Predict CFs with DiCE model wrapper
        cf_labels, _ = dice_predict(model_obj, cf_df, data_interface, reference_df=reference_df)
        validity = float((cf_labels == POSITIVE_LABEL).mean())

        L1s, L2s, Gs, Ss, Fs, Ps = [], [], [], [], [], []
        for cf in cfs:
            L1, L2 = l1_l2(f, cf, num_cols, num_scaler)
            G = gower_row(f, cf, num_cols, cat_cols, ranges)
            S = sparsity_count(f, cf, all_feats)
            F = feasibility_ok(f, cf, IMMUTABLE, DIRECTIONAL)
            P = knn_plausibility(train_dataset, pd.DataFrame([cf]), num_cols, k=10)
            L1s.append(L1); L2s.append(L2); Gs.append(G); Ss.append(S); Fs.append(F); Ps.append(P)

        out.append({
            "method": method_name, "fid": fid,
            "validity": validity,
            "L1": float(np.nanmean(L1s)),
            "L2": float(np.nanmean(L2s)),
            "Gower": float(np.nanmean(Gs)),
            "sparsity": float(np.nanmean(Ss)),
            "feasibility": float(np.mean(Fs)),
            "plausibility": float(np.nanmean(Ps)),
        })
    return pd.DataFrame(out)

df_rag_cfe = evaluate_pairs(m, d, rag_cfe_pairs, "rag_cfe", reference_df=train_dataset[all_feats])
df_dice_cfe = evaluate_pairs(m, d, dice_cfe_pairs, "dice_cfe", reference_df=train_dataset[all_feats])
per_factual = pd.concat([df_rag_cfe, df_dice_cfe], ignore_index=True).sort_values(["method","fid"])

summary = (
    per_factual.drop(columns=["fid"])
    .groupby("method")
    .mean(numeric_only=True)
    .reset_index()
)

print("\n=== Aggregate summary ===")
print(summary.to_string(index=False))