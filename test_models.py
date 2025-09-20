from pathlib import Path
import joblib
import pandas as pd
import numpy as np

API_DIR = Path("api")  # ajuste se necessário

# ---- exemplo de payload real (edite como quiser) ----
payload = {
    "marca": "Volkswagen",
    "modelo": "Gol 1.0",
    "anoModelo": 2018,
    "mesReferencia": 7,
    "anoReferencia": 2020,
}

def predict_with_artifact(pkl_path: Path, payload: dict) -> float:
    obj = joblib.load(pkl_path)

    # CASO A: é um Pipeline sklearn (tem .predict direto)
    if hasattr(obj, "predict"):
        X = pd.DataFrame([payload], columns=["marca","modelo","anoModelo","mesReferencia","anoReferencia"])
        yhat = obj.predict(X)[0]
        return float(yhat)

    # CASO B: é o formato "dict" que salvamos no notebook 04 (árvores)
    if isinstance(obj, dict) and "model" in obj and "encoder" in obj:
        model = obj["model"]
        enc = obj["encoder"]
        cat_cols = obj["cat_cols"]
        num_cols = obj["num_cols"]

        # montar X exatamente como no treino
        X_df = pd.DataFrame([payload])
        X_cat_raw = X_df[cat_cols].astype(str).fillna("__NA__")
        X_num = X_df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

        # transformar com o encoder salvo e empilhar
        X_cat = enc.transform(X_cat_raw)
        X_all = np.hstack([X_cat.values, X_num.values])
        yhat = model.predict(X_all)[0]
        return float(yhat)

    raise ValueError(f"Formato de artefato não reconhecido: {pkl_path.name}")

# --- 1) Testar UM arquivo específico ---
one_file = API_DIR / "Ridge.pkl"            # troque para qualquer outro .pkl
print(one_file.name, "=>", predict_with_artifact(one_file, payload))

# --- 2) Ou testar TODOS os .pkl da pasta ---
for pkl in sorted(API_DIR.glob("*.pkl")):
    try:
        pred = predict_with_artifact(pkl, payload)
        print(f"{pkl.name:20s} => R$ {pred:,.2f}")
    except Exception as e:
        print(f"{pkl.name:20s} => ERRO: {e}")
