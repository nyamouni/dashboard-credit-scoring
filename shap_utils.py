import pickle
import pandas as pd
import shap

# Charger le modèle (assure-toi que le chemin est correct depuis ton app Streamlit)
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Liste exacte des features utilisées par le modèle
FEATURES = list(model.feature_names_in_)

# Fonction de preprocessing simplifié
def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])
    
    # S'assurer qu'on a bien toutes les colonnes attendues par le modèle
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0  # Valeur par défaut

    df = df[FEATURES]  # Réordonner
    return df

# Fonction de prédiction + SHAP
def explain_prediction(input_dict):
    X = preprocess_input(input_dict)

    # Création de l'explainer SHAP (TreeExplainer pour LightGBM)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    return shap_values, X
