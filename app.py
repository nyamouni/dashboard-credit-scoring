import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
import pickle
import plotly.graph_objects as go
import shap

# -----------------------------
# CONFIG APP
# -----------------------------
st.set_page_config(page_title="Dashboard Crédit Scoring", layout="wide")
st.title("Dashboard Crédit Scoring")
st.markdown("Bienvenue sur l’outil de prédiction et d’explication des décisions d’octroi de crédit.")

# -----------------------------
# DONNÉES DE RÉFÉRENCE
# -----------------------------
@st.cache_data
def load_reference_data():
    url = "https://nrdnsniperbot.site/application_train.csv"  

    response = requests.get(url)
    if response.status_code != 200:
        st.error("Erreur lors du téléchargement du fichier CSV.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")
        return pd.DataFrame()

    if "CODE_GENDER" not in df.columns:
        st.write("Colonnes du fichier chargé :", df.columns.tolist())
        st.stop()

    df["APP_CODE_GENDER"] = df["CODE_GENDER"].map({"F": 0, "M": 1})
    df["APP_FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].map({"N": 0, "Y": 1})
    df["APP_FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].map({"N": 0, "Y": 1})

    df.rename(columns={
        "AMT_INCOME_TOTAL": "APP_AMT_INCOME_TOTAL",
        "AMT_CREDIT": "APP_AMT_CREDIT",
        "EXT_SOURCE_2": "APP_EXT_SOURCE_2",
        "EXT_SOURCE_3": "APP_EXT_SOURCE_3"
    }, inplace=True)

    return df.sample(frac=0.25, random_state=42).reset_index(drop=True)

df_ref = load_reference_data()

# -----------------------------
# CHARGEMENT DU MODÈLE
# -----------------------------
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
features_used = list(model.feature_names_in_)

# -----------------------------
# SAISIE CLIENT
# -----------------------------
st.sidebar.subheader("Méthode de sélection du client")
client_mode = st.sidebar.radio("Choisir un mode :", ["Saisie manuelle", "Client existant", "Client aléatoire"])

if client_mode == "Client existant":
    selected_id = st.sidebar.selectbox("Choisir l'ID du client", df_ref.index)
    selected_client = df_ref.loc[selected_id]
elif client_mode == "Client aléatoire":
    selected_client = df_ref.sample(1).iloc[0]
else:
    selected_client = None

# -----------------------------
# FORMULAIRE CLIENT
# -----------------------------
if selected_client is not None:
    gender = int(selected_client["APP_CODE_GENDER"])
    own_car = int(selected_client["APP_FLAG_OWN_CAR"])
    own_realty = int(selected_client["APP_FLAG_OWN_REALTY"])
    income_total = int(selected_client["APP_AMT_INCOME_TOTAL"])
    credit_amt = int(selected_client["APP_AMT_CREDIT"])
    ext_source_2 = float(selected_client["APP_EXT_SOURCE_2"])
    ext_source_3 = float(selected_client["APP_EXT_SOURCE_3"])
else:
    gender = 0
    own_car = 0
    own_realty = 0
    income_total = 50000
    credit_amt = 200000
    ext_source_2 = 0.5
    ext_source_3 = 0.5

gender = st.sidebar.selectbox("Sexe", [0, 1], index=gender, format_func=lambda x: "Femme" if x == 0 else "Homme")
own_car = st.sidebar.selectbox("Possède une voiture ?", [0, 1], index=own_car)
own_realty = st.sidebar.selectbox("Possède un bien immobilier ?", [0, 1], index=own_realty)
income_total = st.sidebar.number_input("Revenu total (€)", min_value=0, value=income_total)
credit_amt = st.sidebar.number_input("Montant du crédit (€)", min_value=0, value=credit_amt)
education = st.sidebar.selectbox("Niveau d'éducation", [
    "Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary"])
income_type = st.sidebar.selectbox("Type de revenu", [
    "Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"])
family_status = st.sidebar.selectbox("Statut familial", [
    "Married", "Single / not married", "Civil marriage", "Widow"])
house_type = st.sidebar.selectbox("Type de logement", [
    "House / apartment", "Municipal apartment", "Rented apartment", "Office apartment", "With parents"])
ext_source_2 = st.sidebar.slider("EXT_SOURCE_2", 0.0, 1.0, ext_source_2)
ext_source_3 = st.sidebar.slider("EXT_SOURCE_3", 0.0, 1.0, ext_source_3)

input_data = {
    "APP_CODE_GENDER": gender,
    "APP_FLAG_OWN_CAR": own_car,
    "APP_FLAG_OWN_REALTY": own_realty,
    "APP_AMT_INCOME_TOTAL": income_total,
    "APP_AMT_CREDIT": credit_amt,
    "APP_NAME_EDUCATION_TYPE": education,
    "APP_NAME_INCOME_TYPE": income_type,
    "APP_NAME_FAMILY_STATUS": family_status,
    "APP_HOUSETYPE_MODE": house_type,
    "APP_EXT_SOURCE_2": ext_source_2,
    "APP_EXT_SOURCE_3": ext_source_3
}

# -----------------------------
# PRÉTRAITEMENT
# -----------------------------
def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])
    for col in features_used:
        if col not in df.columns:
            df[col] = 0
    return df[features_used]

# -----------------------------
# PREDICTION
# -----------------------------
st.subheader("Résultat du score")

if st.sidebar.button("Prédire"):
    import numpy as np

    # Nettoyage des NaN, inf pour compatibilité JSON
    def clean_json_compatible(data):
        return {k: (0 if (pd.isna(v) or v in [np.inf, -np.inf]) else v) for k, v in data.items()}

    input_data_clean = clean_json_compatible(input_data)

    response = requests.post("https://credit-scoring-api-s00s.onrender.com/predict", json=input_data_clean)

    if response.status_code == 200:
        result = response.json()
        prediction = result["prediction"]
        proba = result["probability"]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Score de probabilité", f"{proba:.2f}", "Accepté" if prediction == 0 else "Refusé")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=proba,
                delta={'reference': 0.345, 'increasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [0, 1]},
                    'steps': [
                        {'range': [0, 0.345], 'color': "lightgreen"},
                        {'range': [0.345, 1], 'color': "salmon"}
                    ],
                    'bar': {'color': "darkblue"}
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Détails client")
            st.dataframe(pd.DataFrame.from_dict(input_data, orient="index", columns=["Valeur"]))

        # ---------------- SHAP LOCAL ----------------
        st.subheader("Explication du score (SHAP)")
        X_input = preprocess_input(input_data)
        explainer = shap.Explainer(model)
        shap_values = explainer(X_input)
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(plt.gcf())
        plt.clf()

        # ---------------- ONGLET VISUALISATION ----------------
        tab1, tab2, tab3 = st.tabs(["Analyse univariée", " Analyse bi-variée", "SHAP global"])

        numerical_cols = ["APP_AMT_INCOME_TOTAL", "APP_AMT_CREDIT", "APP_EXT_SOURCE_2", "APP_EXT_SOURCE_3"]

        # --- Onglet 1 : Univariée
        with tab1:
            selected_feature = st.selectbox("Choisissez une variable à comparer", numerical_cols)
            fig_comp = px.histogram(df_ref, x=selected_feature, nbins=50)
            fig_comp.add_vline(x=input_data[selected_feature], line_color="red", line_width=3, line_dash="dash")
            st.plotly_chart(fig_comp, use_container_width=True)

            colbox, colviolin = st.columns(2)
            with colbox:
                fig_box = px.box(df_ref, y=selected_feature, points="all")
                fig_box.add_scatter(y=[input_data[selected_feature]], mode="markers", marker=dict(color="red", size=10), name="Client")
                st.plotly_chart(fig_box, use_container_width=True)
            with colviolin:
                fig_violin = px.violin(df_ref, y=selected_feature, box=True, points="all")
                fig_violin.add_scatter(y=[input_data[selected_feature]], mode="markers", marker=dict(color="red", size=10), name="Client")
                st.plotly_chart(fig_violin, use_container_width=True)

        # --- Onglet 2 : Bi-variée
        with tab2:
            x_var = st.selectbox("Variable X", numerical_cols, key="x")
            y_var = st.selectbox("Variable Y", numerical_cols, key="y")
            fig_scatter = px.scatter(df_ref, x=x_var, y=y_var, opacity=0.5)
            fig_scatter.add_scatter(x=[input_data[x_var]], y=[input_data[y_var]], mode="markers",
                                    marker=dict(size=12, color="red"), name="Client")
            st.plotly_chart(fig_scatter, use_container_width=True)

        # --- Onglet 3 : SHAP global
        with tab3:
            try:
                st.markdown("Calculé sur 200 clients aléatoires de l’échantillon.")

                sample_df = df_ref.sample(200, random_state=42).copy()

                # S’assurer que toutes les colonnes sont présentes pour le modèle
                for col in features_used:
                    if col not in sample_df.columns:
                        sample_df[col] = 0

                sample_df = sample_df[features_used]

                explainer_global = shap.Explainer(model)
                shap_values_global = explainer_global(sample_df)

                shap.plots.bar(shap_values_global, max_display=10, show=False)
                st.pyplot(plt.gcf())
                plt.clf()

            except Exception as e:
                st.error(f"Erreur SHAP global : {e}")

    else:
        st.error("Erreur lors de la requête à l’API.")


