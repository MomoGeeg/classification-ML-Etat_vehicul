import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st


# Configuration de la page
st.set_page_config(
    page_title="Prédiction de l'état d'un véhicule",
    page_icon="🚗",
    layout="centered",
)

DESCRIPTION = (
    "Ce modèle de machine learning permet de prédire l'état d'un véhicule en partant "
    "de la marque, de l'année, de la transmission, du prix et du quartier."
)


# Chargement des artefacts (mis en cache : chargés une seule fois)

@st.cache_resource
def load_artifacts():
    encoders = jb.load("encoders.joblib")   # encodeurs (Marque, Transmission, Quartier, Etat)
    scaler = jb.load("scaler.joblib")       # normaliseur
    gb = jb.load("gb_model.joblib")         # modèle
    return encoders, scaler, gb


encoders, scaler, gb = load_artifacts()
target_class = encoders['Etat'].classes_  # noms des classes cibles


# Fonction de prédiction simple
def pred_func(marque, annee, transmission, prix, quartier):
    quartier = quartier.strip().title()

    transmission_encoded = encoders['Transmission'].transform([transmission])[0]
    marque_encoded = encoders['Marque'].transform([marque])[0]
    quartier_encoded = encoders['Quartier'].transform([quartier])[0]

    colonnes_modele = ['Marque', 'Année', 'Transmission', 'Prix', 'Quartier']

    df_complet = pd.DataFrame([[
        marque_encoded, annee, transmission_encoded, prix, quartier_encoded
    ]], columns=colonnes_modele)

    # Scaler TOUTES les colonnes, comme à l'entraînement (x1 = scaler.fit_transform(X))
    X_input = pd.DataFrame(scaler.transform(df_complet), columns=colonnes_modele)

    prediction = gb.predict(X_input.values)[0]
    classe_predite = target_class[prediction]

    return classe_predite


# Fonction de prédiction multiple

def pred_func_csv(file):
    df = pd.read_csv(file)
    prediction = []

    for i, row in enumerate(df.iloc[:, :].values):
        try:
            y_pred = pred_func(row[0], row[1], row[2], row[3], row[4])
            prediction.append(y_pred)
        except Exception as e:
            print(f"Erreur ligne {i} : {row} → {e}")
            prediction.append(None)

    df['Etat'] = prediction
    return df


# Interface

st.title("🚗 Prédiction de l'état d'un véhicule")

onglet1, onglet2 = st.tabs(["Prédiction simple", "Prédiction multiple"])

# ----------------------------- Onglet 1 -------------------------------
with onglet1:
    st.subheader("Prédire l'état d'un véhicule avec une entrée")
    st.write(DESCRIPTION)

    with st.form("formulaire_simple"):
        col1, col2 = st.columns(2)
        with col1:
            marque = st.selectbox("Marque", options=list(encoders['Marque'].classes_))
            annee = st.number_input("Année", value=2020, step=1)
            transmission = st.selectbox("Transmission", options=list(encoders['Transmission'].classes_))
        with col2:
            prix = st.number_input("Prix", value=0.0, step=1000.0, format="%.2f")
            quartier = st.selectbox("Quartier", options=list(encoders['Quartier'].classes_))

        soumettre = st.form_submit_button("Prédire", type="primary")

    if soumettre:
        try:
            resultat = pred_func(marque, annee, transmission, prix, quartier)
            st.success(f"**État du véhicule :** {resultat}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# ----------------------------- Onglet 2 -------------------------------
with onglet2:
    st.subheader("Prédire l'état d'un véhicule avec plusieurs entrées")
    st.write(DESCRIPTION)
    st.caption(
        "Le fichier CSV doit contenir, dans cet ordre, les colonnes : "
        "Marque, Année, Transmission, Prix, Quartier."
    )

    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if fichier is not None:
        try:
            with st.spinner("Prédictions en cours…"):
                df_resultat = pred_func_csv(fichier)

            st.success(f"{len(df_resultat)} prédiction(s) effectuée(s).")
            st.dataframe(df_resultat, use_container_width=True)

            st.download_button(
                label="⬇️ Télécharger le fichier CSV",
                data=df_resultat.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")