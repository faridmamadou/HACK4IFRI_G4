import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import time

# Configuration de la page
st.set_page_config(
    page_title="Interface ML Model",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de l'API
API_URL = "https://hack4ifrig4-production.up.railway.app/"  # Modifiez selon votre configuration

# Fonctions utilitaires
def api_request(endpoint, method="GET", data=None):
    """Effectue une requête à l'API"""
    url = f"{API_URL}/{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Erreur lors de la connexion à l'API: {str(e)}")
        return None

def check_api_health():
    """Vérifie si l'API est accessible et si le modèle est chargé"""
    try:
        health = api_request("health")
        if health and health.get("status") == "healthy":
            return True
        return False
    except:
        return False

def get_model_info():
    """Récupère les informations sur le modèle"""
    return api_request("model-info")

def make_prediction(features):
    """Envoie une requête de Classification à l'API"""
    data = {"features": features}
    return api_request("predict", method="POST", data=data)

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Sélectionnez une page",
    ["Accueil", "Classification"]
)

# Vérification de la connexion à l'API
api_status = check_api_health()
model_info = get_model_info() if api_status else None

# Contenu de la page d'accueil
if page == "Accueil":
    st.title("Interface du Modèle de Machine Learning")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Bienvenue sur l'interface du modèle
        
        Cette application permet d'interagir avec un modèle de machine learning via une API FastAPI.
        
        ### Fonctionnalités disponibles:
        - 🔮 Faire des classifications en temps réel
        - 📊 Visualiser les résultats
        
        Utilisez la barre latérale pour naviguer entre les différentes pages.
        """)
    
    with col2:
        st.subheader("État du service")
        if api_status:
            st.success("✅ API connectée")
            st.success("✅ Modèle chargé")
            if model_info:
                st.info(f"Type de modèle: {model_info.get('type', 'Non spécifié')}")
                st.info(f"Nombre de fonctionnalités: {model_info.get('n_features', 'Non spécifié')}")
                if 'classes' in model_info:
                    st.info(f"Classes: {', '.join(map(str, model_info.get('classes', [])))}")
        else:
            st.error("❌ API non connectée")
            st.warning("Vérifiez que l'API FastAPI est en cours d'exécution.")
            st.code("uvicorn main:app --reload")
    
    st.markdown("---")
    
    st.subheader("Architecture du système")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Architecture Back-End (FastAPI)
        
        ```
        ┌─────────────────┐
        │   Client HTTP   │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │   API FastAPI   │
        │    (main.py)    │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │ Modèle ML (pkl) │
        └─────────────────┘
        ```
        """)
    
    with col2:
        st.markdown("""
        ### Architecture Front-End (Streamlit)
        
        ```
        ┌─────────────────┐
        │  App Streamlit  │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │ Requêtes HTTP   │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │   API FastAPI   │
        └─────────────────┘
        ```
        """)

# Page de Classification
elif page == "Classification":
    st.title("Classification avec le modèle")
    
    if not api_status:
        st.error("❌ API non connectée. Impossible de faire des classifications.")
        st.stop()
    
    # Récupération d'informations sur le modèle
    n_features = model_info.get('n_features', 4) if model_info else 4
    feature_names = model_info.get('features_expected', [f"Feature {i+1}" for i in range(n_features)]) if model_info else [f"Feature {i+1}" for i in range(n_features)]
    
    st.subheader("Entrez les caractéristiques")
    
    # Méthode d'entrée
    input_method = st.radio(
        "Méthode d'entrée des données",
        ["Formulaire", "CSV", "Exemple prédéfini"]
    )
    
    features = []
    uploaded_df = None
    
    if input_method == "Formulaire":
        # Création d'un formulaire pour entrer les caractéristiques
        cols = st.columns(min(4, n_features))
        features = []
        
        for i in range(n_features):
            with cols[i % min(4, n_features)]:
                feat_value = st.number_input(
                    label=feature_names[i] if i < len(feature_names) else f"Feature {i+1}",
                    value=0.0,
                    format="%.2f",
                    key=f"feature_{i}"
                )
                features.append(feat_value)
        
        if st.button("Prédire", key="predict_form"):
            with st.spinner('Classification en cours...'):
                result = make_prediction(features)
                
                if result:
                    st.success("Classification réussie!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Classification", value=result.get('classe'))
                    
                    if result.get('probability') is not None:
                        with col2:
                            st.subheader("Probabilités")
                            probs = result.get('probability')
                            
                            if isinstance(probs, list):
                                if 'classes' in model_info:
                                    classes = model_info['classes']
                                    prob_df = pd.DataFrame({
                                        'Classe': [str(c) for c in classes],
                                        'Probabilité': probs
                                    })
                                else:
                                    prob_df = pd.DataFrame({
                                        'Classe': [f"Classe {i}" for i in range(len(probs))],
                                        'Probabilité': probs
                                    })
                                
                                fig = px.bar(prob_df, x='Classe', y='Probabilité', title="Probabilités par classe")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.write(f"Probabilité: {probs}")
    
    elif input_method == "CSV":
        st.info("Téléchargez un fichier CSV contenant les caractéristiques")
        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
        
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.write("Aperçu des données:")
                st.dataframe(uploaded_df.head())
                
                # Vérification du nombre de colonnes
                if uploaded_df.shape[1] != n_features:
                    st.warning(f"Attention: Le modèle s'attend à {n_features} caractéristiques, mais le CSV en contient {uploaded_df.shape[1]}.")
                
                if st.button("Prédire le premier exemple", key="predict_csv_first"):
                    with st.spinner('Classification en cours...'):
                        features = uploaded_df.iloc[0].values.tolist()
                        result = make_prediction(features)
                        
                        if result:
                            st.success("Classification réussie!")
                            st.metric("Classification", value=result.get('prediction'))
                
                # Option pour prédire tous les exemples
                if st.button("Prédire tous les exemples", key="predict_csv_all"):
                    with st.spinner('Classifications en cours...'):
                        progress_bar = st.progress(0)
                        results = []
                        
                        # Limite à 100 exemples pour éviter une surcharge
                        sample_size = min(100, len(uploaded_df))
                        sampled_df = uploaded_df.head(sample_size)
                        
                        for i, row in enumerate(sampled_df.itertuples()):
                            features = list(row)[1:]  # Exclure l'index
                            result = make_prediction(features)
                            if result:
                                results.append(result.get('prediction'))
                            else:
                                results.append(None)
                            progress_bar.progress((i + 1) / sample_size)
                            
                        # Affichage des résultats
                        result_df = sampled_df.copy()
                        result_df['Classification'] = results
                        
                        st.success(f"{len(results)} Classifications effectuées!")
                        st.dataframe(result_df)
                        
                        # Visualisation des résultats
                        if len(results) > 0 and all(r is not None for r in results):
                            st.subheader("Distribution des Classifications")
                            fig = px.histogram(result_df, x='Classification', title="Distribution des Classifications")
                            st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Erreur lors du traitement du fichier CSV: {str(e)}")
    
    elif input_method == "Exemple prédéfini":
        # Exemples prédéfinis (à adapter selon votre modèle)
        example_sets = {
            "Exemple 1": [5.1, 3.5, 1.4, 0.2],
            "Exemple 2": [6.7, 3.1, 4.4, 1.4],
            "Exemple 3": [6.3, 2.8, 5.1, 1.5]
        }
        
        # Ajuster le nombre d'exemples en fonction du nombre de caractéristiques
        for key in list(example_sets.keys()):
            if len(example_sets[key]) != n_features:
                example_sets[key] = example_sets[key][:n_features] if len(example_sets[key]) > n_features else example_sets[key] + [0.0] * (n_features - len(example_sets[key]))
        
        selected_example = st.selectbox("Sélectionnez un exemple", list(example_sets.keys()))
        
        features = example_sets[selected_example]
        
        # Affichage des caractéristiques de l'exemple
        example_df = pd.DataFrame([features], columns=feature_names[:n_features])
        st.dataframe(example_df)
        
        if st.button("Classer", key="predict_example"):
            with st.spinner('Classification en cours...'):
                result = make_prediction(features)
                
                if result:
                    st.success("Classification réussie!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Classification", value=result.get('prediction'))
                    
                    if result.get('probability') is not None:
                        with col2:
                            st.subheader("Probabilités")
                            probs = result.get('probability')
                            
                            if isinstance(probs, list):
                                if 'classes' in model_info:
                                    classes = model_info['classes']
                                    prob_df = pd.DataFrame({
                                        'Classe': [str(c) for c in classes],
                                        'Probabilité': probs
                                    })
                                else:
                                    prob_df = pd.DataFrame({
                                        'Classe': [f"Classe {i}" for i in range(len(probs))],
                                        'Probabilité': probs
                                    })
                                
                                fig = px.bar(prob_df, x='Classe', y='Probabilité', title="Probabilités par classe")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.write(f"Probabilité: {probs}")

# Footer
st.markdown("---")
st.markdown("Développé pour le HACK4IFRFI par le Groupe 4")