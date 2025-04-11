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
    ["Accueil", "Classification", "Rapport du projet"]
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
        - 📝 Consulter le rapport du projet
        
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

# Page du rapport
elif page == "Rapport du projet":
    st.title("Rapport du projet")
    
    # Sections du rapport
    st.sidebar.markdown("## Sections du rapport")
    report_section = st.sidebar.radio(
        "Sélectionnez une section",
        ["Introduction", "Préprocessing", "Modélisation", "Évaluation", "Conclusion"]
    )
    
    # Contenu de chaque section
    if report_section == "Introduction":
        st.header("Introduction")
        
        st.markdown("""
        ## Contexte du projet
        
        Ce projet vise à développer un modèle de machine learning pour résoudre le problème de [décrire le problème].
        
        ### Objectifs
        
        - Objectif 1: Analyser les données disponibles
        - Objectif 2: Développer un modèle prédictif performant
        - Objectif 3: Déployer le modèle via une API pour une utilisation en production
        
        ### Données utilisées
        
        Les données proviennent de [source des données] et contiennent [nombre] observations avec [nombre] caractéristiques.
        
        Le jeu de données comprend des informations sur [description des données].
        """)
        
        # Graphique fictif pour illustrer
        st.subheader("Aperçu des données")
        
        # Créer des données fictives pour démonstration
        np.random.seed(42)
        fake_data = pd.DataFrame({
            'Feature 1': np.random.normal(0, 1, 100),
            'Feature 2': np.random.normal(5, 2, 100),
            'Target': np.random.choice(['Class A', 'Class B', 'Class C'], 100)
        })
        
        fig = px.scatter(fake_data, x='Feature 1', y='Feature 2', color='Target', title="Visualisation des données")
        st.plotly_chart(fig, use_container_width=True)
    
    elif report_section == "Préprocessing":
        st.header("Préprocessing des données")
        
        st.markdown("""
        ## Étapes de prétraitement
        
        Le préprocessing des données a impliqué plusieurs étapes essentielles pour préparer les données à l'entraînement du modèle:
        
        ### 1. Nettoyage des données
        
        - Gestion des valeurs manquantes
        - Suppression des doublons
        - Correction des erreurs et anomalies
        
        ### 2. Feature Engineering
        
        - Création de nouvelles caractéristiques
        - Transformation de variables existantes
        - Encodage des variables catégorielles
        
        ### 3. Normalisation et standardisation
        
        - Standardisation des caractéristiques numériques
        - Normalisation des distributions
        """)
        
        # Exemple de code pour le préprocessing
        st.subheader("Exemple de code de préprocessing")
        st.code("""
        # Chargement des données
        import pandas as pd
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        
        # Définition des transformateurs
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Préprocesseur combiné
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Préparation des données
        X_train_processed = preprocessor.fit_transform(X_train)
        """)
        
        # Visualisation du préprocessing
        st.subheader("Impact du préprocessing")
        
        # Données fictives pour démonstration
        np.random.seed(42)
        before_data = pd.DataFrame({
            'Feature': np.random.normal(0, 2, 100)
        })
        
        after_data = pd.DataFrame({
            'Feature': (before_data['Feature'] - before_data['Feature'].mean()) / before_data['Feature'].std()
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.histogram(before_data, x='Feature', title="Avant standardisation")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.histogram(after_data, x='Feature', title="Après standardisation")
            st.plotly_chart(fig2, use_container_width=True)
    
    elif report_section == "Modélisation":
        st.header("Modélisation")
        
        st.markdown("""
        ## Approche de modélisation
        
        ### Modèles évalués
        
        Nous avons testé plusieurs modèles pour résoudre ce problème:
        
        1. **Random Forest**: Un ensemble d'arbres de décision qui améliore la précision et réduit le surapprentissage
        2. **Gradient Boosting**: Une méthode d'ensemble qui construit progressivement des modèles
        3. **Support Vector Machine**: Efficace pour les problèmes à haute dimension
        4. **Neural Network**: Un réseau de neurones multicouche pour capturer des relations complexes
        
        ### Méthodologie d'entraînement
        
        - Validation croisée à k-plis (k=5)
        - Recherche d'hyperparamètres via GridSearchCV
        - Échantillonnage stratifié pour gérer les déséquilibres de classe
        """)
        
        # Visualisation de la performance des modèles
        st.subheader("Comparaison des modèles")
        
        # Données fictives pour démonstration
        model_data = pd.DataFrame({
            'Modèle': ['Random Forest', 'Gradient Boosting', 'SVM', 'Neural Network'],
            'Précision': [0.92, 0.94, 0.88, 0.91],
            'Rappel': [0.90, 0.92, 0.85, 0.89],
            'F1-Score': [0.91, 0.93, 0.86, 0.90]
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=model_data['Modèle'],
            y=model_data['Précision'],
            name='Précision',
            marker_color='royalblue'
        ))
        
        fig.add_trace(go.Bar(
            x=model_data['Modèle'],
            y=model_data['Rappel'],
            name='Rappel',
            marker_color='indianred'
        ))
        
        fig.add_trace(go.Bar(
            x=model_data['Modèle'],
            y=model_data['F1-Score'],
            name='F1-Score',
            marker_color='green'
        ))
        
        fig.update_layout(
            title='Performance des différents modèles',
            xaxis_title='Modèle',
            yaxis_title='Score',
            barmode='group',
            yaxis=dict(range=[0.8, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Code d'entraînement
        st.subheader("Exemple de code d'entraînement")
        st.code("""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV, cross_val_score
        
        # Définition du modèle
        rf = RandomForestClassifier(random_state=42)
        
        # Hyperparamètres à tester
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Recherche d'hyperparamètres
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Entraînement
        grid_search.fit(X_train, y_train)
        
        # Meilleur modèle
        best_model = grid_search.best_estimator_
        
        # Enregistrement du modèle
        import joblib
        joblib.dump(best_model, 'model.pkl')
        """)
    
    elif report_section == "Évaluation":
        st.header("Évaluation du modèle")
        
        st.markdown("""
        ## Résultats de l'évaluation
        
        ### Métriques de performance
        
        Notre modèle final a été évalué sur un ensemble de test indépendant avec les résultats suivants:
        
        - **Précision**: 0.94
        - **Rappel**: 0.92
        - **F1-Score**: 0.93
        - **AUC-ROC**: 0.97
        
        ### Analyse des erreurs
        
        Une analyse des erreurs a révélé que le modèle avait tendance à confondre certaines classes dans des cas spécifiques.
        """)
        
        # Matrice de confusion
        st.subheader("Matrice de confusion")
        
        # Données fictives pour démonstration
        cm = np.array([
            [45, 3, 2],
            [1, 50, 4],
            [2, 3, 40]
        ])
        
        classes = ['Classe A', 'Classe B', 'Classe C']
        
        fig = px.imshow(
            cm,
            x=classes,
            y=classes,
            color_continuous_scale='Blues',
            labels=dict(x="Classification", y="Réalité", color="Nombre")
        )
        
        fig.update_layout(
            title='Matrice de confusion',
            xaxis_title='Classe prédite',
            yaxis_title='Classe réelle'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Courbe ROC
        st.subheader("Courbe ROC")
        
        # Données fictives pour la courbe ROC
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - np.exp(-5 * fpr)
        
        fig = px.line(
            x=fpr, y=tpr,
            labels={'x': 'Taux de faux positifs', 'y': 'Taux de vrais positifs'},
            title='Courbe ROC'
        )
        
        fig.add_shape(
            type='line',
            line=dict(dash='dash', width=1),
            x0=0, x1=1, y0=0, y1=1
        )
        
        fig.update_layout(
            xaxis_range=[0, 1],
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.subheader("Importance des caractéristiques")
        
        # Données fictives pour l'importance des caractéristiques
        feature_importance = pd.DataFrame({
            'Feature': ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'],
            'Importance': [0.35, 0.25, 0.20, 0.15, 0.05]
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Importance des caractéristiques'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif report_section == "Conclusion":
        st.header("Conclusion et perspectives")
        
        st.markdown("""
        ## Conclusion
        
        Ce projet nous a permis de développer un modèle de machine learning performant pour [décrire le problème]. 
        
        Nous avons obtenu d'excellents résultats avec un modèle de Gradient Boosting, atteignant une précision de 94% sur l'ensemble de test.
        
        ### Principaux apprentissages
        
        1. L'importance du préprocessing des données pour obtenir des performances optimales
        2. L'efficacité des méthodes d'ensemble pour ce type de problème
        3. La nécessité d'une validation croisée rigoureuse pour éviter le surapprentissage
        
        ## Perspectives
        
        Pour améliorer davantage le modèle, nous pourrions:
        
        - Collecter davantage de données pour les classes minoritaires
        - Explorer des architectures de deep learning plus complexes
        - Intégrer des données externes supplémentaires
        - Mettre en place un système de monitoring pour détecter la dérive conceptuelle
        
        ## Déploiement
        
        Le modèle a été déployé avec succès via une API FastAPI, permettant son intégration facile dans différentes applications.
        """)
        
        # Visualisation du workflow complet
        st.subheader("Workflow complet du projet")
        
        workflow_code = """
        graph TD
          A[Collecte de données] --> B[Préprocessing]
          B --> C[Feature Engineering]
          C --> D[Entraînement du modèle]
          D --> E[Évaluation]
          E --> F[Déploiement API]
          F --> G[Interface utilisateur]
          E --> H{Performance suffisante?}
          H -->|Non| B
          H -->|Oui| F
        """
        
        st.graphviz_chart(workflow_code)
        
        # Calendrier du projet
        st.subheader("Calendrier du projet")
        
        project_timeline = pd.DataFrame({
            'Tâche': ['Collecte de données', 'Préprocessing', 'Modélisation', 'Évaluation', 'Déploiement'],
            'Début': pd.to_datetime(['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15', '2023-03-01']),
            'Fin': pd.to_datetime(['2023-01-15', '2023-01-31', '2023-02-15', '2023-02-28', '2023-03-15'])
        })
        
        fig = px.timeline(
            project_timeline, 
            x_start='Début', 
            x_end='Fin', 
            y='Tâche',
            title='Calendrier du projet'
        )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Tâche'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Développé pour le HACK4IFRFI par le Groupe 4")