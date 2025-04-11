# 🧠 Prédiction des Stades de la Maladie Rénale Chronique (MRC) avec l'IA

Ce projet vise à développer un modèle d’intelligence artificielle robuste, capable de **prédire avec la meilleure précision possible les différents stades de la maladie rénale chronique (MRC)**, à partir de données médicales de patients. Il s'inscrit dans une démarche de médecine prédictive et personnalisée, pour assister les professionnels de santé dans leur prise de décision.

## 🎯 Objectif principal

Développer un modèle robuste capable de **prédire les stades de la maladie rénale chronique**, avec un haut niveau de précision.

## 🧩 Objectifs spécifiques

- 📄 **Explorer et comprendre les données médicales** de patients du CNHU/HKM atteints de MRC : examens biologiques, antécédents médicaux, facteurs de risque, etc.
- 🧹 **Effectuer un traitement optimal des données** pour garantir leur qualité et pertinence.
- 🤖 **Construire et évaluer plusieurs modèles de machine learning** pour la classification des stades de la MRC.
- 📈 **Assurer l’interprétabilité et/ou l’explicabilité des meilleurs modèles**, afin de faciliter leur adoption en milieu médical.
- 🔍 **Identifier les variables médicales déterminantes** dans la progression de la maladie.
- 🩺 **Contribuer à la mise en place d’un outil d’aide à la décision** destiné aux professionnels de santé.

## 🛠️ Technologies et outils

- Python (pandas, scikit-learn, matplotlib, seaborn)
- Machine Learning : XGBoost, Random Forest, Logistic Regression
- Méthodes d'interprétabilité : SHAP, permutation importance
- Analyse exploratoire et visualisation de données
- (Optionnel : Streamlit / FastAPI pour l’interface ou l’API)

## ⚙️ Installation

```bash
# Cloner le dépôt
https://github.com/faridmamadou/HACK4IFRI_G4.git

# Se déplacer dans le répertoire
cd HACK4IFRI_G4

# Installer les dépendances
pip install -r requirements.txt

# Lancer le script principal
python main.py
