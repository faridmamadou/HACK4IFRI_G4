from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List, Union, Dict, Any
import uvicorn
import os
from contextlib import asynccontextmanager

# Variable globale pour stocker le modèle chargé
model = None

# Chemin vers le modèle (à ajuster selon votre configuration)
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.pkl")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de lifespan pour charger et décharger le modèle"""
    # Chargement du modèle au démarrage
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Modèle chargé depuis {MODEL_PATH}")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        model = None
    
    yield  # L'application s'exécute ici
    
    # Nettoyage à l'arrêt de l'application si nécessaire
    model = None
    print("Ressources libérées")

# Initialisation de l'application FastAPI avec le lifespan
app = FastAPI(
    title="API d'inférence pour modèle ML",
    description="Une API pour faire des prédictions avec un modèle de machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Définition du schéma de données d'entrée
class PredictionInput(BaseModel):
    features: List[Union[float, int]]
    
    class Config:
        schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]  # Exemple pour un modèle de classification d'Iris
            }
        }

# Définition du schéma de données de sortie
class PredictionOutput(BaseModel):
    prediction: Union[float, int, str, List]
    probability: Union[float, List[float], Dict[str, float], None] = None

# Fonction pour charger le modèle 
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Modèle rechargé depuis {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        model = None
        return False

@app.get("/")
def read_root():
    """Page d'accueil basique de l'API"""
    return {"message": "API d'inférence ML. Accédez à /docs pour la documentation interactive."}

@app.get("/health")
def health_check():
    """Endpoint pour vérifier l'état de l'API"""
    if model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Endpoint pour faire des prédictions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé")
    
    try:
        # Conversion des features en array numpy
        features = np.array(input_data.features).reshape(1, -1)
        
        # Prédiction
        prediction = model.predict(features)[0]
        
        # Certains modèles proposent predict_proba pour les probabilités
        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(features)[0].tolist()
        
        return PredictionOutput(
            prediction=prediction,
            probability=probability
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

@app.get("/model-info")
def model_info():
    """Renvoie des informations sur le modèle chargé"""
    if model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé")
    
    info = {
        "type": type(model).__name__,
        "features_expected": None,
    }
    
    # Ajout d'informations supplémentaires selon le type de modèle
    if hasattr(model, "feature_names_in_"):
        info["features_expected"] = model.feature_names_in_.tolist()
    if hasattr(model, "classes_"):
        info["classes"] = model.classes_.tolist()
    if hasattr(model, "n_features_in_"):
        info["n_features"] = model.n_features_in_
    
    return info

@app.post("/reload-model")
def reload_model():
    """Recharge le modèle à partir du fichier"""
    success = load_model()
    if not success:
        raise HTTPException(status_code=500, detail="Échec du rechargement du modèle")
    return {"status": "Le modèle a été rechargé avec succès"}

# Point d'entrée pour exécuter l'application avec uvicorn
if __name__ == "__main__":
    # Définition du port (par défaut 8000)
    port = int(os.environ.get("PORT", 8000))
    
    # Démarrage du serveur
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)