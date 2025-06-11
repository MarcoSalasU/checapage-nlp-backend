
import joblib
from feature_extract import extract_feature_vector

def predict(img_path, html_path):
    vector = extract_feature_vector(html_path, img_path)
    forest = joblib.load('saved_models/forest.pkl')  # Ajuste: ruta relativa directa
    p = forest.predict([vector])
    return p  # Ajuste: retorno necesario para Flask
