import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # ❌ Desactiva GPU (causa cuInit error)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # ❌ Reduce logs pesados

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import pickle

app = Flask(__name__)

# --- Cargar modelos SOLO UNA VEZ ---
scaler = joblib.load('models/scaler.pkl')

with open('models/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

rf = joblib.load('models/random_forest.pkl')
xgb = joblib.load('models/xgboost.pkl')
mlp = joblib.load('models/mlp_calibrated.pkl')
svm = joblib.load('models/svm.pkl')
knn = joblib.load('models/knn.pkl')

# ❗ Modelo Keras solo cargarlo (no compilar, no calibrar, no re-entrenar)
deep = tf.keras.models.load_model('models/deep_learning_model.h5')

# --- Página inicial ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Función para decodificar etiqueta ---
def decode_label(pred):
    if isinstance(pred[0], str):
        return pred[0]
    return encoder.inverse_transform(pred)[0]

# --- Endpoint de predicción ---
@app.route('/predict', methods=['POST'])
def predict():

    columnas = [
        'IMC', 'Indice Braquial (IBB)', 'TIBIAL LATERAL', 'Talla',
        'EVALUACIÓN IMC_Grado I de Sobrepeso', 'EVALUACIÓN.1_Metrocórmico Tronco Medio',
        'EVALUACIÓN.10_Bueno', 'EVALUACIÓN.10_Excelente', 'EVALUACIÓN.11_Excelente',
        'EVALUACIÓN.11_Promedio Bajo', 'EVALUACIÓN.12_No Adecuado',
        'EVALUACIÓN.2_Extr. Inf. Medias', 'EVALUACIÓN.4_NORMAL',
        'EVALUACIÓN.6_Muy Elevado', 'EVALUACIÓN.7_EXCELENTE',
        'INTERPRETACION AGB_Reserva Calorica Normal',
        'RESULTADO IBIA/BILIO_Fuerza/Potencia',
        'RESULTADO Indice Cural_Pierna larga o muslo corto',
        'RESULTADO Indice Cural_Pierna muy corta o muslo muy largo',
        'Resultado ICP_Constitución muy débil'
    ]

    valores = [float(request.form[col]) for col in columnas]
    sample = pd.DataFrame([valores], columns=columnas)
    sample_scaled = scaler.transform(sample)

    # --- Predicciones ligeras ---
    proba_rf = rf.predict_proba(sample_scaled)[0]
    proba_xgb = xgb.predict_proba(sample_scaled)[0]
    proba_mlp = mlp.predict_proba(sample_scaled)[0]
    proba_svm = svm.predict_proba(sample_scaled)[0]
    proba_knn = knn.predict_proba(sample_scaled)[0]

    # --- Deep Learning SOLO predict ---
    logits_dl = deep.predict(sample_scaled)
    proba_dl = tf.nn.softmax(logits_dl, axis=1).numpy()[0]

    pred_rf = decode_label(rf.predict(sample_scaled))
    pred_xgb = decode_label(xgb.predict(sample_scaled))
    pred_mlp = decode_label(mlp.predict(sample_scaled))
    pred_svm = decode_label(svm.predict(sample_scaled))
    pred_knn = decode_label(knn.predict(sample_scaled))
    pred_dl = decode_label([np.argmax(proba_dl)])

    # --- Formatear probabilidades ---
    def format_probs(probs):
        etiquetas = encoder.classes_
        return ", ".join([f"{et}: {prob:.2f}" for et, prob in zip(etiquetas, probs)])

    tabla_resultados = [
        {"modelo": "Random Forest", "pred": pred_rf, "probs": format_probs(proba_rf)},
        {"modelo": "XGBoost", "pred": pred_xgb, "probs": format_probs(proba_xgb)},
        {"modelo": "MLP", "pred": pred_mlp, "probs": format_probs(proba_mlp)},
        {"modelo": "SVM", "pred": pred_svm, "probs": format_probs(proba_svm)},
        {"modelo": "KNN", "pred": pred_knn, "probs": format_probs(proba_knn)},
        {"modelo": "Deep Learning", "pred": pred_dl, "probs": format_probs(proba_dl)},
    ]

    # --- Ensemble ---
    predicciones = [pred_rf, pred_xgb, pred_mlp, pred_svm, pred_knn, pred_dl]
    voto_mayoritario = max(set(predicciones), key=predicciones.count)

    idx_reducida = list(encoder.classes_).index("REDUCIDA")
    conf_promedio = np.mean([
        proba_rf[idx_reducida],
        proba_xgb[idx_reducida],
        proba_mlp[idx_reducida],
        proba_dl[idx_reducida]
    ])

    return render_template('result.html',
                            tabla=tabla_resultados,
                            ensemble=voto_mayoritario,
                            confianza=round(conf_promedio, 2))

if __name__ == '__main__':
    app.run()
