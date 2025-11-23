from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import pickle
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV   # 🔧 NUEVO

app = Flask(__name__)

# --- Cargar modelos ---
scaler = joblib.load('models/scaler.pkl')

with open('models/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

rf = joblib.load('models/random_forest.pkl')
xgb = joblib.load('models/xgboost.pkl')
mlp = joblib.load('models/mlp_calibrated.pkl')
svm = joblib.load('models/svm.pkl')
knn = joblib.load('models/knn.pkl')
deep = tf.keras.models.load_model('models/deep_learning_model.h5')

# --- 🔧 Calibración del MLP ---
# Intentamos aplicar una calibración ligera si hay datos de validación disponibles
try:
    X_val = np.load('data/X_val.npy')
    y_val = np.load('data/y_val.npy')

    print("Calibrando MLP con CalibratedClassifierCV...")
    mlp_calibrated = CalibratedClassifierCV(mlp, cv='prefit', method='sigmoid')
    mlp_calibrated.fit(X_val, y_val)
    mlp = mlp_calibrated
except Exception as e:
    print(f"[AVISO] No se pudo calibrar MLP automáticamente: {e}")

# --- 🔧 Calibración del modelo Keras (temperature scaling) ---
def apply_temperature_scaling(model, X_val, temperature=2.0):
    logits = model.predict(X_val)
    scaled_logits = logits / temperature
    probs = tf.nn.softmax(scaled_logits, axis=1).numpy()
    return probs

# Intentamos aplicar la calibración de temperatura si hay datos de validación
try:
    X_val = np.load('data/X_val.npy')
    y_val = np.load('data/y_val.npy')
    print("Calibrando red Keras con temperature scaling...")
    deep_temperature = 2.0  # Puedes ajustar este valor
except Exception as e:
    print(f"[AVISO] No se pudo calibrar el modelo Keras automáticamente: {e}")
    deep_temperature = 1.0  # Default sin calibrar

# --- Función para decodificar etiquetas ---
def decode_label(pred):
    if isinstance(pred[0], str):
        return pred[0]
    else:
        return encoder.inverse_transform(pred)[0]

# --- Página inicial ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Procesar formulario ---
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener datos del formulario (20 campos)
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

    # --- Predicciones ---
    proba_rf = rf.predict_proba(sample_scaled)[0]
    proba_xgb = xgb.predict_proba(sample_scaled)[0]
    proba_mlp = mlp.predict_proba(sample_scaled)[0]
    proba_knn = knn.predict_proba(sample_scaled)[0] if hasattr(knn, "predict_proba") else None

    # 🔧 Calibrar salida del modelo Keras
    logits_dl = deep.predict(sample_scaled)
    scaled_logits = logits_dl / deep_temperature
    proba_dl = tf.nn.softmax(scaled_logits, axis=1).numpy()[0]

    pred_rf = decode_label(rf.predict(sample_scaled))
    pred_xgb = decode_label(xgb.predict(sample_scaled))
    pred_mlp = decode_label(mlp.predict(sample_scaled))
    pred_svm = decode_label(svm.predict(sample_scaled))
    pred_knn = decode_label(knn.predict(sample_scaled))
    pred_dl = decode_label([np.argmax(proba_dl)])

    # --- Formatear probabilidades ---
    def format_probs(probs):
        if probs is None:
            return "(decisión, no prob)"
        etiquetas = encoder.classes_
        formatted = ", ".join([f"{et}: {prob:.2f}" for et, prob in zip(etiquetas, probs)])
        return f"[{formatted}]"

    # --- Crear tabla de resultados ---
    tabla_resultados = [
        {"modelo": "Random Forest", "pred": pred_rf, "probs": format_probs(proba_rf)},
        {"modelo": "XGBoost", "pred": pred_xgb, "probs": format_probs(proba_xgb)},
        {"modelo": "MLP", "pred": pred_mlp, "probs": format_probs(proba_mlp)},
        {"modelo": "SVM", "pred": pred_svm, "probs": format_probs(svm.predict_proba(sample_scaled)[0])},
        {"modelo": "KNN (k=5)", "pred": pred_knn, "probs": format_probs(proba_knn)},
        {"modelo": "Deep Learning (Keras)", "pred": pred_dl, "probs": format_probs(proba_dl)},
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

# --- Mostrar datos de prueba ---
@app.route("/test-data")
def show_test_data():
    X_test = pd.read_csv("data/X_test_prepared.csv")
    y_test = pd.read_csv("data/y_test_prepared.csv")
    data = X_test.copy()
    data["Etiqueta Real"] = y_test
    records = data.to_dict(orient="records")
    columns = data.columns.tolist()
    return render_template("test_data.html", records=records, columns=columns)


if __name__ == '__main__':
    app.run(debug=True)
