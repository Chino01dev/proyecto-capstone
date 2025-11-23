# calibrate_mlp_cv.py
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1) Cargar X_train, y_train (CSV que guardaste)
X = pd.read_csv('data/X_train_prepared.csv')
y = pd.read_csv('data/y_train_prepared.csv').squeeze()

# 2) Opcional: re-entrenar el MLP aquí o usar el existente
# Si quieres reentrenarlo:
scaler = joblib.load('models/scaler.pkl')   # si usaste scaler
X_scaled = scaler.transform(X)

mlp = MLPClassifier(hidden_layer_sizes=(128,64,32), max_iter=500, random_state=42)
# CalibratedClassifierCV con cv=3 reentrena internamente
calibrated = CalibratedClassifierCV(mlp, method='sigmoid', cv=3)
calibrated.fit(X_scaled, y)

joblib.dump(calibrated, 'models/mlp_calibrated.pkl')
print("✅ MLP entrenado y calibrado guardado en models/mlp_calibrated.pkl")
