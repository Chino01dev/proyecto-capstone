import pandas as pd
import numpy as np

# Cargar los datos de prueba existentes
X_test = pd.read_csv("data/X_test_prepared.csv")
y_test = pd.read_csv("data/y_test_prepared.csv")

# Convertir a arrays NumPy
X_val = X_test.to_numpy()
y_val = y_test.to_numpy().ravel()  # Asegura que sea un vector 1D

# Guardar como archivos .npy para uso directo en calibración
np.save("data/X_val.npy", X_val)
np.save("data/y_val.npy", y_val)

print("✅ Archivos X_val.npy y y_val.npy generados correctamente.")
print(f"Dimensiones X_val: {X_val.shape}")
print(f"Dimensiones y_val: {y_val.shape}")
