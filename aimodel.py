import pandas as pd

# Cargar el CSV (puede ser desde archivo o directamente desde el texto que compartiste)
df = pd.read_csv('sleep_quality.csv')  # cambia por el nombre de tu archivo

# Convertir 'yes'/'no' a 1/0
df['Alcohol'] = df['Alcohol'].map({'yes': 1, 'no': 0})

# Opcional: eliminar columnas que no aportan (como ID)
df = df.drop(columns=['ID'])

# Variables predictoras (todas menos la etiqueta)
X = df.drop(columns=['Label'])

# Etiqueta
y = df['Label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Cambio aquí: importamos KNeighborsClassifier y usamos KNN
from sklearn.neighbors import KNeighborsClassifier

# Definir el modelo KNN, por ejemplo con k=5 vecinos (puedes ajustar k)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

import joblib
joblib.dump(model, 'modelo_sueno.pkl')
joblib.dump(scaler, 'scaler.pkl')
