import pandas as pd

df = pd.read_csv('sleep_quality.csv')

df['Alcohol'] = df['Alcohol'].map({'yes': 1, 'no': 0})

df = df.drop(columns=['ID'])

X = df.drop(columns=['Label'])
y = df['Label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

print('KNN algorithm:')
print("Accuracy:", accuracy_score(y_test, y_pred))

import joblib
joblib.dump(model, 'modelo_sueno.pkl')
joblib.dump(scaler, 'scaler.pkl')

