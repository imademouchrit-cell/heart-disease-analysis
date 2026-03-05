# Analyse prédictive des maladies cardiaques
# Dataset : heart_disease_prediction.csv (270 patients)
# Outils : pandas, matplotlib, scikit-learn


# Importer les bibliothèques

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Charger le fichier CSV

df = pd.read_csv('heart_disease_prediction.csv')


# Exploration du dataset

print("-" * 60)
print("EXPLORATION DU DATASET MALADIES CARDIAQUES")
print("-" * 60)

print("\n1. Les 5 premières lignes:")
print(df.head())

print("\n2. Informations générales:")
print(f"Nombre de lignes : {len(df)}")
print(f"Nombre de colonnes : {len(df.columns)}")

print("\n3. Noms des colonnes:")
print(df.columns.tolist())

print("\n4. Types de données:")
print(df.dtypes)

print("\n5. Statistiques de base:")
print(df.describe())

print("\n6. Valeurs manquantes:")
print(df.isnull().sum())

print("\n7. Distribution de la maladie cardiaque:")
print(df['Heart Disease'].value_counts())


# Créer la colonne "target"

df['target'] = (df['Heart Disease'] == 'Presence').astype(int)

print(f"\nPourcentage avec maladie: {(df['target'].sum() / len(df)) * 100:.1f}%")


# Histogramme de l'âge (tous les patients)

plt.figure(figsize=(8, 5))
plt.hist(df['Age'], bins=20, color='blue', edgecolor='black')
plt.title("Distribution de l'âge")
plt.xlabel("Âge")
plt.ylabel("Nombre de patients")
plt.tight_layout()
plt.show()


# Histogramme comparatif malades vs sains

plt.figure(figsize=(10, 6))
plt.hist(df[df['target'] == 0]['Age'], bins=20, alpha=0.5, label='Sans maladie', color='green', edgecolor='black')
plt.hist(df[df['target'] == 1]['Age'], bins=20, alpha=0.5, label='Avec maladie', color='red', edgecolor='black')
plt.title("Distribution de l'âge par groupe (Malades vs Sains)")
plt.xlabel("Âge")
plt.ylabel("Nombre de patients")
plt.legend()
plt.tight_layout()
plt.show()


# Préparer les données pour le modèle
# On utilise 4 variables numériques pertinentes du dataset

features = ['Age', 'Cholesterol', 'Max HR', 'BP']

X = df[features]
y = df['target']


# Séparer les données en train/test (80% / 20%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

print(f"\nDonnées d'entraînement : {len(X_train)} patients")
print(f"Données de test        : {len(X_test)} patients")


# Entraîner le modèle de régression logistique

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# Évaluer le modèle

y_prediction = model.predict(X_test)

print("\n8. Évaluation du modèle:")
print(f"Précision du modèle : {accuracy_score(y_test, y_prediction) * 100:.1f}%")
print("\nRapport de classification:")
print(classification_report(y_test, y_prediction, target_names=['Sans maladie', 'Avec maladie']))


# Visualiser l'importance des variables

coefficients = pd.Series(model.coef_[0], index=features)
coefficients.sort_values().plot(kind='barh', color='blue', figsize=(8, 4))
plt.title("Importance des variables (coefficients de la régression logistique)")
plt.xlabel("Coefficient")
plt.axvline(x=0, color='red', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.show()


# Conclusion

print("\nCONCLUSION")
print("-" * 60)
print(f"Le modèle prédit correctement {accuracy_score(y_test, y_prediction) * 100:.1f}% des cas.")
print("\nCoefficients des variables (influence sur la maladie cardiaque):")
print(coefficients.sort_values(ascending=False))
print("\nUn coefficient positif signifie que la variable augmente le risque.")
print("Un coefficient négatif signifie qu'elle le diminue.")

print("\nSigne des coefficients :")
print("BP et Cholesterol ont un coefficient positif")
print("- Une tension artérielle ou un cholestérol élevé augmente le risque de maladie cardiaque")

print("\nMax HR et Age ont un coefficient négatif")
print("- Une fréquence cardiaque maximale élevée diminue le risque de maladie cardiaque")
print("(l'âge influence indirectement la maladie cardiaque via d'autres variables comme la tension et le cholestérol)")

print("\nVariable la plus influente :")
print("- Max HR (-0.041) est la variable qui influence le plus la prédiction")

print("\nConclusion médicale :")
print("- Le modèle retrouve des résultats cohérents avec la littérature médicale")
print("- Cela valide la pertinence de nos variables et de notre approche")