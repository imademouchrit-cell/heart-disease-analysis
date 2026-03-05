# Analyse des maladies cardiaques

Projet réalisé en autonomie pour pratiquer l'analyse de données et le machine learning en Python.
L'idée était de partir d'un vrai dataset médical et d'aller jusqu'à un modèle de prédiction fonctionnel.

# Ce que fait le projet

À partir d'un dataset de 270 patients, j'explore les données, je visualise la distribution des patients
selon leur âge et leur état de santé, puis j'entraîne un modèle de régression logistique pour prédire
la présence d'une maladie cardiaque.

# Fichiers

- `analyse_des_maladies_cardiaques.py` — le script principal
- `Prédiction des maladies cardiaques.csv` — le dataset utilisé

# Variables utilisées pour le modèle

- Age
- Cholesterol
- BP (tension artérielle)
- Max HR (fréquence cardiaque maximale)

# Librairies

```
pandas
matplotlib
scikit-learn
```

# Résultats

Le modèle identifie la fréquence cardiaque maximale et la tension artérielle comme les variables
les plus influentes. Les résultats sont cohérents avec ce qu'on trouve dans la littérature médicale.

# Lancer le projet

```bash
pip install pandas matplotlib scikit-learn
python heart_disease_analysis.py
```
