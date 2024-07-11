from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Supposez que vos données sont dans un DataFrame pandas
import pandas as pd

# Chargement des données (remplacez par vos données réelles)
data = pd.read_csv('handball_matches.csv')

# Séparation des caractéristiques (X) et de la cible (y)
X = data.drop('result', axis=1)
y = data['result']

# Séparation des ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Préparation du pipeline de transformation des données
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['feature1', 'feature2', ...]),  # Remplacez par vos caractéristiques numériques
        ('cat', OneHotEncoder(), ['feature3', 'feature4', ...])    # Remplacez par vos caractéristiques catégorielles
    ])

# Création du pipeline complet avec un modèle (par exemple, Régression Logistique)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédictions
predictions = model.predict(X_test)
