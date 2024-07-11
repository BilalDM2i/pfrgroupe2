# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:23:22 2024

@author: Administrateur
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Charger les données des fichiers CSV
train_df = pd.read_csv('Matchs_toutes_competitions_transforme.csv')
predict_df = pd.read_csv('Matchs_uniquement_2024_transforme.csv')

# Afficher les premières lignes des DataFrames
print(train_df.head())
print(predict_df.head())

# Encodage des noms des équipes en entiers
label_encoder = LabelEncoder()
train_df['home_team_encoded'] = label_encoder.fit_transform(train_df['Home_Team'])
train_df['away_team_encoded'] = label_encoder.fit_transform(train_df['Away_Team'])

predict_df['home_team_encoded'] = label_encoder.transform(predict_df['Home_Team'])
predict_df['away_team_encoded'] = label_encoder.transform(predict_df['Away_Team'])

# Séparer les features et les labels pour l'entraînement
X = train_df[['home_team_encoded', 'away_team_encoded']]
y_home = train_df['Resultat_Home_Team']
y_away = train_df['Resultat_Away_Team']

# Séparer les données pour l'entraînement et les tests
X_train, X_test, y_train_home, y_test_home = train_test_split(X, y_home, test_size=0.2, random_state=42)
_, _, y_train_away, y_test_away = train_test_split(X, y_away, test_size=0.2, random_state=42)

# Entraîner le modèle pour les résultats des équipes à domicile
home_model = RandomForestClassifier(random_state=42)
home_model.fit(X_train, y_train_home)

# Entraîner le modèle pour les résultats des équipes extérieures
away_model = RandomForestClassifier(random_state=42)
away_model.fit(X_train, y_train_away)

# Prédire les résultats des matchs de test pour l'équipe à domicile
home_predictions = home_model.predict(X_test)
print(f'Accuracy for home team predictions: {accuracy_score(y_test_home, home_predictions)}')

# Prédire les résultats des matchs de test pour l'équipe extérieure
away_predictions = away_model.predict(X_test)
print(f'Accuracy for away team predictions: {accuracy_score(y_test_away, away_predictions)}')

# Prédire les résultats pour les nouveaux matchs (predict_df)
X_predict = predict_df[['home_team_encoded', 'away_team_encoded']]
predict_df['home_result_pred'] = home_model.predict(X_predict)
predict_df['away_result_pred'] = away_model.predict(X_predict)

"""
# Ajuster les prédictions pour garantir qu'il n'y ait pas de matchs avec deux résultats à 0
for idx, row in predict_df.iterrows():
    if row['home_result_pred'] == 0 and row['away_result_pred'] == 0:
        # Si les deux équipes ont 0, nous décidons arbitrairement de faire gagner l'équipe à domicile
        predict_df.at[idx, 'home_result_pred'] = 1
        predict_df.at[idx, 'away_result_pred'] = 0
    elif row['home_result_pred'] == 1 and row['away_result_pred'] == 1:
        # Si les deux équipes ont 1, nous décidons arbitrairement de faire perdre l'équipe à domicile
        predict_df.at[idx, 'home_result_pred'] = 0
        predict_df.at[idx, 'away_result_pred'] = 1
"""
# Gérer le cas où la France joue à domicile même si elle est listée comme Away_Team
for idx, row in predict_df.iterrows():
    if row['Away_Team'] == 'France':
        # Inverser les résultats si la France joue à domicile en tant qu'équipe extérieure
        predict_df.at[idx, 'home_result_pred'], predict_df.at[idx, 'away_result_pred'] = (
            predict_df.at[idx, 'away_result_pred'], predict_df.at[idx, 'home_result_pred'])


# Sélectionner uniquement les colonnes nécessaires
final_predict_df = predict_df[['Date', 'Home_Team', 'Away_Team', 'home_result_pred', 'away_result_pred']]

# Sauvegarder les résultats prédits dans un nouveau fichier CSV
final_predict_df.to_csv('predicted_results_v3.csv', index=False)
print("Les résultats prédits ont été sauvegardés dans 'predicted_results_v3.csv'.")
