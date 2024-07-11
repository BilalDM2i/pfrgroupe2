# -*- coding: utf-8 -*-

# Predictions des poules puis quarts puis demi puis finales

# Imports
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt

#%%

# Fonctions
# Fonction pour determiner les quarts de finale à  partir des classements des poules A et B
def calculer_quarts_de_finale(poule_A_df, poule_B_df):
    quarts_de_finale = []

    # Qualification directe pour les quarts de finale
    quart_1 = (poule_A_df.iloc[0]['team'], poule_B_df.iloc[3]['team'])  # 1er de A vs 4ème de B
    quart_2 = (poule_B_df.iloc[0]['team'], poule_A_df.iloc[3]['team'])  # 1er de B vs 4ème de A
    quart_3 = (poule_A_df.iloc[1]['team'], poule_B_df.iloc[2]['team'])  # 2ème de A vs 3ème de B
    quart_4 = (poule_B_df.iloc[1]['team'], poule_A_df.iloc[2]['team'])  # 2ème de B vs 3ème de A

    quarts_de_finale.append(quart_1)
    quarts_de_finale.append(quart_4)
    quarts_de_finale.append(quart_2)
    quarts_de_finale.append(quart_3)

    return quarts_de_finale


# Fonction pour creer un graphique en barres pour une poule
def plot_poule(poule_df, title):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(poule_df['team'], poule_df['points'], color='skyblue')
    plt.xlabel('Équipe')
    plt.ylabel('Points')
    plt.title(title)
    plt.xticks(rotation=45)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + 0.05, round(yval, 2))  # Ajouter le score au-dessus de chaque barre
    plt.tight_layout()
    plt.show()


# Fonction pour ajouter les variables du train pour quarts, demis et finale
def ajout_variables_train(matchs, train_df):
    data = []
    # Pour chaque match extraire les donnees correspondantes de train_df
    for home_team, away_team in matchs:
        # Recuperer les donnees pour l'equipe à  domicile
        home_data = train_df[train_df['Home_Team'] == home_team].iloc[0]
        home_team_encoded = home_data['home_team_encoded']
        home_Age_Moyen = home_data['Home_Age_Moyen']
        home_Taille_Moyenne = home_data['Home_Taille_Moyenne']
        home_Selection_Moyenne = home_data['Home_Selection_Moyenne']
        home_Buts_Moyens = home_data['Home_Buts_Moyens']
        home_Rank = home_data['Home_Rank']

        # Recuperer les donnees pour l'equipe à  l'exterieur
        away_data = train_df[train_df['Home_Team'] == away_team].iloc[0]
        away_team_encoded = away_data['away_team_encoded']
        away_Age_Moyen = away_data['Away_Age_Moyen']
        away_Taille_Moyenne = away_data['Away_Taille_Moyenne']
        away_Selection_Moyenne = away_data['Away_Selection_Moyenne']
        away_Buts_Moyens = away_data['Away_Buts_Moyens']
        away_Rank = away_data['Away_Rank']

        # Creer un dictionnaire pour chaque match de quart de finale
        match_data = {
            'Home_Team': home_team,
            'Away_Team': away_team,
            'home_team_encoded': home_team_encoded,
            'away_team_encoded': away_team_encoded,
            'Home_Age_Moyen': home_Age_Moyen,
            'Home_Taille_Moyenne': home_Taille_Moyenne,
            'Home_Selection_Moyenne': home_Selection_Moyenne,
            'Home_Buts_Moyens': home_Buts_Moyens,
            'Home_Rank': home_Rank,
            'Away_Age_Moyen': away_Age_Moyen,
            'Away_Taille_Moyenne': away_Taille_Moyenne,
            'Away_Selection_Moyenne': away_Selection_Moyenne,
            'Away_Buts_Moyens': away_Buts_Moyens,
            'Away_Rank': away_Rank,
        }

        data.append(match_data)

    return data

#%%

# ----------------------------------------------------------------------------------------------------------------------
# POULES
# ----------------------------------------------------------------------------------------------------------------------

# Charger les fichiers train et predict df (cree dans le script 0_prep_train_predict_df.py)
train_df = pd.read_csv('train_df.csv')
predict_df = pd.read_csv('predict_df.csv')

print(train_df.head())
print(predict_df.head())

# Encodage des noms des equipes
all_teams = pd.concat(
    [train_df['Home_Team'], train_df['Away_Team'], predict_df['Home_Team'], predict_df['Away_Team']]).unique()
label_encoder = LabelEncoder()
label_encoder.fit(all_teams)

train_df['home_team_encoded'] = label_encoder.transform(train_df['Home_Team'])
train_df['away_team_encoded'] = label_encoder.transform(train_df['Away_Team'])
predict_df['home_team_encoded'] = label_encoder.transform(predict_df['Home_Team'])
predict_df['away_team_encoded'] = label_encoder.transform(predict_df['Away_Team'])

# Creer une colonne qui indique si l'equipe à  domicile a gagne (1) ou perdu (0)
train_df['match_result'] = (train_df['Resultat_Home_Team'] > train_df['Resultat_Away_Team']).astype(int)
# WARNING : on pourrait sinon ici mettre les resultats des matchs, les scores ?

#%%

#On ajoute les valeurs minimales pour les Nan (données pour équipes non qualifiées)

# Définir une fonction pour obtenir les valeurs minimales des colonnes
def get_min_values(df, columns):
    return {col: df[col].min() for col in columns}

columns_to_impute = ['home_team_encoded', 'away_team_encoded', 'Home_Age_Moyen', 'Home_Taille_Moyenne',
                     'Home_Selection_Moyenne', 'Home_Buts_Moyens', 'Away_Age_Moyen',
                     'Away_Taille_Moyenne', 'Away_Selection_Moyenne', 'Away_Buts_Moyens',
                     'Home_Rank', 'Away_Rank']

# Obtenir les valeurs minimales pour les colonnes à imputer
min_values = get_min_values(train_df, columns_to_impute)

# Créer un imputeur pour chaque colonne avec les valeurs minimales correspondantes
imputers = {col: SimpleImputer(strategy='constant', fill_value=min_value) for col, min_value in min_values.items()}

# Appliquer l'imputation à chaque colonne pour les données d'entraînement
for col, imputer in imputers.items():
    train_df[col] = imputer.fit_transform(train_df[[col]])

# Appliquer l'imputation à chaque colonne pour les données de prédiction
for col, imputer in imputers.items():
    predict_df[col] = imputer.transform(predict_df[[col]])

#%%

# Caracteristiques (features) pour le modèle
X = train_df[['home_team_encoded', 'away_team_encoded', 'Home_Age_Moyen', 'Home_Taille_Moyenne',
              'Home_Selection_Moyenne', 'Home_Buts_Moyens', 'Away_Age_Moyen',
              'Away_Taille_Moyenne', 'Away_Selection_Moyenne', 'Away_Buts_Moyens',
              'Home_Rank', 'Away_Rank']]
y = train_df['match_result']

# Validation croisee pour evaluer le modèle
match_model = RandomForestClassifier(
    criterion='gini',
    max_depth=6,
    n_estimators=300,
    random_state=42
)
cross_val_scores = cross_val_score(match_model, X, y, cv=5, scoring='accuracy')
print(f'Cross-validated accuracy: {cross_val_scores.mean()}')

# Entraîner le modèle
match_model.fit(X, y)

# Predire les resultats pour les nouveaux matchs (predict_df)
X_predict = predict_df[['home_team_encoded', 'away_team_encoded', 'Home_Age_Moyen', 'Home_Taille_Moyenne',
                        'Home_Selection_Moyenne', 'Home_Buts_Moyens', 'Away_Age_Moyen',
                        'Away_Taille_Moyenne', 'Away_Selection_Moyenne', 'Away_Buts_Moyens',
                        'Home_Rank', 'Away_Rank']]
predict_df['match_result_pred'] = match_model.predict(X_predict)

# Colonne indiquant le gagnant
predict_df['Winning_Team'] = predict_df.apply(
    lambda row: row['Home_Team'] if row['match_result_pred'] == 1 else row['Away_Team'],
    axis=1
)

# Selection de colonnes
poules_predict_df = predict_df[['Date', 'Home_Team', 'Away_Team', 'match_result_pred', 'Winning_Team']]

# Sauvegarder les resultats predits
poules_predict_df.to_csv('predictions_poules.csv', index=False)
print("Les resultats predits pour les matchs de poules ont ete sauvegardes dans 'predictions_poules.csv'.")

# Affichage des résultats des prédictions de poules
print("\nRésultats prédits pour les matchs de poules :\n", poules_predict_df)

# WARNING : pas de match nul possible... pb ou pas ?

# ----------------------------------------------------------------------------------------------------------------------
# CLASSEMENT POULES
# ----------------------------------------------------------------------------------------------------------------------
# Définir les poules et les équipes qui y appartiennent
poule_A_teams = ['Espagne', 'Croatie', 'Allemagne', 'Slovenie', 'Suede', 'Japon']
poule_B_teams = ['Danemark', 'Norvege', 'Hongrie', 'France', 'Egypte', 'Argentine']

# Initialiser un dictionnaire pour stocker les points des équipes
points = {team: 0 for team in poule_A_teams + poule_B_teams}

# Parcourir les résultats des matchs et mettre à jour les points // Attention il n'y a pas le cas du match nul
for idx, row in poules_predict_df.iterrows():
    Home_Team = row['Home_Team']
    Away_Team = row['Away_Team']
    Winning_Team = row['Winning_Team']

    if Winning_Team == Home_Team:
        if Home_Team in points:
            points[Home_Team] += 2
    elif Winning_Team == Away_Team:
        if Away_Team in points:
            points[Away_Team] += 2

# Créer un DataFrame pour les points par équipe
points_df = pd.DataFrame(list(points.items()), columns=['team', 'points'])

# Séparer les points par poule
poule_A_points = points_df[points_df['team'].isin(poule_A_teams)].sort_values(by='points', ascending=False).reset_index(
    drop=True)
poule_B_points = points_df[points_df['team'].isin(poule_B_teams)].sort_values(by='points', ascending=False).reset_index(
    drop=True)

# Appel de la fonction pour calculer les quarts de finale
quarts_de_finale = calculer_quarts_de_finale(poule_A_points, poule_B_points)

# Afficher les graphiques pour chaque poule
plot_poule(poule_A_points, 'Classement Poule A')
plot_poule(poule_B_points, 'Classement Poule B')

# Affichage des quarts de finale
print("\nTableau des quarts de finale :")
for i, match in enumerate(quarts_de_finale, start=1):
    print(f"Quart de finale {i}: {match[0]} contre {match[1]}")

# Convertir les quarts de finale en DataFrame
quarts_de_finale_df = pd.DataFrame(quarts_de_finale, columns=['Home_Team', 'Away_Team'])

print(quarts_de_finale_df)

#%%

# ----------------------------------------------------------------------------------------------------------------------
# QUARTS
# ----------------------------------------------------------------------------------------------------------------------
# Faire maintenant les prédictions sur quarts_de_finale_df, il faut ajouter les infos pour faire la prédiction

# WARNING : à vérifier / optimiser

quarts_de_finale_matches = [
    (quarts_de_finale_df.loc[0, 'Home_Team'], quarts_de_finale_df.loc[0, 'Away_Team']),
    (quarts_de_finale_df.loc[1, 'Home_Team'], quarts_de_finale_df.loc[1, 'Away_Team']),
    (quarts_de_finale_df.loc[2, 'Home_Team'], quarts_de_finale_df.loc[2, 'Away_Team']),
    (quarts_de_finale_df.loc[3, 'Home_Team'], quarts_de_finale_df.loc[3, 'Away_Team'])
]


quarts_de_finale_data = ajout_variables_train(quarts_de_finale_matches, train_df)
quarts_de_finale_data = pd.DataFrame(quarts_de_finale_data)

# Caractéristiques pour la prédiction
X_quarts = quarts_de_finale_data[['home_team_encoded', 'away_team_encoded',
                                  'Home_Age_Moyen', 'Home_Taille_Moyenne', 'Home_Selection_Moyenne', 'Home_Buts_Moyens',
                                  'Away_Age_Moyen', 'Away_Taille_Moyenne', 'Away_Selection_Moyenne', 'Away_Buts_Moyens',
                                  'Home_Rank', 'Away_Rank']]

# Utiliser le modèle déjà entraîné pour prédire les résultats
quarts_de_finale_data['match_result_pred'] = match_model.predict(X_quarts)

# Ajouter une colonne pour indiquer le gagnant
quarts_de_finale_data['Winning_Team'] = quarts_de_finale_data.apply(
    lambda row: row['Home_Team'] if row['match_result_pred'] == 1 else row['Away_Team'],
    axis=1
)

# Sélectionner uniquement les colonnes nécessaires pour les prédictions
quarts_predict_df = quarts_de_finale_data[['Home_Team', 'Away_Team', 'match_result_pred', 'Winning_Team']]

print("\nPrédictions des quarts de finale : \n", quarts_predict_df)

# Sauvegarder les résultats prédits dans un fichier CSV
quarts_predict_df.to_csv('predictions_quarts.csv', index=False)
print("Les résultats prédits des quarts de finale ont été sauvegardés dans 'predictions_quarts.csv'.")

# Affichage de la précision pour les quarts de finale
quarts_accuracy = np.mean(quarts_de_finale_data['match_result_pred'] == 1)
print(f"Précision des quarts de finale: {quarts_accuracy}")

# ----------------------------------------------------------------------------------------------------------------------
# DEMIS
# ----------------------------------------------------------------------------------------------------------------------


#%%

# WARNING : à vérifier / optimiser
demis_finale_matches = [
    (quarts_predict_df.loc[0, 'Winning_Team'], quarts_predict_df.loc[1, 'Winning_Team']),
    (quarts_predict_df.loc[2, 'Winning_Team'], quarts_predict_df.loc[3, 'Winning_Team'])
]
demis_finale_data = ajout_variables_train(demis_finale_matches, train_df)
demis_finale_data = pd.DataFrame(demis_finale_data)

# Caractéristiques pour la prédiction
X_demis = demis_finale_data[['home_team_encoded', 'away_team_encoded',
                             'Home_Age_Moyen', 'Home_Taille_Moyenne', 'Home_Selection_Moyenne', 'Home_Buts_Moyens',
                             'Away_Age_Moyen', 'Away_Taille_Moyenne', 'Away_Selection_Moyenne', 'Away_Buts_Moyens',
                             'Home_Rank', 'Away_Rank']]

# Utiliser le modèle déjà entraîné pour prédire les résultats
demis_finale_data['match_result_pred'] = match_model.predict(X_demis)

# Ajouter une colonne pour indiquer le gagnant
demis_finale_data['Winning_Team'] = demis_finale_data.apply(
    lambda row: row['Home_Team'] if row['match_result_pred'] == 1 else row['Away_Team'],
    axis=1
)

# Sélectionner uniquement les colonnes nécessaires pour les prédictions
demis_predict_df = demis_finale_data[['Home_Team', 'Away_Team', 'match_result_pred', 'Winning_Team']]

print("\nPrédictions des demi-finales : \n", demis_predict_df)

# Sauvegarder les résultats prédits dans un fichier CSV
demis_predict_df.to_csv('predictions_demis.csv', index=False)
print("Les résultats prédits des demi-finales ont été sauvegardés dans 'predictions_demis.csv'.")

# Affichage de la précision pour les demi-finales
demis_accuracy = np.mean(demis_finale_data['match_result_pred'] == 1)
print(f"Précision des demi-finales: {demis_accuracy}")

# ----------------------------------------------------------------------------------------------------------------------
# FINALES
# ----------------------------------------------------------------------------------------------------------------------

# WARNING : à vérifier / optimiser
finale_match = [(demis_predict_df.loc[0, 'Winning_Team'], demis_predict_df.loc[1, 'Winning_Team'])]

finale_data = ajout_variables_train(finale_match, train_df)
finale_data = pd.DataFrame(finale_data)

# Caractéristiques pour la prédiction
X_finale = finale_data[['home_team_encoded', 'away_team_encoded',
                        'Home_Age_Moyen', 'Home_Taille_Moyenne', 'Home_Selection_Moyenne', 'Home_Buts_Moyens',
                        'Away_Age_Moyen', 'Away_Taille_Moyenne', 'Away_Selection_Moyenne', 'Away_Buts_Moyens',
                        'Home_Rank', 'Away_Rank']]

# Utiliser le modèle déjà entraîné pour prédire les résultats
finale_data['match_result_pred'] = match_model.predict(X_finale)

# Ajouter une colonne pour indiquer le gagnant
finale_data['Winning_Team'] = finale_data.apply(
    lambda row: row['Home_Team'] if row['match_result_pred'] == 1 else row['Away_Team'],
    axis=1
)

# Sélectionner uniquement les colonnes nécessaires pour les prédictions
finale_predict_df = finale_data[['Home_Team', 'Away_Team', 'match_result_pred', 'Winning_Team']]

print("\nPrédiction de la finale : \n", finale_predict_df)

# Sauvegarder les résultats prédits dans un fichier CSV
finale_predict_df.to_csv('predictions_finale.csv', index=False)
print("Les résultats prédits de la finale ont été sauvegardés dans 'predictions_finale.csv'.")

# Affichage de la précision pour la finale
finale_accuracy = np.mean(finale_data['match_result_pred'] == 1)
print(f"Précision de la finale: {finale_accuracy}")
