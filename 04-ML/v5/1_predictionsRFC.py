# -*- coding: utf-8 -*-

# Prédictions des poules puis quarts puis demi puis finale


# Imports
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt

#%%

# Fonctions
# Fonction pour déterminer les quarts de finale à partir des classements des poules A et B
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


# Fonction pour créer un graphique en barres pour une poule
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
        # Récupérer les données pour l'équipe à domicile
        home_data = train_df[train_df['Home_Team'] == home_team].iloc[0]
        home_team_encoded = home_data['home_team_encoded']
        home_age_moyen = home_data['Home_Age_Moyen']
        home_taille_moyenne = home_data['Home_Taille_Moyenne']
        home_selection_moyenne = home_data['Home_Selection_Moyenne']
        home_buts_moyens = home_data['Home_Buts_Moyens']
        home_rank = home_data['Home_Rank']

        # Récupérer les données pour l'equipe à l'exterieur
        away_data = train_df[train_df['Home_Team'] == away_team].iloc[0]
        away_team_encoded = away_data['away_team_encoded']
        away_age_moyen = away_data['Away_Age_Moyen']
        away_taille_moyenne = away_data['Away_Taille_Moyenne']
        away_selection_moyenne = away_data['Away_Selection_Moyenne']
        away_buts_moyens = away_data['Away_Buts_Moyens']
        away_rank = away_data['Away_Rank']

        # Créer un dictionnaire pour chaque match de quart de finale
        match_data = {
            'Home_Team': home_team,
            'Away_Team': away_team,
            'home_team_encoded': home_team_encoded,
            'away_team_encoded': away_team_encoded,
            'Home_Age_Moyen': home_age_moyen,
            'Home_Taille_Moyenne': home_taille_moyenne,
            'Home_Selection_Moyenne': home_selection_moyenne,
            'Home_Buts_Moyens': home_buts_moyens,
            'Home_Rank': home_rank,
            'Away_Age_Moyen': away_age_moyen,
            'Away_Taille_Moyenne': away_taille_moyenne,
            'Away_Selection_Moyenne': away_selection_moyenne,
            'Away_Buts_Moyens': away_buts_moyens,
            'Away_Rank': away_rank
        }

        data.append(match_data)

    return data


#%%

# ----------------------------------------------------------------------------------------------------------------------
# POULES
# ----------------------------------------------------------------------------------------------------------------------

# Charger les fichiers train et predict df (crée dans le script 0_prep_train_predict_df.py)
train_df = pd.read_csv('train_df.csv')
predict_df = pd.read_csv('predict_df.csv')

print(train_df.head())
print(predict_df.head())

# Encodage des noms des équipes
all_teams = pd.concat(
    [train_df['Home_Team'], train_df['Away_Team'], predict_df['Home_Team'], predict_df['Away_Team']]).unique()
label_encoder = LabelEncoder()
label_encoder.fit(all_teams)

train_df['home_team_encoded'] = label_encoder.transform(train_df['Home_Team'])
train_df['away_team_encoded'] = label_encoder.transform(train_df['Away_Team'])
predict_df['home_team_encoded'] = label_encoder.transform(predict_df['Home_Team'])
predict_df['away_team_encoded'] = label_encoder.transform(predict_df['Away_Team'])

# Créer une colonne qui indique si l'équipe à domicile a gagné (1) ou perdu (0) ou si s'il s'agit d'un match nul (2)

# Conditions pour déterminer le résultat du match
conditions = [
    (train_df['Resultat_Home_Team'] > train_df['Resultat_Away_Team']),  # Victoire de l'équipe à domicile
    (train_df['Resultat_Home_Team'] < train_df['Resultat_Away_Team']),  # Défaite de l'équipe à domicile
    (train_df['Resultat_Home_Team'] == train_df['Resultat_Away_Team'])  # Match nul
]

# Valeurs correspondantes pour chaque condition
choices = [1, 0, 2]

# Créer une nouvelle colonne 'match_result' en fonction des conditions
train_df['match_result'] = np.select(conditions, choices)


# Imputations pour variables âge, taille, sélection, buts et rang (quand c'est NA)

# Fonction pour calculer les valeurs moyennes et maximales pour l'imputation
def get_mean_values(df, columns):
    return {col: df[col].mean() for col in columns}


def get_max_values(df, columns):
    return {col: df[col].max() for col in columns}

# Pour taille, âge, sélections et buts on impute par la moyenne
# (car on a les infos uniquement pour les équipes qualifiées)
columns_to_impute = ['Home_Age_Moyen', 'Home_Taille_Moyenne',
                     'Home_Selection_Moyenne', 'Home_Buts_Moyens', 'Away_Age_Moyen',
                     'Away_Taille_Moyenne', 'Away_Selection_Moyenne', 'Away_Buts_Moyens']
# Valeurs moyennes
mean_values = get_mean_values(train_df, columns_to_impute)
# Créer un imputeur pour chaque colonne avec les valeurs moyennes correspondantes
imputers = {col: SimpleImputer(strategy='constant', fill_value=mean_value) for col, mean_value in mean_values.items()}
# Appliquer l'imputation à chaque colonne pour les données d'entraînement
for col, imputer in imputers.items():
    train_df[col] = imputer.fit_transform(train_df[[col]])

# Pour rang
columns_to_impute = ['Home_Rank', "Away_Rank"]
# Valeurs maximales
max_values = get_max_values(train_df, columns_to_impute)
# Créer un imputeur pour chaque colonne avec les valeurs max correspondantes
imputers = {col: SimpleImputer(strategy='constant', fill_value=max_value) for col, max_value in max_values.items()}
# Appliquer l'imputation à chaque colonne pour les données d'entraînement
for col, imputer in imputers.items():
    train_df[col] = imputer.fit_transform(train_df[[col]])


# Caractéristiques (features) pour le modèle
X = train_df[['home_team_encoded', 'away_team_encoded', 'Home_Age_Moyen', 'Home_Taille_Moyenne',
              'Home_Selection_Moyenne', 'Home_Buts_Moyens', 'Away_Age_Moyen',
              'Away_Taille_Moyenne', 'Away_Selection_Moyenne', 'Away_Buts_Moyens',
              'Home_Rank', 'Away_Rank']]
y = train_df['match_result']



# Split des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraîner le modèle

# Définir la grille de paramètres pour GridSearchCV
param_grid = {
    'n_estimators': [200, 250, 300],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}

# Initialiser le modèle RandomForestClassifier
match_model = RandomForestClassifier(random_state=42)

# Initialiser GridSearchCV avec le modèle et la grille de paramètres
grid_search = GridSearchCV(estimator=match_model, param_grid=param_grid, cv=5, scoring='accuracy')

# Ajuster GridSearchCV aux données d'entraînement
grid_search.fit(X_train, y_train)

# Meilleurs paramètres trouvés par GridSearchCV
print(f'Best parameters found: {grid_search.best_params_}')
print(f'Best cross-validated accuracy: {grid_search.best_score_}')

# Obtenir les résultats de GridSearchCV
results = grid_search.cv_results_

# Extraire les paramètres et les scores
mean_test_scores = results['mean_test_score']
std_test_scores = results['std_test_score']
params = results['params']

# Créer une liste de noms de paramètres
param_names = list(params[0].keys())

# Initialiser une figure
plt.figure(figsize=(15, 10))

# Pour chaque paramètre, tracer le score moyen de test en fonction de la valeur du paramètre
for i, param_name in enumerate(param_names):
    # Obtenir les valeurs uniques du paramètre
    param_values = sorted(set([param[param_name] for param in params]))
    
    # Initialiser des listes pour les scores moyens et les écarts-types pour chaque valeur unique du paramètre
    scores = []
    stds = []
    
    # Pour chaque valeur unique du paramètre, obtenir les scores moyens et les écarts-types correspondants
    for value in param_values:
        mean_score = np.mean([mean_test_scores[j] for j in range(len(params)) if params[j][param_name] == value])
        std_score = np.mean([std_test_scores[j] for j in range(len(params)) if params[j][param_name] == value])
        scores.append(mean_score)
        stds.append(std_score)
    
    # Tracer les scores moyens de test pour chaque valeur unique du paramètre
    plt.subplot(len(param_names), 1, i + 1)
    plt.errorbar(param_values, scores, yerr=stds, fmt='-o', capsize=5)
    plt.title(f'Mean test scores for {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Mean test score')
    plt.grid()

# Ajuster la mise en page
plt.tight_layout()
plt.show()

# Utiliser les meilleurs paramètres pour entraîner le modèle final
best_rf = grid_search.best_estimator_
best_rf.fit(X_test, y_test)

# Prédire les résultats pour les nouveaux matchs (predict_df)
X_predict = predict_df[['home_team_encoded', 'away_team_encoded', 'Home_Age_Moyen', 'Home_Taille_Moyenne',
                        'Home_Selection_Moyenne', 'Home_Buts_Moyens', 'Away_Age_Moyen',
                        'Away_Taille_Moyenne', 'Away_Selection_Moyenne', 'Away_Buts_Moyens',
                        'Home_Rank', 'Away_Rank']]
predict_df['match_result_pred'] = best_rf.predict(X_predict)


"""
match_model = RandomForestClassifier(
    criterion='gini',
    max_depth=6,
    n_estimators=300,
    random_state=42
)




match_model.fit(X_train, y_train)

# Évaluer le modèle avec validation croisée

cross_val_scores = cross_val_score(match_model, X_train, y_train, cv=5, scoring='accuracy')
print(f'Cross-validated accuracy: {cross_val_scores.mean()}')


# Évaluer le modèle sur l'ensemble de test
test_accuracy = match_model.score(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')


# Predire les resultats pour les nouveaux matchs (predict_df)
X_predict = predict_df[['home_team_encoded', 'away_team_encoded', 'Home_Age_Moyen', 'Home_Taille_Moyenne',
                        'Home_Selection_Moyenne', 'Home_Buts_Moyens', 'Away_Age_Moyen',
                        'Away_Taille_Moyenne', 'Away_Selection_Moyenne', 'Away_Buts_Moyens',
                        'Home_Rank', 'Away_Rank']]
predict_df['match_result_pred'] = match_model.predict(X_predict)
"""

# Colonne indiquant le gagnant ou Nul si match nul
predict_df['Winning_Team'] = predict_df.apply(
    lambda row: row['Home_Team'] if row['match_result_pred'] == 1 else
                (row['Away_Team'] if row['match_result_pred'] == 0 else 'Nul'),
    axis=1
)

# Selection de colonnes
poules_predict_df = predict_df[['Date', 'Home_Team', 'Away_Team', 'Resultat_Home_Team', 'Resultat_Away_Team', 'match_result_pred', 'Winning_Team']]

# Sauvegarder les résultats prédits
poules_predict_df.to_csv('predictions_poules.csv', index=False)
print("Les résultats prédits pour les matchs de poules ont été sauvegardéss dans 'predictions_poules.csv'.")

# Affichage des résultats des prédictions de poules
print("\nRésultats prédits pour les matchs de poules :\n", poules_predict_df)

# ----------------------------------------------------------------------------------------------------------------------
# CLASSEMENT POULES
# ----------------------------------------------------------------------------------------------------------------------
# Définir les poules et les équipes qui y appartiennent
poule_A_teams = ['Espagne', 'Croatie', 'Allemagne', 'Slovenie', 'Suede', 'Japon']
poule_B_teams = ['Danemark', 'Norvege', 'Hongrie', 'France', 'Egypte', 'Argentine']

# Initialiser un dictionnaire pour stocker les points des équipes
points = {team: 0 for team in poule_A_teams + poule_B_teams}

# Parcourir les résultats des matchs et mettre à jour les points // Attention il n'y a pas le cas du match nul
# Attribuer les points en fonction des résultats prédits
for index, row in poules_predict_df.iterrows():
    home_team = row['Home_Team']
    away_team = row['Away_Team']
    result = row['match_result_pred']

    if result == 1:  # Victoire de l'équipe à domicile
        points[home_team] += 2
    elif result == 0:  # Défaite de l'équipe à domicile
        points[away_team] += 2
    else:  # Match nul
        points[home_team] += 1
        points[away_team] += 1

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

# ----------------------------------------------------------------------------------------------------------------------
# QUARTS
# ----------------------------------------------------------------------------------------------------------------------
# Faire maintenant les prédictions sur quarts_de_finale_df, il faut ajouter les infos pour faire la prédiction

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
quarts_de_finale_data['match_result_pred'] = best_rf.predict(X_quarts)

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


# ----------------------------------------------------------------------------------------------------------------------
# DEMIS
# ----------------------------------------------------------------------------------------------------------------------

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
demis_finale_data['match_result_pred'] = best_rf.predict(X_demis)

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


# ----------------------------------------------------------------------------------------------------------------------
# FINALES
# ----------------------------------------------------------------------------------------------------------------------

finale_match = [(demis_predict_df.loc[0, 'Winning_Team'], demis_predict_df.loc[1, 'Winning_Team'])]

finale_data = ajout_variables_train(finale_match, train_df)
finale_data = pd.DataFrame(finale_data)

# Caractéristiques pour la prédiction
X_finale = finale_data[['home_team_encoded', 'away_team_encoded',
                        'Home_Age_Moyen', 'Home_Taille_Moyenne', 'Home_Selection_Moyenne', 'Home_Buts_Moyens',
                        'Away_Age_Moyen', 'Away_Taille_Moyenne', 'Away_Selection_Moyenne', 'Away_Buts_Moyens',
                        'Home_Rank', 'Away_Rank']]

# Utiliser le modèle déjà entraîné pour prédire les résultats
finale_data['match_result_pred'] = best_rf.predict(X_finale)

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

