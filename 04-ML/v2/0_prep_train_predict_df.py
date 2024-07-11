# Prépa fichier train et predict df pour intégrer dans le modèle
# Emie
# Création 03/07/2024
# Modif 04/07/2024

import pandas as pd
from unidecode import unidecode

# -----------------------------------------------------------------------------------------------------
# Fichier sur les joueurs (attention on a scrappé les infos uniquement pour les 12 équipes qualifiées)
# -----------------------------------------------------------------------------------------------------

joueurs = pd.read_csv("Players_avec_ID.csv", encoding="latin1")
# nettoyage de la variable Pays : en minuscule, sans accent, pour que ça soit facilement fusionnable
joueurs['Pays'] = joueurs['Pays'].str.strip()
joueurs['Pays'] = joueurs['Pays'].apply(lambda x: unidecode(x))

# calcul âge, taille, nb Selections et nb buts moyens par équipe
age_moy = joueurs.groupby('Pays')['Age'].mean()
taille_moy = joueurs.groupby('Pays')['Taille'].mean()
nb_sel_moy = joueurs.groupby('Pays')['Selection'].mean()
nb_buts_moy = joueurs.groupby('Pays')['Buts'].mean()

# -----------------------------------------------------------------------------------------------------
# Fichier sur le rang mondial des équipes au 1er juillet 2024
# -----------------------------------------------------------------------------------------------------

rang = pd.read_csv("../classement_handball_01072024.csv")
# nettoyage de la variable Nation : en minuscule, sans accent, pour que ça soit facilement fusionnable
rang['Nation'] = rang['Nation'].str.strip()
rang['Nation'] = rang['Nation'].apply(lambda x: unidecode(x))

# -----------------------------------------------------------------------------------------------------
# Fichier train de base
# -----------------------------------------------------------------------------------------------------

train_df = pd.read_csv('Matchs_toutes_competitions_transforme.csv')

# nettoyage des variables pays : en minuscule, sans accent, pour que ça soit facilement fusionnable
train_df['Home_Team'] = train_df['Home_Team'].str.strip()
train_df['Home_Team'] = train_df['Home_Team'].apply(lambda x: unidecode(x))

train_df['Away_Team'] = train_df['Away_Team'].str.strip()
train_df['Away_Team'] = train_df['Away_Team'].apply(lambda x: unidecode(x))

# fusion avec les moyennes calculées, pour les équipes Home et ensuite pour les équipes Away
train_df = train_df.merge(age_moy, left_on='Home_Team', right_on='Pays', how='left')
train_df = train_df.merge(taille_moy, left_on='Home_Team', right_on='Pays', how='left')
train_df = train_df.merge(nb_sel_moy, left_on='Home_Team', right_on='Pays', how='left')
train_df = train_df.merge(nb_buts_moy, left_on='Home_Team', right_on='Pays', how='left')

# on renomme les colonnes
train_df.rename(columns={
    'Age': 'Home_Age_Moyen',
    'Taille': 'Home_Taille_Moyenne',
    'Selection': 'Home_Selection_Moyenne',
    'Buts': 'Home_Buts_Moyens'
}, inplace=True)

# fusion pour away team
train_df = train_df.merge(age_moy, left_on='Away_Team', right_on='Pays', how='left')
train_df = train_df.merge(taille_moy, left_on='Away_Team', right_on='Pays', how='left')
train_df = train_df.merge(nb_sel_moy, left_on='Away_Team', right_on='Pays', how='left')
train_df = train_df.merge(nb_buts_moy, left_on='Away_Team', right_on='Pays', how='left')

# on renomme les colonnes
train_df.rename(columns={
    'Age': 'Away_Age_Moyen',
    'Taille': 'Away_Taille_Moyenne',
    'Selection': 'Away_Selection_Moyenne',
    'Buts': 'Away_Buts_Moyens'
}, inplace=True)

# on fusionne avec le rang
train_df = train_df.merge(rang, left_on='Home_Team', right_on='Nation', how='left')

# on renomme
train_df.rename(columns={
    'Rank': 'Home_Rank'
}, inplace=True)

# idem pour away team
train_df = train_df.merge(rang, left_on='Away_Team', right_on='Nation', how='left')
train_df.rename(columns={
    'Rank': 'Away_Rank'
}, inplace=True)

# on supprime les colonnes inutiles
train_df.drop(columns=["Nation_x", "Nation_y", "Points_x", "Points_y", "Match_count_x", "Match_count_y"], inplace=True)

# affichage du df
print(train_df.head().to_string())

# sauvegarde du df
train_df.to_csv('train_df.csv', index=False)


#-----------------------------------------------------------------------------------------------------
# Fichier predict de base
#-----------------------------------------------------------------------------------------------------
predict_df = pd.read_csv('Matchs_uniquement_2024_transforme.csv')

# nettoyage des variables pays : en minuscule, sans accent, pour que ça soit facilement fusionnable
predict_df['Home_Team'] = predict_df['Home_Team'].str.strip()
predict_df['Home_Team'] = predict_df['Home_Team'].apply(lambda x: unidecode(x))

predict_df['Away_Team'] = predict_df['Away_Team'].str.strip()
predict_df['Away_Team'] = predict_df['Away_Team'].apply(lambda x: unidecode(x))

# fusion avec les moyennes calculées, pour les équipes Home et ensuite pour les équipes Away
predict_df = predict_df.merge(age_moy, left_on='Home_Team', right_on='Pays', how='left')
predict_df = predict_df.merge(taille_moy, left_on='Home_Team', right_on='Pays', how='left')
predict_df = predict_df.merge(nb_sel_moy, left_on='Home_Team', right_on='Pays', how='left')
predict_df = predict_df.merge(nb_buts_moy, left_on='Home_Team', right_on='Pays', how='left')

# on renomme les colonnes
predict_df.rename(columns={
    'Age': 'Home_Age_Moyen',
    'Taille': 'Home_Taille_Moyenne',
    'Selection': 'Home_Selection_Moyenne',
    'Buts': 'Home_Buts_Moyens'
}, inplace=True)

# fusion pour away team
predict_df = predict_df.merge(age_moy, left_on='Away_Team', right_on='Pays', how='left')
predict_df = predict_df.merge(taille_moy, left_on='Away_Team', right_on='Pays', how='left')
predict_df = predict_df.merge(nb_sel_moy, left_on='Away_Team', right_on='Pays', how='left')
predict_df = predict_df.merge(nb_buts_moy, left_on='Away_Team', right_on='Pays', how='left')

# on renomme les colonnes
predict_df.rename(columns={
    'Age': 'Away_Age_Moyen',
    'Taille': 'Away_Taille_Moyenne',
    'Selection': 'Away_Selection_Moyenne',
    'Buts': 'Away_Buts_Moyens'
}, inplace=True)

# on fusionne avec le rang
predict_df = predict_df.merge(rang, left_on='Home_Team', right_on='Nation', how='left')

# on renomme
predict_df.rename(columns={
    'Rank': 'Home_Rank'
}, inplace=True)

# idem pour away team
predict_df = predict_df.merge(rang, left_on='Away_Team', right_on='Nation', how='left')
predict_df.rename(columns={
    'Rank': 'Away_Rank'
}, inplace=True)

# on supprime les colonnes inutiles
predict_df.drop(columns=["Nation_x", "Nation_y", "Points_x", "Points_y", "Match_count_x", "Match_count_y"], inplace=True)

# affichage du df
print(predict_df.head().to_string())

# sauvegarde du df
predict_df.to_csv('predict_df.csv', index=False)