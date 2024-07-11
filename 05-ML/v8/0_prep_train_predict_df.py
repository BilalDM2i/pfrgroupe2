# Préparation des dataframes "train" et "predict" utilisés dans les modèles (cf. scripts 1_***_predictions)
# Date de création : 03/07/2024
# Date de dernière modification : 10/07/2024

import pandas as pd
from unidecode import unidecode

# -----------------------------------------------------------------------------------------------------
# Fichier sur les joueurs
# (attention : on a scrappé les infos uniquement pour les 12 équipes qualifiées)
# -----------------------------------------------------------------------------------------------------

#  Importation du fichier
players = pd.read_csv("_Players_avec_ID.csv", encoding="latin1")

print("Fichier joueurs")
print(players.describe())
print(players.dtypes)

# Nettoyage de la variable Pays (espace, accent)
players['Pays'] = players['Pays'].str.strip()
players['Pays'] = players['Pays'].apply(lambda x: unidecode(x))

# Calcul des moyennes, par équipe, sur les variables âge, taille, nb sélections et nb buts
age_moy = players.groupby('Pays')['Age'].mean().round(1)
taille_moy = players.groupby('Pays')['Taille'].mean().round(1)
nb_sel_moy = players.groupby('Pays')['Selection'].mean().round(1)
nb_buts_moy = players.groupby('Pays')['Buts'].mean().round(1)

print("Age moyen", age_moy)
print("Taille moyenne", taille_moy)
print("Nb de sélections moyen", nb_sel_moy)
print("Nb buts moyen", nb_buts_moy)

# -----------------------------------------------------------------------------------------------------
# Fichier sur le rang mondial des équipes au 1er janvier de chaque année entre 2000 et 2024
# -----------------------------------------------------------------------------------------------------
rankings = pd.read_csv('_classements_handball_2000_2024.csv')

print("Fichier rangs")
print(rankings.describe())
print(rankings.dtypes)

rankings['Date'] = pd.to_datetime(rankings['Date'], format='%d/%m/%Y')
rankings['Year'] = rankings['Date'].dt.year
rankings.drop(columns=['Date'], inplace=True)

# -----------------------------------------------------------------------------------------------------
# Fichier train de base
# -----------------------------------------------------------------------------------------------------
train_df = pd.read_csv('_Matchs_toutes_competitions_transforme.csv')

print("Fichier train (matchs antérieurs)")
print(train_df.describe())
print(train_df.dtypes)


# Fonction pour faire la fusion entre train ou predict df et les fichiers joueurs et rangs
def fusion(df):
    # Variable date / année pour la fusion
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year

    # Nettoyage de la variable Pays (espace, accent)
    df['Home_Team'] = df['Home_Team'].str.strip()
    df['Home_Team'] = df['Home_Team'].apply(lambda x: unidecode(x))

    df['Away_Team'] = df['Away_Team'].str.strip()
    df['Away_Team'] = df['Away_Team'].apply(lambda x: unidecode(x))

    # Fusion avec les moyennes calculées, pour les équipes Home et ensuite pour les équipes Away
    df = df.merge(age_moy, left_on='Home_Team', right_on='Pays', how='left')
    df = df.merge(taille_moy, left_on='Home_Team', right_on='Pays', how='left')
    df = df.merge(nb_sel_moy, left_on='Home_Team', right_on='Pays', how='left')
    df = df.merge(nb_buts_moy, left_on='Home_Team', right_on='Pays', how='left')

    # On renomme les colonnes
    df.rename(columns={
        'Age': 'Home_Age_Moyen',
        'Taille': 'Home_Taille_Moyenne',
        'Selection': 'Home_Selection_Moyenne',
        'Buts': 'Home_Buts_Moyens'
    }, inplace=True)

    # Fusion pour away team
    df = df.merge(age_moy, left_on='Away_Team', right_on='Pays', how='left')
    df = df.merge(taille_moy, left_on='Away_Team', right_on='Pays', how='left')
    df = df.merge(nb_sel_moy, left_on='Away_Team', right_on='Pays', how='left')
    df = df.merge(nb_buts_moy, left_on='Away_Team', right_on='Pays', how='left')

    # On renomme les colonnes
    df.rename(columns={
        'Age': 'Away_Age_Moyen',
        'Taille': 'Away_Taille_Moyenne',
        'Selection': 'Away_Selection_Moyenne',
        'Buts': 'Away_Buts_Moyens'
    }, inplace=True)

    # On fusionne avec le rang
    df = df.merge(rankings, left_on=['Year', 'Home_Team'], right_on=['Year', 'Nation'], how='left')
    df.rename(columns={'Rank': 'Home_Rank'}, inplace=True)
    # On supprime les colonnes inutiles
    df.drop(columns=['Nation', 'Points', 'Match_count'], inplace=True)

    # Idem pour away team
    df = df.merge(rankings, left_on=['Year', 'Away_Team'], right_on=['Year', 'Nation'], how='left')
    df.rename(columns={'Rank': 'Away_Rank'}, inplace=True)
    df.drop(columns=['Nation', 'Points', 'Match_count'], inplace=True)

    # On supprime les colonnes inutiles
    df.drop(columns=['Year'], inplace=True)

    # Affichage du df
    print(df.head().to_string())

    # # Vérif des valeurs manquantes (pendant le dev uniquement)
    # df_na = df[(df['Home_Rank'].isnull())]
    # unique_countries_na = df_na['Home_Team'].unique()
    # print(unique_countries_na)
    # df_na = df[(df['Away_Rank'].isnull())]
    # unique_countries_na = df_na['Away_Team'].unique()
    # print(unique_countries_na)

    return df


# Appel de la fonction
train_df = fusion(df=train_df)

# -----------------------------------------------------------------------------------------------------
# Fichier predict de base
# -----------------------------------------------------------------------------------------------------

predict_df = pd.read_csv('_Matchs_uniquement_2024_transforme.csv')

print("Fichier predict (matchs de poules JO 2024 à venir)")
print(predict_df.describe())
print(predict_df.dtypes)

# Appel de la fonction
predict_df = fusion(df=predict_df)

# Vérification du type des données
print(train_df.dtypes)
print(predict_df.dtypes)

#  Modif du type pour le rang pour train_df
train_df["Home_Rank"] = train_df["Home_Rank"].fillna(0.0).astype('int64')
train_df["Away_Rank"] = train_df["Away_Rank"].fillna(0.0).astype('int64')

# Vérif
print(train_df.dtypes)

# Sauvegarde des df
train_df.to_csv('_train_df.csv', index=False)
predict_df.to_csv('_predict_df.csv', index=False)
