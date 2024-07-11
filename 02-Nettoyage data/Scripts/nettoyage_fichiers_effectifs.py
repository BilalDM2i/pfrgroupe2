# Nettoyage des fichiers effectifs
# Emie
# Création : 11/06/2024
# Dernière modif :

import os
import pandas as pd
import datetime
import dateparser

# Déclaration de variables
path_raw_data = "fichiers_effectifs/raw/"
path_cleaned_data = "fichiers_effectifs/cleaned/"


# Déclaration de fonctions
def calcul_age(birth_date):
    today = datetime.date.today()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))


# Nettoyage
for filename in os.listdir(path_raw_data):
    cleaned_file = pd.read_csv(os.path.join(path_raw_data, filename))

    # Si fichier France on retire les colonnes poids et "depuis" car on l'a uniquement pour ce pays
    if "france" in filename:
        cleaned_file.drop("Poids", axis=1, inplace=True)
        cleaned_file.drop("Depuis", axis=1, inplace=True)

    # Si fichier Japon on retire la chaine "(en)" à la fin de chaque nom de joueur
    if "japon" in filename:
        cleaned_file['Nom'] = cleaned_file['Nom'].str.replace("(en)", "").str.strip()

    # Pour tous les fichiers
    # Renommer la colonne P en poste et la colonne Sél en Sélection
    cleaned_file.rename(columns={'P.': 'Poste'}, inplace=True)
    cleaned_file.rename(columns={'Sél.': 'Sélection'}, inplace=True)

    # Calculer l’âge à la place de la date de naissance
    # Nettoyage de la colonne : retrait des caractères après la DDN + des espaces
    cleaned_file['Date de naissance'] = cleaned_file['Date de naissance'].str.split('(').str[0].str.strip()
    # Conversion de la colonne en format date
    cleaned_file['Date de naissance'] = cleaned_file['Date de naissance'].apply(lambda x: dateparser.parse(x))
    # Calcul de l'âge
    cleaned_file['Age'] = cleaned_file['Date de naissance'].apply(
        lambda x: calcul_age(x) if pd.notnull(x) else None)
    cleaned_file.drop("Date de naissance", axis=1, inplace=True)

    # Mettre la taille en numérique (enlever le “m” à la fin de la chaine de caractère avant)
    cleaned_file['Taille'] = cleaned_file['Taille'].str.replace("m", "").str.replace(",", ".").str.strip()
    cleaned_file['Taille'] = pd.to_numeric(cleaned_file['Taille'])

    print(f"Fichier nettoyé :\n {cleaned_file.to_string()}")

    filename2 = filename.split(".csv")[0] + "_cleaned" + ".csv"
    cleaned_file.to_csv(os.path.join(path_cleaned_data, filename2), index=False)
    print(f"Les données ont été sauvegardées dans {path_cleaned_data}")
