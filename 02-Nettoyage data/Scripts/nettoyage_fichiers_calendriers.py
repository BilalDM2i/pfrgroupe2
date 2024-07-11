# Nettoyage des fichiers calendriers
# Emie
# Création : 11/06/2024
# Dernière modif :

import os
import pandas as pd
import dateparser

# Déclaration de variables
path_raw_data = "fichiers_calendriers/raw/"
path_cleaned_data = "fichiers_calendriers/cleaned/"

# Nettoyage : suppression de la colonne score et création d'une colonne date heure du match sous format date python
for filename in os.listdir(path_raw_data):
    cleaned_file = pd.read_csv(os.path.join(path_raw_data, filename))

    cleaned_file.drop("Score", axis=1, inplace=True)

    cleaned_file['Date_heure_match'] = cleaned_file['Date'] + " " + cleaned_file['Heure']
    cleaned_file['Date_heure_match'] = cleaned_file['Date_heure_match'].apply(lambda x: dateparser.parse(x))

    print(f"Fichier nettoyé :\n {cleaned_file.to_string()}")

    filename2 = filename.split(".csv")[0] + "_cleaned" + ".csv"
    cleaned_file.to_csv(os.path.join(path_cleaned_data, filename2), index=False)
    print(f"Les données ont été sauvegardées dans {path_cleaned_data}")
