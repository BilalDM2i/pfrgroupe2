# Nettoyage du fichier bilan par nations JO
# Emie
# Création : 19/06/2024
# Dernière modif : 27/06/2024

import pandas as pd
import numpy as np
from fonctions_utiles import *

df = pd.read_csv("resultats_scraping/bilan_nations_jo_hommes.csv")
print(df.to_string())

# pour la Croatie mettre les données de la Yougoslavie pour période où elle était membre de la Yougoslavie
df.iloc[13, 2:7] = df.iloc[7, 2:7]
print(df.to_string())

# conserver uniquement les nations qualifiées pour 2024
# pays_qualifies = ("Hongrie", "Argentine", "Croatie", "Danemark", "France", "Suède",
#                   "Allemagne", "Égypte", "Espagne", "Japon", "Norvège", "Slovénie")
# df = df.loc[df["Nation"].isin(pays_qualifies)]
# print(df.to_string())

# remplacer les boycott par nan
df.replace("boycott", np.nan, inplace=True)
print(df.to_string())

# année sur 4 chiffres : reprise du code de Cécile
# on selectionne la 1e ligne avec les entetes
years = df.columns

# on va modifier pour les valeurs numeriques les anne yy en yyyy. et on distingue 19yy e 20yy
new_years = []
for year in years:
    if year.isdigit():
        if 0 <= int(year) <= 50:
            new_year = '20' + year
        else:
            new_year = '19' + year
    else:
        new_year = year
    new_years.append(new_year)

print(new_years)
df.columns = new_years
print(df.head(20))
print(df.shape)

# Sauvegarder les données dans un fichier CSV
sauvegarde_df_csv(df, "bilan_nations_jo_hommes_cleaned.csv")

print(df.to_string())