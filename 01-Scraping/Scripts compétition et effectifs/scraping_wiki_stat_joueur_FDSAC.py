# Scraping des statistiques joueurs (site wikipédia)
# Emie
# Création : 11/06/2024
# Dernière modif : 19/06/2024

import requests
from bs4 import BeautifulSoup
from fonctions_utiles import *

# Pour France, Danemark, Suède, Argentine, Croatie (j'ai fait les deux derniers pays car Bilal a eu un souci donc je les relance)
urls = {"France": "https://fr.wikipedia.org/wiki/%C3%89quipe_de_France_masculine_de_handball",
        "Danemark": "https://fr.wikipedia.org/wiki/%C3%89quipe_du_Danemark_masculine_de_handball#Effectif_actuel",
        "Suède": "https://fr.wikipedia.org/wiki/%C3%89quipe_de_Su%C3%A8de_masculine_de_handball",
        "Argentine": "https://fr.wikipedia.org/wiki/%C3%89quipe_d%27Argentine_masculine_de_handball#Effectif",
        "Croatie": "https://fr.wikipedia.org/wiki/%C3%89quipe_de_Croatie_masculine_de_handball#Effectif_actuel",
        }

for pays, url in urls.items():
    print(f"Pays : {pays}")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    if pays == "France":
        tables = soup.find_all('table', {'class': 'wikitable'})
        table = tables[15]
    else:
        table = soup.find('table', {'class': 'sortable'})

    # Noms des colonnes
    headers = [th.text.strip() for th in table.find_all('th')]

    # Extraire les lignes du tableau
    rows = []
    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        row_data = [cell.text.strip() for cell in cells]
        rows.append(row_data)

    # Créer un dataframe avec les données extraites
    df_effectif = pd.DataFrame(rows)
    df_effectif.columns = headers[0:df_effectif.shape[1]]

    # Nettoyage colonne date de naissance
    df_effectif['Date de naissance'] = df_effectif['Date de naissance'].str.split('(').str[0].str.strip()

    # Retrait des lignes totalement vides
    df_effectif.dropna(how='all', inplace=True)

    # Afficher le df
    print(df_effectif.to_string())

    # Sauvegarder les données dans un fichier CSV
    sauvegarde_df_csv(df_effectif, "resultats_scraping/effectif_" + pays + "_hommes.csv")
