# Scraping des statistiques JO (site wikipédia)
# Emie
# Création : 04/06/2024
# Dernière modif : 19/06/2024

import requests
from bs4 import BeautifulSoup
import numpy as np
from fonctions_utiles import *

# URL de la page Wikipédia
url = "https://fr.wikipedia.org/wiki/Handball_aux_Jeux_olympiques#Hommes"

# Envoyer une requête HTTP pour obtenir le contenu de la page
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# -------------------------------------------------
# Palmarès JO
# -------------------------------------------------

# Trouver le tableau du palmarès des hommes
table = soup.find('table', {'class': 'wikitable alternance'})

# Extraire les lignes du tableau
rows = []
for row in table.find_all('tr')[1:]:
    cells = row.find_all('td')
    row_data = [cell.text.strip() for cell in cells]
    rows.append(row_data)

# Créer un dataframe avec les données extraites
df_palmares = pd.DataFrame(rows, columns=["Année", "Ville", "Or", "Argent", "Bronze"])

# Enlever les années futures
df_palmares = retrait_annees_futures(df_palmares)

# Afficher le df
print(df_palmares.to_string())

# Sauvegarder les données dans un fichier CSV
sauvegarde_df_csv(df_palmares, "resultats_scraping/palmares_handball_jo_hommes.csv")

# -------------------------------------------------
#  Tableau des médailles
# -------------------------------------------------
table = soup.find('table', {'class': 'wikitable sortable'})

# Extraire les lignes du tableau
rows = []
for row in table.find_all('tr')[1:]:
    cells = row.find_all(['th', 'td'])
    row_data = [cell.text.strip() for cell in cells]

    # Gérer les cas où le rang est en double (ou plus)
    if len(row_data) == 6:  # Cas où le rang est fusionné avec la ligne précédente
        row_data.insert(0, previous_row[0])
    elif len(row_data) < 6:  # Cas où les données sont incomplètes, continuez la ligne précédente
        continue

    rows.append(row_data)
    previous_row = row_data

# Créer un dataframe avec les données extraites
df_tableau_medailles = pd.DataFrame(rows, columns=["Rang", "Nation", "Or", "Argent", "Bronze", "Total", "Der"])
df_tableau_medailles = nettoyage_colonne_nation(df_tableau_medailles)

# Afficher le df
print(df_tableau_medailles.to_string())

# Sauvegarder les données dans un fichier CSV
sauvegarde_df_csv(df_tableau_medailles, "resultats_scraping/tableau_medailles_jo_hommes.csv")

# -------------------------------------------------
# Bilan par nations
# -------------------------------------------------
table = soup.find(id="Bilan_par_nation").find_next('table')

# Noms des colonnes
headers = []
for th in table.find_all('th'):
    headers.append(th.text.strip())
# ou en plus court : headers = [th.text.strip() for th in table.find_all('th')]

# Récup des infos
countries = []

for row in table.find_all('tr')[1:]:
    cells = row.find_all('td')
    country_cells = []
    for cell in cells:
        # Interpréter les images de médailles et autre code (si le <td> contient un <span> avec 'data-sort-value', si = 0 or, 88 forfait, etc.)
        span = cell.find('span', {'data-sort-value': True})
        if span:
            if span['data-sort-value'] == '0':
                cell_text = '1'
            elif span['data-sort-value'] == '30':
                cell_text = 'boycott'
            elif span['data-sort-value'] == '98':
                cell_text = np.nan
            else:
                cell_text = span['data-sort-value']
        else:
            cell_text = cell.text.strip()

        if cell.has_attr('colspan'):
            colspan = int(cell['colspan'])
            for i in range(0, colspan):
                country_cells.append(cell_text)
        else:
            country_cells.append(cell_text)

    countries.append(country_cells)

# Df
df_bilan_nation = pd.DataFrame(countries, columns=headers[:len(countries[0])])

# Nettoyage de la colonne nation
df_bilan_nation = nettoyage_colonne_nation(df_bilan_nation)

# Retrait des lignes totalement vides
df_bilan_nation = df_bilan_nation.dropna(how='all')

# Afficher le df
print(df_bilan_nation.to_string())

# Sauvegarder les données dans un fichier CSV
sauvegarde_df_csv(df_bilan_nation, "resultats_scraping/bilan_nations_jo_hommes.csv")
