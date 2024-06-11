import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import numpy as np

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
# print(table)

# Extraire les lignes du tableau
rows = []
for row in table.find_all('tr')[1:]:
    cells = row.find_all('td')
    row_data = [cell.text.strip() for cell in cells]
    rows.append(row_data)

# Créer un dataframe avec les données extraites
df_palmares = pd.DataFrame(rows, columns=["Année", "Ville", "Or", "Argent", "Bronze"])

# Enlever les années futures
df_palmares['Année'] = pd.to_numeric(df_palmares['Année'])
date = datetime.date.today()
df_palmares = df_palmares[df_palmares['Année'] < date.year]
print(df_palmares.to_string())

# Sauvegarder les données dans un fichier CSV
file_path_palmares = "resultats_scraping/palmares_handball_jo_hommes.csv"
df_palmares.to_csv(file_path_palmares, index=False)
print(f"Les données ont été sauvegardées dans {file_path_palmares}")

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
df_tableau_medailles['Nation'] = df_tableau_medailles['Nation'].str.replace(r"\(.*?\)", "", regex=True).str.strip()

# Afficher le DataFrame
print(df_tableau_medailles)

# Sauvegarder les données dans un fichier CSV
file_path_tab_medailles = "resultats_scraping/tableau_medailles_jo_hommes.csv"
df_tableau_medailles.to_csv(file_path_tab_medailles, index=False)
print(f"Les données ont été sauvegardées dans {file_path_tab_medailles}")

# -------------------------------------------------
# Bilan par nations
# -------------------------------------------------
table = soup.find(id="Bilan_par_nation").find_next('table')

# Noms des colonnes
headers = []
for th in table.find_all('th'):
    headers.append(th.text.strip())
# ou en plus court : headers = [th.text.strip() for th in table.find_all('th')]

# Initialiser une liste pour stocker les lignes et un dictionnaire pour les cellules fusionnées
rows = []
rowspan_dict = {}

# Extraire les lignes du tableau
for row in table.find_all('tr')[1:]:
    cells = row.find_all(['td', 'th'])
    row_data = []
    col_index = 0

    while col_index < len(cells):
        # Traiter les cellules fusionnées verticalement
        while col_index in rowspan_dict and rowspan_dict[col_index] > 0:
            row_data.append(rows[-1][col_index])
            rowspan_dict[col_index] -= 1
            col_index += 1

        cell = cells[col_index]

        # Vérifier si le <td> contient un <span> avec 'data-sort-value'
        # si = 1 or, 2 argent, 3 bronze, 30 boycott, 98 na
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
            cell_text = cell.get_text(strip=True)

        # Traiter les cellules fusionnées horizontalement et verticalement
        if cell.has_attr('rowspan'):
            rowspan_dict[col_index] = int(cell['rowspan']) - 1

        if cell.has_attr('colspan'):
            colspan = int(cell['colspan'])
            row_data.extend([cell_text] * colspan)
            col_index += colspan
        else:
            row_data.append(cell_text)
            col_index += 1

    rows.append(row_data)

df_bilan_nation = pd.DataFrame(rows, columns=headers[:len(rows[0])])

# Nettoyage de la colonne nation
df_bilan_nation['Nation'] = df_bilan_nation['Nation'].str.replace(r"\[.*?\]", "",
                                                                  regex=True)  # Supprimer les annotations de référence
df_bilan_nation['Nation'] = df_bilan_nation['Nation'].str.replace(r"/", "", regex=True)  # Supprimer les barres obliques
df_bilan_nation['Nation'] = df_bilan_nation['Nation'].str.strip()  # Enlever les espaces blancs au début et à la fin

# Retrait lignes totalement vides
df_bilan_nation = df_bilan_nation.dropna(how='all')

print(df_bilan_nation)

# Sauvegarder les données dans un fichier CSV
file_path_bilan_nations = "resultats_scraping/bilan_nations_jo_hommes.csv"
df_bilan_nation.to_csv(file_path_bilan_nations, index=False)
print(f"Les données ont été sauvegardées dans {file_path_bilan_nations}")
