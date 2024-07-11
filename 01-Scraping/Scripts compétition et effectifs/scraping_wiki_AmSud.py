import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import numpy as np
import sys
import os


# URL de la page Wikipédia
url = "https://fr.wikipedia.org/wiki/Championnat_d%27Am%C3%A9rique_du_Sud_et_centrale_masculin_de_handball"

# Envoyer une requête HTTP pour obtenir le contenu de la page
try:
    response = requests.get(url)
    response.raise_for_status()
    response.encoding = 'utf-8'  # Assurez-vous que l'encodage est correct
except requests.exceptions.RequestException as e:
    print(f"Erreur lors de la requête HTTP : {e}")
    sys.exit()

soup = BeautifulSoup(response.text, 'html.parser')


# Trouver tous les tableaux de la page
tables = soup.find_all('table')

# Vérifier le nombre de tableaux trouvés
print(f"Nombre de tableaux trouvés : {len(tables)}")

# Supposons que le tableau désiré est le 11ème tableau (index 10)
table_index = 4

# Utiliser pandas pour lire le tableau HTML
try:
    df = pd.read_html(str(tables[table_index]))[0]
except ValueError as e:
    print(f"Erreur lors de la lecture du tableau : {e}")
    sys.exit()

# Afficher les premières lignes et les colonnes du DataFrame pour vérifier la structure
print(df)
print(df.columns)

#%%

# Initialiser une liste pour stocker les lignes et un dictionnaire pour les cellules fusionnées

table=tables[table_index]
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

        if col_index >= len(cells):
            break

        cell = cells[col_index]

        # Vérifier si le <td> contient un <span> avec 'data-sort-value'
        span = cell.find('span', {'data-sort-value': True})
        if span:
            cell_text = span['data-sort-value']
            if cell_text == '0':
                cell_text = '1'
            elif cell_text == '30':
                cell_text = 'boycott'
            elif cell_text == '98':
                cell_text = np.nan
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

# Vérifier et ajuster les colonnes avant de créer le DataFrame
max_cols = max(len(row) for row in rows)
print(f"Nombre maximum de colonnes: {max_cols}")

# Noms des colonnes
headers = [th.get_text(strip=True) for th in table.find_all('th')]
headers = headers[:max_cols]

# Ajuster les lignes pour qu'elles aient toutes le même nombre de colonnes
adjusted_rows = []
for row in rows:
    if len(row) < max_cols:
        row.extend([None] * (max_cols - len(row)))
    adjusted_rows.append(row)

df_bilan_nation = pd.DataFrame(adjusted_rows, columns=headers)

# Afficher le DataFrame pour vérifier les colonnes disponibles
print(df_bilan_nation)

# Nettoyage de la colonne 'Nation' si elle existe
if 'Nation' in df_bilan_nation.columns:
    df_bilan_nation['Nation'] = df_bilan_nation['Nation'].str.replace(r"\[.*?\]", "", regex=True)  # Supprimer les annotations de référence
    df_bilan_nation['Nation'] = df_bilan_nation['Nation'].str.replace(r"/", "", regex=True)  # Supprimer les barres obliques
    df_bilan_nation['Nation'] = df_bilan_nation['Nation'].str.strip()  # Enlever les espaces blancs au début et à la fin

# Retrait des lignes totalement vides
df_bilan_nation = df_bilan_nation.dropna(how='all')

print(df_bilan_nation)

# Sauvegarder les données dans un fichier CSV avec encodage UTF-8
file_path_bilan_nations = "resultats_scraping/bilan_nations_AmSud_hommes.csv"
df_bilan_nation.to_csv(file_path_bilan_nations, index=False, encoding='utf-8-sig')
print(f"Les données ont été sauvegardées dans {file_path_bilan_nations}")