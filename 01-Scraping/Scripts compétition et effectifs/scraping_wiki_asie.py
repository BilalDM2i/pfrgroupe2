import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import numpy as np
import os


# URL de la page Wikipédia
url = "https://fr.wikipedia.org/wiki/Championnat_d%27Asie_masculin_de_handball"

# Envoyer une requête HTTP pour obtenir le contenu de la page
try:
    response = requests.get(url)
    response.raise_for_status()
    response.encoding = 'utf-8'  # Assurez-vous que l'encodage est correct
except requests.exceptions.RequestException as e:
    print(f"Erreur lors de la requête HTTP : {e}")
    sys.exit()

soup = BeautifulSoup(response.text, 'html.parser')



#%%

# -------------------------------------------------
# Palmarès JO
# -------------------------------------------------

# Trouver le tableau du palmarès des hommes
table = soup.find('table', {'class': 'wikitable alternance'})
if table is None:
    raise ValueError("Le tableau 'wikitable alternance' n'a pas été trouvé sur la page.")

# Extraire les lignes du tableau
rows = []
for row in table.find_all('tr')[1:]:
    cells = row.find_all('td')
    if len(cells) == 5:  # S'assurer que la ligne a bien 5 colonnes
        row_data = [cell.get_text(strip=True) for cell in cells]
        rows.append(row_data)
    else:
        print(f"Erreur : la ligne {row} n'a pas le bon nombre de colonnes ({len(cells)}) et sera ignorée.")

# Créer un dataframe avec les données extraites
df_palmares = pd.DataFrame(rows, columns=["Année", "Ville", "Or", "Argent", "Bronze"])

# Nettoyer la colonne 'Année' en retirant les caractères non numériques
df_palmares['Année'] = df_palmares['Année'].str.extract('(\d{4})')[0]

# Convertir en valeurs numériques
df_palmares['Année'] = pd.to_numeric(df_palmares['Année'])
date = datetime.date.today()
df_palmares = df_palmares[df_palmares['Année'] < date.year]
print(df_palmares.to_string())

# Sauvegarder les données dans un fichier CSV avec encodage UTF-8
os.makedirs("resultats_scraping", exist_ok=True)
file_path_palmares = "resultats_scraping/palmares_handball_asie_hommes.csv"
df_palmares.to_csv(file_path_palmares, index=False, encoding='utf-8-sig')
print(f"Les données ont été sauvegardées dans {file_path_palmares}")


#%%
# -------------------------------------------------
#  Tableau des médailles
# -------------------------------------------------
table = soup.find('table', {'class': 'wikitable'})
if table is None:
    raise ValueError("Le tableau 'wikitable' n'a pas été trouvé sur la page.")

# Extraire les lignes du tableau
rows = []
previous_row = None
for row in table.find_all('tr')[1:]:
    cells = row.find_all(['th', 'td'])
    row_data = [cell.get_text(strip=True) for cell in cells]
    rows.append(row_data)
    print(row_data)

print(rows)   


#%%

# Vérifier si la liste 'rows' n'est pas vide avant d'appeler max()
if rows:
    max_cols = max(len(row) for row in rows)
    print(f"Nombre maximum de colonnes: {max_cols}")

    # Ajuster les colonnes en fonction du nombre maximum de colonnes
    columns = ["Rang", "Nation", "Or", "Argent", "Bronze", "Total", "Der"]
    columns = columns[:max_cols]

    # Ajuster les lignes pour qu'elles aient toutes le même nombre de colonnes
    adjusted_rows_fixed = []
    for row in rows:
        if len(row) < max_cols:
            row.extend([None] * (max_cols - len(row)))
        adjusted_rows_fixed.append(row)

    # Créer le DataFrame avec les colonnes ajustées
    df_tableau_medailles = pd.DataFrame(adjusted_rows_fixed, columns=columns)

    # Afficher les longueurs de ligne
    for i, row in enumerate(rows):
        print(f"Ligne {i + 1} a {len(row)} colonnes: {row}")

else:
    print("Aucune donnée extraite du tableau, impossible de déterminer le nombre maximum de colonnes.")


#%%

# Vérifier et ajuster les colonnes avant de créer le DataFrame
max_cols = max(len(row) for row in rows)
print(f"Nombre maximum de colonnes: {max_cols}")
columns = ["Rang", "Nation", "Or", "Argent", "Bronze", "Total", "Der"]
columns = columns[:max_cols]

# Ajuster les lignes pour qu'elles aient toutes le même nombre de colonnes
adjusted_rows = []
for row in rows:
    if len(row) < max_cols:
        row.extend([None] * (max_cols - len(row)))
    adjusted_rows.append(row)

df_tableau_medailles = pd.DataFrame(adjusted_rows, columns=columns)
df_tableau_medailles['Nation'] = df_tableau_medailles['Nation'].str.replace(r"\(.*?\)", "", regex=True).str.strip()

# Afficher le DataFrame
print(df_tableau_medailles)

#%%

# Sauvegarder les données dans un fichier CSV avec encodage UTF-8
file_path_tab_medailles = "resultats_scraping/tableau_medailles_asie_hommes.csv"
df_tableau_medailles.to_csv(file_path_tab_medailles, index=False, encoding='utf-8-sig')
print(f"Les données ont été sauvegardées dans {file_path_tab_medailles}")
#%%

# -------------------------------------------------
# Bilan par nations
# -------------------------------------------------
bilan_table = soup.find(id="Bilan_par_nation")
if bilan_table:
    table = bilan_table.find_next('table')
else:
    raise ValueError("Le tableau 'Bilan par nation' n'a pas été trouvé sur la page.")

# Noms des colonnes
headers = [th.get_text(strip=True) for th in table.find_all('th')]

# Afficher les en-têtes pour diagnostic
print(f"En-têtes extraits : {headers}")

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
file_path_bilan_nations = "resultats_scraping/bilan_nations_asie_hommes.csv"
df_bilan_nation.to_csv(file_path_bilan_nations, index=False, encoding='utf-8-sig')
print(f"Les données ont été sauvegardées dans {file_path_bilan_nations}")
