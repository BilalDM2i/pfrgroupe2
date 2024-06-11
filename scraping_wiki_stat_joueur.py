import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import numpy as np

# -------------------------------------------------
# Danemark
# -------------------------------------------------
url = "https://fr.wikipedia.org/wiki/%C3%89quipe_du_Danemark_masculine_de_handball#Effectif_actuel"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
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
df_effectif_danemark = pd.DataFrame(rows, columns=headers)
# faire une vérification sur le nb de colonnes

# Nettoyage colonne date de naissance
df_effectif_danemark['Date de naissance'] = df_effectif_danemark['Date de naissance'].str.split('(').str[0].str.strip()
print(df_effectif_danemark.to_string())

# Sauvegarder les données dans un fichier CSV
file_path_danemark = "resultats_scraping/effectif_danemark_hommes.csv"
df_effectif_danemark.to_csv(file_path_danemark, index=False)
print(f"Les données ont été sauvegardées dans {file_path_danemark}")


# -------------------------------------------------
# Suède
# -------------------------------------------------
url = "https://fr.wikipedia.org/wiki/%C3%89quipe_de_Su%C3%A8de_masculine_de_handball"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
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
df_effectif_suede = pd.DataFrame(rows, columns=headers)
# faire une vérification sur le nb de colonnes

# Nettoyage colonne date de naissance
df_effectif_suede['Date de naissance'] = df_effectif_suede['Date de naissance'].str.split('(').str[0].str.strip()
print(df_effectif_suede.to_string())

# Sauvegarder les données dans un fichier CSV
file_path_suede = "resultats_scraping/effectif_suede_hommes.csv"
df_effectif_suede.to_csv(file_path_suede, index=False)
print(f"Les données ont été sauvegardées dans {file_path_suede}")


# -------------------------------------------------
# France
# -------------------------------------------------
url = "https://fr.wikipedia.org/wiki/%C3%89quipe_de_France_masculine_de_handball"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

tables = soup.find_all('table', {'class': 'wikitable'})
table = tables[15]

# Noms des colonnes
headers = [th.text.strip() for th in table.find_all('th')]

# Extraire les lignes du tableau
rows = []
for row in table.find_all('tr')[1:]:
    cells = row.find_all('td')
    row_data = [cell.text.strip() for cell in cells]
    rows.append(row_data)

# Créer un dataframe avec les données extraites
df_effectif_france = pd.DataFrame(rows, columns=headers[0:10])
# faire une vérification sur le nb de colonnes

# Retrait lignes totalement vides
df_effectif_france = df_effectif_france.dropna(how='all')

# Nettoyage colonne date de naissance
df_effectif_france['Date de naissance'] = df_effectif_france['Date de naissance'].str.split('(').str[0].str.strip()
print(df_effectif_france.to_string())

# Sauvegarder les données dans un fichier CSV
file_path_france = "resultats_scraping/effectif_france_hommes.csv"
df_effectif_france.to_csv(file_path_france, index=False)
print(f"Les données ont été sauvegardées dans {file_path_france}")

