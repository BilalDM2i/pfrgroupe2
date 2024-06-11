# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:23:28 2024

@author: Administrateur
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import sys

# URL de la page Wikipédia
url = "https://fr.wikipedia.org/wiki/%C3%89quipe_d%27Espagne_masculine_de_handball"

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
table_index = 10

# Utiliser pandas pour lire le tableau HTML
try:
    df = pd.read_html(str(tables[table_index]))[0]
except ValueError as e:
    print(f"Erreur lors de la lecture du tableau : {e}")
    sys.exit()

# Afficher les premières lignes et les colonnes du DataFrame pour vérifier la structure
print(df)
print(df.columns)

# Modifier cette ligne en fonction des colonnes réelles trouvées dans le tableau
# Voici un exemple possible
columns_of_interest = ["No", "P.", "Nom", "Date de naissance", "Taille", "Sél.", "Buts", "Club"]

# Vérifier si les colonnes existent dans le DataFrame
if all(col in df.columns for col in columns_of_interest):
    df = df[columns_of_interest]
else:
    print(f"Erreur : Les colonnes attendues ne sont pas présentes dans le tableau.")
    print(f"Colonnes trouvées : {df.columns}")
    sys.exit()

# Sauvegarder les données dans un fichier CSV avec encodage UTF-8
os.makedirs("resultats_scraping", exist_ok=True)
file_path = "resultats_scraping/joueurs_equipe_espagne_handball.csv"
df.to_csv(file_path, index=False, encoding='utf-8-sig')  # UTF-8 avec signature BOM pour une meilleure compatibilité
print(f"Les données ont été sauvegardées dans {file_path}")



