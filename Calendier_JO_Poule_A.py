# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:38:38 2024

@author: Administrateur
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup

def scrape_wikipedia_table(url, table_index=0):
    # Envoyer une requête GET à l'URL de la page Wikipedia
    response = requests.get(url)
    response.raise_for_status()  # Assurez-vous que la requête a réussi

    # Utiliser BeautifulSoup pour analyser le contenu HTML de la page
    soup = BeautifulSoup(response.text, 'html.parser')

    # Trouver tous les tableaux sur la page
    tables = soup.find_all('table', {'class': 'wikitable'})

    # Vérifier que l'index du tableau est valide
    if table_index >= len(tables):
        raise IndexError(f"Tableau indexé {table_index} non trouvé. Il y a seulement {len(tables)} tableaux sur cette page.")

    # Extraire le tableau spécifié
    table = tables[table_index]

    # Utiliser Pandas pour lire le HTML du tableau
    df = pd.read_html(str(table))[0]

    return df

def save_to_csv(df, file_name):
    # Sauvegarder le DataFrame en fichier CSV
    df.to_csv(file_name, index=False, encoding='utf-8-sig')




table_index = 5  # Modifier cet index pour choisir un autre tableau sur la page
url = 'https://fr.wikipedia.org/wiki/Handball_aux_Jeux_olympiques_d%27%C3%A9t%C3%A9_de_2024'
try:
    df = scrape_wikipedia_table(url, table_index)
    save_to_csv(df, 'calendrier_jo_poule_A.csv')
    print("Le tableau a été scrappé et sauvegardé en tant que calendrier_jo_poule_A.csv")
except Exception as e:
    print(f"Une erreur s'est produite : {e}")
