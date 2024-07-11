# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:07:21 2024

@author: Administrateur
"""

import pandas as pd

# Lecture du fichier CSV
df = pd.read_csv('Matchs_toutes_competitions.csv')

# Convertir la colonne 'Date' au format datetime si ce n'est pas déjà fait
df['Date'] = pd.to_datetime(df['Date'])

# Formater la colonne des dates selon le format requis
df['Date'] = df['Date'].dt.strftime('%d.%m.%Y %H:%M')

# Afficher le DataFrame avec la colonne des dates reformatée
print(df)

# Exporter le DataFrame dans un fichier CSV nommé "Matchs_toutes_competitions_avec2024_cleaned.csv"
df.to_csv('Matchs_toutes_competitions_cleaned.csv', index=False)