# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:07:03 2024

@author: Administrateur
"""

# DONNE CHAMPIONNAT DU MONDE MASCULIN - PALMARES 

import requests
from bs4 import BeautifulSoup 
import csv

url="https://fr.wikipedia.org/wiki/Championnat_du_monde_masculin_de_handball#Palmar%C3%A8s"

#♦ on fait une requete url 
reponse=requests.get(url)
html=reponse.content
soup = BeautifulSoup(html, 'html.parser')

print("titre du site", soup.title)

# Localiser la section "Palmarès"
palmares_section = soup.find(id="Palmarès").find_next('table')

# Initialiser une liste pour stocker les données
data = []

# Parcourir les lignes du tableau
for row in palmares_section.find_all('tr'):
    cols = row.find_all('td')

    if len(cols) > 1:  # S'assurer qu'il s'agit d'une ligne de données et non d'un en-tête
        year = cols[0].text.strip()
        country = cols[1].text.strip()
        winners = [winner.text.strip() for winner in cols[2:]]
        data.append([year] + [country]+ winners)
         
      


# Spécifier le nom du fichier CSV
filename = "handball_palmares_monde.csv"

with open(filename, mode='w', newline='',encoding='utf-8') as file:
    writer = csv.writer(file)    
    # Écrire l'en-tête
    header = ["edition", "Year", "Country hote", "Finale Gagnant 1", "Score", "Finaliste - 2e", "Winner3", "Score petite finale", "Winner 4"]
    data.insert(0, header)
   # Écrire les données
    writer.writerows(data)

print('les donnees ont ete sauvegardees')    


    