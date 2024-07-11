import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import sys

# URL de la page Wikipédia
url = "https://fr.wikipedia.org/wiki/%C3%89quipe_du_Japon_masculine_de_handball"

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

# Supposons que le tableau désiré est le 11ème tableau (index 10)
table_index = 7

# Utiliser pandas pour lire le tableau HTML
try:
    df = pd.read_html(str(tables[table_index]))[0]
except ValueError as e:
    print(f"Erreur lors de la lecture du tableau : {e}")
    sys.exit()

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

print(df)
# Supprimer les lignes en utilisant les index spécifiques
indices_to_remove = [0,3,7,14,17]  # Par exemple, supprimer les lignes aux indices 3 et 7
df = df.drop(indices_to_remove)
print(df)
# Sauvegarder les données dans un fichier CSV avec encodage UTF-8
os.makedirs("resultats_scraping", exist_ok=True)
file_path = "resultats_scraping/joueurs_equipe_japon_handball.csv"
df.to_csv(file_path, index=False, encoding='utf-8-sig')  # UTF-8 avec signature BOM pour une meilleure compatibilité
print(f"Les données ont été sauvegardées dans {file_path}")
