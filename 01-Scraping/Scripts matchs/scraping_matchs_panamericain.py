from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import pandas as pd
#from fonctions_utiles import *
import re


# Configurer les options du navigateur
chrome_options = Options()
chrome_options.add_argument("--headless")  # Exécuter en mode headless (sans interface graphique)
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Initialiser le WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Liste pour stocker les données de tous les matchs
all_match_data = []

# Année de début du scraping
year = 2023

# Boucle pour scraper les données jusqu'à l'année 2000
while year > 1999:
    # URL
    url = f'https://www.flashscore.fr/handball/monde/jeux-panamericain-{year}/resultats/'

    # Ouvrir la page
    driver.get(url)

    # Attendre que le contenu soit chargé
    time.sleep(5)

    # Récupérer le contenu de la page
    page_content = driver.page_source

    # Analyser le contenu HTML avec BeautifulSoup
    soup = BeautifulSoup(page_content, 'html.parser')

    # Trouver tous les matchs
    matches = soup.find_all('div', class_='event__match')

    # Pour chaque match, extraire date équipes et scores
    for match in matches:
        # Date du match
        match_date = match.find("div", class_="event__time").text.strip()
        # Enlever ce qui est écrit après date et heure (tir au buts, forfait, etc)
        match_date = re.match(r"\d{2}\.\d{2}\.\s\d{2}:\d{2}", match_date).group()

        # Equipes
        home_team = match.find("div", class_="event__participant--home").text.strip()
        away_team = match.find("div", class_="event__participant--away").text.strip()

        # Scores
        score_home_team = match.find("div", class_="event__score--home").text.strip()
        score_away_team = match.find("div", class_="event__score--away").text.strip()

        # Remplir la liste de tous les matchs
        all_match_data.append({
            'Date': match_date,
            'Home_Team': home_team,
            'Away_Team': away_team,
            'Score_Home_Team': score_home_team,
            'Score_Away_Team': score_away_team,
            'Year': year
        })

    # Edition précédente pour la suite de la boucle
    year -= 2

# Fermer le navigateur
driver.quit()

# DF de tous les matchs
df = pd.DataFrame(all_match_data)

# Ajouter l'année aux dates
df['Date'] = df.apply(lambda row: f"{row['Date'][:-7]}.{row['Year']} {row['Date'][-5:]}", axis=1)

# Afficher le DF
print(df.to_string())

# Sauvegarder les données dans un fichier CSV
file_path_panam = "panamericain.csv"
df.to_csv(file_path_asie, index=False)
print(f"Les donn�es ont �t� sauvegard�es dans {file_path_panam}")

