# Scraping des résultats des mondiaux (scores de chaque match)
# Emie
# Création : 28/06/2024
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import pandas as pd
from fonctions_utiles import *

# Configurer les options du navigateur
chrome_options = Options()
chrome_options.add_argument("--headless")  # Exécuter en mode headless (sans interface graphique)
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Initialiser le WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# URL de la page à scraper
url = 'https://www.flashscore.fr/handball/monde/championnat-du-monde-2023/resultats/'

# Ouvrir la page
driver.get(url)

# Attendre que le contenu soit chargé
time.sleep(5)

# Récupérer le contenu de la page
page_content = driver.page_source

# Fermer le navigateur
driver.quit()

# Analyser le contenu HTML avec BeautifulSoup
soup = BeautifulSoup(page_content, 'html.parser')

# Trouver tous les éléments qui contiennent les informations de match
matches = soup.find_all('div', class_='event__match')

# Parcourir chaque match et extraire les informations
match_data = []
for match in matches:
    # Date du match
    match_date = match.find("div", class_="event__time").text.strip()
    # Gestion du après prolongation : s'il y a une prolongation, on enlève la mention de la date et on ajoute "oui" dans une colonne après prolongration
    if "Après prol." in match_date:
        match_date = match_date.replace("Après prol.", "").strip()
    #     extra_time = "Oui"
    # else:
    #     match_date = match_date
    #     extra_time = "Non"

    # Equipes
    home_team = match.find("div", class_="event__participant--home").text.strip()
    away_team = match.find("div", class_="event__participant--away").text.strip()

    # Scores
    score_home_team = match.find("div", class_="event__score--home").text.strip()
    score_away_team = match.find("div", class_="event__score--away").text.strip()

    # Alimentation d'un DF
    match_data.append({
        'Date': match_date,
        'Home Team': home_team,
        'Away Team': away_team,
        'Score Home Team': score_home_team,
        'Score Away Team': score_away_team,
        # 'Extra Time': extra_time
    })

# DF
df = pd.DataFrame(match_data)

# Afficher le DF
print(df.to_string())

# Sauvegarder les données dans un fichier CSV
sauvegarde_df_csv(df, "resultats_scraping/mondiaux_2023.csv")
