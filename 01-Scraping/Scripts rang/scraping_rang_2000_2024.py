# Scraping des rangs de 2000 à 2024 au 1er janvier de chaque année
# Emie
# Création : 02/07/2024
# Dernière modif : 08/07/2024

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
from unidecode import unidecode

# Configurer les options du navigateur
chrome_options = Options()
chrome_options.add_argument("--headless")  # Exécuter en mode headless (sans interface graphique)
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Initialiser le WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Liste pour stocker les données de classement
all_rankings = []

# URL
url = 'https://handballranking.com/'

# Ouvrir la page
driver.get(url)
time.sleep(5)

# Pour chaque 1er janvier entre 2000 et 2024
current_year = 2000
current_date = datetime(current_year, 1, 1)

while current_year < 2025:

    print(current_year)

    # Trouver et remplir le champ de date
    date_input = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable(
            (By.ID, 'txtDate')))  # récupérer l'élément txt Date, attendre un peu pour voir s'il est bien cliquable
    date_input.clear()  # vider le champ date avant de le remplir pour être sûr qu'il n'y ait pas de mauvaise date
    date_input.send_keys(current_date.strftime('%d/%m/%Y'))  # remplir le champ date

    # Trouver et cliquer sur le bouton "GetList"
    get_list_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, 'input.btn.btn-primary')))  # idem attente que le bouton get list soit bien cliquable
    get_list_button.click()  # on clique

    # Attendre le chargement de la page
    time.sleep(5)

    # Récupérer le contenu de la page
    page_content = driver.page_source

    # Analyser le contenu HTML avec BeautifulSoup
    soup = BeautifulSoup(page_content, 'html.parser')

    # Récupérer le tableau
    rankings_table = soup.find('table')

    rows = rankings_table.find_all('tr')[1:]  # Ignorer l'en-tête de la table

    # Pour chaque ligne on récupère tout sauf drapeau et stats
    for row in rows:
        cols = row.find_all('td')
        rank = cols[0].text.strip()
        nation = cols[2].text.strip()
        points = cols[3].text.strip()
        match_count = cols[4].text.strip()

        # On alimente la liste
        all_rankings.append({
            'Date': current_date.strftime('%d/%m/%Y'),
            'Rank': rank,
            'Nation': nation,
            'Points': points,
            'Match_count': match_count
        })

    current_year += 1
    current_date = datetime(current_year, 1, 1)

# Fermer le navigateur
driver.quit()

# DF
df = pd.DataFrame(all_rankings)
df['Points'] = df['Points'].str.extract(r'([0-9.]+)', expand=False)
df['Rank'] = df['Rank'].str.extract(r'([0-9.]+)', expand=False)

print(df.to_string())

# Changement des noms
translations = {
    'Sweden': 'Suède',
    'Russia': 'Russie',
    'Germany': 'Allemagne',
    'Spain': 'Espagne',
    'Serbia': 'Serbie',
    'France': 'France',
    'Cuba': 'Cuba',
    'Egypt': 'Égypte',
    'Iceland': 'Islande',
    'Croatia': 'Croatie',
    'Denmark': 'Danemark',
    'Hungary': 'Hongrie',
    'Republic of Korea': 'Corée du Sud',
    'Slovenia': 'Slovénie',
    'Tunisia': 'Tunisie',
    'Belarus': 'Biélorussie',
    'Norway': 'Norvège',
    'Switzerland': 'Suisse',
    'Argentina': 'Argentine',
    'Romania': 'Roumanie',
    'Czechia': 'Republique Tcheque',
    'Lithuania': 'Lituanie',
    'Portugal': 'Portugal',
    'Turkey': 'Turquie',
    'Algeria': 'Algérie',
    'Poland': 'Pologne',
    'Ukraine': 'Ukraine',
    'Brazil': 'Brésil',
    'Austria': 'Autriche',
    'Moldova': 'Moldavie',
    'North Macedonia': 'Macédoine du Nord',
    'Bosnia and Herzegovina': 'Bosnie-Herzégovine',
    'Canada': 'Canada',
    'Italy': 'Italie',
    'Ivory Coast': 'Côte d\'Ivoire',
    'United States of America': 'USA',
    'Congo': 'Congo',
    'China': 'Chine',
    'Slovakia': 'Slovaquie',
    'Nigeria': 'Nigéria',
    'Israel': 'Israël',
    'Cyprus': 'Chypre',
    'Finland': 'Finlande',
    'Latvia': 'Lettonie',
    'Australia': 'Australie',
    'Morocco': 'Maroc',
    'Georgia': 'Géorgie',
    'Mexico': 'Mexique',
    'Japan': 'Japon',
    'Greece': 'Grèce',
    'Saudi Arabia': 'Arabie Saoudite',
    'Kuwait': 'Koweït',
    'Estonia': 'Estonie',
    'Paraguay': 'Paraguay',
    'Belgium': 'Belgique',
    'Bulgaria': 'Bulgarie',
    'Netherlands': 'Pays-Bas',
    'Luxembourg': 'Luxembourg',
    'Uruguay': 'Uruguay',
    'Puerto Rico': 'Porto Rico',
    'Greenland': 'Groenland',
    'Dominican Republic': 'République Dominicaine',
    'Colombia': 'Colombie',
    'Cameroon': 'Cameroun',
    'Costa Rica': 'Costa Rica',
    'Armenia': 'Arménie',
    'Malta': 'Malte',
    'Ireland': 'Irlande',
    'Angola': 'Angola',
    'Chile': 'Chili',
    'Iran': 'Iran',
    'Guatemala': 'Guatemala',
    'Azerbaijan': 'Azerbaïdjan',
    'Great Britain': 'Grande Bretagne',
    'Qatar': 'Qatar',
    'Faroe Islands': 'Îles Féroé',
    'Senegal': 'Sénégal',
    'Bahrain': 'Bahreïn',
    'DR Congo': 'RD Congo',
    'Gabon': 'Gabon',
    'England': 'Angleterre',
    'United Arab Emirates': 'Émirats Arabes Unis',
    'Scotland': 'Écosse',
    'Montenegro': 'Monténégro',
    'New Zealand': 'Nouvelle-Zélande',
    'Venezuela': 'Venezuela',
    'Jordan': 'Jordanie',
    'Lebanon': 'Liban',
    'Cook Islands': 'Îles Cook',
    'Uzbekistan': 'Ouzbékistan',
    'Oman': 'Oman',
    'Iraq': 'Iraq',
    'Nicaragua': 'Nicaragua',
    'Libya': 'Libye',
    'Kosovo': 'Kosovo',
    'Honduras': 'Honduras',
    'El Salvador': 'Salvador',
    'Syria': 'Syrie',
    'Kenya': 'Kenya',
    'Andorra': 'Andorre',
    'Albania': 'Albanie',
    'Peru': 'Pérou',
    'Panama': 'Panama',
    'Dominica': 'Dominique',
    'India': 'Inde',
    'Hong Kong': 'Hong Kong',
    'Bolivia': 'Bolivie',
    'Cape Verde': 'Cap-Vert',
    'Guinea': 'Guinée',
    'Kazakhstan': 'Kazakhstan',
    'Zambia': "Zambie",
    "Chinese Taipei": "Taipei Chinois"
}

df['Nation'] = df['Nation'].map(lambda x: translations.get(x, x))
df['Nation'] = df['Nation'].str.strip()
df['Nation'] = df['Nation'].apply(lambda x: unidecode(x))

print(df.to_string())

# Sauvegarder les données dans un fichier CSV
df.to_csv("classements_handball_2000_2024.csv", index=False)
