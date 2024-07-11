import pandas as pd
import matplotlib.pyplot as plt

# Charger les données du fichier CSV
predict_df = pd.read_csv('predicted_results_v3.csv')

# Définir les poules et les équipes qui y appartiennent
poule_A_teams = ['Espagne', 'Croatie', 'Allemagne', 'Slovénie', 'Suède', 'Japon']
poule_B_teams = ['Danemark', 'Norvège', 'Hongrie', 'France', 'Egypte', 'Argentine']

# Initialiser un dictionnaire pour stocker les points des équipes
points = {team: 0 for team in poule_A_teams + poule_B_teams}

# Parcourir les résultats des matchs et mettre à jour les points
for idx, row in predict_df.iterrows():
    Home_Team = row['Home_Team']
    Away_Team = row['Away_Team']
    home_result = row['home_result_pred']
    away_result = row['away_result_pred']

    if home_result == 1 and away_result == 0:
        # Home team wins
        if Home_Team in points:
            points[Home_Team] += 2
    elif home_result == 0 and away_result == 1:
        # Away team wins
        if Away_Team in points:
            points[Away_Team] += 2
    elif home_result == 1 and away_result == 1:
        # Draw
        if Home_Team in points:
            points[Home_Team] += 1
        if Away_Team in points:
            points[Away_Team] += 1

# Créer un DataFrame pour les points par équipe
points_df = pd.DataFrame(list(points.items()), columns=['team', 'points'])

# Séparer les points par poule
poule_A_points = points_df[points_df['team'].isin(poule_A_teams)].sort_values(by='points', ascending=False).reset_index(drop=True)
poule_B_points = points_df[points_df['team'].isin(poule_B_teams)].sort_values(by='points', ascending=False).reset_index(drop=True)

# Fonction pour déterminer les quarts de finale à partir des classements des poules A et B
def calculer_quarts_de_finale(poule_A_df, poule_B_df):
    quarts_de_finale = []
    
    # Qualification directe pour les quarts de finale
    quart_1 = (poule_A_df.iloc[0]['team'], poule_B_df.iloc[3]['team'])  # 1er de A vs 4ème de B
    quart_2 = (poule_B_df.iloc[0]['team'], poule_A_df.iloc[3]['team'])  # 1er de B vs 4ème de A
    quart_3 = (poule_A_df.iloc[1]['team'], poule_B_df.iloc[2]['team'])  # 2ème de A vs 3ème de B
    quart_4 = (poule_B_df.iloc[1]['team'], poule_A_df.iloc[2]['team'])  # 2ème de B vs 3ème de A
    
    quarts_de_finale.append(quart_1)
    quarts_de_finale.append(quart_2)
    quarts_de_finale.append(quart_3)
    quarts_de_finale.append(quart_4)
    
    return quarts_de_finale

# Appel de la fonction pour calculer les quarts de finale
quarts_de_finale = calculer_quarts_de_finale(poule_A_points, poule_B_points)

# Fonction pour créer un graphique de barres pour une poule
def plot_poule(poule_df, title):
    plt.figure(figsize=(10, 6))
    plt.bar(poule_df['team'], poule_df['points'], color='skyblue')
    plt.xlabel('Équipe')
    plt.ylabel('Points')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Afficher les graphiques pour chaque poule
plot_poule(poule_A_points, 'Classement Poule A')
plot_poule(poule_B_points, 'Classement Poule B')

# Affichage des quarts de finale
print("\nTableau des quarts de finale :")
for i, match in enumerate(quarts_de_finale, start=1):
    print(f"Quart de finale {i}: {match[0]} contre {match[1]}")

# Convertir les quarts de finale en DataFrame
quarts_de_finale_df = pd.DataFrame(quarts_de_finale, columns=['Home_Team', 'Away_Team'])

# Écrire les quarts de finale dans un fichier CSV
quarts_de_finale_df.to_csv('quarts_de_finale.csv', index=False)
print("\nLes quarts de finale ont été enregistrés dans le fichier 'quarts_de_finale.csv'.")
