# Fonctions utiles pour plusieurs scripts
# Emie
# Création : 19/06/2024
# Dernière modif :

import pandas as pd
import datetime


def sauvegarde_df_csv(df, file_with_path):
    df.to_csv(file_with_path, index=False)
    print(f"Les données ont été sauvegardées dans {file_with_path}")


def nettoyage_colonne_nation(df):
    # si temps revoir le if et le gérer comme exception
    if "Nation" not in df:
        print("Il n'y a pas de colonnes Nation dans ce dataframe")
    else:
        df['Nation'] = df['Nation'].str.replace(r"\[.*?\]", "", regex=True)  # Supprimer les annotations de référence
        df['Nation'] = df['Nation'].str.replace(r"/", "", regex=True)  # Supprimer les slash
        df['Nation'] = df['Nation'].str.strip()  # Enlever les espaces blancs au début et à la fin
    return df


def retrait_annees_futures(df):
    # si temps revoir le if et le gérer comme exception
    date = datetime.date.today()
    if "Année" not in df:
        print("Il n'y a pas de colonnes Année dans ce dataframe")
    else:
        df['Année'] = pd.to_numeric(df['Année'])
        df = df[df['Année'] < date.year]
    return df
