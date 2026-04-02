import pandas as pd

CHEMIN_FICHIER = r"C:\Users\Edouard\Documents\Edouard\Projet RL\Fichier replay\Database\07f8655e-4036-4a9a-beba-78593b7dfe36.parquet"

df = pd.read_parquet(CHEMIN_FICHIER)

# 1. Forcer l'affichage de TOUTES les colonnes sans les cacher (...)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000) # Élargir l'affichage dans le terminal

# 2. Afficher tous les noms de colonnes disponibles
print("=== LISTE DES 115 COLONNES DISPONIBLES ===")
print(df.columns.tolist())

# 3. Afficher les données du ballon (On prend les frames 100 à 105, quand ça bouge)
print("\n" + "="*50)
print("=== DONNÉES DE LA BALLE (Frames 100 à 105) ===")
colonnes_balle = [col for col in df.columns if 'ball' in col.lower() and 'cam' not in col.lower()]
print(df[colonnes_balle].iloc[100:105])

# 4. Afficher les données physiques et les actions du BOT "ΣΩΖΔ(1)"
print("\n" + "="*50)
print("=== DONNÉES DU BOT ΣΩΖΔ(1) (Frames 100 à 105) ===")
# On cherche toutes les colonnes qui contiennent son nom, sans prendre le label
colonnes_bot = [col for col in df.columns if 'ΣΩΖΔ(1)' in col and 'is_bot' not in col]
print(df[colonnes_bot].iloc[100:105])