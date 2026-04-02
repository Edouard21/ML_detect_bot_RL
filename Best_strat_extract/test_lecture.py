import pandas as pd

# Remplace par le nom exact d'un de tes fichiers générés dans ton dossier Database
CHEMIN_FICHIER = r"C:\Users\Edouard\Documents\Edouard\Projet RL\Fichier replay\Database\07f8655e-4036-4a9a-beba-78593b7dfe36.parquet"

# 1. Charger les données ultra-rapidement
df = pd.read_parquet(CHEMIN_FICHIER)

# 2. Filtrer uniquement les colonnes qu'on a créées pour les labels (is_bot)
colonnes_labels = [col for col in df.columns if "is_bot" in col]

print(f"Dimensions du match : {df.shape[0]} frames x {df.shape[1]} infos")
print("\nVoici les labels de tes joueurs pour l'IA :")
print(df[colonnes_labels].head(1)) # Affiche la première ligne pour voir qui est 0 ou 1