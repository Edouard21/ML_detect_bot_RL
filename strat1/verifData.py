import pandas as pd

# 1. Charger ton fichier
df = pd.read_parquet("6a008b64-deba-411e-bad1-c9a40ff9ebd8.parquet")

# 2. Afficher la taille (Lignes = frames du match, Colonnes = informations)
print(f"Dimensions : {df.shape[0]} frames x {df.shape[1]} colonnes")

# 3. Afficher les noms des colonnes
print("\nListe des colonnes :", df.columns.tolist())

# 4. Afficher un aperçu des 5 premières frames
print("\nAperçu des données :")
print(df.head())