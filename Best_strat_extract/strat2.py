import carball
import pandas as pd
import glob
import os

DOSSIER_REPLAYS = r"C:\Users\Edouard\Documents\Edouard\Projet RL\Fichier replay\Bot"
DOSSIER_SORTIE = r"C:\Users\Edouard\Documents\Edouard\Projet RL\Fichier replay\Database"
LISTE_BOTS = ["Nexto", "Necto", "Element", "Seer"]

def preparer_dataset_complet():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    fichiers_replays = glob.glob(os.path.join(DOSSIER_REPLAYS, "*.replay"))
    print(f"{len(fichiers_replays)} replays trouvés. Début de l'extraction complète...")
    
    for fichier in fichiers_replays:
        nom_fichier = os.path.basename(fichier)
        try:
            # 1. Analyse complète
            manager = carball.analyze_replay_file(fichier)
            df = manager.get_data_frame()
            
            # 2. Aplatir le format MultiIndex de Carball
            df.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else str(col) for col in df.columns.values]
            
            # --- NOUVEAU : NETTOYAGE DES TYPES ---
            # On force toutes les colonnes de Carball en float32 pour corriger le mélange booléen/entier
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
            
            # 3. Ajouter les labels pour l'IA
            df['game_id'] = str(nom_fichier)
            
            for joueur in manager.game.players:
                pseudo = joueur.name
                
                # On étiquette le joueur si c'est un bot connu ou un invité split-screen
                est_un_bot = 1 if (pseudo in LISTE_BOTS) or ("(1)" in pseudo) else 0
                df[f"{pseudo}_is_bot"] = est_un_bot
                
            # 4. Sauvegarde au format Parquet
            chemin_sortie = os.path.join(DOSSIER_SORTIE, f"{nom_fichier.replace('.replay', '')}.parquet")
            df.to_parquet(chemin_sortie, index=False)
            print(f"[SUCCES] -> {nom_fichier} (Joueurs gardés: {len(manager.game.players)})")
            
        except Exception as e:
            print(f"[ERREUR] sur {nom_fichier} : {e}")

if __name__ == "__main__":
    preparer_dataset_complet()