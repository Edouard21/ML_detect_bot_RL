import subtr_actor
import pandas as pd
import glob
import os

# --- PARAMETRES ---
DOSSIER_REPLAYS = r"C:\Users\Edouard\Documents\Edouard\Projet RL\Fichier replay\Bot" # Adapte le chemin
DOSSIER_SORTIE = r"C:\Users\Edouard\Documents\Edouard\Projet RL\Fichier replay\Database"

GLOBAL_FEATURES = ["BallRigidBody"] 
PLAYER_FEATURES = ["PlayerRigidBody", "PlayerBoost", "PlayerAnyJump"]

# Mets ici les pseudos exacts des bots présents dans tes replays
LISTE_BOTS = ["Nexto", "Necto", "Element", "Seer", "Kamael", "NextoBot"] 

def preparer_dataset_ml():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    fichiers_replays = glob.glob(os.path.join(DOSSIER_REPLAYS, "*.replay"))
    print(f"{len(fichiers_replays)} replays trouvés. Début de l'extraction...")
    
    # Récupération du dictionnaire des noms (une seule fois pour gagner du temps)
    headers_info = subtr_actor.get_column_headers(
        global_feature_adders=GLOBAL_FEATURES, 
        player_feature_adders=PLAYER_FEATURES
    )
    
    for fichier in fichiers_replays:
        nom_fichier = os.path.basename(fichier)
        
        try:
            # 1. Extraction brute
            meta, matrice = subtr_actor.get_ndarray_with_info_from_replay_filepath(
                fichier, 
                global_feature_adders=GLOBAL_FEATURES, 
                player_feature_adders=PLAYER_FEATURES, 
                fps=15.0 
            )
            df = pd.DataFrame(matrice)
            
            # 2. Nommage dynamique des colonnes
            noms_colonnes = headers_info['global_headers'].copy()
            nb_colonnes_joueurs = df.shape[1] - len(noms_colonnes)
            nb_joueurs = nb_colonnes_joueurs // len(headers_info['player_headers'])
            
            # On génère les noms pour les joueurs (P0_, P1_, etc.)
            for i in range(nb_joueurs):
                for header in headers_info['player_headers']:
                    noms_colonnes.append(f"P{i}_{header}")
            
            df.columns = noms_colonnes
            
            # 3. Ajout du contexte et des LABELS (Humain ou Bot)
            df['game_id'] = nom_fichier
            
            # Extraction de la liste des pseudos depuis la meta
            noms_joueurs = []
            replay_meta = meta.get('replay_meta', {})
            
            for item in replay_meta.get('all_headers', []):
                if item[0] == 'PlayerStats': # C'est ici que se cachent les pseudos !
                    for stats in item[1]:
                        noms_joueurs.append(stats.get('Name', 'Unknown'))
            
            # On assigne les pseudos et le label Bot aux colonnes
            for i in range(nb_joueurs):
                pseudo = "Unknown"
                if i < len(noms_joueurs):
                    pseudo = noms_joueurs[i]
                
                df[f"P{i}_name"] = pseudo
                df[f"P{i}_is_bot"] = 1 if pseudo in LISTE_BOTS else 0
                
            # 4. Sauvegarde
            chemin_sortie = os.path.join(DOSSIER_SORTIE, f"{nom_fichier.replace('.replay', '')}.parquet")
            df.to_parquet(chemin_sortie, index=False)
            print(f"[SUCCES] -> {nom_fichier} (Joueurs: {nb_joueurs})")
            
        except Exception as e:
            print(f"[ERREUR] sur {nom_fichier} : {e}")

if __name__ == "__main__":
    preparer_dataset_ml()