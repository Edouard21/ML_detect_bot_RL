import os
import glob
import matplotlib.pyplot as plt
import view_all_framesV2 as v2

def visualiser_ballcam(replay_path):
    print(f"Analyse réseau de : {os.path.basename(replay_path)}...")
    
    # 1. Extraction rapide des métadonnées
    meta = v2.subtr_actor.get_replay_meta(replay_path)
    noms_joueurs = v2.extraire_noms_joueurs(meta)
    
    # Filtrage du split-screen comme d'habitude
    split_info = v2.detecter_split_screen(noms_joueurs)
    indices_valides = [i for i in range(len(noms_joueurs)) if i not in split_info]
    mapping_noms_vers_slots = {noms_joueurs[idx]: idx for idx in indices_valides}
    noms_joueurs_valides = list(mapping_noms_vers_slots.keys())
    
    # 2. Obtenir le nombre de frames réseau exact (pour eviter de traiter la physique lourde)
    with open(replay_path, "rb") as f:
        full_replay = v2.subtr_actor.parse_replay(f.read())
    nb_frames = len(full_replay["network_frames"]["frames"])
    
    # 3. Utilisation de la nouvelle fonction V2 pour récupérer juste les inputs !
    df_net = v2.extraire_inputs_du_reseau(replay_path, noms_joueurs_valides, mapping_noms_vers_slots, nb_frames)
    
    if df_net.empty:
        print("Aucune donnée réseau extraite.")
        return
        
    # 4. Tracé du graphe
    plt.figure(figsize=(14, 7))
    
    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx_couleur, (nom, slot) in enumerate(mapping_noms_vers_slots.items()):
        col_ballcam = f"P{slot}_ballcam"
        if col_ballcam in df_net.columns:
            # On décale chaque joueur sur l'axe Y pour ne pas que les courbes se superposent (0, puis 2, puis 4...)
            offset_y = idx_couleur * 1.5 
            
            # y_data contiendra 0 ou 1, + le décalage
            y_data = df_net[col_ballcam] + offset_y
            
            # Utilisation de step() pour faire des "crénaux" bien carrés (allumé/éteint)
            plt.step(df_net.index, y_data, where='post', label=nom, color=couleurs[idx_couleur % len(couleurs)], linewidth=2)
            
            # Ajout d'un petit texte au dessus de la courbe pour identifier la ligne
            plt.text(0, offset_y + 1.05, f"{nom}", fontsize=10, fontweight='bold', color=couleurs[idx_couleur % len(couleurs)])

    plt.title(f"Analyse des Toggle Ballcam\nFichier: {os.path.basename(replay_path)}", fontsize=14)
    plt.xlabel("Temps (Frames Réseau ~30fps)", fontsize=12)
    plt.ylabel("Statut de la Ballcam (Activé en haut / Désactivé en bas)", fontsize=12)
    
    # Suppression des graduations Y car l'axe est qualitatif
    plt.yticks([]) 
    
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    # Affichage de la fenêtre interactive (Remplace Jupyter)
    plt.show()

if __name__ == "__main__":
    # Trouver le premier replay du dossier
    fichiers = glob.glob(os.path.join(v2.DOSSIER_REPLAYS, "**", "*.replay"), recursive=True)
    if fichiers:
        print("Lancement du visualiseur sur le replay n°0...")
        visualiser_ballcam(fichiers[0])
    else:
        print("Aucun fichier replay trouvé dans", v2.DOSSIER_REPLAYS)
