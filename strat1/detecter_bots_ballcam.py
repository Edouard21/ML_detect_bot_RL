import os
import glob
import json
import pandas as pd
import subtr_actor
import view_all_framesV2 as v2

# Dossier cible spécifié par l'utilisateur
DOSSIER_BOTS = r"C:\Users\Edouard\Documents\Edouard\Projet RL\Script Extraction info game\replay\bot"

def detecter_toujours_ballcam():
    print(f"🔍 Analyse des replays dans : {DOSSIER_BOTS}")
    
    fichiers = glob.glob(os.path.join(DOSSIER_BOTS, "*.replay"))
    if not fichiers:
        print("❌ Aucun replay trouvé dans ce dossier.")
        return
    
    suspects = set()
    
    for idx, replay_path in enumerate(fichiers):
        nom_base = os.path.basename(replay_path)
        print(f"[{idx+1}/{len(fichiers)}] Analyse de {nom_base}...")
        
        try:
            # 1. Extraction Metadata pour les noms
            meta = v2.subtr_actor.get_replay_meta(replay_path)
            noms_joueurs = v2.extraire_noms_joueurs(meta)
            
            # Gestion split-screen
            split_info = v2.detecter_split_screen(noms_joueurs)
            indices_valides = [i for i in range(len(noms_joueurs)) if i not in split_info]
            mapping_noms_vers_slots = {noms_joueurs[idx]: idx for idx in indices_valides}
            noms_joueurs_valides = list(mapping_noms_vers_slots.keys())
            
            # 2. Nombre de frames
            with open(replay_path, "rb") as f:
                full_replay = v2.subtr_actor.parse_replay(f.read())
            nb_frames = len(full_replay["network_frames"]["frames"])
            
            # 3. Extraction des inputs (Ballcam)
            # On utilise la fonction de v2 qui est déjà robuste
            df_net = v2.extraire_inputs_du_reseau(replay_path, noms_joueurs_valides, mapping_noms_vers_slots, nb_frames)
            
            if df_net.empty:
                continue
                
            # 4. Vérification pour chaque joueur
            for nom, slot in mapping_noms_vers_slots.items():
                col_ballcam = f"P{slot}_ballcam"
                if col_ballcam in df_net.columns:
                    series = df_net[col_ballcam]
                    
                    # Heuristique :
                    # 1. Un humain change de vue régulièrement.
                    # 2. Un bot a souvent 0 changements (reste sur Ballcam ou Carcam tout le long).
                    
                    nb_changements = (series.shift() != series).sum() - 1 # -1 car la 1ère frame compte
                    moyenne_active = series.mean()
                    
                    # On considère suspect si 0 changements et moyenne > 90% (toujours active)
                    if nb_changements <= 1 and moyenne_active > 0.90:
                        print(f"   ⚠️  SUSPECT : {nom} | Changements: {nb_changements} | Active: {moyenne_active*100:.1f}%")
                        suspects.add(nom)
                    elif nb_changements <= 1 and moyenne_active < 0.10:
                        # Toujours désactivée (carcam) - peut aussi être un bot, mais l'utilisateur a demandé "toujours active"
                        pass
                else:
                    # Si la colonne n'existe pas, peut-être que personne n'a touché au bouton de tout le match ?
                    # ou que l'objet network n'a pas été trouvé.
                    pass
                        
        except Exception as e:
            print(f"   ❌ Erreur sur {nom_base} : {e}")

    # 5. Export JSON
    resultat = sorted(list(suspects))
    output_path = os.path.join(os.path.dirname(DOSSIER_BOTS), "suspects_ballcam.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(resultat, f, ensure_ascii=False, indent=4)
        
    print("\n" + "="*40)
    print(f"✅ Analyse terminée.")
    print(f"👥 Nombre de suspects trouvés : {len(resultat)}")
    print(f"📄 Liste sauvegardée dans : {output_path}")
    print("="*40)
    print(json.dumps(resultat, indent=4))

if __name__ == "__main__":
    detecter_toujours_ballcam()
