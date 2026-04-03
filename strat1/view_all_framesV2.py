"""
Script d'extraction des données joueurs frame par frame des replays Rocket League.
Utilise subtr_actor pour extraire les données joueurs et exporte en JSON.
Gère les cas d'écran scindé (split-screen) où un joueur peut apparaître en double.
Supporte les matchs 1v1 (2 joueurs), 2v2 (4 joueurs), 3v3 (6 joueurs), etc.
"""

import subtr_actor
import pandas as pd
import numpy as np
import glob
import os
import re
import sys
import json

# --- PARAMETRES ---
DOSSIER_REPLAYS = r"C:\Users\Edouard\Documents\Edouard\Projet RL\Script Extraction info game\replay\bot"
DOSSIER_SORTIE_JSON = r"C:\Users\Edouard\Documents\Edouard\Projet RL\Script Extraction info game\database"

GLOBAL_FEATURES = []
PLAYER_FEATURES = ["PlayerRigidBody", "PlayerBoost", "PlayerAnyJump"]

# --- PARAMETRES MACHINE LEARNING ---
# Variables pour sélectionner exactement les infos voulues pour l'entrainement.
# Les noms doivent correspondre aux "player_headers" (ex: "position x", "linear velocity y", "boost level (raw replay units)")
COLONNES_A_GARDER = [
    "position x", "position y", "position z",
    "linear velocity x", "linear velocity y", "linear velocity z",
    "rotation x", "rotation y", "rotation z",
    "boost level (raw replay units)", "any_jump_active",
    "steer", "throttle", "ballcam"
]

TAILLE_SEQUENCE = 60  # Nombre de frames consécutives dans une seule séquence (ex: 60 frames = 4 secondes)
PAS_GLISSEMENT = 30   # Décalage glissant : on génère une séquence, on avance de 30 frames, on recoupe 60 frames, etc.

# Bots connus
LISTE_BOTS = ["/",
        "Discon Ψ",
        "HDZFRJHRDJ",
        "KaydopMaFaitBan",
        "MushyIdelis",
        "OptionalVorname",
        "Road2SSL-Evample",
        "StormX732",
        "TheFluff RL",
        "champ rylan.",
        "cutefemboyfeet .",
        "kemo-4",
        "this ur peak LOL",
        "top g zltj",
        "ty doesnt bot",
        "wesoo. .",
        "μ μ μ zay μ μ",
        "μʔμ"]

# Pattern split-screen : pseudos comme "Joueur(1)", "Joueur(2)", etc.
SPLIT_SCREEN_PATTERN = re.compile(r"^(.+)\((\d+)\)$")


def extraire_noms_joueurs(meta, chemin_replay=None):
    """
    Extrait les noms des joueurs depuis les métadonnées du replay.
    Si les métadonnées sont vides, tente une extraction via le flux réseau (fallback).
    """
    noms_joueurs = []
    replay_meta = meta.get('replay_meta', {})
    for item in replay_meta.get('all_headers', []):
        if item[0] == 'PlayerStats':
            for stats in item[1]:
                noms_joueurs.append(stats.get('Name', 'Unknown'))
    
    # Fallback : Si aucun nom trouvé dans les headers, on fouille le réseau
    if not noms_joueurs and chemin_replay:
        try:
            with open(chemin_replay, "rb") as f:
                d = subtr_actor.parse_replay(f.read())
            
            obj_pri_name = None
            for i, obj in enumerate(d['objects']):
                if obj == 'Engine.PlayerReplicationInfo:PlayerName': 
                    obj_pri_name = i
                    break
            
            if obj_pri_name is not None:
                noms_trouves = set()
                # On scanne les 2000 premières frames réseau pour trouver les noms
                for f in d['network_frames']['frames'][:2000]:
                    for a in f['updated_actors']:
                        if a['object_id'] == obj_pri_name:
                            nom = a['attribute'].get('String')
                            if nom: noms_trouves.add(nom)
                noms_joueurs = list(noms_trouves)
        except:
            pass
            
    return noms_joueurs


def detecter_split_screen(noms_joueurs):
    """
    Détecte les joueurs en écran scindé.
    Retourne un dict {index: info} pour les joueurs split-screen.
    """
    split_screen_info = {}
    for i, nom in enumerate(noms_joueurs):
        match = SPLIT_SCREEN_PATTERN.match(nom)
        if match:
            split_screen_info[i] = {
                'nom_complet': nom,
                'nom_base': match.group(1),
                'numero': int(match.group(2))
            }
    return split_screen_info


def construire_dataframe(meta, matrice, headers_info):
    """Construit un DataFrame avec des colonnes nommées à partir de la matrice."""
    df = pd.DataFrame(matrice)

    noms_colonnes = headers_info['global_headers'].copy()
    nb_colonnes_joueurs = df.shape[1] - len(noms_colonnes)
    nb_joueurs = nb_colonnes_joueurs // len(headers_info['player_headers'])

    for i in range(nb_joueurs):
        for header in headers_info['player_headers']:
            noms_colonnes.append(f"P{i}_{header}")

    df.columns = noms_colonnes
    return df, nb_joueurs


def creer_sequences_joueur(df, slot_matrice, colonnes_cibles, taille_seq, pas):
    """
    Extrait le bloc de colonnes d'un joueur précis et génère une matrice 3D
    transformée en liste de listes en utilisant l'algorithme de fenêtre glissante.
    """
    # 1. Identifier les noms exacts des colonnes dans le DataFrame brut
    colonnes_brutes = []
    # Parfois, les en-têtes retournés par subtr_actor peuvent varier légèrement selon la version,
    # on essaie de matcher le plus intelligemment possible si besoin, mais ici on va direct à la source
    for col in df.columns:
        if col.startswith(f"P{slot_matrice}_"):
            nom_racine = col.replace(f"P{slot_matrice}_", "")
            if nom_racine in colonnes_cibles:
                colonnes_brutes.append(col)
                
    # 2. Extraire et nettoyer les données
    if not colonnes_brutes:
        return []
        
    df_joueur = df[colonnes_brutes].astype('float32')
    # Remplacer les valeurs manquantes/NaN par 0.0 par sécurité pour l'IA
    donnees_matrice = df_joueur.fillna(0.0).values
    
    sequences_listes = []
    
    # 3. Fenêtre glissante (Sliding Window)
    total_frames = len(donnees_matrice)
    for i in range(0, total_frames - taille_seq + 1, pas):
        # On découpe le bloc [i : i + 60] -> 60 lignes, N colonnes
        bloc = donnees_matrice[i : i + taille_seq]
        # On convertit le bloc Numpy 2D en liste de listes standard Python
        sequences_listes.append(bloc.tolist())
        
    return sequences_listes


def preparer_dataframe_ml_global(nom_fichier, df, nb_joueurs_matrice, noms_joueurs, split_info):
    """
    Compile un nouveau DataFrame constitué uniquement des listes de listes (séquences)
    prêtes à l'emploi par les modèles d'IA.
    """
    # Filtrer l'écran scindé de la même manière pour trouver les bons mapping
    indices_valides = [i for i in range(len(noms_joueurs)) if i not in split_info]
    indices_valides = indices_valides[:nb_joueurs_matrice]
    mapping_matrice = {idx: slot for slot, idx in enumerate(indices_valides)}
    
    lignes_finales = []
    
    # Pour chaque joueur valide, on extrait ses séquences glissantes
    for idx_joueur, slot_matrice in mapping_matrice.items():
        nom = noms_joueurs[idx_joueur]
        est_un_bot = 1 if (nom in LISTE_BOTS) else 0
        
        # Appel à notre générateur de séquences
        sequences = creer_sequences_joueur(df, slot_matrice, COLONNES_A_GARDER, TAILLE_SEQUENCE, PAS_GLISSEMENT)
        
        # On ajoute chaque séquence construite au dataset d'export global
        for num_seq, seq in enumerate(sequences):
            lignes_finales.append({
                "game_id": str(nom_fichier),
                "joueur": str(nom),
                "is_bot": est_un_bot,
                "sequence_index": num_seq,
                "sequence_data": seq
            })
            
    df_export = pd.DataFrame(lignes_finales)
    return df_export



def extraire_inputs_du_reseau(fichier, noms_joueurs_valides, mapping_noms_vers_slots, nb_frames_matrice):
    """
    Extrait les inputs bruts du parseur réseau (steer, throttle, ballcam) et les associe
    dynamiquement aux bons noms de joueurs et aux slots (P0, P1...) de la matrice.
    """
    import subtr_actor
    import pandas as pd
    with open(fichier, "rb") as f:
        full_replay_data = subtr_actor.parse_replay(f.read())
        
    frames_data = full_replay_data["network_frames"]["frames"]
    objects = full_replay_data["objects"]
    
    O_PRI_NAME = None
    O_PAWN_PRI = None
    O_CAM_PRI = None
    O_CAM_BEHIND = None
    O_CAM_SEC = None
    O_STEER = None
    O_THROTTLE = None

    for i, obj in enumerate(objects):
        if obj == "Engine.PlayerReplicationInfo:PlayerName": O_PRI_NAME = i
        elif obj == "Engine.Pawn:PlayerReplicationInfo": O_PAWN_PRI = i
        elif obj == "TAGame.CameraSettingsActor_TA:PRI": O_CAM_PRI = i
        elif obj == "TAGame.CameraSettingsActor_TA:bUsingBehindView": O_CAM_BEHIND = i
        elif obj == "TAGame.CameraSettingsActor_TA:bUsingSecondaryCamera": O_CAM_SEC = i
        elif obj == "TAGame.Vehicle_TA:ReplicatedSteer": O_STEER = i
        elif obj == "TAGame.Vehicle_TA:ReplicatedThrottle": O_THROTTLE = i
        
    pri_to_name = {}
    car_to_name = {}
    cam_to_name = {}
    
    net_inputs_by_frame = []
    current_inputs = {
        name: {"steer": 0.0, "throttle": 0.0, "ballcam": False}
        for name in noms_joueurs_valides
    }
    
    for frame in frames_data:
        for actor_update in frame.get("updated_actors", []):
            actor_id = actor_update.get("actor_id")
            object_id = actor_update.get("object_id")
            attr = actor_update.get("attribute", {})
            
            if object_id == O_PRI_NAME:
                name = attr.get("String")
                if name: pri_to_name[actor_id] = name
                continue
                
            if object_id in (O_PAWN_PRI, O_CAM_PRI):
                active_actor = attr.get("ActiveActor", {}) if isinstance(attr, dict) else {}
                if isinstance(active_actor, dict) and active_actor.get("active"):
                    pri_actor_id = active_actor.get("actor")
                    name = pri_to_name.get(pri_actor_id)
                    if name and name in noms_joueurs_valides:
                        if object_id == O_PAWN_PRI:
                            car_to_name[actor_id] = name
                        else:
                            cam_to_name[actor_id] = name
                continue
                
            if object_id in (O_STEER, O_THROTTLE) and actor_id in car_to_name:
                name = car_to_name[actor_id]
                value = attr.get("Float")
                if value is None:
                    value = attr.get("Byte")
                    if value is not None:
                        value = (value - 128) / 127.0
                        
                if value is not None:
                    if object_id == O_STEER:
                        current_inputs[name]["steer"] = float(value)
                    else:
                        current_inputs[name]["throttle"] = float(value)
                continue
                
            if object_id in (O_CAM_BEHIND, O_CAM_SEC) and actor_id in cam_to_name:
                name = cam_to_name[actor_id]
                current_inputs[name]["ballcam"] = bool(attr.get("Boolean", False))
                
        net_inputs_by_frame.append({
            name: {
                "steer": vals["steer"],
                "throttle": vals["throttle"],
                "ballcam": vals["ballcam"]
            } for name, vals in current_inputs.items()
        })
        
    nb_net_frames = len(net_inputs_by_frame)
    if nb_net_frames == 0 or nb_frames_matrice == 0:
        return pd.DataFrame()
        
    df_reseau_records = []
    
    for state_frame_idx in range(nb_frames_matrice):
        net_frame_idx = int(nb_net_frames * state_frame_idx / max(1, nb_frames_matrice))
        net_frame_idx = min(max(0, net_frame_idx), nb_net_frames - 1)
        
        snapshot = net_inputs_by_frame[net_frame_idx]
        record_plat = {}
        for name, vals in snapshot.items():
            if name not in mapping_noms_vers_slots: continue
            slot = mapping_noms_vers_slots[name]
            record_plat[f"P{slot}_steer"] = vals["steer"]
            record_plat[f"P{slot}_throttle"] = vals["throttle"]
            record_plat[f"P{slot}_ballcam"] = float(1.0 if vals["ballcam"] else 0.0)
            
        df_reseau_records.append(record_plat)
        
    return pd.DataFrame(df_reseau_records)


def main():

    print("=" * 80)
    print("  🎮 EXTRACTION DATAFRAME DES DONNÉES JOUEURS - ROCKET LEAGUE")
    print("  📂 Détection split-screen | Export format .csv")
    print("=" * 80)

    os.makedirs(DOSSIER_SORTIE_JSON, exist_ok=True)

    # Récupérer les headers une seule fois
    headers_info = subtr_actor.get_column_headers(
        global_feature_adders=GLOBAL_FEATURES,
        player_feature_adders=PLAYER_FEATURES
    )

    # Scanner les replays (récursif)
    fichiers = glob.glob(os.path.join(DOSSIER_REPLAYS, "**", "*.replay"), recursive=True)
    print(f"\n  📁 Dossier source  : {DOSSIER_REPLAYS}")
    print(f"   Dossier sortie  : {DOSSIER_SORTIE_JSON}")
    print(f"  📄 {len(fichiers)} fichiers .replay trouvés\n")

    if not fichiers:
        print("  ❌ Aucun fichier replay trouvé !")
        return

    resultats = []

    for idx, fichier in enumerate(fichiers):
        nom_fichier = os.path.basename(fichier)
        chemin_relatif = os.path.relpath(fichier, DOSSIER_REPLAYS)

        try:
            # Extraction via subtr_actor
            meta, matrice = subtr_actor.get_ndarray_with_info_from_replay_filepath(
                fichier,
                global_feature_adders=GLOBAL_FEATURES,
                player_feature_adders=PLAYER_FEATURES,
                fps=30.0
            )

            # Construction du DataFrame
            df, nb_joueurs_matrice = construire_dataframe(meta, matrice, headers_info)

            # Extraction des noms (avec fallback réseau si besoin)
            noms_joueurs = extraire_noms_joueurs(meta, fichier)

            # Détection split-screen
            split_info = detecter_split_screen(noms_joueurs)
            split_msg = ""
            if split_info:
                noms_split = [info['nom_complet'] for info in split_info.values()]
                split_msg = f" | ⚠️ Split-screen: {', '.join(noms_split)}"

            # Nombre réel de joueurs
            nb_joueurs_total = max(nb_joueurs_matrice, len(noms_joueurs))

            # --- EXTRACTION RESEAU DYNAMIQUE ---
            indices_valides = [i for i in range(len(noms_joueurs)) if i not in split_info]
            indices_valides = indices_valides[:nb_joueurs_matrice]
            mapping_noms_vers_slots = {noms_joueurs[idx]: slot for slot, idx in enumerate(indices_valides)}
            noms_joueurs_valides = list(mapping_noms_vers_slots.keys())

            df_net = extraire_inputs_du_reseau(fichier, noms_joueurs_valides, mapping_noms_vers_slots, len(df))
            if not df_net.empty:
                df = pd.concat([df, df_net], axis=1)

            # Construction du DataFrame formaté en listes de listes (séquences)
            df_final = preparer_dataframe_ml_global(nom_fichier, df, nb_joueurs_matrice, noms_joueurs, split_info)

            # Sauvegarde au format CSV
            nom_sortie = nom_fichier.replace('.replay', '.csv')
            chemin_sortie = os.path.join(DOSSIER_SORTIE_JSON, nom_sortie)
            df_final.to_csv(chemin_sortie, index=False)

            joueurs_str = ", ".join(noms_joueurs)
            print(f"  [{idx + 1}/{len(fichiers)}] ✅ {chemin_relatif}")
            print(f"       {nb_joueurs_total}j ({nb_joueurs_matrice} dans matrice) | {len(df_final)} séquences générées | Joueurs: {joueurs_str}{split_msg}")
            
            # Affichage du Dataset comme demandé
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(df_final.head(10)) # On n'affiche que les 10 premières lignes pour ne pas flood le terminal

            resultats.append({
                "fichier": chemin_relatif,
                "nb_joueurs": nb_joueurs_total,
                "nb_frames": len(df),
                "joueurs": noms_joueurs,
                "split_screen": len(split_info) > 0,
                "succes": True
            })

        except Exception as e:
            print(f"  [{idx + 1}/{len(fichiers)}] ❌ {chemin_relatif} : {e}")
            resultats.append({
                "fichier": chemin_relatif,
                "nb_joueurs": 0,
                "nb_frames": 0,
                "joueurs": [],
                "split_screen": False,
                "succes": False,
                "erreur": str(e)
            })

    # --- Rapport final ---
    print("\n" + "=" * 80)
    print("  📊 RAPPORT FINAL")
    print("=" * 80)

    succes = [r for r in resultats if r['succes']]
    echecs = [r for r in resultats if not r['succes']]

    print(f"  ✅ Réussis  : {len(succes)}/{len(resultats)}")
    print(f"  ❌ Échoués  : {len(echecs)}/{len(resultats)}")

    # Répartition par nombre de joueurs
    par_nb_joueurs = {}
    for r in succes:
        nb = r['nb_joueurs']
        par_nb_joueurs[nb] = par_nb_joueurs.get(nb, 0) + 1

    if par_nb_joueurs:
        print("\n  🎮 Répartition par type de match :")
        for nb, count in sorted(par_nb_joueurs.items()):
            mode = {2: "1v1", 4: "2v2", 6: "3v3"}.get(nb, f"{nb} joueurs")
            print(f"      {mode} : {count} replays")

    # Split-screen
    avec_split = [r for r in succes if r['split_screen']]
    if avec_split:
        print(f"\n  ⚠️  {len(avec_split)} replays avec split-screen détecté :")
        for r in avec_split:
            print(f"      {r['fichier']} ({', '.join(r['joueurs'])})")

    if echecs:
        print(f"\n  ❌ Détail des erreurs :")
        for r in echecs:
            print(f"      {r['fichier']} : {r.get('erreur', '?')}")

    # Sauvegarder le rapport en JSON
    chemin_rapport = os.path.join(DOSSIER_SORTIE_JSON, "_rapport_extraction.json")
    with open(chemin_rapport, 'w', encoding='utf-8') as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)
    print(f"\n  📄 Rapport sauvegardé : {chemin_rapport}")


if __name__ == "__main__":
    main()
