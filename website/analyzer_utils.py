import subtr_actor
import pandas as pd
import numpy as np
import os
import re

# Colonnes physiques à conserver (noms originaux subtr_actor)
COLONNES_PHYSIQUES = [
    "position x", "position y", "position z",
    "rotation x", "rotation y", "rotation z",
    "linear velocity x", "linear velocity y", "linear velocity z",
    "boost level (raw replay units)"
]

# Colonnes réseau à ajouter
COLONNES_RESEAU = ["steer", "throttle", "ballcam", "handbrake"]

# Pattern split-screen : pseudos comme "Joueur(1)", "Joueur(2)", etc.
SPLIT_SCREEN_PATTERN = re.compile(r"^(.+)\((\d+)\)$")

def extraire_noms_joueurs(info, chemin_replay=None):
    noms_joueurs = []
    replay_meta = info.get('replay_meta', {})
    for item in replay_meta.get('all_headers', []):
        if item[0] == 'PlayerStats':
            for stats in item[1]:
                noms_joueurs.append(stats.get('Name', 'Unknown'))
    
    if not noms_joueurs and chemin_replay:
        try:
            with open(chemin_replay, "rb") as f:
                d = subtr_actor.parse_replay(f.read())
            obj_pri_name = next((i for i, o in enumerate(d['objects']) if o == 'Engine.PlayerReplicationInfo:PlayerName'), None)
            if obj_pri_name is not None:
                noms_trouves = set()
                for f in d['network_frames']['frames'][:2000]:
                    for a in f['updated_actors']:
                        if a['object_id'] == obj_pri_name:
                            nom = a['attribute'].get('String')
                            if nom: noms_trouves.add(nom)
                noms_joueurs = list(noms_trouves)
        except: pass
    return noms_joueurs

def detecter_split_screen(noms_joueurs):
    split_screen_info = {}
    for i, nom in enumerate(noms_joueurs):
        match = SPLIT_SCREEN_PATTERN.match(nom)
        if match:
            split_screen_info[i] = {'nom_complet': nom, 'nom_base': match.group(1), 'numero': int(match.group(2))}
    return split_screen_info

def construire_dataframe_brut(info, matrice):
    df = pd.DataFrame(matrice)
    h = info['column_headers']
    
    nb_global = len(h['global_headers'])
    nb_player_cols = len(h['player_headers'])
    nb_slots = (df.shape[1] - nb_global) // nb_player_cols
    
    noms_complets = list(h['global_headers'])
    for i in range(nb_slots):
        for hp in h['player_headers']:
            noms_complets.append(f"P{i}_{hp}")
            
    df.columns = noms_complets
    return df, nb_slots

def extraire_inputs_du_reseau_complet(fichier, noms_joueurs_valides, mapping_noms_vers_slots, nb_frames_matrice):
    with open(fichier, "rb") as f:
        full_replay_data = subtr_actor.parse_replay(f.read())
    
    frames_data = full_replay_data["network_frames"]["frames"]
    objects = full_replay_data["objects"]
    
    target_objs = {
        "Engine.PlayerReplicationInfo:PlayerName": "O_PRI_NAME",
        "Engine.Pawn:PlayerReplicationInfo": "O_PAWN_PRI",
        "TAGame.CameraSettingsActor_TA:PRI": "O_CAM_PRI",
        "TAGame.CameraSettingsActor_TA:bUsingBehindView": "O_CAM_BEHIND",
        "TAGame.CameraSettingsActor_TA:bUsingSecondaryCamera": "O_CAM_SEC",
        "TAGame.Vehicle_TA:ReplicatedSteer": "O_STEER",
        "TAGame.Vehicle_TA:ReplicatedThrottle": "O_THROTTLE",
        "TAGame.Vehicle_TA:bReplicatedHandbrake": "O_HANDBRAKE"
    }
    obj_ids = {val: next((i for i, o in enumerate(objects) if o == key), None) for key, val in target_objs.items()}

    pri_to_name = {}
    car_to_name = {}
    cam_to_name = {}
    
    net_inputs_by_frame = []
    current_inputs = {name: {"steer": 0.0, "throttle": 0.0, "ballcam": False, "handbrake": False} for name in noms_joueurs_valides}
    
    for frame in frames_data:
        for actor_update in frame.get("updated_actors", []):
            actor_id = actor_update.get("actor_id")
            object_id = actor_update.get("object_id")
            attr = actor_update.get("attribute", {})
            
            if object_id == obj_ids["O_PRI_NAME"]:
                name = attr.get("String")
                if name: pri_to_name[actor_id] = name
            
            elif object_id in (obj_ids["O_PAWN_PRI"], obj_ids["O_CAM_PRI"]):
                active_actor = attr.get("ActiveActor", {}) if isinstance(attr, dict) else {}
                if isinstance(active_actor, dict) and active_actor.get("active"):
                    name = pri_to_name.get(active_actor.get("actor"))
                    if name in noms_joueurs_valides:
                        if object_id == obj_ids["O_PAWN_PRI"]: car_to_name[actor_id] = name
                        else: cam_to_name[actor_id] = name

            elif object_id in (obj_ids["O_STEER"], obj_ids["O_THROTTLE"], obj_ids["O_HANDBRAKE"]) and actor_id in car_to_name:
                name = car_to_name[actor_id]
                if object_id == obj_ids["O_HANDBRAKE"]:
                    current_inputs[name]["handbrake"] = bool(attr.get("Boolean", False))
                else:
                    value = attr.get("Float")
                    if value is None:
                        value = attr.get("Byte")
                        if value is not None: value = (value - 128) / 127.0
                    if value is not None:
                        if object_id == obj_ids["O_STEER"]: current_inputs[name]["steer"] = float(value)
                        else: current_inputs[name]["throttle"] = float(value)

            elif object_id in (obj_ids["O_CAM_BEHIND"], obj_ids["O_CAM_SEC"]) and actor_id in cam_to_name:
                name = cam_to_name[actor_id]
                current_inputs[name]["ballcam"] = bool(attr.get("Boolean", False))
                
        net_inputs_by_frame.append({n: dict(v) for n, v in current_inputs.items()})
    
    if not net_inputs_by_frame: return pd.DataFrame()
    
    df_reseau_records = []
    nb_net = len(net_inputs_by_frame)
    for state_idx in range(nb_frames_matrice):
        net_idx = min(max(0, int(nb_net * state_idx / max(1, nb_frames_matrice))), nb_net - 1)
        snap = net_inputs_by_frame[net_idx]
        rec = {}
        for name, vals in snap.items():
            slot = mapping_noms_vers_slots[name]
            rec[f"P{slot}_steer"] = vals["steer"]
            rec[f"P{slot}_throttle"] = vals["throttle"]
            rec[f"P{slot}_ballcam"] = float(1.0 if vals["ballcam"] else 0.0)
            rec[f"P{slot}_handbrake"] = float(1.0 if vals["handbrake"] else 0.0)
        df_reseau_records.append(rec)
        
    return pd.DataFrame(df_reseau_records)


def extract_players_from_replay(replay_path: str, output_dir: str) -> list:
    """
    Extrait les données de chaque joueur d'un .replay et sauvegarde un .csv par joueur.
    Retourne la liste des dictionnaires avec nom et chemin CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    nom_base_replay = os.path.basename(replay_path).replace('.replay', '')
    
    info, matrice = subtr_actor.get_ndarray_with_info_from_replay_filepath(replay_path, fps=30.0)
    df, nb_slots = construire_dataframe_brut(info, matrice)
    
    noms_joueurs = extraire_noms_joueurs(info, replay_path)
    split_info = detecter_split_screen(noms_joueurs)
    indices_valides = [i for i in range(len(noms_joueurs)) if i not in split_info][:nb_slots]
    mapping_noms_vers_slots = {noms_joueurs[i]: slot for slot, i in enumerate(indices_valides)}
    
    df_net = extraire_inputs_du_reseau_complet(replay_path, list(mapping_noms_vers_slots.keys()), mapping_noms_vers_slots, len(df))
    if not df_net.empty: 
        df = pd.concat([df, df_net], axis=1)
    
    resultats = []
    for nom, slot in mapping_noms_vers_slots.items():
        cols_joueur_physique = [f"P{slot}_{c}" for c in COLONNES_PHYSIQUES]
        cols_joueur_reseau = [f"P{slot}_{c}" for c in COLONNES_RESEAU]
        
        cols_finales = [c for c in (cols_joueur_physique + cols_joueur_reseau) if c in df.columns]
        
        df_joueur = df[cols_finales].copy()
        rename_map = {c: c.replace(f"P{slot}_", "") for c in cols_finales}
        df_joueur.rename(columns=rename_map, inplace=True)
        
        nom_propre = "".join(x for x in nom if x.isalnum() or x in " -_").strip()
        nom_sortie = f"{nom_propre}_{nom_base_replay}.csv"
        chemin_csv = os.path.join(output_dir, nom_sortie)
        df_joueur.to_csv(chemin_csv, index=False)
        
        resultats.append({"nom_joueur": nom_propre, "csv_path": chemin_csv})
        
    return resultats
