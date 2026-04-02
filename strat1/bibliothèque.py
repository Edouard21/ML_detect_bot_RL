import subtr_actor
import pprint

replay_test = "C:/Users/Edouard/Documents/Edouard/Projet RL/Script Extraction info game/6a008b64-deba-411e-bad1-c9a40ff9ebd8.replay"

meta, matrice = subtr_actor.get_ndarray_with_info_from_replay_filepath(
    replay_test, 
    global_feature_adders=["BallRigidBody"], 
    player_feature_adders=["PlayerRigidBody"]
)

print("--- IDENTITÉ DES COLONNES DE LA MATRICE ---")
pprint.pprint(meta['column_headers'])