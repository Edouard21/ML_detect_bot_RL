import subtr_actor

# Optionnel: on peut ajouter "PlayerCamera" si c'est géré, etc.
GLOBAL_FEATURES = []
PLAYER_FEATURES = ["PlayerRigidBody", "PlayerBoost", "PlayerAnyJump"]

def main():
    print("=" * 60)
    print("📋 LISTE DES COLONNES DISPONIBLES POUR L'EXTRACTION")
    print("=" * 60)

    # Récupérer la définition des colonnes générées par subtr_actor
    headers_info = subtr_actor.get_column_headers(
        global_feature_adders=GLOBAL_FEATURES,
        player_feature_adders=PLAYER_FEATURES
    )

    print("\n🌍 COLONNES GLOBALES (Communes à la partie) :")
    global_headers = headers_info.get('global_headers', [])
    if global_headers:
        for idx, col in enumerate(global_headers):
            print(f"  - \"{col}\"")
    else:
        print("  (Aucune colonne globale avec les features actuels)")

    print("\n🏎️ COLONNES JOUEUR (A copier dans COLONNES_A_GARDER) :")
    player_headers = headers_info.get('player_headers', [])
    if player_headers:
        for idx, col in enumerate(player_headers):
            print(f"  - \"{col}\"")
    else:
        print("  (Aucune)")

    print("\n" + "=" * 60)
    print("💡 Astuce : Copie/Colle les lignes exactes (avec les guillemets)")
    print("dans la liste COLONNES_A_GARDER de ton script principal.")
    print("=" * 60)

if __name__ == "__main__":
    main()
