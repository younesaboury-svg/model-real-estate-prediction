"""
Module de prédiction — LOCATION (loyers) | GateOne.immo Marrakech
Charge les modèles .pkl de location depuis models_location/ et expose
une fonction de prédiction par catégorie, fidèle à la logique des
notebooks d'origine (entraînement sur loyers mensuels).
"""
import os
import joblib
import numpy as np
import pandas as pd

MODEL_DIR_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_location")

_CATEGORIES_LOCATION = ["villas", "maisons", "bureaux", "appartements", "magasins", "riads"]
_artefacts_cache_location = {}

AVERTISSEMENTS_LOCATION = {
    "maisons": "Incertitude élevée (±25-30%) : dataset d'entraînement réduit (~100 lignes).",
    "riads": "Incertitude élevée (±20-25%) : dataset d'entraînement réduit (71 lignes).",
}


def load_all_models_location():
    for cat in _CATEGORIES_LOCATION:
        path = os.path.join(MODEL_DIR_LOCATION, f"model_{cat}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modèle locatif manquant : {path}")
        _artefacts_cache_location[cat] = joblib.load(path)
    return list(_artefacts_cache_location.keys())


def get_loaded_categories_location():
    return list(_artefacts_cache_location.keys())


def _get_art(cat: str):
    if cat not in _artefacts_cache_location:
        raise RuntimeError(f"Le modèle locatif '{cat}' n'est pas chargé.")
    return _artefacts_cache_location[cat]


def get_equip_features_location(cat: str):
    return _get_art(cat).get("equip_features", [])


# ══════════════════════════════════════════════════════════
# VILLAS
# ══════════════════════════════════════════════════════════
def predict_villa(surface_totale, surface_habitable, chambres, salle_de_bain,
                   nb_etages, salons, caution, localisation,
                   equipements=None, use_model="lgb"):
    art = _get_art("villas")
    eq = equipements or {}
    row = {
        'surface_totale': surface_totale,
        'surface_habitable': surface_habitable,
        'chambres': chambres,
        'salle_de_bain': salle_de_bain,
        'nb_etages': nb_etages,
        'salons': salons,
        'caution': caution,
        'localisation': localisation,
    }
    for ef in art['equip_features']:
        row[ef] = eq.get(ef, 0)

    X_pred = pd.DataFrame([row])[art['feature_cols']]
    X_pred[art['cat_features']] = art['ord_enc'].transform(X_pred[art['cat_features']])
    X_pred = art['target_enc'].transform(X_pred)

    model = art['xgb_model'] if use_model == "xgb" else art['lgb_model']
    return int(round(np.expm1(model.predict(X_pred)[0])))


# ══════════════════════════════════════════════════════════
# MAISONS
# ══════════════════════════════════════════════════════════
def predict_maison(surface_habitable, surface_terrain=None,
                    chambres=3, salles_bain=2, etage=1, salons=2,
                    localisation='Inconnue', equipements=None):
    art = _get_art("maisons")
    eq = equipements or {}
    if surface_terrain is None:
        surface_terrain = surface_habitable

    loc = localisation if localisation not in art['rare_quartiers'] else 'Autre'
    row = {
        'surface_habitable': surface_habitable,
        'surface_terrain': surface_terrain,
        'chambres': chambres,
        'salles_bain': salles_bain,
        'etage': etage,
        'salons': salons,
        'localisation_grp': loc,
    }
    for ef in art['equip_features']:
        row[ef] = eq.get(ef, 0)

    X_pred = pd.DataFrame([row])[art['feature_cols']]
    X_pred = art['target_enc'].transform(X_pred)

    return int(round(np.expm1(art['model'].predict(X_pred)[0])))


# ══════════════════════════════════════════════════════════
# BUREAUX
# ══════════════════════════════════════════════════════════
def predict_bureau(surface_m2, nb_etage, nb_pieces, chambres, salle_de_bain, frais_syndic,
                    localisation, equipements=None, use_model="lgb"):
    art = _get_art("bureaux")
    eq = equipements or {}
    row = {
        'surface_m2': surface_m2,
        'nb_etage': nb_etage,
        'nb_pieces': nb_pieces,
        'chambres': chambres,
        'salle_de_bain': salle_de_bain,
        'frais_syndic': frais_syndic,
        'localisation': localisation,
    }
    for ef in art['equip_features']:
        row[ef] = eq.get(ef, 0)

    X_pred = pd.DataFrame([row])[art['feature_cols']]
    X_pred = art['target_enc'].transform(X_pred)

    model = art['xgb_model'] if use_model == "xgb" else art['lgb_model']
    return int(round(np.expm1(model.predict(X_pred)[0])))


# ══════════════════════════════════════════════════════════
# APPARTEMENTS
# ══════════════════════════════════════════════════════════
def predict_appartement(surface_totale, surface_habitable, chambres, salle_de_bain,
                         nb_etages, salons, frais_syndic,
                         type_appartement, caution, localisation,
                         equipements=None, use_model="lgb"):
    art = _get_art("appartements")
    eq = equipements or {}
    row = {
        'surface_totale': surface_totale,
        'surface_habitable': surface_habitable,
        'chambres': chambres,
        'salle_de_bain': salle_de_bain,
        'nb_etages': nb_etages,
        'salons': salons,
        'frais_syndic': frais_syndic,
        'type_appartement': type_appartement,
        'caution': caution,
        'localisation': localisation,
    }
    for ef in art['equip_features']:
        row[ef] = eq.get(ef, 0)

    X_pred = pd.DataFrame([row])[art['feature_cols']]
    X_pred[art['cat_features']] = art['ord_enc'].transform(X_pred[art['cat_features']])
    X_pred = art['target_enc'].transform(X_pred)

    model = art['xgb_model'] if use_model == "xgb" else art['lgb_model']
    return int(round(np.expm1(model.predict(X_pred)[0])))


# ══════════════════════════════════════════════════════════
# MAGASINS
# ══════════════════════════════════════════════════════════
def predict_magasin(surface_m2, salle_de_bain=0, localisation='Inconnue',
                     equipements=None, use_model="lgb"):
    art = _get_art("magasins")
    eq = equipements or {}
    row = {
        'surface_m2': surface_m2,
        'salle_de_bain': salle_de_bain,
        'localisation': localisation,
    }
    for ef in art['equip_features']:
        row[ef] = eq.get(ef, 0)

    X_pred = pd.DataFrame([row])[art['feature_cols']]
    X_pred = art['target_enc'].transform(X_pred)

    model = art['xgb_model'] if use_model == "xgb" else art['lgb_model']
    return int(round(np.expm1(model.predict(X_pred)[0])))


# ══════════════════════════════════════════════════════════
# RIADS
# ══════════════════════════════════════════════════════════
def predict_riad(surface_m2, surface_habitable=None,
                  nb_etages=3, chambres=5, salles_de_bain=4, salons=2,
                  caution='2 mois', localisation='Ancienne Médina',
                  equipements=None):
    art = _get_art("riads")
    eq = equipements or {}
    if surface_habitable is None:
        surface_habitable = surface_m2

    loc = localisation if localisation not in art['rare_quartiers'] else 'Autre'

    row = {
        "Surface (m²)": surface_m2,
        "Surface habitable (m²)": surface_habitable,
        "Nombre d'étages": nb_etages,
        "Chambres": chambres,
        "Salles de bain": salles_de_bain,
        "Salons": salons,
        "Caution": caution,
        "localisation_grp": loc,
    }
    for ef in art['equip_features']:
        row[ef] = eq.get(ef, 0)

    X_pred = pd.DataFrame([row])[art['feature_cols']]
    X_pred[art['cat_features']] = art['ord_enc'].transform(X_pred[art['cat_features']])
    X_pred = art['target_enc'].transform(X_pred)

    return int(round(np.expm1(art['model'].predict(X_pred)[0])))
