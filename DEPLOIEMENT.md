# Mettre MedShield Lab en ligne (obtenir un lien)

## Option A — Streamlit Community Cloud (gratuit, le plus simple)
Lien obtenu : `https://TON-NOM.streamlit.app`

1. Crée un compte sur **github.com**, puis **New repository** → nom : `medshield` (Public, ou Private).
2. Dans le dépôt : **Add file → Upload files**. Glisse **le contenu** du dossier `medshield`
   (`app.py`, `core/`, `views/`, `models/`, `assets/`, `.streamlit/`, `requirements.txt`…), **pas le dossier lui-même** :
   `app.py` doit être à la racine du dépôt. Clique **Commit changes**.
3. Va sur **share.streamlit.io** → **Continue with GitHub** → **Create app**.
   - Repository : `ton-compte/medshield` · Branch : `main` · Main file path : `app.py`
   - (Advanced settings → Python 3.12)
   - Choisis un sous-domaine (ex. `medshield-hajar`) puis **Deploy**.
4. Attends 5 à 10 minutes (installation de PyTorch). Ton lien est prêt.

À savoir : l'app se met en veille après 12 h sans visite (un clic la réveille) ; mémoire limitée (~2,7 Go).
Si tu vois « Oh no / resource limits », utilise l'option B.

## Option B — Hugging Face Spaces (gratuit, 16 Go de RAM, meilleur pour PyTorch)
Lien obtenu : `https://huggingface.co/spaces/TON-COMPTE/medshield`

1. Compte sur **huggingface.co** → **New Space** → SDK : **Streamlit** → Hardware : **CPU basic (free)** → Create.
2. **Files → Add file → Upload files** : envoie le contenu du dossier `medshield` (comme ci-dessus).
3. Ouvre `README.md` du Space et mets **tout en haut** :
```
---
title: MedShield Lab
emoji: 🛡️
sdk: streamlit
app_file: app.py
---
```
4. Le Space se construit seul (quelques minutes).

## Avant de publier : vrais modèles médicaux
Les modèles fournis sont synthétiques (démo). Pour de vrais résultats : en local, `python entrainer_tout.py`,
puis ajoute les fichiers du dossier `models/` au dépôt. (Les données MedMNIST se téléchargent au premier usage.)

## Sans Internet : montrer le projet en classe
`streamlit run app.py` (ou `lancer.bat`) puis ouvre http://localhost:8501.
Sur le même Wi-Fi, les autres appareils peuvent ouvrir l'adresse « Network URL » affichée dans le terminal.
