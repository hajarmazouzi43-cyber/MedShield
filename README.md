# 🛡️ MedShield Lab — Sujet 25 : robustesse du diagnostic par imagerie médicale

Simulateur d'attaques adversariales (FGSM, PGD) sur des classifieurs d'images médicales
(**Vision Transformer** et CNN), avec **explicabilité** (attention rollout / Grad-CAM), **détecteur** (IDS),
**purification** (IPS), **adversarial training**, **benchmark** et **rapport PDF**.

## Lancement
```bash
pip install -r requirements.txt
streamlit run app.py            # ou lancer.bat (Windows) / lancer.sh (Linux, Mac)
```
Le dossier `models/` contient des modèles de **démo synthétique** (fonctionnent hors-ligne).
Pour de vrais résultats médicaux (à citer dans le rapport) :
```bash
python entrainer_tout.py        # entraîne ViT, CNN, ViT+adv, CNN+adv sur PneumoniaMNIST et DermaMNIST
```
(ou page **⚙️ Modèles**). Les données MedMNIST se téléchargent automatiquement.

## Mettre en ligne (avoir un lien)
Voir **DEPLOIEMENT.md** (Streamlit Community Cloud ou Hugging Face Spaces).

## Pages
| Page | Contenu |
|---|---|
| 🏠 Accueil | Boutons **Article** et **Simulateur**, image du thème, présentation du projet, infos de l'auteure |
| 📄 Article | Lecture en ligne de l'article, **export PDF / Word / HTML / Markdown / texte**, résumé, remplacement du PDF |
| ⚙️ Modèles | Entraînement standard ou **adversarial training** |
| 🔬 Simulateur | Image → attaque → diagnostic avant/après → XAI → **détecteur** → **purification** → **rapport PDF** → courbe de robustesse |
| 📊 Benchmark | Précision sous attaque, ASR, detection rate, FPR, F1, temps, précision après purification, **ROC**, export CSV |
| 📘 Guide | Les 8 rubriques du rapport (problématique, objectifs, architecture, conception, use case, organigramme, algorithmes, vue de test) |

## Article (page 📄)
Le PDF est dans `assets/article.pdf`. Pour le mettre à jour : page **📄 Article → Remplacer l'article** (ou remplace le fichier puis relance l'app) ;
les exports (`assets/export/`) sont recréés automatiquement. Le Word est reconstruit à partir du PDF : texte modifiable, figures/tableaux/équations en images.

## Structure du code
```
core/data.py      données MedMNIST / synthétiques     core/detector.py  détecteur (algorithme 2)
core/models.py    TinyViT, CNN                        core/defense.py   purification (gauss / médian)
core/attacks.py   FGSM, PGD, bruit                    core/bench.py     benchmark + ROC
core/xai.py       rollout, Grad-CAM (par lots)        core/report.py    rapport PDF
core/train.py     entraînement standard / adversarial core/viz.py       affichage                           core/article.py   lecture + exports de l'article
```

## Détecteur (core/detector.py)
Trois signaux calculés sur l'image reçue : **s** (distance de Mahalanobis des features aux images propres),
**c** (corrélation entre la carte XAI de l'image et celle de l'image purifiée), **p** (décalage de prédiction après purification).
Verdict *ADVERSARIAL* si `s > τs` ou `c < τe` ou `p > τp` ; seuils calibrés sur des images **propres** pour un FPR cible de 5 %.

## À mettre à jour dans l'article
- **Fig. 8 / section IX.B** : l'architecture est *Streamlit + modules Python* (pas PHP + REST).
- **Section IX** : remplacer « experimental evaluation is left for future work » par les résultats du benchmark (Table + courbes + ROC).
- **Section IX.A** : ajouter O7 (atténuation : purification et adversarial training) et l'export de rapport.
- **Limites à citer** : images 28×28, modèles non pré-entraînés, détecteur évalué contre des attaques *non adaptatives*.
