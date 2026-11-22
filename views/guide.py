import streamlit as st
from core import ui

ui.hero("📘 Guide de réalisation", "Les 8 rubriques du rapport demandé, prêtes à reprendre dans le document final.")

DOT = 'bgcolor="transparent"; node [fontname="Helvetica"];'
t = st.tabs(["1 Problématique", "2 Objectifs", "3 Architecture", "4 Conception", "5 Use case", "6 Organigramme", "7 Algorithmes", "8 Vue de test"])

with t[0]:
    st.markdown("""
Un diagnostic produit par un modèle d'IA est archivé et utilisé pour décider d'un traitement. Une **perturbation imperceptible**
ajoutée à l'image (par un attaquant sur le réseau, un initié ou une station compromise) peut inverser ce diagnostic **sans trace visible** :
l'**intégrité** (le « I » de la triade CIA) est violée alors que la disponibilité est préservée.

**Question centrale** : dans quelle mesure les modèles Transformer résistent-ils à ces perturbations, et l'explicabilité (XAI)
peut-elle aider à détecter ou atténuer l'attaque ?
""")
with t[1]:
    st.markdown("""
- **O1** charger une image médicale ; **O2** la classer avec un ViT (et un CNN de référence) ;
- **O3** injecter FGSM / PGD avec un ε réglable ; **O4** comparer image originale, attaquée, perturbation, diagnostic et confiance ;
- **O5** afficher l'explication (attention rollout / Grad-CAM) avant et après ;
- **O6** détecter l'attaque (detection rate, FPR, F1, temps) ; **O7** atténuer (purification, adversarial training) ;
- **O8** évaluer sur le jeu de test (benchmark) et exporter un rapport.
""")
with t[2]:
    st.graphviz_chart(f"""digraph G {{ {DOT} rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#1d4ed8", fontcolor="white"];
      UI [label="Interface Streamlit\\n(Accueil · Modèles · Simulateur · Benchmark · Guide)"];
      CORE [label="Services Python (core/)"]; UI -> CORE;
      subgraph cluster {{ label="core/"; fontcolor="gray"; color="gray";
        data [label="data\\nMedMNIST / synthétique"]; models [label="models\\nViT · CNN"]; train [label="train\\nstandard · adversarial"];
        attacks [label="attacks\\nFGSM · PGD · bruit", fillcolor="#b91c1c"]; xai [label="xai\\nrollout · Grad-CAM"];
        detector [label="detector\\nIDS (algo 2)", fillcolor="#0f766e"]; defense [label="defense\\npurification", fillcolor="#0f766e"];
        bench [label="bench\\nmétriques"]; report [label="report\\nPDF"]; }}
      CORE -> {{data models train attacks xai detector defense bench report}};
      STO [label="Stockage\\nmodels/*.pt · données · rapports", shape=cylinder, fillcolor="#475569"]; CORE -> STO; }}""")
    st.caption("Remarque : l'article décrit une architecture PHP + API REST ; cette implémentation utilise Streamlit et des modules Python. Adapte la Fig. 8 en conséquence.")
with t[3]:
    st.markdown("""
**Modèle de données**
| Entité | Attributs |
|---|---|
| Image | tenseur (C×28×28) ∈ [0,1], étiquette, jeu de données |
| Modèle | clé (vit, cnn, vit_adv, cnn_adv), poids, classes, historique d'entraînement |
| Attaque | type (FGSM/PGD/bruit), ε, pas, image attaquée, perturbation |
| Détection | s, c, p, seuils (τs, τe, τp), verdict, temps |
| Résultat | précision sous attaque, ASR, detection rate, FPR, F1, temps |

**Modules** : `data`, `models`, `train`, `attacks`, `xai`, `detector`, `defense`, `bench`, `report`, `viz`.

**Modèles IA** : TinyViT (patchs 4×4, 4 blocs, 4 têtes) et petit CNN, avec variantes *adversarial training* (PGD à 3 pas, perte 50 % propre / 50 % adversariale).
""")
with t[4]:
    st.graphviz_chart(f"""digraph G {{ {DOT} rankdir=LR; node [fontcolor="white", style=filled];
      U [label="Utilisateur\\n(analyste / étudiant)", shape=box, fillcolor="#475569"];
      node [shape=ellipse, fillcolor="#1d4ed8"];
      u1 [label="Choisir / téléverser\\nune image"]; u2 [label="Classer (ViT / CNN)"]; u3 [label="Choisir attaque\\net budget ε"];
      u4 [label="Injecter l'attaque", fillcolor="#b91c1c"]; u5 [label="Comparer\\nles diagnostics"]; u6 [label="Voir la carte XAI"];
      u7 [label="Lancer le détecteur", fillcolor="#0f766e"]; u8 [label="Purifier et\\nreclassifier", fillcolor="#0f766e"];
      u9 [label="Lancer le benchmark"]; u10 [label="Entraîner (standard /\\nadversarial)"]; u11 [label="Exporter rapport\\n(PDF / CSV)"];
      U -> {{u1 u3 u7 u8 u9 u10 u11}}; u1 -> u2 [style=dashed, label="«include»"]; u3 -> u4 [style=dashed, label="«include»"];
      u4 -> u5 [style=dashed, label="«include»"]; u5 -> u6 [style=dashed, label="«extend»"]; }}""")
    st.caption("L'attaquant est modélisé par le module de simulation, pas comme acteur humain.")
with t[5]:
    st.graphviz_chart(f"""digraph G {{ {DOT} node [shape=box, style="rounded,filled", fillcolor="#1d4ed8", fontcolor="white"];
      a [label="Début"]; b [label="Charger image x\\n+ modèle f"]; c [label="Classer : y0 = f(x)"]; d [label="Choisir attaque, ε"];
      e [label="Injecter : x' = attaque(x)", fillcolor="#b91c1c"]; f [label="Classer : y1 = f(x')"];
      g [label="Expliquer x et x'\\n(rollout / Grad-CAM)"]; h [label="Détecteur(x')\\ns, c, p", fillcolor="#0f766e"];
      q [label="s>τs ou c<τe\\nou p>τp ?", shape=diamond, fillcolor="#b45309"]; adv [label="ADVERSARIAL\\npurifier + reclassifier", fillcolor="#0f766e"];
      cl [label="CLEAN"]; k [label="Afficher diagnostics,\\ncartes, métriques"]; z [label="Fin / export rapport"];
      a->b->c->d->e->f->g->h->q; q->adv [label="oui"]; q->cl [label="non"]; adv->k; cl->k; k->z; }}""")
with t[6]:
    st.markdown("**Algorithme 1 : injection d'attaque (PGD ; FGSM = 1 pas avec α = ε)**")
    st.code("""Entrées : image x, étiquette y, modèle f, budget ε, pas α, itérations T
x0 ← x
pour t = 0 à T-1 :
    g  ← ∇x L(f(xt), y)
    xt+1 ← xt + α · signe(g)
    xt+1 ← clip(xt+1, x-ε, x+ε)      # projection sur la boule ε
    xt+1 ← clip(xt+1, 0, 1)          # image valide
retourner x_adv = xT""", language="text")
    st.markdown("**Algorithme 2 : détecteur (implémenté dans `core/detector.py`)**")
    st.code("""Entrées : image reçue x', modèle f, seuils τs, τe, τp
démarrer chrono
F ← features(f, x')                       # token CLS (ViT) / dernière conv (CNN)
s ← distance de Mahalanobis(F, statistiques des images propres)
xp ← purifier(x')                          # flou gaussien ou médian
c ← corrélation(XAI(x'), XAI(xp))          # cohérence de l'explication
p ← ‖softmax(f(x')) − softmax(f(xp))‖₁     # décalage de prédiction
si s > τs ou c < τe ou p > τp :
    étiqueter ADVERSARIAL ; (optionnel) reclassifier f(xp)
sinon : étiqueter CLEAN
arrêter chrono ; retourner étiquette, s, c, p, temps
# Calibration : seuils choisis sur images PROPRES de validation pour un FPR cible de 5 %""", language="text")
with t[7]:
    st.markdown("""
**Scénarios** : jeu de test MedMNIST (PneumoniaMNIST, DermaMNIST) ; ensembles propres et attaqués pour ε ∈ {1, 2, 4, 8, 12, 16, 24}/255 ;
attaques FGSM, PGD et bruit témoin ; modèles ViT, CNN et leurs versions *adversarial training*.

**Métriques** (page **📊 Benchmark**) : précision sous attaque, attack success rate, précision après purification,
detection rate, FPR, F1 de détection, temps de détection par image, courbes ROC et AUC.

**Limites à citer dans le rapport** : images 28×28 ; modèles entraînés from scratch (non pré-entraînés) ;
détecteur évalué contre des attaques *non adaptatives* (un attaquant qui connaît le détecteur peut le contourner) ;
un seul tirage d'images (résultats à moyenner sur plusieurs graines pour une étude rigoureuse).
""")
