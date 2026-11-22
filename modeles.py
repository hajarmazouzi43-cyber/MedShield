import pandas as pd
import streamlit as st
from core import ui, data, train, models

ui.hero("⚙️ Modèles", "Entraîne un Vision Transformer et un CNN, en version standard ou avec adversarial training (défense).")

c1, c2, c3 = st.columns(3)
ds = c1.selectbox("Jeu de données", list(data.DATASETS), format_func=data.DATASETS.get)
key = c2.selectbox("Modèle", list(models.KEYS), format_func=models.KEYS.get)
epochs = c3.slider("Époques", 2, 30, 8)
if models.is_adv(key):
    st.info("**Adversarial training** : à chaque lot, des exemples PGD (3 pas, ε aléatoire ≤ 8/255) sont générés et "
            "ajoutés à la perte (min-max, éq. 3 de l'article). L'entraînement est ~3× plus lent.")
if ds != "synthetic":
    st.info("Les jeux MedMNIST sont téléchargés automatiquement au premier entraînement (connexion requise).")
if train.exists(ds, key):
    st.success("Un modèle existe déjà pour cette combinaison : le réentraîner le remplace.")

if st.button("🚀 Entraîner", type="primary"):
    bar = st.progress(0.0, "Préparation des données…")
    try:
        hist = train.train(ds, key, epochs=epochs, progress=lambda p, m: bar.progress(p, m))
        bar.progress(1.0, "Terminé")
        st.success(f"Modèle enregistré · précision test finale : {hist[-1]['test_acc']:.1%}")
        st.line_chart(pd.DataFrame(hist).set_index("epoch")[["test_acc"]])
    except Exception as e:  # réseau absent, dataset introuvable…
        st.error(f"Échec de l'entraînement : {e}")

st.subheader("Modèles disponibles")
rows = []
for d in data.DATASETS:
    for k in models.KEYS:
        if train.exists(d, k):
            ck = train.load_model(d, k)[1]
            rows.append({"Jeu de données": d, "Modèle": models.KEYS[k],
                         "Précision test": f"{ck['history'][-1]['test_acc']:.1%}",
                         "Époques": len(ck["history"]), "Durée (s)": round(ck["seconds"])})
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True) if rows else st.caption("Aucun modèle pour l'instant.")
