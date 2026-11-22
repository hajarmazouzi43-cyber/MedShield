import numpy as np
import pandas as pd
import streamlit as st
from core import ui, data, train, models, attacks, bench

ui.hero("📊 Benchmark", "Précision sous attaque, ASR, détection (detection rate, FPR, F1, temps) et purification, en un clic.")

avail = {d: [k for k in models.KEYS if train.exists(d, k)] for d in data.DATASETS}
avail = {d: ks for d, ks in avail.items() if ks}
if not avail:
    st.warning("Aucun modèle entraîné. Va d'abord dans **⚙️ Modèles**.")
    st.stop()

c1, c2 = st.columns(2)
ds = c1.selectbox("Jeu de données", list(avail), format_func=data.DATASETS.get)
keys = c2.multiselect("Modèles à comparer", avail[ds], default=avail[ds][:2], format_func=models.KEYS.get)
c3, c4, c5 = st.columns(3)
atks = c3.multiselect("Attaques", ["FGSM", "PGD", "Bruit"], default=["FGSM", "PGD"])
eps_list = c4.multiselect("Budgets ε (sur 255)", [1, 2, 4, 8, 12, 16, 24], default=[2, 4, 8, 16])
n = c5.slider("Images de test", 50, 500, 200, 50)

if st.button("▶️ Lancer le benchmark", type="primary", disabled=not (keys and atks and eps_list)):
    bar = st.progress(0.0, "Calibration du détecteur…")
    st.session_state["bench"] = (ds, bench.run(ds, keys, atks, sorted(eps_list), n, progress=lambda p, m: bar.progress(p, m)))
    bar.progress(1.0, "Terminé")

if "bench" in st.session_state:
    d0, df = st.session_state["bench"]
    st.subheader(f"Résultats — {data.DATASETS[d0]}")
    pct = [c for c in df.columns if c not in ("Modèle", "Attaque", "ε (/255)", "Temps détection (ms/img)")]
    st.dataframe(df.style.format({**{c: "{:.1%}" for c in pct}, "Temps détection (ms/img)": "{:.2f}"}),
                 width="stretch", hide_index=True)
    st.download_button("⬇️ Télécharger le CSV", df.to_csv(index=False).encode("utf-8"), f"benchmark_{d0}.csv", "text/csv")

    df["Série"] = df["Modèle"] + " · " + df["Attaque"]
    for col, title in [("Précision sous attaque", "Précision sous attaque (courbe de robustesse)"),
                       ("Detection rate", "Taux de détection selon ε"),
                       ("Précision après purification", "Précision après purification selon ε")]:
        st.markdown(f"**{title}**")
        st.line_chart(df.pivot(index="ε (/255)", columns="Série", values=col))

    st.subheader("Courbes ROC du détecteur")
    r1, r2, r3 = st.columns(3)
    rk = r1.selectbox("Modèle", keys or list(df["Modèle"].unique()), format_func=lambda k: models.KEYS.get(k, k)) \
        if keys else None
    ra = r2.selectbox("Attaque ", [a for a in ["FGSM", "PGD"] if a in df["Attaque"].values] or ["FGSM"])
    re_ = r3.selectbox("ε ", sorted(df["ε (/255)"].unique()), index=min(2, len(df["ε (/255)"].unique()) - 1))
    if rk and st.button("Tracer la ROC"):
        out = bench.roc(d0, rk, ra, re_)
        names = {"s": "Anomalie des features", "c": "Cohérence XAI", "p": "Décalage de prédiction"}
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.5, 4))
        for k, (f, t, auc) in out.items():
            ax.plot(f, t, label=f"{names[k]} (AUC {auc:.2f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.set_xlabel("FPR"); ax.set_ylabel("Detection rate"); ax.legend(fontsize=7)
        st.pyplot(fig)
else:
    st.info("Lance le benchmark pour obtenir les tableaux et courbes de la section « Vue de test » de l'article.")
