import numpy as np
import pandas as pd
import streamlit as st
import torch
from core import ui, data, train, attacks, xai, viz, models, detector, defense, report

ui.hero("🔬 Simulateur d'attaque", "Choisis une image, une attaque, puis clique sur le bouton : tu vois si le diagnostic change et si le système détecte l'attaque.")

avail = [(d, k) for d in data.DATASETS for k in models.KEYS if train.exists(d, k)]
if not avail:
    st.warning("Aucun modèle trouvé dans le dossier `models/`. Va dans la page **⚙️ Modèles** pour en entraîner un.")
    st.stop()


@st.cache_resource
def get_model(ds, key, stamp):
    return train.load_model(ds, key)


@st.cache_data
def get_test(ds):
    X, y, _ = data.load(ds, "test")
    return X, y


@st.cache_resource
def get_det(ds, key, stamp):
    return detector.get(ds, key)


def stamp(ds, key):
    return train.path(ds, key).stat().st_mtime


# ------------------------------------------------------------------ guide
with st.expander("📖 Guide d'utilisation (30 secondes)", expanded=True):
    st.markdown("""
| | Étape | Ce que tu fais |
|---|---|---|
| **1** | **Image** | Choisis une image médicale avec le curseur « Image n° ». |
| **2** | **Attaque** | Choisis le type d'attaque et sa force (**Forte** = effet le plus visible). |
| **3** | **Lancer** | Clique sur le gros bouton **🚀 Lancer l'attaque**. |
| **4** | **Résultat** | Lis le bandeau coloré : 🔴 diagnostic inversé = attaque réussie · 🟢 attaque détectée = le système se défend. |

**Principe :** l'attaque ajoute à l'image un bruit invisible à l'œil. Le modèle change pourtant de diagnostic. Le détecteur et la purification essaient de réagir.
""")

# ------------------------------------------------------------------ choix (3 colonnes)
datasets = sorted({d for d, _ in avail})
c1, c2, c3 = st.columns([1.1, 1.2, 1])

with c1:
    st.markdown("### 1️⃣ Image")
    ds = st.selectbox("Jeu de données", datasets, format_func=data.DATASETS.get) if len(datasets) > 1 else datasets[0]
    Xte, yte = get_test(ds)
    idx = st.slider("Image n°", 0, len(yte) - 1, 1)
    st.image(viz.to_img(torch.from_numpy(Xte[idx])), width=170)

with c2:
    st.markdown("### 2️⃣ Attaque")
    atk = st.radio("Type d'attaque", list(attacks.ATTACKS), format_func=attacks.ATTACKS.get, index=1)
    strength = st.select_slider("Force de l'attaque", ["Faible", "Moyenne", "Forte"], value="Forte",
                                help="Faible : ε = 4/255 · Moyenne : ε = 8/255 · Forte : ε = 16/255")
    eps255 = {"Faible": 4, "Moyenne": 8, "Forte": 16}[strength]

with c3:
    st.markdown("### 3️⃣ Lancer")
    st.write("")
    go = st.button("🚀 Lancer l'attaque", type="primary", width="stretch")
    st.caption("Les résultats s'affichent juste en dessous.")

with st.expander("⚙️ Options avancées (facultatif)"):
    a1, a2, a3 = st.columns(3)
    keys_ds = [k for d, k in avail if d == ds]
    key = a1.radio("Modèle", keys_ds, format_func=models.KEYS.get)
    pur = a2.selectbox("Purification", ["gauss", "median", "none"], format_func=defense.PURIFIERS.get)
    steps = a3.slider("Itérations PGD", 2, 40, 10)


# ------------------------------------------------------------------ calcul
def run_sim():
    model, ck = get_model(ds, key, stamp(ds, key))
    classes = ck["classes"]
    x0 = torch.from_numpy(Xte[idx:idx + 1]); y = int(yte[idx])
    eps = eps255 / 255
    xa = attacks.run(atk, model, x0, torch.tensor([y]), eps, steps)
    xp = defense.purify(xa, pur)
    with torch.no_grad():
        p0, pa, pp = (model(v).softmax(1)[0] for v in (x0, xa, xp))
    m0, ma = xai.explain(model, x0), xai.explain(model, xa)
    det = get_det(ds, key, stamp(ds, key))
    r0, ra = det.detect(model, x0), det.detect(model, xa)
    return dict(ds=ds, key=key, atk=atk, eps255=eps255, steps=steps, pur=pur, idx=idx, classes=classes, y=y,
                x0=x0, xa=xa, xp=xp, p0=p0, pa=pa, pp=pp, m0=m0, ma=ma, kind=model.kind,
                det=dict(ts=det.ts, te=det.te, tp=det.tp), r0={k: v for k, v in r0.items()}, ra={k: v for k, v in ra.items()},
                linf=float((xa - x0).abs().max()) * 255, l2=float((xa - x0).norm()))


if go:
    with st.spinner("Attaque en cours… (la première fois, le détecteur se calibre : quelques secondes)"):
        st.session_state["sim"] = run_sim()

R = st.session_state.get("sim")
if not R:
    st.info("👆 Choisis tes réglages puis clique sur **🚀 Lancer l'attaque**.")
    st.stop()

# ------------------------------------------------------------------ résultat
classes, y = R["classes"], R["y"]
pc, pa, pp = int(R["p0"].argmax()), int(R["pa"].argmax()), int(R["pp"].argmax())
corr, iou = xai.shift(R["m0"], R["ma"])
ps = viz.psnr(R["x0"].numpy(), R["xa"].numpy())
flag_a, flag_c = bool(R["ra"]["flag"][0]), bool(R["r0"]["flag"][0])

st.divider()
st.markdown('<div id="resultats"></div>', unsafe_allow_html=True)
st.header("Résultat")
if go:  # défilement automatique vers les résultats juste après le clic
    ui.scroll_to("resultats")
st.caption(f"Image n°{R['idx']} · {models.KEYS[R['key']]} · {attacks.ATTACKS[R['atk']]} · ε = {R['eps255']}/255 "
           f"· vérité terrain : **{classes[y]}**")

# 1) L'attaque a-t-elle marché ?
if pc != y:
    ui.banner("⚠️ Le modèle se trompait déjà sur l'image propre. Choisis une autre image (curseur « Image n° ») pour une démonstration claire.", "warn")
elif pa != y:
    ui.banner(f"🔴 ATTAQUE RÉUSSIE : le diagnostic passe de « {classes[pc]} » (correct) à « {classes[pa]} » (faux), "
              f"avec un bruit de seulement {R['linf']:.0f}/255.", "bad")
else:
    ui.banner(f"🟢 Attaque sans effet : le diagnostic reste « {classes[pa]} ». Essaie une force plus grande ou une autre image.", "ok")

# 2) Le détecteur l'a-t-il vue ?
if flag_a:
    ui.banner(f"🛡️ ATTAQUE DÉTECTÉE par le système en {R['ra']['ms']:.1f} ms.", "ok")
else:
    ui.banner("🕳️ Attaque NON détectée : le système considère l'image comme normale.", "bad")
if flag_c:
    st.caption("Note : l'image propre est aussi signalée (faux positif ; le détecteur vise environ 5 % de faux positifs).")

# 3) La purification a-t-elle réparé ?
if pa != y and pp == y:
    ui.banner(f"✅ PURIFICATION RÉUSSIE : après nettoyage de l'image, le diagnostic redevient « {classes[pp]} ».", "ok")
elif pa != y:
    ui.banner(f"La purification ne rétablit pas le bon diagnostic (« {classes[pp]} »).", "warn")

# Images
st.subheader("Avant / après")
cols = st.columns(5)
pert = viz.perturbation_img(R["x0"][0], R["xa"][0], R["eps255"] / 255)
for c, (im, cap) in zip(cols, [(viz.to_img(R["x0"][0]), f"Originale → {classes[pc]}"),
                               (viz.to_img(R["xa"][0]), f"Attaquée → {classes[pa]}"),
                               (pert, "Bruit ajouté (amplifié)"),
                               (viz.to_img(R["xp"][0]), f"Purifiée → {classes[pp]}")]):
    c.image(im, width="stretch"); c.markdown(f'<div class="cap">{cap}</div>', unsafe_allow_html=True)
with cols[4]:
    st.bar_chart(pd.DataFrame({"Avant": R["p0"].numpy(), "Après": R["pa"].numpy()}, index=classes), horizontal=True, stack=False)
    st.markdown('<div class="cap">Confiance du modèle</div>', unsafe_allow_html=True)

st.subheader("Là où le modèle regarde (explication)")
meth = "Attention rollout" if R["kind"] == "vit" else "Grad-CAM"
e1, e2, e3 = st.columns(3)
e1.image(viz.overlay(R["x0"][0], R["m0"]), width="stretch"); e1.markdown(f'<div class="cap">{meth} · image propre</div>', unsafe_allow_html=True)
e2.image(viz.overlay(R["xa"][0], R["ma"]), width="stretch"); e2.markdown(f'<div class="cap">{meth} · image attaquée</div>', unsafe_allow_html=True)
e3.image(viz.overlay(R["x0"][0], np.abs(R["m0"] - R["ma"]), 0.6), width="stretch"); e3.markdown('<div class="cap">Différence</div>', unsafe_allow_html=True)

# Rapport PDF
det = R["det"]; ra = R["ra"]
lines = [f"Modèle : {models.KEYS[R['key']]}   |   Attaque : {R['atk']} (ε = {R['eps255']}/255)   |   Image n°{R['idx']}",
         f"Diagnostic avant : {classes[pc]} ({R['p0'][pc]:.1%})   après attaque : {classes[pa]} ({R['pa'][pa]:.1%})   après purification : {classes[pp]} ({R['pp'][pp]:.1%})",
         f"Vérité terrain : {classes[y]}   |   Perturbation : Linf = {R['linf']:.1f}/255, L2 = {R['l2']:.2f}, PSNR = {ps:.1f} dB",
         f"Décalage de l'explication : 1-IoU = {1 - iou:.0%}, corrélation = {corr:.2f}",
         f"Détecteur : {'ADVERSARIAL' if flag_a else 'CLEAN'} | s={ra['s'][0]:.2f} (τs={det['ts']:.2f}) c={ra['c'][0]:.2f} (τe={det['te']:.2f}) "
         f"p={ra['p'][0]:.2f} (τp={det['tp']:.2f}) | {ra['ms']:.1f} ms"]
probs = "\n".join(f"{c[:18]:18s} {a:5.1%} -> {b:5.1%}" for c, a, b in zip(classes, R["p0"].numpy(), R["pa"].numpy()))
pdf = report.make_pdf({"xai": meth, "lines": lines, "probs": probs}, R["x0"][0].numpy(), R["xa"][0].numpy(), R["xp"][0].numpy(),
                      R["m0"], R["ma"], pert)
st.download_button("📄 Télécharger le rapport PDF", pdf, "rapport_medshield.pdf", "application/pdf")

# Détails (repliés)
with st.expander("📐 Détails techniques et courbe de robustesse"):
    ui.cards([
        ("Perturbation max", f"{R['linf']:.1f}/255", f"L2 = {R['l2']:.2f} · PSNR {ps:.1f} dB"),
        ("Décalage explication", f"{(1 - iou):.0%}", f"IoU = {iou:.2f} · corr. = {corr:.2f}"),
        ("Signaux détecteur", f"s={ra['s'][0]:.1f}", f"c={ra['c'][0]:.2f} · p={ra['p'][0]:.2f}"),
    ])
    st.write("")
    if st.button("📈 Calculer la précision en fonction de ε (150 images)"):
        _, ck = get_model(R["ds"], R["key"], stamp(R["ds"], R["key"]))
        Xt, yt = get_test(R["ds"])
        sel = np.random.default_rng(0).choice(len(yt), min(150, len(yt)), replace=False)
        xs, ys = torch.from_numpy(Xt[sel]), torch.from_numpy(yt[sel])
        mdl = get_model(R["ds"], R["key"], stamp(R["ds"], R["key"]))[0]
        eps_list = [0, 1, 2, 4, 8, 12, 16, 24]
        accs = []
        for e in eps_list:
            xadv = attacks.run(R["atk"], mdl, xs, ys, e / 255, R["steps"]) if e else xs
            with torch.no_grad():
                accs.append(float((mdl(xadv).argmax(1) == ys).float().mean()))
        st.line_chart(pd.DataFrame({"Précision": accs}, index=eps_list).rename_axis("ε (sur 255)"))
        st.caption("Plus la courbe chute vite, moins le modèle est robuste.")
st.info("Pour des chiffres complets (ASR, detection rate, FPR, F1…) utilise la page **📊 Benchmark**.")
