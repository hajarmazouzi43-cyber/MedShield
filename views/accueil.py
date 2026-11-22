from pathlib import Path
import streamlit as st

HERO = Path(__file__).resolve().parent.parent / "assets" / "hero.png"

# =============================================================== style : accueil BLANC + animations
st.markdown("""
<style>
/* ---------- fond blanc avec halos de couleur ---------- */
.stApp{background-color:#ffffff!important;color:#0f172a!important;
 background-image:
  radial-gradient(900px 520px at 6% -8%, rgba(29,78,216,.14), transparent 60%),
  radial-gradient(800px 520px at 100% 2%, rgba(16,185,129,.16), transparent 55%),
  radial-gradient(900px 500px at 55% 108%, rgba(124,58,237,.12), transparent 60%)!important}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stToolbar"] *,[data-testid="stHeader"] *{color:#0f172a!important}
[data-testid="stSidebar"]{background:#f1f5f9!important;border-right:1px solid #e2e8f0}
[data-testid="stSidebar"] *{color:#0f172a!important}
[data-testid="stSidebarNav"] a[aria-current="page"]{background:#dbeafe!important}
.block-container{max-width:1180px;padding-top:2.2rem}

/* ---------- 2 gros boutons de navigation ---------- */
.st-key-btn_article button,.st-key-btn_sim button{
 height:84px;border:0!important;border-radius:20px!important;color:#fff!important;position:relative;overflow:hidden;
 box-shadow:0 14px 34px rgba(29,78,216,.35);transition:transform .2s ease, box-shadow .2s ease;animation:fadeUp .7s ease both}
.st-key-btn_article button{background:linear-gradient(120deg,#1d4ed8 0%,#0ea5e9 100%)!important}
.st-key-btn_sim button{background:linear-gradient(120deg,#059669 0%,#0d9488 55%,#0891b2 100%)!important;box-shadow:0 14px 34px rgba(5,150,105,.35);animation-delay:.1s}
.st-key-btn_article button *,.st-key-btn_sim button *{color:#fff!important;font-size:1.35rem!important;font-weight:800!important;letter-spacing:.01em}
.st-key-btn_article button:hover,.st-key-btn_sim button:hover{transform:translateY(-5px) scale(1.02);box-shadow:0 22px 44px rgba(15,23,42,.30)}
.st-key-btn_article button:before,.st-key-btn_sim button:before{content:"";position:absolute;top:0;left:-70%;width:45%;height:100%;
 background:linear-gradient(100deg,transparent,rgba(255,255,255,.45),transparent);transform:skewX(-20deg);animation:shine 3.2s ease-in-out infinite}

/* ---------- image du thème ---------- */
.st-key-heroimg img{border-radius:28px;box-shadow:0 30px 70px rgba(29,78,216,.30);animation:floaty 6s ease-in-out infinite}

/* ---------- titre ---------- */
.pill{display:inline-block;padding:6px 16px;border-radius:999px;background:#e0e7ff;color:#3730a3;font-weight:700;font-size:.85rem;
 letter-spacing:.04em;text-transform:uppercase;animation:fadeUp .8s ease both}
.wow-title{font-size:4rem;line-height:1.05;font-weight:900;margin:.35em 0 .15em 0;
 background:linear-gradient(90deg,#1d4ed8,#0ea5e9,#10b981,#7c3aed,#1d4ed8);background-size:300% auto;
 -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;color:transparent;animation:flow 7s linear infinite}
.wow-sub{font-size:1.45rem;font-weight:600;color:#334155;margin-bottom:1.1rem;animation:fadeUp 1s ease both}

/* ---------- paragraphe ---------- */
.homeintro{background:rgba(255,255,255,.88);border:1px solid #e2e8f0;border-left:7px solid #1d4ed8;border-radius:20px;
 padding:26px 30px;box-shadow:0 12px 34px rgba(15,23,42,.08);font-size:1.12rem;line-height:1.8;color:#1e293b;text-align:justify;animation:fadeUp 1.1s ease both}
.homeintro b{color:#1d4ed8}

/* ---------- 4 cartes ---------- */
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin:26px 0 8px 0}
.fcard{background:#fff;border:1px solid #e2e8f0;border-radius:20px;padding:22px 18px;text-align:center;
 box-shadow:0 8px 24px rgba(15,23,42,.06);transition:transform .25s,box-shadow .25s,border-color .25s;animation:fadeUp 1.2s ease both}
.fcard:hover{transform:translateY(-8px);box-shadow:0 22px 44px rgba(29,78,216,.22);border-color:#93c5fd}
.fcard .ic{font-size:2.3rem;line-height:1}
.fcard .t{font-weight:800;font-size:1.08rem;margin:10px 0 4px 0;color:#0f172a}
.fcard .d{font-size:.92rem;color:#475569;line-height:1.45}
@media(max-width:900px){.grid4{grid-template-columns:repeat(2,1fr)}.wow-title{font-size:2.8rem}}

/* ---------- carte auteure ---------- */
.me{display:flex;gap:24px;align-items:center;padding:28px 32px;border-radius:26px;margin-top:10px;
 background:linear-gradient(120deg,#eff6ff 0%,#ecfdf5 100%);border:1px solid #bfdbfe;box-shadow:0 18px 44px rgba(29,78,216,.14);animation:fadeUp 1.2s ease both}
.me .av{flex:0 0 92px;height:92px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:2rem;font-weight:900;color:#fff;
 background:linear-gradient(135deg,#1d4ed8,#10b981);box-shadow:0 10px 26px rgba(29,78,216,.40)}
.me .nm{font-size:1.7rem;font-weight:900;color:#0f172a}
.me .rl{color:#334155;font-size:1.02rem;line-height:1.75}
.me a{color:#1d4ed8;font-weight:700;text-decoration:none}
.me .proj{margin-top:6px;color:#475569;font-size:.92rem}
.foot{text-align:center;color:#94a3b8;font-size:.85rem;margin:22px 0 6px 0}

/* ---------- animations ---------- */
@keyframes fadeUp{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:none}}
@keyframes floaty{0%,100%{transform:translateY(0)}50%{transform:translateY(-9px)}}
@keyframes flow{to{background-position:300% center}}
@keyframes shine{0%{left:-70%}55%,100%{left:130%}}
</style>
""", unsafe_allow_html=True)

# =============================================================== 2 boutons visibles en haut
c1, c2 = st.columns(2, gap="large")
with c1:
    with st.container(key="btn_article"):
        if st.button("📄   Lire l'article scientifique", width="stretch"):
            st.switch_page("views/article.py")
with c2:
    with st.container(key="btn_sim"):
        if st.button("🔬   Lancer le simulateur", width="stretch"):
            st.switch_page("views/simulateur.py")
st.write("")

# =============================================================== image du thème
with st.container(key="heroimg"):
    st.image(str(HERO), width="stretch")

# =============================================================== titre + paragraphe
st.markdown("""
<div style="text-align:center;margin-top:18px">
  <span class="pill">Projet de module · Cybersécurité · Sujet 25</span>
  <div class="wow-title">🛡️ MedShield Lab</div>
  <div class="wow-sub">Robustesse du diagnostic par imagerie médicale face aux attaques adversariales</div>
</div>
<div class="homeintro">
Les systèmes d'intelligence artificielle aident aujourd'hui à diagnostiquer des radiographies, des lésions cutanées ou des images
de la rétine. Or une modification <b>invisible à l'œil</b> de l'image suffit parfois à inverser le diagnostic du modèle, sans laisser
de trace : c'est une attaque contre l'<b>intégrité</b> du diagnostic. Ce projet réunit deux livrables : un <b>article de synthèse</b>
sur les attaques contre les modèles <b>Transformer</b> (Vision Transformer) et le rôle de l'<b>IA explicable (XAI)</b>, et un
<b>simulateur</b> qui injecte l'attaque (FGSM, PGD), montre l'effet sur le diagnostic et sur l'explication, <b>détecte</b> l'attaque,
la <b>neutralise</b> par purification et mesure les performances (précision sous attaque, taux de détection, faux positifs).
</div>

<div class="grid4">
  <div class="fcard"><div class="ic">🎯</div><div class="t">Attaques FGSM · PGD</div><div class="d">Une perturbation imperceptible qui inverse le diagnostic</div></div>
  <div class="fcard"><div class="ic">🧠</div><div class="t">ViT et CNN</div><div class="d">Transformer comparé à un CNN, standard ou adversarial training</div></div>
  <div class="fcard"><div class="ic">🔍</div><div class="t">IA explicable (XAI)</div><div class="d">Attention rollout et Grad-CAM avant / après l'attaque</div></div>
  <div class="fcard"><div class="ic">🛡️</div><div class="t">Détection + purification</div><div class="d">Un IDS / IPS pour l'image, avec benchmark chiffré</div></div>
</div>
""", unsafe_allow_html=True)

# =============================================================== infos de l'auteure (en bas)
st.write("")
st.markdown("""
<div class="me">
  <div class="av">HM</div>
  <div>
    <div class="nm">Hajar MAZOUZI</div>
    <div class="rl">🎓 Étudiante en informatique · <b>ENSA Berrechid</b> — Université Hassan 1er, Maroc<br>
    ✉️ <a href="mailto:mazouzi.ensa@uhp.ac.ma">mazouzi.ensa@uhp.ac.ma</a></div>
    <div class="proj">Projet de module — Cybersécurité · Sujet 25 : robustesse du diagnostic par imagerie médicale (Transformer + XAI)</div>
  </div>
</div>
<div class="foot">MedShield Lab · 2026</div>
""", unsafe_allow_html=True)
