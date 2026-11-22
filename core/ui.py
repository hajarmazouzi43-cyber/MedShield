"""Habillage commun de l'application (thème, cartes, en-tête)."""
import streamlit as st

CSS = """
<style>
.block-container{padding-top:2rem;max-width:1250px}
h1,h2,h3{letter-spacing:-.01em}
.hero{padding:26px 30px;border-radius:18px;margin-bottom:18px;
 background:linear-gradient(135deg,#0f766e 0%,#1d4ed8 60%,#7c3aed 100%);color:#fff}
.hero h1{margin:0 0 6px 0;font-size:2.1rem;color:#fff}
.hero p{margin:0;opacity:.92;font-size:1.02rem}
.card{background:#131c2e;border:1px solid #22304a;border-radius:14px;padding:14px 16px;height:100%}
.card .t{font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;color:#8fa3c4}
.card .v{font-size:1.65rem;font-weight:700;margin-top:2px}
.card .s{font-size:.82rem;color:#8fa3c4;margin-top:2px}
.ok{color:#34d399}.bad{color:#f87171}.warn{color:#fbbf24}
.banner{padding:14px 18px;border-radius:12px;font-weight:600;margin:10px 0}
.banner.bad{background:#3b1219;border:1px solid #f87171;color:#fecaca}
.banner.ok{background:#0f2a22;border:1px solid #34d399;color:#bbf7d0}
.banner.warn{background:#33270a;border:1px solid #fbbf24;color:#fde68a}
.cap{color:#8fa3c4;font-size:.85rem;text-align:center;margin-top:-6px}
button[kind="primary"],button[data-testid="stBaseButton-primary"]{
 background:linear-gradient(135deg,#0f766e 0%,#1d4ed8 100%)!important;color:#fff!important;border:0!important;
 font-weight:700!important;font-size:1.15rem!important;padding:.85rem 1.4rem!important;border-radius:14px!important;
 box-shadow:0 6px 20px rgba(29,78,216,.45);transition:transform .15s,box-shadow .15s}
button[kind="primary"]:hover,button[data-testid="stBaseButton-primary"]:hover{
 transform:translateY(-2px);box-shadow:0 10px 26px rgba(29,78,216,.6)}
.author{display:flex;gap:18px;align-items:center;background:#131c2e;border:1px solid #22304a;border-radius:16px;padding:18px 22px;margin-top:8px}
.author .av{flex:0 0 64px;height:64px;border-radius:50%;background:linear-gradient(135deg,#0f766e,#1d4ed8);display:flex;align-items:center;
 justify-content:center;font-size:1.5rem;font-weight:700;color:#fff}
.author .n{font-size:1.2rem;font-weight:700;color:#fff}.author .r{color:#a9b9d6;font-size:.95rem;line-height:1.5}
.author a{color:#7cc4ff;text-decoration:none}
.homeintro{font-size:1.05rem;line-height:1.7;color:#d6e0f2;text-align:justify}
.stDownloadButton button{border-radius:12px!important;font-weight:600!important}
</style>
"""


def setup():
    """À appeler une seule fois, dans app.py (point d'entrée)."""
    st.set_page_config(page_title="MedShield Lab", page_icon="🛡️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def card(title, value, sub="", cls=""):
    return f'<div class="card"><div class="t">{title}</div><div class="v {cls}">{value}</div><div class="s">{sub}</div></div>'


def cards(items):
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        c.markdown(card(*it), unsafe_allow_html=True)


def banner(text, cls):
    st.markdown(f'<div class="banner {cls}">{text}</div>', unsafe_allow_html=True)


def scroll_to(anchor_id):
    """Fait défiler la page jusqu'à l'élément #anchor_id (fonctionne avec les anciennes et nouvelles versions de Streamlit)."""
    html = ("<script>let n=0;const t=setInterval(()=>{"
            f"const el=window.parent.document.getElementById('{anchor_id}');"
            "if(el){el.scrollIntoView({behavior:'smooth',block:'start'});clearInterval(t);}"
            "if(++n>40)clearInterval(t);},150);</script>")
    if hasattr(st, "iframe"):                      # Streamlit récent
        st.iframe(html, height=1)
    else:                                          # Streamlit plus ancien
        import streamlit.components.v1 as components
        components.html(html, height=0)


def author_card():
    """Bloc « auteur » affiché en bas de l'accueil."""
    st.markdown("""
<div class="author"><div class="av">HM</div><div>
<div class="n">Hajar MAZOUZI</div>
<div class="r">Étudiante en informatique · ENSA Berrechid — Université Hassan 1er, Maroc<br>
✉️ <a href="mailto:mazouzi.ensa@uhp.ac.ma">mazouzi.ensa@uhp.ac.ma</a><br>
Projet de module — Cybersécurité · Sujet 25 : robustesse du diagnostic par imagerie médicale (Transformer + XAI)</div>
</div></div>""", unsafe_allow_html=True)
