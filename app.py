"""MedShield Lab : point d'entrée. Lancer avec :  streamlit run app.py"""
import streamlit as st
from core import ui

ui.setup()
pg = st.navigation([
    st.Page("views/accueil.py", title="Accueil", icon="🏠", default=True),
    st.Page("views/article.py", title="Article", icon="📄"),
    st.Page("views/simulateur.py", title="Simulateur", icon="🔬"),
    st.Page("views/modeles.py", title="Modèles", icon="⚙️"),
    st.Page("views/benchmark.py", title="Benchmark", icon="📊"),
    st.Page("views/guide.py", title="Guide de réalisation", icon="📘"),
])
with st.sidebar:
    st.caption("🛡️ **MedShield Lab**  \nHajar MAZOUZI · ENSA Berrechid")
pg.run()
