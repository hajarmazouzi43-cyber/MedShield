import re
import streamlit as st
from core import ui, article

ui.hero("📄 Article scientifique",
        "Lis l'article en ligne, télécharge-le en PDF, Word, HTML, Markdown ou texte, ou remplace-le par une nouvelle version.")

if not article.PDF.exists():
    st.warning("Aucun article trouvé : téléverse un PDF dans l'onglet « Remplacer l'article ».")


@st.cache_data(show_spinner="Préparation de la lecture…")
def pages(mtime):
    return article.render_pages(article.PDF)


@st.cache_data(show_spinner="Création des exports (Word, HTML, Markdown, texte)…")
def exports(mtime):
    return {k: p.read_bytes() for k, p in article.ensure_exports(article.PDF).items()}


def abstract_text():
    f = article.EXPORT / "article.txt"
    if f.exists():
        m = re.search(r"Abstract—(.+?)\n\s*\n", f.read_text(encoding="utf-8"), re.S)
        return m.group(1).strip() if m else ""
    return ""


if article.PDF.exists():
    mt = article.PDF.stat().st_mtime
    try:
        info = article.stats()
        c = st.columns(3)
        c[0].metric("Pages", info["pages"]); c[1].metric("Mots", f"{info['words']:,}".replace(",", " ")); c[2].metric("Format d'origine", "PDF (IEEE)")
    except Exception:
        pass

    # ---------------- téléchargements
    st.subheader("⬇️ Télécharger / exporter")
    files = exports(mt) if article.available() else {}
    mimes = {"docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "html": "text/html",
             "md": "text/markdown", "txt": "text/plain"}
    cols = st.columns(5)
    cols[0].download_button("📕 PDF (original)", article.PDF.read_bytes(), "article.pdf", "application/pdf", type="primary", width="stretch")
    for col, (k, label) in zip(cols[1:], article.FORMATS.items()):
        if k in files:
            col.download_button(label, files[k], f"article.{k}", mimes[k], width="stretch")
        else:
            col.button(label, disabled=True, width="stretch", key=f"dis_{k}")
    if not files:
        st.info("Les exports Word / HTML / Markdown / texte nécessitent `pymupdf` et `python-docx` : `pip install -r requirements.txt`.")
    st.caption("Le **PDF** est la version fidèle. Le **Word** est reconstruit à partir du PDF : texte, titres et listes sont modifiables ; "
               "figures, tableaux, équations et algorithmes sont insérés comme images. Les formules écrites dans une phrase peuvent être "
               "imparfaites : vérifie-les avant de modifier.")

    tab1, tab2, tab3 = st.tabs(["📖 Lire en ligne", "📝 Résumé", "🔄 Remplacer l'article"])

    with tab1:
        imgs = pages(mt)
        n = len(imgs)
        if "art_page" not in st.session_state:
            st.session_state["art_page"] = 1
        st.session_state["art_page"] = min(max(1, st.session_state["art_page"]), n)
        p1, p2, p3 = st.columns([1, 3, 1])
        if p1.button("◀ Précédente", disabled=st.session_state["art_page"] <= 1, width="stretch"):
            st.session_state["art_page"] -= 1; st.rerun()
        if p3.button("Suivante ▶", disabled=st.session_state["art_page"] >= n, width="stretch"):
            st.session_state["art_page"] += 1; st.rerun()
        p2.slider("Page", 1, n, key="art_page")
        _, mid, _ = st.columns([1, 4, 1])
        mid.image(imgs[st.session_state["art_page"] - 1], width="stretch")

    with tab2:
        ab = abstract_text()
        st.markdown(f"**Résumé (Abstract)**\n\n{ab}" if ab else "Le résumé sera affiché une fois les exports créés.")

    with tab3:
        st.write("Téléverse une nouvelle version PDF : elle remplace l'article actuel et tous les exports sont recréés automatiquement.")
        up = st.file_uploader("Nouveau PDF", type=["pdf"])
        if up is not None and st.button("Remplacer l'article", type="primary"):
            article.PDF.write_bytes(up.getvalue())
            with st.spinner("Création des exports…"):
                if article.available():
                    article.build_all(article.PDF)
            st.cache_data.clear()
            st.session_state["art_page"] = 1
            st.success("Article remplacé.")
            st.rerun()
