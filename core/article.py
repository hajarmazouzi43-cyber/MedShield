"""Article : lecture en ligne (pages PDF -> images) et exports PDF / Word / HTML / Markdown / texte.

L'export Word est reconstruit à partir de la structure du PDF (colonnes IEEE) : titres, paragraphes et listes
sont du texte éditable ; figures, tableaux, équations et algorithmes sont insérés comme images découpées dans le PDF.
Dépendances : pymupdf, python-docx.
"""
import base64
import io
import re
from collections import Counter
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets"
PDF = ASSETS / "article.pdf"
EXPORT = ASSETS / "export"
FORMATS = {"docx": "Word (.docx)", "html": "Page web (.html)", "md": "Markdown (.md)", "txt": "Texte brut (.txt)"}

ROMAN = r"(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)"


def _fitz():
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    return fitz


def available():
    try:
        _fitz(); import docx  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- lecture en ligne
def render_pages(pdf_path=PDF, zoom=1.7):
    """Liste de PNG (bytes), une par page."""
    fitz = _fitz()
    out = []
    with fitz.open(str(pdf_path)) as doc:
        for p in doc:
            out.append(p.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png"))
    return out


def stats(pdf_path=PDF):
    fitz = _fitz()
    with fitz.open(str(pdf_path)) as doc:
        text = " ".join(p.get_text() for p in doc)
        meta = doc.metadata or {}
        return {"pages": len(doc), "words": len(text.split()), "title": meta.get("title", "")}


# --------------------------------------------------------------------------- extraction structurée
def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


def _smallcaps(s):
    """'I. I NTRODUCTION' -> 'I. INTRODUCTION'."""
    for _ in range(3):
        s = re.sub(r"\b([A-Z]) ([A-Z]{2,})", r"\1\2", s)
    return s


def _block_info(b):
    chars, sizes, math, text_lines = 0, Counter(), 0, []
    for l in b["lines"]:
        t = "".join(s["text"] for s in l["spans"])
        text_lines.append(_norm(t))
        for s in l["spans"]:
            n = len(s["text"].strip())
            chars += n
            sizes[round(s["size"], 1)] += n
            f = s["font"]
            if f.startswith(("CMMI", "CMEX", "CMSY10", "CMR10", "CMMI10", "CMBX")) and not f.startswith("CMSY7"):
                if f.startswith("CMBX"):
                    continue
                math += n
    size = sizes.most_common(1)[0][0] if sizes else 0
    return {"bbox": tuple(b["bbox"]), "size": size, "lines": [t for t in text_lines if t], "chars": chars,
            "mathfrac": (math / chars) if chars else 0}


def _join_lines(lines, vocab):
    out = ""
    for ln in lines:
        if not out:
            out = ln
        elif out.endswith("-") and ln[:1].islower() and len(out) > 2 and out[-2].isalpha():
            left = re.search(r"([A-Za-z]+)-$", out)
            right = re.match(r"([A-Za-z]+)", ln)
            if left and right:
                joined = (left.group(1) + right.group(1)).lower()
                hyph = (left.group(1) + "-" + right.group(1)).lower()
                if hyph in vocab and joined not in vocab:
                    out += ln
                else:
                    out = out[:-1] + ln
            else:
                out += ln
        else:
            out += " " + ln
    return out


def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _extract(pdf_path):
    """Retourne une liste d'éléments ordonnés : dict(kind, text/png, caption)."""
    fitz = _fitz()
    doc = fitz.open(str(pdf_path))
    # vocabulaire (mots rencontrés à l'intérieur des lignes) pour décider des traits d'union de fin de ligne
    vocab = set()
    pages_blocks = []
    for page in doc:
        d = page.get_text("dict")
        blocks = [b for b in d["blocks"] if b["type"] == 0 and b["lines"]]
        pages_blocks.append((d, blocks))
        for b in blocks:
            for l in b["lines"]:
                for w in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", "".join(s["text"] for s in l["spans"])[:-1]):
                    vocab.add(w.lower())

    items, in_refs = [], False
    for pno, page in enumerate(doc):
        pw, ph = page.rect.width, page.rect.height
        mid = pw / 2
        d, blocks = pages_blocks[pno]
        infos = [_block_info(b) for b in blocks]
        roman_heads = [i["bbox"] for i in infos if re.match(rf"^{ROMAN}\.\s+[A-Z]", _smallcaps(_norm(" ".join(i["lines"])))) and i["chars"] < 90]
        WHITE = {"ACKNOWLEDGMENTS", "ACKNOWLEDGMENT", "REFERENCES", "AUTHOR BIOGRAPHY", "APPENDIX"}
        members = []     # éléments pouvant former une figure : (bbox, kind, info)
        texts = []
        for inf in infos:
            t = _norm(" ".join(inf["lines"]))
            if not t:
                continue
            if not in_refs and re.fullmatch(r"R\s?EFERENCES", _smallcaps(t).replace(" ", "")) or t.upper().replace(" ", "") == "REFERENCES":
                in_refs = True
            is_eq = (not in_refs) and ((inf["mathfrac"] >= 0.5 and inf["chars"] < 90) or
                                        (re.fullmatch(r"\.?\s*\(\d+\)", t) is not None))
            sc0 = _smallcaps(t)
            is_heading = (re.match(rf"^{ROMAN}\.\s+[A-Z]", sc0) is not None and len(t) < 90) or \
                         (sc0.isupper() and 6 <= len(sc0) < 40 and len(inf["lines"]) == 1 and
                          (sc0 in WHITE or any(0 <= inf["bbox"][1] - h[3] < 16 and inf["bbox"][0] < h[2] and h[0] < inf["bbox"][2] for h in roman_heads)))
            is_algo = (not in_refs) and re.match(r"^(Algorithm\s+\d+\s+[A-Z]|Require:|\d+:\s)", t) is not None
            small_max = 9.1 if pno > 0 else 8.6
            is_small = (not in_refs) and (0 < inf["size"] <= small_max) and not is_heading
            is_small = is_small or is_algo
            if is_eq or is_small:
                members.append({"bbox": inf["bbox"], "kind": "eq" if is_eq else "small", "text": t})
            else:
                lines = inf["lines"]
                cut = next((k for k in range(1, len(lines)) if re.match(r"^[A-H]\.\s+\S", lines[k])), None)
                if cut and re.match(rf"^{ROMAN}\.\s+[A-Z]", _smallcaps(lines[0])):
                    x0, y0, x1, y1 = inf["bbox"]
                    ym = y0 + (y1 - y0) * cut / len(lines)
                    texts.append({**inf, "lines": lines[:cut], "bbox": (x0, y0, x1, ym), "text": " ".join(lines[:cut])})
                    texts.append({**inf, "lines": lines[cut:], "bbox": (x0, ym, x1, y1), "text": " ".join(lines[cut:])})
                else:
                    texts.append({**inf, "text": t})
        if not in_refs:
            for b in d["blocks"]:
                if b["type"] == 1:
                    members.append({"bbox": tuple(b["bbox"]), "kind": "img", "text": ""})
            try:
                for dr in page.get_drawings():
                    r = dr["rect"]
                    if r.width * r.height > 0.4 * pw * ph or (r.width > 0.9 * pw):
                        continue
                    members.append({"bbox": (r.x0, r.y0, r.x1, r.y1), "kind": "draw", "text": ""})
            except Exception:
                pass

        def col(bb):
            w = bb[2] - bb[0]
            return "F" if w > 0.55 * pw else ("L" if (bb[0] + bb[2]) / 2 < mid else "R")

        # regroupement des membres en clusters (figures / tableaux / équations / algorithmes)
        n, parent = len(members), list(range(len(members)))
        pad = 14
        for i in range(n):
            for j in range(i + 1, n):
                a, b = members[i]["bbox"], members[j]["bbox"]
                both_eq = members[i]["kind"] == "eq" and members[j]["kind"] == "eq"
                px, py = (60, 4) if both_eq else (pad, pad)
                if a[0] - px < b[2] and b[0] - px < a[2] and a[1] - py < b[3] and b[1] - py < a[3]:
                    ci, cj = col(a), col(b)
                    if ci == cj or "F" in (ci, cj):
                        parent[_find(parent, i)] = _find(parent, j)
        clusters = {}
        for i in range(n):
            clusters.setdefault(_find(parent, i), []).append(members[i])
        page_items = []
        cl_list = []
        for cl in clusters.values():
            kinds = {m["kind"] for m in cl}
            if kinds <= {"draw"}:
                continue
            x0 = min(m["bbox"][0] for m in cl); y0 = min(m["bbox"][1] for m in cl)
            x1 = max(m["bbox"][2] for m in cl); y1 = max(m["bbox"][3] for m in cl)
            if (x1 - x0) < 8 or (y1 - y0) < 6:
                continue
            cap = ""
            for m in cl:
                if re.match(r"^(Fig\.|TABLE|Algorithm)\s", m["text"]):
                    mm = re.match(r"^TABLE\s+([IVX]+)\s+(.*)$", m["text"])
                    cap = f"TABLE {mm.group(1)} — {_smallcaps(mm.group(2))}" if mm else re.split(r"\sRequire:", m["text"])[0]
                    break
            cl_list.append({"bbox": [x0, y0, x1, y1], "cap": cap, "only_eq": all(m["kind"] in ("eq", "draw") for m in cl),
                            "col": col((x0, y0, x1, y1))})
        # une figure découpée en plusieurs morceaux : on rattache le morceau sans légende à celui qui porte « Fig. »
        changed = True
        while changed:
            changed = False
            cl_list.sort(key=lambda c: c["bbox"][1])
            for i, c in enumerate(cl_list):
                for j, o in enumerate(cl_list):
                    if i == j or c["only_eq"] or o["only_eq"] or c["col"] != o["col"]:
                        continue
                    gap = c["bbox"][1] - o["bbox"][3]
                    if 0 <= gap < 32 and not o["cap"] and c["cap"].startswith("Fig."):
                        c["bbox"] = [min(c["bbox"][0], o["bbox"][0]), o["bbox"][1], max(c["bbox"][2], o["bbox"][2]), c["bbox"][3]]
                        c["col"] = col(tuple(c["bbox"]))
                        cl_list.pop(j); changed = True; break
                if changed:
                    break
        for c in cl_list:
            x0, y0, x1, y1 = c["bbox"]
            clip = fitz.Rect(max(x0 - 3, 0), max(y0 - 3, 0), min(x1 + 3, pw), min(y1 + 3, ph))
            png = page.get_pixmap(matrix=fitz.Matrix(2.6, 2.6), clip=clip).tobytes("png")
            page_items.append({"kind": "eq" if c["only_eq"] else "fig", "png": png, "caption": c["cap"],
                               "bbox": (x0, y0, x1, y1), "col": c["col"], "wpt": x1 - x0})
        def inside_cluster(bb):
            cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
            return any(c["bbox"][0] - 2 <= cx <= c["bbox"][2] + 2 and c["bbox"][1] - 2 <= cy <= c["bbox"][3] + 2 for c in cl_list)
        texts = [t for t in texts if not inside_cluster(t["bbox"])]
        for t in texts:
            page_items.append({"kind": "text", "info": t, "bbox": t["bbox"], "col": col(t["bbox"])})

        # ordre de lecture : en-tête de la page 1, puis bandes délimitées par les éléments pleine largeur
        header = [it for it in page_items if pno == 0 and it["bbox"][1] < 262]
        rest = [it for it in page_items if it not in header]
        header.sort(key=lambda it: it["bbox"][1])
        fulls = sorted([it for it in rest if it["col"] == "F"], key=lambda it: it["bbox"][1])
        cols = [it for it in rest if it["col"] != "F"]
        ordered, top = list(header), -1
        for fw in fulls + [None]:
            bottom = fw["bbox"][1] if fw else 1e9
            band = [it for it in cols if top <= it["bbox"][1] < bottom or (fw is None and it["bbox"][1] >= top)]
            band = [it for it in band if not (fw is None and it["bbox"][1] < top)]
            for c in ("L", "R"):
                ordered += sorted([it for it in band if it["col"] == c], key=lambda it: it["bbox"][1])
            if fw:
                ordered.append(fw)
                top = fw["bbox"][3]
        for it in ordered:
            it["page"] = pno
        items += ordered

    # ---- mise en forme des éléments
    out = []
    last_h1_y, last_h1_page = -99, -1
    for it in items:
        if it["kind"] in ("fig", "eq"):
            out.append({"kind": it["kind"], "png": it["png"], "caption": it["caption"], "wpt": it["wpt"], "full": it["col"] == "F"})
            continue
        inf = it["info"]
        txt = _join_lines(inf["lines"], vocab)
        first_y = it["bbox"][1]
        if it["page"] == 0 and first_y < 262:
            if inf["size"] > 20:
                if out and out[-1]["kind"] == "title":
                    out[-1]["text"] += " " + txt
                else:
                    out.append({"kind": "title", "text": txt})
            else:
                out.append({"kind": "author", "text": txt})
            continue
        sc = _smallcaps(txt)
        letters = [c for c in sc.split(".", 1)[-1] if c.isalpha()]
        up_ratio = (sum(c.isupper() for c in letters) / len(letters)) if letters else 0
        if re.match(rf"^{ROMAN}\.\s+[A-Z]", sc) and len(sc) < 90 and up_ratio > 0.8:
            out.append({"kind": "h1", "text": sc}); last_h1_y, last_h1_page = it["bbox"][3], it["page"]; continue
        if sc.isupper() and 3 < len(sc) < 40 and not any(ch.isdigit() for ch in sc):
            if out and out[-1]["kind"] == "h1" and not out[-1].get("closed") and it["bbox"][1] - last_h1_y < 16 and it["page"] == last_h1_page:
                out[-1]["text"] += " " + sc; out[-1]["closed"] = True; continue
            out.append({"kind": "h1", "text": sc}); last_h1_y, last_h1_page = it["bbox"][3], it["page"]; continue
        if re.match(r"^[A-H]\.\s+\S", txt) and len(txt) < 90 and not txt.endswith("."):
            out.append({"kind": "h2", "text": txt}); continue
        if txt.startswith("•"):
            out.append({"kind": "li", "text": txt.lstrip("• ").strip()}); continue
        if re.match(r"^\[\d+\]", txt):
            out.append({"kind": "ref", "text": txt}); continue
        if re.match(r"^\d+\)\s", txt) and it["bbox"][0] - (49 if it["col"] == "L" else 312) > 6:
            for piece in re.split(r"(?<=[.!?])\s+(?=\d\)\s)", txt):
                out.append({"kind": "ol", "text": piece})
            continue
        out.append({"kind": "p", "text": txt})

    # continuation de paragraphes coupés par une colonne ou une page
    merged = []
    for e in out:
        if e["kind"] == "p" and merged and merged[-1]["kind"] == "p" and e["text"][:1].islower() \
                and not re.search(r"[.!?:]$", merged[-1]["text"]):
            merged[-1]["text"] += " " + e["text"]
        else:
            merged.append(e)
    doc.close()
    return merged


# --------------------------------------------------------------------------- exports
def _split_label(text):
    m = re.match(r"^(Abstract\s?—|Index Terms\s?—|\d\)\s[^:.]{3,60}[:.])\s*(.*)$", text)
    return (m.group(1), m.group(2)) if m else ("", text)


def to_docx(pdf_path=PDF, out=EXPORT / "article.docx"):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, Inches, RGBColor
    from docx.oxml.ns import qn
    items = _extract(pdf_path)
    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Inches(0.9); sec.top_margin = sec.bottom_margin = Inches(0.9)
    st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    for name, size in (("Heading 1", 13), ("Heading 2", 11.5)):
        h = doc.styles[name]; h.font.name = "Times New Roman"; h.font.size = Pt(size); h.font.bold = True
        h.font.color.rgb = RGBColor(0, 0, 0)
        h.element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman"); h.element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    title = next((e["text"] for e in items if e["kind"] == "title"), "Article")
    doc.core_properties.title = title
    authors = [e["text"] for e in items if e["kind"] == "author"]
    doc.core_properties.author = authors[0] if authors else ""
    for e in items:
        k = e["kind"]
        if k == "title":
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(e["text"]); r.bold = True; r.font.size = Pt(18)
        elif k == "author":
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(e["text"]); r.font.size = Pt(11)
        elif k == "h1":
            doc.add_heading(e["text"], level=1)
        elif k == "h2":
            doc.add_heading(e["text"], level=2)
        elif k in ("p", "li", "ref", "ol"):
            style = "List Bullet" if k == "li" else None
            p = doc.add_paragraph(style=style)
            if k == "ol":
                p.paragraph_format.left_indent = Inches(0.3)
            if k == "ref":
                p.paragraph_format.left_indent = Inches(0.35); p.paragraph_format.first_line_indent = Inches(-0.35)
                p.add_run(e["text"]).font.size = Pt(9)
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                lab, rest = _split_label(e["text"])
                if lab:
                    rr = p.add_run(lab + " "); rr.bold = True; rr.italic = lab.startswith(("Abstract", "Index"))
                p.add_run(rest)
        elif k in ("fig", "eq"):
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            w = min(6.5, max(0.6, e["wpt"] / 72 * (0.95 if e["full"] else 1.25)))
            p.add_run().add_picture(io.BytesIO(e["png"]), width=Inches(min(w, 6.5)))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    return out


def _plain(items, md):
    lines = []
    for e in items:
        k = e["kind"]
        if k == "title": lines.append(("# " if md else "") + e["text"] + "\n")
        elif k == "author": lines.append(e["text"] + ("  " if md else ""))
        elif k == "h1": lines.append("\n" + ("## " if md else "") + e["text"] + "\n")
        elif k == "h2": lines.append("\n" + ("### " if md else "") + e["text"] + "\n")
        elif k == "li": lines.append("- " + e["text"])
        elif k in ("p", "ref", "ol"): lines.append(("\n" if k != "ol" else "  ") + e["text"] + ("\n" if k == "p" else ""))
        elif k == "eq": lines.append("\n[équation — voir le PDF]\n")
        elif k == "fig":
            lines.append("\n[" + (e["caption"] or "Figure / tableau") + " — voir le PDF]\n")
    return "\n".join(lines).strip() + "\n"


def to_md(pdf_path=PDF, out=EXPORT / "article.md", items=None):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(_plain(items or _extract(pdf_path), True), encoding="utf-8"); return out


def to_txt(pdf_path=PDF, out=EXPORT / "article.txt", items=None):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(_plain(items or _extract(pdf_path), False), encoding="utf-8"); return out


def to_html(pdf_path=PDF, out=EXPORT / "article.html", items=None):
    import html as H
    items = items or _extract(pdf_path)
    body = []
    for e in items:
        k, t = e["kind"], H.escape(e.get("text", ""))
        if k == "title": body.append(f"<h1>{t}</h1>")
        elif k == "author": body.append(f'<p class="au">{t}</p>')
        elif k == "h1": body.append(f"<h2>{t}</h2>")
        elif k == "h2": body.append(f"<h3>{t}</h3>")
        elif k == "li": body.append(f"<ul><li>{t}</li></ul>")
        elif k == "ref": body.append(f'<p class="ref">{t}</p>')
        elif k in ("p", "ol"):
            lab, rest = _split_label(e["text"])
            body.append(f"<p{' style=\"margin-left:1.5em\"' if k == 'ol' else ''}>{('<b>' + H.escape(lab) + '</b> ') if lab else ''}{H.escape(rest)}</p>")
        elif k in ("fig", "eq"):
            b64 = base64.b64encode(e["png"]).decode()
            body.append(f'<p class="fig"><img src="data:image/png;base64,{b64}" style="max-width:{min(100, int(e["wpt"] / 5.2))}%"></p>')
    css = ("body{font-family:'Times New Roman',serif;max-width:860px;margin:40px auto;padding:0 18px;line-height:1.45;color:#111}"
           "h1{text-align:center;font-size:1.7em}.au{text-align:center;margin:0}p{text-align:justify}.ref{font-size:.85em;text-indent:-2em;margin-left:2em}"
           ".fig{text-align:center}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(f"<!doctype html><html lang='en'><meta charset='utf-8'><title>Article</title><style>{css}</style><body>{''.join(body)}</body></html>",
                         encoding="utf-8")
    return out


def build_all(pdf_path=PDF):
    """Régénère tous les exports (docx, html, md, txt) à partir du PDF."""
    items = _extract(pdf_path)
    to_md(pdf_path, items=items); to_txt(pdf_path, items=items); to_html(pdf_path, items=items)
    to_docx(pdf_path)
    return {k: EXPORT / f"article.{k}" for k in FORMATS}


def ensure_exports(pdf_path=PDF):
    """Retourne {format: chemin} ; régénère si absent ou plus ancien que le PDF."""
    paths = {k: EXPORT / f"article.{k}" for k in FORMATS}
    stale = any((not p.exists()) or p.stat().st_mtime < Path(pdf_path).stat().st_mtime for p in paths.values())
    if stale and available():
        build_all(pdf_path)
    return {k: p for k, p in paths.items() if p.exists()}
