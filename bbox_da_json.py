"""Ricostruisce le bounding box dei campi di un JSON su un PDF con PP-OCRv6.

Per ogni foglia del JSON (valore scalare, anche dentro liste e tabelle) cerca
sulla pagina il tratto di testo più simile e ne restituisce la box. La ricerca
è un allineamento locale su parole (Smith-Waterman): funziona anche per frasi
lunghe, che vanno a capo o che il modello ha leggermente riscritto.

Scrive in output:
  - <nome>_bbox.json   : stesso albero del JSON di input, ogni foglia diventa
                         {page, bbox_page [ymin, xmin, ymax, xmax] in punti PDF,
                          score, testo_trovato, ms}
  - <nome>_campi.pdf   : il PDF con le box in verde (ok) o arancio (dubbie)
  - <nome>_tempi.json  : tempi di ogni fase

Uso: python bbox_da_json.py file.pdf campi.json [cartella_output]
        [--tier medium|small|tiny] [--dpi 300] [--det-max 1600] [--threads 4] [--riusa-ocr]
"""

import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

def _threads_da_argv(default: int = 4) -> int:
    """Legge --threads N / --threads=N prima di argparse (vedi sotto)."""
    for i, a in enumerate(sys.argv):
        if a.startswith("--threads="):
            v = a.split("=", 1)[1]
        elif a == "--threads" and i + 1 < len(sys.argv):
            v = sys.argv[i + 1]
        else:
            continue
        try:
            return int(v)
        except ValueError:
            return default  # argparse darà l'errore vero più avanti
    return default


THREADS = _threads_da_argv()
# I limiti sui thread vanno messi PRIMA di importare paddle e numpy:
# le librerie li leggono all'avvio e dopo non si cambiano più.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = str(THREADS)

import numpy as np  # noqa: E402
import pymupdf  # noqa: E402
from rapidfuzz import fuzz, process  # noqa: E402

# ── Parametri dell'allineamento ──────────────────────────────────────
SIM_MIN = 0.72        # sotto questa somiglianza due parole non si accoppiano
                      # (un refuso OCR su una parola di 6+ lettere resta sopra 0.8;
                      # parole diverse come "impegna"/"impresa" stanno a 0.62)
GAP_PAGINA = 0.15     # costo per saltare una parola della pagina sulla stessa riga
GAP_RIGA = 1.00       # costo per saltare la prima parola di un'altra riga: scoraggia
                      # i "salti" che pescano una parola comoda dalla riga sopra
GAP_QUERY = 0.60      # costo per saltare una parola del valore (il modello ha riassunto)
SCORE_OK = 0.80       # punteggio medio per parola: sopra = verde
SCORE_MIN = 0.55      # sotto = campo non trovato
TOLLERANZA_ANCORA = 0.15  # con un'ancora di riga, un'occorrenza sulla riga giusta vince anche se
                          # l'OCR l'ha letta un po' peggio di una copia su un'altra riga
RAGGIO_RIGA = 10.0        # punti: entro questa distanza verticale visiva si è "sulla stessa riga"

_PUNCT = " .,;:!?()[]{}\"'«»€$%/\\-_"
_NUM_RE = re.compile(r"^[\d.,]+$")
_DATE_DMY = re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$")
_DATE_YMD = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")


def _senza_accenti(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def canonico(tok: str) -> str:
    """Forma confrontabile di una parola: minuscolo, senza accenti né
    punteggiatura di contorno, decimali con il punto, date come gg.mm.aaaa."""
    t = _senza_accenti(tok.strip().lower()).replace("’", "'").replace("`", "'")
    t = t.replace("€", "eur").strip(_PUNCT)
    if not t:
        return ""
    m = _DATE_YMD.match(t)
    if m:
        return f"{int(m.group(3)):02d}.{int(m.group(2)):02d}.{m.group(1)}"
    m = _DATE_DMY.match(t)
    if m:
        return f"{int(m.group(1)):02d}.{int(m.group(2)):02d}.{m.group(3)}"
    if _NUM_RE.match(t):
        if "," in t:
            t = t.replace(".", "").replace(",", ".")  # 1.200,50 -> 1200.50
        elif re.fullmatch(r"\d{1,3}(\.\d{3})+", t):
            t = t.replace(".", "")  # 1.200 (migliaia all'italiana) -> 1200
        try:
            f = float(t)
            return f"{f:.4f}".rstrip("0").rstrip(".")  # 500,00 == 500.0 == 500
        except ValueError:
            return t
    return t


def e_numerico(tok: str) -> bool:
    return bool(_NUM_RE.match(tok)) and any(ch.isdigit() for ch in tok)


_NUM_IN_TEXT_RE = re.compile(r"\d[\d.,]*")


def sotto_span(q: str, testo_pagina: str) -> tuple[int, int] | None:
    """Posizione (inizio, fine) del token canonico *q* dentro una parola di
    pagina più lunga, oppure None.

    Serve quando l'OCR incolla due parole ("500,00mensili",
    "Interventisemestrali", "3396417302/3280280411"): il valore cercato è
    dentro, e la box va ritagliata su quella porzione.
    """
    raw = _senza_accenti(testo_pagina.lower()).replace("’", "'")
    if q == "eur":
        pos = raw.find("€")  # "500,00€" o "€500,00"
        return (pos, pos + 1) if pos >= 0 else None
    if e_numerico(q):
        for m in _NUM_IN_TEXT_RE.finditer(raw):
            prima = raw[m.start() - 1] if m.start() > 0 else ""
            dopo = raw[m.end()] if m.end() < len(raw) else ""
            if prima.isdigit() or dopo.isdigit():
                continue
            # "12" dentro "06/12/2012" è un pezzo di data, non un valore:
            # accanto a un separatore si accettano solo numeri lunghi
            # (i due telefoni "3396417302/3280280411")
            if (prima in "/-.:" and prima) or (dopo in "/-.:" and dopo):
                if sum(ch.isdigit() for ch in q) < 4:
                    continue
            if canonico(m.group()) == q:
                return m.start(), m.end()
        return None
    if len(q) < 4 or len(raw) <= len(q):
        return None  # "mi", "via": troppo corti per cercarli dentro altre parole

    def _ai_bordi(a: int, b: int) -> bool:
        # due parole incollate: il token sta all'inizio o alla fine, con un
        # carattere di tolleranza per il refuso finale ("semestrale" in
        # "interventisemestrali"); "sede" dentro "possedere" non conta
        return a <= 1 or b >= len(raw) - 1

    pos = raw.find(q)
    if pos >= 0 and _ai_bordi(pos, pos + len(q)):
        return pos, pos + len(q)
    # tolleranza agli errori OCR dentro la parola incollata ("semestrale" in "interventisemestrali")
    al = fuzz.partial_ratio_alignment(q, raw)
    if al is not None and al.score >= 85 and _ai_bordi(al.dest_start, al.dest_end):
        return al.dest_start, al.dest_end
    return None


# ── OCR ──────────────────────────────────────────────────────────────


class Parola:
    """Una parola OCR. x0..y1 sono nello spazio della pagina (quello in cui si
    disegna e si esporta); vx0..vy1 nello spazio visivo, cioè come la pagina
    si vede dopo la rotazione (/Rotate): è lì che le righe sono orizzontali,
    quindi ordine di lettura, ancore di riga e ritagli usano quelli."""

    __slots__ = ("testo", "canon", "x0", "y0", "x1", "y1", "vx0", "vy0", "vx1", "vy1", "riga", "derot")

    def __init__(self, testo: str, box, scala: float):
        self.testo = testo
        self.canon = canonico(testo)
        self.x0, self.y0, self.x1, self.y1 = (float(v) * scala for v in box)
        self.vx0, self.vy0, self.vx1, self.vy1 = self.x0, self.y0, self.x1, self.y1  # riassegnati da parole_pagina
        self.riga = 0  # indice della riga OCR di appartenenza (assegnato da parole_pagina)
        self.derot = None  # matrice visivo -> pagina, se la pagina è ruotata

    @property
    def y_vis(self) -> float:
        return (self.vy0 + self.vy1) / 2


def carica_ocr(tier: str, threads: int, det_max: int):
    from paddleocr import PaddleOCR

    return PaddleOCR(
        text_detection_model_name=f"PP-OCRv6_{tier}_det",
        text_recognition_model_name=f"PP-OCRv6_{tier}_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        return_word_box=True,  # box di ogni parola, non solo della riga
        # Il rilevamento delle righe lavora su una copia ridotta della pagina
        # (lato lungo <= det_max px): a 300 DPI la pagina intera è 2480x3508
        # e costa 4 volte tanto. Il riconoscimento dei ritagli resta a piena
        # risoluzione, quindi la qualità del testo non cambia.
        text_det_limit_type="max",
        text_det_limit_side_len=det_max,
        device="cpu",
        cpu_threads=threads,
    )


def rasterizza(page: pymupdf.Page, dpi: int) -> np.ndarray:
    """Immagine della pagina come array BGR (il formato che PaddleOCR legge),
    senza passare da un file."""
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def ocr_pagina(ocr, img: np.ndarray, page: pymupdf.Page, dpi: int) -> list[dict]:
    """Righe OCR della pagina: testo della riga più i frammenti con box
    (in punti PDF, nello spazio della pagina) da cui PaddleOCR l'ha composta."""
    res = ocr.predict(input=img)[0]
    scala = 72 / dpi
    # get_pixmap applica la rotazione della pagina (/Rotate): le box OCR
    # sono nello spazio ruotato e vanno riportate a quello della pagina,
    # che è quello in cui si disegna e si esporta.
    derot = page.derotation_matrix if page.rotation else None

    def _box(b):
        r = pymupdf.Rect(*(float(v) * scala for v in b))
        if derot is not None:
            r = (r * derot).normalize()
        return [r.x0, r.y0, r.x1, r.y1]

    return [
        {
            "testo": testo,
            "frammenti": [{"testo": w, "box": _box(b)} for w, b in zip(parole, boxes)],
        }
        for testo, parole, boxes in zip(res["rec_texts"], res["text_word"], res["text_word_boxes"])
    ]


def parole_da_riga(riga: dict) -> list[Parola]:
    """Ricompone le parole vere di una riga a partire dai frammenti.

    Con return_word_box PaddleOCR spezza la riga a ogni segno di
    punteggiatura ("06/12/2012" -> "06", "/", "12", "/", "2012") e a volte
    incolla due parole ("500,00 mensili" -> "500", ",", "00mensili").
    Il testo della riga (rec_texts) ha invece gli spazi giusti: le parole
    si prendono da lì e la box di ciascuna si ricava dai frammenti che ne
    coprono i caratteri, interpolando in X dentro il frammento se la
    parola ne copre solo una parte.
    """
    fr = [f for f in riga["frammenti"] if f["testo"].replace(" ", "")]
    # mappa: posizione del carattere (senza spazi) -> (frammento, indice nel frammento)
    mappa = []
    for fi, f in enumerate(fr):
        L = len(f["testo"].replace(" ", ""))
        mappa.extend((fi, ci, L) for ci in range(L))
    tokens = riga["testo"].split()
    if sum(len(t) for t in tokens) != len(mappa):
        # la riga e i frammenti non combaciano: si tengono i frammenti,
        # reincollando quelli attaccati in X ("06","/","12" -> "06/12")
        out = []
        for f in fr:
            x0, y0, x1, y1 = f["box"]
            if out and x0 <= out[-1].x1 + 0.5:
                prev = out[-1]
                out[-1] = Parola(prev.testo + f["testo"].strip(), [prev.x0, min(prev.y0, y0), x1, max(prev.y1, y1)], 1.0)
            else:
                out.append(Parola(f["testo"].strip(), f["box"], 1.0))
        return out
    out = []
    pos = 0
    for tok in tokens:
        span = mappa[pos : pos + len(tok)]
        pos += len(tok)
        x0 = y0 = float("inf")
        x1 = y1 = float("-inf")
        for fi in sorted({s[0] for s in span}):
            fx0, fy0, fx1, fy1 = fr[fi]["box"]
            cis = [ci for f2, ci, _ in span if f2 == fi]
            L = span[0][2] if span[0][0] == fi else next(s[2] for s in span if s[0] == fi)
            w = fx1 - fx0
            x0 = min(x0, fx0 + w * min(cis) / L)
            x1 = max(x1, fx0 + w * (max(cis) + 1) / L)
            y0, y1 = min(y0, fy0), max(y1, fy1)
        out.append(Parola(tok, [x0, y0, x1, y1], 1.0))
    return out


def parole_pagina(righe: list[dict], page: pymupdf.Page | None = None) -> list[Parola]:
    """Parole della pagina in ordine di lettura (alto→basso, sinistra→destra)
    nello spazio visivo: con *page* ruotata (/Rotate) le coordinate di pagina
    hanno gli assi scambiati e senza questa correzione le "righe" sarebbero
    le colonne. Le colonne affiancate si intercalano, ma l'allineamento
    tollera i salti."""
    rot = page.rotation_matrix if page is not None and page.rotation else None
    derot = page.derotation_matrix if rot is not None else None
    blocchi = []
    for riga in righe:
        ps = parole_da_riga(riga)
        if not ps:
            continue
        if rot is not None:
            for p in ps:
                r = (pymupdf.Rect(p.x0, p.y0, p.x1, p.y1) * rot).normalize()
                p.vx0, p.vy0, p.vx1, p.vy1 = r.x0, r.y0, r.x1, r.y1
                p.derot = derot
        yc = sum(p.y_vis for p in ps) / len(ps)
        h = sum(p.vy1 - p.vy0 for p in ps) / len(ps)
        blocchi.append((yc, h, min(p.vx0 for p in ps), ps))
    # Righe OCR sulla stessa riga visiva (centri a meno di mezza altezza)
    # vanno nello stesso gruppo, ordinate da sinistra a destra.
    blocchi.sort(key=lambda r: r[0])
    gruppi: list[list] = []
    for b in blocchi:
        if gruppi and abs(b[0] - gruppi[-1][0][0]) < 0.5 * max(b[1], gruppi[-1][0][1]):
            gruppi[-1].append(b)
        else:
            gruppi.append([b])
    out = []
    k = 0
    for g in gruppi:
        for _, _, _, ps in sorted(g, key=lambda r: r[2]):
            for p in ps:
                p.riga = k
                out.append(p)
            k += 1
    return out


# ── Allineamento locale (Smith-Waterman su parole) ───────────────────


SIM_INCOLLATA = 0.95  # token trovato dentro una parola di pagina più lunga
MAX_CANDIDATI = 64    # allineamenti a pari punteggio restituiti: nelle tabelle lo stesso valore
                      # ("00002", "171") può ripetersi su decine di righe e l'ancora di riga
                      # deve poterle vedere tutte, altrimenti dalla nona in poi ripiega sull'ottava


def token_valore(valore: str) -> list[str]:
    """Token canonici del valore da cercare. Il simbolo di valuta accanto a
    un importo ("500,00 €") non si cerca: sulla pagina è spesso incollato
    al numero e un token mancante dimezzerebbe il punteggio."""
    q = [t for t in (canonico(t) for t in valore.split()) if t]
    if len(q) > 1:
        q = [t for t in q if t != "eur"]
    return q


def allinea(parole: list[Parola], valore: str) -> list[tuple[float, list[tuple[int, int]]]]:
    """Cerca il tratto di *parole* più simile a *valore*.

    Ritorna gli allineamenti col punteggio massimo (fino a MAX_CANDIDATI:
    un valore ripetuto in pagina ne produce uno per occorrenza), ciascuno
    come (punteggio medio per token del valore, coppie (token, parola)).
    """
    q = token_valore(valore)
    if not q or not parole:
        return []
    p = [w.canon for w in parole]
    sim = process.cdist(q, p, scorer=fuzz.ratio, workers=1) / 100.0
    # numeri e parole cortissime si accettano solo uguali: "30" non deve
    # entrare in "300" e "eur" non deve somigliare a "per"
    p_arr = np.array(p)
    for i, t in enumerate(q):
        if e_numerico(t) or len(t) <= 3:
            sim[i] = np.where(p_arr == t, 1.0, 0.0)
    # parole incollate dall'OCR: il token è dentro una parola più lunga
    for i, t in enumerate(q):
        for j in np.flatnonzero(sim[i] < SIM_INCOLLATA):
            if len(parole[j].testo) > len(t) and sotto_span(t, parole[j].testo) is not None:
                sim[i, j] = SIM_INCOLLATA
    sim = np.where(sim >= SIM_MIN, sim, -1.0)
    n, m = len(q), len(p)
    # costo per saltare la parola di pagina j: più caro se apre una riga nuova
    gap = [GAP_PAGINA] * m
    for j in range(1, m):
        if parole[j].riga != parole[j - 1].riga:
            gap[j] = GAP_RIGA
    H = np.zeros((n + 1, m + 1))
    tb = np.zeros((n + 1, m + 1), dtype=np.int8)  # 1 diag, 2 salta pagina, 3 salta query
    for i in range(1, n + 1):
        hi, hp, si = H[i], H[i - 1], sim[i - 1]
        for j in range(1, m + 1):
            d = hp[j - 1] + si[j - 1]
            u = hi[j - 1] - gap[j - 1]
            l = hp[j] - GAP_QUERY
            best = d if d >= u and d >= l else (u if u >= l else l)
            if best <= 0:
                continue
            hi[j] = best
            tb[i, j] = 1 if best == d else (2 if best == u else 3)
    best = float(H.max())
    if best <= 0:
        return []
    # tutte le celle al punteggio massimo (a meno di arrotondamento):
    # un valore ripetuto in pagina finisce qui una volta per occorrenza
    celle = np.argwhere(H >= best - 1e-9)
    esiti = []
    visti: set[frozenset] = set()
    for i, j in celle[:MAX_CANDIDATI * 4]:
        i, j = int(i), int(j)
        coppie = []
        while i > 0 and j > 0 and tb[i, j]:
            if tb[i, j] == 1:
                coppie.append((i - 1, j - 1))
                i, j = i - 1, j - 1
            elif tb[i, j] == 2:
                j -= 1
            else:
                i -= 1
        chiave = frozenset(j for _, j in coppie)
        if not coppie or chiave in visti:
            continue
        visti.add(chiave)
        esiti.append((best / n, sorted(coppie, key=lambda c: c[1])))
        if len(esiti) >= MAX_CANDIDATI:
            break
    return esiti


def box_coppia(q_tok: str, w: Parola) -> tuple[float, float, float, float]:
    """Box (spazio pagina) della parola abbinata; se il token è solo una
    porzione della parola (OCR incollato) si ritaglia lungo la direzione del
    testo, cioè la X visiva, in proporzione ai caratteri."""
    if w.canon == q_tok:
        return w.x0, w.y0, w.x1, w.y1
    span = sotto_span(q_tok, w.testo)
    if span is None:
        return w.x0, w.y0, w.x1, w.y1
    n = max(len(w.testo), 1)
    larg = w.vx1 - w.vx0
    vis = pymupdf.Rect(w.vx0 + larg * span[0] / n, w.vy0, w.vx0 + larg * span[1] / n, w.vy1)
    if w.derot is not None:
        vis = (vis * w.derot).normalize()
    return vis.x0, vis.y0, vis.x1, vis.y1


def cerca(pagine: list[list[Parola]], valore: str, ancora: tuple[int, float] | None = None,
          solo_pagina: bool = False) -> dict:
    """Prova ogni pagina e tiene l'allineamento migliore.

    *ancora* = (pagina, y visiva) della riga a cui il valore appartiene:
    a pari punteggio vince l'occorrenza più vicina (valori ripetuti nelle
    tabelle: "giornaliera" su più righe). Senza ancora vince la prima.
    L'esito riporta anche *y_vis*, il centro verticale visivo del testo
    trovato, da usare come ancora per gli altri campi della stessa riga.
    """
    t0 = time.perf_counter()
    q = token_valore(valore)
    candidati = []
    for n, parole in enumerate(pagine, start=1):
        for score, coppie in allinea(parole, valore):
            boxes = [box_coppia(q[i], parole[j]) for i, j in coppie]
            bbox = [
                round(min(b[1] for b in boxes), 1),
                round(min(b[0] for b in boxes), 1),
                round(max(b[3] for b in boxes), 1),
                round(max(b[2] for b in boxes), 1),
            ]
            y_vis = sum(parole[j].y_vis for _, j in coppie) / len(coppie)
            candidati.append((score, n, bbox, coppie, parole, y_vis))
    best_score = max((c[0] for c in candidati), default=0.0)
    migliori = [c for c in candidati if c[0] >= best_score - 1e-9]
    if ancora is not None and len(candidati) > 1:
        # Valori ripetuti in tabella: tra le occorrenze con punteggio vicino al
        # massimo vince quella sulla riga dell'ancora (stessa pagina, entro
        # RAGGIO_RIGA); a parità di riga il punteggio; poi la distanza.
        # (con solo_pagina basta che stia sulla pagina dell'ancora: è il caso
        # dei PDF con più documenti dall'intestazione uguale)
        pg, yc = ancora
        vicini = [c for c in candidati if c[0] >= best_score - TOLLERANZA_ANCORA]
        raggio = float("inf") if solo_pagina else RAGGIO_RIGA
        vicini.sort(key=lambda c: (c[1] != pg, abs(c[5] - yc) > raggio, -c[0], abs(c[5] - yc)))
        if vicini[0][1] == pg and abs(vicini[0][5] - yc) <= raggio:
            migliori = [vicini[0]] + [c for c in migliori if c is not vicini[0]]
        else:
            migliori.sort(key=lambda c: (c[1] != pg, abs(c[5] - yc)))
    if not migliori or best_score < SCORE_MIN:
        esito = {"page": None, "bbox_page": None, "y_vis": None, "score": round(best_score, 3), "copertura": 0.0,
                 "testo_trovato": None, "candidati": len(migliori)}
    else:
        score, n, bbox, coppie, parole, y_vis = migliori[0]
        esito = {
            "page": n,
            "bbox_page": bbox,
            "y_vis": round(y_vis, 1),
            "score": round(score, 3),
            # quota di token del valore effettivamente abbinati: l'allineamento
            # è locale, una frase trovata solo a metà ha score e copertura bassi
            "copertura": round(len(coppie) / len(q), 3),
            "testo_trovato": " ".join(parole[j].testo for _, j in coppie),
            "candidati": len(migliori),  # >1 = valore ripetuto in pagina
        }
    esito["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return esito


# ── Albero del JSON ──────────────────────────────────────────────────


def _foglia_vuota() -> dict:
    return {"page": None, "bbox_page": None, "y_vis": None, "score": 0.0, "copertura": 0.0, "testo_trovato": None, "candidati": 0, "ms": 0.0}


def righe_del_json(nodo, percorso: str = "", doc: str | None = None, righe: dict | None = None) -> dict[str, tuple[str, int, str]]:
    """{percorso della riga: (tipo, numero, documento)} per ogni dizionario dentro una lista.

    Tipo "dettaglio" se la riga contiene solo scalari (le righe di tabella),
    "testata" se ha contenitori annidati (es. un articolo con le sue pezze).
    La numerazione è consecutiva per tipo dentro ogni elemento della lista
    più esterna (il "documento"), così la riga di dettaglio N è la N-esima
    riga della tabella di quel documento.
    """
    if righe is None:
        righe = {}
    if isinstance(nodo, dict):
        for k, v in nodo.items():
            righe_del_json(v, f"{percorso}.{k}" if percorso else str(k), doc, righe)
    elif isinstance(nodo, list):
        for i, v in enumerate(nodo):
            p = f"{percorso}[{i}]"
            if isinstance(v, dict):
                annidati = any(isinstance(x, (dict, list)) for x in v.values())
                if doc is None and annidati:
                    d = p  # elemento della lista più esterna con struttura: è un documento, non una riga
                else:
                    d = doc if doc is not None else ""
                    tipo = "testata" if annidati else "dettaglio"
                    n = 1 + sum(1 for t, _, dd in righe.values() if t == tipo and dd == d)
                    righe[p] = (tipo, n, d)
                righe_del_json(v, p, d, righe)
            else:
                righe_del_json(v, p, doc if doc is not None else p, righe)
    return righe


def etichetta_riga(percorso: str, righe: dict) -> str | None:
    """Etichetta di riga per un campo: "r7" per la settima riga di dettaglio,
    "A2" per la seconda riga articolo, None se il campo non sta in una riga."""
    genitore = percorso.rpartition(".")[0]
    if genitore in righe:
        tipo, n, _ = righe[genitore]
        return f"r{n}" if tipo == "dettaglio" else f"A{n}"
    return None


def scrivi_etichetta(page: pymupdf.Page, bbox: list[float], testo: str, colore, fontsize: float = 6) -> None:
    """Scrive *testo* appena sopra a sinistra della box, dritto anche se la
    pagina è ruotata: la posizione si calcola nello spazio visivo e il testo
    si ruota in senso contrario alla pagina."""
    y0, x0, y1, x1 = bbox
    r = pymupdf.Rect(x0, y0, x1, y1)
    if page.rotation:
        v = (r * page.rotation_matrix).normalize()
        punto = pymupdf.Point(v.x0, max(v.y0 - 1.5, 6)) * page.derotation_matrix
        page.insert_text(punto, testo, fontsize=fontsize, color=colore, rotate=page.rotation)
    else:
        page.insert_text((x0, max(y0 - 1.5, 6)), testo, fontsize=fontsize, color=colore)


def etichette_di_riga(voci: list[tuple[str, int, list[float]]], righe: dict, doc: pymupdf.Document) -> list[tuple[int, list[float], str]]:
    """Da (percorso, pagina, bbox) delle foglie trovate sceglie, per ogni riga
    di tabella, una sola etichetta: quella della cella più a sinistra (in
    senso visivo). Ritorna (pagina, bbox, testo)."""
    per_riga: dict[tuple[str, int], tuple[float, list[float], str]] = {}
    for percorso, n, bbox in voci:
        testo = etichetta_riga(percorso, righe)
        if testo is None or bbox is None:
            continue
        page = doc[n - 1]
        y0, x0, y1, x1 = bbox
        r = pymupdf.Rect(x0, y0, x1, y1)
        if page.rotation:
            r = (r * page.rotation_matrix).normalize()
        chiave = (percorso.rpartition(".")[0], n)
        if chiave not in per_riga or r.x0 < per_riga[chiave][0]:
            per_riga[chiave] = (r.x0, bbox, testo)
    return [(n, bbox, testo) for (_, n), (_, bbox, testo) in per_riga.items()]


def visita_riga(riga: dict, pagine, percorso: str, esiti: list) -> dict:
    """Una riga di tabella (dict dentro una lista): prima i campi con una
    sola occorrenza in pagina, che fissano la posizione della riga; poi
    quelli ripetuti ("giornaliera" su più righe) scelti vicino a quella."""
    out = {}
    ambigui = []
    y_riga: list[tuple[int, float]] = []
    for k, v in riga.items():
        p = f"{percorso}.{k}" if percorso else str(k)
        if isinstance(v, (dict, list)) or v is None or str(v).strip() == "":
            out[k] = visita(v, pagine, p, esiti)
            continue
        e = cerca(pagine, str(v))
        if e["candidati"] > 1:
            ambigui.append((k, v, p))
            continue
        out[k] = e
        esiti.append((p, str(v), e))
        if e["page"] is not None:
            y_riga.append((e["page"], e["y_vis"]))
    ancora = None
    if y_riga:
        pg = max(set(pg for pg, _ in y_riga), key=[pg for pg, _ in y_riga].count)
        ys = sorted(y for p_, y in y_riga if p_ == pg)
        ancora = (pg, ys[len(ys) // 2])
    for k, v, p in ambigui:
        e = cerca(pagine, str(v), ancora=ancora)
        out[k] = e
        esiti.append((p, str(v), e))
    return {k: out[k] for k in riga}  # stesso ordine delle chiavi di input


def visita(nodo, pagine, percorso: str, esiti: list) -> object:
    """Ricostruisce l'albero sostituendo ogni foglia con il suo esito.
    Ogni dizionario passa da visita_riga: i suoi campi con una sola
    occorrenza fissano la posizione, quelli ripetuti (i totali che
    coincidono con l'ultima riga, "giornaliera" su più righe) si scelgono
    vicino a essa invece che alla prima occorrenza in pagina."""
    if isinstance(nodo, dict):
        return visita_riga(nodo, pagine, percorso, esiti)
    if isinstance(nodo, list):
        return [visita(v, pagine, f"{percorso}[{i}]", esiti) for i, v in enumerate(nodo)]
    if nodo is None or str(nodo).strip() == "":
        return _foglia_vuota()
    esito = cerca(pagine, str(nodo))
    esiti.append((percorso, str(nodo), esito))
    return esito


_ELEMENTO_LISTA = re.compile(r"\[\d+\]")


def riancora_per_documento(esiti: list, pagine: list) -> int:
    """Secondo passaggio per i PDF con più documenti dall'intestazione uguale.

    Un valore ripetuto (candidati > 1) può finire sulla copia meglio letta,
    anche se sta in un altro documento. Qui ogni elemento di lista (il DDT,
    l'articolo, la riga) prende come pagina quella dove stanno i suoi valori
    univoci; i valori ripetuti finiti altrove vengono ricercati con quell'ancora.
    Modifica gli esiti sul posto; ritorna quanti ne ha spostati.
    """
    univoci: dict[str, list[tuple[int, float]]] = {}
    for percorso, _, e in esiti:
        if e["page"] is None or e["candidati"] != 1:
            continue
        for m in _ELEMENTO_LISTA.finditer(percorso):
            univoci.setdefault(percorso[: m.end()], []).append((e["page"], e["y_vis"]))
    spostati = 0
    for percorso, valore, e in esiti:
        if e["page"] is None:
            continue
        antenati = [percorso[: m.end()] for m in _ELEMENTO_LISTA.finditer(percorso)]
        for antenato in reversed(antenati):  # dal più vicino
            dati = univoci.get(antenato)
            if not dati:
                continue
            pg = max(set(p for p, _ in dati), key=[p for p, _ in dati].count)
            if e["page"] == pg:
                break
            # sta su un'altra pagina: si accetta la copia sulla pagina del documento
            # solo se l'OCR l'ha letta quasi altrettanto bene (TOLLERANZA_ANCORA)
            ys = sorted(y for p, y in dati if p == pg)
            nuovo = cerca(pagine, valore, ancora=(pg, ys[len(ys) // 2]), solo_pagina=True)
            if nuovo["page"] == pg:
                nuovo["ms"] = round(e["ms"] + nuovo["ms"], 1)
                e.update(nuovo)
                spostati += 1
            break
    return spostati


def disegna(doc: pymupdf.Document, esiti: list, righe: dict | None = None) -> None:
    """Box verdi (ok) o arancio (dubbie). Niente nomi di campo: l'unica
    scritta è il numero di riga, una volta per riga di tabella, sulla cella
    più a sinistra (righe = righe_del_json(campi))."""
    for percorso, _valore, e in esiti:
        if e["page"] is None:
            continue
        page = doc[e["page"] - 1]
        y0, x0, y1, x1 = e["bbox_page"]
        colore = (0, 0.6, 0) if e["score"] >= SCORE_OK else (1, 0.5, 0)
        shape = page.new_shape()
        shape.draw_rect(pymupdf.Rect(x0 - 1, y0 - 1, x1 + 1, y1 + 1))
        shape.finish(color=colore, width=0.8)
        shape.commit()
    if righe:
        voci = [(p, e["page"], e["bbox_page"]) for p, _, e in esiti if e["page"] is not None]
        for n, bbox, testo in etichette_di_riga(voci, righe, doc):
            scrivi_etichetta(doc[n - 1], bbox, testo, (0, 0.4, 0))


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("json_campi")
    ap.add_argument("out_dir", nargs="?", default=None)
    ap.add_argument("--tier", default="medium", choices=["medium", "small", "tiny"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--det-max", type=int, default=1600, help="lato massimo (px) dell'immagine usata per rilevare le righe")
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--riusa-ocr", action="store_true", help="ricarica le parole OCR salvate da una corsa precedente")
    args = ap.parse_args()

    pdf = Path(args.pdf)
    out_dir = Path(args.out_dir) if args.out_dir else pdf.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    campi = json.loads(Path(args.json_campi).read_text(encoding="utf-8"))
    tempi = {"threads": args.threads, "tier": args.tier, "dpi": args.dpi, "det_max": args.det_max, "pagine": []}
    t_start = time.perf_counter()

    doc = pymupdf.open(pdf)
    ocr_json = out_dir / f"{pdf.stem}_ocr.json"
    if args.riusa_ocr and ocr_json.exists():
        # L'OCR è la fase lenta: per iterare sul JSON dei campi si riusa
        # quello salvato dalla corsa precedente.
        righe_pagine = json.loads(ocr_json.read_text(encoding="utf-8"))
        tempi["caricamento_modello_s"] = 0.0
        tempi["pagine"] = [{"pagina": n, "render_s": 0.0, "ocr_s": 0.0} for n in range(1, len(righe_pagine) + 1)]
        print(f"OCR riusato da {ocr_json.name}")
    else:
        t0 = time.perf_counter()
        ocr = carica_ocr(args.tier, args.threads, args.det_max)
        tempi["caricamento_modello_s"] = round(time.perf_counter() - t0, 2)
        righe_pagine = []
        for n, page in enumerate(doc, start=1):
            t0 = time.perf_counter()
            img = rasterizza(page, args.dpi)
            t1 = time.perf_counter()
            righe_pagine.append(ocr_pagina(ocr, img, page, args.dpi))
            t2 = time.perf_counter()
            tempi["pagine"].append({"pagina": n, "render_s": round(t1 - t0, 2), "ocr_s": round(t2 - t1, 2)})
            print(f"pagina {n}: {len(righe_pagine[-1])} righe OCR, render {t1 - t0:.2f} s + OCR {t2 - t1:.2f} s")
        ocr_json.write_text(json.dumps(righe_pagine, ensure_ascii=False), encoding="utf-8")

    pagine = [parole_pagina(righe, page) for righe, page in zip(righe_pagine, doc)]
    for info, parole in zip(tempi["pagine"], pagine):
        info["parole"] = len(parole)

    t0 = time.perf_counter()
    esiti = []
    albero = visita(campi, pagine, "", esiti)
    tempi["riancorati"] = riancora_per_documento(esiti, pagine)
    tempi["matching_s"] = round(time.perf_counter() - t0, 3)
    tempi["campi"] = len(esiti)
    tempi["campi_trovati"] = sum(1 for _, _, e in esiti if e["page"] is not None)
    tempi["totale_s"] = round(time.perf_counter() - t_start, 2)

    for percorso, valore, e in esiti:
        stato = "OK " if e["page"] and e["score"] >= SCORE_OK else ("?? " if e["page"] else "-- ")
        print(f"{stato}{e['ms']:6.1f} ms  score={e['score']:.2f}  p{e['page']}  {percorso}: {valore[:60]!r}")

    disegna(doc, esiti, righe_del_json(campi))
    doc.save(out_dir / f"{pdf.stem}_campi.pdf")
    (out_dir / f"{pdf.stem}_bbox.json").write_text(json.dumps(albero, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{pdf.stem}_tempi.json").write_text(json.dumps(tempi, indent=2), encoding="utf-8")
    print(
        f"\nmodello {tempi['caricamento_modello_s']} s, render {sum(p['render_s'] for p in tempi['pagine']):.2f} s, "
        f"OCR {sum(p['ocr_s'] for p in tempi['pagine']):.2f} s, matching {tempi['matching_s']} s, "
        f"totale {tempi['totale_s']} s ({tempi['campi_trovati']}/{tempi['campi']} campi trovati, {args.threads} thread)"
    )
    print(f"Risultati in {out_dir}")


if __name__ == "__main__":
    main()
