"""Anonimizza i documenti di prova (PDF + JSON estratto) per poterli committare.

Per ogni documento della mappa:
  - trova sulle pagine ogni stringa sensibile (ricerca testuale sui PDF con
    testo, parole OCR in cache sulle scansioni, con tolleranza ai refusi),
    la cancella con una redazione bianca (testo e pixel) e ci scrive sopra il
    valore fittizio, dritto anche sulle pagine ruotate;
  - sbianca le zone indicate (loghi, firme, blocchi di intestazione), date
    come frazioni della pagina vista [ymin, xmin, ymax, xmax];
  - applica le stesse sostituzioni ai valori del JSON, mantenendo maiuscole
    e minuscole dell'originale.
Numeri di documento, date, righe e importi restano com'erano: i test
restano ripetibili sui file anonimi.

La mappa è un JSON (da NON committare: contiene i dati veri):
{
  "famiglie": {"nome": {"sostituzioni": [["originale", "fittizio"], ...],
                        "zone_bianche": [{"pagine": "tutte" | [1, 2], "vis": [ymin, xmin, ymax, xmax],
                                          "senza_testo": true}]}},
  "documenti": [{"nome": "ddt_esempio_1", "famiglia": "nome", "pdf": "...", "json": "...",
                 "ocr": "<nome>_ocr.json (solo scansioni)", "sostituzioni": [...], "zone_bianche": [...],
                 "rasterizza": 200}]
}
"rasterizza" (dpi, anche a livello di famiglia) rende ogni pagina come
immagine prima di anonimizzarla: obbligatorio per i PDF nativi.
"senza_testo" su una zona: le stringhe trovate lì dentro vengono cancellate
ma non riscritte (blocchi di intestazione con dati non usati dal JSON).

Uso: python anonimizza.py mappa.json cartella_uscita
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pymupdf
from rapidfuzz import fuzz

sys.argv, _argv = ["x"], sys.argv  # bbox_da_json legge --threads da argv all'import
import bbox_da_json as B  # noqa: E402

sys.argv = _argv

SOGLIA_OCR = 85  # somiglianza minima (0-100), senza spazi, tra la stringa cercata e le parole OCR:
                 # regge "C0SSAT0" letto con gli zeri, non "PROMANO" per "ROMA" grazie al vincolo di lunghezza
LARGHEZZA_COURIER = 0.62  # larghezza di un carattere Courier in frazione del corpo


# ── Ricerca delle occorrenze ─────────────────────────────────────────


def occorrenze_testo(page: pymupdf.Page, testo: str) -> list[pymupdf.Rect]:
    """PDF con testo: rettangoli di ogni occorrenza (ricerca senza distinzione
    di maiuscole; una stringa a cavallo di due righe dà più rettangoli)."""
    visti = []
    for variante in {testo, testo.upper(), testo.lower()}:
        for r in page.search_for(variante):
            if not any(abs(r.x0 - v.x0) < 1 and abs(r.y0 - v.y0) < 1 for v in visti):
                visti.append(r)
    return visti


def _ritaglio_parole(finestra: list[B.Parola], inizio: int, fine: int) -> pymupdf.Rect:
    """Box (spazio pagina) dei caratteri [inizio, fine) della finestra, contati
    sulle forme canoniche senza spazi: le parole incollate dall'OCR
    ("MOHAMEDVIA") si ritagliano in proporzione ai caratteri, lungo la X vista."""
    vis = None
    pos = 0
    for w in finestra:
        L = len(w.canon)
        a, b = max(inizio, pos), min(fine, pos + L)
        if b > a and L > 0:
            larg = w.vx1 - w.vx0
            r = pymupdf.Rect(w.vx0 + larg * (a - pos) / L, w.vy0, w.vx0 + larg * (b - pos) / L, w.vy1)
            vis = r if vis is None else vis | r
        pos += L
    if vis is None:
        return pymupdf.Rect()
    derot = finestra[0].derot
    return (vis * derot).normalize() if derot is not None else vis


def occorrenze_ocr(parole: list[B.Parola], testo: str) -> list[pymupdf.Rect]:
    """Scansioni: finestre di parole consecutive sulla stessa riga OCR che
    contengono la stringa cercata. Il confronto è senza spazi, così le parole
    incollate o spezzate dall'OCR non contano, e la box copre solo i
    caratteri corrispondenti."""
    q = [t for t in (B.canonico(t) for t in testo.split()) if t]
    if not q:
        return []
    obiettivo = "".join(q)
    trovati: list[tuple[float, pymupdf.Rect]] = []
    n = len(parole)
    for i in range(n):
        for lung in range(max(1, len(q) - 2), len(q) + 3):
            if i + lung > n:
                break
            finestra = parole[i : i + lung]
            if any(w.riga != parole[i].riga for w in finestra):
                break
            testo_fin = "".join(w.canon for w in finestra)
            # la finestra deve coprire (quasi) tutta la stringa: altrimenti "VIA 2 GIUGNO,"
            # senza "13" passerebbe e il numero civico resterebbe in pagina
            if not 0.9 * len(obiettivo) <= len(testo_fin) <= 1.8 * len(obiettivo):
                continue
            if len(obiettivo) >= 6:
                al = fuzz.partial_ratio_alignment(obiettivo, testo_fin)
                if al is None or al.score < SOGLIA_OCR:
                    continue
                punteggio, a, b = al.score, al.dest_start, al.dest_end
            else:  # stringhe corte: solo corrispondenza intera, per non prenderle dentro altre parole
                punteggio = fuzz.ratio(obiettivo, testo_fin)
                if punteggio < SOGLIA_OCR:
                    continue
                a, b = 0, len(testo_fin)
            r = _ritaglio_parole(finestra, a, b)
            if r.is_empty:
                continue
            # tieni la migliore tra le finestre che si sovrappongono
            sovrapposte = [k for k, (s, t) in enumerate(trovati) if (t & r).get_area() > 0.3 * min(t.get_area(), r.get_area())]
            if sovrapposte and all(trovati[k][0] >= punteggio for k in sovrapposte):
                continue
            for k in sorted(sovrapposte, reverse=True):
                trovati.pop(k)
            trovati.append((punteggio, r))
    return [r for _, r in trovati]


# ── Redazione e riscrittura ──────────────────────────────────────────


def rect_visivo(r: pymupdf.Rect, page: pymupdf.Page) -> pymupdf.Rect:
    return (r * page.rotation_matrix).normalize() if page.rotation else pymupdf.Rect(r)


def rect_pagina(vis: pymupdf.Rect, page: pymupdf.Page) -> pymupdf.Rect:
    return (vis * page.derotation_matrix).normalize() if page.rotation else pymupdf.Rect(vis)


def ritaglia(r: pymupdf.Rect, coperti: list, page: pymupdf.Page) -> pymupdf.Rect:
    """Toglie da *r* la parte già coperta da un'altra sostituzione sulla stessa
    riga (parole incollate dall'OCR come "MOHAMEDVIA" finiscono in due
    finestre): si taglia lungo la direzione del testo, nello spazio visivo."""
    v = rect_visivo(r, page)
    for c in coperti:
        cv = rect_visivo(c, page)
        inter = v & cv
        if inter.is_empty or inter.height < 0.5 * min(v.height, cv.height):
            continue
        if cv.x0 <= v.x0:
            v.x0 = max(v.x0, cv.x1)  # coperto a sinistra
        else:
            v.x1 = min(v.x1, cv.x0)  # coperto a destra
    return rect_pagina(v, page) if v.x1 > v.x0 else pymupdf.Rect()


def scrivi_sopra(page: pymupdf.Page, r: pymupdf.Rect, testo: str) -> None:
    """Scrive *testo* dentro il rettangolo (spazio pagina), adattando il corpo
    del carattere a larghezza e altezza viste."""
    v = rect_visivo(r, page)
    if not testo:
        return
    fontsize = min(v.height * 0.8, v.width / (LARGHEZZA_COURIER * len(testo)))
    fontsize = max(2.5, fontsize)
    punto = pymupdf.Point(v.x0, v.y1 - v.height * 0.22) * (page.derotation_matrix if page.rotation else pymupdf.Matrix(1, 1))
    page.insert_text(punto, testo, fontsize=fontsize, fontname="cour", color=(0, 0, 0), rotate=page.rotation)


QUALITA_JPEG = 75  # le scansioni originali sono spesso PNG o JPEG a qualità massima: decine di MB


def ricomprimi_immagini(page: pymupdf.Page) -> None:
    """Riscrive le immagini della pagina (già redatte) come JPEG, a parità di
    risoluzione e rotazione: le scansioni scendono a una frazione del peso."""
    for info in page.get_images(full=True):
        xref = info[0]
        pix = pymupdf.Pixmap(page.parent, xref)
        if pix.alpha or pix.n > 3:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        page.parent.update_stream(xref, pix.tobytes("jpeg", jpg_quality=QUALITA_JPEG), compress=False)
        # l'oggetto immagine deve dichiarare la nuova codifica
        page.parent.xref_set_key(xref, "Filter", "/DCTDecode")
        page.parent.xref_set_key(xref, "ColorSpace", "/DeviceRGB" if pix.n == 3 else "/DeviceGray")
        page.parent.xref_set_key(xref, "BitsPerComponent", "8")
        for chiave in ("DecodeParms", "SMask", "Mask"):
            page.parent.xref_set_key(xref, chiave, "null")


def anonimizza_pdf(pdf: Path, uscita: Path, sostituzioni: list, zone: list, parole_pagine: list | None,
                   rasterizza: int | None = None) -> dict:
    """Ritorna {stringa: occorrenze trovate} per il rapporto.

    Con *rasterizza* (dpi) ogni pagina viene prima resa come immagine e il
    PDF di uscita contiene solo quella: è la via sicura per i PDF nativi, dove
    la riscrittura del contenuto può perdere il modulo (riquadri, diciture) e
    dove testo, font e metadati originali resterebbero nel file.
    """
    doc = pymupdf.open(pdf)
    conteggi = {orig: 0 for orig, _ in sostituzioni}
    nuovo_doc = pymupdf.open() if rasterizza else None
    for n, page in enumerate(doc, start=1):
        # 1. dove sta ogni stringa (sulla pagina originale)
        da_scrivere = []  # (rect pagina, testo fittizio)
        coperti = []
        for orig, nuovo in sostituzioni:  # dalla più lunga: "STABILIMENTO DI ANCONA" prima di "ANCONA"
            if parole_pagine is not None:
                rects = occorrenze_ocr(parole_pagine[n - 1], orig)
            else:
                rects = occorrenze_testo(page, orig)
            for r in rects:
                if any((c & r).get_area() > 0.5 * r.get_area() for c in coperti):
                    continue  # per più di metà dentro una sostituzione già fatta (più lunga)
                r = ritaglia(r, coperti, page)
                if r.is_empty or r.width < 2 or r.height < 2:
                    continue
                conteggi[orig] += 1
                coperti.append(r)
                da_scrivere.append((r, nuovo))
        # 2. la pagina su cui lavorare: l'originale, o la sua immagine
        if nuovo_doc is not None:
            pix = page.get_pixmap(dpi=rasterizza, alpha=False)
            page_out = nuovo_doc.new_page(width=page.rect.width, height=page.rect.height)
            page_out.insert_image(page_out.rect, stream=pix.tobytes("jpeg", jpg_quality=QUALITA_JPEG))
            page_out.set_rotation(0)
            # le pagine rese hanno rotazione 0 e coordinate = pagina vista
            da_scrivere = [(rect_visivo(r, page), t) for r, t in da_scrivere]
        else:
            page_out = page
        # 3. redazioni: stringhe trovate e zone
        for r, _ in da_scrivere:
            page_out.add_redact_annot(r + (-0.5, -0.5, 0.5, 0.5), fill=(1, 1, 1))
        zone_pagina = []
        for z in zone:
            pagine = z.get("pagine", "tutte")
            if pagine != "tutte" and n not in pagine:
                continue
            y0, x0, y1, x1 = z["vis"]
            w, h = page.rect.width, page.rect.height  # page.rect è già la pagina vista
            vis = pymupdf.Rect(x0 * w, y0 * h, x1 * w, y1 * h)
            rz = rect_pagina(vis, page_out)
            if z.get("senza_testo"):
                zone_pagina.append(rz)
            page_out.add_redact_annot(rz, fill=(1, 1, 1))
        page_out.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_PIXELS, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
        if nuovo_doc is None:
            ricomprimi_immagini(page_out)
        for r, nuovo in da_scrivere:
            if any(z.contains(r) for z in zone_pagina):
                continue  # zona "senza_testo" (blocchi di intestazione): niente testo fittizio
            scrivi_sopra(page_out, r, nuovo)
    out = nuovo_doc if nuovo_doc is not None else doc
    out.set_metadata({})  # niente autore, produttore o date del file originale
    out.del_xml_metadata()
    out.save(uscita, garbage=4, deflate=True)
    return conteggi


def _sostituisci(valore: str, sostituzioni: list) -> str:
    for orig, nuovo in sostituzioni:
        def _rimpiazza(m, nuovo=nuovo):
            t = m.group(0)
            return nuovo.upper() if t.isupper() else (nuovo.lower() if t.islower() else nuovo)
        valore = re.sub(re.escape(orig), _rimpiazza, valore, flags=re.IGNORECASE)
    return valore


def anonimizza_json(nodo, sostituzioni: list):
    if isinstance(nodo, dict):
        return {k: anonimizza_json(v, sostituzioni) for k, v in nodo.items()}
    if isinstance(nodo, list):
        return [anonimizza_json(v, sostituzioni) for v in nodo]
    if isinstance(nodo, str):
        return _sostituisci(nodo, sostituzioni)
    return nodo


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mappa")
    ap.add_argument("uscita")
    args = ap.parse_args()
    mappa = json.loads(Path(args.mappa).read_text(encoding="utf-8"))
    uscita = Path(args.uscita)
    uscita.mkdir(parents=True, exist_ok=True)

    for d in mappa["documenti"]:
        fam = mappa.get("famiglie", {}).get(d.get("famiglia", ""), {})
        sostituzioni = [tuple(x) for x in fam.get("sostituzioni", []) + d.get("sostituzioni", [])]
        sostituzioni.sort(key=lambda s: -len(s[0]))
        zone = fam.get("zone_bianche", []) + d.get("zone_bianche", [])
        pdf = Path(d["pdf"]).expanduser()
        parole_pagine = None
        if d.get("ocr"):
            doc = pymupdf.open(pdf)
            righe = json.loads(Path(d["ocr"]).read_text(encoding="utf-8"))
            parole_pagine = [B.parole_pagina(r, page) for r, page in zip(righe, doc)]
        conteggi = anonimizza_pdf(pdf, uscita / f"{d['nome']}.pdf", sostituzioni, zone, parole_pagine,
                                  rasterizza=d.get("rasterizza", fam.get("rasterizza")))
        campi = json.loads(Path(d["json"]).read_text(encoding="utf-8"))
        (uscita / f"{d['nome']}.json").write_text(json.dumps(anonimizza_json(campi, sostituzioni), ensure_ascii=False, indent=2), encoding="utf-8")
        zero = [o for o, c in conteggi.items() if c == 0]
        print(f"{d['nome']}: {sum(conteggi.values())} sostituzioni sul PDF, {len(zone)} zone sbiancate"
              + (f"; NON trovate in pagina: {zero}" if zero else ""))


if __name__ == "__main__":
    main()
