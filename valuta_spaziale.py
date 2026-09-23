"""Valuta le box di Gemini come indicazione spaziale, non come box di parola.

Per ogni foglia del JSON legge, attraverso le parole OCR in cache, cosa c'è
dentro la box di Gemini e classifica:
  - cella esatta      : la box coincide (IoU > 0) con la box di parola dell'OCR
  - altra occorrenza  : dentro la box c'è il testo del valore, ma in un'altra
                        riga o pagina rispetto alla scelta dell'OCR
  - testo diverso     : dentro la box c'è altro (o l'OCR non ha letto nulla)
  - nessuna box       : Gemini non ha restituito la foglia
e conta quante box stanno sulle pagine del documento a cui la foglia
appartiene (--pagine "0:1;1:2,3,4" = documenti[0] a pagina 1, [1] alle 2-4;
senza opzione la pagina non si valuta).

Con --pdf aggiunge un controllo che NON dipende dall'OCR, per le righe di
tabella (dizionari dentro liste con soli scalari): le celle di una stessa
riga devono stare sulla stessa linea visiva (entro 5 punti, tenendo conto
della rotazione della pagina) e le righe devono seguire l'ordine dei numeri.
Lo calcola sia per Gemini sia per l'OCR, così si vede chi sbaglia riga.

Uso: python valuta_spaziale.py campi.json cartella_output nome_pdf_senza_estensione
        [--gemini suffisso] [--pagine "0:1;1:2,3,4"] [--pdf file.pdf]
Legge <nome>_ocr.json, <nome>_gruppi_bbox.json e
<nome>_vertex_<suffisso>.json (default foglie-permissivo_gemini-3.8-flash).
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import pymupdf

sys.argv, _argv = ["x"], sys.argv  # bbox_da_json legge --threads da argv all'import
import bbox_da_json as B  # noqa: E402
import bbox_vertex as V  # noqa: E402
from bbox_gruppi import frazione_dentro  # noqa: E402
from rapidfuzz import fuzz  # noqa: E402

sys.argv = _argv


def y_visivo(bbox: list[float], page: pymupdf.Page) -> float:
    """Centro verticale della box come la si vede (dopo la rotazione della pagina)."""
    y0, x0, y1, x1 = bbox
    r = (pymupdf.Rect(x0, y0, x1, y1) * page.rotation_matrix).normalize()
    return (r.y0 + r.y1) / 2


def coerenza_righe(nome: str, celle: dict[str, tuple[int, list[float]]], righe: dict, doc: pymupdf.Document) -> None:
    """celle: {percorso foglia: (pagina, bbox)}. Stampa quante celle stanno sulla
    linea della propria riga e quante coppie di righe consecutive sono fuori ordine."""
    per_riga: dict[str, list] = {}
    for k, (pg, bb) in celle.items():
        gen = k.rpartition(".")[0]
        if gen in righe and righe[gen][0] == "dettaglio":
            per_riga.setdefault(gen, []).append((pg, bb))
    if not per_riga:
        return
    tot = in_riga = coerenti = 0
    per_pagina: dict[tuple, list] = {}
    for gen, lst in per_riga.items():
        pagine = [pg for pg, _ in lst]
        pg = max(set(pagine), key=pagine.count)
        ys = [y_visivo(bb, doc[pg - 1]) for p_, bb in lst if p_ == pg]
        med = statistics.median(ys)
        ok = sum(1 for p_, bb in lst if p_ == pg and abs(y_visivo(bb, doc[pg - 1]) - med) <= 5)
        tot += len(lst)
        in_riga += ok
        coerenti += ok == len(lst)
        per_pagina.setdefault((righe[gen][2], pg), []).append((righe[gen][1], med))
    viol = coppie = sovrapposte = 0
    for lst in per_pagina.values():
        lst.sort()
        for (_, y1), (_, y2) in zip(lst, lst[1:]):
            coppie += 1
            viol += y2 <= y1 + 3
            sovrapposte += abs(y2 - y1) <= 3  # due righe del JSON sulla stessa linea fisica: riga duplicata
    print(f"  righe [{nome}]: celle sulla linea della propria riga {in_riga}/{tot} ({100 * in_riga / tot:.0f}%), "
          f"righe interamente coerenti {coerenti}/{len(per_riga)}, righe consecutive fuori ordine {viol}/{coppie}, "
          f"righe del JSON sovrapposte alla stessa linea (duplicati dell'estrazione): {sovrapposte}")


def parole_dentro(parole, bbox):
    return [p for p in parole if frazione_dentro(p, bbox) >= 0.5]


def contiene(valore: str, parole_box) -> bool:
    """Il testo del valore sta tra le parole OCR dentro la box."""
    q = B.token_valore(valore)
    if not q:
        return False
    if len(q) <= 2:
        return all(any(t == p.canon or (len(t) >= 3 and B.sotto_span(t, p.testo) is not None) for p in parole_box) for t in q)
    return fuzz.token_set_ratio(" ".join(q), " ".join(p.canon for p in parole_box)) >= 75


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_campi")
    ap.add_argument("out_dir")
    ap.add_argument("stem")
    ap.add_argument("--gemini", default="foglie-permissivo_gemini-3.8-flash")
    ap.add_argument("--pagine", default=None)
    ap.add_argument("--pdf", default=None, help="il PDF: attiva il controllo di coerenza delle righe")
    args = ap.parse_args()
    out = Path(args.out_dir)

    pagine_doc: dict[int, set] = {}
    if args.pagine:
        for parte in args.pagine.split(";"):
            i, pg = parte.split(":")
            pagine_doc[int(i)] = {int(x) for x in pg.split(",")}

    campi = json.loads(Path(args.json_campi).read_text(encoding="utf-8"))
    valori = dict(V.foglie_del_json(campi))
    pagine = [B.parole_pagina(r) for r in json.loads((out / f"{args.stem}_ocr.json").read_text(encoding="utf-8"))]
    ocr = V.esiti_foglie(json.loads((out / f"{args.stem}_gruppi_bbox.json").read_text(encoding="utf-8"))["campi"])
    g = json.loads((out / f"{args.stem}_vertex_{args.gemini}.json").read_text(encoding="utf-8"))
    box_per_foglia: dict[str, list] = {}
    for p in g["pagine"]:
        for r in p["regioni"]:
            box_per_foglia.setdefault(r["etichetta"], []).append((p["page"], r["bbox_page"]))

    c = dict(giusta=0, altra=0, altro=0, nessuna=0, pag_ok=0, con_box=0, rec=0)
    righe_sbagliate = []
    foglie = [(k, v) for k, v in valori.items() if not k.endswith("controllo_assente")]
    for k, v in foglie:
        boxes = box_per_foglia.get(k, [])
        if not boxes:
            c["nessuna"] += 1
            continue
        c["con_box"] += 1
        if pagine_doc:
            idx = int(k.split("[")[1].split("]")[0]) if "[" in k else 0
            if any(pg in pagine_doc.get(idx, set()) for pg, _ in boxes):
                c["pag_ok"] += 1
        eo = ocr.get(k, {})
        esito = "altro"
        for pg, bb in boxes:
            if eo.get("page") == pg and V.iou(bb, eo["bbox_page"]) > 0:
                esito = "giusta"
                break
        if esito != "giusta":
            for pg, bb in boxes:
                if contiene(v, parole_dentro(pagine[pg - 1], bb)):
                    esito = "altra" if eo.get("page") else "rec"
                    break
        if esito == "rec":
            c["rec"] += 1
            c["giusta"] += 1
        else:
            c[esito] += 1
        if esito == "altra":
            righe_sbagliate.append(k)
    n = len(foglie)
    print(f"{args.stem}: {n} foglie (senza controllo), Gemini {args.gemini}")
    print(f"  box presente: {c['con_box']} ({100 * c['con_box'] / n:.0f}%)"
          + (f", sulla pagina del proprio documento: {c['pag_ok']} ({100 * c['pag_ok'] / max(c['con_box'], 1):.0f}% delle box)" if pagine_doc else ""))
    print(f"  testo giusto, cella esatta: {c['giusta']} ({100 * c['giusta'] / n:.0f}%)  [recuperate dove l'OCR non trovava: {c['rec']}]")
    print(f"  testo giusto, altra occorrenza: {c['altra']} ({100 * c['altra'] / n:.0f}%)")
    print(f"  testo diverso: {c['altro']} ({100 * c['altro'] / n:.0f}%)   nessuna box: {c['nessuna']} ({100 * c['nessuna'] / n:.0f}%)")
    controllo = [k for k in valori if k.endswith("controllo_assente") and k in box_per_foglia]
    print(f"  campi di controllo con una box (allucinazioni): {len(controllo)}")
    if righe_sbagliate:
        print("  altra occorrenza, esempi:", ", ".join(k.split(".", 1)[-1] for k in righe_sbagliate[:8]))
    if args.pdf:
        doc = pymupdf.open(args.pdf)
        righe = V.righe_del_json(campi)
        coerenza_righe("Gemini", {k: b[0] for k, b in box_per_foglia.items()}, righe, doc)
        coerenza_righe("OCR", {k: (e["page"], e["bbox_page"]) for k, e in ocr.items() if e.get("page")}, righe, doc)


if __name__ == "__main__":
    main()
