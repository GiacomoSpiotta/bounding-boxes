"""Bounding box "di gruppo" (liste e dizionari del JSON) sopra bbox_da_json.

Riusa OCR, allineamento e disegno di bbox_da_json.py, importato come modulo
e non modificato. Rispetto a quello script aggiunge due cose:

  1. Per ogni nodo lista/dizionario del JSON un aggregato di gruppo: l'unione
     delle box delle foglie sottostanti, divisa in blocchi (uno per pagina, e
     un blocco nuovo quando tra due foglie c'è un salto verticale maggiore di
     SALTO_BLOCCO x altezza pagina), score minimo e medio, foglie trovate,
     dubbie e non trovate, e le parole OCR che stanno dentro il blocco ma che
     nessuna foglia copre: è il testo della sezione che il modello ha saltato.
  2. Gli elementi delle liste di valori scalari (es. "clausole") che ricorrono
     più volte in pagina si scelgono vicino agli elementi contigui della
     stessa lista già trovati, come bbox_da_json fa già per i campi delle
     righe di tabella (dizionari dentro liste) con la mediana della riga.
     Le foglie restano l'output primario: sono loro a dire se un valore non
     è in pagina.

Scrive in output:
  - <nome>_gruppi_bbox.json : {"campi":  stesso albero di bbox_da_json,
                               "gruppi": {percorso: aggregato}} ("" = radice)
  - <nome>_gruppi.pdf       : box delle foglie (verde ok / arancio dubbie),
                               blocchi di gruppo tratteggiati in blu, parole
                               che nessuna foglia copre campite in rosso
  - <nome>_gruppi_tempi.json

Uso: python bbox_gruppi.py file.pdf campi.json [cartella_output]
        [--tier medium|small|tiny] [--dpi 300] [--det-max 1600] [--threads 4] [--riusa-ocr]
Con --riusa-ocr legge <nome>_ocr.json salvato da bbox_da_json o da una
corsa precedente di questo script nella stessa cartella di output.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# bbox_da_json fissa i thread dagli argomenti e importa numpy: va importato
# prima di qualsiasi altra libreria numerica.
import bbox_da_json as B  # noqa: E402
import pymupdf  # noqa: E402

FRAZIONE_DENTRO = 0.5   # una parola sta "dentro" una box se almeno metà della sua area è coperta
SALTO_BLOCCO = 0.20     # frazione dell'altezza pagina: un salto verticale maggiore tra due foglie
                        # dello stesso gruppo apre un blocco nuovo (campi in testa e a piè pagina)


# ── Visita del JSON con ancore di gruppo ─────────────────────────────


def _yc(e: dict) -> float:
    """Centro verticale visivo dell'esito: è la coordinata giusta per dire
    "stessa riga" anche sulle pagine ruotate."""
    return e["y_vis"]


def _ruota(bbox: list[float], m) -> list[float]:
    """Applica la matrice *m* a una box [ymin, xmin, ymax, xmax]; None = identità."""
    if m is None:
        return list(bbox)
    y0, x0, y1, x1 = bbox
    r = (pymupdf.Rect(x0, y0, x1, y1) * m).normalize()
    return [r.y0, r.x0, r.y1, r.x1]


def calcola_ancora(trovati: list[tuple[int, int, float]], ordinale: int, modo: str) -> tuple[int, float] | None:
    """(pagina, y) verso cui attirare un valore ripetuto in pagina.

    *trovati*: (ordinale, pagina, y) degli altri scalari del gruppo già
    localizzati. La pagina è la più frequente tra loro. La y con modo
    "vicini" sta a metà tra l'elemento precedente e il successivo già
    trovati (liste: l'ordine del JSON segue quello della pagina); con modo
    "mediana" è la mediana del gruppo (righe di tabella: l'ordine delle
    chiavi non dice nulla sulla posizione).
    """
    if not trovati:
        return None
    pagine = [pg for _, pg, _ in trovati]
    pg = max(set(pagine), key=pagine.count)
    stessa = sorted((o, y) for o, p_, y in trovati if p_ == pg)
    if modo == "vicini":
        prima = [y for o, y in stessa if o < ordinale]
        dopo = [y for o, y in stessa if o > ordinale]
        if prima and dopo:
            return pg, (prima[-1] + dopo[0]) / 2
        if prima:
            return pg, prima[-1]
        if dopo:
            return pg, dopo[0]
    ys = [y for _, y in stessa]
    return pg, ys[len(ys) // 2]


def visita(nodo, pagine, doc, percorso: str, esiti: list, gruppi: list, modo: str | None = None):
    """Come bbox_da_json.visita, ma ogni contenitore registra il proprio
    aggregato in *gruppi* e gli scalari ripetuti in pagina si cercano con
    un'ancora presa dagli altri scalari del gruppo (*modo*: "vicini" per le
    liste, "mediana" per i dizionari dentro liste, None per gli altri
    dizionari, che restano come in bbox_da_json)."""
    if not isinstance(nodo, (dict, list)):
        if nodo is None or str(nodo).strip() == "":
            return B._foglia_vuota()
        e = B.cerca(pagine, str(nodo))
        esiti.append((percorso, str(nodo), e))
        return e

    e_lista = isinstance(nodo, list)
    items = list(enumerate(nodo)) if e_lista else list(nodo.items())

    def sotto(k):
        if e_lista:
            return f"{percorso}[{k}]"
        return f"{percorso}.{k}" if percorso else str(k)

    n0, pos = len(esiti), len(gruppi)
    out = {}
    ambigui = []
    trovati: list[tuple[int, int, float]] = []
    for ordinale, (k, v) in enumerate(items):
        p = sotto(k)
        if isinstance(v, dict):
            out[k] = visita(v, pagine, doc, p, esiti, gruppi, "mediana" if e_lista else None)
        elif isinstance(v, list):
            out[k] = visita(v, pagine, doc, p, esiti, gruppi, "vicini")
        elif v is None or str(v).strip() == "":
            out[k] = B._foglia_vuota()
        else:
            e = B.cerca(pagine, str(v))
            if modo and e["candidati"] > 1:
                ambigui.append((ordinale, k, v, p))  # si decide dopo, vicino agli altri
                continue
            out[k] = e
            esiti.append((p, str(v), e))
            if e["page"] is not None:
                trovati.append((ordinale, e["page"], _yc(e)))
    for ordinale, k, v, p in ambigui:  # in ordine di posizione nel gruppo
        e = B.cerca(pagine, str(v), ancora=calcola_ancora(trovati, ordinale, modo))
        out[k] = e
        esiti.append((p, str(v), e))
        if e["page"] is not None and modo == "vicini":
            trovati.append((ordinale, e["page"], _yc(e)))  # aiuta i successivi

    agg, non_coperte = aggrega("lista" if e_lista else "dizionario", esiti[n0:], pagine, doc)
    gruppi.insert(pos, (percorso, agg, non_coperte))  # prima dei sottogruppi
    if e_lista:
        return [out[i] for i in range(len(nodo))]
    return {k: out[k] for k in nodo}


def _ricalcola_gruppi(nodo, esiti: list, pagine, doc, percorso: str, gruppi: list) -> None:
    """Ricostruisce gli aggregati di gruppo dagli esiti già calcolati (dopo
    riancora_per_documento), senza nuove ricerche: stesso ordine di visita."""
    if not isinstance(nodo, (dict, list)):
        return
    pos = len(gruppi)
    items = list(enumerate(nodo)) if isinstance(nodo, list) else list(nodo.items())
    for k, v in items:
        p = f"{percorso}[{k}]" if isinstance(nodo, list) else (f"{percorso}.{k}" if percorso else str(k))
        _ricalcola_gruppi(v, esiti, pagine, doc, p, gruppi)
    prefisso = percorso
    foglie = [x for x in esiti if not prefisso or x[0] == prefisso or x[0].startswith(prefisso + ".") or x[0].startswith(prefisso + "[")]
    agg, non_coperte = aggrega("lista" if isinstance(nodo, list) else "dizionario", foglie, pagine, doc)
    gruppi.insert(pos, (percorso, agg, non_coperte))


# ── Aggregato di gruppo ──────────────────────────────────────────────


def frazione_dentro(p: B.Parola, bbox: list[float]) -> float:
    """Quota dell'area della parola che cade dentro *bbox* [ymin, xmin, ymax, xmax]."""
    y0, x0, y1, x1 = bbox
    inter = max(0.0, min(p.x1, x1) - max(p.x0, x0)) * max(0.0, min(p.y1, y1) - max(p.y0, y0))
    area = (p.x1 - p.x0) * (p.y1 - p.y0)
    return inter / area if area > 0 else 0.0


def _unione(boxes: list[list[float]]) -> list[float]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def blocchi_pagina(boxes: list[list[float]], altezza: float) -> list[list[list[float]]]:
    """Divide le box (stessa pagina) in blocchi di foglie verticalmente
    contigue: un salto maggiore di SALTO_BLOCCO x altezza apre un blocco."""
    blocchi: list[list[list[float]]] = []
    for b in sorted(boxes, key=lambda b: b[0]):
        if blocchi and b[0] - max(x[2] for x in blocchi[-1]) <= SALTO_BLOCCO * altezza:
            blocchi[-1].append(b)
        else:
            blocchi.append([b])
    return blocchi


def aggrega(tipo: str, foglie: list, pagine, doc: pymupdf.Document) -> tuple[dict, dict[int, list[B.Parola]]]:
    """Aggregato di un contenitore a partire dagli esiti delle sue foglie
    (percorso, valore, esito). I blocchi si formano nello spazio visivo
    (salti verticali come li si vede, anche su pagine ruotate) e si esportano
    nello spazio pagina. Ritorna anche, per pagina, le parole OCR dei suoi
    blocchi che nessuna foglia copre (servono al disegno)."""
    trovate = [(p, e) for p, _, e in foglie if e["page"] is not None]
    agg = {
        "tipo": tipo,
        "foglie": len(foglie),
        "trovate": len(trovate),
        "ok": sum(1 for _, e in trovate if e["score"] >= B.SCORE_OK),
        "score_min": round(min(e["score"] for _, e in trovate), 3) if trovate else None,
        "score_medio": round(sum(e["score"] for _, e in trovate) / len(trovate), 3) if trovate else None,
        "dubbie": [p for p, e in trovate if e["score"] < B.SCORE_OK],
        "non_trovate": [p for p, _, e in foglie if e["page"] is None],
        "blocchi": [],
    }
    non_coperte: dict[int, list[B.Parola]] = {}
    for n in sorted({e["page"] for _, e in trovate}):
        boxes = [e["bbox_page"] for _, e in trovate if e["page"] == n]
        parole = pagine[n - 1]
        page = doc[n - 1]
        rot = page.rotation_matrix if page.rotation else None
        derot = page.derotation_matrix if page.rotation else None
        for blocco in blocchi_pagina([_ruota(b, rot) for b in boxes], page.rect.height):
            unione = _ruota(_unione(blocco), derot)
            dentro = [p for p in parole if frazione_dentro(p, unione) >= FRAZIONE_DENTRO]
            fuori = [p for p in dentro if not any(frazione_dentro(p, b) >= FRAZIONE_DENTRO for b in boxes)]
            agg["blocchi"].append({
                "page": n,
                "bbox_page": [round(v, 1) for v in unione],
                "foglie": len(blocco),
                "parole_dentro": len(dentro),
                "parole_coperte": len(dentro) - len(fuori),
                "parole_non_coperte": len(fuori),
                "copertura_parole": round((len(dentro) - len(fuori)) / len(dentro), 3) if dentro else None,
                "testo_non_coperto": " ".join(p.testo for p in fuori),
            })
            non_coperte.setdefault(n, []).extend(fuori)
    return agg, non_coperte


# ── Disegno ──────────────────────────────────────────────────────────


def _profondita(percorso: str) -> int:
    return 0 if not percorso else 1 + percorso.count(".") + percorso.count("[")


def disegna_gruppi(doc: pymupdf.Document, gruppi: list, righe_tab: dict | None = None) -> None:
    """Blocchi di gruppo tratteggiati in blu (margine più largo per i gruppi
    esterni, così i sottogruppi restano dentro) e, per la radice, le parole
    che nessuna foglia copre campite in rosso: è ciò che manca nel JSON.
    L'unica scritta è il numero di riga sui blocchi che sono righe di tabella."""
    blu = (0.1, 0.3, 0.9)
    righe_tab = righe_tab or {}
    prof_max = max((_profondita(p) for p, _, _ in gruppi), default=1)
    for percorso, agg, non_coperte in gruppi:
        if not percorso:
            for n, parole in non_coperte.items():
                shape = doc[n - 1].new_shape()
                for p in parole:
                    shape.draw_rect(pymupdf.Rect(p.x0, p.y0, p.x1, p.y1))
                shape.finish(color=None, fill=(1, 0, 0), fill_opacity=0.15)
                shape.commit()
            continue
        margine = 3 + 2 * (prof_max - _profondita(percorso))
        for blocco in agg["blocchi"]:
            page = doc[blocco["page"] - 1]
            y0, x0, y1, x1 = blocco["bbox_page"]
            shape = page.new_shape()
            shape.draw_rect(pymupdf.Rect(x0 - margine, y0 - margine, x1 + margine, y1 + margine))
            shape.finish(color=blu, width=0.6, dashes="[3 2] 0")
            shape.commit()
            if percorso in righe_tab:
                tipo, n, _ = righe_tab[percorso]
                B.scrivi_etichetta(page, blocco["bbox_page"], f"r{n}" if tipo == "dettaglio" else f"A{n}", blu)


# ── Main ─────────────────────────────────────────────────────────────


def ocr_documento(doc: pymupdf.Document, args, out_dir: Path, tempi: dict) -> list[list[dict]]:
    """Righe OCR di ogni pagina, dal file salvato (--riusa-ocr) o da PaddleOCR.
    Stesso file <nome>_ocr.json di bbox_da_json, così le due corse si riusano a vicenda."""
    ocr_json = out_dir / f"{Path(args.pdf).stem}_ocr.json"
    if args.riusa_ocr and ocr_json.exists():
        righe_pagine = json.loads(ocr_json.read_text(encoding="utf-8"))
        tempi["caricamento_modello_s"] = 0.0
        tempi["pagine"] = [{"pagina": n, "render_s": 0.0, "ocr_s": 0.0} for n in range(1, len(righe_pagine) + 1)]
        print(f"OCR riusato da {ocr_json.name}")
        return righe_pagine
    t0 = time.perf_counter()
    ocr = B.carica_ocr(args.tier, args.threads, args.det_max)
    tempi["caricamento_modello_s"] = round(time.perf_counter() - t0, 2)
    righe_pagine = []
    for n, page in enumerate(doc, start=1):
        t0 = time.perf_counter()
        img = B.rasterizza(page, args.dpi)
        t1 = time.perf_counter()
        righe_pagine.append(B.ocr_pagina(ocr, img, page, args.dpi))
        t2 = time.perf_counter()
        tempi["pagine"].append({"pagina": n, "render_s": round(t1 - t0, 2), "ocr_s": round(t2 - t1, 2)})
        print(f"pagina {n}: {len(righe_pagine[-1])} righe OCR, render {t1 - t0:.2f} s + OCR {t2 - t1:.2f} s")
    ocr_json.write_text(json.dumps(righe_pagine, ensure_ascii=False), encoding="utf-8")
    return righe_pagine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("json_campi")
    ap.add_argument("out_dir", nargs="?", default=None)
    ap.add_argument("--tier", default="medium", choices=["medium", "small", "tiny"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--det-max", type=int, default=1600, help="lato massimo (px) dell'immagine usata per rilevare le righe")
    ap.add_argument("--threads", type=int, default=B.THREADS)
    ap.add_argument("--riusa-ocr", action="store_true", help="ricarica le righe OCR salvate da una corsa precedente")
    args = ap.parse_args()

    pdf = Path(args.pdf)
    out_dir = Path(args.out_dir) if args.out_dir else pdf.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    campi = json.loads(Path(args.json_campi).read_text(encoding="utf-8"))
    tempi = {"threads": args.threads, "tier": args.tier, "dpi": args.dpi, "det_max": args.det_max, "pagine": []}
    t_start = time.perf_counter()

    doc = pymupdf.open(pdf)
    righe_pagine = ocr_documento(doc, args, out_dir, tempi)
    pagine = [B.parole_pagina(righe, page) for righe, page in zip(righe_pagine, doc)]
    righe_tab = B.righe_del_json(campi)
    for info, parole in zip(tempi["pagine"], pagine):
        info["parole"] = len(parole)

    t0 = time.perf_counter()
    esiti: list = []
    gruppi: list = []
    albero = visita(campi, pagine, doc, "", esiti, gruppi)
    tempi["riancorati"] = B.riancora_per_documento(esiti, pagine)
    if tempi["riancorati"]:
        # gli aggregati di gruppo vanno ricalcolati sugli esiti spostati
        gruppi_nuovi: list = []
        _ricalcola_gruppi(campi, esiti, pagine, doc, "", gruppi_nuovi)
        gruppi[:] = gruppi_nuovi
    tempi["matching_s"] = round(time.perf_counter() - t0, 3)
    tempi["campi"] = len(esiti)
    tempi["campi_trovati"] = sum(1 for _, _, e in esiti if e["page"] is not None)
    tempi["gruppi"] = len(gruppi)
    tempi["totale_s"] = round(time.perf_counter() - t_start, 2)

    for percorso, valore, e in esiti:
        stato = "OK " if e["page"] and e["score"] >= B.SCORE_OK else ("?? " if e["page"] else "-- ")
        print(f"{stato}{e['ms']:6.1f} ms  score={e['score']:.2f}  p{e['page']}  {percorso}: {valore[:60]!r}")

    print("\nGRUPPI")
    for percorso, agg, _ in gruppi:
        smin = f"{agg['score_min']:.2f}" if agg["score_min"] is not None else "-"
        print(f"{percorso or '(radice)':40s} {agg['tipo']:10s} {agg['trovate']}/{agg['foglie']} trovate, {agg['ok']} ok, score min {smin}")
        for b in agg["blocchi"]:
            print(f"    p{b['page']} {b['bbox_page']}  {b['foglie']} foglie, {b['parole_dentro']} parole, {b['parole_non_coperte']} non coperte")
            if b["testo_non_coperto"]:
                print(f"       non coperte: {b['testo_non_coperto'][:110]!r}")

    B.disegna(doc, esiti, righe_tab)
    disegna_gruppi(doc, gruppi, righe_tab)
    doc.save(out_dir / f"{pdf.stem}_gruppi.pdf")
    uscita = {"campi": albero, "gruppi": {percorso: agg for percorso, agg, _ in gruppi}}
    (out_dir / f"{pdf.stem}_gruppi_bbox.json").write_text(json.dumps(uscita, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{pdf.stem}_gruppi_tempi.json").write_text(json.dumps(tempi, indent=2), encoding="utf-8")
    print(
        f"\nmodello {tempi['caricamento_modello_s']} s, render {sum(p['render_s'] for p in tempi['pagine']):.2f} s, "
        f"OCR {sum(p['ocr_s'] for p in tempi['pagine']):.2f} s, matching {tempi['matching_s']} s, "
        f"totale {tempi['totale_s']} s ({tempi['campi_trovati']}/{tempi['campi']} campi trovati, "
        f"{tempi['gruppi']} gruppi, {args.threads} thread)"
    )
    print(f"Risultati in {out_dir}")


if __name__ == "__main__":
    main()
