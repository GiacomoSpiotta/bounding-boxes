"""Box di regione da Mistral OCR 4 (blocchi di paragrafo con etichetta).

Manda il PDF all'endpoint OCR di Mistral (La Plateforme) con
include_blocks=True: OCR 4 restituisce per ogni pagina i blocchi con
tipo (text, title, list, table, image, header, footer, signature, ...),
box e contenuto. Qui i blocchi si riportano in punti PDF, si disegnano e,
se nella cartella di output c'è il risultato di bbox_gruppi.py, si
confrontano con i blocchi di gruppo calcolati dall'OCR locale: per ogni
gruppo, l'unione dei blocchi Mistral che cadono nella sua box e l'IoU.

Le coordinate dei blocchi sono in pixel dell'immagine di pagina che
Mistral rende internamente (campo "dimensions": dpi, width, height);
si riscalano sulle dimensioni della pagina PDF.

Scrive in output:
  - <nome>_mistral.json : blocchi per pagina in punti PDF, IoU per gruppo,
                          tempi, modello e pagine fatturate
  - <nome>_mistral.pdf  : blocchi Mistral in viola con il tipo; in verde
                          tratteggiato i blocchi OCR dei gruppi confrontati

Uso: python bbox_mistral.py file.pdf [cartella_output]
        [--modello mistral-ocr-latest] [--chiave XXXX]
La chiave si prende da --chiave, poi dalla variabile MISTRAL_API_KEY,
poi dal file .env nella cartella corrente.
"""

import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pymupdf

ENDPOINT = "https://api.mistral.ai/v1/ocr"
FRAZIONE_DENTRO = 0.5  # un blocco Mistral "sta" in un gruppo se almeno metà della sua area è dentro


# ── Chiave e chiamata ────────────────────────────────────────────────


def chiave_api(esplicita: str | None) -> str:
    if esplicita:
        return esplicita
    if os.environ.get("MISTRAL_API_KEY"):
        return os.environ["MISTRAL_API_KEY"]
    env = Path(".env")
    if env.exists():
        m = re.search(r"^MISTRAL_API_KEY\s*=\s*['\"]?([^'\"\n\r]+)", env.read_text(encoding="utf-8"), re.M)
        if m:
            return m.group(1).strip()
    raise SystemExit("chiave Mistral mancante: --chiave, MISTRAL_API_KEY o .env")


def ocr_mistral(pdf: Path, modello: str, chiave: str) -> tuple[dict, int]:
    """Una chiamata OCR sull'intero PDF. Ritorna (risposta, ms)."""
    b64 = base64.b64encode(pdf.read_bytes()).decode()
    body = {
        "model": modello,
        "document": {"type": "document_url", "document_url": f"data:application/pdf;base64,{b64}"},
        "include_blocks": True,
        "include_image_base64": False,
    }
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {chiave}", "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            risposta = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Mistral HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
    return risposta, round((time.perf_counter() - t0) * 1000)


# ── Geometria ────────────────────────────────────────────────────────


def box_in_punti(blocco: dict, dims: dict, page: pymupdf.Page) -> list[float]:
    """Da pixel dell'immagine Mistral (o frazioni 0-1) a [ymin, xmin, ymax, xmax]
    in punti PDF della pagina, tenendo conto della rotazione."""
    x0, y0, x1, y1 = (float(blocco[k]) for k in ("top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y"))
    w_px, h_px = float(dims.get("width") or 0), float(dims.get("height") or 0)
    if max(x1, y1) <= 1.0:  # coordinate normalizzate
        sx, sy = page.rect.width, page.rect.height
    elif w_px and h_px:  # pixel dell'immagine resa da Mistral
        sx, sy = page.rect.width / w_px, page.rect.height / h_px
    else:  # senza dimensioni: si assume il dpi dichiarato
        s = 72 / float(dims.get("dpi") or 72)
        sx = sy = s
    r = pymupdf.Rect(x0 * sx, y0 * sy, x1 * sx, y1 * sy)
    if page.rotation:
        r = (r * page.derotation_matrix).normalize()
    return [round(r.y0, 1), round(r.x0, 1), round(r.y1, 1), round(r.x1, 1)]


def _area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a: list[float], b: list[float]) -> float:
    inter = _inter(a, b)
    unione = _area(a) + _area(b) - inter
    return round(inter / unione, 3) if unione > 0 else 0.0


def unione(boxes: list[list[float]]) -> list[float]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("out_dir", nargs="?", default=None)
    ap.add_argument("--modello", default="mistral-ocr-latest", help="alias di OCR 4; oppure un id esplicito")
    ap.add_argument("--chiave", default=None)
    args = ap.parse_args()

    pdf = Path(args.pdf)
    out_dir = Path(args.out_dir) if args.out_dir else pdf.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    risposta, ms = ocr_mistral(pdf, args.modello, chiave_api(args.chiave))
    uso = risposta.get("usage_info") or {}
    print(f"Mistral OCR: modello {risposta.get('model')}, {len(risposta.get('pages', []))} pagine in {ms} ms, "
          f"pagine fatturate {uso.get('pages_processed')}")

    blocchi_ocr: dict[str, list[dict]] = {}
    f_gruppi = out_dir / f"{pdf.stem}_gruppi_bbox.json"
    if f_gruppi.exists():
        for percorso, agg in json.loads(f_gruppi.read_text(encoding="utf-8"))["gruppi"].items():
            if percorso:
                blocchi_ocr[percorso] = agg["blocchi"]
        print(f"confronto con i blocchi OCR di {f_gruppi.name}")

    doc = pymupdf.open(pdf)
    viola, verde = (0.6, 0.1, 0.8), (0, 0.6, 0)
    esito = {"modello": risposta.get("model"), "ms": ms, "usage_info": uso, "pagine": [], "gruppi": []}
    per_pagina: dict[int, list[list[float]]] = {}
    for p in risposta.get("pages", []):
        n = int(p.get("index", 0)) + 1
        page = doc[n - 1]
        dims = p.get("dimensions") or {}
        blocchi = p.get("blocks") or []
        voci = []
        shape = page.new_shape()
        for b in blocchi:
            bbox = box_in_punti(b, dims, page)
            voci.append({"tipo": b.get("type"), "bbox_page": bbox, "contenuto": (b.get("content") or "")[:200]})
            y0, x0, y1, x1 = bbox
            shape.draw_rect(pymupdf.Rect(x0, y0, x1, y1))
            page.insert_text((x0, max(y0 - 2, 6)), str(b.get("type")), fontsize=5, color=viola)
        shape.finish(color=viola, width=0.8)
        shape.commit()
        per_pagina[n] = [v["bbox_page"] for v in voci]
        tipi = sorted({v["tipo"] for v in voci})
        print(f"\npagina {n}: {len(voci)} blocchi, dimensions={dims}, tipi={tipi}")
        for v in voci:
            print(f"  {str(v['tipo']):12s} {v['bbox_page']}  {v['contenuto'][:60]!r}")
        esito["pagine"].append({"page": n, "dimensions": dims, "blocchi": voci})

    # confronto: per ogni blocco di gruppo OCR, unione dei blocchi Mistral che vi cadono dentro
    ious = []
    for percorso, blocchi in blocchi_ocr.items():
        for bo in blocchi:
            n, box_g = bo["page"], bo["bbox_page"]
            dentro = [bm for bm in per_pagina.get(n, []) if _area(bm) > 0 and _inter(bm, box_g) / _area(bm) >= FRAZIONE_DENTRO]
            voce = {"percorso": percorso, "page": n, "bbox_ocr": box_g, "blocchi_mistral": len(dentro)}
            if dentro:
                u = unione(dentro)
                voce["bbox_mistral"] = [round(v, 1) for v in u]
                voce["iou"] = iou(u, box_g)
                ious.append(voce["iou"])
                s2 = doc[n - 1].new_shape()
                s2.draw_rect(pymupdf.Rect(box_g[1], box_g[0], box_g[3], box_g[2]))
                s2.finish(color=verde, width=0.6, dashes="[3 2] 0")
                s2.commit()
            esito["gruppi"].append(voce)
    if esito["gruppi"]:
        print("\nGRUPPI (blocco OCR locale <-> unione dei blocchi Mistral che vi cadono)")
        for v in esito["gruppi"]:
            iou_s = f"IoU {v['iou']:.2f}" if "iou" in v else "nessun blocco Mistral dentro"
            print(f"  {v['percorso']:24s} p{v['page']} {v['blocchi_mistral']} blocchi  {iou_s}")
        if ious:
            print(f"  IoU medio {sum(ious) / len(ious):.2f} su {len(ious)} blocchi di gruppo")

    doc.save(out_dir / f"{pdf.stem}_mistral.pdf")
    (out_dir / f"{pdf.stem}_mistral.json").write_text(json.dumps(esito, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRisultati in {out_dir}")


if __name__ == "__main__":
    main()
