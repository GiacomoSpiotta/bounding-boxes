"""Prova di PP-OCRv6 su una cartella di PDF.

Per ogni PDF: rasterizza ogni pagina, rileva le righe di testo e scrive in output
  - <nome>_boxes.pdf : il PDF originale con le box disegnate in rosso
  - <nome>_boxes.txt : una riga per box, con pagina, punteggio, box in punti PDF e testo

Uso: python test_ppocrv6.py cartella_input [cartella_output] [tier]
     tier = medium (default) | small | tiny
"""

import sys
import tempfile
from pathlib import Path

import pymupdf as fitz
from paddleocr import PaddleOCR

DPI = 300
SCALA = 72 / DPI  # da pixel a punti PDF


def carica_ocr(tier: str) -> PaddleOCR:
    return PaddleOCR(
        text_detection_model_name=f"PP-OCRv6_{tier}_det",
        text_recognition_model_name=f"PP-OCRv6_{tier}_rec",
        use_doc_orientation_classify=False,  # niente raddrizzamento pagina
        use_doc_unwarping=False,             # niente correzione curvatura
        use_textline_orientation=False,      # le righe dei DDT sono orizzontali
    )


def ocr_pagina(ocr: PaddleOCR, page: fitz.Page, tmp_dir: Path) -> list[tuple[str, float, list]]:
    """Restituisce (testo, punteggio, poligono in punti PDF) per ogni riga trovata."""
    png = tmp_dir / "pagina.png"
    page.get_pixmap(dpi=DPI).save(png)
    res = ocr.predict(input=str(png))[0]
    righe = []
    for testo, score, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):
        poly_pt = [(float(x) * SCALA, float(y) * SCALA) for x, y in poly]
        righe.append((testo, float(score), poly_pt))
    return righe


def disegna(page: fitz.Page, poligoni: list[list]) -> None:
    shape = page.new_shape()
    for poly in poligoni:
        shape.draw_polyline(poly + [poly[0]])  # chiude il poligono
    shape.finish(color=(1, 0, 0), width=0.7)
    shape.commit()


def elabora_pdf(ocr: PaddleOCR, pdf_path: Path, out_dir: Path, tmp_dir: Path) -> int:
    doc = fitz.open(pdf_path)
    righe_txt = []
    totale = 0
    for n, page in enumerate(doc, start=1):
        righe = ocr_pagina(ocr, page, tmp_dir)
        totale += len(righe)
        disegna(page, [poly for _, _, poly in righe])
        for testo, score, poly in righe:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            bbox = [round(v, 1) for v in (min(ys), min(xs), max(ys), max(xs))]  # [ymin, xmin, ymax, xmax] come il painter
            righe_txt.append(f"p{n}  {score:.2f}  {bbox}  {testo!r}")
    doc.save(out_dir / f"{pdf_path.stem}_boxes.pdf")
    (out_dir / f"{pdf_path.stem}_boxes.txt").write_text("\n".join(righe_txt), encoding="utf-8")
    return totale


def main() -> None:
    in_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else in_dir / "output"
    tier = sys.argv[3] if len(sys.argv) > 3 else "medium"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(p for p in in_dir.iterdir() if p.suffix.lower() == ".pdf")
    if not pdfs:
        print(f"Nessun PDF in {in_dir}")
        return

    ocr = carica_ocr(tier)
    with tempfile.TemporaryDirectory() as tmp:
        for pdf in pdfs:
            n = elabora_pdf(ocr, pdf, out_dir, Path(tmp))
            print(f"{pdf.name}: {n} righe")
    print(f"\nRisultati in {out_dir}")


if __name__ == "__main__":
    main()