"""Box di regione da Gemini su Vertex AI (Gemini Enterprise Agent Platform).

Esempio di come un modello Gemini restituisce bounding box "a livello di
regione" per una pagina PDF, senza OCR locale. Gemini risponde con
box_2d = [ymin, xmin, ymax, xmax] normalizzate su 0-1000 rispetto
all'immagine; qui si riportano in punti PDF, si disegnano e, se nella
cartella di output c'è il risultato di bbox_gruppi.py, si confrontano con
i blocchi di gruppo calcolati dall'OCR (IoU per gruppo).

Tre modalità:
  --modo gruppi  (default, richiede campi.json): per ogni gruppo del JSON
                 (oggetto o lista) presente in pagina, la box che racchiude
                 tutti i suoi valori. È l'analogo delle box di gruppo di
                 bbox_gruppi.py, ma chiesto direttamente al modello.
  --modo foglie  (richiede campi.json): per ogni valore scalare del JSON
                 presente in pagina, la box stretta sul testo del valore.
                 È il test decisivo: se reggesse, l'OCR locale non servirebbe.
                 Confronto per IoU con le box di parola di bbox_da_json,
                 separando valori corti (numeri, date) e lunghi (frasi).
  --modo layout  : tutte le regioni della pagina (intestazione, paragrafi,
                 elenchi, tabelle, firme, piè di pagina) con un'etichetta.

Scrive in output:
  - <nome>_vertex_<modo>_<modello>.json : box per pagina in punti PDF, IoU
                                con le box OCR, tempi e token usati
  - <nome>_vertex_<modo>_<modello>.pdf  : box Gemini in blu; in verde
                                tratteggiato le box OCR corrispondenti

Uso: python bbox_vertex.py file.pdf [campi.json] [cartella_output]
        [--modo gruppi|foglie|layout] [--modello gemini-3.8-flash] [--location global]
        [--dpi 110] [--thinking minimal|low|high] [--credenziali auth/chiave.json]
Le credenziali sono un file JSON di service account; se non indicato si
prende il primo *.json nella cartella auth/.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pymupdf
from pydantic import BaseModel

from bbox_da_json import etichette_di_riga, righe_del_json, scrivi_etichetta  # noqa: F401 (righe_del_json riusato altrove)


class Regione(BaseModel):
    etichetta: str
    box_2d: list[int]  # [ymin, xmin, ymax, xmax] su scala 0-1000


class RegionePagina(BaseModel):
    """Regione di una chiamata sull'intero documento: porta anche la pagina."""
    etichetta: str
    pagina: int  # da 1
    box_2d: list[int]  # [ymin, xmin, ymax, xmax] su scala 0-1000 rispetto a quella pagina


ETICHETTE_LAYOUT = (
    "intestazione, mittente, destinatario, titolo, riferimenti, paragrafo, elenco, "
    "voce_elenco, tabella, riga_tabella, importo, firma, pie_pagina, timbro, altro"
)


# ── Credenziali e client ─────────────────────────────────────────────


def credenziali(percorso: str | None) -> tuple[str, str]:
    """Imposta GOOGLE_APPLICATION_CREDENTIALS e ritorna (file, project_id)."""
    if percorso is None:
        candidati = sorted(Path("auth").glob("*.json"))
        if not candidati:
            raise SystemExit("nessuna chiave in auth/: passa --credenziali file.json")
        percorso = str(candidati[0])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = percorso
    project = json.loads(Path(percorso).read_text(encoding="utf-8")).get("project_id")
    if not project:
        raise SystemExit(f"{percorso}: manca project_id")
    return percorso, project


def client_vertex(project: str, location: str):
    from google import genai

    return genai.Client(vertexai=True, project=project, location=location)


# ── Chiamata al modello ──────────────────────────────────────────────


def _config(modello: str, thinking: str, schema):
    from google.genai import types

    if modello.startswith("gemini-2.5"):
        pensiero = types.ThinkingConfig(thinking_budget=0 if thinking == "minimal" else 1024)
    else:
        pensiero = types.ThinkingConfig(thinking_level=thinking)
    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        thinking_config=pensiero,
        temperature=0.0,
        # risoluzione alta: le box su testo denso ne beneficiano
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    )


def _esegui(client, modello: str, contenuti: list, config, tipo):
    t0 = time.perf_counter()
    risposta = client.models.generate_content(model=modello, contents=contenuti, config=config)
    ms = round((time.perf_counter() - t0) * 1000)
    regioni = [tipo.model_validate(r) for r in json.loads(risposta.text)]
    u = risposta.usage_metadata
    meta = {
        "ms": ms,
        "token_prompt": getattr(u, "prompt_token_count", None),
        "token_risposta": getattr(u, "candidates_token_count", None),
        "token_pensiero": getattr(u, "thoughts_token_count", None),
    }
    return regioni, meta


def chiama(client, modello: str, png: bytes, prompt: str, thinking: str) -> tuple[list[Regione], dict]:
    """Una chiamata a Gemini con l'immagine di una pagina + prompt, risposta
    JSON tipizzata. Ritorna le regioni e i metadati (token, ms)."""
    from google.genai import types

    contenuti = [types.Part.from_bytes(data=png, mime_type="image/png"), prompt]
    return _esegui(client, modello, contenuti, _config(modello, thinking, list[Regione]), Regione)


def chiama_intero(client, modello: str, pagine_png: list[bytes] | None, pdf_bytes: bytes | None,
                  prompt: str, thinking: str) -> tuple[list[RegionePagina], dict]:
    """Una sola chiamata con tutto il documento: o le pagine come immagini,
    ognuna preceduta da "Pagina n", o il PDF così com'è (rendering a carico
    di Gemini). La risposta porta la pagina di ogni regione."""
    from google.genai import types

    contenuti: list = []
    if pdf_bytes is not None:
        contenuti.append(types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))
    else:
        for n, png in enumerate(pagine_png or [], start=1):
            contenuti.append(f"Pagina {n}:")
            contenuti.append(types.Part.from_bytes(data=png, mime_type="image/png"))
    contenuti.append(prompt)
    return _esegui(client, modello, contenuti, _config(modello, thinking, list[RegionePagina]), RegionePagina)


def prompt_gruppi(campi: dict, gruppi: list[str], n: int, tot: int) -> str:
    return (
        f"Questa è la pagina {n} di {tot} di un documento. Da questo documento è stato estratto il JSON seguente:\n\n"
        f"{json.dumps(campi, ensure_ascii=False, indent=1)}\n\n"
        "I gruppi del JSON (oggetti e liste) sono, per percorso:\n- " + "\n- ".join(gruppi) + "\n\n"
        "Per ogni gruppo i cui valori compaiono in QUESTA pagina restituisci una sola regione con "
        "etichetta = percorso del gruppo e box_2d = la box più stretta che racchiude tutti i valori "
        "di quel gruppo visibili in questa pagina (solo il testo dei valori, non le etichette del modulo). "
        "Ometti i gruppi che non compaiono in questa pagina. Non inventare regioni. "
        "box_2d è [ymin, xmin, ymax, xmax] con coordinate intere normalizzate su 0-1000 rispetto all'immagine."
    )


def prompt_foglie(foglie: list[tuple[str, str]], n: int | None, tot: int, permissivo: bool = False,
                  righe: dict | None = None) -> str:
    """Prompt della modalità foglie. Con n=None la richiesta riguarda l'intero
    documento in una chiamata sola e ogni regione deve indicare la pagina."""
    righe = righe or {}
    singoli, per_riga = [], {}
    for p, v in foglie:
        genitore, _, campo = p.rpartition(".")
        if genitore in righe:
            per_riga.setdefault(genitore, []).append((campo, v))
        else:
            singoli.append(f"- {p}: {v}")
    elenco = "Valori singoli (percorso nel JSON: valore):\n" + "\n".join(singoli)
    if per_riga:
        elenco += (
            "\n\nRighe di tabella. Le righe di dettaglio sono CONSECUTIVE nella tabella, una sotto l'altra, "
            "nello stesso ordine in cui sono elencate qui: la riga di dettaglio N è la N-esima riga di dettaglio "
            "del documento (se il documento ha più pagine, la numerazione continua dalla pagina precedente; usa il "
            "progressivo stampato e i codici univoci della riga per riconoscerla). Fai attenzione a NON mischiare "
            "le righe: i valori della riga N vanno cercati solo sulla N-esima riga, anche se lo stesso numero "
            "compare in altre righe. Il percorso di ogni campo è <percorso riga>.<campo>.\n"
        )
        for genitore, campi_riga in per_riga.items():
            tipo, num, _ = righe[genitore]
            nome = f"riga di dettaglio {num}" if tipo == "dettaglio" else f"riga articolo {num}"
            elenco += f"- {nome} ({genitore}): " + " | ".join(f"{c}={v}" for c, v in campi_riga) + "\n"
    if n is None:
        base = (
            f"Il documento allegato ha {tot} pagine, in ordine (pagina 1, 2, ...). Da questo documento sono stati "
            f"estratti i valori seguenti.\n\n{elenco}\n\n"
            "Per ogni valore che compare nel documento restituisci una regione con etichetta = percorso completo "
            "del valore, pagina = numero della pagina in cui compare (da 1) e box_2d = la box più stretta possibile "
            "attorno al solo testo del valore in QUELLA pagina (non l'etichetta del modulo, non il resto della riga). "
            "Se il valore va a capo la box copre tutte le sue righe. "
        )
    else:
        base = (
            f"Questa è la pagina {n} di {tot} di un documento. Da questo documento sono stati estratti i valori seguenti.\n\n"
            f"{elenco}\n\n"
            "Per ogni valore che compare in QUESTA pagina restituisci una regione con etichetta = percorso completo "
            "del valore e box_2d = la box più stretta possibile attorno al solo testo del valore (non l'etichetta del "
            "modulo, non il resto della riga). Se il valore va a capo la box copre tutte le sue righe. "
        )
    if permissivo:
        regole = (
            "Sii il più inclusivo possibile: conta come presente anche un valore scritto in forma leggermente "
            "diversa (parola flessa o al plurale come 'giornaliera' per 'giornalieri', abbreviata, con maiuscole "
            "diverse, data o numero in altro formato come 2012-12-06 per 06/12/2012 o 500.0 per 500,00), oppure "
            "contenuto in una riga o frase più lunga: in quel caso la box copre solo la parte che corrisponde. "
            "Se un valore compare più volte restituisci una regione per ogni occorrenza. "
            "Ometti un valore solo se sei certo che non ci sia nulla di simile" + (" nel documento. " if n is None else " in questa pagina. ")
        )
    else:
        regole = "Ometti i valori che non compaiono" + (" nel documento" if n is None else " in questa pagina") + "; non inventare regioni. "
    return base + regole + "box_2d è [ymin, xmin, ymax, xmax] con coordinate intere normalizzate su 0-1000 rispetto all'immagine."


def prompt_layout(n: int, tot: int) -> str:
    return (
        f"Questa è la pagina {n} di {tot} di un documento. Individua tutte le regioni della pagina: "
        "ogni blocco di testo, intestazione, blocco mittente/destinatario, titolo, elenco, tabella, firma, "
        "piè di pagina. Restituisci per ciascuna etichetta (una tra: " + ETICHETTE_LAYOUT + ") e "
        "box_2d = [ymin, xmin, ymax, xmax] con coordinate intere normalizzate su 0-1000 rispetto all'immagine. "
        "Le box devono essere strette sul contenuto e non sovrapporsi tra loro."
    )


# ── Geometria ────────────────────────────────────────────────────────


def gruppi_del_json(nodo, percorso: str = "") -> list[str]:
    """Percorsi di tutti i contenitori (esclusa la radice), come in bbox_gruppi."""
    out = []
    if isinstance(nodo, dict):
        items = [(k, v, f"{percorso}.{k}" if percorso else k) for k, v in nodo.items()]
    elif isinstance(nodo, list):
        items = [(i, v, f"{percorso}[{i}]") for i, v in enumerate(nodo)]
    else:
        return out
    for _, v, p in items:
        if isinstance(v, (dict, list)):
            out.append(p)
            out.extend(gruppi_del_json(v, p))
    return out


def foglie_del_json(nodo, percorso: str = "") -> list[tuple[str, str]]:
    """(percorso, valore) di ogni scalare non vuoto, stessi percorsi di bbox_da_json."""
    if isinstance(nodo, dict):
        return [f for k, v in nodo.items() for f in foglie_del_json(v, f"{percorso}.{k}" if percorso else str(k))]
    if isinstance(nodo, list):
        return [f for i, v in enumerate(nodo) for f in foglie_del_json(v, f"{percorso}[{i}]")]
    if nodo is None or str(nodo).strip() == "":
        return []
    return [(percorso, str(nodo))]


def esiti_foglie(albero, percorso: str = "") -> dict[str, dict]:
    """Appiattisce l'albero degli esiti di bbox_da_json in {percorso: esito}."""
    if isinstance(albero, dict) and "bbox_page" in albero:
        return {percorso: albero}
    out = {}
    if isinstance(albero, dict):
        for k, v in albero.items():
            out.update(esiti_foglie(v, f"{percorso}.{k}" if percorso else str(k)))
    elif isinstance(albero, list):
        for i, v in enumerate(albero):
            out.update(esiti_foglie(v, f"{percorso}[{i}]"))
    return out


def e_corto(valore: str) -> bool:
    """Valori corti: numeri, date, codici, nomi di una o due parole."""
    return len(valore.split()) <= 2


def box_in_punti(box_2d: list[int], page: pymupdf.Page, pix_w: int, pix_h: int, dpi: int) -> list[float]:
    """Da [ymin, xmin, ymax, xmax] su 0-1000 (spazio dell'immagine, già ruotata
    come get_pixmap) a [ymin, xmin, ymax, xmax] in punti PDF della pagina."""
    ymin, xmin, ymax, xmax = (max(0, min(1000, v)) for v in box_2d)
    scala = 72 / dpi
    r = pymupdf.Rect(xmin / 1000 * pix_w * scala, ymin / 1000 * pix_h * scala,
                     xmax / 1000 * pix_w * scala, ymax / 1000 * pix_h * scala)
    if page.rotation:
        r = (r * page.derotation_matrix).normalize()
    return [round(r.y0, 1), round(r.x0, 1), round(r.y1, 1), round(r.x1, 1)]


def iou(a: list[float], b: list[float]) -> float:
    inter_h = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_w = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_h * inter_w
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])  # noqa: E731
    unione = area(a) + area(b) - inter
    return round(inter / unione, 3) if unione > 0 else 0.0


def _centro_dentro(a: list[float], b: list[float]) -> bool:
    """Il centro di *a* cade dentro *b* ([ymin, xmin, ymax, xmax])."""
    cy, cx = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] <= cy <= b[2] and b[1] <= cx <= b[3]


def riepilogo_foglie(esito: dict, foglie_ocr: dict[str, dict], valori: dict[str, str]) -> None:
    """Per ogni foglia la migliore box Gemini sulla pagina dell'OCR; poi
    statistiche separate per valori corti e lunghi, e i casi di disaccordo."""
    migliore: dict[str, float] = {}
    centrata: dict[str, bool] = {}  # il centro di una box cade dentro l'altra: "punta alla parola giusta"
    pagine_gemini: dict[str, set] = {}
    for p in esito["pagine"]:
        for r in p["regioni"]:
            e = r["etichetta"]
            pagine_gemini.setdefault(e, set()).add(p["page"])
            if "iou_ocr" in r:
                if r["iou_ocr"] >= migliore.get(e, -1.0):
                    migliore[e] = r["iou_ocr"]
                    centrata[e] = _centro_dentro(r["bbox_page"], r["bbox_ocr"]) or _centro_dentro(r["bbox_ocr"], r["bbox_page"])
    trovate_ocr = {p for p, e in foglie_ocr.items() if e.get("page") is not None}
    non_ocr = {p for p, e in foglie_ocr.items() if e.get("page") is None}
    print("\nFOGLIE: box Gemini contro box di parola dell'OCR (per foglia la migliore sulla stessa pagina)")
    for nome, insieme in (("corte (<= 2 parole)", {p for p in trovate_ocr if e_corto(valori.get(p, ""))}),
                          ("lunghe", {p for p in trovate_ocr if not e_corto(valori.get(p, ""))})):
        if not insieme:
            continue
        ious = [migliore.get(p, 0.0) for p in insieme]
        soglie = f"> 0: {sum(1 for v in ious if v > 0)}, " + ", ".join(f">= {s}: {sum(1 for v in ious if v >= s)}" for s in (0.3, 0.5, 0.8))
        centrate = sum(1 for p in insieme if centrata.get(p))
        mancanti = sum(1 for p in insieme if p not in migliore)
        print(f"  {nome:20s} {len(insieme):2d} foglie: IoU medio {sum(ious) / len(ious):.2f}, {soglie}, "
              f"centro nella parola giusta: {centrate}, senza box Gemini sulla pagina OCR: {mancanti}")
    peggiori = sorted(((migliore.get(p, 0.0), p) for p in trovate_ocr), key=lambda t: t[0])[:8]
    print("  peggiori:", ", ".join(f"{p} {v:.2f}" for v, p in peggiori))
    for p in sorted(non_ocr):
        dove = pagine_gemini.get(p)
        print(f"  {p}: l'OCR non lo trova; Gemini " + (f"lo mette a pagina {sorted(dove)} (inventato?)" if dove else "lo omette (concorde)"))
    for p in sorted(trovate_ocr):
        dove = pagine_gemini.get(p, set())
        if dove and foglie_ocr[p]["page"] not in dove:
            print(f"  {p}: OCR a p{foglie_ocr[p]['page']}, Gemini solo a p{sorted(dove)}")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("json_campi", nargs="?", default=None)
    ap.add_argument("out_dir", nargs="?", default=None)
    ap.add_argument("--modo", default="gruppi", choices=["gruppi", "foglie", "layout"])
    ap.add_argument("--modello", default="gemini-3.8-flash")
    ap.add_argument("--location", default="global")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--thinking", default="low", choices=["minimal", "low", "high"])
    ap.add_argument("--permissivo", action="store_true",
                    help="modo foglie: chiede una box anche per forme flesse, formati diversi e ogni occorrenza")
    ap.add_argument("--intero", action="store_true",
                    help="modo foglie: una sola chiamata con tutte le pagine, Gemini indica la pagina di ogni box")
    ap.add_argument("--pdf-nativo", action="store_true",
                    help="con --intero: manda il PDF così com'è invece delle pagine rese come immagini")
    ap.add_argument("--pagine-doc", default=None,
                    help='con --intero: una chiamata per documento (elemento della lista più esterna) con le sue '
                         'sole pagine e foglie, es. "0:1;1:2,3,4" = documenti[0] a pagina 1, [1] alle 2-4')
    ap.add_argument("--credenziali", default=None)
    args = ap.parse_args()

    pdf = Path(args.pdf)
    out_dir = Path(args.out_dir) if args.out_dir else pdf.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    campi = json.loads(Path(args.json_campi).read_text(encoding="utf-8")) if args.json_campi else None
    if args.modo in ("gruppi", "foglie") and campi is None:
        raise SystemExit(f"--modo {args.modo} richiede campi.json")
    gruppi = gruppi_del_json(campi) if campi is not None else []
    foglie = foglie_del_json(campi) if campi is not None else []
    valori = dict(foglie)
    righe = righe_del_json(campi) if campi is not None else {}

    file_chiave, project = credenziali(args.credenziali)
    client = client_vertex(project, args.location)
    print(f"Vertex AI: progetto {project}, location {args.location}, modello {args.modello}, chiave {Path(file_chiave).name}")

    # blocchi e foglie OCR di bbox_gruppi.py (o foglie di bbox_da_json), se ci sono, per il confronto
    blocchi_ocr: dict[str, list[dict]] = {}
    foglie_ocr: dict[str, dict] = {}
    f_gruppi = out_dir / f"{pdf.stem}_gruppi_bbox.json"
    f_foglie = out_dir / f"{pdf.stem}_bbox.json"
    if args.modo == "gruppi" and f_gruppi.exists():
        for percorso, agg in json.loads(f_gruppi.read_text(encoding="utf-8"))["gruppi"].items():
            if percorso:
                blocchi_ocr[percorso] = agg["blocchi"]
        print(f"confronto con i blocchi OCR di {f_gruppi.name}")
    if args.modo == "foglie":
        if f_gruppi.exists():
            foglie_ocr = esiti_foglie(json.loads(f_gruppi.read_text(encoding="utf-8"))["campi"])
            print(f"confronto con le foglie OCR di {f_gruppi.name}")
        elif f_foglie.exists():
            foglie_ocr = esiti_foglie(json.loads(f_foglie.read_text(encoding="utf-8")))
            print(f"confronto con le foglie OCR di {f_foglie.name}")

    doc = pymupdf.open(pdf)
    intero = args.intero and args.modo == "foglie"
    nome_modo = args.modo + ("-permissivo" if args.permissivo and args.modo == "foglie" else "")
    if intero:
        nome_modo += "-perdoc" if args.pagine_doc else ("-pdf" if args.pdf_nativo else "-intero")
    pagine_doc: dict[int, list[int]] = {}
    if args.pagine_doc:
        for parte in args.pagine_doc.split(";"):
            i, pg = parte.split(":")
            pagine_doc[int(i)] = sorted(int(x) for x in pg.split(","))
    esito = {"modello": args.modello, "location": args.location, "modo": nome_modo, "dpi": args.dpi,
             "thinking": args.thinking, "pagine": []}
    blu, verde = (0.1, 0.3, 0.9), (0, 0.6, 0)

    # dimensioni dell'immagine di ogni pagina, per riportare le box in punti:
    # con il PDF nativo Gemini rende da sé, e conta solo il rapporto d'aspetto
    pix_pagine = {}
    png_pagine = {}
    for n, page in enumerate(doc, start=1):
        if intero and args.pdf_nativo:
            pix_pagine[n] = (page.rect.width, page.rect.height, 72)
        else:
            pix = page.get_pixmap(dpi=args.dpi, alpha=False)
            pix_pagine[n] = (pix.width, pix.height, args.dpi)
            png_pagine[n] = pix.tobytes("png")

    regioni_intero: dict[int, list] = {}
    meta_intero = None
    if intero and pagine_doc:
        # una chiamata per documento: le sue pagine (con il numero vero) e le sue foglie
        meta_intero = {"ms": 0, "token_prompt": 0, "token_risposta": 0, "token_pensiero": 0}
        lista = re.match(r"([^\[]*)\[\d+\]", foglie[0][0]).group(1)  # nome della lista più esterna
        for i, pgs in pagine_doc.items():
            prefisso = f"{lista}[{i}]"
            foglie_doc = [(p, v) for p, v in foglie if p.startswith(prefisso + ".") or p == prefisso]
            prompt = prompt_foglie(foglie_doc, None, len(pgs), args.permissivo, righe).replace(
                f"Il documento allegato ha {len(pgs)} pagine, in ordine (pagina 1, 2, ...).",
                f"Il documento allegato occupa le pagine {', '.join(map(str, pgs))} del PDF, allegate in ordine e "
                f"ognuna preceduta dal suo numero: usa quel numero come 'pagina'.")
            from google.genai import types
            contenuti: list = []
            for n in pgs:
                contenuti += [f"Pagina {n}:", types.Part.from_bytes(data=png_pagine[n], mime_type="image/png")]
            contenuti.append(prompt)
            regioni_doc, meta = _esegui(client, args.modello, contenuti,
                                        _config(args.modello, args.thinking, list[RegionePagina]), RegionePagina)
            for r in regioni_doc:
                if r.pagina in pgs:
                    regioni_intero.setdefault(r.pagina, []).append(Regione(etichetta=r.etichetta, box_2d=r.box_2d))
            for chiave in meta_intero:
                meta_intero[chiave] += meta[chiave] or 0
            print(f"\ndocumento {prefisso} (pagine {pgs}, {len(foglie_doc)} foglie): {len(regioni_doc)} regioni in {meta['ms']} ms "
                  f"(token prompt {meta['token_prompt']}, risposta {meta['token_risposta']})")
        esito["chiamata_intera"] = meta_intero
    elif intero:
        prompt = prompt_foglie(foglie, None, len(doc), args.permissivo, righe)
        regioni_doc, meta_intero = chiama_intero(
            client, args.modello,
            None if args.pdf_nativo else [png_pagine[n] for n in range(1, len(doc) + 1)],
            pdf.read_bytes() if args.pdf_nativo else None, prompt, args.thinking)
        fuori = [r for r in regioni_doc if not 1 <= r.pagina <= len(doc)]
        for r in regioni_doc:
            if 1 <= r.pagina <= len(doc):
                regioni_intero.setdefault(r.pagina, []).append(Regione(etichetta=r.etichetta, box_2d=r.box_2d))
        print(f"\ndocumento intero ({'PDF nativo' if args.pdf_nativo else f'{len(doc)} immagini'}): {len(regioni_doc)} regioni "
              f"in {meta_intero['ms']} ms (token prompt {meta_intero['token_prompt']}, risposta {meta_intero['token_risposta']}, "
              f"pensiero {meta_intero['token_pensiero']}); con pagina fuori intervallo: {len(fuori)}")
        esito["chiamata_intera"] = meta_intero

    for n, page in enumerate(doc, start=1):
        pix_w, pix_h, dpi_pag = pix_pagine[n]
        if intero:
            regioni = regioni_intero.get(n, [])
            meta = {"ms": 0, "token_prompt": 0, "token_risposta": 0, "token_pensiero": None}
            print(f"\npagina {n}: {len(regioni)} regioni (dalla chiamata intera)")
        else:
            png = png_pagine[n]
            if args.modo == "gruppi":
                prompt = prompt_gruppi(campi, gruppi, n, len(doc))
            elif args.modo == "foglie":
                prompt = prompt_foglie(foglie, n, len(doc), args.permissivo, righe)
            else:
                prompt = prompt_layout(n, len(doc))
            regioni, meta = chiama(client, args.modello, png, prompt, args.thinking)
            print(f"\npagina {n}: {len(regioni)} regioni in {meta['ms']} ms "
                  f"(token prompt {meta['token_prompt']}, risposta {meta['token_risposta']}, pensiero {meta['token_pensiero']})")
        voci = []
        shape = page.new_shape()
        for r in regioni:
            if len(r.box_2d) != 4:
                continue
            bbox = box_in_punti(r.box_2d, page, pix_w, pix_h, dpi_pag)
            voce = {"etichetta": r.etichetta, "box_2d": r.box_2d, "bbox_page": bbox}
            y0, x0, y1, x1 = bbox
            shape.draw_rect(pymupdf.Rect(x0, y0, x1, y1))
            riga = f"  {r.etichetta:32s} {bbox}"
            if r.etichetta in blocchi_ocr:
                # IoU col blocco OCR di questa pagina che gli somiglia di più
                candidati = [b for b in blocchi_ocr[r.etichetta] if b["page"] == n]
                if candidati:
                    migliore = max(candidati, key=lambda b: iou(bbox, b["bbox_page"]))
                    voce["iou_ocr"] = iou(bbox, migliore["bbox_page"])
                    voce["bbox_ocr"] = migliore["bbox_page"]
                    riga += f"  IoU con blocco OCR {voce['iou_ocr']:.2f}"
                    by0, bx0, by1, bx1 = migliore["bbox_page"]
                    s2 = page.new_shape()
                    s2.draw_rect(pymupdf.Rect(bx0, by0, bx1, by1))
                    s2.finish(color=verde, width=0.6, dashes="[3 2] 0")
                    s2.commit()
                else:
                    riga += "  (l'OCR non ha questo gruppo in pagina)"
            elif args.modo == "foglie":
                if r.etichetta not in valori:
                    riga += "  (etichetta non tra le foglie: scartare)"
                else:
                    voce["valore"] = valori[r.etichetta]
                    voce["corto"] = e_corto(valori[r.etichetta])
                    eo = foglie_ocr.get(r.etichetta)
                    if eo and eo.get("page") == n:
                        voce["iou_ocr"] = iou(bbox, eo["bbox_page"])
                        voce["bbox_ocr"] = eo["bbox_page"]
                        riga += f"  IoU con foglia OCR {voce['iou_ocr']:.2f}"
                        by0, bx0, by1, bx1 = eo["bbox_page"]
                        s2 = page.new_shape()
                        s2.draw_rect(pymupdf.Rect(bx0, by0, bx1, by1))
                        s2.finish(color=verde, width=0.6, dashes="[3 2] 0")
                        s2.commit()
                    elif eo and eo.get("page") is None:
                        riga += "  (l'OCR dice: NON in pagina)"
                    elif eo:
                        riga += f"  (l'OCR lo mette a p{eo['page']})"
            elif r.etichetta not in gruppi and args.modo == "gruppi":
                riga += "  (etichetta non tra i gruppi: scartare)"
            print(riga)
            if args.modo == "layout":
                scrivi_etichetta(page, bbox, r.etichetta, blu, fontsize=5)
            voci.append(voce)
        shape.finish(color=blu, width=0.8)
        shape.commit()
        if args.modo in ("foglie", "gruppi"):
            # nessun nome di campo sul PDF: solo il numero di riga, una volta per riga di tabella
            if args.modo == "foglie":
                candidati = [(v["etichetta"], n, v["bbox_page"]) for v in voci]
            else:
                candidati = [(f"{v['etichetta']}.x", n, v["bbox_page"]) for v in voci if v["etichetta"] in righe]
            for _, bbox_r, testo in etichette_di_riga(candidati, righe, doc):
                scrivi_etichetta(page, bbox_r, testo, blu)
        esito["pagine"].append({"page": n, "larghezza_px": pix_w, "altezza_px": pix_h, **meta, "regioni": voci})

    doc.save(out_dir / f"{pdf.stem}_vertex_{nome_modo}_{args.modello}.pdf")
    (out_dir / f"{pdf.stem}_vertex_{nome_modo}_{args.modello}.json").write_text(json.dumps(esito, ensure_ascii=False, indent=2), encoding="utf-8")
    tot_ms = meta_intero["ms"] if meta_intero else sum(p["ms"] for p in esito["pagine"])
    with_iou = [r["iou_ocr"] for p in esito["pagine"] for r in p["regioni"] if "iou_ocr" in r]
    riassunto = f"\n{len(doc)} pagine, {tot_ms} ms totali di chiamate"
    if with_iou and args.modo == "gruppi":
        riassunto += f", IoU medio con i blocchi OCR {sum(with_iou) / len(with_iou):.2f} su {len(with_iou)} gruppi"
    print(riassunto)
    if args.modo == "foglie" and foglie_ocr:
        riepilogo_foglie(esito, foglie_ocr, valori)
    print(f"Risultati in {out_dir}")


if __name__ == "__main__":
    main()
