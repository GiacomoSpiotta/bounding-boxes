"""Estrazione di un JSON "tipo" da un PDF con Gemini su Vertex AI.

Serve a produrre l'input per bbox_da_json / bbox_gruppi / bbox_vertex quando
non si ha già un JSON estratto: manda tutte le pagine come immagini e chiede
un JSON con la struttura di un documento di trasporto (DDT). I valori vanno
copiati alla lettera dalla pagina, come stringhe, così l'allineamento OCR
li ritrova. Un PDF può contenere più DDT: la radice è {"documenti": [...]}.

Aggiunge a ogni documento un campo "controllo_assente" con una frase che NON
è nel documento: serve a verificare che chi localizza non inventi box.

Uso: python estrai_json.py file.pdf [uscita.json] [--tipo tintoria|siderurgia]
        [--modello gemini-3.8-flash] [--location global] [--dpi 110] [--credenziali auth/chiave.json]
Senza uscita scrive esempio_<nome>.json nella cartella corrente.
--tipo sceglie lo schema: "tintoria" (DDT di lavorazione con righe pezza),
"siderurgia" (DDT con righe ordine / posizione / marca trave / pezzi / peso).
"""

import argparse
import json
import time
from pathlib import Path

import pymupdf

from bbox_vertex import client_vertex, credenziali

CONTROLLO_ASSENTE = "Consegna prevista entro trenta giorni lavorativi presso il magazzino di Torino"

SCHEMA = """{
  "documenti": [
    {
      "mittente": {"ragione_sociale": "", "indirizzo": "", "cod_fiscale_piva": ""},
      "per_conto_di": {"codice": "", "ragione_sociale": "", "indirizzo": "", "piva": ""},
      "destinatario": {"ragione_sociale": "", "indirizzo": ""},
      "ddt": {"numero": "", "data": "", "causale": "", "trasportatore": "", "trasporto_a_cura": "",
              "data_ritiro": "", "ora_ritiro": "", "aspetto_beni": "", "termini_consegna": ""},
      "articoli": [
        {"articolo": "", "order_nr": "", "ord_cliente": "", "disegno": "", "composizione": "",
         "pezze": [
           {"n": "", "pack_nr": "", "distinta": "", "colore": "", "pezza": "",
            "mtr_gr": "", "kg_gr": "", "mtr_fin": "", "kg_fin": "", "peso_lordo": ""}
         ],
         "subtotale": {"pezze": "", "mtr_gr": "", "kg_gr": "", "mtr_fin": "", "kg_fin": ""}}
      ],
      "totali": {"colli": "", "pezze": "", "mtr_gr": "", "kg_gr": "", "mtr_fin": "", "kg_fin": "",
                 "peso_lordo": "", "kg_netti": "", "kg_lordi": ""}
    }
  ]
}"""

REGOLE_TINTORIA = (
    "- 'articoli': una voce per ogni codice articolo (es. CS003120-001 /02) con i suoi dati di riga "
    "(ORDER Nr., ORD.CLIENTE, DISEGNO, COMPOSIZIONE) e in 'pezze' una voce per ogni riga di pezza sotto di esso "
    "(n progressivo, PACK NR., DISTINTA, COLORE, numero pezza, MTR GR., KG GR., MTR FIN, KG FIN, PESO LORDO). "
    "Se una colonna è vuota in pagina lascia la stringa vuota.\n"
    "- 'subtotale' e 'totali' come stampati nelle righe Subtotale e TOTALI; 'kg_netti' e 'kg_lordi' dalle "
    "voci 'KG netti tot' e 'KG lordi tot'.\n"
)

SCHEMA_SIDERURGIA = """{
  "documenti": [
    {
      "mittente": {"ragione_sociale": "", "sede_legale": "", "partita_iva": "", "stabilimento": ""},
      "destinatario": {"ragione_sociale": "", "indirizzo": "", "piva": ""},
      "documento": {"tipo": "", "numero": "", "data": "", "cli_for": "", "causale": ""},
      "riferimento_ordine": {"numero": "", "data": ""},
      "cura_trasporto": "", "aspetto_beni": "", "mg": "",
      "vettore": {"ragione_sociale": "", "indirizzo": "", "targa": "", "piva": ""},
      "ordine_interno": "", "targa_automezzo": "", "data_ora_ritiro": "", "ora_ritiro": "",
      "articoli": [
        {"articolo": "", "descrizione": "", "quantita": "", "um": "",
         "righe": [
           {"ordine": "", "pos_ordine": "", "marca_trave": "", "n_pezzi": "", "peso": ""}
         ],
         "totali": {"n_pezzi": "", "peso": ""}}
      ],
      "destinazione_merce": {"ragione_sociale": "", "indirizzo": ""},
      "quantita_totale": ""
    }
  ]
}"""

REGOLE_SIDERURGIA = (
    "- Di norma il PDF contiene un solo DDT su più pagine (Pag 1, 2, ...): l'intestazione si ripete uguale su "
    "ogni pagina e le righe della tabella continuano; è un DDT nuovo solo se cambia il Numero.\n"
    "- 'mittente' è l'azienda dell'intestazione (sede legale, partita IVA, stabilimento); 'destinatario' è lo "
    "'Spett/le' in alto a destra; 'vettore' con indirizzo, targa/sigla (es. CT/8708039/W) e partita IVA.\n"
    "- 'articoli': una voce per ogni riga articolo (es. TS6322 con descrizione, quantità e Um); in 'righe' una voce "
    "per OGNI riga della tabella sottostante, nell'ordine in cui compaiono pagina dopo pagina, con Ordine, "
    "Pos.Ordine, Marca trave, N.pezzi e Peso; se Ordine o Pos.Ordine non sono stampati su quella riga lascia la "
    "stringa vuota, non ripetere quelli della riga sopra.\n"
    "- 'totali' dalla riga 'Totali .....'; 'quantita_totale' dal riquadro 'Quantità Totale'.\n"
)

SCHEMI = {"tintoria": (SCHEMA, REGOLE_TINTORIA), "siderurgia": (SCHEMA_SIDERURGIA, REGOLE_SIDERURGIA)}


def prompt_estrazione(tipo: str) -> str:
    schema, regole = SCHEMI[tipo]
    return (
        "Le immagini sono le pagine, in ordine, di un PDF che contiene uno o più documenti di trasporto (DDT). "
        "Estrai TUTTI i DDT in un JSON con esattamente questa struttura:\n\n"
        + schema + "\n\n"
        "Regole:\n"
        "- Copia i valori alla lettera come stampati (stessi separatori decimali, stesse date, stesse maiuscole), "
        "sempre come stringhe. Non calcolare nulla, non normalizzare.\n"
        "- Indirizzi: le righe dell'indirizzo unite da uno spazio.\n"
        + regole +
        "- Non inventare valori: se un dato non c'è, stringa vuota.\n"
        "Rispondi solo con il JSON."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("uscita", nargs="?", default=None)
    ap.add_argument("--tipo", default="tintoria", choices=sorted(SCHEMI))
    ap.add_argument("--modello", default="gemini-3.8-flash")
    ap.add_argument("--location", default="global")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--credenziali", default=None)
    args = ap.parse_args()

    from google.genai import types

    pdf = Path(args.pdf)
    uscita = Path(args.uscita) if args.uscita else Path(f"esempio_{pdf.stem}.json")
    _, project = credenziali(args.credenziali)
    client = client_vertex(project, args.location)

    doc = pymupdf.open(pdf)
    contenuti = []
    for page in doc:
        png = page.get_pixmap(dpi=args.dpi, alpha=False).tobytes("png")
        contenuti.append(types.Part.from_bytes(data=png, mime_type="image/png"))
    contenuti.append(prompt_estrazione(args.tipo))
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_level="low"),
        temperature=0.0,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    )
    t0 = time.perf_counter()
    r = client.models.generate_content(model=args.modello, contents=contenuti, config=config)
    ms = round((time.perf_counter() - t0) * 1000)
    dati = json.loads(r.text)
    if not isinstance(dati, dict) or "documenti" not in dati:
        dati = {"documenti": dati if isinstance(dati, list) else [dati]}
    for d in dati["documenti"]:
        d["controllo_assente"] = CONTROLLO_ASSENTE

    uscita.write_text(json.dumps(dati, ensure_ascii=False, indent=2), encoding="utf-8")
    u = r.usage_metadata
    n_doc = len(dati["documenti"])
    n_pezze = sum(len(a.get("pezze", a.get("righe", []))) for d in dati["documenti"] for a in d.get("articoli", []))
    print(f"{pdf.name}: {len(doc)} pagine -> {n_doc} DDT, {n_pezze} righe di dettaglio, {ms} ms, "
          f"token prompt {u.prompt_token_count}, risposta {u.candidates_token_count}")
    print(f"scritto {uscita}")


if __name__ == "__main__":
    main()
