"""Reading the Morning report's positions table, which is an image inside the PDF (owner decision
2026-11-28: OCR plus a check).

RapidOCR (rapidocr-onnxruntime, offline ONNX models) reads the embedded images of the first pages; the
image whose text holds the header words `Currency` and `Gain/Loss` is the positions table. Words are
grouped into table rows by their vertical centre and ordered left to right. `parse.morning_rows` then
checks every row arithmetically, so an OCR misread is flagged rather than shown.

The OCR engine is injectable (`engine=`) so tests never load it.
"""

from __future__ import annotations

import io

ROW_TOLERANCE_PX = 8


def _engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def group_rows(words: list) -> list:
    """[(x, y_centre, text)] -> rows of texts, top to bottom, each left to right."""
    rows = []
    for x, y, t in sorted(words, key=lambda w: w[1]):
        if rows and abs(rows[-1]['y'] - y) <= ROW_TOLERANCE_PX:
            rows[-1]['w'].append((x, t))
            rows[-1]['y'] = (rows[-1]['y'] + y) / 2
        else:
            rows.append({'y': y, 'w': [(x, t)]})
    return [[t for _, t in sorted(r['w'])] for r in rows]


def words_of(result) -> list:
    out = []
    for box, text, _conf in result or []:
        ys = [p[1] for p in box]
        out.append((float(box[0][0]), (min(ys) + max(ys)) / 2, str(text)))
    return out


def read_positions_table(pdf_bytes: bytes, pages: int = 3, engine=None) -> list:
    """The table's rows of cell texts, or [] when no image holds it."""
    from pypdf import PdfReader
    engine = engine or _engine()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    for page in reader.pages[:pages]:
        for im in page.images:
            if len(im.data) < 10_000:          # logos and icons
                continue
            result, _ = engine(im.data)
            rows = group_rows(words_of(result))
            flat = ' '.join(' '.join(r) for r in rows)
            if 'Currency' in flat and 'Gain/Loss' in flat:
                return rows
    return []
