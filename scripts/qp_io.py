import csv
import re
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def col_idx(ref):
    s = "".join(ch for ch in ref if ch.isalpha())
    n = 0
    for ch in s:
        n = n * 26 + (ord(ch.upper()) - 64)
    return n


def safe_float(s):
    s = norm(s)
    if not s:
        return None
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
        try:
            return float(s)
        except ValueError:
            return None
    return None


def split_code_text(raw):
    raw = norm(raw)
    if "^" in raw:
        a, b = raw.split("^", 1)
        return norm(a), norm(b)
    return raw, ""


def read_xlsx_first_sheet(path):
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rel_map = {x.attrib.get("Id"): x.attrib.get("Target") for x in rels}
        sh = wb.find("m:sheets/m:sheet", NS)
        rid = sh.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        target = rel_map[rid]
        sp = "xl/" + target.lstrip("/")

        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in ss.findall("m:si", NS):
                shared.append("".join(t.text or "" for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")))

        root = ET.fromstring(z.read(sp))
        rows = root.find("m:sheetData", NS).findall("m:row", NS)
        mat = []
        for r in rows:
            rec = {}
            for c in r.findall("m:c", NS):
                i = col_idx(c.attrib.get("r", "A1"))
                t = c.attrib.get("t")
                v = c.find("m:v", NS)
                isel = c.find("m:is", NS)
                val = ""
                if t == "s" and v is not None and v.text is not None:
                    k = int(v.text)
                    if 0 <= k < len(shared):
                        val = shared[k]
                elif t == "inlineStr" and isel is not None:
                    val = "".join(x.text or "" for x in isel.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"))
                elif v is not None and v.text is not None:
                    val = v.text
                rec[i] = norm(val)
            mat.append(rec)

    hdr = mat[0]
    ncol = max(hdr)
    headers = [norm(hdr.get(i, f"C{i:03d}")) for i in range(1, ncol + 1)]
    rows_dense = [[norm(r.get(i, "")) for i in range(1, ncol + 1)] for r in mat[1:]]
    return headers, rows_dense


def numeric_matrix(rows_dense):
    n = len(rows_dense)
    m = len(rows_dense[0]) if rows_dense else 0
    arr = np.full((n, m), np.nan, dtype=float)
    markers = []
    for i in range(n):
        for j in range(m):
            left, txt = split_code_text(rows_dense[i][j])
            v = safe_float(left)
            if v is not None:
                arr[i, j] = v
            if txt:
                markers.append((i + 1, j + 1, txt))
    return arr, markers


def write_dict_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_rows_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fmt(x):
    if isinstance(x, float):
        if np.isnan(x):
            return ""
        if abs(x - round(x)) < 1e-12:
            return str(int(round(x)))
        return f"{x:.6f}"
    return str(x)
