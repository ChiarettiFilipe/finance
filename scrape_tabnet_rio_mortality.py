#!/usr/bin/env python3
"""Scrape Rio TabNet mortality data (SIM, 2006+) into a normalized CSV.

Output columns:
- neighborhood_code
- neighborhood
- cause
- one column per age range (as returned by TabNet)

This script uses only Python's standard library.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

TABNET_URL = "https://tabnet.rio.rj.gov.br/cgi-bin/dh?sim/definicoes/sim_apos2005.def"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class SelectControl:
    name: str
    options: List[Tuple[str, str]] = field(default_factory=list)  # (value, label)
    selected: Optional[str] = None


@dataclass
class FormSpec:
    action: str
    method: str
    hidden_inputs: Dict[str, str]
    selects: Dict[str, SelectControl]
    label_to_select_name: Dict[str, str]


class FormParser(HTMLParser):
    """Parses the first form and associates row labels to select names."""

    def __init__(self) -> None:
        super().__init__()
        self.in_form = False
        self.form_action = ""
        self.form_method = "post"
        self.hidden_inputs: Dict[str, str] = {}
        self.selects: Dict[str, SelectControl] = {}

        self._in_td = False
        self._current_td_text: List[str] = []
        self._last_td_label = ""

        self._in_select = False
        self._current_select_name: Optional[str] = None
        self._current_option_value: Optional[str] = None
        self._current_option_text: List[str] = []

        self.label_to_select_name: Dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr = {k.lower(): (v or "") for k, v in attrs}

        if tag == "form" and not self.in_form:
            self.in_form = True
            self.form_action = attr.get("action", "")
            self.form_method = attr.get("method", "post").lower()
            return

        if not self.in_form:
            return

        if tag == "td":
            self._in_td = True
            self._current_td_text = []

        elif tag == "input" and attr.get("type", "").lower() == "hidden":
            name = attr.get("name", "")
            if name:
                self.hidden_inputs[name] = attr.get("value", "")

        elif tag == "select":
            name = attr.get("name", "")
            if name:
                self._in_select = True
                self._current_select_name = name
                self.selects.setdefault(name, SelectControl(name=name))
                if self._last_td_label and self._last_td_label not in self.label_to_select_name:
                    self.label_to_select_name[self._last_td_label] = name

        elif tag == "option" and self._in_select and self._current_select_name:
            self._current_option_value = attr.get("value", "")
            self._current_option_text = []
            if "selected" in attr:
                self.selects[self._current_select_name].selected = self._current_option_value

    def handle_data(self, data: str) -> None:
        if self._in_td:
            self._current_td_text.append(data)
        if self._current_option_value is not None:
            self._current_option_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self.in_form:
            self.in_form = False
            return

        if not self.in_form:
            return

        if tag == "td" and self._in_td:
            self._in_td = False
            td_text = clean_text("".join(self._current_td_text))
            if td_text:
                self._last_td_label = td_text

        elif tag == "option" and self._current_option_value is not None and self._current_select_name:
            label = clean_text("".join(self._current_option_text))
            self.selects[self._current_select_name].options.append((self._current_option_value, label))
            self._current_option_value = None
            self._current_option_text = []

        elif tag == "select":
            self._in_select = False
            self._current_select_name = None


def clean_text(raw: str) -> str:
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize(text: str) -> str:
    stripped = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in stripped if not unicodedata.combining(ch))
    stripped = stripped.lower().strip()
    return re.sub(r"\s+", " ", stripped)


def fetch(url: str, data: Optional[Dict[str, str]] = None, timeout: int = 60) -> str:
    encoded = None
    if data is not None:
        encoded = urlencode(data, doseq=True).encode("latin-1", errors="ignore")
    req = Request(url, data=encoded, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()

    for enc in ("latin-1", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def parse_form(page_html: str, base_url: str) -> FormSpec:
    parser = FormParser()
    parser.feed(page_html)
    if not parser.form_action:
        raise RuntimeError("Could not locate TabNet form in the response.")

    action_url = urljoin(base_url, parser.form_action)
    return FormSpec(
        action=action_url,
        method=parser.form_method,
        hidden_inputs=parser.hidden_inputs,
        selects=parser.selects,
        label_to_select_name=parser.label_to_select_name,
    )


def find_select_by_label(form: FormSpec, label_hint: str) -> str:
    target = normalize(label_hint)

    for label, name in form.label_to_select_name.items():
        if target in normalize(label):
            return name

    for name in form.selects:
        if target in normalize(name):
            return name

    available = ", ".join(sorted(form.selects.keys()))
    raise KeyError(f"Select not found for label hint '{label_hint}'. Available: {available}")




def find_first_select_by_labels(form: FormSpec, label_hints: List[str]) -> str:
    """Return first matching select name from a list of label hints."""
    last_error: Optional[Exception] = None
    for hint in label_hints:
        try:
            return find_select_by_label(form, hint)
        except Exception as exc:  # try next hint
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise KeyError("No label hints provided.")

def option_value_by_hint(select: SelectControl, hint: str) -> str:
    target = normalize(hint)

    for value, label in select.options:
        if target in normalize(label):
            return value
        if target in normalize(value):
            return value

    raise KeyError(f"Option '{hint}' not found in select '{select.name}'.")


def build_base_payload(form: FormSpec) -> Dict[str, str]:
    payload: Dict[str, str] = dict(form.hidden_inputs)
    for name, sel in form.selects.items():
        if sel.selected is not None:
            payload[name] = sel.selected
        elif sel.options:
            payload[name] = sel.options[0][0]
    return payload


def extract_tables(html_text: str) -> List[List[List[str]]]:
    tables: List[List[List[str]]] = []
    for table_html in re.findall(r"<table[^>]*>.*?</table>", html_text, flags=re.I | re.S):
        rows = []
        for tr in re.findall(r"<tr[^>]*>.*?</tr>", table_html, flags=re.I | re.S):
            cells = re.findall(r"<(?:td|th)[^>]*>(.*?)</(?:td|th)>", tr, flags=re.I | re.S)
            row = [clean_text(cell) for cell in cells]
            if row:
                rows.append(row)
        if rows:
            tables.append(rows)
    return tables


def parse_tabnet_matrix(result_html: str) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    tables = extract_tables(result_html)
    if not tables:
        raise RuntimeError("No tables found in TabNet response.")

    matrix = max(tables, key=lambda t: sum(len(r) for r in t))
    header = matrix[0]
    if len(header) < 2:
        raise RuntimeError("Unexpected table format: missing age-range columns.")

    age_ranges = header[1:]
    body_rows: List[Tuple[str, List[str]]] = []

    for row in matrix[1:]:
        if len(row) < 2:
            continue
        bairro = row[0]
        if normalize(bairro) in {"total", "ignorado"}:
            continue
        values = row[1:]
        if len(values) < len(age_ranges):
            values.extend([""] * (len(age_ranges) - len(values)))
        body_rows.append((bairro, values[: len(age_ranges)]))

    return age_ranges, body_rows


def split_bairro(raw: str) -> Tuple[str, str]:
    match = re.match(r"^(\d+)\s*[- ]\s*(.+)$", raw)
    if match:
        return match.group(1), match.group(2).strip()

    match = re.match(r"^(\d+)\s+(.+)$", raw)
    if match:
        return match.group(1), match.group(2).strip()

    return "", raw.strip()


def clean_number(value: str) -> str:
    v = value.strip()
    if not v or v == "-":
        return "0"
    v = v.replace(".", "").replace(" ", "")
    return v


def scrape(output_csv: str, max_causes: Optional[int], sleep_s: float) -> None:
    landing = fetch(TABNET_URL)
    form = parse_form(landing, TABNET_URL)

    line_name = find_select_by_label(form, "Linha")
    col_name = find_select_by_label(form, "Coluna")

    bairro_name = find_select_by_label(form, "Bairro Resid")
    age_name = find_select_by_label(form, "Faixa")
    cause_name = find_first_select_by_labels(
        form,
        [
            "Causa (Cap CID10)",
            "Causa (Capital CID10)",
            "Causa (Capítulo CID10)",
            "Causa",
        ],
    )
    munic_name = find_select_by_label(form, "Munic Resid")
    uf_name = find_select_by_label(form, "UF Res")

    base_payload = build_base_payload(form)
    base_payload[line_name] = option_value_by_hint(form.selects[line_name], "Bairro")
    base_payload[col_name] = option_value_by_hint(form.selects[col_name], "Faixa")

    base_payload[munic_name] = option_value_by_hint(form.selects[munic_name], "330455")
    base_payload[uf_name] = option_value_by_hint(form.selects[uf_name], "RJ")

    # Keep all categories for neighborhood and age filter dimensions.
    base_payload[bairro_name] = option_value_by_hint(form.selects[bairro_name], "Todas")
    base_payload[age_name] = option_value_by_hint(form.selects[age_name], "Todas")

    cause_options = [
        (value, label)
        for value, label in form.selects[cause_name].options
        if value and normalize(label) not in {"total", "todas", "todos"}
    ]

    if max_causes is not None:
        cause_options = cause_options[:max_causes]

    all_rows: List[Dict[str, str]] = []
    age_headers: List[str] = []

    for idx, (cause_value, cause_label) in enumerate(cause_options, start=1):
        payload = dict(base_payload)
        payload[cause_name] = cause_value

        html_result = fetch(form.action, data=payload)
        current_ages, matrix_rows = parse_tabnet_matrix(html_result)

        if not age_headers:
            age_headers = current_ages

        for bairro_raw, values in matrix_rows:
            code, bairro = split_bairro(bairro_raw)
            row = {
                "neighborhood_code": code,
                "neighborhood": bairro,
                "cause": cause_label,
            }
            for age_label, count in zip(age_headers, values):
                row[age_label] = clean_number(count)
            all_rows.append(row)

        print(f"[{idx}/{len(cause_options)}] scraped cause: {cause_label}")
        if sleep_s > 0:
            time.sleep(sleep_s)

    if not all_rows:
        raise RuntimeError("No rows collected. Check filters and page availability.")

    fieldnames = ["neighborhood_code", "neighborhood", "cause", *age_headers]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved {len(all_rows)} rows to {output_csv}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape Rio TabNet SIM mortality data and export to CSV."
    )
    parser.add_argument(
        "--output",
        default="rio_mortality_by_neighborhood_cause_age.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--max-causes",
        type=int,
        default=None,
        help="Optional limit for number of causes (debug/testing).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to wait between cause requests.",
    )
    args = parser.parse_args()

    try:
        scrape(output_csv=args.output, max_causes=args.max_causes, sleep_s=args.sleep)
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
