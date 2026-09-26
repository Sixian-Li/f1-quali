"""Parse public qualifying pages without executing embedded scripts."""

import re
import time
import unicodedata
from datetime import date
from html.parser import HTMLParser


def normalized(value):
    value = unicodedata.normalize("NFKD", str(value))
    return re.sub(r"[^a-z0-9]", "", value.encode("ascii", "ignore").decode().lower())


class ResultPage(HTMLParser):
    """Read server-rendered table cells without executing embedded application JS."""

    def __init__(self):
        super().__init__()
        self.links, self.tables, self.headings, self.paragraphs = [], [], [], []
        self.table = self.row = self.cell = self.heading = self.paragraph = None

    def handle_starttag(self, tag, attrs):
        if tag == "a" and "href" in dict(attrs):
            self.links.append(dict(attrs)["href"])
        if tag == "table":
            self.table = []
        if self.table is not None and tag == "tr":
            self.row = []
        if self.table is not None and tag in {"th", "td"}:
            self.cell = []
        if tag == "h1":
            self.heading = []
        if tag == "p":
            self.paragraph = []

    def handle_data(self, value):
        value = value.strip()
        if value:
            for current in (self.cell, self.heading, self.paragraph):
                if current is not None:
                    current.append(value)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append(" ".join(self.cell))
            self.cell = None
        if tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
        if tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None
        if tag == "h1" and self.heading is not None:
            self.headings.append(" ".join(self.heading))
            self.heading = None
        if tag == "p" and self.paragraph is not None:
            self.paragraphs.append(" ".join(self.paragraph))
            self.paragraph = None


def parse_page(payload):
    parser = ResultPage()
    parser.feed(payload.decode("utf-8"))
    return parser


def lap_millis(value):
    if not value or value in {"-", "—", "DNF", "DNS", "DEL"}:
        return None
    match = re.fullmatch(r"(?:(\d+):)?(\d{1,2})\.(\d{3})", value)
    if not match:
        raise ValueError(f"Unrecognized official lap: {value!r}")
    minutes, seconds, fraction = match.groups()
    if minutes is not None and int(seconds) >= 60:
        raise ValueError(f"Invalid lap seconds component: {value}")
    millis = 60000 * int(minutes or 0) + 1000 * int(seconds) + int(fraction)
    if millis <= 0:
        raise ValueError("Official lap must be positive")
    return millis


def validate_page_identity(parsed, event, url):
    title = " | ".join(parsed.headings)
    if str(event.year) not in title or not title.upper().endswith("QUALIFYING"):
        raise ValueError(f"Official page identity/session mismatch: {url}: {title}")
    token = {
        "great-britain": "brit",
        "germany": "deutschland",
        "hungary": "magyar",
        "belgium": "belg",
        "italy": "ital",
        "turkey": "turk",
        "china": "chin",
        "canada": "canad",
        "spain": "span",
        "australia": "austral",
        "russia": "russi",
        "europe": "europ",
        "brazil": "brasil",
        "austria": "osterreich",
    }.get(event.grandPrixId, event.grandPrixId)
    if normalized(token) not in normalized(title):
        raise ValueError(f"Official page GP heading mismatch: {url}: {title}")
    dates = [
        p
        for p in parsed.paragraphs
        if re.fullmatch(r"(?:(?:\d{1,2})(?: [A-Za-z]{3})? - )?\d{1,2} [A-Za-z]{3} \d{4}", p)
    ]
    if len(dates) != 1:
        raise ValueError(f"Ambiguous official event date: {url}: {dates}")
    ending = re.search(r"(\d{1,2} [A-Za-z]{3} \d{4})$", dates[0]).group(1)
    displayed_date = date(*time.strptime(ending, "%d %b %Y")[:3]).isoformat()
    # A header-date discrepancy is preserved and quarantined, never silently
    # converted into a session timestamp or discarded from the evidence archive.
    return title, dates[0], displayed_date == event.date


def numeric_position(text):
    return int(text) if text.isdecimal() and int(text) > 0 else None


def driver_identity(raw, drivers, eligible):
    # The final uppercase three-letter text is the displayed abbreviation, not
    # part of the driver's name. Match names, not car numbers or table order.
    name = re.sub(r"\s+[A-Z]{3}$", "", raw).strip()
    aliases = {"carlossainz": "carlos-sainz-jr"}
    key = normalized(name)
    if key in aliases and aliases[key] in eligible:
        return aliases[key], "explicit_name_alias"
    candidates = drivers[drivers.id.isin(eligible)]
    matches = candidates[
        candidates.apply(
            lambda r: (
                key
                in {
                    normalized(r["name"]),
                    normalized(r.fullName),
                    normalized(str(r.firstName) + " " + str(r.lastName)),
                }
            ),
            axis=1,
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"Official driver cannot be uniquely matched: {raw!r}")
    return matches.id.iloc[0], "exact_accent_punctuation_normalized_name"


def constructor_identity(team, year):
    key = normalized(team)
    prefixes = [
        ("redbull", "red-bull"),
        ("rbr", "red-bull"),
        ("str", "toro-rosso"),
        ("tororosso", "toro-rosso"),
        ("forceindia", "force-india"),
        ("mclaren", "mclaren"),
        ("mercedes", "mercedes"),
        ("ferrari", "ferrari"),
        ("renault", "renault"),
        ("williams", "williams"),
        ("sauber", "sauber"),
        ("hrt", "hrt"),
        ("hispania", "hrt"),
        ("virgin", "virgin"),
        ("marussia", "marussia"),
        ("manor", "marussia"),
        ("caterham", "caterham"),
        ("lotus", "lotus-racing" if year <= 2011 else "lotus-f1"),
    ]
    found = [identity for prefix, identity in prefixes if key.startswith(prefix)]
    if len(set(found)) != 1:
        raise ValueError(f"Unreviewed official constructor: {team!r} ({year})")
    return found[0]
