"""Build the English RM page from the Chinese page without changing rating data.

Run after editing docs/index.html or the translation catalog; --check is read-only.
The existing Chinese page is the source for shared styles and application logic.
"""

import argparse
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAN = re.compile(r"[\u4e00-\u9fff]")
SCRIPTS = re.compile(r"<script\b[^>]*>.*?</script>", re.S)


def english_page(chinese, catalog):
    blocks = SCRIPTS.findall(chinese)
    if len(blocks) != 3 or 'type="application/json" id="ratings-data"' not in blocks[1]:
        raise ValueError("Expected bundled Plotly, inert rating data and the application")
    shell = chinese
    for i, block in enumerate(blocks):
        shell = shell.replace(block, f"__SCRIPT_{i}__", 1)

    used = set()

    def translate(text, *, attribute=False):
        key = text.strip()
        if not HAN.search(key):
            return text
        if key not in catalog["html"]:
            raise ValueError(f"Missing English translation: {key}")
        used.add(key)
        value = html.escape(catalog["html"][key], quote=attribute)
        # Chinese punctuation often joins markup without spaces. English still
        # needs a space around the translated F1DB attribution and strong text.
        if key == "评分来自":
            value += " "
        if key == "及其贡献者，v2026.14.0，":
            value = " " + value + " "
        return text.replace(key, value, 1)

    shell = re.sub(r"(?<=>)[^<>]+(?=<)", lambda m: translate(m[0]), shell)
    shell = re.sub(r'(aria-label|placeholder|title)="([^"]+)"',
                   lambda m: f'{m[1]}="{translate(m[2], attribute=True)}"', shell)
    if used != set(catalog["html"]):
        raise ValueError(f"Stale HTML translations: {set(catalog['html']) - used}")
    shell = shell.replace('<html lang="zh-CN">', '<html lang="en">', 1)
    shell = shell.replace('hreflang="zh-CN" aria-current="page"', 'hreflang="zh-CN"', 1)
    shell = shell.replace('hreflang="en">English', 'hreflang="en" aria-current="page">English', 1)

    app = blocks[2]
    app, count = re.subn(r"  // BEGIN PLOTLY_ZH\n.*?  // END PLOTLY_ZH\n", "", app, flags=re.S)
    if count != 1 or app.count("locale:'zh-CN'") != 1:
        raise ValueError("Expected the Chinese Plotly locale and configuration")
    app = app.replace("locale:'zh-CN'", "locale:'en-US'")
    for source, target in catalog["javascript"].items():
        if source not in app:
            raise ValueError(f"Stale JavaScript translation: {source}")
        app = app.replace(source, target)
    if HAN.search(app):
        raise ValueError("Untranslated Chinese remains in the English application")
    blocks[2] = app
    for i, block in enumerate(blocks):
        shell = shell.replace(f"__SCRIPT_{i}__", block, 1)
    return shell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the English page is stale")
    args = parser.parse_args()
    chinese = (ROOT / "docs/index.html").read_text()
    catalog = json.loads((ROOT / "scripts/rating_page_en.json").read_text())
    english = english_page(chinese, catalog)
    target = ROOT / "docs/en.html"
    if args.check:
        if not target.exists() or target.read_text() != english:
            raise SystemExit("English page is stale; run scripts/build_rating_languages.py")
        print("English page matches the Chinese source and translations.")
    else:
        target.write_text(english)
        print("Built docs/en.html; rating data and bundled Plotly are unchanged.")


if __name__ == "__main__":
    main()
