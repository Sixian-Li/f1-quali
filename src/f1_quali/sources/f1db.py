"""F1DB CSV release adapter (CC BY 4.0). Reads only explicitly requested tables."""

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from f1_quali.sources.cache import fetch_snapshot


class F1DB:
    def __init__(self, root: Path, source: dict, *, offline: bool = False):
        self.version = source["version"]
        self.digest = source["sha256"]
        name = "f1db-csv.zip"
        self.path = root / "f1db" / self.version / name
        url = f"https://github.com/f1db/f1db/releases/download/{self.version}/{name}"
        self._zip = ZipFile(BytesIO(fetch_snapshot(url, self.path, self.digest, offline=offline)))

    def table(self, name: str) -> pd.DataFrame:
        with self._zip.open(f"f1db-{name}.csv") as file:
            return pd.read_csv(file, low_memory=False)
