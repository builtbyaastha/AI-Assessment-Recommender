"""Loads catalog.json into memory and exposes basic lookups."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from app.models.schemas import CatalogItem

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent / "catalog.json"


class Catalog:
    def __init__(self, path: Path | str = DEFAULT_CATALOG_PATH):
        raw = json.loads(Path(path).read_text())
        self.items: List[CatalogItem] = [CatalogItem(**row) for row in raw]
        self._by_id: Dict[str, CatalogItem] = {item.id: item for item in self.items}
        self._by_name_lower: Dict[str, CatalogItem] = {
            item.name.lower(): item for item in self.items
        }

    def get(self, item_id: str) -> Optional[CatalogItem]:
        return self._by_id.get(item_id)

    def find_by_name(self, name: str) -> Optional[CatalogItem]:
        # exact match first, then a loose substring match - good enough for
        # "compare X and Y" where X/Y won't always match the catalog name exactly
        name_lower = name.lower().strip()
        if name_lower in self._by_name_lower:
            return self._by_name_lower[name_lower]
        for item in self.items:
            if name_lower in item.name.lower() or item.name.lower() in name_lower:
                return item
        return None

    def __len__(self) -> int:
        return len(self.items)


_singleton: Optional[Catalog] = None


def get_catalog() -> Catalog:
    global _singleton
    if _singleton is None:
        _singleton = Catalog()
    return _singleton
