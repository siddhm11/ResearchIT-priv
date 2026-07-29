"""
Shared Jinja2 environment with custom filters.
Import `templates` from here instead of creating it per-router.
"""
import json
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="app/templates")


def _tojson_parse(value: str | None) -> list:
    """Parse a JSON-encoded string into a Python list. Returns [] on error."""
    if not value:
        return []
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (ValueError, TypeError):
        return []


templates.env.filters["tojson_parse"] = _tojson_parse
