"""
Read-only monday.com API v2 client.

Design notes:
  * GraphQL over the documented /v2 endpoint. No SDK, so the dependency surface
    stays tiny and the query complexity stays visible.
  * Cursor pagination via items_page / next_items_page, because boards here are
    a few hundred items and the legacy `limit/page` args are deprecated.
  * Retries with exponential backoff on 429 / 5xx and on monday's
    COMPLEXITY_BUDGET_EXHAUSTED, which is returned as HTTP 200 with an errors[].
  * A short TTL cache so a multi-tool answer does not re-fetch both boards for
    every tool call in the same turn.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from .config import settings

log = logging.getLogger(__name__)

API_URL = "https://api.monday.com/v2"
API_VERSION = "2024-10"

_BOARD_QUERY = """
query ($boardId: ID!, $limit: Int!) {
  boards(ids: [$boardId]) {
    id
    name
    columns { id title type }
    items_page(limit: $limit) {
      cursor
      items {
        id
        name
        column_values { id text type }
      }
    }
  }
}
"""

_NEXT_PAGE_QUERY = """
query ($cursor: String!, $limit: Int!) {
  next_items_page(cursor: $cursor, limit: $limit) {
    cursor
    items {
      id
      name
      column_values { id text type }
    }
  }
}
"""


class MondayError(RuntimeError):
    """Raised when monday.com cannot serve the request at all."""


@dataclass
class RawBoard:
    board_id: str
    name: str
    columns: list[dict]                 # [{id,title,type}]
    items: list[dict]                   # [{id,name,column_values:[...]}]
    fetched_at: float = field(default_factory=time.time)

    @property
    def column_titles(self) -> dict[str, str]:
        return {c["id"]: c["title"] for c in self.columns}


class MondayClient:
    def __init__(self, token: str | None = None, timeout: int = 30):
        self.token = token or settings.monday_token
        self.timeout = timeout
        self._cache: dict[str, RawBoard] = {}
        self._session = requests.Session()

    # -- transport ---------------------------------------------------------

    def _post(self, query: str, variables: dict) -> dict:
        if not self.token:
            raise MondayError(
                "No monday.com API token configured. Set MONDAY_API_TOKEN "
                "(monday.com -> avatar -> Developers -> My access tokens)."
            )
        headers = {
            "Authorization": self.token,
            "API-Version": API_VERSION,
            "Content-Type": "application/json",
        }
        delay = 1.0
        last_error = None
        for attempt in range(settings.max_retries):
            try:
                response = self._session.post(
                    API_URL, json={"query": query, "variables": variables},
                    headers=headers, timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
            else:
                if response.status_code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {response.status_code}"
                elif response.status_code == 401:
                    raise MondayError("monday.com rejected the token (401). "
                                      "Check MONDAY_API_TOKEN.")
                else:
                    try:
                        payload = response.json()
                    except ValueError:
                        raise MondayError(
                            f"monday.com returned non-JSON (HTTP {response.status_code})."
                        )
                    errors = payload.get("errors") or payload.get("error_message")
                    if errors:
                        text = str(errors)
                        if "COMPLEXITY" in text.upper() or "minute" in text.lower():
                            last_error = f"rate/complexity limit: {text}"
                        else:
                            raise MondayError(f"monday.com GraphQL error: {text}")
                    else:
                        return payload["data"]

            sleep_for = delay + random.uniform(0, 0.4)
            log.warning("monday.com retry %s/%s after %s (sleeping %.1fs)",
                        attempt + 1, settings.max_retries, last_error, sleep_for)
            time.sleep(sleep_for)
            delay *= 2

        raise MondayError(f"monday.com unavailable after {settings.max_retries} "
                          f"attempts ({last_error}).")

    # -- fetching ----------------------------------------------------------

    def fetch_board(self, board_id: str, force: bool = False) -> RawBoard:
        """Fetch every item on a board, following the cursor to the end."""
        cached = self._cache.get(board_id)
        if cached and not force and (time.time() - cached.fetched_at) < settings.cache_ttl:
            return cached

        data = self._post(_BOARD_QUERY, {"boardId": str(board_id),
                                         "limit": settings.page_size})
        boards = data.get("boards") or []
        if not boards:
            raise MondayError(
                f"Board {board_id} not found, or the token has no access to it. "
                "Check the ID in the board URL and that the token's user is a "
                "member of the workspace."
            )
        board = boards[0]
        page = board["items_page"]
        items = list(page["items"])
        cursor = page.get("cursor")

        while cursor:
            data = self._post(_NEXT_PAGE_QUERY,
                              {"cursor": cursor, "limit": settings.page_size})
            page = data["next_items_page"]
            items.extend(page["items"])
            cursor = page.get("cursor")

        raw = RawBoard(board_id=str(board["id"]), name=board["name"],
                       columns=board["columns"], items=items)
        self._cache[board_id] = raw
        log.info("Fetched board %s (%s): %d items", board["name"], board_id, len(items))
        return raw

    def list_boards(self) -> list[dict[str, Any]]:
        """Helper for setup: shows board IDs so the user can fill in .env."""
        data = self._post("query { boards(limit:50) { id name items_count } }", {})
        return data.get("boards", [])

    def invalidate(self) -> None:
        self._cache.clear()
