"""
monday.com GraphQL client (read-only).

Design notes:
  - Uses items_page cursor pagination (API v2, required since 2023-10).
  - Never mutates. Only `boards` and `items_page` queries are issued.
  - Column values are read from `column_values { id text value column { title } }`
    so the agent addresses columns by their human title, not by monday's
    generated column IDs. That means re-importing the board (which produces new
    IDs) does not break the agent.
  - Retries on 429 and 5xx with exponential backoff. monday's complexity budget
    is per-minute, so a backoff is usually enough.
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests

API_URL = "https://api.monday.com/v2"
API_VERSION = "2026-07"  # override with MONDAY_API_VERSION if this is retired
PAGE_SIZE = 100
MAX_RETRIES = 4


class MondayError(RuntimeError):
    """Raised when monday.com cannot serve the request."""


class MondayClient:
    def __init__(self, api_token: str, timeout: int = 30):
        if not api_token:
            raise MondayError("No monday.com API token configured.")
        self.api_token = api_token
        self.timeout = timeout
        self.session = requests.Session()

    # --- transport ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": self.api_token,
            "Content-Type": "application/json",
            "API-Version": API_VERSION,
        }

    def _post(self, query: str, variables: dict[str, Any] | None = None) -> dict:
        payload = {"query": query, "variables": variables or {}}
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.post(
                    API_URL,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = f"Network error talking to monday.com: {exc}"
                time.sleep(2**attempt)
                continue

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"monday.com returned HTTP {response.status_code}"
                time.sleep(2**attempt)
                continue

            if response.status_code == 401:
                raise MondayError(
                    "monday.com rejected the API token (401). Check that "
                    "MONDAY_API_TOKEN is a personal API token from "
                    "Admin > API, and that it has not been regenerated."
                )

            if response.status_code != 200:
                raise MondayError(
                    f"monday.com returned HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                )

            body = response.json()
            if "errors" in body and body["errors"]:
                messages = "; ".join(
                    e.get("message", str(e)) for e in body["errors"]
                )
                # Complexity errors are worth retrying once the budget resets.
                if "complexity" in messages.lower() and attempt < MAX_RETRIES - 1:
                    last_error = messages
                    time.sleep(10)
                    continue
                raise MondayError(f"monday.com GraphQL error: {messages}")

            return body.get("data", {})

        raise MondayError(
            f"monday.com unreachable after {MAX_RETRIES} attempts. {last_error}"
        )

    # --- queries -----------------------------------------------------------

    def board_name(self, board_id: str) -> str:
        query = """
        query ($ids: [ID!]) {
          boards (ids: $ids) { id name }
        }
        """
        data = self._post(query, {"ids": [str(board_id)]})
        boards = data.get("boards") or []
        if not boards:
            raise MondayError(
                f"Board {board_id} not found, or the token's account cannot see it."
            )
        return boards[0]["name"]

    def fetch_board_rows(self, board_id: str) -> list[dict[str, Any]]:
        """
        Return every item on the board as a flat dict keyed by column title.

        The item's own name is exposed as '__name' and its id as '__id' so
        they never collide with a real column title.
        """
        query = """
        query ($ids: [ID!], $limit: Int!, $cursor: String) {
          boards (ids: $ids) {
            items_page (limit: $limit, cursor: $cursor) {
              cursor
              items {
                id
                name
                column_values {
                  id
                  text
                  value
                  column { title }
                }
              }
            }
          }
        }
        """

        rows: list[dict[str, Any]] = []
        cursor: str | None = None

        while True:
            data = self._post(
                query,
                {"ids": [str(board_id)], "limit": PAGE_SIZE, "cursor": cursor},
            )
            boards = data.get("boards") or []
            if not boards:
                raise MondayError(f"Board {board_id} returned no data.")

            page = boards[0].get("items_page") or {}
            for item in page.get("items", []):
                row: dict[str, Any] = {
                    "__id": item.get("id"),
                    "__name": item.get("name"),
                }
                for cv in item.get("column_values", []):
                    title = (cv.get("column") or {}).get("title") or cv.get("id")
                    row[title] = cv.get("text")
                rows.append(row)

            cursor = page.get("cursor")
            if not cursor:
                break

        return rows

    def health_check(self) -> dict[str, Any]:
        query = "query { me { name email } }"
        data = self._post(query)
        return data.get("me") or {}
