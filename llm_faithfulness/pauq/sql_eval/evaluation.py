"""Compatibility shim — re-exports the symbols the codebase actually uses."""

from .process_sql import Schema, get_sql

__all__ = ["Schema", "get_sql"]
