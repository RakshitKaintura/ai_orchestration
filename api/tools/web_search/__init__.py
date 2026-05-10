"""
api/tools/web_search/__init__.py — public exports for web_search subpackage.
"""
from api.tools.web_search.tool import web_search
from api.tools.web_search.fixtures import FIXTURES, KEYWORD_MAP, select_fixture

__all__ = ["web_search", "FIXTURES", "KEYWORD_MAP", "select_fixture"]
