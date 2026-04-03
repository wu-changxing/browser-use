"""
useful_tools - Framework-agnostic tool wrappers for browser-use.

Exposes browser-use capabilities as tools that any external agent framework
(OpenAI function calling, LangChain, ConnectOnion, etc.) can consume.
"""

from useful_tools.browser_use_tool import BrowserTool

__all__ = ['BrowserTool']
