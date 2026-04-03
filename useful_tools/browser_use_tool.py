"""
BrowserTool — wraps browser-use as a single callable tool for external agent frameworks.

Usage:

    # As a standalone tool (no LLM, direct browser control)
    tool = BrowserTool()
    await tool.start()
    result = await tool.execute(action="navigate", url="https://example.com")
    result = await tool.execute(action="click", index=3)
    await tool.stop()

    # As a high-level agent tool (uses LLM internally to complete a task)
    tool = BrowserTool(llm=ChatOpenAI(model="gpt-4o"))
    await tool.start()
    result = await tool.execute(action="run_task", task="Find the price of Bitcoin")
    await tool.stop()

    # Expose schema for external agent frameworks (OpenAI, LangChain, etc.)
    schema = tool.parameters_schema  # JSON Schema dict
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field

from browser_use.agent.views import ActionResult
from browser_use.browser import BrowserSession
from browser_use.browser.profile import BrowserProfile
from browser_use.llm.base import BaseChatModel
from browser_use.tools.service import Tools

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input schema: what external agents pass to this tool
# ---------------------------------------------------------------------------


class BrowserToolInput(BaseModel):
	"""Input schema for BrowserTool.execute().

	The external agent picks an action and provides the relevant parameters.
	"""

	action: str = Field(
		description=(
			'The browser action to perform. One of: '
			'run_task, navigate, click, type, scroll_down, scroll_up, '
			'go_back, send_keys, get_state, screenshot, done, wait, '
			'switch_tab, close_tab, extract'
		),
	)

	# -- High-level agent action --
	task: str | None = Field(default=None, description='Task description for run_task action (requires LLM)')

	# -- Navigation --
	url: str | None = Field(default=None, description='URL for navigate action')

	# -- Element interaction --
	index: int | None = Field(default=None, description='Element index from browser state for click/type actions')
	text: str | None = Field(default=None, description='Text to type for type action, or keys for send_keys')
	clear: bool = Field(default=True, description='Whether to clear the field before typing')

	# -- Scroll --
	pages: float = Field(default=1.0, description='Number of pages to scroll (0.5=half, 1=full, 10=bottom/top)')

	# -- Tab management --
	tab_id: str | None = Field(default=None, description='Tab ID for switch_tab/close_tab actions')

	# -- Extract --
	query: str | None = Field(default=None, description='Query for extract action')


# ---------------------------------------------------------------------------
# BrowserTool — the main class
# ---------------------------------------------------------------------------


class BrowserTool:
	"""A framework-agnostic tool wrapping browser-use for external agent consumption.

	Designed to plug into any agent framework that supports tool/function calling:
	  - OpenAI function calling: use name, description, parameters_schema
	  - LangChain: adapt via StructuredTool.from_function(tool.execute, ...)
	  - ConnectOnion or any custom framework: call tool.execute(**params)
	"""

	name: str = 'browser_use'
	description: str = (
		'Control a real browser to complete web tasks. '
		'Supports navigation, clicking elements, typing text, scrolling, '
		'extracting content, taking screenshots, managing tabs, and running '
		'full autonomous tasks with an AI agent. '
		'Use get_state to see the current page and available elements before interacting.'
	)

	def __init__(
		self,
		llm: BaseChatModel | None = None,
		headless: bool = True,
		browser_profile: BrowserProfile | None = None,
	):
		"""
		Args:
			llm: Optional LLM for run_task action (autonomous agent mode).
				Not needed for direct browser control actions.
			headless: Whether to run Chrome in headless mode.
			browser_profile: Optional custom browser profile.
		"""
		self._llm = llm
		self._headless = headless
		self._browser_profile = browser_profile
		self._tools: Tools | None = None
		self._session: BrowserSession | None = None
		self._started = False

	# -- Lifecycle ----------------------------------------------------------

	async def start(self) -> 'BrowserTool':
		"""Start the browser session. Must be called before execute()."""
		if self._started:
			return self

		profile = self._browser_profile or BrowserProfile(
			headless=self._headless,
			user_data_dir=None,
		)
		self._session = BrowserSession(browser_profile=profile)
		await self._session.start()
		self._tools = Tools()
		self._started = True
		logger.info('BrowserTool started')
		return self

	async def stop(self) -> None:
		"""Stop the browser session and release resources."""
		if self._session:
			await self._session.kill()
			self._session = None
		self._tools = None
		self._started = False
		logger.info('BrowserTool stopped')

	async def __aenter__(self) -> 'BrowserTool':
		return await self.start()

	async def __aexit__(self, *exc) -> None:
		await self.stop()

	# -- Schema for external frameworks ------------------------------------

	@property
	def parameters_schema(self) -> dict[str, Any]:
		"""JSON Schema for the tool input, suitable for OpenAI function calling etc."""
		return BrowserToolInput.model_json_schema()

	def to_openai_tool(self) -> dict[str, Any]:
		"""Return an OpenAI-compatible function tool definition."""
		return {
			'type': 'function',
			'function': {
				'name': self.name,
				'description': self.description,
				'parameters': self.parameters_schema,
			},
		}

	# -- Element finding helpers -------------------------------------------

	async def _get_selector_map(self) -> dict:
		"""Refresh and return the current selector map."""
		assert self._session is not None, 'BrowserTool not started'
		await self._session.get_browser_state_summary()
		return await self._session.get_selector_map()

	# -- Main execute method -----------------------------------------------

	async def execute(self, **kwargs: Any) -> str:
		"""Execute a browser action. This is the main entry point for external agents.

		Args:
			**kwargs: Parameters matching BrowserToolInput fields.

		Returns:
			String result describing what happened (suitable for LLM consumption).
		"""
		assert self._started and self._tools and self._session, 'BrowserTool not started. Call start() first.'

		params = BrowserToolInput(**kwargs)
		action = params.action.lower().strip()

		try:
			result = await self._dispatch(action, params)
			return result
		except Exception as e:
			error_msg = f'BrowserTool error ({action}): {e}'
			logger.error(error_msg)
			return error_msg

	async def _dispatch(self, action: str, params: BrowserToolInput) -> str:
		"""Route to the correct action handler."""
		assert self._tools is not None and self._session is not None

		if action == 'run_task':
			return await self._run_task(params)
		elif action == 'navigate':
			return await self._navigate(params)
		elif action == 'click':
			return await self._click(params)
		elif action == 'type':
			return await self._type(params)
		elif action == 'scroll_down':
			return await self._scroll(down=True, pages=params.pages)
		elif action == 'scroll_up':
			return await self._scroll(down=False, pages=params.pages)
		elif action == 'go_back':
			return await self._go_back()
		elif action == 'send_keys':
			return await self._send_keys(params)
		elif action == 'get_state':
			return await self._get_state()
		elif action == 'screenshot':
			return await self._screenshot()
		elif action == 'wait':
			return await self._wait(params)
		elif action == 'switch_tab':
			return await self._switch_tab(params)
		elif action == 'close_tab':
			return await self._close_tab(params)
		elif action == 'extract':
			return await self._extract(params)
		else:
			return f'Unknown action: {action}. Available: run_task, navigate, click, type, scroll_down, scroll_up, go_back, send_keys, get_state, screenshot, wait, switch_tab, close_tab, extract'

	# -- Action implementations --------------------------------------------

	async def _run_task(self, params: BrowserToolInput) -> str:
		"""Run a full autonomous browser task using the Agent + LLM."""
		if not self._llm:
			return 'Error: run_task requires an LLM. Pass llm= to BrowserTool constructor.'
		if not params.task:
			return 'Error: run_task requires a task parameter.'

		from browser_use import Agent

		agent = Agent(
			task=params.task,
			llm=self._llm,
			browser_session=self._session,
		)
		history = await agent.run()
		final = history.final_result()
		return final if final else 'Task completed (no explicit result returned).'

	async def _navigate(self, params: BrowserToolInput) -> str:
		if not params.url:
			return 'Error: navigate requires a url parameter.'
		result: ActionResult = await self._tools.navigate(  # type: ignore[union-attr]
			url=params.url, new_tab=False, browser_session=self._session,
		)
		return result.extracted_content or result.error or 'Navigated.'

	async def _click(self, params: BrowserToolInput) -> str:
		if params.index is None:
			return 'Error: click requires an index parameter. Use get_state to see available elements.'
		result: ActionResult = await self._tools.click(  # type: ignore[union-attr]
			index=params.index, browser_session=self._session,
		)
		return result.extracted_content or result.error or 'Clicked.'

	async def _type(self, params: BrowserToolInput) -> str:
		if params.index is None:
			return 'Error: type requires an index parameter.'
		if params.text is None:
			return 'Error: type requires a text parameter.'
		result: ActionResult = await self._tools.input(  # type: ignore[union-attr]
			index=params.index, text=params.text, clear=params.clear, browser_session=self._session,
		)
		return result.extracted_content or result.error or 'Typed.'

	async def _scroll(self, down: bool, pages: float) -> str:
		result: ActionResult = await self._tools.scroll(  # type: ignore[union-attr]
			down=down, pages=pages, browser_session=self._session,
		)
		return result.extracted_content or result.error or 'Scrolled.'

	async def _go_back(self) -> str:
		result: ActionResult = await self._tools.go_back(  # type: ignore[union-attr]
			browser_session=self._session,
		)
		return result.extracted_content or result.error or 'Navigated back.'

	async def _send_keys(self, params: BrowserToolInput) -> str:
		if not params.text:
			return 'Error: send_keys requires a text parameter (e.g. "Enter", "Escape", "Control+a").'
		result: ActionResult = await self._tools.send_keys(  # type: ignore[union-attr]
			keys=params.text, browser_session=self._session,
		)
		return result.extracted_content or result.error or 'Keys sent.'

	async def _get_state(self) -> str:
		"""Return the current browser state as a string the external agent can read."""
		assert self._session is not None
		state = await self._session.get_browser_state_summary()
		url = state.url or '(no url)'
		dom_repr = state.dom_state.llm_representation() if state.dom_state else '(empty page)'
		# Truncate if too long
		if len(dom_repr) > 10000:
			dom_repr = dom_repr[:10000] + '\n... (truncated)'
		return f'URL: {url}\n\nPage elements:\n{dom_repr}'

	async def _screenshot(self) -> str:
		result: ActionResult = await self._tools.screenshot(  # type: ignore[union-attr]
			browser_session=self._session,
		)
		return result.extracted_content or result.error or 'Screenshot taken.'

	async def _wait(self, params: BrowserToolInput) -> str:
		seconds = int(params.pages) if params.pages else 3  # reuse pages field for seconds
		result: ActionResult = await self._tools.wait(  # type: ignore[union-attr]
			seconds=seconds, browser_session=self._session,
		)
		return result.extracted_content or result.error or f'Waited {seconds}s.'

	async def _switch_tab(self, params: BrowserToolInput) -> str:
		if not params.tab_id:
			return 'Error: switch_tab requires a tab_id parameter.'
		result: ActionResult = await self._tools.switch(  # type: ignore[union-attr]
			tab_id=params.tab_id, browser_session=self._session,
		)
		return result.extracted_content or result.error or 'Switched tab.'

	async def _close_tab(self, params: BrowserToolInput) -> str:
		if not params.tab_id:
			return 'Error: close_tab requires a tab_id parameter.'
		result: ActionResult = await self._tools.close(  # type: ignore[union-attr]
			tab_id=params.tab_id, browser_session=self._session,
		)
		return result.extracted_content or result.error or 'Closed tab.'

	async def _extract(self, params: BrowserToolInput) -> str:
		if not params.query:
			return 'Error: extract requires a query parameter.'
		if not self._llm:
			return 'Error: extract requires an LLM. Pass llm= to BrowserTool constructor.'
		result: ActionResult = await self._tools.extract(  # type: ignore[union-attr]
			query=params.query,
			browser_session=self._session,
			page_extraction_llm=self._llm,
		)
		return result.extracted_content or result.error or 'Extraction returned no content.'
