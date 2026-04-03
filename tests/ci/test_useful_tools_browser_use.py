"""
Tests for useful_tools.BrowserTool.

Part 1: Tool interface tests (schema, lifecycle, error handling)
Part 2: Direct browser action tests (no LLM — navigate, click, type, scroll, get_state)
Part 3: E2E agent test (BrowserTool.run_task with a mock LLM driving the Agent)
"""

import asyncio
import json

import pytest
from pytest_httpserver import HTTPServer

from browser_use.browser import BrowserSession
from browser_use.browser.profile import BrowserProfile

from useful_tools.browser_use_tool import BrowserTool, BrowserToolInput


# ---------------------------------------------------------------------------
# Test pages
# ---------------------------------------------------------------------------

INDEX_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Tool Test Home</title></head>
<body>
  <h1 id="heading">Welcome</h1>
  <a href="/form" id="form-link">Go to form</a>
  <div style="height:2000px"><p>Tall content</p></div>
  <p id="bottom">Bottom of page</p>
</body>
</html>
"""

FORM_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Tool Test Form</title></head>
<body>
  <h1>Contact</h1>
  <form id="contact-form">
    <input type="text" id="name" name="name" placeholder="Name">
    <input type="text" id="email" name="email" placeholder="Email">
    <button type="submit" id="submit-btn">Send</button>
  </form>
  <div id="result" style="display:none;">Submitted!</div>
  <script>
    document.getElementById('contact-form').addEventListener('submit', e => {
      e.preventDefault();
      document.getElementById('result').style.display = 'block';
    });
  </script>
</body>
</html>
"""

RESULT_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Task Result</title></head>
<body>
  <h1>Result Page</h1>
  <p id="answer">The answer is 42</p>
</body>
</html>
"""


@pytest.fixture(scope='session')
def http_server():
	server = HTTPServer()
	server.start()
	server.expect_request('/').respond_with_data(INDEX_HTML, content_type='text/html')
	server.expect_request('/form').respond_with_data(FORM_HTML, content_type='text/html')
	server.expect_request('/result').respond_with_data(RESULT_HTML, content_type='text/html')
	yield server
	server.stop()


@pytest.fixture(scope='session')
def base_url(http_server):
	return f'http://{http_server.host}:{http_server.port}'


# ---------------------------------------------------------------------------
# Part 1: Tool interface tests
# ---------------------------------------------------------------------------


class TestBrowserToolInterface:
	"""Verify schema, lifecycle, and error handling — no browser needed."""

	def test_schema_is_valid_json_schema(self):
		"""parameters_schema returns a valid JSON Schema dict."""
		tool = BrowserTool()
		schema = tool.parameters_schema
		assert isinstance(schema, dict)
		assert 'properties' in schema
		assert 'action' in schema['properties']
		# Verify key fields exist
		for field in ('url', 'index', 'text', 'task', 'query', 'tab_id'):
			assert field in schema['properties'], f'{field} missing from schema'

	def test_openai_tool_format(self):
		"""to_openai_tool() returns a well-formed OpenAI function tool dict."""
		tool = BrowserTool()
		openai_def = tool.to_openai_tool()
		assert openai_def['type'] == 'function'
		assert openai_def['function']['name'] == 'browser_use'
		assert 'parameters' in openai_def['function']
		assert 'description' in openai_def['function']

	def test_input_model_validates(self):
		"""BrowserToolInput accepts valid params and rejects bad ones."""
		# Valid
		inp = BrowserToolInput(action='navigate', url='https://example.com')
		assert inp.action == 'navigate'
		assert inp.url == 'https://example.com'

		# Minimal
		inp2 = BrowserToolInput(action='get_state')
		assert inp2.action == 'get_state'

	async def test_execute_before_start_fails(self):
		"""Calling execute() without start() raises AssertionError."""
		tool = BrowserTool()
		with pytest.raises(AssertionError, match='not started'):
			await tool.execute(action='get_state')

	async def test_context_manager(self, base_url):
		"""async with BrowserTool() starts and stops correctly."""
		async with BrowserTool(headless=True) as tool:
			assert tool._started
			result = await tool.execute(action='navigate', url=f'{base_url}/')
			assert 'Navigated' in result
		assert not tool._started

	async def test_unknown_action_returns_error(self, base_url):
		"""Unknown action names return a helpful error string, not an exception."""
		async with BrowserTool(headless=True) as tool:
			result = await tool.execute(action='dance')
			assert 'Unknown action' in result
			assert 'navigate' in result  # lists available actions


# ---------------------------------------------------------------------------
# Part 2: Direct browser action tests (no LLM)
# ---------------------------------------------------------------------------


class TestBrowserToolDirectActions:
	"""Test each action works with a real Chrome browser and local HTTP server."""

	@pytest.fixture(scope='class')
	async def tool(self):
		"""Single BrowserTool instance shared across this test class."""
		t = BrowserTool(headless=True)
		await t.start()
		yield t
		await t.stop()

	async def test_navigate(self, tool, base_url):
		result = await tool.execute(action='navigate', url=f'{base_url}/')
		assert 'Navigated' in result

		url = await tool._session.get_current_page_url()
		assert base_url in url

	async def test_get_state(self, tool, base_url):
		await tool.execute(action='navigate', url=f'{base_url}/')
		result = await tool.execute(action='get_state')
		assert 'URL:' in result
		assert 'Welcome' in result or 'heading' in result

	async def test_click_link(self, tool, base_url):
		await tool.execute(action='navigate', url=f'{base_url}/')

		# Find the link index
		selector_map = await tool._get_selector_map()
		link_idx = None
		for idx, el in selector_map.items():
			if el.tag_name.lower() == 'a' and 'form-link' in (el.attributes.get('id', '') or ''):
				link_idx = idx
				break
		assert link_idx is not None, 'Could not find form-link'

		result = await tool.execute(action='click', index=link_idx)
		assert 'Clicked' in result or 'clicked' in result.lower()

		await asyncio.sleep(0.3)
		url = await tool._session.get_current_page_url()
		assert '/form' in url

	async def test_type_into_field(self, tool, base_url):
		await tool.execute(action='navigate', url=f'{base_url}/form')

		selector_map = await tool._get_selector_map()
		name_idx = None
		for idx, el in selector_map.items():
			if el.tag_name.lower() == 'input' and (el.attributes.get('name') == 'name'):
				name_idx = idx
				break
		assert name_idx is not None

		result = await tool.execute(action='type', index=name_idx, text='Alice')
		assert 'Typed' in result or 'typed' in result.lower() or 'Input' in result

		# Verify via CDP
		cdp = await tool._session.get_or_create_cdp_session()
		val = await cdp.cdp_client.send.Runtime.evaluate(
			params={'expression': "document.getElementById('name').value"},
			session_id=cdp.session_id,
		)
		assert val.get('result', {}).get('value') == 'Alice'

	async def test_scroll_down(self, tool, base_url):
		await tool.execute(action='navigate', url=f'{base_url}/')
		await asyncio.sleep(0.3)
		result = await tool.execute(action='scroll_down', pages=2)
		assert 'Scrolled' in result or 'scrolled' in result.lower()

	async def test_scroll_up(self, tool, base_url):
		result = await tool.execute(action='scroll_up', pages=1)
		assert 'Scrolled' in result or 'scrolled' in result.lower()

	async def test_go_back(self, tool, base_url):
		await tool.execute(action='navigate', url=f'{base_url}/')
		await tool.execute(action='navigate', url=f'{base_url}/form')
		result = await tool.execute(action='go_back')
		assert 'Navigated back' in result or 'back' in result.lower()
		await asyncio.sleep(0.5)
		url = await tool._session.get_current_page_url()
		assert '/form' not in url

	async def test_missing_params_return_error_string(self, tool):
		"""Missing required params return error strings, not exceptions."""
		r1 = await tool.execute(action='navigate')  # no url
		assert 'Error' in r1

		r2 = await tool.execute(action='click')  # no index
		assert 'Error' in r2

		r3 = await tool.execute(action='type', index=1)  # no text
		assert 'Error' in r3

	async def test_full_form_workflow(self, tool, base_url):
		"""Navigate → type → click submit → verify result. No LLM."""
		await tool.execute(action='navigate', url=f'{base_url}/form')

		selector_map = await tool._get_selector_map()

		name_idx = email_idx = submit_idx = None
		for idx, el in selector_map.items():
			tag = el.tag_name.lower()
			attrs = el.attributes or {}
			if tag == 'input' and attrs.get('name') == 'name':
				name_idx = idx
			elif tag == 'input' and attrs.get('name') == 'email':
				email_idx = idx
			elif tag == 'button' and attrs.get('id') == 'submit-btn':
				submit_idx = idx

		assert name_idx is not None
		assert email_idx is not None
		assert submit_idx is not None

		await tool.execute(action='type', index=name_idx, text='Bob')
		await tool.execute(action='type', index=email_idx, text='bob@test.com')
		await tool.execute(action='click', index=submit_idx)
		await asyncio.sleep(0.3)

		# Verify via CDP
		cdp = await tool._session.get_or_create_cdp_session()
		display = await cdp.cdp_client.send.Runtime.evaluate(
			params={'expression': "document.getElementById('result').style.display"},
			session_id=cdp.session_id,
		)
		assert display.get('result', {}).get('value') == 'block'


# ---------------------------------------------------------------------------
# Part 3: E2E agent test (run_task with mock LLM)
# ---------------------------------------------------------------------------


class TestBrowserToolE2EAgent:
	"""Test that BrowserTool.run_task correctly drives the full Agent loop."""

	async def test_run_task_without_llm_returns_error(self, base_url):
		"""run_task without an LLM returns an error, not an exception."""
		async with BrowserTool(headless=True) as tool:
			result = await tool.execute(action='run_task', task='do something')
			assert 'Error' in result
			assert 'LLM' in result

	async def test_run_task_without_task_returns_error(self, base_url):
		"""run_task without a task param returns an error."""
		async with BrowserTool(headless=True) as tool:
			result = await tool.execute(action='run_task')
			assert 'Error' in result
			assert 'task' in result.lower()

	async def test_run_task_with_mock_agent(self, base_url):
		"""Full e2e: BrowserTool.run_task uses Agent + mock LLM to complete a browser task."""
		from tests.ci.conftest import create_mock_llm

		# Mock LLM will: 1) navigate to /result, 2) call done with extracted answer
		actions = [
			json.dumps({
				'thinking': 'I need to navigate to the result page',
				'evaluation_previous_goal': 'Starting task',
				'memory': 'Need to find the answer',
				'next_goal': 'Navigate to result page',
				'action': [{'navigate': {'url': f'{base_url}/result'}}],
			}),
			json.dumps({
				'thinking': 'I can see the answer on this page',
				'evaluation_previous_goal': 'Successfully navigated to result page',
				'memory': 'Found the answer: 42',
				'next_goal': 'Report the answer',
				'action': [{'done': {'text': 'The answer is 42', 'success': True}}],
			}),
		]
		mock_llm = create_mock_llm(actions=actions)

		tool = BrowserTool(llm=mock_llm, headless=True)
		await tool.start()
		try:
			result = await tool.execute(action='run_task', task='Find the answer on the result page')

			assert '42' in result
		finally:
			await tool.stop()
