"""
Test that browser-use tools/functions work independently without the Agent/LLM layer.

This verifies the "headless automation" use case: calling navigate, click, input, scroll,
extract content, etc. directly via the Tools API with only a BrowserSession — no AI needed.
"""

import asyncio

import pytest
from pytest_httpserver import HTTPServer

from browser_use.agent.views import ActionResult
from browser_use.browser import BrowserSession
from browser_use.browser.profile import BrowserProfile
from browser_use.tools.service import Tools


FORM_PAGE = """
<!DOCTYPE html>
<html>
<head><title>Test Form</title></head>
<body>
	<h1>Contact Form</h1>
	<form id="contact-form">
		<label for="name">Name:</label>
		<input type="text" id="name" name="name" placeholder="Enter name">
		<label for="email">Email:</label>
		<input type="text" id="email" name="email" placeholder="Enter email">
		<button type="submit" id="submit-btn">Submit</button>
	</form>
	<div id="result" style="display:none;">Form submitted!</div>
	<script>
		document.getElementById('contact-form').addEventListener('submit', function(e) {
			e.preventDefault();
			document.getElementById('result').style.display = 'block';
		});
	</script>
</body>
</html>
"""

LINK_PAGE = """
<!DOCTYPE html>
<html>
<head><title>Links Page</title></head>
<body>
	<h1>Navigation Test</h1>
	<a href="/destination" id="dest-link">Go to destination</a>
	<p>Some paragraph text for scrolling tests.</p>
	<div style="height: 2000px;">Tall content to enable scrolling</div>
	<p id="bottom-text">Bottom of the page</p>
</body>
</html>
"""

DESTINATION_PAGE = """
<!DOCTYPE html>
<html>
<head><title>Destination</title></head>
<body>
	<h1>You arrived!</h1>
	<p>This is the destination page.</p>
</body>
</html>
"""


@pytest.fixture(scope='session')
def http_server():
	server = HTTPServer()
	server.start()
	server.expect_request('/form').respond_with_data(FORM_PAGE, content_type='text/html')
	server.expect_request('/links').respond_with_data(LINK_PAGE, content_type='text/html')
	server.expect_request('/destination').respond_with_data(DESTINATION_PAGE, content_type='text/html')
	yield server
	server.stop()


@pytest.fixture(scope='session')
def base_url(http_server):
	return f'http://{http_server.host}:{http_server.port}'


@pytest.fixture(scope='module')
async def browser_session():
	session = BrowserSession(
		browser_profile=BrowserProfile(headless=True, user_data_dir=None, keep_alive=True)
	)
	await session.start()
	yield session
	await session.kill()


@pytest.fixture(scope='function')
def tools():
	return Tools()


class TestToolsWithoutAgent:
	"""Prove that Tools can drive a full browser workflow with zero LLM involvement."""

	async def test_navigate_and_read_title(self, tools, browser_session, base_url):
		"""Navigate to a page and verify we landed there."""
		result = await tools.navigate(url=f'{base_url}/links', new_tab=False, browser_session=browser_session)
		assert isinstance(result, ActionResult)

		url = await browser_session.get_current_page_url()
		assert '/links' in url

	async def test_click_link_navigates(self, tools, browser_session, base_url):
		"""Click a link by index and verify navigation to the destination."""
		# Navigate to links page
		await tools.navigate(url=f'{base_url}/links', new_tab=False, browser_session=browser_session)

		# Build the selector map so we can find the link's index
		await browser_session.get_browser_state_summary()
		selector_map = await browser_session.get_selector_map()

		link_index = None
		for idx, element in selector_map.items():
			if element.tag_name.lower() == 'a' and 'dest-link' in (element.attributes.get('id', '') or ''):
				link_index = idx
				break

		assert link_index is not None, 'Could not find the destination link in selector map'

		result = await tools.click(index=link_index, browser_session=browser_session)
		assert isinstance(result, ActionResult)

		# Wait for navigation
		await asyncio.sleep(0.5)

		url = await browser_session.get_current_page_url()
		assert '/destination' in url

	async def test_fill_form_fields(self, tools, browser_session, base_url):
		"""Navigate to a form, type into input fields by index."""
		await tools.navigate(url=f'{base_url}/form', new_tab=False, browser_session=browser_session)

		# Build selector map
		await browser_session.get_browser_state_summary()
		selector_map = await browser_session.get_selector_map()

		# Find the name and email input indices
		name_index = None
		email_index = None
		for idx, element in selector_map.items():
			if element.tag_name.lower() == 'input':
				name_attr = element.attributes.get('name', '') or ''
				if name_attr == 'name':
					name_index = idx
				elif name_attr == 'email':
					email_index = idx

		assert name_index is not None, 'Could not find name input'
		assert email_index is not None, 'Could not find email input'

		# Type into both fields
		result_name = await tools.input(index=name_index, text='Alice', browser_session=browser_session)
		assert isinstance(result_name, ActionResult)

		result_email = await tools.input(index=email_index, text='alice@example.com', browser_session=browser_session)
		assert isinstance(result_email, ActionResult)

		# Verify values via CDP
		cdp_session = await browser_session.get_or_create_cdp_session()
		name_val = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={'expression': "document.getElementById('name').value"},
			session_id=cdp_session.session_id,
		)
		assert name_val.get('result', {}).get('value') == 'Alice'

		email_val = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={'expression': "document.getElementById('email').value"},
			session_id=cdp_session.session_id,
		)
		assert email_val.get('result', {}).get('value') == 'alice@example.com'

	async def test_scroll_page(self, tools, browser_session, base_url):
		"""Scroll down on a page with tall content."""
		await tools.navigate(url=f'{base_url}/links', new_tab=False, browser_session=browser_session)
		await asyncio.sleep(0.3)

		result = await tools.scroll(direction='down', amount=3, browser_session=browser_session)
		assert isinstance(result, ActionResult)
		assert result.extracted_content is not None

	async def test_go_back_after_navigation(self, tools, browser_session, base_url):
		"""Navigate forward then back, verifying history works."""
		await tools.navigate(url=f'{base_url}/links', new_tab=False, browser_session=browser_session)
		await tools.navigate(url=f'{base_url}/form', new_tab=False, browser_session=browser_session)

		url_before_back = await browser_session.get_current_page_url()
		assert '/form' in url_before_back

		result = await tools.go_back(browser_session=browser_session)
		assert isinstance(result, ActionResult)
		await asyncio.sleep(0.5)

		url_after_back = await browser_session.get_current_page_url()
		assert '/links' in url_after_back

	async def test_full_workflow_no_llm(self, tools, browser_session, base_url):
		"""
		End-to-end workflow: navigate -> fill form -> click submit -> verify result.
		Zero LLM calls. This is the core proof that functions work standalone.
		"""
		# 1. Navigate to form
		await tools.navigate(url=f'{base_url}/form', new_tab=False, browser_session=browser_session)

		# 2. Build selector map
		await browser_session.get_browser_state_summary()
		selector_map = await browser_session.get_selector_map()

		# 3. Find elements
		name_index = None
		email_index = None
		submit_index = None
		for idx, element in selector_map.items():
			tag = element.tag_name.lower()
			attrs = element.attributes or {}
			if tag == 'input' and attrs.get('name') == 'name':
				name_index = idx
			elif tag == 'input' and attrs.get('name') == 'email':
				email_index = idx
			elif tag == 'button' and attrs.get('id') == 'submit-btn':
				submit_index = idx

		assert name_index is not None
		assert email_index is not None
		assert submit_index is not None

		# 4. Fill form
		await tools.input(index=name_index, text='Bob', browser_session=browser_session)
		await tools.input(index=email_index, text='bob@test.com', browser_session=browser_session)

		# 5. Click submit
		await tools.click(index=submit_index, browser_session=browser_session)
		await asyncio.sleep(0.3)

		# 6. Verify the "Form submitted!" div is now visible
		cdp_session = await browser_session.get_or_create_cdp_session()
		result_display = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={'expression': "document.getElementById('result').style.display"},
			session_id=cdp_session.session_id,
		)
		assert result_display.get('result', {}).get('value') == 'block', 'Form submission did not trigger result display'

		result_text = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={'expression': "document.getElementById('result').textContent"},
			session_id=cdp_session.session_id,
		)
		assert result_text.get('result', {}).get('value') == 'Form submitted!'
