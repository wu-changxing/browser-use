"""
Example: Using browser-use Tools directly without any Agent or LLM.

This demonstrates how to drive a real browser (Chrome) purely with imperative
Python code — navigate, click, type into forms, scroll, go back — using the
same action functions the Agent uses, but without any AI in the loop.

Usage:
    python examples/no_llm_tools.py
"""

import asyncio
import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread

from browser_use.agent.views import ActionResult
from browser_use.browser import BrowserSession
from browser_use.browser.profile import BrowserProfile
from browser_use.tools.service import Tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A tiny local web app served by the stdlib so this example is self-contained
# ---------------------------------------------------------------------------

INDEX_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Demo App</title></head>
<body>
  <h1 id="heading">Welcome to the Demo</h1>
  <nav>
    <a href="/about" id="about-link">About</a> |
    <a href="/form" id="form-link">Contact Form</a>
  </nav>
  <p>This page is served locally for the example.</p>
  <div style="height:2000px">
    <p>Scroll down…</p>
  </div>
  <p id="footer">You reached the bottom!</p>
</body>
</html>
"""

ABOUT_HTML = """\
<!DOCTYPE html>
<html>
<head><title>About</title></head>
<body>
  <h1>About Page</h1>
  <p id="info">browser-use tools running without an LLM.</p>
  <a href="/">Back to home</a>
</body>
</html>
"""

FORM_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Contact</title></head>
<body>
  <h1>Contact Form</h1>
  <form id="contact-form">
    <label for="name">Name:</label>
    <input type="text" id="name" name="name"><br>
    <label for="email">Email:</label>
    <input type="text" id="email" name="email"><br>
    <button type="submit" id="submit-btn">Send</button>
  </form>
  <div id="result" style="display:none;">Thanks for submitting!</div>
  <script>
    document.getElementById('contact-form').addEventListener('submit', e => {
      e.preventDefault();
      document.getElementById('result').style.display = 'block';
    });
  </script>
</body>
</html>
"""

PAGES = {
	'/': INDEX_HTML,
	'/about': ABOUT_HTML,
	'/form': FORM_HTML,
}


class _Handler(SimpleHTTPRequestHandler):
	def do_GET(self):
		path = self.path.split('?')[0]
		html = PAGES.get(path)
		if html:
			self.send_response(200)
			self.send_header('Content-Type', 'text/html')
			self.end_headers()
			self.wfile.write(html.encode())
		else:
			self.send_error(404)

	def log_message(self, *args):
		pass  # silence request logs


def _start_local_server(port: int = 8769) -> str:
	server = HTTPServer(('127.0.0.1', port), _Handler)
	t = Thread(target=server.serve_forever, daemon=True)
	t.start()
	return f'http://127.0.0.1:{port}'


# ---------------------------------------------------------------------------
# The main wrapper class that drives the browser without any LLM
# ---------------------------------------------------------------------------


class BrowserAutomation:
	"""
	Thin wrapper around browser-use's Tools + BrowserSession.
	No Agent. No LLM. Just direct function calls.
	"""

	def __init__(self, headless: bool = True):
		self.tools = Tools()
		self.session = BrowserSession(
			browser_profile=BrowserProfile(
				headless=headless,
				user_data_dir=None,
			)
		)

	async def start(self):
		await self.session.start()
		return self

	async def stop(self):
		await self.session.kill()

	# -- helpers to find elements by attribute in the selector map ----------

	async def _refresh_selector_map(self) -> dict:
		await self.session.get_browser_state_summary()
		return await self.session.get_selector_map()

	async def _find_element(self, **match_attrs) -> int:
		"""
		Return the highlight index of the first element whose attributes
		contain all of the given key/value pairs.

		Example:  await self._find_element(id='about-link')
		          await self._find_element(tag='input', name='email')
		"""
		selector_map = await self._refresh_selector_map()
		for idx, el in selector_map.items():
			attrs = el.attributes or {}
			tag = el.tag_name.lower()

			ok = True
			for k, v in match_attrs.items():
				if k == 'tag':
					if tag != v:
						ok = False
				elif attrs.get(k) != v:
					ok = False
			if ok:
				return idx

		available = [(idx, el.tag_name, el.attributes) for idx, el in selector_map.items()]
		raise LookupError(f'No element matching {match_attrs}. Available: {available}')

	# -- public action wrappers --------------------------------------------

	async def navigate(self, url: str) -> ActionResult:
		return await self.tools.navigate(url=url, new_tab=False, browser_session=self.session)

	async def click(self, **match_attrs) -> ActionResult:
		idx = await self._find_element(**match_attrs)
		return await self.tools.click(index=idx, browser_session=self.session)

	async def type_text(self, text: str, **match_attrs) -> ActionResult:
		idx = await self._find_element(**match_attrs)
		return await self.tools.input(index=idx, text=text, browser_session=self.session)

	async def scroll_down(self, pages: float = 1.0) -> ActionResult:
		return await self.tools.scroll(down=True, pages=pages, browser_session=self.session)

	async def scroll_up(self, pages: float = 1.0) -> ActionResult:
		return await self.tools.scroll(down=False, pages=pages, browser_session=self.session)

	async def go_back(self) -> ActionResult:
		return await self.tools.go_back(browser_session=self.session)

	async def current_url(self) -> str:
		return await self.session.get_current_page_url()


# ---------------------------------------------------------------------------
# Run the demo
# ---------------------------------------------------------------------------


async def main():
	base_url = _start_local_server()
	logger.info(f'Local server running at {base_url}')

	bot = BrowserAutomation(headless=True)
	await bot.start()

	try:
		# 1. Navigate to home page
		logger.info('--- Step 1: Navigate to home page ---')
		await bot.navigate(f'{base_url}/')
		assert '//' in await bot.current_url()
		logger.info(f'  URL: {await bot.current_url()}')

		# 2. Click the "About" link
		logger.info('--- Step 2: Click About link ---')
		await bot.click(id='about-link')
		await asyncio.sleep(0.3)
		url = await bot.current_url()
		assert '/about' in url, f'Expected /about, got {url}'
		logger.info(f'  URL: {url}')

		# 3. Go back to home
		logger.info('--- Step 3: Go back ---')
		await bot.go_back()
		await asyncio.sleep(0.3)
		url = await bot.current_url()
		logger.info(f'  URL: {url}')

		# 4. Scroll down on home page
		logger.info('--- Step 4: Scroll down ---')
		await bot.scroll_down(pages=3)
		logger.info('  Scrolled down 3 pages')

		# 5. Navigate to form, fill it, submit
		logger.info('--- Step 5: Fill and submit form ---')
		await bot.navigate(f'{base_url}/form')
		await bot.type_text('Alice', id='name')
		await bot.type_text('alice@example.com', id='email')
		await bot.click(id='submit-btn')
		await asyncio.sleep(0.3)

		# Verify submission via CDP
		cdp = await bot.session.get_or_create_cdp_session()
		result = await cdp.cdp_client.send.Runtime.evaluate(
			params={'expression': "document.getElementById('result').style.display"},
			session_id=cdp.session_id,
		)
		assert result['result']['value'] == 'block', 'Form submission failed'
		logger.info('  Form submitted successfully!')

		logger.info('=== All steps passed — Tools work without any LLM! ===')

	finally:
		await bot.stop()


if __name__ == '__main__':
	asyncio.run(main())
