import datetime
import asyncio
import json
import os.path
import string
from datetime import datetime, timedelta
import random
from playwright.async_api import Page, async_playwright
from lxml import html

import pandas as pd

DATE_FORMAT = '%Y-%m-%d'

VIEWPORTS = [
    {'width': 1024, 'height': 1366},
]


def generate_ucs(length=8) -> str:
    characters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


def update_session(name: str, new_data: dict) -> None:
    """Updates or creates a session file."""
    os.makedirs("sessions", exist_ok=True)
    session_file_path = os.path.join("sessions", f"{name}.json")
    try:
        with open(session_file_path, "w") as file:
            json.dump(new_data, file, indent=4)
        print(f"Session {name} successfully updated.")
    except Exception as e:
        print(f"Failed to update session {name}: {e}")


async def perform_random_mouse_movements(page: Page, min_actions: int = 2, max_actions: int = 5,
                                         min_pause: float = 0.5, max_pause: float = 1.0):
    """
    Performs random mouse movements and scrolling actions on a Playwright page
    without any clicking.
    """
    # Get page dimensions
    dimensions = await page.evaluate('''() => {
        return {
            width: document.documentElement.clientWidth,
            height: document.documentElement.clientHeight,
            scrollHeight: document.documentElement.scrollHeight
        }
    }''')

    viewport_width = dimensions['width']
    viewport_height = dimensions['height']
    scroll_height = dimensions['scrollHeight']

    # Determine number of actions to perform
    num_actions = random.randint(min_actions, max_actions)

    for _ in range(num_actions):
        # Choose a random action (no clicks)
        action = random.choice(['move', 'hover_element', 'scroll_down', 'scroll_up', 'pause'])

        # Random pause between actions
        await asyncio.sleep(random.uniform(min_pause, max_pause))

        if action == 'move':
            # Move to random position
            x = random.randint(0, viewport_width)
            y = random.randint(0, viewport_height)
            await page.mouse.move(x, y)

        elif action == 'hover_element':
            # Find a random clickable element and hover over it
            clickable_elements = await page.evaluate('''() => {
                const elements = Array.from(document.querySelectorAll('a, button, [role="button"], input[type="submit"]'));
                return elements.map(el => {
                    const rect = el.getBoundingClientRect();
                    return {
                        x: rect.left + rect.width / 2,
                        y: rect.top + rect.height / 2,
                        width: rect.width,
                        height: rect.height,
                        visible: rect.width > 0 && rect.height > 0 &&
                                rect.top >= 0 && rect.left >= 0 &&
                                rect.bottom <= window.innerHeight &&
                                rect.right <= window.innerWidth
                    };
                }).filter(el => el.visible);
            }''')

            if clickable_elements:
                element = random.choice(clickable_elements)
                await page.mouse.move(element['x'], element['y'])

        elif action == 'scroll_down':
            # Random scroll down amount (between 100 and 800 pixels)
            scroll_amount = random.randint(100, 800)
            current_scroll = await page.evaluate('window.scrollY')

            # Ensure we don't scroll past the bottom
            max_scroll = scroll_height - viewport_height
            if current_scroll < max_scroll:
                # Use smooth scrolling
                await page.evaluate(f'window.scrollBy({{ top: {scroll_amount}, behavior: "smooth" }})')

        elif action == 'scroll_up':
            # Random scroll up amount (between 100 and 400 pixels)
            if await page.evaluate('window.scrollY') > 0:
                scroll_amount = random.randint(100, 400)
                await page.evaluate(f'window.scrollBy({{ top: -{scroll_amount}, behavior: "smooth" }})')

        elif action == 'pause':
            # Just pause for a random time (already handled above)
            pass

    # Final random pause
    await asyncio.sleep(random.uniform(min_pause, max_pause))


async def scroll_down(page: Page, button_selector: str, scroll_amount: int) -> None:
    try:
        await page.mouse.wheel(0, scroll_amount)
        await asyncio.sleep(random.uniform(1, 3))
        await page.click(button_selector, timeout=1000)
    except Exception as e:
        print(f"failed to press button: {e}")


def read_session(session_name: str) -> dict | None:
    """Reads a specific session file."""
    session_file_path = os.path.join("sessions", f"{session_name}.json")
    try:
        if not os.path.exists(session_file_path):
            print(
                f"Session file for {session_name} not found, not using session."
            )
            return None
        with open(session_file_path, "r") as file:
            content = file.read()
            if not content.strip():
                print(
                    f"Session file for {session_name} is empty, not using session."
                )
                return None
            return json.loads(content)
    except json.JSONDecodeError:
        print(
            f"Failed to decode session {session_name}, not using session."
        )
        return None
    except Exception as e:
        print(f"Unexpected error reading session {session_name}: {e}")
        return None


def read_cookies(cookies_name: str) -> dict | None:
    """Reads a specific session file."""
    cookies_file_path = os.path.join("cookies", f"{cookies_name}-cookies.json")
    try:
        if not os.path.exists(cookies_file_path):
            print(
                f"Cookies file for {cookies_name} not found, not using cookies."
            )
            return None
        with open(cookies_file_path, "r") as file:
            content = file.read()
            if not content.strip():
                print(
                    f"Cookies file for {cookies_name} is empty, not using cookies."
                )
                return None
            return json.loads(content)
    except json.JSONDecodeError:
        print(
            f"Failed to decode cookies {cookies_name}, not using cookies."
        )
        return None
    except Exception as e:
        print(f"Unexpected error reading cookies {cookies_name}: {e}")
        return None


class Scraper:
    def __init__(self, date: datetime):
        """Initialize scraper with a date"""
        self.date = date.strftime(DATE_FORMAT)

    def __str__(self):
        """String representation for session/cookie naming"""
        return f"scraper_{self.date}"

    @staticmethod
    def create_url(date: datetime, base_url: str = 'https://mivzakim.net/view/category/17/date/') -> str:
        """Create URL from date"""
        year = date.strftime('%Y')
        month = date.strftime('%m')
        day = date.strftime('%d')
        date_str = f"{year}-{month}-{day}"
        return f"{base_url}{date_str}"

    async def _get_page_source(self, url, response_url=None,
                               headers: dict = None) -> str:
        """
        Async get page source using Playwright
        :param url: website url
        :return: page source
        """

        async def _handle_response(response) -> None:
            if (response.url.startswith(response_url)) and response.status == 200 and response.url.endswith('.json'):
                with open("flights.json", "w") as f:
                    flights = await response.json()
                    json.dump(flights, f, indent=4)

        session = read_session(session_name=self.__str__())
        cookies = read_cookies(cookies_name=self.__str__())

        # Ensure directories exist
        os.makedirs("cookies", exist_ok=True)
        os.makedirs("sessions", exist_ok=True)

        async with async_playwright() as pw:
            browser = await pw.firefox.launch(
                headless=False,
                firefox_user_prefs={
                    "security.insecure_connection_text.enabled": True,
                    "security.insecure_connection_text.pbmode.enabled": True
                }
            )
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36',
                viewport=random.choice(VIEWPORTS),
                storage_state=session,
                ignore_https_errors=True  # Ignore SSL certificate errors
            )

            if headers:
                await context.set_extra_http_headers(headers)

            if cookies:
                await context.add_cookies(cookies)

            # Create page from context
            page = await context.new_page()

            # Write flights response to json file
            if response_url:
                page.on('response', _handle_response)

            # Simulate human
            await page.goto(url, timeout=100000)
            await page.wait_for_load_state('domcontentloaded')
            await perform_random_mouse_movements(page)
            await page.wait_for_timeout(2000)
            await page.keyboard.press("PageDown")
            await page.wait_for_load_state('domcontentloaded')

            # Get page source
            full_page_source = await page.content()

            with open(f"./cookies/{self.__str__()}-cookies.json", "w+") as f:
                f.write(json.dumps(await context.cookies()))

            # Save latest Session
            update_session(new_data=await context.storage_state(), name=self.__str__())

            await browser.close()
        return full_page_source

    def _get_data(self, page_source: str,
                  xpath: str = '/html/body/div[1]/div[4]/div[1]/div[3]/table/tbody/tr/td[4]/a') -> pd.DataFrame:
        """Extract data from page source using XPath"""
        # Parse with lxml for XPath support
        tree = html.fromstring(page_source)

        # Extract headlines using XPath - get text content
        headlines = tree.xpath(xpath + '/@title')

        # Alternative: if tbody is not in HTML, try without it
        if not headlines:
            xpath_no_tbody = xpath.replace('/tbody', '')
            headlines = tree.xpath(xpath_no_tbody + '/@title')

        # Clean and store headlines
        all_headlines = []
        all_dates = []
        for headline in headlines:
            cleaned_headline = headline.strip()
            if cleaned_headline:  # Only add non-empty headlines
                all_headlines.append(cleaned_headline)
                all_dates.append(self.date)

        df = pd.DataFrame({
            'date': all_dates,
            'headline': all_headlines
        })

        print(f"Total headlines collected for {self.date}: {len(df)}")

        return df

    async def write_data(self) -> pd.DataFrame:
        """
        Get data for this scraper's date
        :return: DataFrame with scraped data
        """
        return await self.scrape_from_page(xpath='/html/body/div[1]/div[4]/div[1]/div[3]/table/tbody/tr/td[4]/a')

    async def scrape_from_page(self, xpath: str, response_url=None, headers=None) -> pd.DataFrame:
        """
        Extract the data from the website
        """
        # Convert date string back to datetime for create_url
        date_obj = datetime.strptime(self.date, DATE_FORMAT)
        url = self.create_url(date_obj)

        page_source = await self._get_page_source(url=url, response_url=response_url, headers=headers)
        return self._get_data(page_source, xpath)


def clean_dir(dir_name: str) -> None:
    """Clean all files in a directory"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        return
    for file in os.listdir(dir_name):
        file_path = os.path.join(dir_name, file)
        if os.path.isfile(file_path):  # Only delete files
            os.remove(file_path)


def purge() -> None:
    """Purge cookies and sessions"""
    clean_dir('cookies')
    clean_dir('sessions')


async def scrape_single_date(date_obj: datetime) -> pd.DataFrame:
    """
    Scrape data for a single date
    :param date_obj: datetime object to scrape
    :return: DataFrame with scraped data
    """
    try:
        # Create scraper instance for this date
        scraper = Scraper(date_obj)

        # Get data
        df = await scraper.scrape_from_page(xpath='/html/body/div[1]/div[4]/div[1]/div[3]/table/tbody/tr/td[4]/a')

        print(f"Completed scraping for date: {date_obj.strftime(DATE_FORMAT)}")
        return df

    except Exception as e:
        print(f"Error scraping date {date_obj.strftime(DATE_FORMAT)}: {e}")
        return pd.DataFrame()


async def scrape_and_save(dates: list) -> None:
    """
    Main flow of the collecting process - scrapes all dates concurrently
    :param dates: list of datetime objects to scrape
    """
    print(f"Starting concurrent scraping for {len(dates)} dates...")

    # Create tasks for all dates to run concurrently
    tasks = [scrape_single_date(date) for date in dates]

    # Run all tasks concurrently and gather results
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Combine all dataframes
    all_data = []
    for result in results:
        if isinstance(result, pd.DataFrame) and not result.empty:
            all_data.append(result)
        elif isinstance(result, Exception):
            print(f"Task failed with exception: {result}")

    if all_data:
        # Combine all data
        df = pd.concat(all_data, ignore_index=True).drop_duplicates().reset_index(drop=True)

        # Load existing data if file exists
        file_name = "headlines.csv"
        if os.path.exists(file_name):
            existing_df = pd.read_csv(file_name)
            df = pd.concat([existing_df, df]).drop_duplicates().reset_index(drop=True)

        # Save to CSV
        df.to_csv(file_name, index=False)
        print(f"Data written to {file_name}")
        print(f"Total unique headlines: {len(df)}")
    else:
        print("No data collected")

    print("All tasks completed")


def get_data() -> None:
    """
    Running the main flow of collecting all headlines data
    :return: headlines.csv file
    """
    # Get dates for past 7 days
    dates = [datetime.now() - timedelta(days=i) for i in range(7)]

    # Run async scraping
    asyncio.run(scrape_and_save(dates))

    # Final cleanup
    purge()
    print("Data collection complete!")


if __name__ == "__main__":
    get_data()