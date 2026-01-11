import datetime
import asyncio
import json
import os.path
from datetime import datetime
import random
from playwright.async_api import async_playwright
from lxml import html

import pandas as pd

from mivazakim_scraper import Scraper
from utils import read_session, read_cookies, VIEWPORTS, perform_random_mouse_movements, update_session


class SearchScraper(Scraper):
    """
    Scraper that searches for specific keywords on the website
    """
    def __init__(self, date: datetime, keywords: set, num_pages: int = 1, start_date: datetime = None):
        """
        Initialize search scraper with keywords
        :param date: The specific date to scrape
        :param keywords: Set of words to search for
        :param num_pages: Number of pages to scrape for each keyword (default: 1)
        :param start_date: Optional start date for reference
        """
        super().__init__(date, start_date)
        self.keywords = keywords
        self.num_pages = num_pages

    def __str__(self):
        """String representation for session/cookie naming"""
        # Use first keyword for naming (sanitized)
        first_keyword = list(self.keywords)[0].replace(' ', '_') if self.keywords else 'search'
        return f"search_scraper_{first_keyword}_{self.date}"

    def _get_search_data(self, page_source: str) -> pd.DataFrame:
        """
        Extract data from search results page with different structure
        After search, the page has a different structure:
        - Dates are in <div class="date"> under <div class="dateandlegends">
        - Headlines are in <td class="nf_title"> with <a> tags containing title attribute
        """
        # Parse with lxml for XPath support
        tree = html.fromstring(page_source)

        all_headlines = []
        all_dates = []

        # Find all date separators
        date_divs = tree.xpath('//div[@class="dateandlegends"]/div[@class="date"]')

        for date_div in date_divs:
            # Get the date text (format: DD.MM.YYYY)
            date_text = date_div.text_content().strip()

            # Convert date format from DD.MM.YYYY to YYYY-MM-DD
            try:
                date_parts = date_text.split('.')
                if len(date_parts) == 3:
                    day, month, year = date_parts
                    formatted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                else:
                    formatted_date = self.date  # Fallback to scraper's date
            except:
                formatted_date = self.date  # Fallback to scraper's date

            # Find the next table after this date div
            # Get parent dateandlegends div, then find the following sibling table
            dateandlegends_div = date_div.getparent()
            following_table = dateandlegends_div.getnext()

            if following_table is not None and following_table.tag == 'table':
                # Extract all headlines from this table
                headline_links = following_table.xpath('.//td[@class="nf_title nf_rtl"]/a | .//td[@class="nf_title"]/a')

                for link in headline_links:
                    # Get title attribute
                    title = link.get('title', '').strip()
                    if title:
                        all_headlines.append(title)
                        all_dates.append(formatted_date)

        df = pd.DataFrame({
            'date': all_dates,
            'headline': all_headlines
        })

        print(f"Total headlines collected from search: {len(df)}")

        return df

    async def _search_and_get_page_source(self, keyword: str, response_url=None, headers: dict = None) -> tuple[str, str]:
        """
        Search for a keyword and get page source
        :param keyword: The keyword to search
        :return: tuple of (page source, search result URL)
        """
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
                ignore_https_errors=True
            )

            if headers:
                await context.set_extra_http_headers(headers)

            if cookies:
                await context.add_cookies(cookies)

            # Create page from context
            page = await context.new_page()

            # Navigate to base URL first
            base_url = 'https://mivzakim.net/view/category/17'
            await page.goto(base_url, timeout=100000)
            await page.wait_for_load_state('domcontentloaded')

            # Perform random mouse movements
            await perform_random_mouse_movements(page, min_actions=2, max_actions=5)

            # Find and fill search box
            search_input_xpath = '/html/body/div[1]/div[3]/div[1]/form/div/input[2]'
            search_result_url = None
            try:
                # Wait for search input to be available
                await page.wait_for_selector(f'xpath={search_input_xpath}', timeout=10000)

                # Type the keyword into search box
                search_input = page.locator(f'xpath={search_input_xpath}')
                await search_input.fill(keyword)
                await page.wait_for_timeout(50)

                # Submit the search (press Enter)
                await search_input.press('Enter')

                # Wait for navigation to complete (new URL will be generated)
                await page.wait_for_load_state('domcontentloaded', timeout=30000)
                await page.wait_for_timeout(50)

                # Get the search result URL
                search_result_url = page.url

                # Perform additional human-like actions
                await page.keyboard.press("PageDown")
                await page.wait_for_load_state('domcontentloaded')

                print(f"Searched for '{keyword}', current URL: {search_result_url}")

            except Exception as e:
                print(f"Error during search for keyword '{keyword}': {e}")

            # Get page source
            full_page_source = await page.content()

            # Save cookies and session
            with open(f"./cookies/{self.__str__()}-cookies.json", "w+") as f:
                f.write(json.dumps(await context.cookies()))

            update_session(new_data=await context.storage_state(), name=self.__str__())

            await browser.close()

        return full_page_source, search_result_url

    async def _get_page_source_from_url(self, url: str, headers: dict = None) -> str:
        """
        Get page source from a specific URL (for pagination)
        :param url: The URL to fetch
        :return: page source
        """
        session = read_session(session_name=self.__str__())
        cookies = read_cookies(cookies_name=self.__str__())

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
                ignore_https_errors=True
            )

            if headers:
                await context.set_extra_http_headers(headers)

            if cookies:
                await context.add_cookies(cookies)

            page = await context.new_page()

            # Navigate to the URL
            await page.goto(url, timeout=100000)
            await page.wait_for_load_state('domcontentloaded')

            # Wait for content to load
            # await page.wait_for_selector('div.dateandlegends', timeout=10000)

            # Perform some human-like actions
            await perform_random_mouse_movements(page, min_actions=2, max_actions=4)
            await page.keyboard.press("PageDown")
            await page.wait_for_timeout(50)

            # Get page source
            full_page_source = await page.content()

            # Save cookies and session
            with open(f"./cookies/{self.__str__()}-cookies.json", "w+") as f:
                f.write(json.dumps(await context.cookies()))

            update_session(new_data=await context.storage_state(), name=self.__str__())

            await browser.close()

        return full_page_source

    async def scrape_from_search(self) -> pd.DataFrame:
        """
        Search for all keywords and extract headlines from multiple pages
        Note: xpath parameter is ignored for search results as they use a different structure
        :return: DataFrame with all results
        """
        all_dataframes = []

        for keyword in self.keywords:
            print(f"Searching for keyword: '{keyword}' across {self.num_pages} page(s)")
            try:
                # Perform search and get first page
                page_source, search_result_url = await self._search_and_get_page_source(keyword)

                if not search_result_url:
                    print(f"Could not get search result URL for keyword '{keyword}'")
                    continue

                # Scrape first page
                df = self._get_search_data(page_source)
                df['keyword'] = keyword
                df['page'] = 1
                all_dataframes.append(df)

                print(f"  Page 1: Found {len(df)} headlines")

                # Scrape additional pages if num_pages > 1
                for page_num in range(2, self.num_pages + 1):
                    # Construct pagination URL
                    paginated_url = f"{search_result_url}/page/{page_num}"

                    print(f"  Scraping page {page_num}: {paginated_url}")

                    try:
                        # Get page source for this page
                        page_source = await self._get_page_source_from_url(paginated_url)

                        # Extract data
                        df_page = self._get_search_data(page_source)

                        if df_page.empty:
                            print(f"    Page {page_num} is empty, stopping pagination for '{keyword}'")
                            break

                        df_page['keyword'] = keyword
                        df_page['page'] = page_num
                        all_dataframes.append(df_page)

                        print(f"    Page {page_num}: Found {len(df_page)} headlines")

                        # Small delay between pages
                        await asyncio.sleep(random.uniform(1, 2))

                    except Exception as e:
                        print(f"    Error scraping page {page_num} for keyword '{keyword}': {e}")
                        break

                # Delay between different keywords
                await asyncio.sleep(random.uniform(1, 2))

            except Exception as e:
                print(f"Error scraping keyword '{keyword}': {e}")
                continue

        # Combine all results
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True).drop_duplicates().reset_index(drop=True)
            print(f"Total headlines found across all keywords and pages: {len(combined_df)}")
            return combined_df
        else:
            return pd.DataFrame()


