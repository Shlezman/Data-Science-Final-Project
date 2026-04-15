import asyncio
import json
import os
from datetime import datetime
import random
from lxml import html

import pandas as pd

from mivzakim_scraper import Scraper
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
        first_keyword = list(self.keywords)[0].replace(' ', '_') if self.keywords else 'search'
        return f"search_scraper_{first_keyword}_{self.date}"

    def _get_search_data(self, page_source: str) -> pd.DataFrame:
        """
        Extract data from search results page.
        Extracts all fields per row: date, source, hour, popularity, headline
        (matching _get_data field extraction pattern).
        """
        tree = html.fromstring(page_source)

        data = []

        # Find all date separators
        date_divs = tree.xpath('//div[@class="dateandlegends"]/div[@class="date"]')

        for date_div in date_divs:
            date_text = date_div.text_content().strip()

            # Convert date format from DD.MM.YYYY to YYYY-MM-DD
            try:
                date_parts = date_text.split('.')
                if len(date_parts) == 3:
                    day, month, year = date_parts
                    formatted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                else:
                    formatted_date = self.date
            except:
                formatted_date = self.date

            # Find the next table after this date div
            dateandlegends_div = date_div.getparent()
            following_table = dateandlegends_div.getnext()

            if following_table is not None and following_table.tag == 'table':
                rows = following_table.xpath('.//tr')

                for row in rows:
                    source = row.xpath('./td[1]/@title')
                    hour = row.xpath('./td[2]/text()')
                    popularity = row.xpath('./td[3]/@class')
                    headline = row.xpath('./td[4]/a/@title')

                    headline_text = headline[0].strip() if headline else None
                    if headline_text:
                        data.append({
                            'date': formatted_date,
                            'source': source[0].strip() if source else None,
                            'hour': hour[0].strip() if hour else None,
                            'popularity': popularity[0] if popularity else None,
                            'headline': headline_text
                        })

        df = pd.DataFrame(data)
        print(f"Total headlines collected from search: {len(df)}")
        return df

    async def _search_and_get_page_source(self, browser, keyword: str, headers: dict = None) -> tuple[str, str]:
        """
        Search for a keyword and get page source using the shared browser.
        :param browser: Shared Playwright browser instance
        :param keyword: The keyword to search
        :return: tuple of (page source, search result URL)
        """
        session = read_session(session_name=self.__str__())
        cookies = read_cookies(cookies_name=self.__str__())

        os.makedirs("cookies", exist_ok=True)
        os.makedirs("sessions", exist_ok=True)

        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36',
            viewport=random.choice(VIEWPORTS),
            storage_state=session,
            ignore_https_errors=True
        )

        page = None
        try:
            if headers:
                await context.set_extra_http_headers(headers)
            if cookies:
                await context.add_cookies(cookies)

            page = await context.new_page()

            # Navigate to base URL
            base_url = 'https://mivzakim.net/view/category/17'
            await page.goto(base_url, timeout=100000)
            await page.wait_for_load_state('domcontentloaded')

            await perform_random_mouse_movements(page, min_actions=2, max_actions=5)

            # Find and fill search box
            search_input_xpath = '/html/body/div[1]/div[3]/div[1]/form/div/input[2]'
            search_result_url = None
            try:
                await page.wait_for_selector(f'xpath={search_input_xpath}', timeout=10000)

                search_input = page.locator(f'xpath={search_input_xpath}')
                await search_input.fill(keyword)
                await page.wait_for_timeout(50)

                await search_input.press('Enter')

                await page.wait_for_load_state('domcontentloaded', timeout=30000)
                await page.wait_for_timeout(50)

                search_result_url = page.url

                await page.keyboard.press("PageDown")
                await page.wait_for_load_state('domcontentloaded')

                print(f"Searched for '{keyword}', current URL: {search_result_url}")

            except Exception as e:
                print(f"Error during search for keyword '{keyword}': {e}")

            full_page_source = await page.content()

            with open(f"./cookies/{self.__str__()}-cookies.json", "w+") as f:
                f.write(json.dumps(await context.cookies()))

            update_session(new_data=await context.storage_state(), name=self.__str__())

            return full_page_source, search_result_url

        finally:
            if page:
                await page.close()
            await context.close()

    async def _get_page_source_from_url(self, browser, url: str, headers: dict = None) -> str:
        """
        Get page source from a specific URL (for pagination) using the shared browser.
        :param browser: Shared Playwright browser instance
        :param url: The URL to fetch
        :return: page source
        """
        session = read_session(session_name=self.__str__())
        cookies = read_cookies(cookies_name=self.__str__())

        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36',
            viewport=random.choice(VIEWPORTS),
            storage_state=session,
            ignore_https_errors=True
        )

        page = None
        try:
            if headers:
                await context.set_extra_http_headers(headers)
            if cookies:
                await context.add_cookies(cookies)

            page = await context.new_page()

            await page.goto(url, timeout=100000)
            await page.wait_for_load_state('domcontentloaded')

            await perform_random_mouse_movements(page, min_actions=2, max_actions=4)
            await page.keyboard.press("PageDown")
            await page.wait_for_timeout(50)

            full_page_source = await page.content()

            with open(f"./cookies/{self.__str__()}-cookies.json", "w+") as f:
                f.write(json.dumps(await context.cookies()))

            update_session(new_data=await context.storage_state(), name=self.__str__())

            return full_page_source

        finally:
            if page:
                await page.close()
            await context.close()

    async def scrape_from_search(self, browser, output_file: str = "../headlines_search.csv") -> pd.DataFrame:
        """
        Search for all keywords and extract headlines from multiple pages.
        Uses a shared browser instance. Saves results to CSV.
        :param browser: Shared Playwright browser instance
        :param output_file: Path to output CSV file
        :return: DataFrame with all results
        """
        all_dataframes = []

        for keyword in self.keywords:
            print(f"Searching for keyword: '{keyword}' across {self.num_pages} page(s)")
            try:
                # Perform search and get first page
                page_source, search_result_url = await asyncio.wait_for(
                    self._search_and_get_page_source(browser, keyword),
                    timeout=60.0
                )

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
                    paginated_url = f"{search_result_url}/page/{page_num}"
                    print(f"  Scraping page {page_num}: {paginated_url}")

                    try:
                        page_source = await asyncio.wait_for(
                            self._get_page_source_from_url(browser, paginated_url),
                            timeout=60.0
                        )

                        df_page = self._get_search_data(page_source)

                        if df_page.empty:
                            print(f"    Page {page_num} is empty, stopping pagination for '{keyword}'")
                            break

                        df_page['keyword'] = keyword
                        df_page['page'] = page_num
                        all_dataframes.append(df_page)

                        print(f"    Page {page_num}: Found {len(df_page)} headlines")

                        await asyncio.sleep(random.uniform(1, 2))

                    except asyncio.TimeoutError:
                        print(f"    !!! TIMEOUT: Page {page_num} for keyword '{keyword}'. Skipping.")
                        continue

                    except Exception as e:
                        print(f"    Error scraping page {page_num} for keyword '{keyword}': {e}")
                        break

                await asyncio.sleep(random.uniform(1, 2))

            except asyncio.TimeoutError:
                print(f"!!! TIMEOUT: Search for keyword '{keyword}'. Skipping.")
                continue

            except Exception as e:
                print(f"Error scraping keyword '{keyword}': {e}")
                continue

        # Combine all results
        if all_dataframes:
            df = pd.concat(all_dataframes, ignore_index=True).drop_duplicates(
            ).reset_index(drop=True).dropna(subset=["headline"])

            if not df.empty:
                if os.path.exists(output_file):
                    existing_df = pd.read_csv(output_file)
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates().reset_index(drop=True)
                    new_records = len(combined_df) - len(existing_df)
                    combined_df.to_csv(output_file, mode='w', header=True, index=False)
                    print(f"Added {new_records} new records to {output_file} (total: {len(combined_df)})")
                else:
                    df.to_csv(output_file, mode='w', header=True, index=False)
                    print(f"Created {output_file} with {len(df)} records")

            print(f"Total headlines found across all keywords and pages: {len(df)}")
            return df
        else:
            print("No data collected from search")
            return pd.DataFrame()
