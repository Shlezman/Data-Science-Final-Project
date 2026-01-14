import json
import os
from datetime import datetime
import random

import asyncio
import pandas as pd
from lxml import html
from playwright.async_api import async_playwright

from utils import DATE_FORMAT, read_session, read_cookies, VIEWPORTS, perform_random_mouse_movements, update_session

SOURCE = '/html/body/div[1]/div[4]/div[1]/div[3]/table/tbody/tr/td[1]'  # title
# title
HEADLINE_PATH = '/html/body/div[1]/div[4]/div[1]/div[3]/table/tbody/tr/td[4]/a'
# class name
IMPORTANCE_LEVEL = '/html/body/div[1]/div[4]/div[1]/div[3]/table/tbody/tr/td[3]'
HOUR = '/html/body/div[1]/div[4]/div[1]/div[3]/table/tbody/tr/td[2]'  # content
BASE_URL = 'https://mivzakim.net/view/category/1/date/'


class Scraper:
    def __init__(self, date: datetime, start_date: datetime = None, num_pages: int = 2):
        """
        Initialize scraper with a date
        :param date: The specific date to scrape
        :param start_date: Optional start date for reference (e.g., campaign start)
        """
        self.date = date.strftime(DATE_FORMAT)
        self.start_date = start_date.strftime(
            DATE_FORMAT) if start_date else None
        self.num_pages = num_pages

    def __str__(self):
        """String representation for session/cookie naming"""
        return f"scraper_{self.date}"

    @staticmethod
    def create_url(date: datetime, base_url: str = BASE_URL) -> str:
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
                headless=True,
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
            try:

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
            finally:
                await browser.close()
                return full_page_source

    def _get_data(self, page_source: str) -> pd.DataFrame:
        """Extract table data from page source using XPath"""

        tree = html.fromstring(page_source)

        # Base XPath to rows (handle tbody absence)
        rows = tree.xpath(
            '/html/body/div[1]/div[4]/div[1]/div[3]/table/tbody/tr')
        if not rows:
            rows = tree.xpath(
                '/html/body/div[1]/div[4]/div[1]/div[3]/table/tr')

        data = []

        for row in rows:
            # Extract per row
            source = row.xpath('./td[1]/@title')
            hour = row.xpath('./td[2]/text()')
            importance = row.xpath('./td[3]/@class')
            headline = row.xpath('./td[4]/a/@title')

            data.append({
                'date': self.date,
                'source': source[0].strip() if source else None,
                'hour': hour[0].strip() if hour else None,
                'importance_level': importance[0] if importance else None,
                'headline': headline[0].strip() if headline else None
            })

        return pd.DataFrame(data)

    async def scrape_from_page(self, response_url=None, headers=None, output_file: str = "../headlines.csv") -> pd.DataFrame:
        """
        Extract the data from the website
        """
        # Convert date string back to datetime for create_url
        date_obj = datetime.strptime(self.date, DATE_FORMAT)
        url = self.create_url(date_obj)
        all_dataframes = []
        for page_num in range(1, self.num_pages):
            # Construct pagination URL
            paginated_url = f"{url}/page/{page_num}"

            print(f"  Scraping page {page_num}: {paginated_url}")

            try:
                page_source = await self._get_page_source(url=paginated_url, response_url=response_url,
                                                          headers=headers)
                # Get page source for this page

                # Extract data
                df_page = self._get_data(page_source)

                if df_page.empty:
                    print(f"Page {page_num} is empty")
                    break

                all_dataframes.append(df_page)

                print(f"Page {page_num}: Found {len(df_page)} headlines")

                # Small delay between pages
                await asyncio.sleep(random.uniform(1, 2))

            except Exception as e:
                print(f"    Error scraping page {page_num}: {e}")
                break

        df = pd.concat(all_dataframes, ignore_index=True).drop_duplicates(
        ).reset_index(drop=True).dropna(subset=["headline"])
        if not df.empty:
            # Check if file exists
            if os.path.exists(output_file):
                # Read existing data
                existing_df = pd.read_csv(output_file)

                # Combine with new data and remove duplicates
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates().reset_index(drop=True)

                # Calculate how many new records were added
                new_records = len(combined_df) - len(existing_df)

                # Save back to file
                combined_df.to_csv(output_file, mode='w',
                                   header=True, index=False)
                print(
                    f"Added {new_records} new records to {output_file} (total: {len(combined_df)})")
            else:
                # Create new file with header
                df = df.drop_duplicates().reset_index(drop=True)
                df.to_csv(output_file, mode='w', header=True, index=False)
                print(f"Created {output_file} with {len(df)} records")

            print(
                f"Completed scraping for date: {date_obj.strftime(DATE_FORMAT)}")
        else:
            print(
                f"No data scraped for date: {date_obj.strftime(DATE_FORMAT)}")

        return df
