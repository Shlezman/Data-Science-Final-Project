import json
import os
from datetime import datetime
import random

import pandas as pd
from lxml import html
from playwright.async_api import async_playwright

from utils import DATE_FORMAT, read_session, read_cookies, VIEWPORTS, perform_random_mouse_movements, update_session

class Scraper:
    def __init__(self, date: datetime, start_date: datetime = None):
        """
        Initialize scraper with a date
        :param date: The specific date to scrape
        :param start_date: Optional start date for reference (e.g., campaign start)
        """
        self.date = date.strftime(DATE_FORMAT)
        self.start_date = start_date.strftime(DATE_FORMAT) if start_date else None

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

        # Extract headlines using XPath - get title attribute
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

