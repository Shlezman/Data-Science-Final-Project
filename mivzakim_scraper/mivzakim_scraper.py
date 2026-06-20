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

    async def _get_page_source(self, browser, url, response_url=None,
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

        # Assign context inside try so that if browser.new_context() raises
        # (e.g. the browser process crashed), the finally clause does not
        # attempt to call .close() on an undefined name and mask the original error.
        context = None
        page = None
        try:
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

            if response_url:
                page.on('response', _handle_response)

            # ניווט ופעולות
            await page.goto(url, timeout=100000)
            await page.wait_for_load_state('domcontentloaded')
            await perform_random_mouse_movements(page)
            await page.wait_for_timeout(2000)
            await page.keyboard.press("PageDown")
            await page.wait_for_load_state('domcontentloaded')

            full_page_source = await page.content()

            # שמירת נתונים
            with open(f"./cookies/{self.__str__()}-cookies.json", "w+") as f:
                f.write(json.dumps(await context.cookies()))

            update_session(new_data=await context.storage_state(), name=self.__str__())

            return full_page_source  # החזרת הערך בתוך ה-try (או בסוף הפונקציה), לא ב-finally

        finally:
            # סגירת העמוד וה-context בלבד. לא סוגרים את ה-browser!
            if page:
                await page.close()
            if context:
                await context.close()

    def _get_data(self, page_source: str) -> pd.DataFrame:
        """Extract table data from page source using XPath"""

        tree = html.fromstring(page_source)

        # Class-based row selection — robust to layout/nesting changes. A headline
        # row is any <tr> containing the title cell; this skips holytime/separator
        # rows and does NOT depend on a fixed /html/body/div[…] path (the old
        # absolute XPath silently broke when the site was redesigned).
        rows = tree.xpath('//tr[td[contains(@class, "nf_title")]]')

        data = []
        for row in rows:
            source = row.xpath('./td[contains(@class, "nf_feed")]/@title')
            hour = row.xpath('./td[contains(@class, "nf_time")]/text()')
            importance = row.xpath('./td[starts-with(@class, "p")][1]/@class')
            # Headline lives in the <a title="…"> of the nf_title cell; fall back to
            # the link text if @title is absent.
            headline = row.xpath('./td[contains(@class, "nf_title")]/a/@title')
            if not headline:
                headline = row.xpath('./td[contains(@class, "nf_title")]/a//text()')

            data.append({
                'date': self.date,
                'source': source[0].strip() if source else None,
                'hour': hour[0].strip() if hour else None,
                'popularity': importance[0] if importance else None,
                'headline': headline[0].strip() if headline else None,
            })

        return pd.DataFrame(data)

    async def scrape_from_page(self, browser, response_url=None, headers=None, output_file: str | None = None) -> pd.DataFrame:
        """
        Extract the data from the website.

        Writes to a PER-DATE file (``../headlines_<date>.csv`` by default) so
        concurrent dates in a batch never read-modify-write the same file and
        clobber each other (the prior shared-``headlines.csv`` race that lost most
        of every batch). The caller (scrape_dates) globs + concatenates these.
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
                page_source = await asyncio.wait_for(
                    self._get_page_source(browser, url=paginated_url, response_url=response_url, headers=headers),
                    timeout=60.0
                )
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

            except asyncio.TimeoutError:
                print(f"    !!! TIMEOUT: Page {page_num} took too long using shared browser. Skipping.")
                continue

            except Exception as e:
                print(f"    Error scraping page {page_num}: {e}")
                break

        # Guard against every page timing out — pd.concat([]) raises ValueError.
        if not all_dataframes:
            print("No pages returned data — nothing to write")
            return pd.DataFrame()

        raw_rows = sum(len(d) for d in all_dataframes)
        df = pd.concat(all_dataframes, ignore_index=True).drop_duplicates().reset_index(drop=True)
        df = df.dropna(subset=["headline"])
        df = df[df["headline"].astype(str).str.strip() != ""]

        if df.empty:
            # Rows were parsed but every headline is empty → selector/markup mismatch.
            # Warn LOUDLY (the old code silently wrote nothing, which read downstream as
            # "site exhausted, 0 inserted").
            if raw_rows > 0:
                print(f"!! WARNING: parsed {raw_rows} rows but ALL headlines empty for "
                      f"{self.date} — headline selector likely broken (site markup changed).")
            else:
                print(f"No data scraped for date: {self.date}")
            return pd.DataFrame()

        # Per-date file → no shared-file read-modify-write, so concurrent dates can't
        # clobber each other. scrape_dates globs + concatenates these.
        out = output_file or f"../headlines_{self.date}.csv"
        df.to_csv(out, mode='w', header=True, index=False, encoding="utf-8")
        print(f"Wrote {len(df)} headlines for {self.date} -> {out}")
        return df
