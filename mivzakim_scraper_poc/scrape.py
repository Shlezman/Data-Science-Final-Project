import os
from datetime import datetime, timedelta

import asyncio
import pandas as pd

from mivzakim_search_scraper import SearchScraper
from utils import DATE_FORMAT

from mivazakim_scraper import Scraper


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


async def scrape_single_date(date_obj: datetime, pages: int = 100) -> None:
    """
    Scrape data for a single date and write directly to file
    :param date_obj: datetime object to scrape
    :param pages: number of pages to scrape
    """
    try:
        # Create scraper instance for this date
        scraper = Scraper(date_obj, num_pages=pages)

        # Scrape and save directly to file (no return value needed)
        await scraper.scrape_from_page()

        print(f"Completed scraping for date: {date_obj.strftime(DATE_FORMAT)}")

    except Exception as e:
        print(f"Error scraping date {date_obj.strftime(DATE_FORMAT)}: {e}")


async def scrape_batch(dates: list, pages: int = 100) -> None:
    """
    Scrape a batch of dates concurrently (each writes directly to file)
    :param dates: list of datetime objects to scrape
    :param pages: number of pages to scrape per date
    """
    print(f"Starting concurrent scraping for {len(dates)} dates in this batch...")

    # Create tasks for all dates to run concurrently
    tasks = [scrape_single_date(date, pages) for date in dates]

    # Run all tasks concurrently
    await asyncio.gather(*tasks, return_exceptions=True)

    print("Batch completed")

########## search #############

def get_data(start_date: datetime = None, days: int = 7, pages: int = 100, batch_size: int = 30) -> None:
    """
    Running the main flow of collecting all headlines data in batches
    :param start_date: Optional start date for the scraping campaign (defaults to today)
    :param days: Number of days to scrape backwards from start_date
    :param pages: Number of pages to scrape per date
    :param batch_size: Number of days to process in each batch (default 30)
    :return: headlines.csv file
    """
    if start_date is None:
        start_date = datetime.now()

    # Get dates for past N days from start_date
    all_dates = [start_date - timedelta(days=i) for i in range(days)]

    print(f"Scraping {days} days starting from {start_date.strftime(DATE_FORMAT)}")
    print(f"Processing in batches of {batch_size} days")

    # Split dates into batches
    total_batches = (len(all_dates) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(all_dates))
        batch_dates = all_dates[start_idx:end_idx]

        print(f"\n{'=' * 60}")
        print(f"Processing batch {batch_num + 1}/{total_batches}")
        print(f"Dates: {batch_dates[-1].strftime(DATE_FORMAT)} to {batch_dates[0].strftime(DATE_FORMAT)}")
        print(f"{'=' * 60}\n")

        # Run async scraping for this batch
        asyncio.run(scrape_batch(batch_dates, pages))

        # Small delay between batches
        if batch_num < total_batches - 1:
            print(f"\nWaiting 5 seconds before next batch...\n")
            import time
            purge()
            time.sleep(5)

    # Final cleanup
    purge()
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)


def get_search_data(keywords: set, num_pages: int = 1) -> None:
    """
    Running the main flow of collecting headlines using keyword search
    :param keywords: Set of keywords to search for
    :param num_pages: Number of pages to scrape
    :return: headlines_search.csv file
    """

    print(f"Searching for keywords {keywords}")

    # Run async scraping with search
    asyncio.run(scrape_and_save_search([datetime.now()], keywords, num_pages))

    # Final cleanup
    purge()
    print("Search data collection complete!")


async def scrape_search_keywords(date_obj: datetime, keywords: set, num_pages: int = 1) -> pd.DataFrame:
    """
    Scrape data for a single date using keyword search
    :param date_obj: datetime object to scrape
    :param keywords: Set of keywords to search
    :return: DataFrame with scraped data
    """
    try:
        # Create search scraper instance
        scraper = SearchScraper(date_obj, keywords, num_pages)

        # Get data through search
        df = await scraper.scrape_from_search()

        print(f"Completed search scraping for date: {date_obj.strftime(DATE_FORMAT)}")
        return df

    except Exception as e:
        print(f"Error search scraping date {date_obj.strftime(DATE_FORMAT)}: {e}")
        return pd.DataFrame()


async def scrape_and_save_search(dates: list, keywords: set, num_pages: int = 1) -> None:
    """
    Main flow for search-based scraping - scrapes all dates concurrently with keyword search
    :param dates: list of datetime objects to scrape
    :param keywords: Set of keywords to search for
    :param num_pages: Number of pages to scrape
    """
    print(f"Starting concurrent search scraping with keywords: {keywords}")

    # Run search for the date
    results = await scrape_search_keywords(dates[0], keywords, num_pages)

    df = results.drop_duplicates().reset_index(drop=True)

    # Load existing data if file exists
    file_name = "headlines_search.csv"
    if os.path.exists(file_name):
        existing_df = pd.read_csv(file_name)
        df = pd.concat([existing_df, df]).drop_duplicates().reset_index(drop=True)

    # Save to CSV
    df.to_csv(file_name, index=False)
    print(f"Search data written to {file_name}")
    print(f"Total unique headlines: {len(df)}")
    print("All search tasks completed")