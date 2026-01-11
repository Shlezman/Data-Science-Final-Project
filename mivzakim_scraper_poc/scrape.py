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


async def scrape_single_date(date_obj: datetime, pages: int =100) -> pd.DataFrame:
    """
    Scrape data for a single date
    :param date_obj: datetime object to scrape
    :return: DataFrame with scraped data
    """
    try:
        # Create scraper instance for this date
        scraper = Scraper(date_obj, num_pages = pages)

        # Get data
        df = await scraper.scrape_from_page()

        print(f"Completed scraping for date: {date_obj.strftime(DATE_FORMAT)}")
        return df

    except Exception as e:
        print(f"Error scraping date {date_obj.strftime(DATE_FORMAT)}: {e}")
        return pd.DataFrame()


async def scrape_and_save(dates: list, pages: int =100) -> None:
    """
    Main flow of the collecting process - scrapes all dates concurrently
    :param dates: list of datetime objects to scrape
    """
    print(f"Starting concurrent scraping for {len(dates)} dates...")

    # Create tasks for all dates to run concurrently
    tasks = [scrape_single_date(date, pages) for date in dates]

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


def get_data(start_date: datetime = None, days: int = 7, pages: int =100) -> None:
    """
    Running the main flow of collecting all headlines data
    :param start_date: Optional start date for the scraping campaign (defaults to today)
    :param days: Number of days to scrape backwards from start_date
    :return: headlines.csv file
    """
    if start_date is None:
        start_date = datetime.now()

    # Get dates for past N days from start_date
    dates = [start_date - timedelta(days=i) for i in range(days)]

    print(f"Scraping {days} days starting from {start_date.strftime(DATE_FORMAT)}")

    # Run async scraping
    asyncio.run(scrape_and_save(dates, pages))

    # Final cleanup
    purge()
    print("Data collection complete!")


def get_search_data(keywords: set, num_pages: int = 1) -> None:
    """
    Running the main flow of collecting headlines using keyword search
    :param keywords: Set of keywords to search for
    :param start_date: Optional start date for the scraping campaign (defaults to today)
    :return: headlines_search.csv file
    """

    # Get dates for past N days from start_date

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
    :param start_date: Optional start date for reference
    :return: DataFrame with scraped data
    """
    try:
        # Create search scraper instance
        scraper = SearchScraper(date_obj, keywords, num_pages)

        # Get data through search
        df = await scraper.scrape_from_search() #xpath='/html/body/div[1]/div[4]/div[1]/div[3]/table/tbody/tr/td[4]/a'

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
    :param start_date: Optional start date for the scraping campaign
    """
    print(f"Starting concurrent search scraping with keywords: {keywords}")

    # Create tasks for all dates to run concurrently

    # Run all tasks concurrently and gather results
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