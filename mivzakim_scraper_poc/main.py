from datetime import datetime
from scrape import get_data, get_search_data

if __name__ == "__main__":
# Example 1: Regular scraping with custom start date
#     custom_start = datetime(2025, 12, 1)
#     get_data(start_date=custom_start, days=10)
# Example 2: scraping with custom search words
    search_keywords = {'שוק','בחירות'}
    get_search_data(keywords=search_keywords, num_pages=2)