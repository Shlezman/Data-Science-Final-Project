from datetime import datetime
from scrape import get_data

if __name__ == "__main__":
    # Will process 3650 days in batches of 30 (122 batches total)
    get_data(days=3650, pages=100, batch_size=30)

    # Or use smaller batches if still having issues
    # get_data(start_date=custom_start, days=3650, pages=100, batch_size=15)