from datetime import datetime
from scrape import get_data

if __name__ == "__main__":
    resume_date = datetime(2025, 5, 27)
    get_data(start_date=resume_date, days=3450, pages=100, batch_size=10)