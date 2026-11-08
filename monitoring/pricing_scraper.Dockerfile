FROM python:3.12-slim

WORKDIR /app
COPY pricing_scraper.py .

CMD ["python3", "pricing_scraper.py"]