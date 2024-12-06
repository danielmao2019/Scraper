from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import ssl
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time

import random


def get_random_user_agent():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.85 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.85 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.85 Safari/537.36"
    ]
    return random.choice(user_agents)


def get_soup_with_urllib(url: str, headers: dict):
    ssl_context = ssl.create_default_context()
    request = Request(url, headers=headers)
    response = urlopen(request, context=ssl_context)
    html = response.read().decode("utf-8")
    return BeautifulSoup(html, "html.parser")


def get_soup_with_requests(url: str, headers: dict):
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raises an HTTPError for bad responses
    return BeautifulSoup(response.text, "html.parser")


def get_soup_with_selenium(url: str):
    options = Options()
    options.add_argument("--headless")  # Run Chrome in headless mode
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"user-agent={get_random_user_agent()}")
    options.add_argument("accept-language=en-US,en;q=0.9")
    options.add_argument("upgrade-insecure-requests=1")
    options.add_argument("sec-ch-ua=\"Google Chrome\";v=\"118\", \"Not A(Brand\";v=\"8\", \"Chromium\";v=\"118\"")
    options.add_argument("sec-ch-ua-mobile=?0")
    options.add_argument("sec-ch-ua-platform=\"Windows\"")
    options.add_argument("dnt=1")  # Do Not Track
    options.add_argument("accept=text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8")

    # Update the path to your ChromeDriver executable
    driver = webdriver.Chrome(service=Service("/usr/local/bin/chromedriver"), options=options)
    try:
        driver.get(url)
        time.sleep(5)  # Allow time for the page to load
        html = driver.page_source
        return BeautifulSoup(html, "html.parser")
    finally:
        driver.quit()


def get_soup_with_api(api_key: str, article_id: str):
    url = f"https://api.elsevier.com/content/article/pii/{article_id}"
    headers = {
        'Accept': 'application/json',
        'X-ELS-APIKey': api_key,
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()  # Return raw JSON data (not soup)


def get_soup(url: str, api_key: str = None, article_id: str = None):
    assert type(url) == str, f"type(url)={type(url)}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    }

    if (
        url.startswith("https://dl.acm.org/doi") or
        url.startswith("https://ieeexplore.ieee.org") or
        url.startswith("https://www.researchgate.net") or
        url.startswith("https://www.sciencedirect.com")
    ):
        try:
            return get_soup_with_selenium(url)
        except Exception as e3:
            print(f"Failed with Selenium: {e3}")

    # Try urllib.request
    try:
        return get_soup_with_urllib(url, headers)
    except Exception as e1:
        print(f"Failed with urllib.request: {e1}")

    # Try requests
    try:
        return get_soup_with_requests(url, headers)
    except Exception as e2:
        print(f"Failed with requests: {e2}")

    # Try Selenium
    try:
        return get_soup_with_selenium(url)
    except Exception as e3:
        print(f"Failed with Selenium: {e3}")

    # Try Elsevier API (requires api_key and article_id)
    if api_key and article_id:
        try:
            return get_soup_with_api(api_key, article_id)
        except Exception as e4:
            print(f"Failed with Elsevier API: {e4}")

    # Raise an error if all approaches fail
    raise RuntimeError("Failed to fetch the page using all methods.")
