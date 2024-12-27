from typing import Optional
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
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.85 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.5479.52 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.5479.52 Safari/537.36",
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
    # Set Chrome options
    options = Options()
    options.add_argument("--headless=new")  # Updated headless mode for Chrome 131
    options.add_argument("--disable-gpu")  # Disable GPU acceleration
    options.add_argument("--no-sandbox")  # Bypass OS security model
    options.add_argument("--disable-dev-shm-usage")  # Prevent /dev/shm issues in Docker/Linux
    options.add_argument(f"user-agent={get_random_user_agent()}")  # Use a random user-agent
    options.add_argument("accept-language=en-US,en;q=0.9")  # Set preferred languages
    options.add_argument("upgrade-insecure-requests=1")  # Accept HTTPS upgrades
    options.add_argument("dnt=1")  # Do Not Track
    options.add_argument("--disable-blink-features=AutomationControlled")  # Avoid detection
    options.add_argument("--remote-debugging-port=9222")  # For debugging
    options.add_argument("--window-size=1920,1080")  # Set a large enough virtual screen size

    # Update the path to your ChromeDriver executable
    driver = webdriver.Chrome(service=Service("/usr/local/bin/chromedriver"), options=options)
    try:
        driver.get(url)
        time.sleep(5)  # Allow time for the page to load
        html = driver.page_source
        return BeautifulSoup(html, "html.parser")
    finally:
        driver.quit()


def get_soup(url: str, method: Optional[str] = None):
    assert type(url) == str, f"type(url)={type(url)}"
    if method is not None:
        assert type(method) == str, f"{type(method)=}"
        assert method in ['urllib', 'requests', 'selenium'], f"{method=}"
        return globals()[f"get_soup_with_{method}"](url)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    }

    if (
        url.startswith("https://dl.acm.org/doi") or
        url.startswith("https://ieeexplore.ieee.org") or
        url.startswith("https://www.researchgate.net") or
        url.startswith("https://journals.scholarsportal.info") or
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

    # Raise an error if all approaches fail
    raise RuntimeError("Failed to fetch the page using all methods.")
