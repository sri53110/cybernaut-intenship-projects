import os
import time
import pandas as pd

from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# ============================================================
# SETTINGS
# ============================================================

URL = "https://www.coingecko.com/"

CURRENT_FILE = "crypto-price.csv"
HISTORY_FILE = "crypto-price-history.csv"

TOP_COINS = 10
WAIT_TIME = 30


# ============================================================
# CHROME SETTINGS
# ============================================================

options = webdriver.ChromeOptions()

# Run Chrome in background
options.add_argument("--headless=new")

options.add_argument("--window-size=1920,1080")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-notifications")
options.add_argument("--disable-popup-blocking")

# Reduce unnecessary Chrome messages
options.add_argument("--log-level=3")

# User agent
options.add_argument(
    "--user-agent=Mozilla/5.0 "
    "(Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)


# ============================================================
# START PROGRAM
# ============================================================

print("=" * 90)
print("              COINGECKO CRYPTOCURRENCY PRICE TRACKER")
print("=" * 90)

print("\nStarting Chrome in HEADLESS mode...")
print("Chrome window will NOT appear.\n")


# ============================================================
# START SELENIUM
# ============================================================

driver = webdriver.Chrome(
    service=Service(
        ChromeDriverManager().install()
    ),
    options=options
)

crypto_data = []


try:

    # ========================================================
    # OPEN COINGECKO
    # ========================================================

    print("Opening CoinGecko...")

    driver.get(URL)

    wait = WebDriverWait(
        driver,
        WAIT_TIME
    )

    # Wait for table
    wait.until(
        EC.presence_of_element_located(
            (By.TAG_NAME, "table")
        )
    )

    # Allow JavaScript content to finish
    time.sleep(5)

    print("CoinGecko loaded successfully.")


    # ========================================================
    # FIND TABLE
    # ========================================================

    print("\nSearching cryptocurrency table...")

    rows = driver.find_elements(
        By.CSS_SELECTOR,
        "table tbody tr"
    )

    print(
        "Rows found:",
        len(rows)
    )


    # ========================================================
    # SCRAPE TOP COINS
    # ========================================================

    for row in rows:

cddcdccccccc

            cells = row.find_elements(
                By.TAG_NAME,
                "td"
            )

            if not cells:
                continue


            # ------------------------------------------------
            # GET TEXT FROM ALL CELLS
            # ------------------------------------------------

            cell_values = []

            for cell in cells:

                text = cell.text.strip()

                if text:
                    cell_values.append(text)


            if not cell_values:
                continue


            # ------------------------------------------------
            # COIN NAME
            # ------------------------------------------------

            coin_name = "N/A"

            links = row.find_elements(
                By.TAG_NAME,
                "a"
            )

            for link in links:

                text = link.text.strip()

                if not text:
                    continue

                if text.lower() in [
                    "buy",
                    "trade"
                ]:
                    continue

                # Avoid links which are only symbols
                if len(text) > 1:

                    coin_name = text

                    break


            # ------------------------------------------------
            # FALLBACK COIN NAME
            # ------------------------------------------------

            if coin_name == "N/A":

                for value in cell_values:

                    value_lower = value.lower()

                    if (
                        "$" not in value
                        and "%" not in value
                        and value_lower not in [
                            "buy",
                            "trade"
                        ]
                    ):

                        coin_name = value

                        break


            # ------------------------------------------------
            # CURRENT PRICE
            # ------------------------------------------------

            current_price = "N/A"

            dollar_values = []

            for value in cell_values:

                value = value.strip()

                if value.startswith("$"):

                    dollar_values.append(value)


            if dollar_values:

                # First dollar value is normally
                # the current price

                current_price = dollar_values[0]


            # ------------------------------------------------
            # 24H CHANGE
            # ------------------------------------------------

            change_values = []

            for value in cell_values:

                value = value.strip()

                if "%" in value:

                    change_values.append(value)


            change_24h = "N/A"

            if len(change_values) >= 2:

                # Usually:
                # 1h
                # 24h
                # 7d

                change_24h = change_values[1]

            elif len(change_values) == 1:

                change_24h = change_values[0]


            # ------------------------------------------------
            # MARKET CAP
            # ------------------------------------------------

            market_cap = "N/A"

            if len(dollar_values) >= 2:

                # Market cap is generally
                # near the end of the row

                market_cap = dollar_values[-2]


            # ------------------------------------------------
            # 24H VOLUME
            # ------------------------------------------------

            volume_24h = "N/A"

            if len(dollar_values) >= 3:

                # Normally:
                # Price
                # Volume
                # Market Cap
                # FDV

                volume_24h = dollar_values[-3]


            # ------------------------------------------------
            # FULL ROW TEXT
            # ------------------------------------------------

            full_data = " | ".join(
                cell_values
            )


            # ------------------------------------------------
            # TIMESTAMP
            # ------------------------------------------------

            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )


            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if coin_name == "N/A":
                continue


            # ------------------------------------------------
            # STORE DATA
            # ------------------------------------------------

            crypto_data.append({

                "Rank": len(crypto_data) + 1,

                "Coin Name": coin_name,

                "Current Price": current_price,

                "24h Change": change_24h,

                "24h Volume": volume_24h,

                "Market Cap": market_cap,

                "Timestamp": timestamp

            })


        except Exception as error:

            print(
                "Error reading row:",
                error
            )

            continue


finally:

    # ========================================================
    # CLOSE CHROME
    # ========================================================

    driver.quit()

    print(
        "\nChrome background process closed."
    )


# ============================================================
# CHECK DATA
# ============================================================

if not crypto_data:

    print("\n" + "=" * 90)
    print("ERROR")
    print("=" * 90)

    print(
        "No cryptocurrency data was extracted."
    )

    print(
        "CoinGecko webpage structure may have changed."
    )

    exit()


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    crypto_data
)


# ============================================================
# SAVE CURRENT DATA
# ============================================================

# crypto-price.csv contains ONLY the latest scraping result

df.to_csv(
    CURRENT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# SAVE HISTORICAL DATA
# ============================================================

# crypto-price-history.csv keeps ALL previous runs

if os.path.exists(HISTORY_FILE):

    df.to_csv(
        HISTORY_FILE,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8"
    )

else:

    df.to_csv(
        HISTORY_FILE,
        mode="w",
        header=True,
        index=False,
        encoding="utf-8"
    )


# ============================================================
# DISPLAY OUTPUT
# ============================================================

print("\n")
print("=" * 120)
print("                    CURRENT CRYPTOCURRENCY DATA")
print("=" * 120)

print(
    df.to_string(index=False)
)


# ============================================================
# FILE INFORMATION
# ============================================================

print("\n")
print("=" * 120)
print("                         SCRAPING SUCCESS")
print("=" * 120)

print(
    f"\nCoins scraped       : {len(df)}"
)

print(
    f"Current data file   : {os.path.abspath(CURRENT_FILE)}"
)

print(
    f"History data file   : {os.path.abspath(HISTORY_FILE)}"
)

print(
    "\nCurrent CSV         : crypto-price.csv"
)

print(
    "History CSV         : crypto-price-history.csv"
)

print("\n" + "=" * 120)
print("                         PROGRAM COMPLETED")
print("=" * 120)