import re
import shutil
import tempfile
import time
import traceback

import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager


# ==========================================
# SETTINGS
# ==========================================

URL = "https://www.imdb.com/chart/top/"
OUTPUT_FILE = "imdb_movies.csv"
DEBUG_HTML_FILE = "imdb_debug_page.html"


# ==========================================
# START CHROME
# ==========================================

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.add_argument("--disable-notifications")

# Use a fresh, throwaway Chrome profile every run. Without this, a Chrome
# process left over from a previous crashed/interrupted run keeps a lock on
# the default profile, which makes new launches fail or hang silently —
# this is the single most common cause of "Chrome won't open" with Selenium.
profile_dir = tempfile.mkdtemp(prefix="imdb_scraper_profile_")
options.add_argument(f"--user-data-dir={profile_dir}")

# Look like a normal browser, not an automated one
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)

driver = None

try:
    # webdriver-manager automatically downloads the ChromeDriver version
    # that matches your installed Chrome browser, so you never have to
    # manually download/place chromedriver.exe or fight PATH issues.
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
except Exception as e:
    print("webdriver-manager launch failed, trying Selenium's built-in driver manager instead...")
    print("(original error:", e, ")")
    try:
        # Selenium 4.6+ ships its own "Selenium Manager" that can fetch a
        # matching driver without webdriver-manager at all. Falling back to
        # this covers cases where webdriver-manager itself is the problem.
        driver = webdriver.Chrome(options=options)
    except Exception as e2:
        print()
        print("Chrome still failed to launch. Common causes:")
        print(" 1) A leftover chrome.exe / chromedriver.exe process is stuck — check Task Manager and end them")
        print(" 2) Google Chrome itself is not installed, or is a mismatched/very new version")
        print(" 3) An antivirus/firewall is blocking chromedriver.exe from running")
        print(" 4) Corporate/managed PC policy is blocking automated browser launches")
        print()
        print("Full error below:")
        traceback.print_exc()
        raise SystemExit(1)

driver.execute_cdp_cmd(
    "Page.addScriptToEvaluateOnNewDocument",
    {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
)

wait = WebDriverWait(driver, 20)


# ==========================================
# OPEN IMDb
# ==========================================

print("Opening IMDb Top 250...")
driver.get(URL)


# ==========================================
# HANDLE COOKIE / CONSENT OVERLAY (if present)
# ==========================================
# IMDb often shows a GDPR/consent popup that blocks the page from finishing
# rendering the chart. If it's not there, we just move on.

try:
    reject_or_accept = wait.until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "button[data-testid='reject-button'], button[data-testid='accept-button']")
        )
    )
    reject_or_accept.click()
    print("Dismissed cookie/consent banner.")
    time.sleep(1)
except TimeoutException:
    print("No cookie/consent banner detected (or it used a different selector).")


# ==========================================
# WAIT FOR MOVIE LIST TO ACTUALLY LOAD
# ==========================================

try:
    wait.until(
        EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "li.ipc-metadata-list-summary-item")
        )
    )
except TimeoutException:
    print("Movie list never appeared. Saving page source for debugging...")
    with open(DEBUG_HTML_FILE, "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    driver.save_screenshot("imdb_debug_screenshot.png")
    print(f"Saved {DEBUG_HTML_FILE} and imdb_debug_screenshot.png — open these to see what IMDb actually returned.")
    driver.quit()
    shutil.rmtree(profile_dir, ignore_errors=True)
    raise SystemExit("Could not load the movie list. Check the debug HTML/screenshot.")


# ==========================================
# FIND MOVIES
# ==========================================

movie_items = driver.find_elements(By.CSS_SELECTOR, "li.ipc-metadata-list-summary-item")
print("Movies found:", len(movie_items))


# ==========================================
# SCRAPE DATA (text-based, selector-independent)
# ==========================================
# IMDb renames internal CSS classes frequently, which is why relying on
# selectors like "h3" inside each row keeps breaking. Instead, we read the
# plain visible text of each row (which Selenium gives us regardless of the
# underlying markup) and pull out title / year / rating / votes with regex.
#
# A typical row's .text looks like:
#   "1. The Shawshank Redemption\n1994\n2h 22m\nR\n9.3\n(3.2M)"

YEAR_RE = re.compile(r"^\(?(\d{4})\)?$")
YEAR_ANYWHERE_RE = re.compile(r"(19|20)\d{2}")
RATING_RE = re.compile(r"^(\d\.\d)$")
VOTES_RE = re.compile(r"^\(([\d.]+[MK]?)\)$")
RANK_RE = re.compile(r"^#?\d+\.?$")
DURATION_RE = re.compile(r"^\d+h(\s*\d+m)?$|^\d+m$")
CONTENT_RATINGS = {
    "G", "PG", "PG-13", "R", "NC-17", "TV-Y", "TV-Y7", "TV-G", "TV-PG",
    "TV-14", "TV-MA", "TV-13", "NOT RATED", "UNRATED", "APPROVED", "PASSED", "GP",
}

data = []
debug_printed = False

for item in movie_items:
    try:
        raw_text = item.text.strip()
        if not raw_text:
            continue

        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        if not lines:
            continue

        year = ""
        rating = ""
        votes = ""
        rank = ""
        title_candidates = []

        for line in lines:
            if not year:
                m = YEAR_RE.match(line)
                if m:
                    year = m.group(1)
                    continue
            if not rating:
                m = RATING_RE.match(line)
                if m:
                    rating = m.group(1)
                    continue
            if not votes:
                m = VOTES_RE.match(line)
                if m:
                    votes = m.group(1)
                    continue
            if RANK_RE.match(line):
                if not rank:
                    rank = re.sub(r"\D", "", line)  # strip "#" / "." to leave just digits
                continue
            if DURATION_RE.match(line):
                continue
            if line.upper() in CONTENT_RATINGS:
                continue
            title_candidates.append(line)

        # The title is virtually always the longest remaining line — rank
        # badges, ratings, durations, and content-rating codes are all short.
        movie_name = max(title_candidates, key=len) if title_candidates else ""
        if ". " in movie_name and movie_name.split(". ", 1)[0].isdigit():
            movie_name = movie_name.split(". ", 1)[1]

        # Fallback: if no clean standalone year line was found, search the
        # whole row's text for a plausible 4-digit year.
        if not year:
            m = YEAR_ANYWHERE_RE.search(raw_text)
            if m:
                year = m.group(0)

        # ---- MOVIE URL ----
        # The whole row is wrapped in a link to the title page; grab the
        # first href containing "/title/" and normalize it to a full,
        # query-string-free IMDb URL.
        movie_url = ""
        try:
            for link in item.find_elements(By.CSS_SELECTOR, "a[href*='/title/']"):
                href = link.get_attribute("href")
                if href and "/title/" in href:
                    movie_url = href.split("?")[0]
                    break
        except NoSuchElementException:
            pass

        if not movie_name or (not year and not rating):
            if not debug_printed:
                print("\n--- DEBUG: could not fully parse this row. Raw text was: ---")
                print(repr(raw_text))
                print("--- END DEBUG ---\n")
                debug_printed = True
            print("Skipped movie: could not parse row text")
            continue

        data.append({
            "Rank": rank,
            "Movie Name": movie_name,
            "Year": year,
            "IMDb Rating": rating,
            "Vote Count": votes,
            "URL": movie_url,
        })

        print(rank, "|", movie_name, "|", year, "|", rating, "|", votes, "|", movie_url)

    except Exception as error:
        print("Skipped movie:", error)


# ==========================================
# CLOSE BROWSER
# ==========================================

driver.quit()
shutil.rmtree(profile_dir, ignore_errors=True)


# ==========================================
# SAVE CSV
# ==========================================

df = pd.DataFrame(data)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


# ==========================================
# RESULT
# ==========================================

print()
print("======================================")
print("SCRAPING COMPLETED")
print("======================================")
print("Total movies collected:", len(df))
print("CSV file:", OUTPUT_FILE)
print()
print("CSV preview:")
print(df.head(10).to_string(index=False))