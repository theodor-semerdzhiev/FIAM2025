import feedparser
import newspaper
import time
import os
import random
from urllib.parse import urlparse
import socket # For specific error checking
from urllib.error import URLError # For specific error checking
import logging # Use logging for cleaner output control

# --- Configuration ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("newspaper").setLevel(logging.WARNING) # Quieten newspaper's internal logs unless it's a warning/error

# Base output directory
# !! IMPORTANT: Update this path to where your regional folders are !!
# Example: BASE_OUTPUT_DIR = r'C:\_Files\School\Competitions\FIAM2025\data\regional_tuning_data'
BASE_OUTPUT_DIR = r'D:\market_data\text_data' # <-- MAKE SURE THIS PATH IS CORRECT
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
logging.info(f"Base output directory: {BASE_OUTPUT_DIR}")

# Define groups, their countries (codes from Appendix A), feeds, and primary language
# We aim for mostly English sources where possible, except where native language is needed for translation step
# Structure: 'GROUP_NAME': { 'codes': ['COUNTRY_CODE', ...], 'feeds': [('rss_url', 'language_code'), ...]}
# Note: language_code is for newspaper3k's extractor hint

# !!! ADD MANY MORE FEEDS HERE FOR EACH REGION TO MAXIMIZE VOLUME !!!
# Search for "[Publication Name] Business RSS feed", "[Country] financial news RSS", etc.
REGIONAL_GROUPS = {
    'USA': {
        'codes': ['USA'],
        'feeds': [
            ('http://feeds.reuters.com/reuters/businessNews', 'en'),
            ('https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml', 'en'),
            ('https://www.cnbc.com/id/10000115/device/rss/rss.html', 'en'),
            ('https://www.nytimes.com/svc/collections/v1/publish/https://www.nytimes.com/section/business/rss.xml', 'en'),
            ('https://www.marketwatch.com/rss/topstories', 'en'),
            ('https://seekingalpha.com/feed.xml', 'en'),
            ('https://www.economist.com/finance-and-economics/rss.xml', 'en'),
            ('https://www.economist.com/business/rss.xml', 'en'),
            ('https://www.ft.com/rss/home', 'en'),
            ('https://moxie.foxbusiness.com/google-publisher/latest.xml', 'en'),
            ('https://moxie.foxbusiness.com/google-publisher/economy.xml', 'en'),
            ('https://moxie.foxbusiness.com/google-publisher/markets.xml', 'en'),
            ('https://www.nasdaq.com/feed/nasdaq-original/rss.xml', 'en'),
            ('https://www.nasdaq.com/feed/rssoutbound?category=Markets', 'en'),
            ('https://www.nasdaq.com/feed/rssoutbound?category=Technology', 'en'),
            ('https://www.federalreserve.gov/feeds/press_all.xml', 'en'),
            ('https://www.federalreserve.gov/feeds/h41.xml', 'en')



            # Add more: Bloomberg, Fox Business, Investor's Business Daily, etc.
        ]
    },
    'CAN': {
        'codes': ['CAN'],
        'feeds': [
            ('https://www.bnnbloomberg.ca/rss/news', 'en'),
            ('https://financialpost.com/feed/', 'en'),
            ('https://www.theglobeandmail.com/business/rss/rob-commentary/', 'en'),
            ('https://www.theglobeandmail.com/business/rss/industry-news/', 'en'),
            ('https://www.cbc.ca/cmlink/rss-business', 'en'),
            ('https://www.theglobeandmail.com/arc/outboundfeeds/rss/category/business/', 'en'),
            ('https://www.theglobeandmail.com/arc/outboundfeeds/rss/category/business/economy/', 'en'),
            ('https://financialpost.com/category/news/economy/', 'en')
        ]
    },
    'MEX': { # Will require translation later
        'codes': ['MEX'],
        'feeds': [
            ('https://www.elfinanciero.com.mx/rss/rss-ultimas-noticias.xml', 'es'),
            ('https://expansion.mx/rss/economia', 'es'),
            ('https://www.eleconomista.com.mx/rss/empresas.xml', 'es'),
            ('https://www.reforma.com/rss/negocios.xml', 'es'),
            # Add more Mexican sources
        ]
    },
    'GBR': {
        'codes': ['GBR'],
        'feeds': [
            ('http://feeds.reuters.com/reuters/UKBusinessNews', 'en'),
            ('https://feeds.bbci.co.uk/news/business/rss.xml', 'en'),
            ('https://www.theguardian.com/uk/business/rss', 'en'),
            ('https://www.telegraph.co.uk/business/rss.xml', 'en'),
            ('https://www.independent.co.uk/news/business/rss', 'en'),
            ('http://news.sky.com/feeds/rss/business.xml', 'en'),
            ('https://www.ft.com/rss/home/uk', 'en'), # Often paywalled
            ('http://www.telegraph.co.uk/finance/rss', 'en')

            # Add more UK sources (Sky News Business, etc.)
        ]
    },
    'IRL': {
        'codes': ['IRL'],
        'feeds': [
            ('https://www.irishtimes.com/business/rss', 'en'),
            ('https://www.independent.ie/business/rss/', 'en'),
            ('https://www.rte.ie/news/business/rss.xml', 'en'),
            # Add more Irish sources
        ]
    },
    'ANZ': {
        'codes': ['AUS', 'NZL'],
        'feeds': [
            # Australia
            ('https://www.afr.com/rss/markets', 'en'),
            ('https://www.afr.com/rss/companies', 'en'),
            ('https://www.smh.com.au/rss/business.xml', 'en'),
            ('https://www.theage.com.au/rss/business.xml', 'en'),
            ('https://www.theaustralian.com.au/business/rss', 'en'),
            ('https://www.abc.net.au/news/feed/51120/rss.xml', 'en'), # ABC Business
            # New Zealand
            ('https://www.nzherald.co.nz/rss/topic/business/', 'en'),
            ('https://www.stuff.co.nz/rss/business', 'en'),
            ('https://businessdesk.co.nz/rss', 'en'),
            ('https://www.rnz.co.nz/rss/business.xml', 'en'), # RNZ Business
            # Add more ANZ sources
        ]
    },
    'DACH': { # Germany, Austria, Switzerland
        'codes': ['DEU', 'AUT', 'CHE'],
        'feeds': [
             # Germany
            ('https://www.handelsblatt.com/rss/finanzen/', 'de'),
            ('https://www.handelsblatt.com/rss/unternehmen/', 'de'),
            ('https://www.wiwo.de/rss/finanzen', 'de'),
            ('https://www.boerse-zeitung.de/rss/bz-rss.xml', 'de'),
            ('https://www.faz.net/rss/aktuell/wirtschaft/', 'de'), # FAZ Wirtschaft
            # Austria
            ('https://www.diepresse.com/rss/Wirtschaft', 'de'),
            ('https://www.derstandard.at/rss/?ressort=Wirtschaft', 'de'),
            ('https://kurier.at/wirtschaft/rss', 'de'),
            # Switzerland (German part)
            ('https://www.nzz.ch/wirtschaft.rss', 'de'),
            ('https://www.tagesanzeiger.ch/wirtschaft/rss.xml', 'de'),
            # Switzerland (French part)
            ('https://www.letemps.ch/economie/rss', 'fr'),
            # Add more DACH sources
        ]
    },
    'FRA': {
        'codes': ['FRA'],
        'feeds': [
            ('https://www.lesechos.fr/rss/rss_finance-marches.xml', 'fr'),
            ('https://www.lesechos.fr/rss/rss_entreprises.xml', 'fr'),
            ('https://www.latribune.fr/finance/feed.xml', 'fr'),
            ('https://www.lefigaro.fr/rss/figaro_economie.xml', 'fr'),
            ('https://www.lemonde.fr/economie/rss_full.xml', 'fr'),
            # Add more French sources
        ]
    },
    'BENELUX': { # Belgium, Netherlands, Luxembourg
        'codes': ['BEL', 'NLD', 'LUX'],
        'feeds': [
            # Belgium
            ('https://www.lecho.be/rss.xml', 'fr'),
            ('https://www.tijd.be/rss.xml', 'nl'),
            # Netherlands
            ('https://fd.nl/rss', 'nl'),
            ('https://www.rtlnieuws.nl/rss/rtlz.xml', 'nl'),
            ('https://www.nrc.nl/rss/economie/', 'nl'), # NRC Economy
            # Luxembourg
            ('https://www.luxtimes.lu/en/rss/business', 'en'),
            ('https://paperjam.lu/rss.xml', 'fr'),
            ('https://www.wort.lu/de/business/rss', 'de'),
            # Add more Benelux sources
        ]
    },
    'NORDICS': { # Denmark, Finland, Norway, Sweden
        'codes': ['DNK', 'FIN', 'NOR', 'SWE'],
        'feeds': [
            ('https://borsen.dk/rss/', 'da'), # Denmark
            ('https://www.kauppalehti.fi/rss', 'fi'), # Finland
            ('https://www.dn.no/rss/', 'no'), # Norway
            ('https://www.di.se/rss', 'sv'), # Sweden
            # Add more Nordic sources (e.g., business sections of national broadcasters/papers)
        ]
    },
    'SOUTHERN_EU': { # Italy, Spain, Portugal
        'codes': ['ITA', 'ESP', 'PRT'],
        'feeds': [
             # Italy
            ('https://ilsole24ore.com/rss/finanza.xml', 'it'),
            ('https://ilsole24ore.com/rss/imprese.xml', 'it'),
            ('https://www.milanofinanza.it/rss/rss_ultimissime.xml', 'it'),
            ('https://www.repubblica.it/rss/economia/rss2.0.xml', 'it'), # La Repubblica Economia
            # Spain
            ('https://www.expansion.com/rss/portada.xml', 'es'),
            ('https://www.eleconomista.es/rss/rss-empresas-finanzas.php', 'es'),
            ('https://cincodias.elpais.com/rss/cincodias/portada.xml', 'es'), # Cinco Días
            # Portugal
            ('https://www.jornaldenegocios.pt/rss', 'pt'),
            ('https://eco.sapo.pt/feed/', 'pt'), # ECO News
            # Add more Southern EU sources
        ]
    },
    'ASIA_EAST': { # China, Hong Kong, Japan, South Korea, Taiwan
        'codes': ['CHN', 'HKG', 'JPN', 'KOR', 'TWN'],
        'feeds': [
            # China (English)
            ('http://www.chinadaily.com.cn/rss/business_rss.xml', 'en'),
            ('https://www.caixinglobal.com/rss/latest.xml', 'en'),
            # Hong Kong (English)
            ('https://www.scmp.com/rss/91/feed', 'en'),
            ('https://www.thestandard.com.hk/rss/latest_business.php', 'en'),
            # Japan (English)
            ('https://asia.nikkei.com/rss/feed/nar', 'en'),
            ('https://www.japantimes.co.jp/news_category/business/feed/', 'en'),
            ('http://feeds.reuters.com/reuters/JapanmarketsNews', 'en'),
             # South Korea (English)
            ('http://www.koreaherald.com/rss/rss_xml.php?ct=6', 'en'),
            ('http://koreatimes.co.kr/www/rss/biz.xml', 'en'),
            ('https://feed.koreatimes.co.kr/k/economy.xml', 'en'),
            ('https://en.yna.co.kr/RSS/economy.xml', 'en'),
             # Taiwan (English)
            ('https://www.taipeitimes.com/rss/business.xml', 'en'),
            ('https://focustaiwan.tw/rss/list/BUSINESS.aspx', 'en'),
            # Add more East Asian sources
        ]
    },
    'SGP': { # Singapore
        'codes': ['SGP'],
        'feeds': [
            ('https://www.straitstimes.com/news/business/rss.xml', 'en'),
            ('https://www.channelnewsasia.com/rssfeeds/8395954', 'en'),
            ('https://www.businesstimes.com.sg/rss/companies', 'en'),
            # Add more Singaporean sources
        ]
    },
    'ISR': {
        'codes': ['ISR'],
        'feeds': [
            ('https://www.jpost.com/Rss/RssFeedsBusinessAndInnovation.aspx', 'en'),
            ('https://www.timesofisrael.com/business/feed/', 'en'),
            ('https://en.globes.co.il/en/rss.aspx', 'en'),
            ('https://www.calcalistech.com/ctech/rss/articles/0,7340,L-3760431,00.xml', 'en'),
            # Add more Israeli sources
        ]
    },
}

# --- Helper Functions ---

def load_seen_urls(seen_urls_filepath):
    """Load the set of already scraped URLs for a specific group."""
    seen_urls = set()
    if os.path.exists(seen_urls_filepath):
        try:
            with open(seen_urls_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    seen_urls.add(line.strip())
        except Exception as e:
            logging.warning(f"Could not read {seen_urls_filepath}. Starting fresh. Error: {e}")
    return seen_urls

def save_url(url, seen_urls_filepath):
    """Appends a new URL to the group's 'seen' file."""
    try:
        with open(seen_urls_filepath, 'a', encoding='utf-8') as f:
            f.write(url + '\n')
    except Exception as e:
        logging.error(f"Could not write to seen URLs file: {seen_urls_filepath}. Error: {e}")


def save_text(text, output_filepath):
    """Appends the scraped article text to the group's corpus file."""
    try:
        with open(output_filepath, 'a', encoding='utf-8') as f:
            f.write(text + '\n' + '='*80 + '\n') # Add a separator
    except Exception as e:
        logging.error(f"Could not write to corpus file: {output_filepath}. Error: {e}")

def clean_url(url):
    """Removes query parameters and fragments from URL."""
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme if parsed.scheme else ''
        netloc = parsed.netloc if parsed.netloc else ''
        path = parsed.path if parsed.path else ''
        return scheme + "://" + netloc + path
    except Exception:
        return url # Return original url if cleaning fails


# --- Main Scraper Function ---

def scrape_all_groups():
    logging.info("--- Starting Master Autonomous Scraper (FASTER MODE) ---")
    logging.warning("Reduced delays enabled - higher risk of IP blocking.")
    logging.info("This script will cycle through all defined groups indefinitely.")
    logging.info("Press Ctrl+C to stop.\n")

    global_total_new_articles = 0
    MAX_FEED_RETRIES = 1 # Reduce retries for speed
    RETRY_DELAY = 2 # Shorter delay

    # --- SPEED ADJUSTMENT: Reduced article download delay ---
    # Original: random.uniform(2.0, 4.5)
    ARTICLE_DELAY_MIN = 0.5 # Minimum seconds between article downloads
    ARTICLE_DELAY_MAX = 1.5 # Maximum seconds between article downloads
    logging.warning(f"Article download delay set to random between {ARTICLE_DELAY_MIN} and {ARTICLE_DELAY_MAX} seconds.")
    # --- END SPEED ADJUSTMENT ---


    try:
        while True: # Run indefinitely until stopped
            logging.info("******************** Starting New Global Scraping Cycle ********************")
            cycle_new_articles = 0
            groups_to_scrape = list(REGIONAL_GROUPS.keys())
            random.shuffle(groups_to_scrape) # Process groups in random order each cycle

            for group_name in groups_to_scrape:
                group_data = REGIONAL_GROUPS[group_name]
                group_feeds = group_data['feeds']
                group_codes = group_data['codes']

                logging.info(f"--- Processing Group: {group_name} (Countries: {', '.join(group_codes)}) ---")

                group_output_dir = os.path.join(BASE_OUTPUT_DIR, group_name)
                os.makedirs(group_output_dir, exist_ok=True)
                output_filepath = os.path.join(group_output_dir, 'corpus.txt')
                seen_urls_filepath = os.path.join(group_output_dir, 'seen_urls.txt')

                seen_urls = load_seen_urls(seen_urls_filepath)
                group_new_articles = 0
                feeds_to_scrape = list(group_feeds)
                random.shuffle(feeds_to_scrape)

                for feed_url, lang in feeds_to_scrape:
                    # logging.debug(f"  Fetching feed for {group_name}: {feed_url}") # Use debug level for less noise
                    feed = None
                    for attempt in range(MAX_FEED_RETRIES + 1):
                        try:
                            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                            # CORRECTED: Removed the 'timeout' argument from feedparser.parse
                            feed = feedparser.parse(feed_url, agent=headers.get('User-Agent'), request_headers=headers)

                            if feed.bozo and isinstance(feed.bozo_exception, (socket.gaierror, URLError)):
                                raise feed.bozo_exception
                            break # Succeeded or failed for non-network reason
                        except (socket.gaierror, URLError) as net_err:
                            logging.warning(f"  Network Error on attempt {attempt + 1}/{MAX_FEED_RETRIES + 1} for {feed_url}: {net_err}")
                            if attempt < MAX_FEED_RETRIES:
                                time.sleep(RETRY_DELAY)
                            else:
                                logging.error(f"  Max retries reached for feed {feed_url}. Skipping.")
                                break # Exit retry loop after max retries
                        except Exception as e:
                            # Catch the TypeError specifically if it occurs elsewhere, though unlikely now
                            if isinstance(e, TypeError) and 'timeout' in str(e):
                                 logging.error(f"  Internal Error: Still trying to use timeout argument incorrectly for {feed_url}.")
                            else:
                                 logging.error(f"  Unexpected error fetching feed {feed_url} on attempt {attempt + 1}. Error: {e}")
                            break # Don't retry other unexpected errors

                    if feed is None: continue # Skip if fetching failed after retries

                    if feed.bozo:
                        # Log warning for malformed feed
                        warning_msg = f"  Warning: Feed may be malformed: {feed_url}. Parser message: {getattr(feed.bozo_exception, 'getMessage', lambda: str(feed.bozo_exception))()}"
                        logging.warning(warning_msg[:250])
                        continue # Skip processing this feed

                    # --- Process feed entries (rest of the loop is unchanged) ---
                    for entry in feed.entries:
                        url = getattr(entry, 'link', None)
                        if not url: continue
                        url = clean_url(url)
                        if url in seen_urls: continue

                        try:
                            article_config = newspaper.Config()
                            article_config.browser_user_agent = headers.get('User-Agent')
                            article_config.request_timeout = 15 # Shorter article timeout
                            article_config.fetch_images = False
                            article_config.memoize_articles = False
                            article_config.verbose = False # Reduce newspaper's verbosity

                            article = newspaper.Article(url, language=lang, config=article_config)
                            article.download()
                            article.parse()

                            if len(article.text) < 100: # Slightly lower threshold
                                save_url(url, seen_urls_filepath)
                                seen_urls.add(url)
                            else:
                                title = article.title if article.title else "No Title Found"
                                header = f"SOURCE URL: {url}\nTITLE: {title}\nGROUP: {group_name}\n\n"
                                save_text(header + article.text, output_filepath)
                                save_url(url, seen_urls_filepath)
                                seen_urls.add(url)
                                logging.info(f"    SAVED [{group_name}]: {title[:60]}... (Len: {len(article.text)})")
                                group_new_articles += 1
                                cycle_new_articles += 1
                                global_total_new_articles += 1

                            # --- SPEED ADJUSTMENT: Reduced article delay ---
                            time.sleep(random.uniform(ARTICLE_DELAY_MIN, ARTICLE_DELAY_MAX))
                            # --- END SPEED ADJUSTMENT ---

                        except newspaper.article.ArticleException as article_err:
                            if '406 Client Error' in str(article_err):
                                 logging.warning(f"    SKIP (406 Blocked) [{group_name}]: {url}")
                            else:
                                logging.warning(f"    WARN (Newspaper) [{group_name}]: {url}. {str(article_err)[:150]}")
                            save_url(url, seen_urls_filepath)
                            seen_urls.add(url)
                            # time.sleep(0.5) # Shorter pause on error
                        except Exception as e:
                            logging.error(f"    ERROR (General Article) [{group_name}]: {url}. {str(e)[:150]}")
                            save_url(url, seen_urls_filepath)
                            seen_urls.add(url)
                            # time.sleep(1) # Shorter pause on error


            logging.info("******************** End of Global Cycle. Total New Articles This Cycle: {} ********************".format(cycle_new_articles))
            logging.info(f"Total New Articles This Session: {global_total_new_articles}")
            logging.info("Starting next cycle immediately...")
            # --- SPEED ADJUSTMENT: Remove cycle delay ---
            # time.sleep(CYCLE_WAIT_MINUTES * 60)
            # --- END SPEED ADJUSTMENT ---


    except KeyboardInterrupt:
        logging.info("\n--- Master Scraper stopped by user ---")
        logging.info(f"Total articles scraped this session: {global_total_new_articles}")
        logging.info(f"Corpora saved in respective subfolders within {BASE_OUTPUT_DIR}")

if __name__ == "__main__":
    scrape_all_groups()

