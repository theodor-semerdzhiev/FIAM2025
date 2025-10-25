import feedparser
import newspaper
import time
import os
import random
from urllib.parse import urlparse

# --- Configuration ---

# Base output directory
BASE_OUTPUT_DIR = r'D:\market_data\text_data'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
print(f"Base output directory: {BASE_OUTPUT_DIR}")

# Define groups, their countries (codes from Appendix A), feeds, and primary language
# We aim for mostly English sources where possible, except where native language is needed for translation step
# Structure: 'GROUP_NAME': { 'codes': ['COUNTRY_CODE', ...], 'feeds': [('rss_url', 'language_code'), ...]}
# Note: language_code is for newspaper3k's extractor hint
REGIONAL_GROUPS = {
    'USA': { # Special case - you'll likely use 10K/Q data primarily, but scraping news is still useful
        'codes': ['USA'], # USA isn't in Appendix A list, but needed
        'feeds': [
            ('http://feeds.reuters.com/reuters/businessNews', 'en'), # Reuters US Business
            ('https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml', 'en'), # Wall Street Journal US Business
            ('https://www.cnbc.com/id/10000115/device/rss/rss.html', 'en'), # CNBC Top News
            ('https://www.nytimes.com/svc/collections/v1/publish/https://www.nytimes.com/section/business/rss.xml', 'en'),
        ]
    },
    'CAN': {
        'codes': ['CAN'],
        'feeds': [
            ('https://www.bnnbloomberg.ca/rss/news', 'en'),
            ('https://financialpost.com/feed/', 'en'),
            ('https://www.theglobeandmail.com/business/rss/rob-commentary/', 'en'),
            ('https://www.theglobeandmail.com/business/rss/industry-news/', 'en'),
        ]
    },
    'MEX': { # Will require translation later
        'codes': ['MEX'],
        'feeds': [
            ('https://www.elfinanciero.com.mx/rss/rss-ultimas-noticias.xml', 'es'),
            ('https://expansion.mx/rss/economia', 'es'),
            ('https://www.eleconomista.com.mx/rss/empresas.xml', 'es'), # Empresas section
            ('https://www.reforma.com/rss/negocios.xml', 'es'),
        ]
    },
    'GBR': {
        'codes': ['GBR'],
        'feeds': [
            ('http://feeds.reuters.com/reuters/UKBusinessNews', 'en'),
            ('https://feeds.bbci.co.uk/news/business/rss.xml', 'en'),
            ('https://www.theguardian.com/uk/business/rss', 'en'),
            ('https://www.telegraph.co.uk/business/rss.xml', 'en'),
            # ('https://www.ft.com/rss/home/uk', 'en'), # Often paywalled
        ]
    },
    'IRL': {
        'codes': ['IRL'],
        'feeds': [
            ('https://www.irishtimes.com/business/rss', 'en'),
            ('https://www.independent.ie/business/rss/', 'en'),
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
            # New Zealand
            ('https://www.nzherald.co.nz/rss/topic/business/', 'en'),
            ('https://www.stuff.co.nz/rss/business', 'en'),
            ('https://businessdesk.co.nz/rss', 'en'),
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
            # Austria
            ('https://www.diepresse.com/rss/Wirtschaft', 'de'),
            ('https://www.derstandard.at/rss/?ressort=Wirtschaft', 'de'),
            # Switzerland (German part)
            ('https://www.nzz.ch/wirtschaft.rss', 'de'),
            # Switzerland (French part - might add to FRA group instead if preferred)
            ('https://www.letemps.ch/economie/rss', 'fr'),
        ]
    },
    'FRA': { # France only for now (CHE French feed is in DACH)
        'codes': ['FRA'],
        'feeds': [
            ('https://www.lesechos.fr/rss/rss_finance-marches.xml', 'fr'),
            ('https://www.lesechos.fr/rss/rss_entreprises.xml', 'fr'),
            ('https://www.latribune.fr/finance/feed.xml', 'fr'),
            ('https://www.lefigaro.fr/rss/figaro_economie.xml', 'fr'),
        ]
    },
    'BENELUX': { # Belgium, Netherlands, Luxembourg
        'codes': ['BEL', 'NLD', 'LUX'],
        'feeds': [
            # Belgium
            ('https://www.lecho.be/rss.xml', 'fr'), # French part
            ('https://www.tijd.be/rss.xml', 'nl'), # Dutch part
            # Netherlands
            ('https://fd.nl/rss', 'nl'),
            ('https://www.rtlnieuws.nl/rss/rtlz.xml', 'nl'),
            # Luxembourg
            ('https://www.luxtimes.lu/en/rss/business', 'en'), # English version
            ('https://paperjam.lu/rss.xml', 'fr'), # French version
            ('https://www.wort.lu/de/business/rss', 'de'), # German version
        ]
    },
    'NORDICS': { # Denmark, Finland, Norway, Sweden
        'codes': ['DNK', 'FIN', 'NOR', 'SWE'],
        'feeds': [
            ('https://borsen.dk/rss/', 'da'), # Denmark
            ('https://www.kauppalehti.fi/rss', 'fi'), # Finland
            ('https://www.dn.no/rss/', 'no'), # Norway
            ('https://www.di.se/rss', 'sv'), # Sweden
        ]
    },
    'SOUTHERN_EU': { # Italy, Spain, Portugal
        'codes': ['ITA', 'ESP', 'PRT'],
        'feeds': [
             # Italy
            ('https://ilsole24ore.com/rss/finanza.xml', 'it'),
            ('https://ilsole24ore.com/rss/imprese.xml', 'it'),
            ('https://www.milanofinanza.it/rss/rss_ultimissime.xml', 'it'),
            # Spain
            ('https://www.expansion.com/rss/portada.xml', 'es'),
            ('https://www.eleconomista.es/rss/rss-empresas-finanzas.php', 'es'), # Empresas y Finanzas
            # Portugal
            ('https://www.jornaldenegocios.pt/rss', 'pt'),
        ]
    },
    'ASIA_EAST': { # China, Hong Kong, Japan, South Korea, Taiwan
        'codes': ['CHN', 'HKG', 'JPN', 'KOR', 'TWN'],
        'feeds': [
            # China (English)
            ('http://www.chinadaily.com.cn/rss/business_rss.xml', 'en'),
            ('https://www.caixinglobal.com/rss/latest.xml', 'en'),
            # Hong Kong (English)
            ('https://www.scmp.com/rss/91/feed', 'en'), # SCMP Business
            ('https://www.thestandard.com.hk/rss/latest_business.php', 'en'),
            # Japan (English)
            ('https://asia.nikkei.com/rss/feed/nar', 'en'),
            ('https://www.japantimes.co.jp/news_category/business/feed/', 'en'),
            ('http://feeds.reuters.com/reuters/JapanmarketsNews', 'en'),
             # South Korea (English)
            ('http://www.koreaherald.com/rss/rss_xml.php?ct=6', 'en'),
            ('http://koreatimes.co.kr/www/rss/biz.xml', 'en'),
            ('https://en.yna.co.kr/RSS/economy.xml', 'en'),
             # Taiwan (English)
            ('https://www.taipeitimes.com/rss/business.xml', 'en'),
            ('https://focustaiwan.tw/rss/list/BUSINESS.aspx', 'en'), # Focus Taiwan Business
        ]
    },
    'SGP': { # Singapore
        'codes': ['SGP'],
        'feeds': [
            ('https://www.straitstimes.com/news/business/rss.xml', 'en'),
            ('https://www.channelnewsasia.com/rssfeeds/8395954', 'en'), # CNA Business
            ('https://www.businesstimes.com.sg/rss/companies', 'en'),
        ]
    },
    'ISR': {
        'codes': ['ISR'], # Note: Code in Appendix is ISL, but ISO 3166 is ISR. Using ISR.
        'feeds': [
            ('https://www.jpost.com/Rss/RssFeedsBusinessAndInnovation.aspx', 'en'),
            ('https://www.timesofisrael.com/business/feed/', 'en'),
            ('https://en.globes.co.il/en/rss.aspx', 'en'),
            ('https://www.calcalistech.com/ctech/rss/articles/0,7340,L-3760431,00.xml', 'en'),
        ]
    },
    # Note: CHE French feed is included in DACH, adjust if needed
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
            print(f"Warning: Could not read {seen_urls_filepath}. Starting fresh. Error: {e}")
    return seen_urls

def save_url(url, seen_urls_filepath):
    """Appends a new URL to the group's 'seen' file."""
    with open(seen_urls_filepath, 'a', encoding='utf-8') as f:
        f.write(url + '\n')

def save_text(text, output_filepath):
    """Appends the scraped article text to the group's corpus file."""
    with open(output_filepath, 'a', encoding='utf-8') as f:
        f.write(text + '\n' + '='*80 + '\n') # Add a separator

def clean_url(url):
    """Removes query parameters and fragments from URL."""
    parsed = urlparse(url)
    return parsed.scheme + "://" + parsed.netloc + parsed.path


# --- Main Scraper Function ---

def scrape_all_groups():
    print(f"--- Starting Master Autonomous Scraper ---")
    print("This script will cycle through all defined groups indefinitely.")
    print("Press Ctrl+C to stop.\n")

    global_total_new_articles = 0

    try:
        while True: # Run indefinitely until stopped
            print("\n" + "*"*20 + " Starting New Global Scraping Cycle " + "*"*20)
            cycle_new_articles = 0
            groups_to_scrape = list(REGIONAL_GROUPS.keys())
            random.shuffle(groups_to_scrape) # Process groups in random order each cycle

            for group_name in groups_to_scrape:
                group_data = REGIONAL_GROUPS[group_name]
                group_feeds = group_data['feeds']
                group_codes = group_data['codes']

                print(f"\n--- Processing Group: {group_name} (Countries: {', '.join(group_codes)}) ---")

                # Define paths for this group
                group_output_dir = os.path.join(BASE_OUTPUT_DIR, group_name)
                os.makedirs(group_output_dir, exist_ok=True)
                output_filepath = os.path.join(group_output_dir, 'corpus.txt')
                seen_urls_filepath = os.path.join(group_output_dir, 'seen_urls.txt')

                # Load seen URLs for this specific group
                seen_urls = load_seen_urls(seen_urls_filepath)
                print(f"Loaded {len(seen_urls)} seen URLs for {group_name}.")

                group_new_articles = 0
                feeds_to_scrape = list(group_feeds)
                random.shuffle(feeds_to_scrape)

                for feed_url, lang in feeds_to_scrape:
                    print(f"  Fetching feed for {group_name}: {feed_url}")
                    try:
                        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
                        # Set timeout for feed parsing
                        feed = feedparser.parse(feed_url, agent=headers.get('User-Agent'), request_headers=headers, response_headers={}, etag=None, modified=None)


                        if feed.bozo:
                            print(f"  Warning: Feed may be malformed or inaccessible. {feed.bozo_exception}")
                            time.sleep(1) # Small pause if feed is broken
                            continue

                        # print(f"  Found {len(feed.entries)} articles in feed.") # Reduce verbosity

                        for entry in feed.entries:
                            url = getattr(entry, 'link', None)
                            if not url:
                                continue # Skip entry if no link

                            # Clean URL
                            url = clean_url(url)

                            if url in seen_urls:
                                continue

                            # print(f"    New article found: {url}") # Reduce verbosity

                            try:
                                article_config = newspaper.Config()
                                article_config.browser_user_agent = headers.get('User-Agent')
                                article_config.request_timeout = 20 # Increased timeout slightly
                                article_config.fetch_images = False # Don't need images
                                article_config.memoize_articles = False # Don't cache in memory

                                article = newspaper.Article(url, language=lang, config=article_config)
                                article.download()
                                article.parse()

                                if len(article.text) < 150: # Slightly lower threshold
                                    # print(f"    Skipped (too short/paywall/parse error): {url}")
                                    save_url(url, seen_urls_filepath) # Save even if skipped
                                    seen_urls.add(url)
                                else:
                                    title = article.title if article.title else "No Title Found"
                                    # Include group name for clarity
                                    header = f"SOURCE URL: {url}\nTITLE: {title}\nGROUP: {group_name}\n\n"
                                    save_text(header + article.text, output_filepath)
                                    save_url(url, seen_urls_filepath)
                                    seen_urls.add(url)
                                    print(f"    SAVED [{group_name}]: {title[:60]}... (Len: {len(article.text)})") # Print confirmation
                                    group_new_articles += 1
                                    cycle_new_articles += 1
                                    global_total_new_articles += 1

                                # Politeness delay
                                time.sleep(random.uniform(1.5, 4.0))

                            except newspaper.article.ArticleException as article_err:
                                print(f"    ERROR (Newspaper) [{group_name}]: {url}. {article_err}")
                                save_url(url, seen_urls_filepath) # Don't retry newspaper errors often
                                seen_urls.add(url)
                                time.sleep(2) # Short pause on error
                            except Exception as e:
                                print(f"    ERROR (General) [{group_name}]: {url}. {e}")
                                time.sleep(5) # Longer pause on general error

                    except Exception as e:
                        print(f"  ERROR: Failed to fetch RSS feed {feed_url} for {group_name}. Error: {e}")
                        time.sleep(5)

                print(f"--- Finished Group: {group_name}. Added {group_new_articles} new articles this cycle. ---")

            print("\n" + "*"*20 + f" End of Global Cycle. Total New Articles This Cycle: {cycle_new_articles} " + "*"*20)
            print(f"Total New Articles This Session: {global_total_new_articles}")
            # Removed the long wait between cycles as requested
            print("Starting next cycle immediately...")
            # time.sleep(10) # Optional: Add a very short pause between cycles if needed

    except KeyboardInterrupt:
        print("\n--- Master Scraper stopped by user ---")
        print(f"Total articles scraped this session: {global_total_new_articles}")
        print(f"Corpora saved in respective subfolders within {BASE_OUTPUT_DIR}")

if __name__ == "__main__":
    scrape_all_groups()
