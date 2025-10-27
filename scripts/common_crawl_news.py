import os
import logging
from datasets import load_dataset
from urllib.parse import urlparse
import time
import json # To save progress

# --- Configuration ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base output directory (MUST MATCH your RSS scraper output)
# !! IMPORTANT: Update this path !!
# Example: BASE_OUTPUT_DIR = r'C:\_Files\School\Competitions\FIAM2025\data\regional_tuning_data'
BASE_OUTPUT_DIR = r'D:\market_data\text_data' # <-- MAKE SURE THIS PATH IS CORRECT
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
logging.info(f"Base output directory: {BASE_OUTPUT_DIR}")

# --- Target Domains per Region ---
# !!! CRITICAL: EXPAND THESE LISTS SIGNIFICANTLY for better coverage !!!
# Add domains of major national newspapers, financial papers, business news sites for each country
TARGET_DOMAINS = {
    'USA': [
        'bloomberg.com',
        'reuters.com',
        'wsj.com',
        'ft.com',
        'forbes.com',
        'economist.com',
        'cnbc.com',
        'foxbusiness.com',
        'money.cnn.com',
        'marketwatch.com',
        'investing.com',
        'finance.yahoo.com',
        'money.msn.com',
        'seekingalpha.com',
        'fool.com',
        'kiplinger.com',
        'investopedia.com',
        'thestreet.com',
        'morganstanley.com',
        'fisherinvestments.com'
    ],
    'CAN': [
        'theglobeandmail.com',
        'nationalpost.com',
        'thestar.com',
        'ctvnews.ca',
        'apnews.com',
        'timesofindia.indiatimes.com',
        'rob.reportonbusiness.com',
        'investmentexecutive.com',
        'bcbusiness.ca',
        'macleans.ca',
        'corporateknights.com',
        'ised-isde.canada.ca',
        'publications.gc.ca'
    ],
    'MEX': [
        'elfinanciero.com.mx',
        'milenio.com',
        'jornada.com.mx',
        'bloomberglinea.com',
        'elpais.com',
        'forbes.com.mx',
        'scotiabank.com.mx',
        'banamex.com'
    ],
    'GBR': [
        'ft.com',
        'reuters.com',
        'economist.com',
        'theguardian.com',
        'independent.co.uk',
        'cityam.com',
        'thisismoney.co.uk',
        'moneyweek.com',
        'ifamagazine.com',
        'financial-world.com',
        'wealthandfinance.digital',
        'finance-monthly.com',
        'finextra.com',
        'thefintechtimes.com',
        'uktn.tech',
        'sifted.eu',
        'fintechmagazine.com',
        'fintechfutures.com',
        'tomd.co.uk',
        '11fs.com'
    ],
    'IRL': [
        'irishtimes.com',
        'thinkbusiness.ie',
        'businessnews.ie',
        'irishcentral.com',
        'newsnow.co.uk',
        'washingtonpost.com'
    ],
    'ANZ': [
        'nzherald.co.nz',
        'businessnewsaustralia.com',
        'rnz.co.nz',
        'fool.com.au',
        'investordaily.com.au',
        'investmentmagazine.com.au',
        'stockspot.com.au',
        'passiveinvestingaustralia.com',
        'intelligentinvestor.com.au',
        'beehive.govt.nz',
        'economictimes.indiatimes.com'
    ],
    'DACH': [
        'finanzen.net',
        'finanznachrichten.de',
        'handelsblatt.com',
        'boersen-zeitung.de',
        'dw.com',
        'finanz.at',
        'finanznachrichten.at',
        'ig.com',
        'handelszeitung.ch',
        'cash.ch',
        'wirtschaftszeit.ch',
        'snb.ch',
        'swissinfo.ch',
        'finews.com',
        'fintechnews.ch'
    ],
    'FRA': [
        'lesechos.fr',
        'latribune.fr',
        'lefigaro.fr',
        'challenges.fr',
        'investir.fr',
        'lerevenu.com',
        'lagefi.fr',
        'optionfinance.fr',
        'capital.fr',
        'management.fr',
        'mieuxvivrevotreargent.fr',
        'alternatives-economiques.fr',
        'journaldeleconomie.fr',
        'zonebourse.com',
        'presse.economie.gouv.fr'
    ],
    'BENELUX': [
        'ing.be',
        'economie.fgov.be',
        'brusselstimes.com',
        'brusselsmorning.com',
        'wbn.nl',
        'rijksoverheid.nl',
        'rvo.nl',
        'government.nl',
        'deloitte.com',
        'luxse.com',
        'bloomberg.com',
        'morningstar.com',
        'businesstimes.com.sg',
        'aa.com.tr'
    ],
    'NORDICS': [
        'erhvervsnyhederne.dk',
        'erhvervs-tidende.dk',
        'erhvervsstyrelsen.dk',
        'thelocal.dk',
        'eng.em.dk',
        'finimize.com',
        'etla.fi',
        'vm.fi',
        'suomenpankki.fi',
        'bofbulletin.fi',
        'helsinkitimes.fi',
        'finland.fi',
        'finansavisen.no',
        'nrk.no',
        'vg.no',
        'aftenposten.no',
        'regjeringen.no',
        'newsinenglish.no',
        'investing.com',
        'ft.com',
        'tradingview.com',
        'ofm.wa.gov',
        'tv2.no',
        'abcnyheter.no',
        'podcasts.apple.com'
    ],
    'SOUTHERN_EU': [
        'quifinanza.it',
        'teleborsa.it',
        'finanza.repubblica.it',
        'elpais.com',
        'correionegocios.pt',
        'cnnportugal.iol.pt',
        'revistabusinessportugal.pt'
    ],
    'ASIA_EAST': [
        'apnews.com',
        'timesofindia.indiatimes.com',
        'english.kyodonews.net',
        'koreaherald.com',
        'koreatimes.co.kr',
        'mk.co.kr',
        'kedglobal.com',
        'focustaiwan.tw',
        'scmp.com',
        'taiwantoday.tw',
        'cgtn.com',
        'news.cn'
    ],
    'SGP': [
        'theedgesingapore.com',
        'businesstimes.com.sg',
        'timesofindia.indiatimes.com',
        'worldbank.org'
    ],
    'ISR': [
        'en.globes.co.il',
        'jpost.com'
    ]
}

# --- Optional: Keywords for additional filtering (can slow things down) ---
KEYWORDS = None # Set to None to disable keyword filtering for speed

# --- Progress Tracking ---
# Use a different progress file name for this dataset
PROGRESS_FILE = os.path.join(BASE_OUTPUT_DIR, 'multilingual_cc_news_progress.json')
SAVE_PROGRESS_INTERVAL = 10000 # Save progress every N articles processed (increased from debug)
LOG_PROGRESS_INTERVAL = 1000 # Log progress less frequently than debug

# --- Helper Functions ---

def get_domain(url):
    """Extracts the main domain name (e.g., bbc.co.uk) from a URL."""
    try:
        netloc = urlparse(url).netloc
        if not netloc: return None
        parts = netloc.split('.')
        if len(parts) >= 2:
             if parts[-2] in ['co', 'com', 'org', 'gov', 'net'] and len(parts) > 2:
                 return '.'.join(parts[-3:]).lower()
             else:
                 return '.'.join(parts[-2:]).lower()
        else: return netloc.lower()
    except Exception: return None

def save_text_cc(text, output_filepath):
    """Appends the scraped article text to the group's corpus file."""
    try:
        with open(output_filepath, 'a', encoding='utf-8') as f:
            f.write(text.strip() + '\n\n')
    except Exception as e:
        logging.error(f"Could not write to corpus file: {output_filepath}. Error: {e}")

def load_progress():
    """Loads the number of articles already processed."""
    # --- Reinstated original logic ---
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                progress = json.load(f)
                count = progress.get('processed_count', 0)
                logging.info(f"Loaded progress: {count:,} articles previously processed.")
                return count
        except Exception as e:
            logging.warning(f"Could not load progress file ({PROGRESS_FILE}): {e}. Starting from scratch.")
    else:
        logging.info("Progress file ({}) not found. Starting from scratch.".format(PROGRESS_FILE))
    return 0

def save_progress(count):
    """Saves the number of articles processed."""
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({'processed_count': count}, f)
        # logging.debug(f"Saved progress: {count:,}") # Use debug level for less noise
    except Exception as e:
        logging.error(f"Could not save progress ({PROGRESS_FILE}): {e}")

# --- Main Processing Function ---

def process_cc_news():
    logging.info("--- Starting Multilingual Common Crawl News (intfloat/multilingual_cc_news - English Subset) Processor ---") # Updated title
    logging.warning("This will stream a large dataset and can take MANY hours/days.")
    logging.warning("Ensure you have a stable internet connection and sufficient disk space.")

    processed_count_start_of_run = load_progress()
    logging.info(f"Attempting to resume from article number: {processed_count_start_of_run:,}")

    total_added_this_session = 0
    start_time = time.time()
    cc_stream = None # Initialize variable
    dataset_iterable = None # Initialize variable

    try:
        logging.info("Initializing intfloat/multilingual_cc_news stream (English subset)...")
        # --- MODIFIED: Point to the new dataset AND specify the English config ---
        dataset_iterable = load_dataset("intfloat/multilingual_cc_news", languages=["en"], streaming=True, split="train") # Added name='en'
        cc_stream = iter(dataset_iterable) # Directly try to get an iterator
        logging.info("Stream initialized as iterator.")

    except Exception as load_err:
        logging.error(f"Failed to initialize dataset stream correctly: {load_err}", exc_info=True)
        return # Exit if stream loading fails


    current_article_index = 0 # Initialize here, will be updated by skipping logic
    try: # Add a try block around the skipping and processing loops

        # --- Reinstated skipping logic ---
        if processed_count_start_of_run > 0:
            logging.info(f"Attempting to skip the first {processed_count_start_of_run:,} articles...")
            skipped_count = 0
            skip_start_time = time.time()
            try:
                for _ in range(processed_count_start_of_run):
                    next(cc_stream) # Advance the iterator
                    skipped_count += 1
                    if skipped_count % 100000 == 0: # Log progress during skip
                         logging.info(f"Skipped {skipped_count:,} / {processed_count_start_of_run:,} articles...")
                current_article_index = skipped_count # Start counting from here
                logging.info(f"Skipping complete. Took {time.time() - skip_start_time:.2f} seconds.")
            except StopIteration:
                 logging.warning("Stream ended while trying to skip. Maybe the dataset size changed or progress file is wrong?")
                 current_article_index = skipped_count # Adjust count if stream ended early
                 save_progress(current_article_index)
                 return # Cannot continue if stream ended during skip
            except Exception as skip_err:
                 logging.error(f"Error during skipping at article {skipped_count}: {skip_err}", exc_info=True)
                 current_article_index = skipped_count # Save progress up to the error point
                 save_progress(current_article_index)
                 return # Cannot reliably continue
        else:
             current_article_index = 0 # Ensure it starts at 0 if no progress loaded
        # --- End skipping logic ---


        # --- Main Processing Loop ---
        logging.info(f"Starting article processing loop from index {current_article_index:,}...")
        loop_start_time = time.time()
        articles_processed_in_loop = 0 # Count articles processed *within this loop*

        while True: # Keep processing until manually stopped or stream ends
            try:
                article = next(cc_stream)
                current_article_index += 1 # Increment total count
                articles_processed_in_loop += 1 # Increment loop count

                # --- Logging and Saving Progress ---
                if articles_processed_in_loop % LOG_PROGRESS_INTERVAL == 0:
                    now = time.time()
                    loop_elapsed = now - loop_start_time
                    articles_per_sec_loop = articles_processed_in_loop / loop_elapsed if loop_elapsed > 0 else 0
                    logging.info(f"Processing index {current_article_index:,}... (Loop rate: {articles_per_sec_loop:.1f}/s)")

                if current_article_index % SAVE_PROGRESS_INTERVAL == 0:
                     save_progress(current_article_index)

                # --- Article Filtering and Saving Logic ---
                # Check dataset structure - column names might be different
                url = article.get('url')
                text = article.get('text')
                # Try 'domain' first, fallback to parsing url
                domain_raw = article.get('domain')
                domain = domain_raw.lower() if domain_raw else get_domain(url)

                if not url or not text or len(text) < 150: continue
                if not domain: continue

                matched_group = None
                for group_name, domains_list in TARGET_DOMAINS.items():
                    # Check domain match more robustly
                    if any(domain == target_domain or domain.endswith('.' + target_domain) for target_domain in domains_list):
                         matched_group = group_name
                         break

                if matched_group:
                    if KEYWORDS:
                        text_lower = text.lower()
                        if not any(text_lower.find(keyword) != -1 for keyword in KEYWORDS):
                            continue

                    group_output_dir = os.path.join(BASE_OUTPUT_DIR, matched_group)
                    os.makedirs(group_output_dir, exist_ok=True)
                    output_filepath = os.path.join(group_output_dir, 'corpus.txt')

                    save_text_cc(text, output_filepath)
                    total_added_this_session += 1

            except StopIteration:
                logging.info(f"Stream finished cleanly after processing article {current_article_index:,}.")
                save_progress(current_article_index)
                break
            except Exception as e:
                logging.error(f"Error processing article #{current_article_index} (URL: {url if 'url' in locals() and url else 'N/A'}): {e}", exc_info=True)
                save_progress(current_article_index)
                logging.warning("Attempting to continue to next article after error...")
                time.sleep(1)

    except KeyboardInterrupt:
        logging.info("\n--- Multilingual CC News Processor stopped by user ---")
        final_idx_on_stop = current_article_index if 'current_article_index' in locals() else processed_count_start_of_run
        save_progress(final_idx_on_stop)
    except Exception as e:
        logging.error(f"\n--- An unexpected error occurred outside the main processing loop: {e} ---", exc_info=True)
        logging.error("Saving last known progress before exiting.")
        final_idx_on_error = current_article_index if 'current_article_index' in locals() else processed_count_start_of_run
        save_progress(final_idx_on_error)
    finally:
        session_elapsed = time.time() - start_time
        final_processed_index = load_progress() # Read the absolute latest count saved to disk
        processed_this_session = final_processed_index - processed_count_start_of_run

        logging.info(f"Finished session. Processed approximately {processed_this_session:,} articles in {session_elapsed:.2f} seconds.")
        logging.info(f"Added {total_added_this_session:,} articles to corpora during this session.")
        logging.info(f"Total articles processed overall (check progress file): {final_processed_index:,}")

if __name__ == "__main__":
    process_cc_news()

