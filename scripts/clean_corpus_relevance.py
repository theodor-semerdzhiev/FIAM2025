import os
import logging
import time
# Removed langdetect
import random # For sampling lines
import re # For cleaning and simple tokenization
from collections import Counter # For counting keywords

# --- Configuration ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- !! IMPORTANT: SET THESE PATHS !! ---
BASE_OUTPUT_DIR = r'D:\market_data\text_data' # <-- MAKE SURE THIS PATH IS CORRECT
USA_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'USA')
# Input is the potentially large, english corpus
INPUT_CORPUS_FILE = os.path.join(USA_OUTPUT_DIR, 'corpus_en_only.txt')
# Output is the cleaned version
OUTPUT_CORPUS_FILE = os.path.join(USA_OUTPUT_DIR, 'corpus_cleaned.txt')

MIN_ARTICLE_LENGTH = 100 # Minimum characters AFTER cleaning
MAX_ARTICLE_LENGTH = 100000 # Discard articles longer than 100k characters

LOG_INTERVAL = 10000 # Log progress every N articles processed

# --- Text Cleaning Configuration ---
REMOVE_URLS = True
REMOVE_PHONE_NUMBERS = True
REMOVE_DISCLAIMERS = True

# Regex for URLs (handles http, https, www)
URL_REGEX = re.compile(r'https?://\S+|www\.\S+')
# Regex for common NA phone numbers (add formats if needed)
PHONE_REGEX = re.compile(r'\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
# List of common disclaimer patterns (case-insensitive)
# Add more specific phrases or patterns as observed
DISCLAIMER_PATTERNS = [
    re.compile(r"the views and opinions expressed herein.*?Nasdaq, Inc\.", re.IGNORECASE | re.DOTALL),
    re.compile(r"opinions expressed are.*?not necessarily those of.*?", re.IGNORECASE | re.DOTALL),
    re.compile(r"this article represents the opinion of the author.*?", re.IGNORECASE | re.DOTALL),
    re.compile(r"disclosure:.*?author has no position.*?", re.IGNORECASE | re.DOTALL),
    re.compile(r"seeking alpha.*?not a licensed financial advisor.*?", re.IGNORECASE | re.DOTALL),
    # Add generic "forward looking statement" warnings if desired
    re.compile(r"\bforward-looking statements?\b.*?risks uncertainties.*?differ materially.*?", re.IGNORECASE | re.DOTALL),
]


# --- Relevance Filtering Configuration ---
FILTER_BY_KEYWORDS = True
FINANCIAL_KEYWORDS_SET = set([
    "stock", "market", "shares", "equity", "equities", "trade", "trading",
    "investment", "investor", "portfolio", "asset", "assets", "fund", "funds",
    "finance", "financial", "economy", "economic", "gdp", "inflation", "rate",
    "rates", "yield", "bond", "bonds", "currency", "forex", "dollar", "euro",
    "bank", "banking", "fed", "ecb", "central bank", "loan", "credit", "debt",
    "company", "corporate", "corporation", "business", "firm", "industry",
    "earnings", "profit", "loss", "revenue", "income", "growth", "quarter",
    "report", "guidance", "ceo", "cfo", "board", "shareholder", "stakeholder",
    "merger", "acquisition", "ipo", "capital", "venture", "startup",
    "oil", "gas", "energy", "tech", "semiconductor", "pharma", "biotech",
    "retail", "consumer", "real estate",
    "risk", "volatile", "volatility", "downturn", "recession", "crisis", "fraud",
    "bankrupt", "bankruptcy", "default", "bullish", "optimistic", "strong",
    "growth", "rally", "boom", "recover",
    "inc", "ltd", "corp", "llc", "nasdaq", "nyse",
])
# --- REDUCED: Minimum Keyword Density Threshold ---
MIN_KEYWORD_DENSITY = 3.0 # Require only 3 financial keywords per 1000 words


# --- Relevance Analysis Configuration (at the end) ---
RUN_RELEVANCE_ANALYSIS = True
NUM_LINES_TO_SAMPLE = 10000


# --- Helper Function for Keyword Density ---
def calculate_keyword_density(text, keywords_set):
    """Calculates the density of specified keywords in the text."""
    if not text or not keywords_set: return 0.0
    words = re.findall(r'\b\w+\b', text.lower())
    total_word_count = len(words)
    if total_word_count == 0: return 0.0
    financial_word_count = sum(1 for word in words if word in keywords_set)
    density = (financial_word_count / total_word_count) * 1000
    return density

# --- NEW Text Cleaning Function ---
def clean_text(text):
    """Applies configured cleaning rules to the text."""
    if REMOVE_URLS:
        text = URL_REGEX.sub('', text)
    if REMOVE_PHONE_NUMBERS:
        text = PHONE_REGEX.sub('', text)
    if REMOVE_DISCLAIMERS:
        for pattern in DISCLAIMER_PATTERNS:
            text = pattern.sub('', text)

    # Remove excessive whitespace that might result from deletions
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

# --- Main Filtering Script ---

def clean_and_filter_corpus(input_path, output_path):
    logging.info(f"Starting corpus cleaning and filtering process.")
    logging.info(f"Input file: {input_path}")
    logging.info(f"Output file: {output_path}")
    logging.info(f"Cleaning applied: URLs={REMOVE_URLS}, Phones={REMOVE_PHONE_NUMBERS}, Disclaimers={REMOVE_DISCLAIMERS}")
    if FILTER_BY_KEYWORDS:
        logging.info(f"Applying keyword density filter (Min Density: {MIN_KEYWORD_DENSITY:.1f}/1000 words)")
    logging.info(f"Applying length filter (Min: {MIN_ARTICLE_LENGTH}, Max: {MAX_ARTICLE_LENGTH})")

    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return False

    start_time = time.time()
    articles_processed = 0
    articles_written = 0
    current_article_lines = []
    output_file_created = False

    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            output_file_created = True
            logging.info("Starting to process input file...")
            for line_num, line in enumerate(infile):
                stripped_line = line.strip()

                is_separator = not stripped_line
                if is_separator and current_article_lines:
                    # Process the accumulated article buffer
                    raw_article_text = "".join(current_article_lines).strip()
                    articles_processed += 1
                    current_article_lines = [] # Reset buffer

                    # --- Apply Cleaning ---
                    cleaned_article_text = clean_text(raw_article_text)

                    # --- Apply Filters on CLEANED text ---
                    # 1. Length Filter
                    if not (MIN_ARTICLE_LENGTH <= len(cleaned_article_text) <= MAX_ARTICLE_LENGTH):
                        # logging.debug(f"Skipped article {articles_processed} (cleaned length: {len(cleaned_article_text)})")
                        continue

                    # 2. Keyword Density Filter (if enabled)
                    if FILTER_BY_KEYWORDS:
                        density = calculate_keyword_density(cleaned_article_text, FINANCIAL_KEYWORDS_SET)
                        if density < MIN_KEYWORD_DENSITY:
                            # logging.debug(f"Skipped article {articles_processed} (density: {density:.2f})")
                            continue

                    # --- Write CLEANED text if all filters passed ---
                    outfile.write(cleaned_article_text + '\n\n')
                    articles_written += 1

                    # Log progress
                    if articles_processed % LOG_INTERVAL == 0 and articles_processed > 0:
                        elapsed = time.time() - start_time
                        rate = articles_processed / elapsed if elapsed > 0 else 0
                        logging.info(f"Processed: {articles_processed:,} articles | Written (Cleaned): {articles_written:,} | Rate: {rate:.1f} art/s")

                elif not is_separator:
                    current_article_lines.append(line)

            # --- Process the very last article in the file ---
            if current_article_lines:
                raw_article_text = "".join(current_article_lines).strip()
                articles_processed += 1
                cleaned_article_text = clean_text(raw_article_text)

                if MIN_ARTICLE_LENGTH <= len(cleaned_article_text) <= MAX_ARTICLE_LENGTH:
                     passes_keyword_check = True
                     if FILTER_BY_KEYWORDS:
                          density = calculate_keyword_density(cleaned_article_text, FINANCIAL_KEYWORDS_SET)
                          if density < MIN_KEYWORD_DENSITY:
                               passes_keyword_check = False

                     if passes_keyword_check:
                          outfile.write(cleaned_article_text + '\n\n')
                          articles_written += 1

    except FileNotFoundError:
        logging.error(f"Input file disappeared during processing: {input_path}")
        return False
    except Exception as e:
        logging.error(f"An error occurred during file processing: {e}", exc_info=True)
        return False
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        logging.info("--- Corpus Cleaning & Filtering Complete ---")
        logging.info(f"Total raw articles processed: {articles_processed:,}")
        logging.info(f"Total cleaned articles written: {articles_written:,}")
        logging.info(f"Output file: {output_path}")
        logging.info(f"Total time: {total_time:.2f} seconds")

    return output_file_created and articles_processed > 0


# --- Relevance Analysis Function (Operates on the CLEANED output file) ---
def analyze_relevance(filepath, num_samples, keywords_set):
    # --- (This function remains exactly the same as the previous version) ---
    logging.info(f"\n--- Starting Relevance Analysis on: {filepath} ---")
    logging.info(f"Sampling {num_samples:,} lines/articles to check keyword density...")

    if not os.path.exists(filepath):
        logging.error(f"File not found for analysis: {filepath}")
        return

    try:
        # --- Efficiently sample lines from a large file ---
        logging.info("Counting total lines/articles (this may take a while)...")
        total_articles_approx = 0
        with open(filepath, 'rb') as f: # Read bytes for faster counting maybe
            chunk_size = 1024 * 1024 * 10 # 10MB chunks
            separator = b'\n\n'
            buffer = b''
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    # Check remaining buffer
                    if buffer.strip():
                        total_articles_approx += 1
                    break
                buffer += chunk
                while separator in buffer:
                    # Split only once to avoid issues if separator is at chunk boundary
                    _, buffer = buffer.split(separator, 1)
                    total_articles_approx += 1
        logging.info(f"Approximate total articles: {total_articles_approx:,}")

        if total_articles_approx == 0:
            logging.warning("Output file appears empty. Cannot analyze relevance.")
            return

        num_to_sample = min(num_samples, total_articles_approx)
        if num_to_sample < num_samples:
             logging.warning(f"Requested {num_samples} samples, but file only has ~{total_articles_approx} articles. Sampling all.")

        # --- Sampling based on article boundaries ('\n\n') ---
        sampled_articles_text = []
        articles_read = 0
        current_article_lines = []
        # Use reservoir sampling for potentially better random distribution on large files
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                stripped_line = line.strip()
                is_separator = not stripped_line
                if is_separator and current_article_lines:
                    articles_read += 1
                    article_to_sample = "".join(current_article_lines).strip()
                    if len(sampled_articles_text) < num_to_sample:
                        sampled_articles_text.append(article_to_sample)
                    else:
                        s = int(random.random() * articles_read)
                        if s < num_to_sample:
                             sampled_articles_text[s] = article_to_sample
                    current_article_lines = []
                    # Log less frequently during sampling
                    # if articles_read % (total_articles_approx // 10 or 1) == 0:
                    #      logging.info(f"Scanning articles for sampling: {articles_read}/{total_articles_approx}...")

                elif not is_separator:
                    current_article_lines.append(line)
        # Handle last article
        if current_article_lines:
            articles_read += 1
            article_to_sample = "".join(current_article_lines).strip()
            if len(sampled_articles_text) < num_to_sample:
                sampled_articles_text.append(article_to_sample)
            else:
                 s = int(random.random() * articles_read)
                 if s < num_to_sample:
                      sampled_articles_text[s] = article_to_sample
        logging.info(f"Finished sampling {len(sampled_articles_text)} articles.")

        if not sampled_articles_text:
            logging.warning("No text could be sampled. Cannot analyze.")
            return

        logging.info(f"Analyzing {len(sampled_articles_text):,} sampled articles...")
        full_sample_text = " ".join(sampled_articles_text)
        words = re.findall(r'\b\w+\b', full_sample_text.lower())
        total_word_count = len(words)

        if total_word_count == 0:
            logging.warning("Sampled text contains no words after tokenization.")
            return

        keyword_counts = Counter()
        financial_word_count = 0
        for word in words:
            if word in keywords_set:
                keyword_counts[word] += 1
                financial_word_count += 1

        keyword_density = (financial_word_count / total_word_count) * 1000 if total_word_count > 0 else 0.0

        logging.info(f"\n--- Relevance Analysis Results ---")
        logging.info(f"Total words in sample: {total_word_count:,}")
        logging.info(f"Financial keywords found in sample: {financial_word_count:,}")
        logging.info(f"Keyword Density (per 1000 words): {keyword_density:.2f}")

        top_n = 20
        logging.info(f"\nTop {top_n} Financial Keywords Found in Sample:")
        for keyword, count in keyword_counts.most_common(top_n):
            logging.info(f"- {keyword}: {count:,}")

        # Heuristic Interpretation (adjusted threshold slightly)
        if keyword_density >= MIN_KEYWORD_DENSITY * 1.5:
             logging.info("\nInterpretation: Keyword density is good, suggesting the filtered corpus is likely relevant.")
        elif keyword_density >= MIN_KEYWORD_DENSITY * 0.75:
             logging.info("\nInterpretation: Keyword density is moderate. Filter seems okay, but check keywords/threshold.")
        else:
             logging.warning("\nInterpretation: Keyword density is low. Corpus might still contain significant noise, or keywords/threshold need adjustment.")

    except Exception as e:
        logging.error(f"An error occurred during relevance analysis: {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    filter_success = False
    try:
        # No langdetect needed here
        filter_success = clean_and_filter_corpus(INPUT_CORPUS_FILE, OUTPUT_CORPUS_FILE)

    except Exception as main_err:
         logging.error(f"An error occurred during filtering setup: {main_err}", exc_info=True)

    if filter_success and RUN_RELEVANCE_ANALYSIS:
         analyze_relevance(OUTPUT_CORPUS_FILE, NUM_LINES_TO_SAMPLE, FINANCIAL_KEYWORDS_SET)
    elif not filter_success:
         logging.error("Filtering step failed or did not run. Skipping relevance analysis.")
    else:
         logging.info("Relevance analysis is disabled (RUN_RELEVANCE_ANALYSIS=False).")
