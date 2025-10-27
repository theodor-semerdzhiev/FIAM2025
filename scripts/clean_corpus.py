import os
import logging
# Removed fasttext import
import time
from langdetect import detect, LangDetectException # Import langdetect

# --- Configuration ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base directory where the USA folder is located
BASE_OUTPUT_DIR = r'D:\market_data\text_data' 


# Define input and output files within the USA directory
USA_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'USA')
INPUT_CORPUS_FILE = os.path.join(USA_OUTPUT_DIR, 'corpus.txt')
OUTPUT_CORPUS_FILE = os.path.join(USA_OUTPUT_DIR, 'corpus_en_only.txt')

MIN_ARTICLE_LENGTH = 100 # Minimum characters for an article to be considered valid
LOG_INTERVAL = 10000 # Log progress every N articles processed
CONFIDENCE_THRESHOLD = 0.90 # Optional: Only keep if langdetect is reasonably confident it's English

# --- Script ---

def filter_english_articles_langdetect(input_path, output_path): # Removed model_path
    """
    Reads a large text corpus, identifies articles separated by double newlines,
    detects language using langdetect, and writes only English articles to a new file.
    """
    logging.info(f"Starting language filtering process using langdetect.")
    logging.info(f"Input file: {input_path}")
    logging.info(f"Output file: {output_path}")
    logging.warning("This process might be slower than fastText.")

    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return
    # Model loading section removed

    start_time = time.time()
    articles_processed = 0
    articles_written = 0
    current_article_lines = []

    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            logging.info("Starting to process input file...")
            for line_num, line in enumerate(infile): # Added line number for better error reporting
                stripped_line = line.strip()

                # --- Article Boundary Logic ---
                # Consider consecutive blank lines as a separator
                is_separator = not stripped_line
                if is_separator and current_article_lines:
                     # Process the accumulated article buffer
                    article_text = "".join(current_article_lines).strip()
                    articles_processed += 1

                    if len(article_text) >= MIN_ARTICLE_LENGTH:
                        try:
                            # langdetect works directly on the text
                            # Keep internal newlines for writing, predict on cleaned text
                            prediction_text = article_text.replace('\n', ' ').replace('\r', ' ')
                            lang_code = detect(prediction_text) # Detect language

                            # Simple check if language is English
                            if lang_code == 'en':
                                # Optional: Add a confidence check if desired, though detect() doesn't directly return it easily
                                # Might need detect_langs() if confidence check is critical.
                                # For simplicity, let's just check 'en' for now.
                                outfile.write(article_text + '\n\n') # Write original article + separator
                                articles_written += 1
                            # else: # Optional: Log detected non-English language
                            #     logging.debug(f"Skipped article {articles_processed} (lang: {lang_code}) near line {line_num}")

                        except LangDetectException:
                            # Handle cases where langdetect cannot reliably determine the language (e.g., too short after cleaning, mixed language)
                            logging.debug(f"Could not reliably detect language for article {articles_processed} near line {line_num}. Skipping.")
                        except Exception as pred_err:
                            logging.warning(f"Error predicting language for article {articles_processed} near line {line_num}: {pred_err}")

                    # Log progress periodically
                    if articles_processed % LOG_INTERVAL == 0 and articles_processed > 0:
                        elapsed = time.time() - start_time
                        rate = articles_processed / elapsed if elapsed > 0 else 0
                        logging.info(f"Processed: {articles_processed:,} articles | Written (EN): {articles_written:,} | Rate: {rate:.1f} art/s")

                    current_article_lines = [] # Reset buffer after processing separator or empty buffer
                elif not is_separator:
                    # Non-blank line, add to buffer (keep original line ending)
                    current_article_lines.append(line)
                # If it's a separator line but the buffer was empty, just ignore it.


            # --- Process the very last article in the file ---
            if current_article_lines:
                article_text = "".join(current_article_lines).strip()
                articles_processed += 1
                if len(article_text) >= MIN_ARTICLE_LENGTH:
                     try:
                        prediction_text = article_text.replace('\n', ' ').replace('\r', ' ')
                        lang_code = detect(prediction_text)
                        if lang_code == 'en':
                            outfile.write(article_text + '\n\n') # Write if English
                            articles_written += 1
                     except LangDetectException:
                         logging.debug(f"Could not reliably detect language for final article. Skipping.")
                     except Exception as pred_err:
                         logging.warning(f"Error predicting language for final article: {pred_err}")

    except FileNotFoundError:
        logging.error(f"Input file disappeared during processing: {input_path}")
        return
    except Exception as e:
        logging.error(f"An error occurred during file processing: {e}", exc_info=True)
        return
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        logging.info("--- Language Filtering Complete ---")
        logging.info(f"Total articles processed: {articles_processed:,}")
        logging.info(f"Total English articles written: {articles_written:,}")
        logging.info(f"Output file: {output_path}")
        logging.info(f"Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    # Install langdetect first: pip install langdetect
    try:
        from langdetect import detect, LangDetectException
        filter_english_articles_langdetect(INPUT_CORPUS_FILE, OUTPUT_CORPUS_FILE)
    except ModuleNotFoundError:
         logging.error("The 'langdetect' library is not installed.")
         logging.error("Please install it by running: pip install langdetect")
    except Exception as main_err:
         logging.error(f"An error occurred: {main_err}", exc_info=True)


    

