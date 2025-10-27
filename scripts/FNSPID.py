import os
import logging
from datasets import load_dataset
# from urllib.parse import urlparse # Not needed if saving all to USA
import time
import json # To save progress

# --- Configuration ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Base output directory
# !! IMPORTANT: Update this path !!
BASE_OUTPUT_DIR = r'D:\market_data\text_data' # <-- MAKE SURE THIS PATH IS CORRECT
USA_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'USA')
os.makedirs(USA_OUTPUT_DIR, exist_ok=True)
USA_CORPUS_FILE = os.path.join(USA_OUTPUT_DIR, 'corpus.txt')
logging.info(f"Output directory (all data): {USA_OUTPUT_DIR}")
logging.info(f"Output corpus file: {USA_CORPUS_FILE}")

# --- Dataset Configuration ---
DATASET_NAME = "Zihan1004/FNSPID"
DATASET_SPLIT = "train"

# --- Column Names (Based on previous user input/assumptions) ---
# !! VERIFY THESE from dataset card or /first-rows if possible !!
TEXT_COLUMN = 'Article'
URL_COLUMN = 'Url' # Keep for logging errors

# --- Progress Tracking ---
PROGRESS_FILE = os.path.join(BASE_OUTPUT_DIR, 'zihan_fnspid_progress.json') # New progress file
SAVE_PROGRESS_INTERVAL = 10000
LOG_PROGRESS_INTERVAL = 1000

# --- Helper Functions ---

def save_text_usa(text, output_filepath):
    """Appends the article text to the USA corpus file."""
    try:
        if isinstance(text, str):
             text_to_save = text.strip()
             if text_to_save:
                 with open(output_filepath, 'a', encoding='utf-8') as f:
                     f.write(text_to_save + '\n\n')
    except Exception as e:
        logging.error(f"Could not write to corpus file: {output_filepath}. Error: {e}")

def load_progress():
    """Loads the last successfully processed index."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                progress = json.load(f)
                count = progress.get('processed_count', 0)
                logging.info(f"Loaded progress: {count:,} articles previously processed.")
                return count
        except Exception as e:
            logging.warning(f"Could not load progress file ({PROGRESS_FILE}): {e}. Starting from 0.")
    else:
        logging.info("Progress file ({}) not found. Starting from index 0.".format(PROGRESS_FILE))
    return 0

def save_progress(count):
    """Saves the last successfully processed index."""
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({'processed_count': count}, f)
    except Exception as e:
        logging.error(f"Could not save progress ({PROGRESS_FILE}): {e}")

# --- Main Processing Function ---

def process_zihan_dataset():
    logging.info(f"--- Starting Dataset Processor for {DATASET_NAME} (USA Only) ---")
    logging.warning("Streaming dataset. This may take several hours depending on size.")

    processed_count_start_of_run = load_progress()
    logging.info(f"Attempting to resume from article index: {processed_count_start_of_run:,}")

    total_added_this_session = 0
    start_time = time.time()
    data_stream = None
    dataset_iterable = None

    try:
        logging.info(f"Initializing {DATASET_NAME} stream using load_dataset...")
        # Use load_dataset directly, hoping it works for this Parquet-based dataset
        dataset_iterable = load_dataset(DATASET_NAME, streaming=True, split=DATASET_SPLIT)
        data_stream = iter(dataset_iterable)
        logging.info("Stream initialized.")

    except Exception as load_err:
        logging.error(f"Failed to initialize dataset stream via load_dataset: {load_err}", exc_info=True)
        # Check if the error is the 'Dataset scripts not supported' one again
        if "Dataset scripts are no longer supported" in str(load_err):
             logging.error("This dataset might still rely on an unsupported loading script.")
        return

    current_article_index = 0
    try:
        # --- Skipping Logic ---
        if processed_count_start_of_run > 0:
            logging.info(f"Attempting to skip the first {processed_count_start_of_run:,} articles...")
            skipped_count = 0
            skip_start_time = time.time()
            try:
                for _ in range(processed_count_start_of_run):
                    next(data_stream)
                    skipped_count += 1
                    if skipped_count % 100000 == 0:
                        logging.info(f"Skipped {skipped_count:,} / {processed_count_start_of_run:,} articles...")
                current_article_index = skipped_count
                logging.info(f"Skipping complete. Took {time.time() - skip_start_time:.2f} seconds.")
            except StopIteration:
                 logging.warning("Stream ended while trying to skip. Progress file might be outdated.")
                 current_article_index = skipped_count
                 save_progress(current_article_index)
                 return
            except Exception as skip_err:
                 logging.error(f"Error during skipping at article {skipped_count}: {skip_err}", exc_info=True)
                 current_article_index = skipped_count
                 save_progress(current_article_index)
                 return
        else:
             current_article_index = 0
        # --- End Skipping ---

        logging.info(f"Starting article processing loop from index {current_article_index:,}...")
        loop_start_time = time.time()
        articles_processed_in_loop = 0

        while True:
            try:
                article = next(data_stream)
                current_article_index += 1
                articles_processed_in_loop += 1

                # --- Logging & Saving ---
                if articles_processed_in_loop % LOG_PROGRESS_INTERVAL == 0:
                    now = time.time()
                    loop_elapsed = now - loop_start_time
                    rate = articles_processed_in_loop / loop_elapsed if loop_elapsed > 0 else 0
                    logging.info(f"Processing index {current_article_index:,}... (Rate: {rate:.1f}/s)")

                if current_article_index % SAVE_PROGRESS_INTERVAL == 0:
                     save_progress(current_article_index)

                # --- Simplified Logic: Save all valid articles to USA corpus ---
                text = article.get(TEXT_COLUMN)
                url = article.get(URL_COLUMN) # Keep url for potential error logging

                # Basic validation
                if not text or not isinstance(text, str) or len(text) < 100: continue # Skip short/empty/non-string text

                # Save the text directly to the USA corpus file
                save_text_usa(text, USA_CORPUS_FILE)
                total_added_this_session += 1
                # --- End Simplified Logic ---

            except StopIteration:
                logging.info(f"Dataset stream finished cleanly after processing article {current_article_index:,}.")
                save_progress(current_article_index)
                break # Exit while loop
            except Exception as e:
                # Log error minimally, save progress, and try to continue
                logging.error(f"Error processing article #{current_article_index} (URL: {url if url else 'N/A'}): {e}", exc_info=False)
                save_progress(current_article_index)
                logging.warning("Attempting to continue to next article...")
                time.sleep(0.5) # Short pause on error

    except KeyboardInterrupt:
        logging.info("\n--- Dataset Processor stopped by user ---")
        final_idx_on_stop = current_article_index if 'current_article_index' in locals() else processed_count_start_of_run
        save_progress(final_idx_on_stop)
    except Exception as e:
        logging.error(f"\n--- An unexpected error occurred outside main loop: {e} ---", exc_info=True)
        final_idx_on_error = current_article_index if 'current_article_index' in locals() else processed_count_start_of_run
        save_progress(final_idx_on_error)
    finally:
        session_elapsed = time.time() - start_time
        final_processed_index = load_progress()
        processed_this_session = final_processed_index - processed_count_start_of_run
        logging.info(f"Finished session. Processed approximately {processed_this_session:,} articles in {session_elapsed:.2f} seconds.")
        logging.info(f"Added {total_added_this_session:,} articles to USA corpus during this session.")
        logging.info(f"Total rows processed overall (check progress file): {final_processed_index:,}")

if __name__ == "__main__":
    process_zihan_dataset()
