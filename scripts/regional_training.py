import os
import torch
from datasets import load_dataset
from transformers import (
    RobertaConfig,
    RobertaTokenizerFast, # Use the Fast version for efficiency
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
import psutil # For checking system RAM
from multiprocess import freeze_support # Use the 'multiprocess' fork

# --- Configuration ---

# 1. DEFINE THE REGION AND PATHS
region_name = "USA"
base_corpus_dir = r'D:\market_data\text_data' # <-- MAKE SURE THIS PATH IS CORRECT
usa_dir = os.path.join(base_corpus_dir, region_name)
scraped_text_file = os.path.join(usa_dir, 'corpus_cleaned.txt')
filings_text_file = os.path.join(usa_dir, 'usa_filings_corpus.txt')

data_files_list = []
if os.path.exists(scraped_text_file):
    data_files_list.append(scraped_text_file)
    print(f"Found scraped corpus: {scraped_text_file}")
else:
    print(f"WARNING: Scraped corpus not found at {scraped_text_file}, skipping.")

if os.path.exists(filings_text_file):
    data_files_list.append(filings_text_file)
    print(f"Found filings corpus: {filings_text_file}")
else:
    print(f"WARNING: Filings corpus not found at {filings_text_file}, skipping.")

if not data_files_list:
    print("ERROR: No input corpus files found. Please run preprocessing scripts.")
    exit()


# 2. CHOOSE BASE MODEL AND OUTPUT DIRECTORY
base_model_name = 'roberta-base'
output_model_dir = f'./{region_name}-fin-roberta-largecorpus'

# 3. TRAINING HYPERPARAMETERS
num_train_epochs = 1
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
save_steps = 10000
logging_steps = 500
max_seq_length = 256
learning_rate=5e-5

# --- Tokenization ---
# Load the tokenizer *once* in the global scope.
# This makes it a read-only object that child processes can safely import.
print(f"Loading tokenizer: {base_model_name}...")
tokenizer = RobertaTokenizerFast.from_pretrained(base_model_name, max_len=max_seq_length)
print("Tokenizer loaded.")

def tokenize_function(examples):
    # This function now safely uses the globally-loaded tokenizer
    return tokenizer(examples['text'],
                     truncation=True,
                     max_length=max_seq_length,
                     padding=False,
                     return_special_tokens_mask=False)

# --- Main Execution Block ---
def main():
    print(f"--- Starting Continued Pre-training for: {region_name} (Combined Corpus) ---")
    print(f"Using base model: {base_model_name}")
    print(f"Loading text data from: {', '.join(data_files_list)}")
    print(f"Output model will be saved to: {output_model_dir}")

    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("WARNING: No GPU detected. Training will be very slow on CPU.")

    os.makedirs(output_model_dir, exist_ok=True)

    # 1. Load and Prepare Dataset
    print("Loading dataset from multiple files...")
    try:
        dataset = load_dataset('text', data_files={'train': data_files_list}, split='train')
    except Exception as e:
        print(f"ERROR: Failed to load dataset from {data_files_list}. Error: {e}")
        exit()

    print("Filtering dataset...")
    # --- MODIFIED: Force num_proc=1 for filter to avoid PermissionError ---
    filtered_dataset = dataset.filter(
        lambda example: example['text'] is not None and len(example['text'].strip()) > 10,
        num_proc=1 # Disable multiprocessing for filter step
    )
    # --- END MODIFICATION ---

    if len(filtered_dataset) == 0:
        print(f"ERROR: No valid lines found in the combined dataset after filtering.")
        exit()
    print(f"Combined dataset loaded and filtered with {len(filtered_dataset):,} lines.")

    # Tokenize the dataset
    print("Tokenizing dataset (this might take a while)...")
    ram_gb = psutil.virtual_memory().available / (1024**3)
    print(f"Available System RAM: {ram_gb:.2f} GB")
    num_cpus = os.cpu_count()
    
    # Keep num_proc=1 if RAM is low, otherwise allow multiprocessing
    # Your 1.75 GB of RAM will force this to 1
    num_proc_tokenizer = 1 if ram_gb < 8 else max(1, min(num_cpus // 2, int(ram_gb // 4)))
    print(f"Using {num_proc_tokenizer} processes for tokenization.")

    # --- THIS IS THE STEP THAT WILL LIKELY FAIL NEXT ---
    try:
        tokenized_dataset = filtered_dataset.map(
            tokenize_function, # Uses the global tokenizer
            batched=True,
            num_proc=num_proc_tokenizer,
            remove_columns=["text"]
        )
    except Exception as e:
        print(f"\n--- FAILED DURING .map() ---")
        print(f"ERROR: {e}")
        print("This is the expected 'RAM Wall'. Your system has {ram_gb:.2f} GB of RAM, which is not enough for this 34GB+ dataset.")
        print("Please move this script to a high-RAM environment like Google Colab Pro.")
        exit()
        
    print("Tokenization complete.")

    # 3. Initialize Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # 4. Load Base Model
    print(f"Loading base model: {base_model_name}...")
    model = RobertaForMaskedLM.from_pretrained(base_model_name)
    print("Base model loaded.")

    # 5. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_model_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=gpu_available,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        report_to="none",
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    # 7. Start Training
    print("--- Starting MLM Training ---")
    print(f"Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps * (torch.cuda.device_count() if gpu_available else 1)}")
    print(f"Epochs: {num_train_epochs}, Max Seq Length: {max_seq_length}")
    print(f"Using mixed precision (FP16): {gpu_available}")

    trainer.train()
    print("--- MLM Training Complete ---")

    # 8. Save the Final Model and Tokenizer
    print(f"Saving final adapted model to {output_model_dir}...")
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"Model and tokenizer saved successfully in {output_model_dir}")


# This guard is CRITICAL for multiprocessing
if __name__ == "__main__":
    freeze_support() # Call this first
    main()           # Then call main()
