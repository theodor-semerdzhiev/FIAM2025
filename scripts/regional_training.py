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

# --- Configuration ---

# 1. DEFINE THE REGION AND PATHS
region_name = "USA" # This script is now specifically for the combined USA data
base_corpus_dir = r'D:\market_data\text_data' # <-- MAKE SURE THIS PATH IS CORRECT
usa_dir = os.path.join(base_corpus_dir, region_name)

# --- MODIFIED: Define paths to BOTH input text files ---
scraped_text_file = os.path.join(usa_dir, 'corpus_cleaned.txt') # Assumes you ran clean_corpus.py
filings_text_file = os.path.join(usa_dir, 'usa_filings_corpus.txt') # Output from preprocess_pickles.py

# Create a list of files to load
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
# --- END MODIFICATION ---


# 2. CHOOSE BASE MODEL AND OUTPUT DIRECTORY
base_model_name = 'roberta-base'
output_model_dir = f'./{region_name}-fin-roberta-largecorpus' # Adjusted name slightly

# 3. TRAINING HYPERPARAMETERS (Optimized for ~12GB VRAM GPU)
num_train_epochs = 1              # Start with 1 epoch, consider more if time allows
per_device_train_batch_size = 4   # Keep batch size conservative for large dataset
gradient_accumulation_steps = 4   # Effective batch size = 16
save_steps = 10000                # Save checkpoints reasonably often on long runs
logging_steps = 500               # Log loss frequently
max_seq_length = 256              # Keep at 256 for memory, 512 if possible & needed
learning_rate=5e-5

# --- Script ---

print(f"--- Starting Continued Pre-training for: {region_name} (Combined Corpus) ---")
print(f"Using base model: {base_model_name}")
print(f"Loading text data from: {', '.join(data_files_list)}")
print(f"Output model will be saved to: {output_model_dir}")

# Check GPU availability
gpu_available = torch.cuda.is_available()
if gpu_available:
    # ...(GPU checks remain the same)...
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("WARNING: No GPU detected. Training will be very slow on CPU.")


os.makedirs(output_model_dir, exist_ok=True)

# 1. Load Tokenizer
# ...(Tokenizer loading remains the same)...
print(f"Loading tokenizer: {base_model_name}...")
tokenizer = RobertaTokenizerFast.from_pretrained(base_model_name, max_len=max_seq_length)
print("Tokenizer loaded.")


# 2. Load and Prepare Dataset
print("Loading dataset from multiple files...")
try:
    # --- MODIFIED: Load dataset from the list of files ---
    dataset = load_dataset('text', data_files={'train': data_files_list}, split='train')
    # --- END MODIFICATION ---
except Exception as e:
    print(f"ERROR: Failed to load dataset from {data_files_list}. Error: {e}")
    exit()

# Filter empty/short lines (important as preprocessing might create some)
dataset = dataset.filter(lambda example: example['text'] is not None and len(example['text'].strip()) > 10)
if len(dataset) == 0:
    print(f"ERROR: No valid lines found in the combined dataset after filtering.")
    exit()
print(f"Combined dataset loaded with {len(dataset):,} lines.")


# Tokenize the dataset
# ...(Tokenization logic remains the same)...
def tokenize_function(examples):
    return tokenizer(examples['text'],
                     truncation=True,
                     max_length=max_seq_length,
                     padding=False,
                     return_special_tokens_mask=False)

print("Tokenizing dataset (this might take a while)...")
ram_gb = psutil.virtual_memory().available / (1024**3)
print(f"Available System RAM: {ram_gb:.2f} GB")
num_cpus = os.cpu_count()
num_proc_tokenizer = max(1, min(num_cpus // 2, int(ram_gb // 4)))
print(f"Using {num_proc_tokenizer} processes for tokenization.")

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=num_proc_tokenizer,
    remove_columns=["text"]
)
print("Tokenization complete.")


# 3. Initialize Data Collator
# ...(Data Collator remains the same)...
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)


# 4. Load Base Model
# ...(Model loading remains the same)...
print(f"Loading base model: {base_model_name}...")
model = RobertaForMaskedLM.from_pretrained(base_model_name)
print("Base model loaded.")


# 5. Define Training Arguments
# ...(Training Arguments remain the same)...
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
# ...(Trainer initialization remains the same)...
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)


# 7. Start Training
# ...(Training execution remains the same)...
print("--- Starting MLM Training ---")
print(f"Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps * (torch.cuda.device_count() if gpu_available else 1)}")
print(f"Epochs: {num_train_epochs}, Max Seq Length: {max_seq_length}")
print(f"Using mixed precision (FP16): {gpu_available}")
trainer.train()
print("--- MLM Training Complete ---")


# 8. Save the Final Model and Tokenizer
# ...(Saving remains the same)...
print(f"Saving final adapted model to {output_model_dir}...")
trainer.save_model(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
print(f"Model and tokenizer saved successfully in {output_model_dir}")

