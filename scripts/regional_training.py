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

# 1. DEFINE THE REGION AND PATHS (MODIFY FOR EACH RUN)
region_name = "USA" # e.g., "USA", "EUROPE", "ANZ", "ASIA_EAST", etc.
# !! IMPORTANT: Update this path to where your regional folders are !!
# Example: base_corpus_dir = r'C:\_Files\School\Competitions\FIAM2025\data\regional_tuning_data'
base_corpus_dir = r'D:\market_data\text_data' # <-- MAKE SURE THIS PATH IS CORRECT
text_file = os.path.join(base_corpus_dir, region_name, 'corpus.txt') # Path to the specific corpus.txt

# 2. CHOOSE BASE MODEL AND OUTPUT DIRECTORY
base_model_name = 'roberta-base'
# Directory where the newly trained model will be saved
# Saved relative to where you run the script
output_model_dir = f'./{region_name}-fin-roberta' # Creates a folder like ./USA-fin-roberta

# 3. TRAINING HYPERPARAMETERS (Optimized for ~12GB VRAM GPU)
num_train_epochs = 1              # Start with 1 epoch for testing, increase later (e.g., 3-5)
per_device_train_batch_size = 4   # REDUCED from 8 for 12GB VRAM. Decrease to 2 if you get Out-of-Memory errors.
gradient_accumulation_steps = 4   # INCREASED from 2 to maintain effective batch size (4*4=16). Increase if you further decrease batch size.
save_steps = 5000                 # Save a checkpoint every N steps. Lower if training is slow and you want more frequent saves.
logging_steps = 200               # Log training loss every N steps. Lower for more frequent updates.
max_seq_length = 256              # Max length for text sequences. 256 is often okay, 512 uses significantly more VRAM.
learning_rate=5e-5                # Standard learning rate for fine-tuning/domain adaptation.

# --- Script ---

print(f"--- Starting Continued Pre-training for: {region_name} ---")
print(f"Using base model: {base_model_name}")

if not os.path.exists(text_file):
    print(f"ERROR: Corpus file not found at {text_file}")
    print("Please ensure the base_corpus_dir and region_name are set correctly.")
    exit()

print(f"Loading text data from: {text_file}")
print(f"Output model will be saved to: {output_model_dir}")

# Check GPU availability
gpu_available = torch.cuda.is_available()
if gpu_available:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("WARNING: No GPU detected. Training will be very slow on CPU.")

# Ensure output directory exists
os.makedirs(output_model_dir, exist_ok=True)

# 1. Load Tokenizer
print(f"Loading tokenizer: {base_model_name}...")
# Use RobertaTokenizerFast for speed
tokenizer = RobertaTokenizerFast.from_pretrained(base_model_name, max_len=max_seq_length)
print("Tokenizer loaded.")

# 2. Load and Prepare Dataset
print("Loading dataset...")
try:
    # Load text file line by line
    dataset = load_dataset('text', data_files={'train': text_file}, split='train')
except Exception as e:
    print(f"ERROR: Failed to load dataset from {text_file}. Error: {e}")
    exit()

# Basic filtering (optional, but good practice)
dataset = dataset.filter(lambda example: example['text'] is not None and len(example['text'].strip()) > 10)
if len(dataset) == 0:
    print(f"ERROR: No valid lines found in {text_file} after filtering.")
    exit()
print(f"Dataset loaded with {len(dataset):,} lines.")


# Tokenize the dataset
def tokenize_function(examples):
    # Tokenize the texts, truncate sequences longer than max_seq_length
    return tokenizer(examples['text'],
                     truncation=True,
                     max_length=max_seq_length,
                     padding=False, # Padding handled dynamically by collator
                     return_special_tokens_mask=False) # Not needed for basic MLM

print("Tokenizing dataset (this might take a while)...")
# Check system RAM before tokenizing
ram_gb = psutil.virtual_memory().available / (1024**3)
print(f"Available System RAM: {ram_gb:.2f} GB")
# Adjust num_proc based on available RAM and cores, be conservative
num_cpus = os.cpu_count()
num_proc_tokenizer = max(1, min(num_cpus // 2, int(ram_gb // 4))) # Heuristic: Use half cores, limit based on RAM
print(f"Using {num_proc_tokenizer} processes for tokenization.")

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=num_proc_tokenizer,
    remove_columns=["text"] # Remove original text column
)
print("Tokenization complete.")

# 3. Initialize Data Collator
# This dynamically creates the [MASK] tokens and labels for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15 # Standard masking probability
)

# 4. Load Base Model
print(f"Loading base model: {base_model_name}...")
# Make sure to load RobertaForMaskedLM for the MLM task
model = RobertaForMaskedLM.from_pretrained(base_model_name)
print("Base model loaded.")

# 5. Define Training Arguments
training_args = TrainingArguments(
    output_dir=output_model_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_strategy="steps", # Save based on steps
    save_steps=save_steps,
    save_total_limit=2, # Keep only the latest 2 checkpoints + the final one
    prediction_loss_only=True, # We only need the loss for pre-training
    fp16=gpu_available, # Use mixed precision if GPU detected
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    report_to="none", # Disable wandb/tensorboard reporting unless configured
    # Add evaluation strategy if you create a small validation split from your corpus
    # evaluation_strategy="steps",
    # eval_steps=save_steps,
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
    # Add eval_dataset=tokenized_eval_dataset if using evaluation
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

