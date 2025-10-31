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
# Base directory where the USA folder is located
BASE_OUTPUT_DIR = r'D:\market_data\text_data' 
USA_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, region_name)

# --- MODIFIED: Point ONLY to the new 2016 filings corpus ---
filings_2016_file = os.path.join(USA_OUTPUT_DIR, 'usa_filings_corpus_2005_2016.txt')

data_files_list = []
if os.path.exists(filings_2016_file):
    data_files_list.append(filings_2016_file)
    print(f"Found 2005-2016 filings corpus: {filings_2016_file}")
else:
    print(f"ERROR: 2005-2016 filings corpus not found at {filings_2016_file}")
    print("Please run the 'preprocess_filings_2016.py' script first.")
    exit()
# --- END MODIFICATION ---


# 2. CHOOSE BASE MODEL AND OUTPUT DIRECTORY
base_model_name = 'roberta-base'
# --- MODIFIED: New output model name ---
output_model_dir = f'./{region_name}-fin-roberta-filings-2016'

# 3. TRAINING HYPERPARAMETERS
# --- MODIFIED: Increased epochs since dataset is smaller ---
num_train_epochs = 1 # Let's train for 3 full passes
per_device_train_batch_size = 32 # Keep this as high as your VRAM allows
gradient_accumulation_steps = 1  # Keep this at 1 for max speed
save_steps = 10000 # Save a checkpoint every 10,000 steps
logging_steps = 500
max_seq_length = 256
learning_rate=5e-5

# --- Tokenization ---
print(f"Loading tokenizer: {base_model_name}...")
tokenizer = RobertaTokenizerFast.from_pretrained(base_model_name, max_len=max_seq_length)
print("Tokenizer loaded.")

def tokenize_function(examples):
    return tokenizer(examples['text'],
                     truncation=True,
                     max_length=max_seq_length,
                     padding=False,
                     return_special_tokens_mask=False)

# --- Main Execution Block ---
def main():
    print(f"--- Starting Continued Pre-training for: {region_name} (Filings 2005-2016) ---")
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
    print("Loading dataset...")
    try:
        dataset = load_dataset('text', data_files={'train': data_files_list}, split='train')
    except Exception as e:
        print(f"ERROR: Failed to load dataset from {data_files_list}. Error: {e}")
        exit()

    print("Filtering dataset...")
    filtered_dataset = dataset.filter(
        lambda example: example['text'] is not None and len(example['text'].strip()) > 10,
        num_proc=1 # num_proc=1 is fine for the filter step
    )

    if len(filtered_dataset) == 0:
        print(f"ERROR: No valid lines found in the dataset after filtering.")
        exit()
    print(f"Dataset loaded and filtered with {len(filtered_dataset):,} lines.")

    # Tokenize the dataset
    print("Tokenizing dataset (this might take a while)...")
    ram_gb = psutil.virtual_memory().available / (1024**3)
    print(f"Available System RAM: {ram_gb:.2f} GB")
    num_cpus = os.cpu_count()
    
    # Use more processes if you have RAM, but 1-2 is fine on your low-RAM system
    num_proc_tokenizer = 1 if ram_gb < 8 else max(1, min(num_cpus // 2, 4))
    print(f"Using {num_proc_tokenizer} processes for tokenization.")

    try:
        tokenized_dataset = filtered_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc_tokenizer,
            remove_columns=["text"]
        )
    except Exception as e:
        print(f"\n--- FAILED DURING .map() ---")
        print(f"ERROR: {e}")
        print(f"This might be a RAM issue. Your system has {ram_gb:.2f} GB of RAM.")
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
        save_total_limit=2, # Keeps the 2 most recent checkpoints
        prediction_loss_only=True,
        fp16=gpu_available, # Use mixed precision if GPU is available
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

    # Use resume_from_checkpoint=True if you want to resume a stopped job
    trainer.train(resume_from_checkpoint=True)
    print("--- MLM Training Complete ---")

    # 8. Save the Final Model and Tokenizer
    print(f"Saving final adapted model to {output_model_dir}...")
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"Model and tokenizer saved successfully in {output_model_dir}")


if __name__ == "__main__":
    freeze_support() # Call this first
    main()           # Then call main()
