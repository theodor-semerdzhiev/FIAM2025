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
import os # Import os for cpu_count

# --- Configuration ---

# 1. DEFINE THE REGION AND PATHS
# --- MODIFIED: Simplified paths for a Linux VM ---
# We assume the script, data, and output will all live in the same directory
# (e.g., /home/jupyter/ or /home/your_user/)

# The corpus file MUST be in the same directory as this python script
filings_2016_file = 'usa_filings_corpus_2005_2016.txt'

data_files_list = []
if os.path.exists(filings_2016_file):
    data_files_list.append(filings_2016_file)
    print(f"Found 2005-2016 filings corpus: {filings_2016_file}")
else:
    print(f"ERROR: 2005-2016 filings corpus not found at {filings_2016_file}")
    print("Please make sure you've copied it from your GCS bucket to this directory.")
    exit()
# --- END MODIFICATION ---


# 2. CHOOSE BASE MODEL AND OUTPUT DIRECTORY
base_model_name = 'roberta-base'
# --- MODIFIED: Output dir will be relative to this script ---
output_model_dir = 'USA-fin-roberta-filings-2016'

# 3. TRAINING HYPERPARAMETERS
num_train_epochs = 3 # Let's train for 3 full passes
per_device_train_batch_size = 128 
gradient_accumulation_steps = 1  # Keep this at 1 for max speed
save_steps = 5000 # Save a checkpoint every 5,000 steps
logging_steps = 100 # Log more frequently
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
    print(f"--- Starting Continued Pre-training (GCP A100) for: USA (Filings 2005-2016) ---")
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
        # GCP VM will have high-speed internet, this will be fast
        dataset = load_dataset('text', data_files={'train': data_files_list}, split='train')
    except Exception as e:
        print(f"ERROR: Failed to load dataset from {data_files_list}. Error: {e}")
        exit()

    print("Filtering dataset...")
    filtered_dataset = dataset.filter(
        lambda example: example['text'] is not None and len(example['text'].strip()) > 10,
        num_proc=1 
    )

    if len(filtered_dataset) == 0:
        print(f"ERROR: No valid lines found in the dataset after filtering.")
        exit()
    print(f"Dataset loaded and filtered with {len(filtered_dataset):,} lines.")

    # Tokenize the dataset
    print("Tokenizing dataset (this might take a while)...")
    ram_gb = psutil.virtual_memory().available / (1024**3)
    print(f"Available System RAM: {ram_gb:.2f} GB")
    
    # --- MODIFIED: Use all available CPU cores on the VM ---
    num_proc_tokenizer = os.cpu_count()
    print(f"Using {num_proc_tokenizer} processes for tokenization.")
    # --- END MODIFICATION ---

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
        save_total_limit=3, # Keep the 3 most recent checkpoints
        prediction_loss_only=True,
        fp16=gpu_available, # Use mixed precision (FP16)
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
    # trainer.train(resume_from_checkpoint=True)
    # --- We are starting from scratch OR resuming from a home checkpoint
    
    # --- IMPORTANT ---
    # Change this line based on what you're doing:
    
    # 1. To start training from scratch:
    # trainer.train()
    
    # 2. To resume from a checkpoint you uploaded (e.g., 'checkpoint-70000'):
    trainer.train(resume_from_checkpoint=True)
    
    print("--- MLM Training Complete ---")

    # 8. Save the Final Model and Tokenizer
    print(f"Saving final adapted model to {output_model_dir}...")
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"Model and tokenizer saved successfully in {output_model_dir}")


if __name__ == "__main__":
    # freeze_support() # Not needed on Linux
    main()
