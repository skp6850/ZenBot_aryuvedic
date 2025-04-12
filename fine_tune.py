import os
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# Step 1: Load the dataset
csv_file = 'qa_output.csv'

save_directory = "./local_llama_model1"
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModelForCausalLM.from_pretrained(save_directory)
tokenizer.pad_token = tokenizer.eos_token

# Step 2: Load and process the dataset
def load_qa_dataset(csv_file):
    dataset = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row['Question']
            answer = row['Answer']
            dataset.append({"input_text": question, "output_text": answer})
    return Dataset.from_dict({"input_text": [d['input_text'] for d in dataset],
                              "output_text": [d['output_text'] for d in dataset]})

qa_dataset = load_qa_dataset(csv_file)

# Sample a fraction of the dataset, e.g., 10% for faster training
sampled_dataset = qa_dataset.train_test_split(test_size=0.90)['train']

# Step 3: Preprocess the dataset for the model
def preprocess_function(examples):
    inputs = [q for q in examples['input_text']]
    targets = [a for a in examples['output_text']]
    # Reduce max_length to 256 to speed up training
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = sampled_dataset.map(preprocess_function, batched=True)

# Step 4: Set training arguments (optimized for faster runs)
training_args = TrainingArguments(
    output_dir="./fine_tuned_llama_model",

    overwrite_output_dir=True,
    per_device_train_batch_size=4,  # Increased batch size for faster runs
    num_train_epochs=1,  # Reduced number of epochs to speed up training
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="epoch",
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=500,  # Log progress less frequently to reduce overhead
    report_to="none",  # Disable external reporting (e.g., to WandB)
    no_cuda=False,  # Ensure CPU is used to avoid CUDA errors
)

# Step 5: Define a data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 6: Initialize the Trainer
trainer = Trainer(

    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Step 7: Fine-tune the model
trainer.train()

# Step 8: Save the fine-tuned model
model.save_pretrained("./fine_tuned_llama_model")
tokenizer.save_pretrained("./fine_tuned_llama_model")

print("Model has been fine-tuned and saved to './fine_tuned_llama_model'")