import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch

dataset = pd.read_csv("New.csv", usecols=["Category", "Question", "Answer"])
print(dataset.head())

print("Categories: ")
print([i for i in dataset['Category'].unique()])

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=42)
print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}, Testing samples: {len(test_data)}")

train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_data)
dataset = DatasetDict({"train": train_dataset,"val": val_dataset, "test": test_dataset})

def preprocess_data(example):
    processed_example = {
        "text": f"Category: {example['Category']} | Question: {example['Question']} | Answer: {example['Answer']}"
    }
    return processed_example

dataset = dataset.map(preprocess_data, remove_columns=["Category", "Question", "Answer"])

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def tokenize_data(example):
    encodings = tokenizer(example["text"], padding="max_length", truncation=True, max_length=150, return_tensors='pt')

    encodings['labels'] = encodings['input_ids'].clone()

    return encodings

tokenized_dataset = dataset.map(tokenize_data, batched=True)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    save_total_limit=2,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("./gpt2-finetuned-final")
tokenizer.save_pretrained("./gpt2-finetuned-final")