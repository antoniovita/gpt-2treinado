from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
    DataCollatorForLanguageModeling
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch


MODEL_CHECKPOINT = "./results_v2"  
DATASET_PATH = "./treinos3.jsonl"
OUTPUT_DIR = "./results_v3"
MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 3
LR = 5e-5
LOGGING_STEPS = 50


dataset = load_dataset('json', data_files=DATASET_PATH)
dataset = dataset["train"].train_test_split(test_size=0.1)


def preprocess(example):
    return {"text": f"{example['prompt']} {example['completion']}"}

dataset = dataset.map(preprocess)


tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

tokenized = dataset.map(tokenize, batched=True)


tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]})


tokenized = tokenized.remove_columns(["text", "prompt", "completion"])
tokenized.set_format("torch")


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

train_dataloader = DataLoader(
    tokenized["train"], shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator
)

eval_dataloader = DataLoader(
    tokenized["test"], batch_size=BATCH_SIZE, collate_fn=data_collator
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_CHECKPOINT,
    pad_token_id=tokenizer.pad_token_id
)


accelerator = Accelerator()

train_dataloader, eval_dataloader, model = accelerator.prepare(
    train_dataloader, eval_dataloader, model
)


optimizer = AdamW(model.parameters(), lr=LR)

num_training_steps = EPOCHS * len(train_dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


progress_bar = tqdm(range(num_training_steps))

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()

        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if step % LOGGING_STEPS == 0 and step > 0:
            avg_loss = total_loss / LOGGING_STEPS
            print(f"Epoch {epoch+1} | Step {step} | Avg Loss: {avg_loss:.4f}")
            total_loss = 0

        progress_bar.update(1)

    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            outputs = model(**batch)
            eval_loss += outputs.loss.detach().float()

    avg_eval_loss = eval_loss / len(eval_dataloader)
    print(f"👉 Epoch {epoch+1} finished | Validation Loss: {avg_eval_loss:.4f}")


model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Modelo treinado e salvo em {OUTPUT_DIR}")
