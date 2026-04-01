import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from eval_model import evaluate_model

model_id = "Qwen/Qwen2.5-0.5B"

# CPU: no quantization, float32
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    dtype=torch.float32
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id = tokenizer.eos_token_id

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("ag_news")

# Use small subsets to keep CPU training feasible
train_dataset = dataset["train"].select(range(500))
test_dataset = dataset["test"].select(range(100))

labels = ['World', 'Sports', 'Business', 'Sci/Tech']


def build_prompt(text):
    return f"""Classify the news article.

Article:
{text}

Return ONLY the number of the correct label.

0 = World
1 = Sports
2 = Business
3 = Sci/Tech

Answer:"""


def tokenize_sample(sample):
    prompt = build_prompt(sample["text"]) + " " + str(sample["label"])
    tokens = tokenizer(prompt, truncation=True, max_length=256, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


train_dataset = train_dataset.map(tokenize_sample, remove_columns=train_dataset.column_names)
train_dataset.set_format("torch")


def extract_label(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    matches = re.findall(r"[0-3]", text)
    if matches:
        return int(matches[-1])
    return -1


# Baseline inference before training
sample = dataset["train"][10]
sample_text = sample["text"]
sample_label = sample["label"]

print("Ground truth label:", sample_label)
print("Sample text:")
print(sample_text[:300])

prompt = build_prompt(sample_text)
inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model prediction (before training):", extract_label(prediction))

# Training
training_args = TrainingArguments(
    output_dir="./lora_cpu_output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=50,
    save_strategy="epoch",
    fp16=False,
    bf16=False,
    dataloader_pin_memory=False,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    data_collator=data_collator,
)

print("\nStarting LoRA fine-tuning on CPU...")
trainer.train()
trainer.save_model("./lora_cpu_output/final")
tokenizer.save_pretrained("./lora_cpu_output/final")
print("Training complete. Model saved to ./lora_cpu_output/final")

# Evaluation
print("\nRunning evaluation on test set...")
model.eval()
device = next(model.parameters()).device  # use whatever device Trainer placed the model on
true_labels = []
pred_labels = []

for sample in tqdm(dataset["test"].select(range(100)), desc="Evaluating"):
    prompt = build_prompt(sample["text"])
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = extract_label(decoded)
    true_labels.append(sample["label"])
    pred_labels.append(pred)

valid = [(t, p) for t, p in zip(true_labels, pred_labels) if p != -1]
if valid:
    t_vals, p_vals = zip(*valid)
    acc = accuracy_score(t_vals, p_vals)
    f1 = f1_score(t_vals, p_vals, average="macro")
    print(f"\nEvaluation Results ({len(valid)}/{len(true_labels)} valid predictions)")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
else:
    print("No valid predictions extracted.")
