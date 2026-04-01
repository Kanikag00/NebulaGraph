import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(model, tokenizer, build_prompt, extract_label, dataset):
    preds = []
    refs = []

    model.eval()
    device = next(model.parameters()).device

    for example in tqdm(dataset, desc="Evaluating"):
        text = example["text"]
        label = example["label"]

        prompt = build_prompt(text)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_label = extract_label(prediction)

        preds.append(pred_label)
        refs.append(label)

    valid = [(r, p) for r, p in zip(refs, preds) if p != -1]
    if valid:
        r_vals, p_vals = zip(*valid)
        acc = accuracy_score(r_vals, p_vals)
        f1 = f1_score(r_vals, p_vals, average="weighted")
        print(f"\nEvaluation Results ({len(valid)}/{len(refs)} valid predictions)")
        print(f"Accuracy : {acc:.4f}")
        print(f"F1 Score : {f1:.4f}")
    else:
        print("No valid predictions extracted.")
