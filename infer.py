import argparse
import csv
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_device(choice):
    if choice == "auto":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available on this machine.")
        return torch.device("mps")
    return torch.device("cpu")

def infer_batch(model, tokenizer, texts, device, max_length=512):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    preds = probs.argmax(axis=1)
    results = []
    for i, text in enumerate(texts):
        pred_id = int(preds[i])
        label = model.config.id2label.get(pred_id, str(pred_id))
        score = float(probs[i, pred_id])
        results.append((label, score, text))
    return results

def write_csv(path, rows, write_header=True):
    exists = os.path.exists(path)
    mode = "a" if exists else "w"
    with open(path, mode, newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists and write_header:
            writer.writerow(["label", "score", "text"])
        for r in rows:
            writer.writerow([r[0], r[1], r[2]])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="output", help="Path to model/tokenizer directory")
    parser.add_argument("--text", help="Single text input")
    parser.add_argument("--input-file", help="Path to file containing one text per line")
    parser.add_argument("--output-csv", help="Optional CSV output path")
    parser.add_argument("--device", choices=["auto","cpu","mps"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    if not args.text and not args.input_file:
        parser.error("Provide either --text or --input-file")

    device = get_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()

    all_results = []
    if args.text:
        results = infer_batch(model, tokenizer, [args.text], device, max_length=args.max_length)
        for label, score, text in results:
            print({"label": label, "score": round(score, 6), "text": text})
        if args.output_csv:
            write_csv(args.output_csv, results)
        return

    # input-file path
    with open(args.input_file, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    for i in range(0, len(lines), args.batch_size):
        batch = lines[i:i+args.batch_size]
        results = infer_batch(model, tokenizer, batch, device, max_length=args.max_length)
        for label, score, text in results:
            print({"label": label, "score": round(score, 6), "text": text})
        all_results.extend(results)
        if args.output_csv:
            write_csv(args.output_csv, results, write_header=(i==0))

if __name__ == "__main__":
    main()