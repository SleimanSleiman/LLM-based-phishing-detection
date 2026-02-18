from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text = "Test email: please update your billing information at http://example.com"
print(tokenizer(text, truncation=True, max_length=128))