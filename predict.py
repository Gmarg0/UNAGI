import os

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, BertTokenizer, BertForSequenceClassification
import torch


def load_model(model_path, is_local):
    # Load the trained model and tokenizer
    # model = DistilBertForSequenceClassification.from_pretrained(model_path, use_safetensors=True)
    # tokenizer = DistilBertTokenizer.from_pretrained(model_path)

    model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=is_local)
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=is_local)
    return model, tokenizer


def predict_quote(model, tokenizer, quote):
    # Prepare the quote for the model
    inputs = tokenizer(quote, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Predict
    with torch.no_grad():
        logits = model(**inputs).logits

    # Convert to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get scores for each character
    return probabilities


# # Load the model from local
# model_name = os.path.join(os.getcwd(), 'centalBERT')
# is_local = True

# Load the model from huggingface
model_name = "GalMargo/centralBert"
is_local = False


model, tokenizer = load_model(model_name, is_local)




# Example quote"
quote = "But they don't know we know they know"
probabilities = predict_quote(model, tokenizer, quote)
print("Probabilities:", probabilities)
# Assuming you know the order of characters in the model output layer:
characters = ['Rachel', 'Ross', 'Monica', 'Chandler', 'Joey', 'Phoebe']
for i, character in enumerate(characters):
    print(f"{character}: {probabilities[0][i].item():.4f}")