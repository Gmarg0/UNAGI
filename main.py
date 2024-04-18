
# Main script to load, preprocess, filter, and train a model
import torch

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from script_processor import DialogueExtractor
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, \
    BertForSequenceClassification, RobertaForSequenceClassification, BertTokenizer, RobertaTokenizer


def main():
    # Path to the directory containing text filesx
    directory = 'data_files'
    extractor = DialogueExtractor()


    # Load and preprocess data
    df = load_data(directory, extractor)
    print("Data Loaded and Preprocessed")

    # Filter for specific characters
    characters_to_include = ['rachel', 'ross', 'monica', 'chandler', 'joey', 'phoebe']
    df = df[df['Character'].isin(characters_to_include)]
    print("Filtered Characters:", characters_to_include)

    # Tokenize the data

    # DistilBE Tokenizer
    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    #BERT Tokenizer
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


    encodings = tokenizer(df['Quote'].tolist(), truncation=True, padding=True, max_length=128)

    # Prepare the dataset
    class FriendsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    labels = pd.factorize(df['Character'])[0]
    dataset = FriendsDataset(encodings, labels)

    # Assuming `dataset` is your full dataset
    train_dataset, eval_dataset = train_test_split(dataset, test_size=0.1)  # 10% for validation




#
# ################## DistilBert ##################
#     # Load model
#     model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
#                                                                 num_labels=len(characters_to_include))
#
#     # Training settings
#     training_args = TrainingArguments(
#         output_dir='./results',
#         num_train_epochs=3,
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=16,
#         warmup_steps=500,
#         weight_decay=0.01,
#         logging_dir='./logs',
#         logging_steps=10,
#         evaluation_strategy="epoch",  # Evaluate the model at the end of each epoch
#         save_strategy="epoch",  # Save the model at the end of each epoch
#         load_best_model_at_end=True  # Load the best model found during training when finished
#     )
#
#     # Initialize Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,  # use the train dataset here
#         eval_dataset=eval_dataset  # use the validation dataset here
#     )
#
#
# ################## Bert ##################
#
#     # Load the pre-trained BERT model
#     model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(characters_to_include))
#
#     # Training settings
#     training_args = TrainingArguments(
#         output_dir='./results_bert',
#         num_train_epochs=3,
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=16,
#         warmup_steps=500,
#         weight_decay=0.01,
#         logging_dir='./logs_bert',
#         logging_steps=10,
#         evaluation_strategy="epoch"
#     )
#
#     # Initialize the Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset  # assuming you have a validation dataset
#     )
#
#
#


    ############## RoBERTa ################

    # Load the pre-trained RoBERTa model
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(characters_to_include))

    # # Training settings distilbert
    # training_args = TrainingArguments(
    #     output_dir='./results_roberta',
    #     num_train_epochs=3,
    #     per_device_train_batch_size=4,
    #     per_device_eval_batch_size=4,
    #     warmup_steps=500,
    #     weight_decay=0.01,
    #     logging_dir='./logs_roberta',
    #     logging_steps=10,
    #     evaluation_strategy="epoch"
    # )

    training_args = TrainingArguments(
        output_dir='./results_roberta',
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Increased from 4 to 8
        per_device_eval_batch_size=8,  # Increased from 4 to 8
        learning_rate=5e-5,  # Adjusted learning rate
        warmup_steps=300,  # Adjusted warmup steps
        weight_decay=0.01,
        logging_dir='./logs_roberta',
        logging_steps=10,
        evaluation_strategy="epoch"
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset  # assuming you have a validation dataset
    )



    #########################################


    # Start training
    trainer.train()

    print("Training complete")
    model_path = 'roBERTa'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

# Function to load data from text files in a directory
def load_data(directory, extractor):
    if os.path.exists('dialogue_df.feather'):
        return pd.read_feather('dialogue_df.feather')
    all_dialogues = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            path = os.path.join(directory, filename)
            with open(path, 'r', encoding='utf-8') as file:
                script = file.read()
                dialogues = extractor.preprocess(script)
                all_dialogues.extend(dialogues)
    dialogue_df = pd.DataFrame(all_dialogues, columns=['Character', 'Quote'])
    dialogue_df.to_feather('dialogue_df.feather')
    return dialogue_df

if __name__ == "__main__":
    main()