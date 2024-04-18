
import pickle
import re
import string

class DialogueExtractor:
    def __init__(self):
        self.pattern = re.compile(r'^(\w+):\s*(.+?)\s*(?=\[|\w+:)', re.MULTILINE | re.DOTALL)

    def preprocess(self, script):
        dialogues = []
        matches = self.pattern.finditer(script)
        for match in matches:
            character = match.group(1).lower()
            quote = match.group(2)
            quote = re.sub(r'\(.*?\)', '', quote)
            quote = quote.translate(str.maketrans('', '', string.punctuation))
            # quote = ' '.join([word for word in quote.split() if word.lower() not in stop_words])
            quote = ' '.join([word.lower() for word in quote.split()])
            dialogues.append((character, quote))
        return dialogues

    def save_preprocessor(self, filename='dialogue_extractor.pkl'):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_preprocessor(cls, filename='dialogue_extractor.pkl'):
        with open(filename, 'rb') as file:
            return pickle.load(file)