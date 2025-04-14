import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import re
import nltk

device = 'mps' if not torch.cuda.is_available() else 'cuda'
#############################################
# 1. Load the Training Dataset and Rename Columns
#############################################

data_path = "./data/SlangIJCNLP/train.tsv"
# If your file does not include a header row, use header=None.
# In our case, we specify header=None and then assign custom column names.
df = pd.read_csv(data_path, sep="\t", header=None)
# Rename columns:
# Based on your info:
# - Column 0: target (non-standard expression)
# - Column 1: explanation
# - Column 2: sentence (context)
# - Column 3: label (ignored)
df.columns = ['target', 'explanation', 'sentence', 'label']
# df = pd.read_csv("hf://datasets/MLBtrio/genz-slang-dataset/all_slangs.csv")


print("Dataset info after renaming columns:")
print(df.info())
print(df.head())

#############################################
# 2. Build Vocabularies
#############################################

# Define special tokens and indices
PAD_TOKEN = "<PAD>"  # used for padding
SOS_TOKEN = "<SOS>"  # start-of-sentence
EOS_TOKEN = "<EOS>"  # end-of-sentence
UNK_TOKEN = "<UNK>"  # unknown token

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

def tokenize_text(text):
    # Simple whitespace and punctuation tokenization; adjust as needed.
    tokens = nltk.word_tokenize(text)
    return tokens

def build_vocab(sentences, special_tokens=SPECIAL_TOKENS):
    vocab = {}
    idx = 0
    for token in special_tokens:
        vocab[token] = idx
        idx += 1
    for sent in sentences:
        for word in tokenize_text(sent):
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

def build_char_vocab(texts, special_tokens=SPECIAL_TOKENS):
    vocab = {}
    idx = 0
    for token in special_tokens:
        vocab[token] = idx
        idx += 1
    for text in texts:
        text = text.strip()
        for ch in text:
            if ch not in vocab:
                vocab[ch] = idx
                idx += 1
    return vocab

# Build vocabularies for each modality:
# - Use context from the "sentence" column (word-level)
# - Use explanation from the "explanation" column (target words for generation)
# - Use the target non-standard expression from the "target" column (character-level)
word_vocab = build_vocab(df["sentence"].map(str).tolist())
explanation_vocab = build_vocab(df["explanation"].map(str).tolist())  # for explanation output tokens
char_vocab = build_char_vocab(df["target"].map(str).tolist())

print("Word vocab size:", len(word_vocab))
print("Explanation vocab size:", len(explanation_vocab))
print("Char vocab size:", len(char_vocab))

#############################################
# 3. Custom Dataset and Collate Function
#############################################
class SlangDataset(Dataset):
    def __init__(self, dataframe, word_vocab2, char_vocab2, explanation_vocab2):
        self.data = dataframe
        self.word_vocab = word_vocab2
        self.char_vocab = char_vocab2
        self.explanation_vocab = explanation_vocab2

    def __len__(self):
        return len(self.data)

    def encode_sentence(self, sentence):
        # Tokenize sentence and convert words to indices; unknowns mapped to UNK.
        tokens = tokenize_text(sentence)
        return [self.word_vocab.get(tok, self.word_vocab[UNK_TOKEN]) for tok in tokens]

    def encode_explanation(self, explanation):
        # Tokenize explanation and add SOS and EOS tokens.
        tokens = tokenize_text(explanation)
        return ([self.explanation_vocab[SOS_TOKEN]] +
                [self.explanation_vocab.get(tok, self.explanation_vocab[UNK_TOKEN]) for tok in tokens] +
                [self.explanation_vocab[EOS_TOKEN]])

    def encode_target(self, target):
        # Encode the target (non-standard expression) as sequence of characters.
        return [self.char_vocab.get(ch, self.char_vocab[UNK_TOKEN]) for ch in list(target.strip())]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sentence_indices = self.encode_sentence(row["sentence"])
        target_char_indices = self.encode_target(row["target"])
        explanation_indices = self.encode_explanation(row["explanation"])
        return (torch.tensor(sentence_indices, dtype=torch.long).to(device),
                torch.tensor(target_char_indices, dtype=torch.long).to(device),
                torch.tensor(explanation_indices, dtype=torch.long).to(device))

def collate_fn(batch):
    # Each batch element is a tuple: (sentence, target_char, explanation)
    sentences, target_chars, explanations = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=word_vocab[PAD_TOKEN])
    target_chars_padded = pad_sequence(target_chars, batch_first=True, padding_value=char_vocab[PAD_TOKEN])
    explanations_padded = pad_sequence(explanations, batch_first=True, padding_value=explanation_vocab[PAD_TOKEN])
    return sentences_padded, target_chars_padded, explanations_padded

# Create dataset and DataLoader
dataset = SlangDataset(df, word_vocab, char_vocab, explanation_vocab)

def get_dataloader(batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader