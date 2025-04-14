import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import re

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
    try:
        text = text.lower().strip()
        return re.findall(r"\w+|[^\w\s]", text)
    except Exception as e:
        print('uhoh')

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
    def __init__(self, dataframe, word_vocab, char_vocab, explanation_vocab):
        self.data = dataframe
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.explanation_vocab = explanation_vocab

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
        return (torch.tensor(sentence_indices, dtype=torch.long),
                torch.tensor(target_char_indices, dtype=torch.long),
                torch.tensor(explanation_indices, dtype=torch.long))

def collate_fn(batch):
    # Each batch element is a tuple: (sentence, target_char, explanation)
    sentences, target_chars, explanations = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=word_vocab[PAD_TOKEN])
    target_chars_padded = pad_sequence(target_chars, batch_first=True, padding_value=char_vocab[PAD_TOKEN])
    explanations_padded = pad_sequence(explanations, batch_first=True, padding_value=explanation_vocab[PAD_TOKEN])
    return sentences_padded, target_chars_padded, explanations_padded

# Create dataset and DataLoader
dataset = SlangDataset(df, word_vocab, char_vocab, explanation_vocab)
BATCH_SIZE = 32
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

#############################################
# 4. Model Definition
#############################################
# (Assuming that the model classes from the previous code snippet are defined in this script.)
# For completeness, include the key model classes again below.

# ---------- Word-Level Encoder ----------
class WordEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(WordEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
    def forward(self, input_seq):
        embedded = self.embedding(input_seq)  # [batch, seq_len, embed_size]
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

# ---------- Character-Level Encoder ----------
class CharEncoder(nn.Module):
    def __init__(self, char_vocab_size, char_embed_size, char_hidden_size, num_layers=1):
        super(CharEncoder, self).__init__()
        self.embedding = nn.Embedding(char_vocab_size, char_embed_size)
        self.lstm = nn.LSTM(char_embed_size, char_hidden_size, num_layers, batch_first=True)
    def forward(self, input_char_seq):
        embedded = self.embedding(input_char_seq)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden[-1]  # return last hidden state

# ---------- Attention Mechanism ----------
class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
    def forward(self, decoder_hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((decoder_hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # [batch, src_len]
        return F.softmax(attention, dim=1)

# ---------- Decoder with Attention ----------
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, encoder_hidden_dim, decoder_hidden_dim, attention, num_layers=1):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim + encoder_hidden_dim, decoder_hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(decoder_hidden_dim + encoder_hidden_dim + embed_dim, output_dim)
    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)  # [batch, 1]
        embedded = self.embedding(input)  # [batch, 1, embed_dim]
        attn_weights = self.attention(hidden[-1], encoder_outputs)  # [batch, src_len]
        attn_weights = attn_weights.unsqueeze(1)  # [batch, 1, src_len]
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch, 1, encoder_hidden_dim]
        lstm_input = torch.cat((embedded, context), dim=2)  # [batch, 1, embed_dim + encoder_hidden_dim]
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = output.squeeze(1)  # [batch, decoder_hidden_dim]
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
        prediction = self.out(torch.cat((output, context, embedded), dim=1))
        return prediction, hidden, cell

# ---------- Dual Encoder Fusion ----------
class DualEncoder(nn.Module):
    def __init__(self, word_encoder, char_encoder, fusion_output_dim):
        super(DualEncoder, self).__init__()
        self.word_encoder = word_encoder
        self.char_encoder = char_encoder
        self.fusion_linear = nn.Linear(word_encoder.lstm.hidden_size + char_encoder.lstm.hidden_size,
                                       fusion_output_dim)
    def forward(self, word_input, char_input):
        word_outputs, hidden, cell = self.word_encoder(word_input)
        char_rep = self.char_encoder(char_input)  # [batch, char_hidden_dim]
        batch_size, src_len, _ = word_outputs.size()
        char_rep_expanded = char_rep.unsqueeze(1).expand(-1, src_len, -1)
        fusion_input = torch.cat((word_outputs, char_rep_expanded), dim=2)
        fused_outputs = self.fusion_linear(fusion_input)
        return fused_outputs, hidden, cell

# ---------- Complete Sequence-to-Sequence Model ----------
class DualEncoderSeq2Seq(nn.Module):
    def __init__(self, dual_encoder, decoder, device):
        super(DualEncoderSeq2Seq, self).__init__()
        self.dual_encoder = dual_encoder
        self.decoder = decoder
        self.device = device
    def forward(self, src_words, src_chars, trg, teacher_forcing_ratio=0.5):
        batch_size = src_words.size(0)
        trg_len = trg.size(1)
        output_dim = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)
        encoder_outputs, hidden, cell = self.dual_encoder(src_words, src_chars)
        dec_hidden = hidden
        dec_cell = cell
        input_token = trg[:, 0]  # <SOS> tokens
        for t in range(1, trg_len):
            prediction, dec_hidden, dec_cell = self.decoder(input_token, dec_hidden, dec_cell, encoder_outputs)
            outputs[:, t, :] = prediction
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
        return outputs

# Set model hyperparameters using the vocabulary sizes from our dataset
WORD_VOCAB_SIZE = len(word_vocab)
CHAR_VOCAB_SIZE = len(char_vocab)
TARGET_VOCAB_SIZE = len(explanation_vocab)

WORD_EMBED_SIZE = 300
CHAR_EMBED_SIZE = 50
DECODER_EMBED_SIZE = 300

WORD_HIDDEN_SIZE = 512
CHAR_HIDDEN_SIZE = 256
FUSION_OUTPUT_DIM = 512        # typically match the encoder hidden size for attention
DECODER_HIDDEN_SIZE = 512
NUM_LAYERS = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate model components
word_encoder = WordEncoder(WORD_VOCAB_SIZE, WORD_EMBED_SIZE, WORD_HIDDEN_SIZE, NUM_LAYERS)
char_encoder = CharEncoder(CHAR_VOCAB_SIZE, CHAR_EMBED_SIZE, CHAR_HIDDEN_SIZE, NUM_LAYERS)
dual_encoder = DualEncoder(word_encoder, char_encoder, FUSION_OUTPUT_DIM)
attention = Attention(encoder_hidden_dim=FUSION_OUTPUT_DIM, decoder_hidden_dim=DECODER_HIDDEN_SIZE)
decoder = Decoder(TARGET_VOCAB_SIZE, DECODER_EMBED_SIZE, FUSION_OUTPUT_DIM, DECODER_HIDDEN_SIZE, attention, NUM_LAYERS)
model = DualEncoderSeq2Seq(dual_encoder, decoder, device).to(device)
print(model)

#############################################
# 5. Training Loop
#############################################
criterion = nn.CrossEntropyLoss(ignore_index=explanation_vocab[PAD_TOKEN])
optimizer = optim.Adam(model.parameters(), lr=0.001)
NUM_EPOCHS = 5

def train(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio=0.5):
    model.train()
    epoch_loss = 0
    for batch_idx, (src_words, src_chars, trg) in enumerate(dataloader):
        src_words = src_words.to(device)
        src_chars = src_chars.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()
        output = model(src_words, src_chars, trg, teacher_forcing_ratio)
        # output shape: [batch_size, trg_len, TARGET_VOCAB_SIZE]
        # Skip the first token (<SOS>) for loss calculation:
        output_dim = output.shape[-1]
        output = output[:, 1:].contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")
    return epoch_loss / len(dataloader)

for epoch in range(1, NUM_EPOCHS + 1):
    loss = train(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio=0.5)
    print(f"Epoch {epoch} Loss: {loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "dual_seq2seq_model.pth")
print("Model training complete and saved as 'dual_seq2seq_model.pth'")