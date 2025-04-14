import torch
import torch.nn as nn
import torch.nn.functional as F

class WordEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size=300, hidden_size=512):
        super(WordEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        embd = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embd)
        return outputs, hidden, cell


class CharEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size=300, hidden_size=512):
        super(CharEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        embd = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embd)
        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, hidden_size=512, output_size=3000):
        super(Decoder, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, context, target):
        import pdb
        pdb.set_trace()
        context = context.unsqueeze(1).repeat(1, target.size(1), 1)
        combined = torch.cat((context, target), dim=2)
        attn_weights = torch.softmax(self.attn(combined), dim=1)
        context = torch.sum(attn_weights * combined, dim=1)
        output, _ = self.lstm(context)
        output = self.fc_out(output)
        return output

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim=512, decoder_hidden_dim=512):
        super(Attention, self).__init__()
        self.attn = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Parameter(torch.rand(decoder_hidden_dim))

    def forward(self, decoder_hidden, encoder_outputs):
        batch_size, seq_len, _ = encoder_outputs.shape
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((decoder_hidden, encoder_outputs), dim=2)))
        energy = torch.sum(energy * self.v, dim=2)
        return torch.softmax(energy, dim=1)

# Output dim will be set in train.py, just a placeholder here
class AttentionDecoder(nn.Module):
    def __init__(self, embedding_dim=300, decoder_hidden_dim=512, encoder_hidden_dim=512, output_size=3000):
        super(AttentionDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, embedding_dim)
        # worth swapping this out for multi-headed attn later?
        self.attention = Attention(encoder_hidden_dim, decoder_hidden_dim)
        self.rnn = nn.LSTM(embedding_dim + encoder_hidden_dim, decoder_hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(embedding_dim + encoder_hidden_dim + decoder_hidden_dim, output_size)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        embedded = self.embedding(input_token).unsqueeze(1)
        attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        combined = torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1)
        prediction = self.fc_out(combined)
        return prediction, hidden, cell

class DualEncoderSeq2Seq(nn.Module):
    def __init__(self, word_encoder, char_encoder, decoder):
        super(DualEncoderSeq2Seq, self).__init__()
        self.word_encoder = word_encoder
        self.char_encoder = char_encoder
        self.decoder = decoder

    def forward(self, context_input, word_input, target_output, teacher_forcing_ratio=0.0):
        batch_size, target_len = target_output.shape
        output_dim = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, target_len, output_dim).to(context_input.device)

        context_outputs, word_hidden, _ = self.word_encoder(context_input)
        _, char_hidden, _ = self.char_encoder(word_input)

        combined_hidden = (word_hidden + char_hidden) / 2
        decoder_hidden = combined_hidden
        decoder_cell = torch.zeros_like(combined_hidden)

        input_token = target_output[:, 0]

        for t in range(1, target_len):
            output, decoder_hidden, decoder_cell = self.decoder(
                input_token, decoder_hidden, decoder_cell, context_outputs
            )
            outputs[:, t] = output
            top1 = output.argmax(1)
            input_token = target_output[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs