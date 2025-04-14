import torch
from dataloader import word_vocab, char_vocab, explanation_vocab
from dual_seq2seq import DualEncoderSeq2Seq, WordEncoder, CharEncoder, AttentionDecoder

device = 'mps' if not torch.cuda.is_available() else 'cuda'

# Should build classes around tokenizers where this is a method
class Tokenizer:
    def __init__(self, vocab):
        # if vocab.get('<unk>')
        self.stoi = vocab
        self.itos = {v: k for k, v in vocab.items()}

def do_inference(model, context_sentence, slang_word, word_tokenizer, char_tokenizer, target_tokenizer, max_len=30):
    model.eval()

    # Tokenize inputs
    context_indices = [word_tokenizer.stoi.get(w, word_tokenizer.stoi['<UNK>']) for w in context_sentence.split()]
    word_indices = [char_tokenizer.stoi.get(c, char_tokenizer.stoi['<UNK>']) for c in slang_word]

    context_tensor = torch.tensor([context_indices], dtype=torch.long).to(device)
    word_tensor = torch.tensor([word_indices], dtype=torch.long).to(device)

    with torch.no_grad():
        # Encode inputs
        context_outputs, word_hidden, _ = model.word_encoder(context_tensor)
        _, char_hidden, _ = model.char_encoder(word_tensor)

        combined_hidden = (word_hidden + char_hidden) / 2
        decoder_hidden = combined_hidden
        decoder_cell = torch.zeros_like(combined_hidden)

        input_token = torch.tensor([target_tokenizer.stoi['<SOS>']], dtype=torch.long).to(device)
        outputs = []

        for _ in range(max_len):
            output, decoder_hidden, decoder_cell = model.decoder(
                input_token, decoder_hidden, decoder_cell, context_outputs
            )
            top1 = output.argmax(1).item()
            if top1 == target_tokenizer.stoi['<EOS>']:
                break
            outputs.append(target_tokenizer.itos[top1])
            input_token = torch.tensor([top1], dtype=torch.long).to(device)

    return ' '.join(outputs)


if __name__ == "__main__":


    word_encoder = WordEncoder(vocab_size=len(word_vocab))
    char_encoder = CharEncoder(vocab_size=len(char_vocab))
    decoder = AttentionDecoder(output_size=len(explanation_vocab))
    model = DualEncoderSeq2Seq(word_encoder, char_encoder, decoder).to(device)
    model.load_state_dict(torch.load('dual_seq2seq.pth', weights_only=True))
    sents = [
        'Her eyebrows on fleek, look at her go!',
        "That phrase is so cheugy, no one says that anymore.",
        "that test was so ass i think i failed",
        'TFW you finish a big project and can finally relax'
    ]
    words = [
        'on fleek',
        'cheugy',
        'ass',
        'TFW'
    ]
    word_tok = Tokenizer(word_vocab)
    char_tok = Tokenizer(char_vocab)
    exp_tok = Tokenizer(explanation_vocab)
    for i in range(len(sents)):
        sent = sents[i]
        word = words[i]
        res = do_inference(model, sent, word, word_tok, char_tok, exp_tok)
        print(res)