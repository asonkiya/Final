# from dataloader import word_vocab, char_vocab, explanation_vocab, get_dataloader  # FOR small dataset
from dataloader_big import get_dataloader, word_vocab, char_vocab, explanation_vocab
from dual_seq2seq import DualEncoderSeq2Seq, CharEncoder, WordEncoder, Decoder, AttentionDecoder
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

device = 'mps' if not torch.cuda.is_available() else 'cuda'

def do_train(model, dataloader, learning_rate=0.002, teacher_forcing_ratio=0.5):

    epoch_loss = 0
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model.train()
    # Stochastic grad descent
    for batch in tqdm(dataloader):
        try:
            word_input, char_input, target = batch # example usage, word itself, explanation
            optimizer.zero_grad()
            output = model(word_input, char_input, target)
            # remove <SOS> for loss calc
            output = output[:, 1:].reshape(-1, output.shape[-1])
            target = target[:, 1:].reshape(-1)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        except Exception as e:
            print("BATCH FAILED")
            print(e)

    torch.save(model.state_dict(), "dual_seq2seq.pth")
    print("Model training complete and saved as 'dual_seq2seq.pth'")

if __name__ == "__main__":
    # dataloader = get_dataloader()
    dataloader = get_dataloader(batch_size=16)
    word_encoder = WordEncoder(vocab_size=len(word_vocab))
    char_encoder = CharEncoder(vocab_size=len(char_vocab))
    decoder = AttentionDecoder(output_size=len(explanation_vocab))
    model = DualEncoderSeq2Seq(word_encoder=word_encoder, char_encoder=char_encoder, decoder=decoder).to(device)
    do_train(model=model, dataloader=dataloader)