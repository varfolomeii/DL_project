import os
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataloader import num2desc, num2code, desc2num, code2num, code_count, desc_count, myDataset
from torch.utils.data import DataLoader

data_folder = 'data/stackoverflow'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, vocab_size, output_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, output_size)

    def forward(self, input):
        return self.embedding(input)


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.2):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.W = nn.Linear(hidden_size, output_size)

    def forward(self, prev_input, prev_hidden, encoder_outputs):
        input = self.embedding(prev_input).view(1, -1, 400)
        input = self.dropout(input)

        if prev_hidden:
            output, hidden = self.lstm(input, prev_hidden)
        else:
            output, hidden = self.lstm(input)
        alpha = encoder_outputs.bmm(hidden[0].view(-1, 400, 1))
        alpha = F.softmax(alpha, -1).view(-1, 1, 400)
        t = alpha.bmm(encoder_outputs)

        h_att = self.tanh(self.W1(hidden[0].view(-1, 1, 400)) + self.W2(t))
        h_att = self.dropout(h_att)
        y = self.W(h_att)
        pred = F.log_softmax(y, dim=1)

        return pred, hidden


def train(encoder, decoder, dataloader, num_epochs=50, lr=0.01):
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=lr)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    for num_epoch in range(num_epochs):
        for j, batch in tqdm(enumerate(dataloader)):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            input_tensor, target_tensor = batch
            if torch.cuda.is_available():
                target_tensor = target_tensor.cuda()
                input_tensor = input_tensor.cuda()
            target_length = target_tensor.size(1)
            encoder_outputs = encoder(input_tensor)
            decoder_input = target_tensor[:, 0]

            loss = 0
            hidden = None
            for i in range(target_length):
                decoder_output, hidden = decoder(decoder_input, hidden, encoder_outputs)
                loss += criterion(decoder_output.squeeze(1), target_tensor[:, i])
                decoder_input = target_tensor[:, i]

            if j % 1000 == 0:
                print('Loss : {}'.format(loss))
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()


def test(encoder, decoder, dataloader):
    for j, batch in tqdm(enumerate(dataloader)):
        input_tensor, target_tensor = batch
        if torch.cuda.is_available():
            target_tensor = target_tensor.cuda()
            input_tensor = input_tensor.cuda()
        target_length = target_tensor.size(1)
        encoder_outputs = encoder(input_tensor)
        decoder_input = target_tensor[:, 0]

        hidden = None
        ans = []
        for i in range(target_length):
            decoder_output, hidden = decoder(decoder_input, hidden, encoder_outputs)
            decoder_input = decoder_output.argmax(dim=-1)
            if j == 0:
                ans.append(num2desc(decoder_input.view(1).item))

        if j == 0:
            print(' '.join(ans))

        if i % 1000 == 0:
            print('Loss : {}'.format(loss))
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()


if __name__ == '__main__':
    dataloader = DataLoader(myDataset('data/stackoverflow/python/train.txt'), batch_size=4)
    encoder = Encoder(code_count, 400).to(device=device)
    decoder = Decoder(400, desc_count).to(device=device)
    train(encoder, decoder, dataloader)
    torch.save(encoder.state_dict(), './')
    torch.save(decoder.state_dict(), './')
    encoder.eval()
    decoder.eval()
    test_dataloader = DataLoader(myDataset('data/stackoverflow/python/test.txt'), batch_size=1)
