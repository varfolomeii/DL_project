import os
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataloader import num2desc, num2code, desc2num, code2num, code_count, desc_count, SIZE, myDataset
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
    def __init__(self, hidden_size, output_size, dropout_p=0.5):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.W = nn.Linear(hidden_size, output_size)

    def forward(self, prev_input, prev_hidden, encoder_outputs):
        input = self.embedding(prev_input).view(1, -1, self.hidden_size)
        input = self.dropout(input)

        if prev_hidden:
            output, hidden = self.lstm(input, prev_hidden)
        else:
            output, hidden = self.lstm(input)
        alpha = encoder_outputs.bmm(hidden[0].view(-1, self.hidden_size, 1))
        alpha = F.softmax(alpha, 1).view(1, 1, -1)
        t = alpha.bmm(encoder_outputs).view(-1, self.hidden_size)

        h_att = self.tanh(self.W1(hidden[0].view(-1, self.hidden_size)) + self.W2(t))
        h_att = self.dropout(h_att)
        y = self.W(h_att)
        pred = F.log_softmax(y, dim=-1)
        return pred, hidden


def train(encoder, decoder, dataloader, num_epochs=20, lr=0.5):
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=lr)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    losses = []

    for num_epoch in range(num_epochs):
        print('Epoch {}/{}'.format(num_epoch, num_epochs))
        epoch_loss = []
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
            for i in range(1, target_length):
                decoder_output, hidden = decoder(decoder_input, hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[:, i])
                decoder_input = target_tensor[:, i]

            epoch_loss.append(loss.view(1).item())
            if j % 1000 == 0:
                losses.append(loss.view(1).item())
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
        print('Loss: {}'.format(np.mean(np.array(epoch_loss))))

    f = plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('iter_num / 1000')
    plt.ylabel('NLLLoss')
    plt.grid()
    plt.show()
    f.savefig('plot.pdf')


def test(encoder, decoder, dataloader):
    for j, batch in tqdm(enumerate(dataloader)):
        input_tensor, target_tensor = batch
        if torch.cuda.is_available():
            target_tensor = target_tensor.cuda()
            input_tensor = input_tensor.cuda()
        encoder_outputs = encoder(input_tensor)
        decoder_input = target_tensor[:, 0]

        input = []
        hidden = None
        for i in range(input_tensor.size(1)):
            input.append(num2code[input_tensor[:, i].item()])
        ans = ['CODE_START']
        while len(ans) < 30 and ans[-1] != 'CODE_END':
            decoder_output, hidden = decoder(decoder_input, hidden, encoder_outputs)
            decoder_input = decoder_output.argmax(dim=-1)
            ans.append(num2desc[decoder_input.view(1).item()])

        if j % 500 == 3:
            print(' '.join(input))
            print(' '.join(ans))


if __name__ == '__main__':
    dataloader = DataLoader(myDataset('data/stackoverflow/python/train.txt'), batch_size=1)
    encoder = Encoder(code_count, SIZE).to(device=device)
    decoder = Decoder(SIZE, desc_count).to(device=device)
    train(encoder, decoder, dataloader)
    torch.save(encoder.state_dict(), './Encoder')
    torch.save(decoder.state_dict(), './Decoder')
    #encoder.load_state_dict(torch.load('./Encoder'))
    #decoder.load_state_dict(torch.load('./Decoder'))
    encoder.eval()
    decoder.eval()
    test_dataloader = DataLoader(myDataset('data/stackoverflow/python/test.txt'), batch_size=1)
    test(encoder, decoder, test_dataloader)
