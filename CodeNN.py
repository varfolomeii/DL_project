import os
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataloader import num2desc, num2code, desc2num, code2num, code_count, desc_count, SIZE, myDataset
from torch.utils.data import DataLoader
from nltk.translate.meteor_score import meteor_score
import nltk
from queue import PriorityQueue

nltk.download('wordnet')

data_folder = 'data/stackoverflow'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 100


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
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.W = nn.Linear(hidden_size, output_size)

    def forward(self, prev_input, prev_hidden, encoder_outputs):
        batch_size = prev_input.shape[0]
        input = self.embedding(prev_input).view(batch_size, 1, self.hidden_size)
        input = self.dropout(input)

        if prev_hidden:
            output, hidden = self.lstm(input, prev_hidden)
        else:
            output, hidden = self.lstm(input)
        alpha = hidden[0].view(batch_size, 1, self.hidden_size).bmm(
            encoder_outputs.view(batch_size, self.hidden_size, -1))
        alpha = F.softmax(alpha, 2)
        t = torch.sum(alpha.view(batch_size, 1, -1) * encoder_outputs.view(batch_size, self.hidden_size, -1), dim=-1)
        h_att = self.tanh(self.W1(hidden[0].view(batch_size, self.hidden_size)) + self.W2(t))
        h_att = self.dropout(h_att)
        y = self.W(h_att)
        pred = F.log_softmax(y, dim=-1)
        return pred, hidden


def train(encoder, decoder, dataloader, val_dataloader, num_epochs=60, lr=0.5):
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=lr)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    losses = []
    meteors = []
    best_meteor = -1
    encoder_dict = {}
    decoder_dict = {}
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
            if j % 20 == 0:
                losses.append(loss.view(1).item())
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
        encoder.eval()
        decoder.eval()
        n = 0
        score = 0
        for val_data in tqdm(val_dataloader):
            input, target = val_data
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            a, b = predict(encoder, decoder, input, target)
            try:
                score += meteor_score(a, b)
                n += 1
            except Exception:
                pass
        score /= n
        print('METEOR: {}'.format(score))
        meteors.append(score)
        if best_meteor == -1 or score > best_meteor:
            best_meteor = score
            encoder_dict = encoder.state_dict()
            decoder_dict = decoder.state_dict()

        encoder.train()
        decoder.train()
        print('Loss: {}'.format(np.mean(np.array(epoch_loss))))

    f = plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('iter_num / 20')
    plt.ylabel('NLLLoss')
    plt.grid()
    plt.show()
    f.savefig('plot1.pdf')

    f = plt.figure()
    plt.plot(range(len(meteors)), meteors)
    plt.xlabel('epoch_num')
    plt.ylabel('Meteor_score')
    plt.grid()
    plt.show()
    f.savefig('meteor1.pdf')
    return encoder_dict, decoder_dict


def test(encoder, decoder, dataloader):
    score = 0
    n = 0
    for j, batch in tqdm(enumerate(dataloader)):
        input_tensor, target_tensor = batch
        if torch.cuda.is_available():
            target_tensor = target_tensor.cuda()
            input_tensor = input_tensor.cuda()
        a, b = predict(encoder, decoder, input_tensor, target_tensor)
        try:
            score += meteor_score(a, b)
            n += 1
        except Exception:
            pass
        if j % 200 == 0:
            print(a)
            print(b)
    score /= n
    print('METEOR: {}'.format(score))


class BeamSearchNode(object):
    def __init__(self, hidden, prev, num, logProb, length):
        self.hidden = hidden
        self.prev = prev
        self.num = num
        self.logp = logProb
        self.length = length

    def eval(self):
        return self.logp / float(self.length - 1 + 1e-6)

    def __lt__(self, other):
        return self.eval() < other.eval()


def beam_decode(decoder, decoder_input, encoder_outputs):
    width = 10
    output = []

    end_node = None
    node = BeamSearchNode(None, None, decoder_input, 0, 1)
    nodes = PriorityQueue()
    nodes.put((-node.eval(), node))
    q = 1
    while q < 2000:
        score, cur_node = nodes.get()
        decoder_input = cur_node.num
        hidden = cur_node.hidden

        if decoder_input.view(1).item() == desc2num['CODE_END'] or cur_node.length > 20:
            end_node = cur_node
            break

        decoder_output, hidden = decoder(decoder_input, hidden, encoder_outputs)

        top, indexes = torch.topk(decoder_output, width)
        for i in range(width):
            index = indexes[0][i].view(1)
            log_p = top[0][i].item()
            node = BeamSearchNode(hidden, cur_node, index, cur_node.logp + log_p, cur_node.length + 1)
            nodes.put((-node.eval(), node))
        q += width - 1

    if not end_node:
        end_node = nodes.get()[1]

    node = end_node
    while node.prev != None:
        output.append(num2desc[node.num.view(1).item()])
        node = node.prev

    output.append('CODE_START')
    output.reverse()
    return output


def predict(encoder, decoder, input, target):
    encoder_outputs = encoder(input)
    decoder_input = target[:, 0]

    target_text = []
    for i in range(target.size(1)):
        target_text.append(num2desc[target[:, i].item()])
        if target_text[-1] == 'CODE_END':
            break
    ans = beam_decode(decoder, decoder_input, encoder_outputs)

    return ' '.join(ans[1:-1]), ' '.join(target_text[1:-1])


if __name__ == '__main__':
    dataloader = DataLoader(myDataset('data/stackoverflow/python/train.txt'), batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(myDataset('data/stackoverflow/python/valid.txt'), batch_size=1)
    encoder = Encoder(code_count, SIZE).to(device=device)
    decoder = Decoder(SIZE, desc_count).to(device=device)
    e_dict, d_dict = train(encoder, decoder, dataloader, val_dataloader)
    torch.save(e_dict, './Encoder1')
    torch.save(d_dict, './Decoder1')
    encoder.load_state_dict(torch.load('./Encoder1'))
    decoder.load_state_dict(torch.load('./Decoder1'))
    encoder.eval()
    decoder.eval()
    test_dataloader = DataLoader(myDataset('data/stackoverflow/python/test.txt'), batch_size=1)
    test(encoder, decoder, test_dataloader)
