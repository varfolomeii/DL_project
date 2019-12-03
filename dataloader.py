import re
import collections
from torch.utils.data import Dataset
import torch

data_folder = 'data/stackoverflow'

PAD = 1
UNK = 2
START = 3
END = 4

desc2num = {"UNK": UNK, "CODE_START": START, "CODE_END": END}
code2num = {"UNK": UNK, "CODE_START": START, "CODE_END": END}
num2desc = {PAD: "UNK", UNK: "UNK", START: "CODE_START", END: "CODE_END"}
num2code = {UNK: "UNK", START: "CODE_START", END: "CODE_END"}


def skipComment(line):
    if '#' in line:
        return line[:line.find('#')]
    else:
        return line


def prepareCode(code):
    code = code.replace('\t', '  ').replace('\\n', '\n')
    lines = code.split('\n')
    lines.append('')
    lines[0] = skipComment(lines[0])
    gap = len(lines[0]) - len(lines[0].lstrip())
    current_gap = [0]
    for i in range(1, len(lines)):
        lines[i] = skipComment(lines[i][gap:])
        gap = len(lines[i]) - len(lines[i].lstrip())
        if gap > current_gap[-1]:
            lines[i] = '{ ' + lines[i]
            current_gap.append(gap)
        if gap < current_gap[-1]:
            while gap < current_gap[-1]:
                lines[i] = '} ' + lines[i]
                current_gap.pop()

    return ' '.join(lines)


def tokenizeDescription(desc):
    desc = desc.strip()
    return re.findall(r"[\w]+|[^\s\w]", desc)


def tokenizeCode(code):
    code = prepareCode(code)
    return [re.sub(r'\s+', ' ', x.strip()) for x in code.split(' ')]


def buildVocab(filename):
    desc_tokens = collections.Counter()
    code_tokens = collections.Counter()

    for line in open(filename, "r"):
        if len(line.strip().split('\t')) < 4:
            continue
        desc, code = line.strip().split('\t')[2], line.strip().split('\t')[3]
        code_tokens.update(tokenizeCode(code))
        desc_tokens.update(tokenizeDescription(desc))

    code_count = 5
    desc_count = 5

    for tok in code_tokens:
        if code_tokens[tok] > 3:
            code2num[tok] = code_count
            num2code[code_count] = tok
            code_count += 1
        else:
            code2num[tok] = UNK

    for tok in desc_tokens:
        if desc_tokens[tok] > 3:
            desc2num[tok] = desc_count
            num2desc[desc_count] = tok
            desc_count += 1
        else:
            desc2num[tok] = UNK

    return code_count, desc_count


code_count, desc_count = buildVocab('data/stackoverflow/python/train.txt')


def tokenizeData(filename):
    dataset = []
    for line in open(filename, 'r'):

        if len(line.strip().split('\t')) < 4:
            continue
        desc, code = line.strip().split('\t')[2], line.strip().split('\t')[3]
        code_tokens = (tokenizeCode(code))
        desc_tokens = (tokenizeDescription(desc))

        code_num = []
        desc_num = []

        for tok in code_tokens:
            if tok not in code2num:
                code2num[tok] = UNK
            code_num.append(code2num[tok])

        l = len(code_num)
        for i in range(400 - l):
            code_num.append(1)

        desc_num.append(desc2num["CODE_START"])
        for tok in desc_tokens:
            if tok not in desc2num:
                desc2num[tok] = UNK
            desc_num.append(desc2num[tok])

        desc_num.append(desc2num["CODE_END"])

        l = len(desc_num)
        for i in range(400 - l):
            desc_num.append(1)

        dataset.append((code_num, desc_num))
    return dataset


class myDataset(Dataset):
    def __init__(self, filename):
        super(Dataset, self).__init__()
        self.dataset = tokenizeData(filename)

    def __getitem__(self, index):
        return torch.tensor(self.dataset[index][0]), torch.tensor(self.dataset[index][1])

    def __len__(self):
        return len(self.dataset)