import argparse
import glob
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from PIL import Image
from torch.autograd import Variable

class Network(nn.Module):
    def __init__(self, dropout_p=0.2):
        super(Network, self).__init__()
        self.c1 = nn.Conv2d(3, 10, kernel_size=5)
        self.c2 = nn.Conv2d(10, 20, kernel_size=3)
        self.c3 = nn.Conv2d(20, 40, kernel_size=3)
        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(3360, 20)
        self.fc2 = nn.Linear(20, 2)
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, x):
        x = F.relu(self.mp(self.c1(x)))
        x = F.relu(self.mp(self.c2(x)))
        x = F.relu(self.mp(self.c3(x)))
        x = x.view(-1, 3360)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)

class dataset(object):
    def __init__(self, path):
        # neg is 0, pos is 1
        pos_im = glob.glob(path + '/pos/*.png')
        neg_im = glob.glob(path + '/neg/*.png')
        img = [(x, 1) for x in pos_im]
        img = img + [(x, 0) for x in neg_im]
        random.shuffle(img)
        self.data = img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img = Image.open(self.data[index][0]).resize((64, 128))

        img = np.array(img).transpose((2, 0, 1))[:3]
        img = img / 255. - 0.5
        img = torch.from_numpy(img).float()
        label = self.data[index][1]
        return img, label

def train(model, loader, optimizer, criterion, epoch, device):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, (input, target) in enumerate(loader):
        input = Variable(input)
        target = Variable(target)
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        accuracy = (output.topk(1)[1].transpose(0,1) == target).float().mean()
        loss.backward()
        optimizer.step()
        losses.update(loss.item())
        accuracies.update(accuracy)
    print('Train: epoch {}\t loss {}\t accuracy {}'.format(epoch, losses.avg, accuracies.avg))

def test(model, loader, criterion, epoch, device):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, (input, target) in enumerate(loader):
        input = Variable(input)
        target = Variable(target)
        input, target = input.to(device), target.to(device)
        output = model(input)

        print(input.norm(), output.norm())

        loss = criterion(output, target)
        accuracy = (output.topk(1)[1].transpose(0,1) == target).float().mean()
        losses.update(loss.item())
        accuracies.update(accuracy)
    print('Test: epoch {}\t loss {}\t accuracy {}'.format(epoch, losses.avg, accuracies.avg))

def test_single(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, (input, target) in enumerate(loader):
        input = Variable(input)
        target = Variable(target)
        input, target = input.to(device), target.to(device)
        output = model(input)

        print(input.norm(), output.norm())

        loss = criterion(output, target)
        accuracy = (output.topk(1)[1].transpose(0,1) == target).float().mean()
        losses.update(loss.item())
        accuracies.update(accuracy)
    print('Test: epoch {}\t loss {}\t accuracy {}'.format(0, losses.avg, accuracies.avg))

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_network(model, path):
    weights = {}
    with open(path, 'rb') as f:
        weights = pickle.load(f)
    model.load_state_dict(weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_dir", help="Training directory")
    parser.add_argument("--test_dir", help="Test directory")
    parser.add_argument("-r", "--rate", help="Learning rate", type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.01, help='Momentum')
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=time.time(), help="Random seed")
    parser.add_argument("--mode", type=str, required=True, help="One of: train, test, both")
    parser.add_argument("--dropout_chance", type=float, default=0.2, help="Dropout chance")
    parser.add_argument("-o", "--output", type=str, required=False, help="Output file")
    parser.add_argument("-i", "--input", type=str, required=False, help="Input file")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Network(args.dropout_chance)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.rate, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    if args.input != None:
        load_network(model, args.input)

    if args.mode == "test":

        if args.test_dir == None:
            print("Please provide a valid test directory")
            exit(-1)

        test_loader = torch.utils.data.DataLoader(dataset(args.test_dir), batch_size=64, shuffle=False, num_workers=4)
        model.train(False)

        test(model, test_loader, criterion, 0, device)
    elif args.mode == "train":

        if args.training_dir == None:
            print("Please provide a valid training directory")
            exit(-1)

        train_loader = torch.utils.data.DataLoader(dataset(args.training_dir), batch_size=64, shuffle=True, num_workers=4)
        model.train(True)

        for epoch in range(args.epochs):
            train(model, train_loader, optimizer, criterion, epoch, device)
    elif args.mode == "both":

        if args.training_dir == None or args.test_dir == None:
            print("Please provide valid training and test directories")
            exit(-1)

        train_loader = torch.utils.data.DataLoader(dataset(args.training_dir), batch_size=64, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(dataset(args.test_dir), batch_size=64, shuffle=False, num_workers=4)

        for epoch in range(args.epochs):

            model.train(True)
            train(model, train_loader, optimizer, criterion, epoch, device)

            model.train(False)
            test(model, test_loader, criterion, epoch, device)
    else:
        print("Invalid mode. Valid options: test, train, both")
        exit(-1)

    if args.output != None:
        state = model.state_dict()
        with open(args.output, 'wb') as f:
            pickle.dump(state, f)
