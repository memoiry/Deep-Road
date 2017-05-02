import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchnet.meter as meter
import time
import os
import argparse
from models import *
from utils import progress_bar

############  Hyper Parameters   ############
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--network', type=str, default='CNN', metavar='N',
                    help='which model to use, CNN|NIN|ResNet')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                    help='which model to use, MNIST|CIFAR-10')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

###############   Model   ##################

if(args.network == 'CNN'):
    cnn = CNN()
elif(args.network == 'NIN'):
    cnn = NIN()
elif(args.network == 'ResNet18'):
    cnn = ResNet18()
elif(args.network == 'ResNet34'):
    cnn = ResNet34()
elif(args.network == 'ResNet50'):
    cnn = ResNet50()
elif(args.network == 'ResNet101'):
    cnn = ResNet101()
elif(args.network == 'ResNet152'):
    cnn = ResNet152()
elif(args.network == 'VGG16'):
    cnn = VGG('VGG16')
elif(args.network == 'GoogLeNet'):
    cnn = GoogLeNet()
elif(args.network == 'DenseNet121'):
    cnn = DenseNet121()
elif(args.network == 'ResNeXt29'):
    cnn = ResNeXt29()

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('pretrained'), 'Error: no checkpoint directory found!'
    state_dict_ = torch.load('./pretrained/cnn_{}.pkl'.format(args.network))
    cnn.load_state_dict(state_dict_)

#if not args.no_cuda:
#    cnn.cuda()
#print(cnn)


#############  DATASET   ################
print('==> Preparing data..')
if args.dataset == 'MNIST':
    if args.network == 'CNN' or args.network == 'NIN':
        transform = transforms.Compose([
            #transforms.RandomCrop(28, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Scale(32),
        #    transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    train_dataset = datasets.MNIST(root = 'data/',
                                   train = True,
                                   transform = transform,
                                   download = True)

    test_dataset = datasets.MNIST(root = 'data/',
                                  train = False,
                                  transform = transform,
                                  download = True)

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = args.batch_size,
                                               shuffle = True)

    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                               batch_size = args.batch_size,
                                               shuffle = False)
else:
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='../data/cifar/', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data/cifar/', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

################   Loss   #################
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
mtr = meter.ConfusionMeter(k=10)

################   Training   #############
def train(epoch):
    cnn.train()
    train_loss = 0
    correct = 0
    total = 0
    print "Training Architecture: {}".format(args.network)
    for i , (images,labels) in enumerate(train_loader):
        #print images
        if not args.no_cuda:
            images = images.cuda()
            labels = labels.cuda()

        images = Variable(images)
        labels = Variable(labels)

        # forward
        optimizer.zero_grad()
        outputs = cnn(images)

        loss = criterion(outputs,labels)

        # backward
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()

        progress_bar(i, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(i+1), 100.*correct/total, correct, total))

def test():
    cnn.eval()

    # training data test
    for images,labels in train_loader:
        if not args.no_cuda:
            images = images.cuda()
        images = Variable(images)

        # forward
        outputs = cnn(images)
        mtr.add(outputs.data, labels)

    trainacc = mtr.value().diagonal().sum()*1.0/len(train_dataset)
    mtr.reset()

    # testing data test
    for images,labels in test_loader:
        if not args.no_cuda:
            images = images.cuda()
        images = Variable(images)

        # forward
        outputs = cnn(images)
        mtr.add(outputs.data, labels)

    testacc = mtr.value().diagonal().sum()*1.0/len(test_dataset)
    mtr.reset()

    # logging
    print('Accuracy on training data is: %f . Accuracy on testing data is: %f. '%(trainacc, testacc) )

##################   Main   ##################
for epoch in range(args.epochs):
    train(epoch)
    test()
torch.save(cnn.state_dict(), 'cnn_{}.pkl'.format(args.network))
