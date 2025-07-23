# we will load a dataset jisse hum train karenge

import torch
import torchvision.transforms as transforms #img ko tensor mein convert karne ke liye
import torchvision.datasets as datasets # MNIST dataset ko load karne ke liye

def loaders(batch_size=64):
    transform=transforms.ToTensor() # image ko tensor mein convert karne ke liye

    #now we will load the MNIST dataset
    train_set=datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    # this one is for testing the model after training 
    test_set=datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    # random data return hoga
    trainloader=torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True
    )
    # testloader will not shuffle the data
    # kyunki humein test karna hai ki model sahi kaam kar raha
    testloader=torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False
    )
    return trainloader, testloader
