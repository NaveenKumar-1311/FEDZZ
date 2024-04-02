import torch
from torchvision import utils as vutils, datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import sys
from flask import Flask
from tqdm import tqdm
from numpy.linalg import norm
import numpy as np
import pandas as pd
import random
from imutils import paths
from random import randint
import os
from PIL import Image as Image
import copy



app = Flask(__name__)
dataset_path = '.'
torch.manual_seed(42)
np.random.seed(42)
cuda3 = torch.device('cuda:3')
train_path = dataset_path+'/Train_GTSRB'
test_csv_file_path = dataset_path+'/Test.csv'

NUM_CLASSES = 43
IMG_SIZE = 150
BATCH_SIZE = 64


data_transforms = transformations = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
    
train_dataset = datasets.ImageFolder(train_path, transform=data_transforms)
num_train = len(train_dataset)
print('number of training samples :', num_train)

train_sampler, test_dataset = torch.utils.data.random_split(train_dataset, (round(0.8*len(train_dataset)), round(0.2*len(train_dataset))))
print('number of testing samples :', len(test_dataset))

def randomize_clients_data(train_size, num_clients, minimum_size):
    assert train_size >= num_clients >= 1
    assert minimum_size*num_clients <= train_size

    data_per_client = []
    max = train_size//num_clients
    for idx in range(num_clients-1):
      data_per_client.append(randint(minimum_size,max))
      max = (train_size-sum(data_per_client))//(num_clients-len(data_per_client))
    data_per_client.append(train_size-sum(data_per_client))
    return data_per_client


def create_clients(data, num_clients=10, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of images and label lists.
        args: 
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
            
    '''
    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]
    training_data_size = len(data)

    train_idx = list(range(training_data_size))
    # np.random.shuffle(train_idx)

    #shard data and place at each client
    # size = len(data)//num_clients
    # shards = [torch.utils.data.Subset(data,list(range(i, i + size))) for i in range(0, size*num_clients, size)]
    # Mimic presence of hetrogenous data size on each client
    shard_size = randomize_clients_data(training_data_size, num_clients, 3000)
    # print('data per client: ', shard_size)
    # shards = [torch.utils.data.Subset(data,list(range(i, i + shard_size[i])))  for i in range(num_clients)]
    shards = [torch.utils.data.Subset(data,train_idx[i : i + shard_size[i]])  for i in range(num_clients)]
    for i in range(len(shards)):
      print('client ' , i , ' : data size: ', len(shards[i]))

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return {client_names[i] : shards[i] for i in range(len(client_names))} 


# In[ ]:




# In[ ]:

def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tf pytorch dataloaderds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        pytorch dataloader object'''

    
    trainloader = torch.utils.data.DataLoader(data_shard, batch_size=BATCH_SIZE,
                                            shuffle=False, drop_last= True, num_workers=2)
    
    return trainloader


# In[ ]:

#process and batch the training data for each client

#process and batch the test set  
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=2, drop_last=True)


# ### Model

# In[ ]:

class CustomCNN(nn.Module):

    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 1)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 1)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 43)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(self.conv1_bn(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(self.conv2_bn(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.conv3_bn(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(self.conv4_bn(x))
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(self.fc1_bn(x))
        x = F.softmax(self.fc2(x), -1)
        return x

def scale_model_weights(client_models, weight_multiplier_list):
    '''function for scaling a models weights'''
    client_model = client_models[0].state_dict()
    for i in range(len(client_models)):
      for k in client_model.keys():
        client_models[i].state_dict()[k].float()*weight_multiplier_list[i]

    return client_models

def server_aggregate(global_model, client_models):
    """
    This function has aggregation method 'mean'
    """
    ### This will take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        #print(torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0))
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
#     for model in client_models:
#         model.load_state_dict(global_model.state_dict())
    return global_model


# In[ ]:

def test(global_model, test_loader):
    """This function test the global model on test data and returns test loss and test accuracy """
    global_model.eval()
    loss = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    test_size = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(cuda3), target.to(cuda3)
            output = global_model(data)
            test_loss += loss(output, target) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            del data
            del target

    test_loss /= test_size
    acc = correct / test_size

    return acc, test_loss


# In[ ]:



def get_probs(model, x, y):
    probs = (model(x))[:, y]
    return torch.diag(probs.data)


# SIMBA
def MSimBA(model, x, y, num_iters, epsilon):
    
    x_org = x.clone()
    n_dims = x.reshape(1, -1).size(1)
    perm = torch.randperm(n_dims)
    org_probs = model(x)
    confused_class = torch.topk(org_probs.squeeze(), 2, dim=0, largest=True, sorted=True).indices[1]
    confused_prob = org_probs[0, confused_class]
    last_prob = get_probs(model, x_org, confused_class)
    new_class = y.clone()
    i = 0
    k = 1

    while ((i < num_iters) and ((y == new_class) or (torch.abs((output)[:, y] - (output)[:, new_class]) <= 0.1))):
        
        diff = torch.zeros(n_dims).to(cuda3)
        diff[perm[i % len(perm)]] = epsilon
        perturbation = diff.reshape(x.size())

        left_prob = get_probs(model, (x - perturbation), confused_class)
        
        if left_prob > last_prob:
            x = (x - perturbation)
            last_prob = left_prob
    
        else:
            right_prob = get_probs(model, (x + perturbation).to(cuda3), confused_class)
            if right_prob > last_prob:
                x = (x + perturbation)
                last_prob = right_prob

        output = model(x)
        new_class = torch.argmax(output)

        i += 1
    return x, model(x), i

def get_random_assignment(n, c):
    arr = np.arange(n)
    np.random.shuffle(arr)
    arr %= c
    return(arr)

def createzones(n, z):
    alpha = n / z
    c = 0
    ZallocG = []
    clients = range(0,n)
    while c < 1000:
        clients2 = range(0,n)
        Z = {}
        for i in range(z):
            if len(clients2) > alpha:
                r = random.sample(clients2, int(alpha))
                if r not in Z.values():
                    Z[i] = r
                    clients2 = [k for k in clients2 if k not in r]
            else:
                Z[i] = clients2
        check = True
        keys = np.array(list(Z.keys()))
        values = np.array(list(Z.values()))
        Zalloc = []
        for i in clients:
            Zalloc.append(keys[np.where(values == i)[0][0]])
        if Zalloc not in ZallocG:
            ZallocG.append(Zalloc)
        c = c + 1
    return ZallocG

clients = create_clients(train_sampler, num_clients=9)
print('Clients created..done')
clients
clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)

learning_rate = 0.01 
model_path = "./FL_GTSRB_25C_WA_homo_3Jan2022_0.pt"

global_model = CustomCNN().to(cuda3)
global_model.load_state_dict(torch.load(model_path, map_location='cpu'))
global_model.eval()

optimizer  = torch.optim.SGD(global_model.parameters(), lr=learning_rate, momentum=0.9)
criterion  = nn.CrossEntropyLoss()

global_acc1, global_loss = test(global_model, test_loader)
print(global_acc1)

#--
model_path = "./FL_GTSRB_25C_WA_homo_3Jan2022_0.pt"

global_model1 = CustomCNN().to(cuda3)
global_model1.load_state_dict(torch.load(model_path, map_location='cpu'))
global_model1.eval()

optimizer1  = torch.optim.SGD(global_model1.parameters(), lr=learning_rate, momentum=0.9)
criterion1  = nn.CrossEntropyLoss()

#--
model_path = "./FL_GTSRB_25C_WA_homo_3Jan2022_0.pt"

global_model2 = CustomCNN().to(cuda3)
global_model2.load_state_dict(torch.load(model_path, map_location='cpu'))
global_model2.eval()

optimizer2  = torch.optim.SGD(global_model2.parameters(), lr=learning_rate, momentum=0.9)
criterion2  = nn.CrossEntropyLoss()

#--
model_path = "./FL_GTSRB_25C_WA_homo_3Jan2022_0.pt"

global_model3 = CustomCNN().to(cuda3)
global_model3.load_state_dict(torch.load(model_path, map_location='cpu'))
global_model3.eval()

optimizer3  = torch.optim.SGD(global_model3.parameters(), lr=learning_rate, momentum=0.9)
criterion3  = nn.CrossEntropyLoss()



GTAA=[]
def execution(mutation, Zalloc,global_acc1, global_model, global_model1, global_model2, global_model3):
    global_epochs = 2
    local_epochs = 1
    oddclients = []

    z1 = []
    z2 = []
    z3 = []
    c = 1
    for i in Zalloc:
        if i == 0:
            z1.append(c)
        elif i == 1:
            z2.append(c)
        else:
            z3.append(c)
        c = c + 1
    
    print ("z", z1, z2, z3)

    for g_epoch in range(global_epochs):
        global_weights1 = global_model1.parameters()
        global_weights2 = global_model2.parameters()
        global_weights3 = global_model3.parameters()

        scaled_local_weight_list = list()
        client_names= list(clients_batched.keys())
        client_names_sel = client_names
        print(client_names_sel)
        client_models = []
        z1client_models = []
        z2client_models = []
        z3client_models = []

        scaling_factor = []
        scaling_factor1 = []
        scaling_factor2 = []
        scaling_factor3 = []


        clients_data_list = []
        clients_data_list1 = []
        clients_data_list2 = []
        clients_data_list3 = []

        selclientids = []
        #loop through each client and create new local model
        for client in tqdm(client_names_sel):
            if int(client[8:]) in z1:
                print("Entering client", client, " of zone z1")
                local_model = CustomCNN().to(cuda3)
                local_model.load_state_dict(global_model2.state_dict())
                local_model = copy.deepcopy(global_model2)
                
                optimizer2  = torch.optim.SGD(local_model.parameters(), lr=learning_rate, momentum=0.9)
                loss_func  = nn.CrossEntropyLoss()
                local_model.train()
                for epoch in range(local_epochs):
                    success = 0
                    total = 0
                    client_data_size = 0
                    for batch_idx, (data, target) in enumerate(clients_batched[client]):
                        images_batch = Variable(data)
                        labels_batch = Variable(target)
                        org_prob = local_model(images_batch.to(cuda3))
                        org_class = torch.argmax(org_prob, dim=1)

                        client_data_size += len(target)
                        optimizer2.zero_grad()
                        output = local_model(images_batch.to(cuda3))
                        loss = criterion2(output, labels_batch.to(cuda3).long())
                        loss.backward()
                        optimizer2.step()
                        del images_batch
                        del labels_batch

                    z1client_models.append(local_model)

                    fc2localmodel = []
                    fc2globalmodel = []
        

                    for name, para in local_model.named_parameters():
                        if name == str('fc2.weight'):
                            fc2localmodel.append(torch.flatten(para).cpu().detach().numpy())

                    for name, para in global_model2.named_parameters():
                        if name == str('fc2.weight'):
                            fc2globalmodel.append(torch.flatten(para).cpu().detach().numpy())

                    cossim = np.dot(np.squeeze(np.asarray(fc2localmodel)),np.squeeze(np.asarray(fc2globalmodel)))/(norm(np.squeeze(np.asarray(fc2localmodel)))*norm(np.asarray(fc2globalmodel)))
                    if cossim > 0.99:
                        client_models.append(local_model)
                        selclientids.append(int(client[8:]))

                    clients_data_list1.append(client_data_size)
                    clients_data_list.append(client_data_size)

                    del local_model
            elif int(client[8:]) in z2:
                print("Entering client", client, " of zone z2")
                local_model = CustomCNN().to(cuda3)
                local_model.load_state_dict(global_model3.state_dict())
                local_model = copy.deepcopy(global_model3)
                
                optimizer3  = torch.optim.SGD(local_model.parameters(), lr=learning_rate, momentum=0.9)
                loss_func  = nn.CrossEntropyLoss()
                local_model.train()
                for epoch in range(local_epochs):
                    success = 0
                    total = 0
                    client_data_size = 0
                    for batch_idx, (data, target) in enumerate(clients_batched[client]):

                        images_batch = Variable(data)
                        labels_batch = Variable(target)
                        org_prob = local_model(images_batch.to(cuda3))
                        org_class = torch.argmax(org_prob, dim=1)

                        client_data_size += len(target)
                        optimizer3.zero_grad()
                        output = local_model(images_batch.to(cuda3))
                        loss = criterion3(output, labels_batch.to(cuda3).long())
                        loss.backward()
                        optimizer3.step()
                        del images_batch
                        del labels_batch

                    z2client_models.append(local_model)

                    fc2localmodel = []
                    fc2globalmodel = []
        

                    for name, para in local_model.named_parameters():
                        if name == str('fc2.weight'):
                            fc2localmodel.append(torch.flatten(para).cpu().detach().numpy())

                    for name, para in global_model3.named_parameters():
                        if name == str('fc2.weight'):
                            fc2globalmodel.append(torch.flatten(para).cpu().detach().numpy())

                    cossim = np.dot(np.squeeze(np.asarray(fc2localmodel)),np.squeeze(np.asarray(fc2globalmodel)))/(norm(np.squeeze(np.asarray(fc2localmodel)))*norm(np.asarray(fc2globalmodel)))
                    if cossim > 0.99:
                        client_models.append(local_model)

                    clients_data_list2.append(client_data_size)
                    clients_data_list.append(client_data_size)

                    del local_model
            else:
                print("Entering client", client, " of zone z3")
                local_model = CustomCNN().to(cuda3)
                local_model.load_state_dict(global_model1.state_dict())
                local_model = copy.deepcopy(global_model1)
                
                optimizer1  = torch.optim.SGD(local_model.parameters(), lr=learning_rate, momentum=0.9)
                loss_func  = nn.CrossEntropyLoss()
                local_model.train()
                for epoch in range(local_epochs):
                    success = 0
                    total = 0
                    client_data_size = 0
                    for batch_idx, (data, target) in enumerate(clients_batched[client]):

                        images_batch = Variable(data)
                        labels_batch = Variable(target)
                        org_prob = local_model(images_batch.to(cuda3))
                        org_class = torch.argmax(org_prob, dim=1)
                        client_data_size += len(target)
                        optimizer1.zero_grad()
                        output = local_model(images_batch.to(cuda3))
                        loss = criterion1(output, labels_batch.to(cuda3).long())
                        loss.backward()
                        optimizer1.step()
                        del images_batch
                        del labels_batch

                    z3client_models.append(local_model)

                    fc2localmodel = []
                    fc2globalmodel = []
        

                    for name, para in local_model.named_parameters():
                        if name == str('fc2.weight'):
                            fc2localmodel.append(torch.flatten(para).cpu().detach().numpy())

                    for name, para in global_model1.named_parameters():
                        if name == str('fc2.weight'):
                            fc2globalmodel.append(torch.flatten(para).cpu().detach().numpy())

                    cossim = np.dot(np.squeeze(np.asarray(fc2localmodel)),np.squeeze(np.asarray(fc2globalmodel)))/(norm(np.squeeze(np.asarray(fc2localmodel)))*norm(np.asarray(fc2globalmodel)))
                    #print(client, cossim)
                    if cossim > 0.99:
                        client_models.append(local_model)
                        selclientids.append(int(client[8:]))


                    clients_data_list3.append(client_data_size)
                    clients_data_list.append(client_data_size)

                    del local_model

        tot_data = sum(clients_data_list)

        tot_data1 = sum(clients_data_list1)
        tot_data2 = sum(clients_data_list2)
        tot_data3 = sum(clients_data_list3)

        scaling_factor = [client_data / tot_data for client_data in clients_data_list]

        scaling_factor1 = [client_data / tot_data1 for client_data in clients_data_list1]
        scaling_factor2 = [client_data / tot_data2 for client_data in clients_data_list2]
        scaling_factor3 = [client_data / tot_data3 for client_data in clients_data_list3]

        #print(client_models)
        print("Selected clients: ", selclientids)
        client_models = scale_model_weights(client_models, scaling_factor)
        global_model = server_aggregate(global_model,client_models)
        global_acc, global_loss = test(global_model, test_loader)
        print('global_epochs: {} | global_loss: {} | global_accuracy: {}'.format(g_epoch+1, global_loss, global_acc))
        #torch.save(global_model'')
        #print("Cosinvalue for epoch:"+str(g_epoch+1)+"--> "+str(oddclients))
        GTAA.append(global_acc)
        #np.save('GTAA_25client_multi13Alabelflip_mean_homo_MSimBA_28June2022.npy', GTAA)
        #np.save('GTAA_25client_cossim8A_21June2022.npy', oddclients)

        np.save('GTAA_9client_het_noattack_withrandomzones_18Jul2022.npy', GTAA)

        z1client_models = scale_model_weights(z1client_models, scaling_factor1)
        global_model1 = server_aggregate(global_model1,z1client_models)

        z2client_models = scale_model_weights(z2client_models, scaling_factor2)
        global_model2 = server_aggregate(global_model2,z2client_models)

        z3client_models = scale_model_weights(z3client_models, scaling_factor3)
        global_model3 = server_aggregate(global_model3,z3client_models)
    
    print("Federated Learning is finished... ok!")
    print("Feedback of Mutation: ", mutation, " is: ", str(global_acc - global_acc1))
    global_acc1 = global_acc
    return global_acc1, global_model1, global_model2, global_model3, global_model


# ZallocG = createzones(9, 3)
for i in range(50):
    ZallocG = get_random_assignment(9, 3)
    global_model = CustomCNN().to(cuda3)
    global_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    global_acc1, global_model1, global_model2, global_model3, global_model = execution(i, ZallocG[i], global_acc1, global_model, global_model1, global_model2, global_model3)
