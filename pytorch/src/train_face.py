import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
import torch.utils.data as util_data
import lr_schedule
from data_list import ImageList
from torch.autograd import Variable

from data.data_loader import load_data_multi
from os.path import join
from tqdm import tqdm

import pdb

optim_dict = {"SGD": optim.SGD}

def image_classification_test(loader, model):
    
    correct = 0.0
    total_num = 0.0

    for data in tqdm(loader):
        inputs = data[0]
        labels = data[1]
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
        # pdb.set_trace()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total_num += len(labels)
       
    accuracy = correct / total_num
    return accuracy

def transfer_classification(config, src):

    datadir = '/home/zhmiao/datasets/MultiPIE/cropped'
    class_num = 6
    batch = 128

    tgt_list = ['mp05', 'mp08', 'mp09', 'mp13', 'mp14']
    tgt_list.remove(src) 

    # data loaders
    dset_loaders = {}

    dset_loaders["source"] = {}
    dset_loaders["source"]["train"] = load_data_multi(src, 'train', batch=batch, 
                                                    rootdir=join(datadir, src), num_channels=3, 
                                                    image_size=224, download=True, 
                                                    kwargs={'num_workers': 1, 'pin_memory': True})
    dset_loaders["source"]["test"] = load_data_multi(src, 'test', batch=batch, 
                                                    rootdir=join(datadir, src), num_channels=3, 
                                                    image_size=224, download=True, 
                                                    kwargs={'num_workers': 1, 'pin_memory': True})

    dset_loaders["target"] = {}
    dset_loaders["target"]["train"] = load_data_multi(tgt_list, 'train', batch=batch, 
                                                    rootdir=datadir, num_channels=3, 
                                                    image_size=224, download=True, 
                                                    kwargs={'num_workers': 1, 'pin_memory': True})
    dset_loaders["target"]["test"] = {}
    for dom in ['mp05', 'mp08', 'mp09', 'mp13', 'mp14', 'mp19']:
        dset_loaders["target"]["test"][dom] = load_data_multi(dom, 'test', batch=batch, 
                                                        rootdir=join(datadir, dom), num_channels=3, 
                                                        image_size=224, download=True, 
                                                        kwargs={'num_workers': 1, 'pin_memory': True})

    ## set loss
    class_criterion = nn.CrossEntropyLoss()
    loss_config = config["loss"]
    transfer_criterion = loss.loss_dict[loss_config["name"]]
    if "params" not in loss_config:
        loss_config["params"] = {}

    # pdb.set_trace()

    ## set base network
    net_config = config["network"]
    base_network = network.network_dict[net_config["name"]]()
    classifier_layer = nn.Linear(base_network.output_num(), class_num)

    # full_net = network.network_dict['ResNet18_full']().cuda()
    # # pdb.set_trace()
    # full_net.load('./ResNet18_net_mp05.pth')
    # full_net = full_net.cuda()
    # full_net.eval()
    # # for dom in ['mp05', 'mp08', 'mp09', 'mp13', 'mp14', 'mp19']:
    # for dom in ['mp05']:
    #     print('\nTesting on %s\n'%dom)
    #     print('Acc: %s\n'%image_classification_test(dset_loaders["target"]["test"][dom], full_net))

    # exit()

    print('Loading ./ResNet18_net_mp05.pth')
    weights = torch.load('./ResNet18_net_mp05.pth')

    feature_weights = {k: weights[k] if k in weights else base_network.state_dict()[k] 
                       for k in base_network.state_dict()}

    base_network.load_state_dict(feature_weights)

    # pdb.set_trace()

    classifier_layer.weight.data = weights['fc.weight'].data
    classifier_layer.bias.data = weights['fc.bias'].data

    # for param in base_network.parameters():
    #     param.requires_grad = False

    ## initialization
    # classifier_layer.weight.data.normal_(0, 0.01)
    # classifier_layer.bias.data.fill_(0.0)

    classifier_layer = classifier_layer.cuda()
    base_network = base_network.cuda()

    ## collect parameters
    parameter_list = [{"params":classifier_layer.parameters(), "lr":0.001, "momentum":0.9, "weight_decay":0.0005, "nesterov":True},
                      {"params":base_network.parameters(), "lr":0.001, "momentum":0.9, "weight_decay":0.0005, "nesterov":True}]

    ## add additional network for some methods
    if loss_config["name"] == "JAN":
        softmax_layer = nn.Softmax(dim=1)
        softmax_layer = softmax_layer.cuda()
           
    # pdb.set_trace()
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list)
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    ## train   
    len_train_source = len(dset_loaders["source"]["train"]) - 1
    len_train_target = len(dset_loaders["target"]["train"]) - 1
    # transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    for i in range(config["num_iterations"]):
        ## test in the train
        if i % config["test_interval"] == 0:
            base_network.eval()
            classifier_layer.eval()
            for dom in ['mp05', 'mp08', 'mp09', 'mp13', 'mp14', 'mp19']:
                print('\nTesting on %s\n'%dom)
                print('Acc: %s\n'%image_classification_test(dset_loaders["target"]["test"][dom], 
                                                nn.Sequential(base_network, classifier_layer)))

        # loss_test = nn.BCELoss()
        ## train one iter
        base_network.train()
        classifier_layer.train()
        optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"]["train"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"]["train"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = Variable(inputs_source).cuda(), Variable(inputs_target).cuda(), Variable(labels_source).cuda()
           
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        features = base_network(inputs)

        outputs = classifier_layer(features)

        # pdb.set_trace()

        classifier_loss = class_criterion(outputs.narrow(0, 0, int(inputs.size(0)/2)), labels_source)
        ## switch between different transfer loss
        if loss_config["name"] == "DAN":
            transfer_loss = transfer_criterion(features.narrow(0, 0, int(features.size(0)/2)),
                                               features.narrow(0, int(features.size(0)/2), int(features.size(0)/2)),
                                               **loss_config["params"])
        elif loss_config["name"] == "RTN":
            ## RTN is still under developing
            transfer_loss = 0
        elif loss_config["name"] == "JAN":
            softmax_out = softmax_layer(outputs)
            transfer_loss = transfer_criterion([features.narrow(0, 0, int(features.size(0)/2)),
                                                softmax_out.narrow(0, 0, int(softmax_out.size(0)/2))],
                                               [features.narrow(0, int(features.size(0)/2), int(features.size(0)/2)),
                                                softmax_out.narrow(0, int(softmax_out.size(0)/2), int(softmax_out.size(0)/2))],
                                               **loss_config["params"])

        total_loss = 0.1 * transfer_loss + classifier_loss
        total_loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Step %s\n'%i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--source', type=str, nargs='?', default='mp05', help="source data")
    parser.add_argument('--target', type=str, nargs='?', default='multi', help="target data")
    parser.add_argument('--loss_name', type=str, nargs='?', default='JAN', help="loss name")
    parser.add_argument('--tradeoff', type=float, nargs='?', default=1.0, help="tradeoff")
    args = parser.parse_args()

    config = {}
    config["num_iterations"] = 20000
    config["test_interval"] = 500
    config["prep"] = [{"name":"source", "type":"image", "test_10crop":False, "resize_size":256, "crop_size":224}, 
                      {"name":"target", "type":"image", "test_10crop":False, "resize_size":256, "crop_size":224}]
    config["loss"] = {"name":args.loss_name, "trade_off":args.tradeoff }
    config["data"] = [{"name":"source", "type":"image"}, 
                      {"name":"target", "type":"image"}]
    config["network"] = {"name":"ResNet18"}
    config["optimizer"] = {"type":"SGD", 
                           "optim_params":{"lr":0.01, "momentum":0.9, "weight_decay":0.0005, "nesterov":True}, 
                           "lr_type":"inv", 
                           "lr_param":{"init_lr":0.0003, "gamma":0.0003, "power":0.75} }
    print(config["loss"])
    transfer_classification(config, src=args.source)
