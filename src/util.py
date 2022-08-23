import torch
import numpy as np
from scipy.spatial.distance import cosine
from src.loss import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_baseline_hubness(model, data, unseen_class):
    ##### hubness calculation
    model.eval()
    with torch.no_grad():
        batch_x_semantic = torch.from_numpy(data['unseen_attribute']).to(device)    
        ### sotmax
        visual_fea = torch.from_numpy(data['unseen_feature']).to(device)
        visual_fea_proj = model(visual_fea.float())
        Dis = torch.cdist(visual_fea_proj.float(), batch_x_semantic.float(), p=2.0)
        Dis_soft = torch.nn.functional.softmax(-10*Dis, dim=1)
        ### histogram
        Hist = torch.sum(Dis_soft, dim=0)
        Hist_mean = torch.mean(Hist)
        Hist_var = torch.var(Hist)
        #### loss
        a = Hist - Hist_mean
        a_pow = torch.pow(a, 3)
        a_pow_sum = torch.sum(a_pow)
        hubness = a_pow_sum/(unseen_class*torch.pow(Hist_var, 1.5))
    return hubness.item()

def calculate_ours_hubness(model, data, unseen_class):
    ## hubness
    model.eval()
    with torch.no_grad():
        batch_x_semantic = torch.from_numpy(data['unseen_attribute']).to(device)    
        batch_x_semantic_proj = model(batch_x_semantic.float())
        ### sotmax
        visual_fea = torch.from_numpy(data['unseen_feature']).to(device)
        Dis = torch.cdist(visual_fea.float(), batch_x_semantic_proj.float(), p=2.0)
        Dis_soft = torch.nn.functional.softmax(-10*Dis, dim=1)
        ### histogram
        Hist = torch.sum(Dis_soft, dim=0)
        Hist_mean = torch.mean(Hist)
        Hist_var = torch.var(Hist)
        #### hubness
        a = Hist - Hist_mean
        a_pow = torch.pow(a, 3)
        a_pow_sum = torch.sum(a_pow)
        hubness = a_pow_sum/(unseen_class*torch.pow(Hist_var, 1.5))
    return hubness.item()

def train_per_epoch_ours_inductive(model, optimizer, step_batch_size, arr, batch_size, data):
    model.train()
    for i in range(0,step_batch_size):
        temp = arr[i*batch_size:(i+1)*batch_size]
        batch_x_visual = data['seen_feature_train'][temp,:]
        # label
        batch_y = data['seen_labels_train'][temp]
        batch_y = batch_y.astype(int)
        batch_y = torch.from_numpy(batch_y)
        batch_y = batch_y.to(device).type(torch.long)
        # visual
        batch_x_visual = torch.from_numpy(batch_x_visual)
        batch_x_visual = batch_x_visual.to(device).float()
        #semantic
        batch_x_semantic = torch.from_numpy(data['seen_attribute']).to(device).float()
        optimizer.zero_grad()
        #### forward
        batch_x_semantic_proj = model(batch_x_semantic.float())
        ### loss calculation
        Loss_p = inductive_distance_loss(batch_x_visual, batch_x_semantic_proj, batch_y)
        Loss = Loss_p
        # backpropagate
        Loss.backward(retain_graph=True)
        optimizer.step()

def train_per_epoch_baseline_inductive(model, optimizer, step_batch_size, arr, batch_size, data):
    model.train()
    for i in range(0,step_batch_size):
        temp = arr[i*batch_size:(i+1)*batch_size]
        batch_x_visual = data['seen_feature_train'][temp,:]
        # label
        batch_y = data['seen_labels_train'][temp]
        batch_y = batch_y.astype(int)
        batch_y = torch.from_numpy(batch_y)
        batch_y = batch_y.to(device).type(torch.long)
        # visual
        batch_x_visual = torch.from_numpy(batch_x_visual)
        batch_x_visual = batch_x_visual.to(device).float()
        #semantic
        batch_x_semantic = torch.from_numpy(data['seen_attribute']).to(device).float()
        optimizer.zero_grad()
        #### forward
        batch_x_visual_proj = model(batch_x_visual.float())
        ### loss calculation
        Loss_p = inductive_distance_loss(batch_x_visual_proj,batch_x_semantic,batch_y)
        Loss = Loss_p 
        # backpropagate
        Loss.backward(retain_graph=True)
        optimizer.step()

def calculate_accuracy_ours(model, data, config):
    ## evaluation
    model.eval()
    with torch.no_grad():
        per = 0
        class_acc = {label:0 for label in np.unique(data['unseen_labels'])}
        batch_x_semantic = torch.from_numpy(data['unseen_attribute']).to(device)
        batch_x_semantic_proj =  model(batch_x_semantic.float())
        for i in range(len(data['unseen_labels'])):
            label = data['unseen_labels'][i]
            batch_x_visual =  np.expand_dims(data['unseen_feature'][i,:], 0)
            batch_x_visual = batch_x_visual.astype(float)
            Distance = np.zeros(config['unseen_class'])
            for k1 in range(config['unseen_class']):
                Distance[k1] = cosine(batch_x_visual,batch_x_semantic_proj.cpu().detach().numpy()[k1,:])
            h = np.argmin(Distance) + config['seen_class']
            if h==label:
                per = per + 1
                class_acc[label] = class_acc[label] + 1
        for label in class_acc:
            class_acc[label] = (class_acc[label]/(data['unseen_labels'] == label).sum())*100
        zsl_acc = (per/len(data['unseen_labels']))*100
        # print(class_acc)

        per = 0
        batch_x_semantic = torch.from_numpy(data['attribute']).to(device)
        batch_x_semantic_proj =   model(batch_x_semantic.float())
        for i in range(len(data['seen_labels_train_test'])):
            label = data['seen_labels_train_test'][i]
            batch_x_visual =  np.expand_dims(data['seen_feature_test'][i,:], 0)
            batch_x_visual = batch_x_visual.astype(float)
            Distance = np.zeros(config['total_class'])
            for k1 in range(config['total_class']):
                Distance[k1] = cosine(batch_x_visual,batch_x_semantic_proj.cpu().detach().numpy()[k1,:])
            h = np.argmin(Distance)
            if h==label:
                per = per + 1
        seen = (per/len(data['seen_labels_train_test']))*100

        per = 0
        batch_x_semantic = torch.from_numpy(data['attribute']).to(device)
        batch_x_semantic_proj =   model(batch_x_semantic.float())
        for i in range(len(data['unseen_labels'])):
            label = data['unseen_labels'][i]
            batch_x_visual =  np.expand_dims(data['unseen_feature'][i,:], 0)
            batch_x_visual = batch_x_visual.astype(float)
            Distance = np.zeros(config['total_class'])
            for k1 in range(config['total_class']):
                Distance[k1] = cosine(batch_x_visual,batch_x_semantic_proj.cpu().detach().numpy()[k1,:])
            h = np.argmin(Distance)
            if h==label:
                per = per + 1
        unseen = (per/len(data['unseen_labels']))*100
        hm = (2*seen*unseen)/(seen+unseen)

        return {'zsl_acc' : zsl_acc, 'gzsl_seen': seen, 'gzsl_unseen': unseen, 'gzsl_hm': hm}

def calculate_accuracy_baseline(model, data, config):
    model.eval()
    with torch.no_grad():
        per = 0
        class_acc = {label:0 for label in np.unique(data['unseen_labels'])}
        batch_x_semantic = torch.from_numpy(data['unseen_attribute']).to(device)
        for i in range(len(data['unseen_labels'])):
            label = data['unseen_labels'][i]
            batch_x_visual =  np.expand_dims(data['unseen_feature'][i,:], 0)
            batch_x_visual = batch_x_visual.astype(float)
            batch_x_visual = torch.from_numpy(batch_x_visual).to(device)
            batch_x_visual_proj = model(batch_x_visual.float())
            Distance = np.zeros(config['unseen_class'])
            for k1 in range(config['unseen_class']):
                Distance[k1] = cosine(batch_x_visual_proj.cpu().detach().numpy(),batch_x_semantic.cpu().detach().numpy()[k1,:])
            h = np.argmin(Distance) + config['seen_class']
            if h==label:
                per = per + 1
                class_acc[label] = class_acc[label] + 1
        for label in class_acc:
            class_acc[label] = (class_acc[label]/(data['unseen_labels'] == label).sum())*100
        zsl_acc = (per/len(data['unseen_labels']))*100
        # print(class_acc)

        per = 0
        batch_x_semantic = torch.from_numpy(data['attribute']).to(device)
        for i in range(len(data['seen_labels_train_test'])):
            label = data['seen_labels_train_test'][i]
            batch_x_visual =  np.expand_dims(data['seen_feature_test'][i,:], 0)
            batch_x_visual = batch_x_visual.astype(float)
            batch_x_visual = torch.from_numpy(batch_x_visual).to(device)
            batch_x_visual_proj = model(batch_x_visual.float())
            Distance = np.zeros(config['total_class'])
            for k1 in range(config['total_class']):
                Distance[k1] = cosine(batch_x_visual_proj.cpu().detach().numpy(),batch_x_semantic.cpu().detach().numpy()[k1,:])
            h = np.argmin(Distance)
            if h==label:
                per = per + 1
        seen = (per/len(data['seen_labels_train_test']))*100

        per = 0
        batch_x_semantic = torch.from_numpy(data['attribute']).to(device)
        for i in range(len(data['unseen_labels'])):
            label = data['unseen_labels'][i]
            batch_x_visual =  np.expand_dims(data['unseen_feature'][i,:], 0)
            batch_x_visual = batch_x_visual.astype(float)
            batch_x_visual = torch.from_numpy(batch_x_visual).to(device)
            batch_x_visual_proj = model(batch_x_visual.float())
            Distance = np.zeros(config['total_class'])
            for k1 in range(config['total_class']):
                Distance[k1] = cosine(batch_x_visual_proj.cpu().detach().numpy(), batch_x_semantic.cpu().detach().numpy()[k1,:])
            h = np.argmin(Distance)
            if h==label:
                per = per + 1
        unseen = (per/len(config['unseen_labels']))*100
        hm = (2*seen*unseen)/(seen+unseen)

        return {'zsl_acc' : zsl_acc, 'gzsl_seen': seen, 'gzsl_unseen': unseen, 'gzsl_hm': hm}

def train_per_epoch_baseline_transductive(model, optimizer, step_batch_size, step_batch_size_unseen, arr, arr_unseen, batch_size, data, config):
    model.train()
    for i in range(0, step_batch_size):
        #semantic
        batch_x_semantic = torch.from_numpy(data['attribute']).to(device).float()
        optimizer.zero_grad()
        #### forward
        batch_x_semantic_proj = model(batch_x_semantic.float())
        #### seen data
        temp = arr[i*int(batch_size/2):(i+1)*int(batch_size/2)]
        batch_x_visual = data['seen_feature_train'][temp,:]
        # label
        batch_y = data['seen_labels_train'][temp]
        batch_y = batch_y.astype(int)
        batch_y = torch.from_numpy(batch_y)
        batch_y_seen = batch_y.to(device).type(torch.long)
        # visual
        batch_x_visual = torch.from_numpy(batch_x_visual)
        batch_x_visual_seen = batch_x_visual.to(device).float()
        ### inductive loss calculation
        Loss_p = inductive_distance_loss(batch_x_visual_seen,batch_x_semantic_proj,batch_y_seen)
        Loss_ind = Loss_p
        #### unseen data
        if i % step_batch_size_unseen ==0:
            np.random.shuffle(arr_unseen)
            i1 = 0
        i1 = i1 + 1
        temp = arr_unseen[i1*int(batch_size/2):(i1+1)*int(batch_size/2)]
        batch_x_visual = data['unseen_feature'][temp,:]
        # visual
        batch_x_visual = torch.from_numpy(batch_x_visual)
        batch_x_visual_unseen = batch_x_visual.to(device)
        loss_tran = transductive_triplet_loss(batch_x_visual_unseen,batch_x_semantic_proj,batch_size,alpha=1.1) 
        Loss_tran = 1.0*loss_tran
        #### final loss
        Loss = Loss_ind + config['alpha_triplet']*Loss_tran
        # backpropagate
        Loss.backward(retain_graph=True)
        optimizer.step()

def train_per_epoch_ours_transductive(model, optimizer, step_batch_size, step_batch_size_unseen, arr, arr_unseen, batch_size, data, config):
    model.train()
    for i in range(0, step_batch_size):
        #semantic
        batch_x_semantic = torch.from_numpy(data['attribute']).to(device).float()
        optimizer.zero_grad()
        #### forward
        batch_x_semantic_proj = model(batch_x_semantic.float())
        #### seen data
        temp = arr[i*int(batch_size/2):(i+1)*int(batch_size/2)]
        batch_x_visual = data['seen_feature_train'][temp,:]
        # label
        batch_y = data['seen_labels_train'][temp]
        batch_y = batch_y.astype(int)
        batch_y = torch.from_numpy(batch_y)
        batch_y_seen = batch_y.to(device).type(torch.long)
        # visual
        batch_x_visual = torch.from_numpy(batch_x_visual)
        batch_x_visual_seen = batch_x_visual.to(device).float()
        ### inductive loss calculation
        loss_ind = inductive_distance_loss(batch_x_visual_seen,batch_x_semantic_proj,batch_y_seen)
        #### unlabel data pseudo labeling
        if i % step_batch_size_unseen ==0:
            np.random.shuffle(arr_unseen)
            i1 = 0
        i1 = i1 + 1
        temp = arr_unseen[i1*int(batch_size/2):(i1+1)*int(batch_size/2)]
        batch_x_visual = data['unlabel_feature'][temp,:]
        # visual
        batch_x_visual = torch.from_numpy(batch_x_visual)
        batch_x_visual_unlabel = batch_x_visual.to(device)
        #### transductive loss
        loss_triplet = transductive_triplet_loss(batch_x_visual_unlabel, batch_x_semantic_proj, alpha=1.1)
        loss_hub_trans = hubness_loss_transductive(batch_x_visual_unlabel, batch_x_semantic_proj, config['total_class'])
        loss_GFSL = QFSL_loss_transductive(batch_x_visual_unlabel,batch_x_semantic_proj)
        #### final loss
        Loss = loss_ind + config['alpha_triplet']*loss_triplet + config['alpha_hubness']*loss_hub_trans + config['alpha_unbiased']*loss_GFSL
        # backpropagate
        Loss.backward(retain_graph=True)
        optimizer.step()
