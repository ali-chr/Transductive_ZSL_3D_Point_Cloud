import torch
import torch.nn as nn
import torch.nn.functional as F


mse = nn.MSELoss()
def inductive_distance_loss(visual_fea, semantic_fea, label):
    loss = mse(visual_fea ,semantic_fea[label,:].float())
    return loss


def transductive_triplet_loss(visual_fea, semantic_fea, batch_size, alpha):
    visual_fea = F.normalize(visual_fea, p=2, dim=1)
    semantic_fea = F.normalize(semantic_fea, p=2, dim=1)
    Dis = torch.cdist(visual_fea.float(), semantic_fea.float(), p=2.0)
    label_pseudo_pos = torch.argmin(Dis, dim=1)
    Dis = torch.cdist(visual_fea.float(), semantic_fea[0:30,:].float(), p=2.0)
    label_pseudo_neg = torch.argmin(Dis, dim=1) 
    loss = 0
    for kk in range(int(batch_size/2)):
        dis_pos = torch.dist(visual_fea[kk,:].float(), semantic_fea[label_pseudo_pos[kk],:].float(), p=2) 
        dis_neg = torch.dist(visual_fea[kk,:].float(), semantic_fea[label_pseudo_neg[kk],:].float(), p=2) 
        loss += (F.relu(dis_pos-dis_neg+alpha))/(batch_size/2)
    return loss


def hubness_loss_transductive(visual_fea, semantic_fea, total_class):
    ### sotmax
    Dis = torch.cdist(visual_fea.float(), semantic_fea.float(), p=2.0)
    ### adaptive weight
    Dis_soft_digit = torch.nn.functional.softmax(-Dis, dim=1) 
    Dis_soft_digit_max,_ = (torch.max(Dis_soft_digit,dim=1))
    weight = torch.mean(-torch.log(Dis_soft_digit_max))
    ## argsoftmax
    Dis_soft = torch.nn.functional.softmax(-10*Dis, dim=1) 
    # histogram 
    Hist = torch.sum(Dis_soft, dim=0)
    Hist_mean = torch.mean(Hist)
    Hist_var = torch.var(Hist)
    #### loss 
    a = Hist - Hist_mean
    a_pow = torch.pow(a, 3)
    a_pow_sum = torch.sum(a_pow)
    loss = a_pow_sum/(total_class*torch.pow(Hist_var, 1.5))
    # weight = 1
    loss = weight*loss
    return loss


def QFSL_loss_transductive(visual_fea, semantic_fea):
    ### sotmax
    Dis = torch.cdist(visual_fea.float(), semantic_fea.float(), p=2.0)
    ### adaptive weight
    Dis_soft_digit = torch.nn.functional.softmax(-Dis, dim=1) 
    Dis_soft_digit_sum = torch.sum(Dis_soft_digit[:,30:], dim=1)
    loss = torch.mean(-torch.log(Dis_soft_digit_sum))
    return loss