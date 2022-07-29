import numpy as np
import scipy.io as sio

class DataUtil():
    def __init__(self, dataset, backbone, config):
        self.dataset = dataset
        self.backbone = backbone
        self.config = config    
        self.data = {}
    
    def get_unseen_data(self):
        ############### load unseen data
        if self.dataset == 'McGill':
            feature_path = '/unseen_McGill.mat'
            label_path = '/unseen_McGill_label.mat'
        elif self.dataset == 'ModelNet':
            feature_path = '/unseen_ModelNet10.mat'
            label_path = '/unseen_ModelNet10_label.mat'
        elif self.dataset == 'ScanObjectNN':
            feature_path = '/unseen_ScanObjectNN.mat'
            label_path = '/unseen_ScanObjectNN_label.mat'
        temp = sio.loadmat(self.config['dataset_path'] + self.backbone + feature_path)
        self.data['unseen_feature'] = temp['data']
        temp = sio.loadmat(self.config['dataset_path'] + self.backbone + label_path) 
        self.data['unseen_labels'] = temp['label'] 
        self.data['unseen_labels'] = np.squeeze(self.data['unseen_labels']) + self.config['seen_class']
    
    def get_seen_data(self):
        ###### load seen train data
        temp = sio.loadmat(self.config['dataset_path'] + self.backbone + '/seen_train.mat')
        self.data['seen_feature_train'] = temp['data']
        temp = sio.loadmat(self.config['dataset_path'] + self.backbone + '/seen_train_label.mat')
        seen_labels = temp['label']
        self.data['seen_labels_train'] = np.squeeze(seen_labels)

        ############# load seen test data
        temp = sio.loadmat(self.config['dataset_path'] + self.backbone + '/seen_test.mat')
        self.data['seen_feature_test'] = temp['data']
        temp = sio.loadmat(self.config['dataset_path'] + self.backbone + '/seen_test_label.mat')
        self.data['seen_labels_train_test'] = temp['label']
        self.data['seen_labels_train_test'] = np.squeeze(self.data['seen_labels_train_test'])

    def get_attribute(self):
        wordvector = sio.loadmat(self.config['dataset_path'] + 'ModelNetwordvector')
        if self.dataset == 'ModelNet':
            seen_index =np.int16([0,3,4,5,6,7,9,10,11,13,15,16,17,18,19,20,21,24,25,26,27,28,29,31,32,34,36,37,38,39])
            temp = wordvector['word']
            self.data['seen_attribute'] = temp[seen_index,:]
            unseen_index =np.int16([1,2,8,12,14,22,23,30,33,35])
            self.data['unseen_attribute'] = temp[unseen_index,:]
        elif self.dataset == 'McGill':
            seen_index = np.int16([0,3,4,5,6,7,9,10,11,13,15,16,17,18,19,20,21,24,25,26,27,28,29,31,32,34,36,37,38,39])
            temp = wordvector['word']
            self.data['seen_attribute'] = temp[seen_index,:]
            wordvector = sio.loadmat(self.config['dataset_path'] + 'McGill_w2v.mat')
            self.data['unseen_attribute'] = wordvector['word']
        elif self.dataset == 'ScanObjectNN':
            seen_index = np.int16([0,1,5,6,7,9,10,11,15,16,17,18,19,20,21,23,24,25,26,27,28,31,34,36,37,39])
            temp = wordvector['word']
            self.data['seen_attribute'] = temp[seen_index,:]
            unseen_index =np.int16([12,22,13,4,33,2,29,30,35])
            v1 = (temp[14,:]+temp[38,:])/2
            v2 =  (temp[8,:]+temp[32,:]+temp[3,:])/3
            v1 = np.expand_dims(v1, 0)
            v2 = np.expand_dims(v2, 0)
            self.data['unseen_attribute'] = temp[unseen_index,:]
            self.data['unseen_attribute'] = np.concatenate((v1, v2, self.data['unseen_attribute']), axis=0)

        self.data['attribute'] = np.append(self.data['seen_attribute'], self.data['unseen_attribute'], axis=0)


    def get_data(self):
        self.get_seen_data()
        self.get_unseen_data()
        self.get_attribute()
        return self.data

