import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import csv
from numpy import linalg
import sys
from torch.utils.data  import DataLoader,IterableDataset
csv.field_size_limit(sys.maxsize)


        
def read_text_iterator(path):
    data_dict=path
    for item in zip(data_dict['text_tensors'],data_dict['audio_tensors'],data_dict['video_tensors'],data_dict['label'],data_dict['speaker']):
       if len(item[0])>0:
        yield item
    
            # if len(row[2])>0:
            #   self.text_ten.append(row[2])
            #   self.aucs_ten.append(row[3])
            #   self.vid_ten.append(row[4])
            #   self.label.append(row[5])


# iterable dataset
class m2h2PDataset(IterableDataset):
    def __init__(self, full_num_lines, iterator):
        super(m2h2PDataset, self).__init__()
        self.iterator = iterator
        self.full_num_lines = full_num_lines
        self.num_lines = full_num_lines
        self.current_pos = None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_pos == self.num_lines - 1:
            raise StopIteration
        

        item = next(self.iterator)
       
        if len(item[1][0].shape)>1:
          item[1][0]=item[1][0].numpy()
          item[1][0]=linalg.norm(item[1][0],ord=2,axis=0)
          item[1][0]=torch.from_numpy(item[1][0])
        vid_tensor=torch.empty(len(item[0]),item[0][0].shape[0])
        text_tensor=torch.empty(len(item[1]),item[1][0].shape[0])
        aucs_tensor=torch.empty(len(item[2]),item[2][0].shape[0])
        #print(text_tensor.shape,aucs_tensor.shape,vid_tensor.shape)
        label=torch.FloatTensor(item[3])
        qmask=torch.FloatTensor([[1,0] if x==1 else [0,1] for x in item[4]])
        umask=torch.FloatTensor([1]*len(item[3]))
        # print(len(item[0]),len(item[1]),len(item[2]))
        # print(item[0][0].shape,item[1][0].shape,item[2][0].shape)
        for i in range(vid_tensor.shape[0]):
           vid_tensor[i]=item[0][i]
           text_tensor[i]=item[1][i]
           aucs_tensor[i]=item[2][i]


        
        if self.current_pos == None:
            self.current_pos = 0
        else:
            self.current_pos += 1
        
        return text_tensor,aucs_tensor,vid_tensor,label,umask,qmask
    
    def __len__(self):
        return self.num_lines
    
    def pos(self):
        return self.current_pos


# create a function to take input filepaths and create iterators
def load_dataset(root_dir):
   
    data_dict = root_dir

    num_lines = 0
    #data_dict=pickle.load(open(src_path, 'rb'), encoding='latin1')
    num_lines = len(data_dict['label'])
   

    

    src_iterator = read_text_iterator(data_dict)

    return m2h2PDataset(num_lines,src_iterator)
    # src_path = root_dir

    # num_lines = 0
    # data_dict=pickle.load(open(src_path, 'rb'), encoding='latin1')
    # num_lines = len(data_dict['label'])

    # print(f'Number of lines: {num_lines}')

    # src_iterator = read_text_iterator(src_path)

    # return m2h2PDataset(num_lines,src_iterator)

