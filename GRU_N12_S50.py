#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy               as np
import pandas              as pd
import h5py                as h5py
import matplotlib.pyplot   as plt
import torch
import torch.nn            as nn
import torch.nn.functional as F
import pickle

import os.path
#from   os              import path
from   IPython.display import clear_output
from   numpy.random    import seed
from   sklearn.utils   import shuffle

np.set_printoptions(precision = 5)

devCPU = torch.device("cpu")
dev    = devCPU

#from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from torch.utils.data import DataLoader, Dataset

plt.rcParams["figure.figsize"] = (6,4)
plt.rcParams["figure.dpi"]     = 200


# In[2]:


def scale_sequence(x,scale):
    return scale*(1.-2.*x)

def truncate_normalize_free_energy(y):
    norm    = np.max(np.abs(y))
    print ("F-norm = ",norm)
    #ynorm   = y / norm
    ynorm = y
    norm = 1.
    return ynorm,norm

def preprocess(x,y,sequenceScale):
    x       = scale_sequence(x,sequenceScale)
    y, norm = truncate_normalize_free_energy(y)
    return x,y,norm

def iterate_batch(x,y,batchSize):
    for i in range(0, x.shape[0], batchSize):
        yield (x[i:i + batchSize], y[i:i + batchSize])

def init_weights(m):
    if isinstance(m, nn.Linear):
        n     =  m.in_features
        width = .5 * np.min ( ( np.sqrt ( 20./ n ) , 1.) )
        m.weight.data.uniform_( -width, width ) 
        # selecting the initial weigths from uniform distributions


# In[6]:


file = '../../../Hdf_files_Row_as_Sequence/Sca_Ext_Abs_N12.hdf5'
N='/N12/'
opt='Sca/'
direc='Long/'

Input = np.linspace(1.13,3.53,81)

col='output1'
Sca_Output_Col_2 = pd.read_hdf(file, N+opt+direc+col)

col='input'
Sca_Sequence = pd.read_hdf(file ,N+opt+direc+col)

opt='Sca/'
direc='Trans/'
col='output1'
Sca_Output_Col_3 = pd.read_hdf(file, N+opt+direc+col)

opt='Ext/'
col='output1'
direc='Long/'
Sca_Output_Col_4 = pd.read_hdf(file, N+opt+direc+col)

opt='Ext/'
col='output1'
direc='Trans/'
Sca_Output_Col_5 = pd.read_hdf(file, N+opt+direc+col)

opt='Abs/'
col='output1'
direc='Long/'
Sca_Output_Col_6 = pd.read_hdf(file, N+opt+direc+col)

opt='Abs/'
col='output1'
direc='Trans/'
Sca_Output_Col_7 = pd.read_hdf(file, N+opt+direc+col)

sequenceScale = 0.7
X1f = Sca_Sequence.to_numpy()
Y1f_1 = Sca_Output_Col_2.to_numpy()
Y1f_2 = Sca_Output_Col_3.to_numpy()
Y1f_3 = Sca_Output_Col_4.to_numpy()
Y1f_4 = Sca_Output_Col_5.to_numpy()
Y1f_5 = Sca_Output_Col_6.to_numpy()
Y1f_6 = Sca_Output_Col_7.to_numpy()

Y1f_N10 = np.concatenate((Y1f_1, Y1f_2, Y1f_3, Y1f_4, Y1f_5), axis=1)

X1f = X1f[:300,:]

print(X1f.shape)
print(Y1f_N10.shape)

X1_N10, Y1_N10, Y1norm_N10 = preprocess(X1f, Y1f_N10, sequenceScale)

shuffleIdx= shuffle(np.arange(X1_N10.shape[0]))
X1_N10    = X1_N10[shuffleIdx]
Y1_N10    = Y1_N10[shuffleIdx]


INDEX = 50

X1_TRAIN_N10 = X1_N10[:INDEX][:]
Y1_TRAIN_N10 = Y1_N10[:INDEX][:]

X1_Train = torch.from_numpy(X1_TRAIN_N10).float()
Y1_Train = torch.from_numpy(Y1_TRAIN_N10).float()

print(X1_Train.shape, Y1_Train.shape)

X1_TEST_N10 = X1_N10[INDEX:][:]
Y1_TEST_N10 = Y1_N10[INDEX:][:]

X1_Test_10 = torch.from_numpy(X1_TEST_N10).float()
Y1_Test_10 = torch.from_numpy(Y1_TEST_N10).float()
print(X1_Test_10.shape, Y1_Test_10.shape)


# In[7]:


class XY_Data(Dataset):
    def __init__(self, X, Y):
        #data loading
        self.input   = X
        self.output  = Y
        self.n_samples = X.shape[0]
        
    def __getitem__(self, index):
        # dataset[0]
        return self.input[index], self.output[index]
        
    def __len__(self):
        # len(dataset)
        return self.n_samples
    
TrainDataset = XY_Data(X1_Train, Y1_Train)


# In[8]:


batch_size = 12

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=TrainDataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)


# In[9]:


# Hyper-parameters

hidden_size     = 128
num_layers      = 2

num_classes     = Y1_Train.shape[1]
num_epochs      = 15000
learning_rate   = 0.1
l2              = 1e-5
input_size      = 1
sequence_length = X1_Train.shape[1]
device          = dev


# In[12]:


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        ## x-> (batchsize, seq, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias=False, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes, bias=False)
         
    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)
        
        out, h_n = self.gru(x, h0)
        ## out -> (batchsize, seq_length =1, hidden_size)
        ## h_n -> (num_layer, N, hidden_size)
        
        out = out[:, -1, :]
        ## out (batchsize, hidden_size)
        
        out = self.fc(out)
        
        return out


# In[11]:


model = RNN(input_size, hidden_size, num_layers, num_classes)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = l2)

# Train the model
n_total_steps = len(train_loader)
print(len(train_loader))

mseHistory = list() # loss history

xTrain = X1_Train 
yTrain = Y1_Train 


# In[9]:


for epoch in range(num_epochs):
    
    
    with torch.no_grad():
            if ( epoch % 10 == 0 ):
                
                xTrain   = xTrain.reshape(-1, sequence_length, input_size).to(device)
                yPred    = model(xTrain)
                M        = yTrain.shape[0]*yTrain.shape[1]
                mseTrain = (yPred - yTrain).pow(2).sum()/M
                
                
                xTest     = X1_Test_10
                yTest     = Y1_Test_10
                xTest     = xTest.reshape(-1, sequence_length, input_size).to(device)
                yPred     = model(xTest)
                M         = yTest.shape[0]*yTest.shape[1]
                mseTest_10 = (yPred -  yTest).pow(2).sum()/M # mean square
                
                
                mseRecord= np.array ( (epoch, float(mseTrain), float(mseTest_10) ))
                
                print ( "rmse/kT ~", mseRecord[0], np.sqrt(mseRecord[1:] ))
                mseHistory.append(mseRecord)
                
                if ( epoch% 500 == 0 ):
                    
                    print(xTest.shape)
                    print(yTest.shape)
                    print(yPred.shape)
                
                    np_yPred = yPred.cpu().detach().numpy()
                    yPred_DF =pd.DataFrame(np_yPred)
                
                    np_yTest = yTest.cpu().detach().numpy()
                    yTest_DF =pd.DataFrame(np_yTest)
                    
                    xTest = xTest.squeeze(2)
                    
                    np_xTest = xTest.cpu().detach().numpy()
                    xTest_DF =pd.DataFrame(np_xTest)
                    
                    fname_1 = './GRU_Individual_SeqProp.hdf5'
                    path_1 = '/N12/sample50/'
                
                    yPred_DF.to_hdf(fname_1, path_1+'Pred'+str(epoch),mode='a')
                
                    yTest_DF.to_hdf(fname_1, path_1+'Target'+str(epoch),mode='a')
                    
                    xTest_DF.to_hdf(fname_1, path_1+'Sequence'+str(epoch),mode='a')
                    
                
    for i, (images, labels) in enumerate(train_loader):  
        
        images = images.reshape(-1, sequence_length, input_size)
    
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# In[10]:


DF_msehist =  pd.DataFrame(mseHistory)

fname_2 = './GRU_Individual_mse.hdf5'
path_2 = '/N12/sample50/msehist/'
DF_msehist.to_hdf(fname_2, path_2, mode='a')


# In[11]:


norm = 1.

case_N8 = DF_msehist


case_N8.columns = ['epoch', 'train_mse', 'T_8',]

plt.figure(1)
plt.xscale("log")
plt.ylim(0.0, 25000) 
plt.ylabel("Test RMSE", fontsize = 12)
plt.xlabel("Epochs", fontsize = 12)

plt.plot( case_N8['epoch'], np.sqrt(case_N8['train_mse'])*norm, 'r.--',  ms = 4, lw=1., label="train")

plt.plot( case_N8['epoch'], np.sqrt(case_N8['T_8'])*norm, 'b.--',  ms = 4, lw=1., label="test")

plt.grid()
plt.legend(loc='best',fontsize=12)
#plt.savefig("./RMSE_case3_250_Included.png",dpi=300, bbox_inches='tight' )


# In[4]:


fname_1 = './GRU_Individual_SeqProp.hdf5'
path_1 = '/N12/sample50/'

Pred_5000 = pd.read_hdf(fname_1, path_1+'Pred14500')
Test_5000 = pd.read_hdf(fname_1, path_1+'Target14500')

Pred_5000 = Pred_5000.to_numpy()
Test_5000 = Test_5000.to_numpy()

print(Pred_5000.shape)
print(Test_5000.shape)


# In[5]:


plt.rcParams["figure.figsize"] = (6,4)
plt.rcParams["figure.dpi"]     = 200

plt.figure(1)
plt.ylabel("Predict")
plt.xlabel("Target")


plt.plot(Test_5000[:,:], Pred_5000[:,:], 'r.', ms = 0.75,)

plt.plot(Test_5000[:,:], Test_5000[:,:], 'g.', ms = 0.75,)


plt.legend(loc='best')
plt.grid()
#plt.savefig("./Results_combineAB.png",dpi=300, bbox_inches='tight' )


# In[ ]:




