#%% Importing Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader 
from torchvision import transforms,datasets,models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import warnings
from IPython.display import display
from sklearn.model_selection import train_test_split
from torchvision.utils import make_grid
from sklearn.metrics import f1_score,accuracy_score, precision_score,recall_score

warnings.filterwarnings('ignore')

torch.manual_seed(42)
#%% Cuda Setup
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#%% Setting the image labels
pre_path =  r"C:\Users\efeta\OneDrive\Desktop"
#pre_path =  "/auto/data2/etarhan"
img_names = pd.read_csv(pre_path+"/MPOX_2/MPOXDBASE.csv")['Data'].values
img_labels = pd.read_csv(pre_path+"/MPOX_2/MPOXDBASE.csv")['Disease'].values
#%% Train_test_split
X_train, X_test, y_train, y_test = train_test_split(img_names,img_labels,test_size = 0.2, shuffle = True)
#%% [Chickenpox Measles Monkeypox Normal]  = [0 0 0 0]
y_train = pd.get_dummies(y_train)
y_train = y_train.to_numpy()
y_train = [torch.FloatTensor(i).to(device) for i in y_train]

y_test = pd.get_dummies(y_test)
y_test = y_test.to_numpy()
y_test = [torch.FloatTensor(i).to(device) for i in y_test]
    #%% Increasing the number of training samples
train_data = []
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomAdjustSharpness(2),
    transforms.RandomPosterize(2),
    transforms.RandomInvert(),
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

for i in range(2):
    for j in range(len(X_train)):
        img =Image.open(pre_path + X_train[j])
        img = train_transforms(img)
        img = img.to(device)
        train_data.append((img,y_train[j]))
        print(f"Training data {i+1}.{j} has done")
#%%
test_data= []
test_label = []
test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    ])

for i in range(len(X_test)):
    img =Image.open(pre_path + X_test[i])
    img = test_transforms(img)
    img = img.to(device)
    test_data.append((img,y_test[i]))
    print(f"Test data {i} has done")

test_label = y_test
#%%train_data
train_loader = DataLoader(train_data,batch_size = 16,shuffle = True)
test_loader = DataLoader(test_data,batch_size = 16, shuffle = True)
#%%
class VGGPOX(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16pret = models.vgg16('IMAGENET1K_V1')
        vgg16pret.classifier = nn.Linear(25088,4096)
        self.vgg16 = vgg16pret
        self.dropout1 = nn.Dropout(0.20)
        self.fc2 = nn.Linear(4096,1000)
        self.dropout2 = nn.Dropout(0.30)
        self.fc3 = nn.Linear(1000,4)
        
    def forward(self,x):
        x = F.relu(self.vgg16(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.softmax(x,dim=1)
#%%
class RESNETPOX(nn.Module):
    def __init__(self):
        super().__init__()
        resnet50pret = models.resnet50('IMAGENET1K_V1')
        resnet50pret.fc = nn.Linear(2048,1000)
        self.resnet50 = resnet50pret
        self.dropout1 = nn.Dropout(0.20)
        self.fc2 = nn.Linear(1000,512)
        self.dropout2 = nn.Dropout(0.30)
        self.fc3 = nn.Linear(512,4)
        
    def forward(self,x):
        x = F.relu(self.resnet50(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.softmax(x,dim=1)
#%%
class EFFICIENTPOX(nn.Module):
    def __init__(self):
        super().__init__()
        efficpret = models.efficientnet_b0('IMAGENET1K_V1')
        efficpret.classifier = nn.Linear(1280,640)
        self.efficientv2 = efficpret
        self.dropout1 = nn.Dropout(0.20)
        self.fc2 = nn.Linear(640,160)
        self.dropout2 = nn.Dropout(0.30)
        self.fc3 = nn.Linear(160,4)
        
    def forward(self,x):
        x = F.relu(self.efficientv2(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.softmax(x,dim=1)
#%%
class MOBILEPOX(nn.Module):
    def __init__(self):
        super().__init__()
        mobilpret = models.mobilenet_v3_large('IMAGENET1K_V1')
        mobilpret.classifier = nn.Linear(960,1280)
        self.mobilenetv3 = mobilpret
        self.dropout1 = nn.Dropout(0.20)
        self.fc2 = nn.Linear(1280,1000)
        self.dropout2 = nn.Dropout(0.30)
        self.fc3 = nn.Linear(1000,4)
        
    def forward(self,x):
        x = F.relu(self.mobilenetv3(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.softmax(x,dim=1)
#%%
class DENSEPOX(nn.Module):
    def __init__(self):
        super().__init__()
        densepret = models.densenet121('IMAGENET1K_V1')
        densepret.classifier = nn.Linear(1024,512)
        self.densenet121 = densepret
        self.dropout1 = nn.Dropout(0.20)
        self.fc2 = nn.Linear(512,256)
        self.dropout2 = nn.Dropout(0.30)
        self.fc3 = nn.Linear(256,4)
        
    def forward(self,x):
        x = F.relu(self.densenet121(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.softmax(x,dim=1)
#%%
class INCEPOX(nn.Module):
    def __init__(self):
        super().__init__()
        inceppret = models.inception_v3('IMAGENET1K_V1')
        inceppret.fc = nn.Linear(2048,512)
        self.inception = inceppret
        self.dropout1 = nn.Dropout(0.20)
        self.fc2 = nn.Linear(512,128)
        self.dropout2 = nn.Dropout(0.30)
        self.fc3 = nn.Linear(128,4)
        
    def forward(self,x):
        x = self.inception(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.softmax(x,dim=1)
#%%
torch.manual_seed(42)
criterion = nn.CrossEntropyLoss()
model = VGGPOX().to(device)
optimizer = torch.optim.SGD(model.parameters(),lr =0.001)
#%%
import time 
start_time = time.time()
epochs = 50

train_losses = []
test_losses = []
train_correct = []
test_correct = []
for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    
    for b,(X_train,y_train) in enumerate(train_loader):
        y_pred = model(X_train)
        loss = criterion(y_pred,y_train)
        
        t_encode = torch.argmax(y_train,dim = 1)
                        
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == t_encode).sum()
        trn_corr += batch_corr
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss)
        train_correct.append(trn_corr)
    print("---------------------------------------------")
    print(f'epoch:{(i+1):2} || loss: {loss.item():10.8f}  || accuracy: {trn_corr.item()*100/(16*(b+1)):7.3f}% || ')
    
    
    with torch.no_grad():
        predicted = np.array([])
        predix = np.array([])
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test)
            loss = criterion(y_val,y_test)  
            predicted = np.hstack((predicted ,torch.argmax(y_val.data, 1).to("cpu").numpy()))
            predix = np.hstack((predix,torch.argmax(y_test,1).to("cpu").numpy()))
        test_losses.append(loss)
        print("--------------")
        print("Test Metrics :")
        print("--------------")
        print(f"Accuracy: %{accuracy_score(predix,predicted)*100}")
        print(f"Recall: %{recall_score(predix,predicted,average = 'macro')*100}")
        print(f"Precision: %{precision_score(predix,predicted,average = 'macro')*100}")
        print(f"F1-Score: %{f1_score(predix,predicted,average = 'macro')*100}")
        print("---------------------------------------------")
        print()
        

print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed

#%%

torch.save(model.state_dict(), 'MPOXMODEL.pt')
