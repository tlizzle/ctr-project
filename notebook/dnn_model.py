'''
How to use: 
Put this script and zip files downloaded from kaggle into a folder, and run the script.
'''
import gzip
import os
import shutil
import csv
import pandas as pd
import math
import datetime
import torch
import torch.nn.functional as F
import numpy as np

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# unzip files
for file in ['train', 'test']:
    with gzip.open(f"{file}.gz", "rb") as f_in:
        with open(f"{file}.csv", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


'''
Descriptions:
Based on data fileds description, most of the variables are categorical variable, 
and only variables which all categories in the test dataset also exist in the training dataset can be used in the analysis,
So only 11 variables were chosen as independent variables.

which are :C1, banner_pos, site_category, app_category, device_type, device_conn_type, C15, C16, C18, C20, hour

Field 'hour' contains date and time, because time period in the test dataset dosen't overlap with train dataset, 
so date won't be useful in prediction. But weekday might be useful.
I treated time (still named it hour) as a continus variable, the others as categorical variable, including weekday.
Categorical variables were processed by embedding layers, then I concatenate all variables together and send it into a linear layer
'''

# Hyper Parameters
EPOCH = 5
BATCH_SIZE = 500000
LR = 0.05              # learning rate


# Load training dataset
variables_name_list = ['C1','banner_pos','site_category','app_category','device_type','device_conn_type','C15','C16','C18','C20','weekday','hour']
usecols = ['C1','banner_pos','site_category','app_category','device_type','device_conn_type','C15','C16','C18','C20','hour']
var_dtype_dict = {var_name:str for var_name in variables_name_list}
var_dtype_dict['click'] = np.int
train_data = pd.read_csv("train.csv", dtype=var_dtype_dict, usecols = usecols + ['click'])

# Seperate dependent from independent variables, remove date digits from 'hour' because I don't believe it useful
X_variables_dict = {i:train_data[i] for i in train_data if i in variables_name_list}
X_variables_dict['weekday'] = [datetime.datetime.strptime(i, '%y%m%d%H').weekday() for i in train_data['hour']]
X_variables_dict['hour'] = [float(i[-2:]) for i in train_data['hour']]
Y_variable = train_data['click']

# Build categorical variables to one-hot encoding mapping dictionaries
idx_to_variables_list = {}
variables_to_idx_dict = {}

print('\nbuild categorical variables to one-hot encoding mapping dictionaries')

for var_name in X_variables_dict:
    unique_var_list = list(set(X_variables_dict[var_name]))
    idx_to_variables_list[var_name] = unique_var_list.copy()
    variables_to_idx_dict[var_name] = {var:idx for idx, var in enumerate(unique_var_list)}.copy()
    print(f"variable name': {var_name}  categories count: {len(idx_to_variables_list[var_name])}")
print('')


# Build a dictionary to record input and output dimensions of every variables
variables_dim = {var_name:{'input':len(idx_to_variables_list[var_name]), 'output':1} if var_name == 'hour' else {'input':len(idx_to_variables_list[var_name]), 'output':math.ceil(len(idx_to_variables_list[var_name])/10)} for var_name in variables_name_list}

# convert raw training data into processable dataset
data = [torch.FloatTensor(X_variables_dict[var_name]) if var_name == 'hour' else torch.LongTensor([variables_to_idx_dict[var_name][i] for i in X_variables_dict[var_name]]) for var_name in variables_name_list] + [torch.FloatTensor(Y_variable)]
tensor_data = torch.utils.data.TensorDataset(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11] ,data[12])
train_loader = torch.utils.data.DataLoader(dataset=tensor_data, batch_size=BATCH_SIZE, shuffle=True)

    
del train_data, X_variables_dict

# DNN
class predictor(torch.nn.Module):
    def __init__(self):
        super(predictor, self).__init__()

        self.C1_embed = torch.nn.Embedding(variables_dim['C1']['input'], math.ceil(variables_dim['C1']['output']))
        self.banner_pos_embed = torch.nn.Embedding(variables_dim['banner_pos']['input'], math.ceil(variables_dim['banner_pos']['output']))
        self.site_category_embed = torch.nn.Embedding(variables_dim['site_category']['input'], math.ceil(variables_dim['site_category']['output']))
        self.app_category_embed = torch.nn.Embedding(variables_dim['app_category']['input'], math.ceil(variables_dim['app_category']['output']))
        self.device_type_embed = torch.nn.Embedding(variables_dim['device_type']['input'], math.ceil(variables_dim['device_type']['output']))
        self.device_conn_type_embed = torch.nn.Embedding(variables_dim['device_conn_type']['input'], math.ceil(variables_dim['device_conn_type']['output']))
        self.C15_embed = torch.nn.Embedding(variables_dim['C15']['input'], math.ceil(variables_dim['C15']['output']))
        self.C16_embed = torch.nn.Embedding(variables_dim['C16']['input'], math.ceil(variables_dim['C16']['output']))
        self.C18_embed = torch.nn.Embedding(variables_dim['C18']['input'], math.ceil(variables_dim['C18']['output']))
        self.C20_embed = torch.nn.Embedding(variables_dim['C20']['input'], math.ceil(variables_dim['C20']['output']))
        self.weekday_embed = torch.nn.Embedding(variables_dim['weekday']['input'], math.ceil(variables_dim['weekday']['output']))

        self.hidden1 = torch.nn.Linear(sum([variables_dim[var_name]['output'] for var_name in variables_name_list]), 10)
        self.hidden2 = torch.nn.Linear(10, 5)
        self.hidden3 = torch.nn.Linear(5, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_input):

        C1 = self.C1_embed(x_input[0])
        banner_pos = self.banner_pos_embed(x_input[1])
        site_category = self.site_category_embed(x_input[2])
        app_category = self.app_category_embed(x_input[3])
        device_type = self.device_type_embed(x_input[4])
        device_conn_type = self.device_conn_type_embed(x_input[5])
        C15 = self.C15_embed(x_input[6])
        C16 = self.C16_embed(x_input[7])
        C18 = self.C18_embed(x_input[8])
        C20 = self.C20_embed(x_input[9])
        weekday = self.weekday_embed(x_input[10])
        hour = x_input[11].unsqueeze(1)

        X = torch.cat((C1,banner_pos,site_category,app_category,device_type,device_conn_type,C15,C16,C18,C20,weekday,hour), dim = 1)
        X = F.relu(self.hidden1(X))
        X = F.relu(self.hidden2(X))
        X = self.hidden3(X)
        X = self.sigmoid(X)

        return X
        
Net = predictor()
print(Net,'\n')

optimizer = torch.optim.Adam(Net.parameters(), lr = LR)
loss_func = torch.nn.BCELoss()

# Training process
print('training start...')
for epoch in range(EPOCH):
    for step, batch_data in enumerate(train_loader):
        prediction = Net(batch_data)
        loss = loss_func(prediction, batch_data[-1].unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(f"Epoch: {epoch}  | step: {step}  | training_loss: {round(float(loss.data),4)}")

del data, tensor_data, train_loader


# Load test dataset
test_data = pd.read_csv("test.csv", dtype=str, usecols = usecols + ['id'])
X_variables_dict = {i:test_data[i] for i in test_data if i in variables_name_list}
X_variables_dict['weekday'] = [datetime.datetime.strptime(i, '%y%m%d%H').weekday() for i in test_data['hour']]
X_variables_dict['hour'] = [float(i[-2:]) for i in test_data['hour']]

data = [torch.FloatTensor(X_variables_dict[var_name]) if var_name == 'hour' else torch.LongTensor([variables_to_idx_dict[var_name][i] for i in X_variables_dict[var_name]]) for var_name in variables_name_list]
tensor_data = torch.utils.data.TensorDataset(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11])

# Prediction output
prediction = Net(tensor_data.tensors).data.numpy()
output = np.concatenate((np.expand_dims(np.array(test_data['id']),1), prediction),1)

with open("summit_form.csv",'w',newline='') as f:
    w = csv.writer(f)
    w.writerows([['id','click']] + list(output))

