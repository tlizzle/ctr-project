import math
from sklearn import metrics
import torch
from torch.utils.data.dataset import Dataset
from deepctr_torch.layers.interaction import InteractingLayer
from typing import List, Union
import datetime
import numpy as np
import pandas as pd
from src.config import Config
from torch.utils.tensorboard import SummaryWriter

class O3Dataset(Dataset):
    def __init__(self, inputs, variables, labels=None):
        self.variables = variables
        self.inputs = torch.stack(list(inputs.values()), dim=1)
        self.labels=labels

    def __getitem__(self, idx):
        if not isinstance(self.labels, type(None)):
            x = self.inputs[idx]
            return {'inputs': dict(zip(self.variables, x)), 'labels': self.labels[idx]}
        else:
            return {'inputs': dict(zip(self.variables, x))}

    def __len__(self):
        return len(self.inputs)

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, validation_loss, previous_validation_loss):
        if validation_loss > previous_validation_loss:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

def data_pre_processing(df: pd.DataFrame, catagory_variables: List[str], mapping_dict:dict = None):
    df['hour'] = df['hour'].astype(str) 
    parsed_datetime = np.array(list(map(lambda datetime: [datetime[:2], datetime[2:4], datetime[4:6], datetime[6:8]], df['hour'])))
    df['year'] = parsed_datetime[:,0]
    df['month'] = parsed_datetime[:,1]
    df['date'] = parsed_datetime[:,2]
    df['hour'] = parsed_datetime[:,3]

    catagory_variables += ['year', 'month', 'date', 'hour']


    if not mapping_dict:
        mapping_dict = {}

        for cat in catagory_variables:
            keys = np.concatenate((np.unique(df[cat]), np.array(["other"])) , axis=0)

            value = [i for i, _ in enumerate(keys)]
            mapping_dict[cat]  = dict(zip(keys, value))


    for cat in catagory_variables:
        mapped_index = list(map(lambda value: 
                            mapping_dict[cat][str(value)]
                                if str(value) in mapping_dict[cat] 
                                    else mapping_dict[cat]['other']  , df[cat]))
        df[cat] = mapped_index
    return df, mapping_dict


def training(model, train_loader, val_loader, experient, lr:float = 0.001, epoch:int= 10, early_stop:bool = False):
    early_stopping = EarlyStopping(tolerance=2, min_delta=10)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=1e-5)
    writer = SummaryWriter(f'run/{experient}')
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.85, verbose=True)

    validationEpoch_loss = []
    for ep in range(epoch):
        step_loss = []
        y_true = []
        y_pred = []

        model.train()
        # for iteration, batch_data in enumerate(train_loader.batch_data()):
        for iteration, batch_data in enumerate(train_loader):
            prediction = model(batch_data['inputs'])
            loss = criterion(prediction, batch_data['labels'])
            train_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_loss.append(train_loss)
            y_true.append(batch_data['labels'].detach().numpy())
            y_pred.append(prediction.detach().numpy())

            print (f'Epoch: [{ep+1}/{epoch}] | iteration: {iteration} | Loss: {np.array(step_loss).mean():.4f}')
            # writer.add_scalar('Loss/train', np.array(step_loss).mean(),  ep * len(train_loader) + iteration)
            # writer.add_scalar('Auc/train', metrics.roc_auc_score(np.concatenate(y_true).ravel(), np.concatenate(y_pred).ravel()), ep * len(train_loader) + iteration)
            
        for name, weight in model.named_parameters():
            writer.add_histogram(name, weight,  ep)
            writer.add_histogram(f'{name}.grad', weight.grad,  ep)
        y_true = np.concatenate(y_true).ravel()
        y_pred = np.concatenate(y_pred).ravel()
        auc = metrics.roc_auc_score(y_true, y_pred)

        step_val_loss = []
        y_val_true = []
        y_val_pred = []

        for iteration, batch_data in enumerate(val_loader):
            with torch.inference_mode(): 
                prediction = model(batch_data['inputs'])
                loss = criterion(prediction, batch_data['labels']).item()

                step_val_loss.append(loss)
                y_val_true.append(batch_data['labels'].detach().numpy())
                y_val_pred.append(prediction.detach().numpy())

                if iteration % 10 == 0:
                    print (f'Epoch: [{ep+1}/{epoch}] | iteration: {iteration} | valLoss: {np.mean(step_val_loss):.4f}')

        y_val_true = np.concatenate(y_val_true).ravel()
        y_val_pred = np.concatenate(y_val_pred).ravel()
        validationEpoch_loss.append(np.mean(step_val_loss))
        var_auc = metrics.roc_auc_score(y_val_true, y_val_pred)


        adjusted_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('lr', adjusted_lr, ep)
        scheduler.step()

        writer.add_scalars('loss', {'train': np.array(step_loss).mean(),
                                    'val':  np.array(step_val_loss).mean()}, ep)

        writer.add_scalars('auc', {'train': auc,
                                    'val':  var_auc}, ep)

        print("=" * 60)
        
        # early stopping
        if early_stop:
            if len(validationEpoch_loss) > 1:
                early_stopping(validationEpoch_loss[ep], validationEpoch_loss[ep-1])
                if early_stopping.early_stop:
                    print("We are at epoch:", ep)
                    break
    return model.eval()


def save_data(data_path: str, name: str, data):
    with open(os.path.join(data_path, name), 'wb') as f:
        pickle.dump(data, f )

def get_data(data_path: str, name: str):
    with open(os.path.join(data_path, name) , 'rb') as f:
        data = pickle.load(f)
    return data

class TupleLoader():
    def __init__(self, dataset, batch_size= 128):
        self.dataset = dataset
        self.size = len(dataset)
        self.batch_size = batch_size
        self.index = np.arange(self.size)
        pass

    def _suffle(self):
        np.random.shuffle(self.index)
        pass

    def batch_data(self):
        self._suffle()
        N, _ = divmod(self.size, self.batch_size)
        if _ > 0:
            N += 1
        return zip(*[ts[self.index].chunk(N) for ts in self.dataset.tensors])


def convert_to_tuple_Dataset(data:dict) -> torch.utils.data.TensorDataset:
    return torch.utils.data.TensorDataset(
        torch.FloatTensor(data['click']),
        torch.LongTensor(data['site_id']),
        torch.LongTensor(data['site_domain']),
        torch.LongTensor(data['app_id']),
        torch.LongTensor(data['device_id']),
        torch.LongTensor(data['device_ip']),
        torch.LongTensor(data['device_model']),
        torch.LongTensor(data['C14']),
        torch.LongTensor(data['C1']),
        torch.LongTensor(data['banner_pos']),
        torch.LongTensor(data['device_type']),
        torch.LongTensor(data['device_conn_type']),
        torch.LongTensor(data['C15']),
        torch.LongTensor(data['C16']),
        torch.LongTensor(data['C18']),
        torch.LongTensor(data['site_category']),
        torch.LongTensor(data['C19']),
        torch.LongTensor(data['C21']),
        torch.LongTensor(data['app_category']),
        torch.LongTensor(data['C20']),
        torch.LongTensor(data['C17']),
        torch.LongTensor(data['app_domain']),
        torch.LongTensor(data['year']),
        torch.LongTensor(data['month']),
        torch.LongTensor(data['date']),
        torch.LongTensor(data['hour']),
        )


def convert_to_tuple_loader(dataset: torch.utils.data.TensorDataset, batch_size: int, val_frac: float= None):
    if val_frac == None:
        train_loader = TupleLoader(dataset, batch_size)
        return train_loader
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1-val_frac, val_frac])
        train_loader =  TupleLoader(train_dataset, batch_size)
        val_loader = TupleLoader(val_dataset, batch_size)
        return [train_loader, val_loader]


def convert_to_data_dict(data, variables, continuous_variables):
    return  {
            variables[i]: torch.FloatTensor(x) if variables[i] in continuous_variables else torch.LongTensor(x)
            for i, x in enumerate(zip(*data[variables].values))
        }

def convert2Loader(dataset: torch.utils.data.TensorDataset, batch_size: int, val_frac: float= None):
    num_workers = Config['training_setting']['num_workers']

    if val_frac == None:
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return data_loader
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1-val_frac, val_frac])
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return [train_loader, val_loader]

def calculate_metrics(model, test_loader):
    metrics_result = {}
    y_true = []
    y_prob = []
    with torch.inference_mode():
        # for batch_data in test_loader.batch_data():
        for batch_data in test_loader:
            prediction = model(batch_data['inputs'])
            y_prob.append(prediction.detach().numpy())
            y_true.append(batch_data['labels'].numpy().astype(int))

    y_true = np.concatenate(y_true).ravel()
    y_prob = np.concatenate(y_prob).ravel()

    auc = metrics.roc_auc_score(y_true, y_prob)

    average_position_score = metrics.average_precision_score(y_true, y_prob, pos_label=1)
    metrics_result["auc"] = auc
    metrics_result["ap"] = average_position_score

    return metrics_result



