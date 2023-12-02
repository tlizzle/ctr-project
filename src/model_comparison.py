import os
from datetime import datetime
import numpy as np
import pickle
from torch.utils.data import DataLoader
import pandas as pd
from typing import List, Union
from torch.utils.data.dataset import Dataset
import torch
from sklearn.model_selection import train_test_split
from sklearn import metrics
from src.models import deepctr, embedding_mlp
import src.utils as utils
from src.config import Config
from src.logger import get_logger

log = get_logger(logger_name= "Model comparison", save_file= False)


BATCH_SIZE = 2**13


def main():
    _train_path = Config['data']['train_path']
    df = pd.read_csv(_train_path, compression = 'gzip', nrows= 100000)

    catagory_variables = ['site_id', 'site_domain', 'app_id', 'device_id', 'device_ip',\
        'device_model', 'C14', 'C1', 'banner_pos', 'device_type',\
        'device_conn_type', 'C15', 'C16', 'C18', 'site_category',\
        'C19', 'C21', 'app_category', 'C20', 'C17', 'app_domain',]

    continuous_variables = []


    data, mapping_dict = utils.data_pre_processing(df, catagory_variables)
    variables = catagory_variables + continuous_variables + ['year', 'month', 'date']
    log.info('Data preprocessed !')


    train_dataset, val_dataset = train_test_split(data, test_size= 0.2, stratify= data["click"])
    val_dataset, test_dataset = train_test_split(val_dataset, test_size= 0.5, stratify= val_dataset["click"])
    log.info("split dataset completed !")


    DATA_FORMATION = 'dict'
    if DATA_FORMATION == 'dict':
        train_dict = utils.convert_to_data_dict(train_dataset, variables, continuous_variables)
        val_dict = utils.convert_to_data_dict(val_dataset, variables, continuous_variables)
        test_dict = utils.convert_to_data_dict(test_dataset, variables, continuous_variables)

        train_dataset = utils.O3Dataset(train_dict, variables, labels=torch.FloatTensor(train_dataset['click'].tolist()).unsqueeze(1))
        val_dataset = utils.O3Dataset(val_dict, variables, labels=torch.FloatTensor(val_dataset['click'].tolist()).unsqueeze(1))
        test_dataset = utils.O3Dataset(test_dict, variables, labels=torch.FloatTensor(test_dataset['click'].tolist()).unsqueeze(1))
        log.info("Convert to torch dataset completed !")

        train_loader = DataLoader(train_dataset, batch_size=2**13, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=2**13, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=2**13, shuffle=True, num_workers=4)
        log.info("Convert to torch data loader completed !")

    else:
        train_dataset = utils.convert_to_tuple_Dataset(train_dataset.to_dict("list"))
        val_dataset = utils.convert_to_tuple_Dataset(val_dataset.to_dict("list"))
        test_dataset = utils.convert_to_tuple_Dataset(test_dataset.to_dict("list"))
        log.info("Convert to torch dataset completed !")

        train_loader = utils.convert_to_tuple_loader(train_dataset, BATCH_SIZE)
        val_loader = utils.convert_to_tuple_loader(val_dataset, BATCH_SIZE)
        test_loader = utils.convert_to_tuple_loader(test_dataset, BATCH_SIZE)
        log.info("Convert to torch data loader completed !")

    model = deepctr.O3Model(
        mapping_dict, continuous_variables=continuous_variables, catagory_variables=catagory_variables + ['year', 'month', 'date'],
        embedding_size=8, last_mlp_size=32,
        omlp_use_bn=False, infer_use_bn=True)
    log.info("Model built !")

    log.info("Model start training !")
    trained_model = utils.training(model, train_loader, val_loader, 'test', lr= 1e-1, epoch= 30)


    metrics_result = utils.calculate_metrics(trained_model, test_loader)
    log.info(f"Model metrics result: {metrics_result} !")

# 2023-11-25 18:00:39 [Model comparison] [INFO] Model metrics result: {'auc': 0.722560925962229, 'ap': 0.31120321162092535} !
    log.info('=' * 60)

    model = embedding_mlp.Embedding_Mlp(mapping_dict, continuous_variables, catagory_variables)
    log.info("Model built !")


    log.info("Model start training !")
    trained_model = utils.training(model, train_loader, val_loader, 'test', lr= 1e-1, epoch= 30)


    metrics_result = utils.calculate_metrics(trained_model, test_loader)
    log.info(f"Model metrics result: {metrics_result} !")

# 2023-11-25 18:06:27 [Model comparison] [INFO] Model metrics result: {'auc': 0.7292035014346546, 'ap': 0.3431573875758657} !

if __name__ == "__main__":
    main()


