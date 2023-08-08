import torch
import numpy as np
from imp import create_imp_model
from datasets import get_agnews
from models import SentenceCNN
from torch.utils.data import DataLoader, TensorDataset
from train import _get_outputs
import pandas as pd
from utils import save_grad_and_out_info


if __name__ == '__main__':
    # create_imp_model(model_type, device=None,
    #                 dataset='ag_news', size_train_batch=64, size_test_batch=1028,
    #                 embed_dim=128, n_epochs=10, late_rewind=False,
    #                 n_iterations=8, pruning_perc=0.5, pruning_type='global', prune_embedding=True):

    # create_imp_model('LSTM', late_rewind=True)

    X_test = torch.load("./datasets/AG_NEWS/1028_X_test.pt")
    Y_test = torch.load("./datasets/AG_NEWS/1028_Y_test.pt")

    folder = "LSTM5"

    _y_out, _pred_grads, _label_grads, _y_pred, _acc_list = save_grad_and_out_info(folder, X_test, Y_test)

    folder = "LSTM6"

    _y_out, _pred_grads, _label_grads, _y_pred, _acc_list = save_grad_and_out_info(folder, X_test, Y_test)

    folder = "LSTM8"

    _y_out, _pred_grads, _label_grads, _y_pred, _acc_list = save_grad_and_out_info(folder, X_test, Y_test)

    folder = "LSTM9"

    _y_out, _pred_grads, _label_grads, _y_pred, _acc_list = save_grad_and_out_info(folder, X_test, Y_test)

    folder = "LSTM10"

    _y_out, _pred_grads, _label_grads, _y_pred, _acc_list = save_grad_and_out_info(folder, X_test, Y_test)

