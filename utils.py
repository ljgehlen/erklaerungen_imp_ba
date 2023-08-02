import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def _get_gradients_and_outputs(X, Y, model, loss_fun, inference_fn=None, device='cpu', batch_size=1):
    """
    calculates outputs and gradients from test data in the model
    requires clean-up of embedding dimension in returned gradients

    parameters:
        X: text data

        Y: label data

        model: model for analysing the data

        loss_fun: loss function for gradient calculation w.r.t outputs

        inference_fn: forward application in model, uses model embedding function

        device: for torch.tensor usage, 'cpu' necessary for autograd.grad input tensors

        batch_size: number of used data instances per calculation round

    return:
        _y_out: torch.tensor with model output per instance, shape[data_instances][data_instances_nmbr_words][nmbr_categories]

        _grads: torch.tensor with model output per instance, shape[data_instances][data_instances_nmbr_words][model_emb_dim]

        _y_pred: torch.tensor with preditction for category, shape[data_instances]

        _acc: float: accuracy of data in model compated to input Y / labels
    """
    _data = DataLoader(TensorDataset(X), shuffle=False, batch_size=batch_size)
    _labels = DataLoader(TensorDataset(Y), shuffle=False, batch_size=batch_size)
    if inference_fn == None:
        inference_fn = model.forward_embedded_softmax

    _y_out = []
    _grads = []
    model.eval()
    for text, label in zip(_data, _labels):
        emb = model.embed_input(text[0]).to(device)
        emb.requires_grad = True
        _y = inference_fn(emb)
        _y_out.append(_y.cpu())
        _loss = loss_fun(_y, label[0])
        grad = torch.autograd.grad(_y, emb, retain_graph=True)[0].data
        _grads.append(grad)
    model.train()
    _y_pred = torch.argmax(torch.vstack(_y_out), dim=1)
    _acc = torch.mean((Y == _y_pred).to(torch.float)).detach().cpu().item()
    return torch.vstack(_y_out), torch.vstack(_grads), _y_pred, _acc


def create_df(data, vocab_size):
    """
    creates dataframe of data and cleans up empty columns

    parameters:
        data: data to encode to Dataframe

        vocab_size: needed for column names based on tokenizer ids

    return:
        data_df: pandas Dataframe one-hot encoded like from data
    """
    data_np = np.zeros((len(data), vocab_size), dtype=np.int8)
    for i, instance in enumerate(data):
        for word in instance:
            data_np[i][word] = 1

    cmns = []
    for i in range(vocab_size):
        cmns.append(str(i + 1))

    data_df = pd.DataFrame(data_np, columns=cmns)

    drop_list = [col for col in data_df.columns if sum(data_df[col]) <= 0]
    data_df.drop(drop_list, axis=1, inplace=True)
    print(f'{len(drop_list)} columns dropped')
    print(f'data shape: {data_df.shape}')

    return data_df
