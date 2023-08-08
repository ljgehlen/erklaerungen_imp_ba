import torch
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from models import SentenceCNN, BiLSTMClassif
from imp import pruning
from tqdm import tqdm

AG_NEWS = ['World', 'Sports', 'Business', 'Sci/Tech']
RESULT_PATH = "./results"


def _get_gradients_and_outputs(x, y, model, inference_fn=None, device='cpu', batch_size=1):
    """
    calculates outputs and gradients from test data in the model

    parameters:
        x: text data

        y: label data

        model: model for analysing the data

        inference_fn: forward application in model, uses model embedding function

        device: for tensor usage

        batch_size: number of used data instances per calculation round

    return:
        _y_out: tensor with model output per instance, shape[data_instances][number_words][number_categories]

        _pred_grads: tensor, gradients per instance i.r.t. prediction, shape[data_instances][number_words][model_emb_dim]

        _label_grads: tensor, gradients per instance i.r.t. label, shape[data_instances][number_words][model_emb_dim]

        _y_pred: tensor with prediction for category, shape[data_instances]

        _acc: float: accuracy of data in model compared to input Y / labels
    """
    _data = DataLoader(TensorDataset(x), shuffle=False, batch_size=batch_size)
    _labels = DataLoader(TensorDataset(y), shuffle=False, batch_size=batch_size)
    if inference_fn is None:
        inference_fn = model.forward_embedded_softmax

    _y_out = []
    _pred_grads = []
    _label_grads = []
    model.eval()
    for text, label in tqdm(zip(_data, _labels)):
        emb = model.embed_input(text[0]).to(device)
        emb.requires_grad = True
        _out = inference_fn(emb)
        _label = torch.gather(_out, dim=1, index=torch.tensor(label).unsqueeze(1))
        grad = torch.autograd.grad(_label, emb, grad_outputs=torch.ones_like(_label), retain_graph=True)[0].data
        _label_grads.append(grad)
        if torch.argmax(_out, dim=1) == label:
            _pred_grads.append(grad)
        else:
            _pred = torch.gather(_out, dim=1, index=torch.argmax(_out, dim=1).unsqueeze(1))
            grad = torch.autograd.grad(_pred, emb, grad_outputs=torch.ones_like(_pred), retain_graph=True)[0].data
            _pred_grads.append(grad)
        _y_out.append(_out.cpu())
    model.train()
    _y_pred = torch.argmax(torch.vstack(_y_out), dim=1)
    _acc = torch.mean((y == _y_pred).to(torch.float)).detach().cpu().item()
    _pred_grads = torch.sum(torch.vstack(_pred_grads), dim=2)
    _label_grads = torch.sum(torch.vstack(_label_grads), dim=2)
    return torch.vstack(_y_out), _pred_grads, _label_grads, _y_pred, _acc


def _get_gradients_and_outputs_over_it(folder: str, x, y):
    """
    calculates outputs and gradients from test data over all iterations

    parameters:
        folder: string in Form of "LSTM1" or "CNN12"

        x: text data

        y: label data

    return:
        _y_out_list: np array with _y_out of _get_gradients_and_outputs over iterations [iterations][_y_out]

        _pred_grads_list: np array with _pred_grads of _get_gradients_and_outputs over iterations [iterations][_pred_grads]

        _label_grads_list: np array with _label_grads of _get_gradients_and_outputs over iterations [iterations][_label_grads]

        _y_pred_list: np array with _y_pred of _get_gradients_and_outputs over iterations [iterations][_y_pred_list]

        _acc_list: float: np array with _acc of _get_gradients_and_outputs over iterations [iterations][_acc]
    """
    path = RESULT_PATH + "/" + folder + "/"

    file = path + 'model_info.json'
    with open(file, 'r') as f:
        model_info = json.load(f)

    n_iterations = model_info['n_iterations']
    _y_out_list = []
    _pred_grads_list = []
    _label_grads_list = []
    _y_pred_list = []
    _acc_list = []

    for i in range(n_iterations):
        model = get_model(folder, i)
        _y_out, _pred_grads, _label_grads, _y_pred, _acc = _get_gradients_and_outputs(x, y, model)
        _y_out_list.append(_y_out.detach().numpy())
        _pred_grads_list.append(_pred_grads.detach().numpy())
        _label_grads_list.append(_label_grads.detach().numpy())
        _y_pred_list.append(_y_pred.detach().numpy())
        _acc_list.append(_acc)
        print(f"finished extracting gradients of iteration: {i}")

    return np.asarray(_y_out_list), np.asarray(_pred_grads_list), np.asarray(_label_grads_list),\
        np.asarray(_y_pred_list), np.asarray(_acc_list)


def save_grad_and_out_info(folder: str, x, y):
    """
        saves gradient and output information over all iterations in folder

    parameters:
        folder: string in Form of "LSTM1" or "CNN12"

        x: text data

        y: label data

    """
    _y_out, _pred_grads, _label_grads, _y_pred, _acc_list = _get_gradients_and_outputs_over_it(folder, x, y)

    path = "./results/" + folder + "/"

    np.save(path + "_y_out.npy", _y_out)
    np.save(path + "_pred_grads.npy", _pred_grads)
    np.save(path + "_label_grads.npy", _label_grads)
    np.save(path + "_y_pred.npy", _y_pred)
    np.save(path + "_acc_list.npy", _acc_list)

    return _y_out, _pred_grads, _label_grads, _y_pred, _acc_list


def get_category(label, dataset='ag_news'):
    """
    maps label to name of category
    """
    category = None
    if dataset == 'ag_news':
        category = AG_NEWS[label]
    return category


def get_words(tokenized_data, vocab):
    """
    maps tokens in tokenized_data to real words in vocab
    """
    words = []
    for token in tokenized_data:
        words.append(vocab.lookup_token(token))
    return words


def compression_rate(n_iterations=8, pruning_perc=0.5):
    """
    generates compression information over iterations i.r.t pruning percentage
    """
    compression_list = []
    for i in range(n_iterations):
        compression_list.append(int(1/((1-pruning_perc)**i)))
    return np.array(compression_list)


def plot_best_accuracies(data_list: list, n_iterations=8, pruning_perc=0.5):
    compression = compression_rate(n_iterations=n_iterations, pruning_perc=pruning_perc)
    iter_points = np.linspace(1, n_iterations, n_iterations, dtype=int)

    plt.figure(figsize=(12, 6))
    fig, ax = plt.subplots()
    for i in range(len(data_list)):
        ax.plot(iter_points,
                np.load(RESULT_PATH+"/"+data_list[i]+"/best_acc_of_it.npy"),
                label=("Model "+str(i+1)))

    ax.set_xticks(iter_points, compression)
    plt.legend()
    plt.show()


def get_model(folder: str, iteration: int):
    """
    returns model in folder with from specific iteration
    loading pruned models needs preparation for _orig and _mask parameters
    """
    path = RESULT_PATH + "/" + folder + "/"

    file = path + 'model_info.json'
    with open(file, 'r') as f:
        model_info = json.load(f)

    model = None
    if model_info['model_type'] == "LSTM":
        model = BiLSTMClassif(n_classes=model_info['n_classes'], embed_dim=model_info['embed_dim'],
                              vocab_size=model_info['vocab_size'], hid_size=64)
        if iteration == 0:
            model.load_state_dict(torch.load(path+'model_it_'+str(iteration)))
        elif 0 < iteration < model_info['n_iterations']:
            model = pruning(model, pruning_type=model_info['pruning_type'], prune_embedding=model_info['prune_embedding'])
            model.load_state_dict(torch.load(path + 'model_it_' + str(iteration)))
    elif model_info['model_type'] == "CNN":
        model = SentenceCNN(n_classes=model_info['n_classes'], embed_dim=model_info['embed_dim'],
                            vocab_size=model_info['vocab_size'])
        if iteration == 0:
            model.load_state_dict(torch.load(path+'model_it_'+str(iteration)))
        elif 0 < iteration < model_info['n_iterations']:
            model = pruning(model, pruning_type=model_info['pruning_type'], prune_embedding=model_info['prune_embedding'])
            model.load_state_dict(torch.load(path + 'model_it_' + str(iteration)))

    return model


def get_euclidean_distance(folder: str, iteration_1, iteration_2, gradient_type='label'):
    path = RESULT_PATH + "/" + folder + "/"

    file = path + 'model_info.json'
    with open(file, 'r') as f:
        model_info = json.load(f)

    gradients = None
    euclidean = None

    if 0 <= iteration_1 < model_info['n_iterations'] and 0 <= iteration_2 < model_info['n_iterations']:
        if gradient_type == 'label':
            gradients = np.load(path + "_label_grads.npy")
        elif gradient_type == 'pred':
            gradients = np.load(path + "_pred_grads.npy")

        euclidean = np.zeros((gradients.shape[1], gradients.shape[2]))
        for i in range(gradients.shape[1]):
            for j in range(gradients.shape[2]):
                euclidean[i][j] = math.sqrt((gradients[iteration_1][i][j] - gradients[iteration_2][i][j]) ** 2)

    return euclidean


def get_euclidean_over_iterations(folder: str, gradient_type='label'):
    path = RESULT_PATH + "/" + folder + "/"

    file = path + 'model_info.json'
    with open(file, 'r') as f:
        model_info = json.load(f)

    euclidean_over_it = []
    for i in range(model_info['n_iterations']-1):
        euclidean_over_it.append(get_euclidean_distance(folder, i, i+1, gradient_type))

    return np.asarray(euclidean_over_it)


def get_variance_over_iterations(folder:str, gradient_type='label'):
    path = RESULT_PATH + "/" + folder + "/"

    file = path + 'model_info.json'
    with open(file, 'r') as f:
        model_info = json.load(f)

    gradients = None
    if gradient_type == 'label':
        gradients = np.load(path + "_label_grads.npy")
    elif gradient_type == 'pred':
        gradients = np.load(path + "_pred_grads.npy")

    return np.var(gradients, axis=2)


def feature_agreement(folder: str, iteration_1, iteration_2, top_k, gradient_type='label'):
    path = RESULT_PATH + "/" + folder + "/"

    file = path + 'model_info.json'
    with open(file, 'r') as f:
        model_info = json.load(f)

    features_agreement = None
    gradients = None
    if 0 <= iteration_1 < model_info['n_iterations'] and 0 <= iteration_2 < model_info['n_iterations']:
        if gradient_type == 'label':
            gradients = np.load(path + "_label_grads.npy")
        elif gradient_type == 'pred':
            gradients = np.load(path + "_pred_grads.npy")

        features_agreement = torch.zeros(gradients.shape[1], dtype=torch.float)

        grad_it_1 = torch.from_numpy(gradients[iteration_1])
        grad_it_2 = torch.from_numpy(gradients[iteration_2])

        values_it_1, indices_it_1 = torch.topk(torch.abs(grad_it_1), top_k)
        values_it_2, indices_it_2 = torch.topk(torch.abs(grad_it_2), top_k)

        for i in range(gradients.shape[1]):
            for indices1 in indices_it_1[i]:
                for indices2 in indices_it_2[i]:
                    if indices1 == indices2:
                        features_agreement[i] += 1
            features_agreement[i] /= top_k

    return features_agreement


def rank_agreement(folder: str, iteration_1, iteration_2, top_k, gradient_type='label'):
    path = RESULT_PATH + "/" + folder + "/"

    file = path + 'model_info.json'
    with open(file, 'r') as f:
        model_info = json.load(f)

    ranks_agreement = None
    gradients = None
    if 0 <= iteration_1 < model_info['n_iterations'] and 0 <= iteration_2 < model_info['n_iterations']:
        if gradient_type == 'label':
            gradients = np.load(path + "_label_grads.npy")
        elif gradient_type == 'pred':
            gradients = np.load(path + "_pred_grads.npy")

        ranks_agreement = torch.zeros(gradients.shape[1], dtype=torch.float)

        grad_it_1 = torch.from_numpy(gradients[iteration_1])
        grad_it_2 = torch.from_numpy(gradients[iteration_2])

        values_it_1, indices_it_1 = torch.topk(torch.abs(grad_it_1), top_k)
        values_it_2, indices_it_2 = torch.topk(torch.abs(grad_it_2), top_k)

        for i in range(gradients.shape[1]):
            for indices1, indices2 in zip(indices_it_1[i], indices_it_2[i]):
                if indices1 == indices2:
                    ranks_agreement[i] += 1
            ranks_agreement[i] /= top_k

    return ranks_agreement


def sign_agreement(folder: str, iteration_1, iteration_2, top_k, gradient_type='label'):
    path = RESULT_PATH + "/" + folder + "/"

    file = path + 'model_info.json'
    with open(file, 'r') as f:
        model_info = json.load(f)

    signs_agreement = None
    gradients = None
    if 0 <= iteration_1 < model_info['n_iterations'] and 0 <= iteration_2 < model_info['n_iterations']:
        if gradient_type == 'label':
            gradients = np.load(path + "_label_grads.npy")
        elif gradient_type == 'pred':
            gradients = np.load(path + "_pred_grads.npy")

        signs_agreement = torch.zeros(gradients.shape[1], dtype=torch.float)

        grad_it_1 = torch.from_numpy(gradients[iteration_1])
        grad_it_2 = torch.from_numpy(gradients[iteration_2])

        values_it_1, indices_it_1 = torch.topk(torch.abs(grad_it_1), top_k)
        values_it_2, indices_it_2 = torch.topk(torch.abs(grad_it_2), top_k)

        for i in range(gradients.shape[1]):
            for indices1 in indices_it_1[i]:
                for indices2 in indices_it_2[i]:
                    if indices1 == indices2 and np.sign(grad_it_1[i][indices1]) == np.sign(grad_it_2[i][indices2]):
                        signs_agreement[i] += 1
            signs_agreement[i] /= top_k

    return signs_agreement


def signed_rank_agreement(folder: str, iteration_1, iteration_2, top_k, gradient_type='label'):
    path = RESULT_PATH + "/" + folder + "/"

    file = path + 'model_info.json'
    with open(file, 'r') as f:
        model_info = json.load(f)

    signed_ranks_agreement = None
    gradients = None
    if 0 <= iteration_1 < model_info['n_iterations'] and 0 <= iteration_2 < model_info['n_iterations']:
        if gradient_type == 'label':
            gradients = np.load(path + "_label_grads.npy")
        elif gradient_type == 'pred':
            gradients = np.load(path + "_pred_grads.npy")

        signed_ranks_agreement = torch.zeros(gradients.shape[1], dtype=torch.float)

        grad_it_1 = torch.from_numpy(gradients[iteration_1])
        grad_it_2 = torch.from_numpy(gradients[iteration_2])

        values_it_1, indices_it_1 = torch.topk(torch.abs(grad_it_1), top_k)
        values_it_2, indices_it_2 = torch.topk(torch.abs(grad_it_2), top_k)

        for i in range(gradients.shape[1]):
            for indices1, indices2 in zip(indices_it_1[i], indices_it_2[i]):
                if indices1 == indices2 and np.sign(grad_it_1[i][indices1]) == np.sign(grad_it_2[i][indices2]):
                    signed_ranks_agreement[i] += 1
            signed_ranks_agreement[i] /= top_k

    return signed_ranks_agreement


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

    columns = []
    for i in range(vocab_size):
        columns.append(str(i + 1))

    data_df = pd.DataFrame(data_np, columns=columns)

    drop_list = [col for col in data_df.columns if sum(data_df[col]) <= 0]
    data_df.drop(drop_list, axis=1, inplace=True)
    print(f'{len(drop_list)} columns dropped')
    print(f'data shape: {data_df.shape}')

    return data_df
