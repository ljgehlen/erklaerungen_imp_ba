import torch
import json
import math
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models import SentenceCNN, BiLSTMClassif
from imp import pruning
from tqdm import tqdm

AG_NEWS = ['World', 'Sports', 'Business', 'Sci/Tech']
RESULT_PATH = "./results"


def _get_gradients_and_outputs(x, y, model, inference_fn=None, device='cpu', batch_size=1):
    '''
    Returns the gradients of input in regard to output of model

        Parameters:
            x (DataLoader): Input data
            y (DataLoader): Label of Inputs
            model (Module): model
            inference_fn (function): model forward function for input
            device (string): device for tensors ('cpu' or 'cuda:...')
            batch_size (int): batch size for gradients extraction

        Returns:
            _y_out (tensor): label distributions of model outputs
            _pred_grads (tensor): gradients in regard to prediction output
            _label_grads (tensor): gradients in regard to input label output
            _y_pred (tensor): predicted labels of model
            _acc (tensor): accuracy of input
    '''

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
    '''
    Returns the gradients of input in regard to output of model over every iteration

        Parameters:
            folder (str): model
            x (DataLoader): Input data
            y (DataLoader): Label of Inputs

        Returns:
            _y_out_list (tensor): list (n_iterations) of label distributions of model outputs
            _pred_grads_list (tensor): list (n_iterations) of gradients in regard to prediction output
            _label_grads_list (tensor): list (n_iterations) of gradients in regard to input label output
            _y_pred_list (tensor): list (n_iterations) of predicted labels of model
            _acc_list (tensor): list (n_iterations) of accuracies of input
    '''
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
    '''
        savens the gradients of input in regard to output of model over every iteration in numpy arrays

            Parameters:
                folder (str): model
                x (DataLoader): Input data
                y (DataLoader): Label of Inputs

            Returns:
                _y_out (tensor): list (n_iterations) of label distributions of model outputs
                _pred_grads (tensor): list (n_iterations) of gradients in regard to prediction output
                _label_grads (tensor): list (n_iterations) of gradients in regard to input label output
                _y_pred (tensor): list (n_iterations) of predicted labels of model
                _acc (tensor): list (n_iterations) of accuracies of input
    '''
    _y_out, _pred_grads, _label_grads, _y_pred, _acc_list = _get_gradients_and_outputs_over_it(folder, x, y)

    path = "./results/" + folder + "/"

    np.save(path + "_y_out.npy", _y_out)
    np.save(path + "_pred_grads.npy", _pred_grads)
    np.save(path + "_label_grads.npy", _label_grads)
    np.save(path + "_y_pred.npy", _y_pred)
    np.save(path + "_acc_list.npy", _acc_list)

    return _y_out, _pred_grads, _label_grads, _y_pred, _acc_list


def get_category(label, dataset='ag_news'):
    """takes label and dataset  and returns name of the category"""
    category = None
    if dataset == 'ag_news':
        category = AG_NEWS[label]
    return category


def get_words(tokenized_data, vocab):
    """takes token and vocabulary and return the word"""
    words = []
    for token in tokenized_data:
        words.append(vocab.lookup_token(token))
    return words


def compression_rate(n_iterations=8, pruning_perc=0.5):
    """takes iterations and compression rate and returns list of compressions over time"""
    compression_list = []
    for i in range(n_iterations):
        compression_list.append(int(1/((1-pruning_perc)**i)))
    return np.array(compression_list)


def get_model(folder: str, iteration: int):
    """
    returns model in folder with from specific iteration
    loading pruned models needs preparation for _orig and _mask parameters

        Parameters:
            folder (str): model
            iteration (int): iterationd of model that should be loaded

        Returns:
            model (Module): loaded model
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


def agreemets(gradients1, gradients2, top_k):
    """
        returns agreements of top features between feature significance (gradients)

        Parameters:
            gradients1 (tensor): gradients of model 1
            gradients2 (tensor): gradients of model 2
            top_k (int): number of top features

        Returns:
            feat (tensor): feature agreement between models
            sign (tensor): sign agreement between models
            rank (tensor): rank agreement between models
            signed rank (tensor): signed rank agreement between models
    """
    feat = torch.zeros(gradients1.shape[0], dtype=torch.float)
    rank = torch.zeros(gradients1.shape[0], dtype=torch.float)
    sign = torch.zeros(gradients1.shape[0], dtype=torch.float)
    signed_rank = torch.zeros(gradients1.shape[0], dtype=torch.float)

    _, indices1 = torch.topk(torch.abs(gradients1), top_k)
    _, indices2 = torch.topk(torch.abs(gradients2), top_k)

    for i in range(gradients1.shape[0]):
        for ind1 in indices1[i]:
            for ind2 in indices2[i]:
                if ind1 == ind2:
                    feat[i] += 1
                    if np.sign(gradients1[i][ind1]) == np.sign(gradients2[i][ind2]):
                        sign[i] += 1
        feat[i] /= top_k
        sign[i] /= top_k

        for ind1, ind2 in zip(indices1[i], indices2[i]):
            if ind1 == ind2:
                rank[i] += 1
                if np.sign(gradients1[i][ind1]) == np.sign(gradients2[i][ind2]):
                    signed_rank[i] += 1
        rank[i] /= top_k
        signed_rank[i] /= top_k

    return feat, rank, sign, signed_rank


def get_agreements_models(folder1: str, folder2: str, iteration=0, top_k=11, gradient_type='label', relevant=None):
    """
        returns agreements between two unique models and same iteration

        Parameters:
            folder1 (str): name for folder of model 1
            folder2 (str): name for folder of model 2
            iteration (int): iteration, which gets compared between both models
            top_k (int): number of top features
            gradient_type (str): compare gradients in regard to label or prediction
            relevant (list[int]): subset of gradients to extract

        Returns:
            feat (tensor): feature agreement between models
            sign (tensor): sign agreement between models
            rank (tensor): rank agreement between models
            signed rank (tensor): signed rank agreement between models
    """
    path1 = RESULT_PATH + "/" + folder1 + "/"
    path2 = RESULT_PATH + "/" + folder2 + "/"

    if gradient_type == 'label':
        gradients1 = np.load(path1 + "_label_grads.npy")
        gradients2 = np.load(path2 + "_label_grads.npy")
    elif gradient_type == 'pred':
        gradients1 = np.load(path1 + "_pred_grads.npy")
        gradients2 = np.load(path2 + "_pred_grads.npy")

    grads1 = None
    grads2 = None

    if relevant is None:
        grads1 = gradients1[iteration]
        grads2 = gradients2[iteration]
    else:
        grads1 = np.zeros((len(relevant), gradients1.shape[2]), dtype=float)
        grads2 = np.zeros((len(relevant), gradients1.shape[2]), dtype=float)

        for i, r in enumerate(relevant):
            grads1[i] = gradients1[iteration][int(r)]
            grads2[i] = gradients2[iteration][int(r)]

    return agreemets(torch.from_numpy(grads1), torch.from_numpy(grads2), top_k)


def get_agreements_iterations(folder: str, iteration1=0, iteration2=1, top_k=11, gradient_type='label', relevant=None):
    """
        returns agreements between two unique models and same iteration

        Parameters:
            folder (str): name for folder of model
            iteration1 (int): first iteration to compare
            iteration2 (int): second iteration to compare
            top_k (int): number of top features
            gradient_type (str): compare gradients in regard to label or prediction
            relevant (list[int]): subset of gradients to extract

        Returns:
            feat (tensor): feature agreement between models
            sign (tensor): sign agreement between models
            rank (tensor): rank agreement between models
            signed rank (tensor): signed rank agreement between models
        """
    path = RESULT_PATH + "/" + folder + "/"

    if gradient_type == 'label':
        gradients = np.load(path + "_label_grads.npy")
    elif gradient_type == 'pred':
        gradients = np.load(path + "_pred_grads.npy")

    grads1 = None
    grads2 = None

    if relevant is None:
        grads1 = gradients[iteration1]
        grads2 = gradients[iteration2]
    else:
        grads1 = np.zeros((len(relevant), gradients.shape[2]), dtype=float)
        grads2 = np.zeros((len(relevant), gradients.shape[2]), dtype=float)

        for i, r in enumerate(relevant):
            grads1[i] = gradients[iteration1][int(r)]
            grads2[i] = gradients[iteration2][int(r)]

    return agreemets(torch.from_numpy(grads1), torch.from_numpy(grads2), top_k)

