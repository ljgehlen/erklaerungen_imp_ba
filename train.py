import copy
import os.path
from time import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.utils.prune as prune
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from datasets import get_agnews
from models import SentenceCNN, BiLSTMClassif


# author: Sebastian

def _get_outputs(inference_fn, data, model, device, batch_size=256):

    _data = DataLoader(TensorDataset(data), shuffle=False, batch_size=batch_size)

    try:
        _y_out = []
        for x in _data:
            _y = inference_fn(x[0].to(device))
            _y_out.append(_y.cpu())
        return torch.vstack(_y_out)
    except RuntimeError as re:
        if "CUDA out of memory" in str(re):
            model.to('cpu')
            outputs = _get_outputs(inference_fn, data, model, 'cpu')
            model.to('cuda')
            return outputs
        else:
            raise re


def _get_predictions(inference_fn, data, model, device):
    return torch.argmax(_get_outputs(inference_fn, data, model, device), dim=1)


def validate(inference_fn, model, X, Y):

    if inference_fn is None:
        inference_fn = model

    model.eval()
    device = next(model.parameters()).device

    _y_pred = _get_predictions(inference_fn, X, model, device)
    model.train()

    acc = torch.mean((Y == _y_pred).to(torch.float)).detach().cpu().item()  # mean expects float, not bool (or int)
    return acc

def train_loop(model, optim, loss_fn, tr_data: DataLoader, te_data: tuple, inference_fn=None, \
               n_batches_max=10, device='cuda'):
    #print(device)
    model.to(device)
    acc_val = []
    losses = []
    n_batches = 0
    _epochs, i_max = 0, 0
    accs = []
    while n_batches <= n_batches_max:
        for i, (text, labels) in enumerate(tr_data, 0):
            acc = validate(inference_fn, model, *te_data)
            accs.append(acc)
            if i % 100 == 0:
                print(f"test acc @ batch {i+_epochs*i_max}/{n_batches_max}: {acc:.4f}")
            text = text.to(device)
            labels = labels.to(device)
            out = model(text)
            loss = loss_fn(out, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())

            n_batches += 1
            if n_batches > n_batches_max:
                break
        i_max = i

    acc_val.append(validate(inference_fn, model, *te_data))
    print("accuracies over test set")
    print(acc_val)
    return model, losses, accs

# author: Lars Gehlen

def pruning(model, pruning_perc, pruning_type):

    # local pruning using l1_unstructured
    if pruning_type == 'local':
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, 'weight', pruning_perc)
            elif isinstance(module, torch.nn.LSTM):
                prune.l1_unstructured(module, 'weight_ih_l0', pruning_perc)
                prune.l1_unstructured(module, 'weight_hh_l0', pruning_perc)
                prune.l1_unstructured(module, 'weight_ih_l0_reverse', pruning_perc)
                prune.l1_unstructured(module, 'weight_hh_l0_reverse', pruning_perc)

    # global pruning using global_unstructured
    elif pruning_type == 'global':

        # get relevant layers
        parameters = []
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                parameters.append((module, "weight"))
            elif isinstance(module, torch.nn.LSTM):
                parameters.extend(((module, "weight_ih_l0"), (module, "weight_hh_l0"),
                                   (module, "weight_ih_l0_reverse"), (module, "weight_hh_l0_reverse")))

        prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=pruning_perc)

    return model

def reset_param_to_initial(model, init_model):

    # save pruning masks
    masks = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            masks.append(module.weight_mask)

            # clean model of pruning masks for proper reset
            prune.remove(module, "weight")

        elif isinstance(module, torch.nn.LSTM):
            masks.extend(((module.weight_ih_l0_mask), (module.weight_hh_l0_mask),
                          (module.weight_ih_l0_reverse_mask), (module.weight_hh_l0_reverse_mask)))

            prune.remove(module, "weight_ih_l0")
            prune.remove(module, "weight_hh_l0")
            prune.remove(module, "weight_ih_l0_reverse")
            prune.remove(module, "weight_hh_l0_reverse")

    # reset model to initialization
    model.load_state_dict(init_model.state_dict())

    # re-add pruning masks to model
    mask_pruner = prune.CustomFromMask(None)
    step = 0
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            mask_pruner.apply(module, 'weight', masks[step])
            step += 1
        elif isinstance(module, torch.nn.LSTM):
            mask_pruner.apply(module, 'weight_ih_l0', masks[step])
            step += 1
            mask_pruner.apply(module, 'weight_hh_l0', masks[step])
            step += 1
            mask_pruner.apply(module, 'weight_ih_l0_reverse', masks[step])
            step += 1
            mask_pruner.apply(module, 'weight_hh_l0_reverse', masks[step])
            step += 1
    return model

def check_perc(model):
    # check percentages of pruned "weight" data
    perc = []
    global_cnt = 0
    global_pruned = 0
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            tup = (module_name,
                   "{:.2f}%".format(100. * float(torch.count_nonzero(module.weight).item())
                                    / float(torch.numel(module.weight))))
            perc.append(tup)
            global_cnt += torch.numel(module.weight)
            global_pruned += torch.count_nonzero(module.weight).item()
        elif isinstance(module, torch.nn.LSTM):
            tup1 = ("{}.ih_l0".format(module_name),
                   "{:.2f}%".format(100. * float(torch.count_nonzero(module.weight_ih_l0).item())
                                    / float(torch.numel(module.weight_ih_l0))))
            tup2 = ("{}.hh_l0".format(module_name),
                    "{:.2f}%".format(100. * float(torch.count_nonzero(module.weight_hh_l0).item())
                                     / float(torch.numel(module.weight_hh_l0))))
            tup3 = ("{}.ih_l0_reverse".format(module_name),
                    "{:.2f}%".format(100. * float(torch.count_nonzero(module.weight_ih_l0_reverse).item())
                                     / float(torch.numel(module.weight_ih_l0_reverse))))
            tup4 = ("{}.hh_l0_reverse".format(module_name),
                    "{:.2f}%".format(100. * float(torch.count_nonzero(module.weight_hh_l0_reverse).item())
                                     / float(torch.numel(module.weight_hh_l0_reverse))))
            perc.extend((tup1, tup2, tup3, tup4))
            global_cnt = global_cnt + torch.numel(module.weight_ih_l0)\
                         + torch.numel(module.weight_hh_l0)\
                         + torch.numel(module.weight_ih_l0_reverse)\
                         + torch.numel(module.weight_hh_l0_reverse)
            global_pruned = global_pruned + torch.count_nonzero(module.weight_ih_l0).item()\
                            + torch.count_nonzero(module.weight_hh_l0).item()\
                            + torch.count_nonzero(module.weight_ih_l0_reverse).item()\
                            + torch.count_nonzero(module.weight_hh_l0_reverse).item()
    print(f"Local (pruned) Layers: {perc}")
    print("Global Model: {:.2f}%".format(100. * float(global_pruned) / float(global_cnt)))

def imp_loop(model, init_model, pruning_type, optimizer, loss_fun, train_set, X_test, Y_test, n_iterations, pruning_perc, n_batches, device):
    print(device)

    loss_list = [None]*(n_iterations)
    test_accuracies_list = [None]*(n_iterations)
    model_active = model

    iterations = 1
    while iterations <= n_iterations:
        print("----------------------------------------")
        print(f"iteration: {iterations}")

        # train first (100%) model
        if iterations == 1:
            model_active, loss_list[iterations-1], test_accuracies_list[iterations-1] = \
                train_loop(model_active, optimizer, loss_fun, train_set, (X_test, Y_test),
                           inference_fn=model.forward_softmax, device=device, n_batches_max=n_batches)
            iterations += 1

        # prune and train reduced models
        else:
            model_active = pruning(model_active, pruning_perc, pruning_type)
            check_perc(model_active)
            model_active = reset_param_to_initial(model_active, init_model)
            model_active, loss_list[iterations-1], test_accuracies_list[iterations-1] = \
                train_loop(model_active, optimizer, loss_fun, train_set, (X_test, Y_test),
                           inference_fn=model.forward_softmax, device=device, n_batches_max=n_batches)
            iterations += 1

    return model_active, loss_list, test_accuracies_list

# author: Sebastian
# author: Lars Gehlen

if __name__ == '__main__':

    size_train_batch = 64
    size_test_batch = 1024
    n_batches = 2000
    embedding_dim = 128
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_set, test_set, size_vocab, n_classes = get_agnews(random_state=42, batch_sizes=(size_train_batch, size_test_batch))

    X_test, Y_test = next(iter(test_set))  # only use first batch as a test set
    Y_test_distr = torch.bincount(Y_test, minlength=n_classes)/size_test_batch
    print(f"class distribution in test set: {Y_test_distr}")  # this should roughly be uniformly distributed

    # CNN
    #model_type = "CNN"
    #model = SentenceCNN(n_classes=n_classes, embed_dim=embedding_dim, vocab_size=size_vocab)
    #init_model = SentenceCNN(n_classes=n_classes, embed_dim=embedding_dim, vocab_size=size_vocab)

    # LSTM
    model_type = "LSTM"
    model = BiLSTMClassif(n_classes=n_classes, embed_dim=embedding_dim, vocab_size=size_vocab, hid_size=64)
    init_model = BiLSTMClassif(n_classes=n_classes, embed_dim=embedding_dim, vocab_size=size_vocab, hid_size=64)

    # copy initial parameters for imp reset
    init_model.load_state_dict(model.state_dict())

    optimizer = Adam(model.parameters())
    loss_fun = torch.nn.CrossEntropyLoss()

    # imp hyperparameters
    n_iterations = 5
        # float between 0.0 and 1.0
    pruning_perc = 0.5
        # local or global
    #pruning_type = 'local'
    pruning_type = 'global'

    _t_start = time()

    #model, loss, test_accuracies = \
    #    train_loop(model, optimizer, loss_fun, train_set, (X_test, Y_test),
    #               inference_fn=model.forward_softmax, device=device, n_batches_max=n_batches)

    models, loss_list, test_accuracies_list = \
        imp_loop(model, init_model, pruning_type, optimizer, loss_fun, train_set, X_test, Y_test, n_iterations, pruning_perc, n_batches, device)

    _t_end = time()
    now = datetime.now()
    print(f"Training finished in {int(_t_end - _t_start)} s")
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")

    #print(loss_list)
    #print(test_accuracies_list)

    results = "./results/" + date_time + " - " + model_type

    if not os.path.exists(results):
        os.makedirs(results)

    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html
    for i in range(n_iterations):
        pruned = f"{(1.0 - ((1-pruning_perc**(i))))*100}% network"

        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('#batches')
        ax1.set_ylim(0, 1.)
        ax1.set_ylabel('test accuracy', color=color)
        ax1.plot(test_accuracies_list[i],  color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:orange'
        ax2.set_ylabel('losses', color=color)  # we already handled the x-label with ax1
        ax2.set_ylim(min(0, min(loss_list[i])), max(loss_list[i]))
        ax2.plot(loss_list[i], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        name = results + f"/{pruned}.png"
        fig.suptitle(pruned)
        plt.savefig(name)
        #plt.show()
