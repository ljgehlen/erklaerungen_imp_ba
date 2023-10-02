import torch
import time
import os
import json
import torch.nn.utils.prune as prune
import numpy as np
from torch.utils.data import DataLoader
from train import train
from datasets import get_agnews
from models import SentenceCNN, BiLSTMClassif
from torch.optim import Adam
from imp_utils import unique_path

RESULT_PATH = "./results"


def pruning(model, pruning_perc=0.5, pruning_type='global', prune_embedding=True):
    # local pruning using l1_unstructured
    if pruning_type == 'local':
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or \
                    (prune_embedding is True and isinstance(module, torch.nn.Embedding)):
                prune.l1_unstructured(module, 'weight', pruning_perc)
            elif isinstance(module, torch.nn.LSTM):
                prune.l1_unstructured(module, 'weight_ih_l0', pruning_perc)
                prune.l1_unstructured(module, 'weight_hh_l0', pruning_perc)
                prune.l1_unstructured(module, 'weight_ih_l0_reverse', pruning_perc)
                prune.l1_unstructured(module, 'weight_hh_l0_reverse', pruning_perc)

    # global pruning using global_unstructured (L1_Norm), Embedding local
    elif pruning_type == 'global':

        # get relevant layers
        parameters = []
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                parameters.append((module, "weight"))
            elif prune_embedding is True and isinstance(module, torch.nn.Embedding):
                prune.l1_unstructured(module, 'weight', pruning_perc)
            elif isinstance(module, torch.nn.LSTM):
                parameters.extend(((module, "weight_ih_l0"), (module, "weight_hh_l0"),
                                   (module, "weight_ih_l0_reverse"), (module, "weight_hh_l0_reverse")))

        prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=pruning_perc)

    # unmodified global pruning using global_unstructured (L1_Norm)
    elif pruning_type == 'global_unmodified':

        # get relevant layers
        parameters = []
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                parameters.append((module, "weight"))
            elif prune_embedding is True and isinstance(module, torch.nn.Embedding):
                parameters.append((module, "weight"))
            elif isinstance(module, torch.nn.LSTM):
                parameters.extend(((module, "weight_ih_l0"), (module, "weight_hh_l0"),
                                (module, "weight_ih_l0_reverse"), (module, "weight_hh_l0_reverse")))

        prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=pruning_perc)

    return model


def pruning_random(model, pruning_perc=0.5, pruning_type='global', prune_embedding=True):
    # local pruning using random_unstructured
    if pruning_type == 'local':
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or \
                    (prune_embedding is True and isinstance(module, torch.nn.Embedding)):
                prune.random_unstructured(module, 'weight', pruning_perc)
            elif isinstance(module, torch.nn.LSTM):
                prune.random_unstructured(module, 'weight_ih_l0', pruning_perc)
                prune.random_unstructured(module, 'weight_hh_l0', pruning_perc)
                prune.random_unstructured(module, 'weight_ih_l0_reverse', pruning_perc)
                prune.random_unstructured(module, 'weight_hh_l0_reverse', pruning_perc)

    # global pruning using global_unstructured (Random), Embedding local
    elif pruning_type == 'global':

        # get relevant layers
        parameters = []
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                parameters.append((module, "weight"))
            elif prune_embedding is True and isinstance(module, torch.nn.Embedding):
                prune.random_unstructured(module, 'weight', pruning_perc)
            elif isinstance(module, torch.nn.LSTM):
                parameters.extend(((module, "weight_ih_l0"), (module, "weight_hh_l0"),
                                   (module, "weight_ih_l0_reverse"), (module, "weight_hh_l0_reverse")))

        prune.global_unstructured(parameters, pruning_method=prune.RandomUnstructured, amount=pruning_perc)

    # unmodified global pruning using global_unstructured
    elif pruning_type == 'global_unmodified':

        # get relevant layers
        parameters = []
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                parameters.append((module, "weight"))
            elif prune_embedding is True and isinstance(module, torch.nn.Embedding):
                parameters.append((module, "weight"))
            elif isinstance(module, torch.nn.LSTM):
                parameters.extend(((module, "weight_ih_l0"), (module, "weight_hh_l0"),
                                   (module, "weight_ih_l0_reverse"), (module, "weight_hh_l0_reverse")))

        prune.global_unstructured(parameters, pruning_method=prune.RandomUnstructured, amount=pruning_perc)

    return model


def reset_model(model, init_model, prune_embedding=True):
    # save pruning masks
    masks = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or \
                (prune_embedding is True and isinstance(module, torch.nn.Embedding)):
            masks.append(module.weight_mask)
            # clean model of pruning masks for proper reset
            prune.remove(module, "weight")
        elif isinstance(module, torch.nn.LSTM):
            masks.extend((module.weight_ih_l0_mask, module.weight_hh_l0_mask,
                          module.weight_ih_l0_reverse_mask, module.weight_hh_l0_reverse_mask))

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
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or \
                (prune_embedding is True and isinstance(module, torch.nn.Embedding)):
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


def check_perc(model, prune_embedding=True):
    # check percentages of pruned "weight" data
    perc = []
    global_cnt = 0
    global_pruned = 0
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or \
                (prune_embedding is True and isinstance(module, torch.nn.Embedding)):
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
            global_cnt = global_cnt + torch.numel(module.weight_ih_l0) \
                         + torch.numel(module.weight_hh_l0) \
                         + torch.numel(module.weight_ih_l0_reverse) \
                         + torch.numel(module.weight_hh_l0_reverse)
            global_pruned = global_pruned + torch.count_nonzero(module.weight_ih_l0).item() \
                            + torch.count_nonzero(module.weight_hh_l0).item() \
                            + torch.count_nonzero(module.weight_ih_l0_reverse).item() \
                            + torch.count_nonzero(module.weight_hh_l0_reverse).item()
    tup = ('Global Model',
           "{:.2f}%".format(100. * float(global_pruned) / float(global_cnt)))
    perc.append(tup)
    print(f"trainable weights remaining: {perc}")


def imp_loop(path, model, init_model, best_model, device, optimizer, loss_fun, n_epochs, late_rewind,
             pruning_type, pruning_perc, prune_embedding, prune_random, n_iterations, tr_data: DataLoader,
             te_data: tuple):

    # data to save
    best_acc_of_it = np.zeros(n_iterations, dtype=float)  # shape: (n_iterations)
    best_epoch_of_it = np.zeros(n_iterations, dtype=int)  # shape: (n_iterations)
    last_epoch_of_it = np.zeros(n_iterations, dtype=int)  # shape: (n_iterations)
    acc_of_epochs_over_it = np.zeros((n_iterations, n_epochs), dtype=float)  # shape: (n_iterations x n_epochs)
    test_losses_of_epoch_over_it = np.zeros((n_iterations, n_epochs), dtype=float)  # shape: (n_iterations x n_epochs)
    train_losses_of_epoch_over_it = np.zeros((n_iterations, n_epochs), dtype=float)  # shape: (n_iterations x n_epochs)

    active_model = model
    patience = int(n_epochs*0.5)

    if not os.path.exists(path):
        os.makedirs(path)

    iteration = 0
    while iteration < n_iterations:
        print("----------------------------------------")
        print(f"iteration: {iteration+1}")

        # prune starting from second iteration
        if iteration > 0:
            print(f"pruning model by {pruning_perc}")
            if prune_random is False:
                active_model = pruning(active_model, pruning_perc, pruning_type, prune_embedding)
            else:
                active_model = pruning_random(active_model, pruning_perc, pruning_type, prune_embedding)
            active_model = reset_model(active_model, init_model, prune_embedding)
            if iteration == 1:
                best_model = pruning(best_model, pruning_perc, pruning_type, prune_embedding)
        check_perc(active_model, prune_embedding)

        early_stop = False
        last_update = 0
        epoch = 0
        test_loss = 0
        train_loss = 0
        test_acc = 0
        best_acc = 0
        best_acc_epoch = 0
        while epoch < n_epochs and early_stop is False:
            print("----------------------------------------")
            print(f"iteration {iteration+1} on epoch {epoch+1}")

            check_perc(model)
            active_model, test_acc, test_loss, train_loss = train(model, optimizer, loss_fun, tr_data, te_data,
                                                                  inference_fn=model.forward_softmax, device=device)

            print(f"accuracies over test set: {test_acc}")
            test_losses_of_epoch_over_it[iteration][epoch] = test_loss
            train_losses_of_epoch_over_it[iteration][epoch] = train_loss
            acc_of_epochs_over_it[iteration][epoch] = test_acc

            if best_acc < test_acc:
                best_acc = test_acc
                best_acc_epoch = epoch
                best_model.load_state_dict(active_model.state_dict())
                last_update = 0
            else:
                last_update += 1

            if last_update >= patience:
                early_stop = True

            # if late rewind is true: set init_model for reset to weights after first epoch of the not pruned model
            if iteration == 0 and epoch == 0 and late_rewind is True:
                init_model.load_state_dict(active_model.state_dict())

            epoch += 1

        last_epoch_of_it[iteration] = epoch-1

        print(f"stopped in iteration {iteration + 1} after epoch {epoch} with acc {best_acc}")
        if early_stop is True:
            print("early stop because of no increase in test accuracy")
            while epoch < n_epochs:
                test_losses_of_epoch_over_it[iteration][epoch] = test_loss
                train_losses_of_epoch_over_it[iteration][epoch] = train_loss
                acc_of_epochs_over_it[iteration][epoch] = test_acc
                epoch += 1

        # save best model, it's accuracy + epoch from all trained epochs
        model_path = path + '/model_it_' + str(iteration)
        torch.save(best_model.state_dict(), model_path)
        best_acc_of_it[iteration] = best_acc
        best_epoch_of_it[iteration] = best_acc_epoch

        iteration += 1

    return best_acc_of_it, best_epoch_of_it, last_epoch_of_it, acc_of_epochs_over_it, test_losses_of_epoch_over_it,\
        train_losses_of_epoch_over_it


def create_imp_model(model_type, model=None, device=None,
                     dataset='ag_news', size_train_batch=64, size_test_batch=1028,
                     embed_dim=128, n_epochs=20, late_rewind=True,
                     n_iterations=8, pruning_perc=0.5, pruning_type='global', prune_embedding=True, prune_random=False):

    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    dataset_loaded = False
    train_set = None
    test_set = None
    vocab_size = 0
    n_classes = 0
    vocab = None
    if dataset == 'ag_news':
        train_set, test_set, vocab_size, n_classes, vocab = get_agnews(random_state=42,
                                                                   batch_sizes=(size_train_batch, size_test_batch))

        X_test, Y_test = next(iter(test_set))
        dataset_loaded = True
    else:
        print(f"could not read and load dataset: {dataset}")

    init_model = None
    best_model = None
    if model_type == "CNN" and dataset_loaded is True:
        if model is None:
            model = SentenceCNN(n_classes=n_classes, embed_dim=embed_dim, vocab_size=vocab_size)
        init_model = SentenceCNN(n_classes=n_classes, embed_dim=embed_dim, vocab_size=vocab_size)
        best_model = SentenceCNN(n_classes=n_classes, embed_dim=embed_dim, vocab_size=vocab_size)
    elif model_type == "LSTM" and dataset_loaded is True:
        if model is None:
            model = BiLSTMClassif(n_classes=n_classes, embed_dim=embed_dim, vocab_size=vocab_size, hid_size=64)
        init_model = BiLSTMClassif(n_classes=n_classes, embed_dim=embed_dim, vocab_size=vocab_size, hid_size=64)
        best_model = BiLSTMClassif(n_classes=n_classes, embed_dim=embed_dim, vocab_size=vocab_size, hid_size=64)

    if model is not None:
        init_model.load_state_dict(model.state_dict())
        best_model.load_state_dict(model.state_dict())

        optimizer = Adam(model.parameters())
        loss_fun = torch.nn.CrossEntropyLoss()

        path = RESULT_PATH + '/' + model_type
        path = unique_path(path)
        _t_start = time.time()

        best_acc_of_it, best_epoch_of_it, last_epoch_of_it, acc_of_epochs_over_it, test_losses_of_epoch_over_it,\
            train_losses_of_epoch_over_it = imp_loop(path, model, init_model, best_model, device, optimizer, loss_fun,
                                                     n_epochs, late_rewind, pruning_type, pruning_perc, prune_embedding,
                                                     prune_random, n_iterations, train_set, (X_test, Y_test))

        _t_end = time.time()
        print(f"Training finished in {int(_t_end - _t_start)} s")

        if not os.path.exists(path):
            os.makedirs(path)

        model_info = {
            'model_type': model_type,
            'dataset': dataset,
            'vocab_size': vocab_size,
            'n_classes': n_classes,
            'embed_dim': embed_dim,
            'n_epochs': n_epochs,
            'late_rewind': late_rewind,
            'n_iterations': n_iterations,
            'pruning_perc': pruning_perc,
            'pruning_type': pruning_type,
            'prune_embedding': prune_embedding
        }
        file = path + '/model_info.json'
        with open(file, 'w') as f:
            json.dump(model_info, f)

        np.save(path + "/acc_of_epochs_over_it.npy", acc_of_epochs_over_it)
        np.save(path + "/test_losses_of_epoch_over_it.npy", test_losses_of_epoch_over_it)
        np.save(path + "/train_losses_of_epoch_over_it.npy", train_losses_of_epoch_over_it)
        np.save(path + "/best_acc_of_it.npy", best_acc_of_it)
        np.save(path + "/best_epoch_of_it.npy", best_epoch_of_it)
        np.save(path + "/last_epoch_of_it.npy", last_epoch_of_it)

        with open(path + '/readme.txt', 'w') as f:
            f.write(str(model_type))
            f.write('\n')
            f.write("late_rewind: " + str(late_rewind))
            f.write('\n')
            f.write("size_train_batch: " + str(size_train_batch) + ", size_test_batch: " + str(size_test_batch))
            f.write('\n')
            f.write("embed_dim: " + str(embed_dim))
            f.write('\n')
            f.write("n_epochs: " + str(n_epochs))
            f.write('\n')
            f.write("n_iterations: " + str(n_iterations))
            f.write('\n')
            f.write("pruning_type: " + str(pruning_type))
            f.write('\n')
            f.write("pruning_perc: " + str(pruning_perc))
            f.write('\n')
            f.write("prune_embedding: " + str(prune_embedding))
            f.write('\n')
            for i in range(n_iterations):
                f.write(str((1-pruning_perc) ** i) + ", loss: " + str(test_losses_of_epoch_over_it[i][best_epoch_of_it[i]]) + ", acc: " +
                        str(best_acc_of_it[i]) + " in epoch: " + str(best_epoch_of_it[i]) + ", after stopping in epoch: " + str(last_epoch_of_it[i]))
                f.write('\n')

        print(f'results saved in {path}')

    else:
        print(f'could not setup model')
