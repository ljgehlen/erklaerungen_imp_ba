import torch
import time
import os
import torch.nn.utils.prune as prune
import numpy as np
from torch.utils.data import DataLoader
from train import train
from datasets import get_agnews
from models import SentenceCNN, BiLSTMClassif
from torch.optim import Adam

RESULT_PATH = "./results"


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


def reset_model(model, init_model):
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
            global_cnt = global_cnt + torch.numel(module.weight_ih_l0) \
                         + torch.numel(module.weight_hh_l0) \
                         + torch.numel(module.weight_ih_l0_reverse) \
                         + torch.numel(module.weight_hh_l0_reverse)
            global_pruned = global_pruned + torch.count_nonzero(module.weight_ih_l0).item() \
                            + torch.count_nonzero(module.weight_hh_l0).item() \
                            + torch.count_nonzero(module.weight_ih_l0_reverse).item() \
                            + torch.count_nonzero(module.weight_hh_l0_reverse).item()
    print(f"Local (pruned) Layers: {perc}")
    print("Global Model: {:.2f}%".format(100. * float(global_pruned) / float(global_cnt)))


def imp_loop(path, model, init_model, best_model, device, optimizer, loss_fun, n_batches, n_epochs,
             pruning_type, pruning_perc, n_iterations, tr_data: DataLoader, te_data: tuple):
    acc_list = [None] * n_iterations
    best_acc_list = [None] * n_iterations
    best_acc_it_list = [None] * n_iterations

    active_model = model

    if not os.path.exists(path):
        os.makedirs(path)

    iteration = 1
    while iteration <= n_iterations:
        print("----------------------------------------")
        print(f"iteration: {iteration}")
        epoch = 1
        pruned = 0

        best_acc = 0
        best_acc_it = 0
        epochs_acc = [None] * n_epochs

        while epoch <= n_epochs:
            print(f"iteration {iteration} on epoch {epoch}")

            if iteration > 1 and pruned == 0:
                active_model = pruning(active_model, pruning_perc, pruning_type)
                best_model = pruning(active_model, pruning_perc, pruning_type)
                active_model = reset_model(active_model, init_model)
                check_perc(active_model)
                pruned = 1

            active_model, last_acc = train(model, optimizer, loss_fun, tr_data, te_data,
                                           inference_fn=model.forward_softmax, device=device, n_batches_max=n_batches)

            epochs_acc[epoch - 1] = last_acc[0]
            if best_acc < last_acc[0]:
                best_acc = last_acc[0]
                best_acc_it = iteration
                best_model.load_state_dict(active_model.state_dict())

            epoch += 1

        print(f"stopped in iteration {iteration} on epoch {epoch} with best acc {best_acc}")
        acc_list[iteration - 1] = epochs_acc
        best_acc_list[iteration - 1] = best_acc
        best_acc_it_list[iteration - 1] = best_acc_it
        temp = path + '/model_' + str(iteration)
        torch.save(best_model.state_dict(), temp)
        iteration += 1

    return active_model, acc_list, best_acc_list, best_acc_it_list


def create_imp_model(code, model_type,
                     size_train_batch=64, n_batches=1875, size_test_batch=1024, embed_dim=128,
                     n_epochs=10, n_iterations=5, pruning_perc=0.5, pruning_type='global'):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_set, test_set, vocab_size, n_classes, vocab = get_agnews(random_state=42,
                                                                   batch_sizes=(size_train_batch, size_test_batch))

    X_test, Y_test = next(iter(test_set))

    model = None
    init_model = None
    best_model = None
    if model_type == "CNN":
        model = SentenceCNN(n_classes=n_classes, embed_dim=embed_dim, vocab_size=vocab_size)
        init_model = SentenceCNN(n_classes=n_classes, embed_dim=embed_dim, vocab_size=vocab_size)
        best_model = SentenceCNN(n_classes=n_classes, embed_dim=embed_dim, vocab_size=vocab_size)
    elif model_type == "LSTM":
        model = BiLSTMClassif(n_classes=n_classes, embed_dim=embed_dim, vocab_size=vocab_size, hid_size=64)
        init_model = BiLSTMClassif(n_classes=n_classes, embed_dim=embed_dim, vocab_size=vocab_size, hid_size=64)
        best_model = BiLSTMClassif(n_classes=n_classes, embed_dim=embed_dim, vocab_size=vocab_size, hid_size=64)

    if model is not None:
        init_model.load_state_dict(model.state_dict())
        best_model.load_state_dict(model.state_dict())

        optimizer = Adam(model.parameters())
        loss_fun = torch.nn.CrossEntropyLoss()

        path = RESULT_PATH + '/' + code
        _t_start = time.time()

        model, acc_list, best_accs_list, best_acc_iteration_list = \
            imp_loop(path, model, init_model, best_model, device, optimizer, loss_fun, n_batches, n_epochs,
                     pruning_type, pruning_perc, n_iterations, train_set, (X_test, Y_test))

        _t_end = time.time()
        print(f"Training finished in {int(_t_end - _t_start)} s")

        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + '/readme.txt', 'w') as f:
            f.write(str(model_type))
            f.write('\n')
            f.write("pruning_type: " + str(pruning_type))
            f.write('\n')
            f.write("pruning_perc: " + str(pruning_perc))
            f.write('\n')
            f.write("n_iterations: " + str(n_iterations))
            f.write('\n')
            f.write(
                "size_train_batch: " + str(size_train_batch) + ", n_batches: " + str(n_batches) + ", n_epochs: " + str(
                    n_epochs))
            f.write('\n')
            for i in range(n_iterations):
                f.write(str(pruning_perc ** i) + ", best acc: " + str(best_accs_list[i]) + " in epoch: " + str(
                    best_acc_iteration_list[i]))
                f.write('\n')
            array_acc_list = np.asarray(acc_list)
            np.save(path + "/acc_list.npy", array_acc_list)
            array_best_accs_list = np.asarray(best_accs_list)
            np.save(path + "/best_accs.npy", array_best_accs_list)
            array_best_acc_iteration_list = np.asarray(best_acc_iteration_list)
            np.save(path + "/best_acc_iterations.npy", array_best_acc_iteration_list)

    else:
        print(f'could not setup model')