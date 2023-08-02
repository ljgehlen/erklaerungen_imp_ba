import torch
from imp import create_imp_model
from datasets import get_agnews, get_agnews_prepraped
from models import SentenceCNN
from torch.utils.data import DataLoader, TensorDataset
from train import _get_outputs, _get_predictions
import pandas as pd
from utils import _get_gradients_and_outputs


if __name__ == '__main__':
    # create_imp_model(code, model_type,
    #                 size_train_batch=64, n_batches=1875, size_test_batch=1024, embed_dim=128,
    #                 n_epochs=10, n_iterations=5, pruning_perc=0.5, pruning_type='global'):

    # create_imp_model('003', 'CNN', n_iterations=2, n_epochs=2)
    device = 'cpu'
    size_train_batch = 64
    size_test_batch = 1028

    train_set, test_set, vocab_size, n_classes, vocab = get_agnews_prepraped(random_state=42,
                                                                   batch_sizes=(size_train_batch, size_test_batch))
    X_test, Y_test = next(iter(test_set))

    embed_dim = 128
    model = SentenceCNN(n_classes=n_classes, embed_dim=embed_dim, vocab_size=vocab_size)
    model.load_state_dict(torch.load('./results/001/model_1'))
    loss_fun = torch.nn.CrossEntropyLoss()

    _y_outs, _grads, _y_pred, _acc = _get_gradients_and_outputs(X_test, Y_test, model, loss_fun)
