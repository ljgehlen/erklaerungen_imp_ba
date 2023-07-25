import torch
from imp import create_imp_model
from datasets import get_agnews
from rules import save_rules, encode_test
from models import SentenceCNN
from torch.utils.data import DataLoader, TensorDataset
from train import _get_outputs, _get_predictions


if __name__ == '__main__':
    # create_imp_model(code, model_type,
    #                 size_train_batch=64, n_batches=1875, size_test_batch=1024, embed_dim=128,
    #                 n_epochs=10, n_iterations=5, pruning_perc=0.5, pruning_type='global'):

    # create_imp_model('003', 'CNN', n_iterations=2, n_epochs=2)
    device = 'cpu'
    size_train_batch = 64
    size_test_batch = 1028

    train_set, test_set, vocab_size, n_classes, vocab = get_agnews(random_state=42,
                                                                   batch_sizes=(size_train_batch, size_test_batch))
    X_test, Y_test = next(iter(test_set))

    n_classes = 4
    embed_dim = 128
    model = SentenceCNN(n_classes=n_classes, embed_dim=embed_dim, vocab_size=vocab_size)
    model.load_state_dict(torch.load('./results/001/model_1'))

    _data = DataLoader(TensorDataset(X_test), shuffle=False, batch_size=1)

    model.zero_grad()
    _y_out = []
    _grads = []
    x = next(iter(_data))
    embedded = model.embed_input(x[0].to(device))
    embedded.requires_grad = True
    _y = model.forward_embedded_softmax(embedded)
    _y_out.append(_y.cpu())
    _grads.append(torch.autograd.grad(_y.sum(), embedded, retain_graph=True)[0].data)

    print(_y_out[0])
    print(torch.argmax(_y_out[0], dim=1))
    print(_grads[0])