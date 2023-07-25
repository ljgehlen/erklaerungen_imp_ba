import torch
from torch.utils.data import DataLoader, TensorDataset


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


def train(model, optim, loss_fn, tr_data: DataLoader, te_data: tuple, inference_fn=None, \
               n_batches_max=10, device='cuda'):
    model.to(device)
    acc_val = []
    n_batches = 0
    _epochs = 0
    while n_batches <= n_batches_max:
        for i, (text, labels) in enumerate(tr_data, 0):
            text = text.to(device)
            labels = labels.to(device)
            out = model(text)
            loss = loss_fn(out, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            n_batches += 1
            if n_batches > n_batches_max:
                break

    acc_val.append(validate(inference_fn, model, *te_data))
    print("accuracies over test set")
    print(acc_val)
    return model, acc_val