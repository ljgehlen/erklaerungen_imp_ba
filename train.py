import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


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


def validate(inference_fn, model, x, y, loss_fn):

    if inference_fn is None:
        inference_fn = model

    model.eval()
    device = next(model.parameters()).device

    out = _get_outputs(inference_fn, x, model, device)
    _y_pred = torch.argmax(out, dim=1)
    loss = loss_fn(out, y)
    model.train()

    acc = torch.mean((y == _y_pred).to(torch.float)).detach().cpu().item()  # mean expects float, not bool (or int)
    return acc, loss


def train(model, optim, loss_fn, tr_data: DataLoader, te_data: DataLoader, inference_fn=None, device='cuda'):
    model.to(device)
    test_acc_val = []
    test_loss_val = []
    train_loss = 0.0
    _epochs = 0
    for text, labels in tqdm(tr_data):
        text = text.to(device)
        labels = labels.to(device)
        out = model(text)
        loss = loss_fn(out, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.item()

    for text, labels in te_data:
        test_acc, test_loss = validate(inference_fn, model, text, labels, loss_fn)
        test_acc_val.append(test_acc)
        test_loss_val.append(test_loss)
    acc = torch.mean(torch.tensor(test_acc_val))
    test_loss = torch.mean(torch.tensor(test_loss_val))
    average_train_loss = train_loss / len(tr_data)
    return model, acc.item(), test_loss.item(), average_train_loss
