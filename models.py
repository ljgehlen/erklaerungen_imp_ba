import torch
from torch import nn
from torch.nn import functional as F


class BiLSTMClassif(nn.Module):

    def __init__(self, n_classes, embed_dim, hid_size=128, vocab_size=None):
        super(BiLSTMClassif, self).__init__()
        if vocab_size is None:
            raise ValueError('vocab size cannot be empty')
        # sparse limits available optimizers to SGD, SparseAdam and Adagrad
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hid_size,
            bidirectional=True,
            batch_first=True,
        )
        self.fc_out = nn.Linear(2 * hid_size, n_classes)

    def get_embeddings_variance(self):
        return torch.var(self.embedding.weight).item()

    def embed_input(self, seqs):
        with torch.no_grad():
            embedded = self.embedding(seqs)
        return embedded

    def forward(self, seqs):  # , offsets):
        embedded = self.embedding(seqs)  # , offsets)
        return self.forward_embedded(embedded)

    def forward_softmax(self, seqs):
        embedded = self.embedding(seqs)
        return self.forward_embedded_softmax(embedded)

    def forward_embedded(self, embedded_seq):
        lstm_out, (_, _) = self.bilstm(embedded_seq)
        h_T = lstm_out[:, -1]
        y = self.fc_out(h_T)
        return y

    def forward_embedded_softmax(self, embedded_seq):
        x = self.forward_embedded(embedded_seq)
        y = torch.softmax(x, dim=1)
        return y


class SentenceCNN(nn.Module):
    '''
    Implementation based on/ close to
        >> Convolutional Neural Networks for Sentence Classification
        >> https://arxiv.org/pdf/1408.5882.pdf
    '''

    def __init__(self, n_classes, embed_dim, vocab_size, pre_trained_weights=None, freeze=False):
        super(SentenceCNN, self).__init__()

        if n_classes == 2:
            n_classes = 1

        kernel_widths = [3, 4, 5]
        out_channels = 100
        self.n_classes = n_classes
        self.embed_dim = embed_dim

        if pre_trained_weights is not None:
            print("use provided embeddings")
            self.embedding = nn.Embedding.from_pretrained(pre_trained_weights, freeze=freeze, sparse=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False)

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kw, embed_dim),
                                              padding_mode='zeros', dtype=torch.float)
                                    for kw in kernel_widths])
        self.pooling = F.max_pool1d
        self.fc = nn.Linear(in_features=len(kernel_widths) * out_channels,
                            out_features=n_classes)
        self.dropout = nn.Dropout()
        self.act = F.relu
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        embed_seq = self.embedding(x)
        return self.forward_embedded(embed_seq)

    def forward_softmax(self, x):
        _y_lin = self.forward(x)
        y = torch.softmax(_y_lin, dim=1)
        return y

    def embed_input(self, seqs):
        with torch.no_grad():
            embedded = self.embedding(seqs)
        return embedded

    def forward_embedded(self, embedded_seq):
        embed_seq = embedded_seq.unsqueeze(1)  # add "channel" as expected by conv
        convolved = [self.act(conv(embed_seq)).squeeze(3) for conv in self.convs]
        lin_in = torch.cat([self.pooling(conv, conv.size(2)) for conv in convolved], 1).float().squeeze(-1)
        lin_in = self.dropout(lin_in)
        out = self.fc(lin_in)
        out = self.act_out(out)
        return out

    def forward_embedded_softmax(self, embedded_seq):
        x = self.forward_embedded(embedded_seq)
        y = torch.softmax(x, dim=1)
        return y
