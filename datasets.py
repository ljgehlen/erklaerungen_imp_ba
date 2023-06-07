import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS
# ----------------------------------------------------------------------------------------------------------------------

DATA_ROOT = './datasets'

# ----------------------------------------------------------------------------------------------------------------------


# author: Sebastian

class TorchRandomSeed(object):
    """
    Class to be used when opening a with clause. On enter sets the random seed for torch based sampling, restores previous state on exit
    """
    def __init__(self, seed):
        self.seed = seed
        self.prev_random_state = None

    def __enter__(self):
        self.prev_random_state = torch.get_rng_state()
        torch.set_rng_state(torch.manual_seed(self.seed).get_state())

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.set_rng_state(self.prev_random_state)


def _get_vocab(name='AG_NEWS', train_iter=None):
    vocab_path = f"{DATA_ROOT}/{name}/vocab.torch"
    print(f"looking for vocab in {vocab_path}")
    try:
        vocab = torch.load(vocab_path)
    except FileNotFoundError:
        print("Vocab not found, building vocab ...")
        tokenizer = get_tokenizer('basic_english')
        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)
        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>', '<pad>'])
        vocab.set_default_index(vocab['<unk>'])
        torch.save(vocab, vocab_path)
        print(f"... done, saved vocab of {len(vocab)} words in {vocab_path}")
    return vocab


def _build_collate_fn(vocab, label_pipeline):
    """
    given a text dataset, returns preprocessing function as required for DataLoader
    :param train_iter: iterator over text dataset
    :return:
    """
    tokenizer = get_tokenizer('basic_english')
    padding_val = vocab['<pad>']

    def text_pipeline(input):
        return vocab(tokenizer(input))

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)

        labels = torch.tensor(label_list, dtype=torch.int64)
        text = pad_sequence(text_list, batch_first=True, padding_value=padding_val)
        return text, labels

    return collate_batch, len(vocab)


def get_agnews(random_state, batch_sizes=(64, 200), root=DATA_ROOT):
    """load AGNews dataset from torchtext"""
    # -, classes = 4
    def label_pipeline(input):
        return int(input) - 1

    with TorchRandomSeed(random_state):
        # base vocab in collate function on training data
        gen_train = torch.Generator(); gen_train.manual_seed(random_state)
        gen_test = torch.Generator(); gen_test.manual_seed(random_state)

        train_iter = AG_NEWS(root, split='train')
        train_iter = to_map_style_dataset(train_iter)

        test_iter = AG_NEWS(root, split='test')
        test_iter = to_map_style_dataset(test_iter)

        vocab = _get_vocab('AG_NEWS', train_iter)
        collate_batch, size_vocab = _build_collate_fn(vocab, label_pipeline)

        train_loader = DataLoader(train_iter, batch_size=batch_sizes[0],
                                  shuffle=True, collate_fn=collate_batch, generator=gen_train)
        test_loader = DataLoader(test_iter, batch_size=batch_sizes[-1], shuffle=True, collate_fn=collate_batch,
                                 generator=gen_test)

    return train_loader, test_loader, size_vocab, 4

