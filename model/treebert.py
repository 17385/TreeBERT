import torch.nn as nn
import torch

class TreeBERT(nn.Module):
    def __init__(self, transformer, path_num, vocab_size):
        super().__init__()
        self.transformer = transformer
        self.mask_lm = TreeMaskedLanguageModel(self.transformer.hidden, vocab_size)
        self.node_order_prediction = NodeOrderPrediction(path_num, self.transformer.hidden)

    def forward(self, x, y):
        enc_output, x, _ = self.transformer(x, y)
        return self.node_order_prediction(enc_output), self.mask_lm(x)

class NodeOrderPrediction(nn.Module):
    """
    2-class classification model : order, disorder
    """

    def __init__(self, path_num, hidden):
        """
        :param hidden: TreeBERT encoder output size
        """
        super().__init__()
        self.linear_1 = nn.Linear(path_num*hidden, hidden)
        self.linear_2 = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = torch.reshape(x,(x.shape[0],-1))
        x = self.linear_1(x)
        return self.softmax(self.linear_2(x))

class TreeMaskedLanguageModel(nn.Module):
    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(x)
