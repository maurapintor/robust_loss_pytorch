import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset


class H5Dataset(TorchDataset):

    def __init__(self, path=None, num_samples=None, indexes=None, classes=None):
        super(H5Dataset, self).__init__()

        self._file_path = path

        self._file = h5py.File(self._file_path, 'r')

        if indexes is not None:  # check first if indexes are passed
            sample_indexes = indexes
        elif num_samples is not None:
            sample_indexes = range(min(num_samples, self._file['X_data'].shape[0]))
            # shuffle
            sample_indexes = np.array(sample_indexes)
            np.random.shuffle(sample_indexes)
        else:
            sample_indexes = None

        y_data = self._file['y_data']

        if classes is not None:
            get_samples = np.zeros(self._file['X_data'].shape[0], dtype=bool)
            for c in classes:
                get_samples[y_data[:].argmax(1) == c] = True
            self._samples = self._file['X_data'][get_samples, ...].transpose(0, 3, 1, 2)
            self._labels = torch.argmax(torch.from_numpy(y_data[get_samples, ...]), dim=1).flatten().long()
        elif sample_indexes is None:
            self._samples = self._file['X_data'][:].transpose(0, 3, 1, 2)
            self._labels = torch.argmax(torch.from_numpy(y_data[:]), dim=1).flatten().long()
        else:
            get_samples = np.zeros(self._file['X_data'].shape[0], dtype=bool)
            get_samples[sample_indexes] = True
            self._samples = self._file['X_data'][get_samples, ...].transpose(0, 3, 1, 2)
            self._labels = torch.argmax(torch.from_numpy(y_data[get_samples, ...]), dim=1).flatten().long()


        # normalize samples
        self._samples = self._samples - self._samples.min()
        self._samples = self._samples / self._samples.max()

        self.classes = np.unique(self._labels)
        self._file.close()


    def __getitem__(self, index):
        input = self._samples[index, ...]
        return input.astype('float32'), self._labels[index]

    def __len__(self):
        return self._samples.shape[0]


class RobustLoss(nn.CrossEntropyLoss):
    def __init__(self, lambda_const):
        super(RobustLoss, self).__init__()
        self.lambda_const = lambda_const

    def forward(self, input, target, x):
        ce_loss = super(RobustLoss, self).forward(input, target)
        grad = torch.autograd.grad(ce_loss, x, create_graph=True)[0]
        grad_norm = torch.norm(grad, p=1)
        loss = ce_loss + self.lambda_const * grad_norm
        return loss

def robust_ce_loss(y_pred, y_true, **kwargs):
    """Computes the cross entropy loss, penalized by
    a term proportional to the norm of the input gradients.
    Using this loss during training, enforces the gradients
    w.r.t. the inputs to remain small, hence improving robustness.

    :param y_pred: output of the model
    :param y_true: ground truth labels
    :param kwargs: keyword arguments for loss. In order
        to compute the robust loss, at least x (input samples)
        and eps (penalty term for the loss) should be passed.
        If nothing is passed, the cross entropy will be
        returned.
    """
    ce_loss = torch.nn.functional.cross_entropy(y_pred, y_true)
    eps = kwargs.get('eps', 0)
    if eps == 0:
        return ce_loss
    x = kwargs.get('inputs', None)
    if x is None:
        raise ValueError("Robust loss requires the input data in "
                         "order to compute the input gradients.")
    grad = torch.autograd.grad(ce_loss, x, create_graph=True)[0]
    grad_norm = torch.norm(grad, p=2)
    loss = ce_loss + eps * grad_norm
    return loss

