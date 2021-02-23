import torch


class Dataloader:
    """
    Dataloader class with methods for creating iterable dataloader objects for training and testing models.

    Attributes (and parameters to constructor):
        - full_training_set: torch dataset for re-training the model with full training set
        - training_set: torch dataset for training model
        - validation_set: torch dataset for validating model
        - test_set: torch dataset for testing model
        - batch_size: integer specifying the batch size
    Methods:
        - training_loader_full: returns dataloader object of full training set in batches, shuffled
        - training_loader: returna dataloader object in batches, shuffled
        - validation_laoder: returna dataloader object in batches, shuffled
        - test_loader: returns dataloader object in batches, not shuffled
        - unpack_full_test_set: returns dataloader object in a single batch covering the full test set, not shuffled

    """
    def __init__(self, full_training_set, training_set, validation_set, test_set, batch_size):
        self.full_training_set = full_training_set
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.batch_size = batch_size

    def training_loader_full(self):
        return torch.utils.data.DataLoader(dataset=self.full_training_set,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           drop_last=True)

    def training_loader(self):
        return torch.utils.data.DataLoader(dataset=self.training_set,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           drop_last=True)

    def validation_loader(self):
        return torch.utils.data.DataLoader(dataset=self.validation_set,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           drop_last=True)

    def test_loader(self):
        return torch.utils.data.DataLoader(dataset=self.test_set,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           drop_last=True)

    def unpack_full_test_set(self):
        return torch.utils.data.DataLoader(dataset=self.test_set,
                                           batch_size=len(self.test_set),
                                           shuffle=False)