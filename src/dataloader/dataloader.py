import torch


class Dataloader:
    """
    Dataloader class with methods for creating iterable dataloader objects for training and testing models.

    Attributes (and parameters to constructor):
        - training_set: torch dataset for training model
        - validation_set: torch dataset for validation model during trainign (estimating the test error)
        - test_set: torch dataset for testing model performance post training
        - training_set_full (not a parameter): full trainingset (train + val) for training the model
        - batch_size: integer specifying the batch size
    Methods:
        - training_loader_full
        - training_loader: returns a dataloader object in batches, shuffled
        - validation_loader: returns dataloader object in batches, not shuffled
        - test_loader: returns dataloader object in a single batch covering the full test set, not shuffled

    """
    def __init__(self, training_set, validation_set, test_set, batch_size):
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.training_set_full = torch.utils.data.ConcatDataset([self.training_set, self.validation_set])
        self.batch_size = batch_size

    def training_loader_full(self):
        return torch.utils.data.DataLoader(dataset=self.training_set_full,
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
                                           batch_size=len(self.test_set),
                                           shuffle=False)
