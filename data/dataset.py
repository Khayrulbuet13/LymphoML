import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from LymphoMNIST.LymphoMNIST import LymphoMNIST

class FilteredLymphoMNIST(Dataset):
    """Filters the dataset to keep only specified labels."""
    def __init__(self, original_dataset, labels_to_keep):
        self.original_dataset = original_dataset
        self.labels_to_keep = labels_to_keep
        self.label_map = {label: i for i, label in enumerate(labels_to_keep)}
        self.indices = [i for i, (_, label) in enumerate(original_dataset) if label in labels_to_keep]

    def __getitem__(self, index):
        original_index = self.indices[index]
        image, label = self.original_dataset[original_index]
        # Map the original label to the new label index if needed
        if hasattr(label, 'item'):
            mapped_label = self.label_map[label.item()]
            return image, torch.tensor(mapped_label)
        else:
            return image, label

    def __len__(self):
        return len(self.indices)

def balanced_weights(dataset, nclasses):
    """Calculate balanced weights for weighted sampling."""
    count = [0] * nclasses
    for _, label in dataset:
        count[label] += 1
    N = float(sum(count))
    weight_per_class = [N / float(count[i]) for i in range(nclasses)]
    return [weight_per_class[label] for _, label in dataset]

def get_dataloaders(train_ds, val_ds, batch_size=64, split=(0.5, 0.5), sampler=None, seed=42, **kwargs):
    """Create train, validation, and test dataloaders."""
    lengths = [int(len(val_ds) * frac) for frac in split]
    lengths[1] += len(val_ds) - sum(lengths)  # Correct split length sum

    generator = torch.Generator().manual_seed(seed)
    val_ds, test_ds = torch.utils.data.random_split(val_ds, lengths, generator=generator)

    shuffle = False if sampler else True
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, **kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kwargs)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kwargs)

    return train_dl, val_dl, test_dl
