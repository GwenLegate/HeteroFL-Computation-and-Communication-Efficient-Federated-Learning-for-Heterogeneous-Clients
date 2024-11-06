import numpy as np
import torch
from torch.utils.data import DataLoader

def dirichlet_split_dataset(ds, num_clients, alpha, num_classes):
    ''' Splits the dataset between num_clients clients and further partitions each clients subset into training
        and test sets
    Args:
        ds: the complete training dataset
        num_clients: number of clients to split between
        alpha: parameter of the dirichlet distribution
        num_classes: number of classes in ds
    Returns:
        data_partition: a nested dict where keys are 'train' and 'test' and each train and test is a dict with keys 0 to
        num_clients which contain the corresponding sample indices for training and testing subsets of each clients partition.
        label_set: a list of lists containing the labels present at each client
    '''
    print(f'Creating non iid client datasets using a Dirichlet distribution')
    try:
        labels = np.array(ds.target)
    except AttributeError:
        labels = np.array(ds.labels)
    dict_partition = {}
    dict_partition['train'] = {}
    dict_partition['test'] = {}
    label_set = []
    multinomial_vals = []
    examples_per_label = []
    for i in range(num_classes):
        examples_per_label.append(int(np.argwhere(labels == i).shape[0]))

    # Each client has a multinomial distribution over classes drawn from a Dirichlet.
    for i in range(num_clients):
        proportion = np.random.dirichlet(alpha * np.ones(num_classes))
        multinomial_vals.append(proportion)

    multinomial_vals = np.array(multinomial_vals)
    example_indices = []

    for k in range(num_classes):
        label_k = np.where(labels == k)[0]
        np.random.shuffle(label_k)
        example_indices.append(label_k)

    example_indices = np.array(example_indices, dtype=object)

    client_samples = [[] for _ in range(num_clients)]
    count = np.zeros(num_classes).astype(int)

    examples_per_client = int(labels.shape[0] / num_clients)

    for k in range(num_clients):
        for i in range(examples_per_client):
            sampled_label = np.argwhere(np.random.multinomial(1, multinomial_vals[k, :]) == 1)[0][0]
            label_indices = example_indices[sampled_label]
            client_samples[k].append(label_indices[count[sampled_label]])
            count[sampled_label] += 1
            if count[sampled_label] == examples_per_label[sampled_label]:
                multinomial_vals[:, sampled_label] = 0
                multinomial_vals = (
                        multinomial_vals /
                        multinomial_vals.sum(axis=1)[:, None])
    for i in range(num_clients):
        np.random.shuffle(np.array(client_samples[i]))
        # create 90/10 train/validation split for each client
        samples = np.array(client_samples[i])
        train_idxs = samples[:int(samples.shape[0] * 0.9)].astype('int64').squeeze()
        validation_idxs = samples[int(samples.shape[0] * 0.9):].astype('int64').squeeze()

        dict_partition['train'][i] = list(train_idxs)
        dict_partition['test'][i] = list(validation_idxs)
        label_set.append(get_client_labels(ds, dict_partition['train'][i], dict_partition['test'][i]))

    return dict_partition, label_set

def get_client_labels(dataset, train_idxs, test_idxs):
    """
    Creates a set of all labels present in both train and validation sets of a client dataset
    Args:
        dataset: the complete dataset being used
        train_idxs: the indices of the training samples for the client
        test_idxs: the indices of the validation samples for the client
        num_workers: how many sub processes to use for data loading
    Returns: Set of all labels present in both train and validation sets of a client dataset.
        """
    all_idxs = np.concatenate((train_idxs, test_idxs), axis=0)
    try:
        labels = [dataset[i]['label'].item() for i in all_idxs]
    except AttributeError:
        labels = [dataset[i]['label'] for i in all_idxs]
    return list(set(labels))
