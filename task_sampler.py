from torch.utils.data.sampler import Sampler
from random import sample
from collections import Counter

class Task_Sampler(Sampler):
    """
    Custom sampler

    Arguments:
        way: number of unique classes
        shot: number of images per class
        labels: list of labels of all images

    Per each iteration returns indexes of images for the batch
    """
    def __init__(self, way, shot, labels):
        self.way = way
        self.shot = shot
        self.labels = labels
        self.unique_labels = set(self.labels)

        num_in_each_label = Counter(self.labels)

        if min(num_in_each_label.values()) < shot:
            raise AttributeError(f'The number of shots selected exceeds the number of images in some folders. Set "shot" to less than {self.shot}')

    def __iter__(self):

        task_labels = sample(self.unique_labels, self.way)
        batch = [sample([idx for idx, val in enumerate(self.labels) if self.labels[idx] == j ], self.shot) for j in task_labels]
        batch = [i for j in batch for i in j]
        print( batch)
        return iter(batch)


    def __len__(self):
        return self.way * self.shot
