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
    def __init__(self, support_way, support_shot,  query_way, query_shot, labels):
        self.support_way = support_way
        self.support_shot = support_shot
        self.query_way = query_way
        self.query_shot = query_shot
        self.labels = labels
        self.unique_labels = set(self.labels)

        num_in_each_label = Counter(self.labels)

        if min(num_in_each_label.values()) < self.support_shot + self.query_shot:
            raise AttributeError(f'The number of shots selected exceeds the number of images in some folders. Set "shot" to less than {self.shot}')

    def __iter__(self):

        task_labels = sample(self.unique_labels, self.support_way)
        task_labels_query = sample(task_labels, self.query_way)
        batch_sup = [sample([idx for idx, val in enumerate(self.labels) if self.labels[idx] == j ], self.support_shot) for j in task_labels]
        batch_query = [sample([idx for idx, val in enumerate(self.labels) if self.labels[idx] == j ], self.query_shot) for j in task_labels_query]
        batch = [i for j in batch_sup+batch_query for i in j]
        return iter(batch)


    def __len__(self):
        return self.way * self.shot
