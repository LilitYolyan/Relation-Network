from torch.utils.data.sampler import Sampler
from random import sample
from collections import Counter

class Task_Sampler(Sampler):
    """Samples few-shot learning tasks (episodes) from a labeled dataset.

    Each call to __iter__ constructs one episode by randomly selecting a
    subset of classes ("ways") and drawing support and query examples
    ("shots") from those classes, following the standard episodic
    training setup used in few-shot learning (e.g. Relation Networks,
    Prototypical Networks).

    Args:
        support_way (int): Number of unique classes to sample for the
            support set in each episode.
        support_shot (int): Number of examples per class to draw for
            the support set.
        query_way (int): Number of classes (drawn from the sampled
            support classes) to use for the query set.
        query_shot (int): Number of examples per class to draw for
            the query set.
        labels (list): Labels for every image/example in the dataset,
            indexed the same way as the underlying dataset.

    Raises:
        AttributeError: If any class has fewer examples than
            support_shot + query_shot, since there wouldn't be enough
            samples to build a full episode for that class.
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
