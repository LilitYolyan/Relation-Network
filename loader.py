from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from torchvision import datasets
from task_sampler import Task_Sampler
import os

class FSL_Dataset(Dataset):
    """
    Creates Dataset Object

    Arguments:
          image_size _ image size for training
          images_path _ path to root folder of images

    """
    def __init__(self, image_size, images_path):
        self.images_path = images_path
        self.transform = transforms.Compose([transforms.Resize(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.dataset_folders = datasets.ImageFolder(os.path.join(self.images_path), transform=self.transform)
        self.labels = [i[1] for i in self.dataset_folders.imgs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        returns list of two tensor: image and label
        """
        image = self.dataset_folders[index]
        return image



def data_loader( way,shot, images_size, images_path):
    """
    Arguments:
        way: number of classes
        shot: number of images per class
        images_size: size of image
        images_path: root path to images

    """
    dataset = FSL_Dataset(image_size=images_size, images_path=images_path)
    labels = dataset.labels
    sampler = Task_Sampler(way, shot, labels)
    loader = DataLoader(dataset, batch_size=1,  sampler = sampler)
    return loader

