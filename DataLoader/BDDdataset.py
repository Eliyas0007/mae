import cv2
import glob
import torch
import numpy
import torchvision.transforms as transforms

from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from DataLoader.SemanticDataLoader import MaskRandomPatch

class BDDdataset(Dataset):
    
    def __init__(self, transforms=None):
        super().__init__()

        self.root_dir = '/home/yiliyasi/Downloads/Datasets/BDD-100k/original_datas/bdd100k_images_100k/bdd100k/images/100k/train'
        # self.root_dir = '/home/yiliyasi/Downloads/Datasets/celeba/celeba/img_align_celeba'
        self.transforms = transforms

        self.all_image_pathes = glob.glob(self.root_dir + '/*.jpg')

    def __len__(self):
        return len(self.all_image_pathes)

    def __getitem__(self, index):
        image_path = self.all_image_pathes[index]
        image = rearrange(numpy.array(cv2.imread(image_path)), 'h w c -> c h w') / 255

        image = torch.from_numpy(image)
        image = self.transforms(image)
        temp = torch.clone(image, memory_format=torch.preserve_format)
        image_masked = MaskRandomPatch(16, 0)(temp)
        
        return image_masked.float(), image.float() 

       
       
if __name__ == '__main__':
    dataset = BDDdataset(transforms=transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomVerticalFlip(),
    ]))
    loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=8)
    batch = next(iter(loader))
    print(batch.shape)