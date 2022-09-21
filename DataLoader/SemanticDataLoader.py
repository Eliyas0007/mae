import os
import cv2
import time
import glob
import numpy
import torch
import random

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from einops import rearrange
from torch.utils.data import Dataset, DataLoader

# torch.manual_seed(1)
class SemanticDataset(Dataset):

    def __init__(self,
                root_dir, 
                num_classes,
                load_type,
                purpose,
                random_mask,
                masking_patch_size=16,
                masking_proportion=0.5,
                transforms=None,
                ) -> None:

        super().__init__()

        self.root_dir = root_dir
        self.transforms = transforms
        self.load_type = load_type
        self.purpose = purpose
        self.num_classes = num_classes
        
        self.random_mask = random_mask
        self.masking_patch_size = masking_patch_size
        self.masking_proportion = masking_proportion

        self._original_images_path = f'/home/yiliyasi/Downloads/Datasets/BDD-100k/original_datas/bdd100k_images_10k/bdd100k/images/10k/{self.purpose}/'
        self._total_image_pathes = os.listdir(root_dir)


    def __len__(self):
        return len(self._total_image_pathes)


    def __getitem__(self, index):
        
        image_folder = self._total_image_pathes[index]
        # print(image_folder)
        image_paths = glob.glob(self.root_dir + '/' + image_folder + '/*.png')
        c, h, w = rearrange(numpy.array(cv2.imread(image_paths[0])), 'h w c -> c h w').shape
        semantic_images = torch.zeros((self.num_classes, c, h, w))

        should_mask = random.randrange(0, 2)

        if self.load_type == 'color_map':
            '''
            NOT DONE!!!!!!!!!!!!!!!!!!
            '''

            semantic_images = self.transforms(semantic_images)
            original = self.transforms(original)
            return semantic_images.float(), original.float()

        elif self.load_type == 'actual_image':
            mask_images = numpy.zeros((self.num_classes, c, h, w))
            raw_image = cv2.imread(self._original_images_path + image_folder + '.jpg')


            original_image = rearrange(numpy.array(raw_image), 'h w c -> c h w') / 255
            for a_path in image_paths:
                raw_image = cv2.imread(a_path)
                tensor_image = rearrange(numpy.array(raw_image), 'h w c -> c h w') / 255
                image_name = a_path.split('/')[-1].split('.')[0]
                if image_name == 'buildings0':
                    mask_images[0] += tensor_image
                elif image_name == 'roads1':
                    mask_images[1] = tensor_image
                elif image_name == 'cars2':
                    mask_images[2] = tensor_image
                elif image_name == 'sky3':
                    mask_images[0] += tensor_image
                elif image_name == 'vegetations4':
                    mask_images[0] += tensor_image
                elif image_name == 'others5':
                    mask_images[0] += tensor_image

            if self.random_mask == True:
                if should_mask == 1:
                    original_image = numpy.zeros_like(original_image)
                    num_of_mask = random.randrange(0, self.num_classes)
                    indexes = random.sample(range(self.num_classes), num_of_mask)
                    for i_s in indexes:
                        mask_images[i_s] = numpy.zeros_like(mask_images[i_s])

                    for i_smtc in range(self.num_classes):
                        original_image += mask_images[i_smtc]

            mask_images = torch.from_numpy(mask_images)
            original_image = torch.from_numpy(original_image)

            mask_images = self.transforms(mask_images.float())
            original_image = self.transforms(original_image.float())
            mask_images = MaskRandomPatch(self.masking_patch_size, self.masking_proportion)(mask_images)
            # mask_images = AddGaussianNoise(0, 1.)(mask_images)
            return  mask_images, original_image
        else:
            raise 'load type not supported!'


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class MaskRandomPatch(object):
    def __init__(self, patch_size=0, percent=0.5):
        self.patch_size = patch_size
        self.percent = percent
        
    def __call__(self, tensor):
        c, h, w = tensor.shape
        total_patches = (h / self.patch_size)**2
        masked_patches = total_patches * self.percent

        randomlist = sorted(random.sample(range(0, int(total_patches)), int(masked_patches)))
        patch_index = 0
        shasha = 0
        for row in range(0, h, self.patch_size):
            # print(row)
            for col in range(0, w, self.patch_size):
                # print(row)
                for masking_index in randomlist:

                    if patch_index == masking_index:
                        shasha += 1
                        tensor[:, row:row+self.patch_size, col:col+self.patch_size] = 0
                        break
                patch_index += 1
        return tensor
    
    def __repr__(self):
        print('haha')
        return None


if __name__ == '__main__':
    path = '/home/yiliyasi/Downloads/Datasets/BDD-100k/customized_datas/colormap_masked_images_train'
    # another_path = '/home/yiliyasi/Downloads/Datasets/BDD-100k/customized_datas/masked_images_train'
    dataset = SemanticDataset(path, load_type='actual_image', num_classes=3, purpose='train', random_mask=False, transforms=transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    # transforms.RandomHorizontalFlip(),
                                    # AddGaussianNoise(0., 0.3)
                                    ]))

    dataset.__getitem__(0)

    start = time.time()
    loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, num_workers=1)
    train, target = next(iter(loader))
    print(train.shape, target.shape)
    end = time.time()
    print('time used: ', end - start)

    # semantics = yaml.safe_load(open('semantics.yaml', 'r'))
    num_of_sample = 0
    for i, image in enumerate(train[num_of_sample]):
        cv2.imwrite(f'./images/data{i}.png', rearrange(image.detach().cpu().numpy(), 'c h w -> h w c') * 255)
    
    cv2.imwrite('./images/target.png', rearrange(target[num_of_sample].detach().cpu().numpy(), 'c h w -> h w c') * 255)
