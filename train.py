import os
import tqdm
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from cv2 import imwrite
from einops import rearrange
# from tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models_mae import MaskedAutoencoderViT
from DataLoader.BDDdataset import BDDdataset
from DataLoader.SemanticDataLoader import SemanticDataset, AddGaussianNoise


# save path for model
model_save_path = './savedModels/'
model_load_path = './savedModels/1_cl_real_p16_mae_b32_step250000_114.pth'

print('Initializing!')
# Hyperparameters
starting_epoch = 115
num_epochs = 5000
learning_rate = 0.001
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# models
image_size = 256
patch_size = 16
num_classes = 1
model = MaskedAutoencoderViT(img_size=image_size, patch_size=patch_size)
model.to(device)

model_save_name = f'{num_classes}_cl_real_p{patch_size}_mae_b{batch_size}'

try: os.mkdir(f'./performance_{model_save_name}')
except: ...

# tensorboard Hyperparameters
load_model = True
writer = SummaryWriter(f'performance_{model_save_name}/')
step = 250001
# netD = Discriminator(in_channels=3)
# netD.apply(weights_init)
# netD = netD.to(device)
 

# Optimizer
optimizer_vae = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer_dis = optim.Adam(netD.parameters(), lr=learning_rate)

# alternative loss function
criterion_ae = nn.MSELoss()
# criterion_dis = nn.BCELoss()

# Data

dataset = BDDdataset(transforms=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip()
]))


path = '/home/yiliyasi/Downloads/Datasets/BDD-100k/customized_datas/colormap_masked_images_train'
# another_path = '/home/yiliyasi/Downloads/Datasets/BDD-100k/customized_datas/masked_images_train'
# dataset = SemanticDataset(path, load_type='actual_image',
#                             num_classes=num_classes, purpose='train',
#                             random_mask=False,
#                             masking_patch_size=patch_size,
#                             masking_proportion=proportion,
#                             transforms=transforms.Compose([
#                                     transforms.Resize((256, 256)),
#                                     transforms.RandomHorizontalFlip()
#                                     ]))


loader =  DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)
print('Data Loaded!')
# b, cl, ch, h, w = next(iter(loader))[0].shape
# print(b, cl, ch, h, w)

if load_model:
    model.load_state_dict(torch.load(model_load_path))
    # netD.load_state_dict(torch.load('/home/yiliyasi/Documents/Projects/Mine/MTAE/savedModels/discriminator_step160000_91.pth'))
    print('state dict loaded!')

print('Start training!')
for epoch in range(starting_epoch, num_epochs):

    print(f'Epoch [current: {epoch} / total: {num_epochs}]')

    with tqdm.tqdm(loader, unit='batch') as tepoch:

        for (train, target) in tepoch:

            # if num_classes == 1:
            #     train = train.unsqueeze(1)
            
            # load datas to gpu
            train = train.to(device)
            target = target.to(device)
            # print(train.shape, target.shape)
            # train discriminator
            # netD.zero_grad()
            # pred_real = netD(target).view(-1)
            # loss_d_real = criterion_dis(pred_real, torch.ones_like(pred_real))
            # loss_d_real.backward()

            # pred_fake = netD(pred[1][0].detach()).view(-1)
            # loss_d_fake = criterion_dis(pred_fake, torch.zeros_like(pred_fake))
            # loss_d_fake.backward()
            # loss_d = loss_d_real + loss_d_fake
            # optimizer_dis.step()

            # train mbvae
            model.zero_grad()
            loss, pred, _ = model(train)
            # loss, recons_loss, kld_loss = MbVAE.total_loss_function(pred, target)
            # loss_recon = criterion_ae(pred, target)
            # d_output = netD(pred[1][0])
            # loss_dis = criterion_dis(d_output, torch.ones_like(d_output))
            # loss = loss_recon + loss_dis
            loss.backward()

            # this is for recurrent networks
            # nn.utils.clip_grad_norm_(MbVAE.parameters(), max_norm=1)  
            optimizer_vae.step()

            if step % 100 == 0:
                # try: os.mkdir(f'/home/yiliyasi/Documents/Projects/Mine/MTAE/performance_{model_save_name}')
                # except: ...
                # print(train[0].shape, pred[0].shape, target[0].shape)
                image = numpy.array(torch.cat([train[0], model.unpatchify(pred)[0], target[0]], dim=2).cpu().detach().numpy())
                imwrite(f'./performance_{model_save_name}/step{step}_{epoch}.png', rearrange(image * 255, 'c h w -> h w c'))

            # print(f'Training Loss : {loss.item()}')
            if step % 25000 == 0 and step != 0:                
                torch.save(model.state_dict(), model_save_path + f'{model_save_name}_step{step}_{epoch}.pth')

                # torch.save(netD.state_dict(), model_save_path + f'discriminator_step{step}_{epoch}.pth')

            writer.add_scalar('reconstruction Loss', loss.item(), global_step=step)
            # writer.add_scalar('KLD Loss', kld_loss.item(), global_step=step)
            # writer.add_scalar('generation Loss', loss_dis.item(), global_step=step)
            # writer.add_scalar('discriminator Loss', loss_d.item(), global_step=step)


            step += 1

        # torch.save(MbVAE.state_dict(), model_save_path + f'{model_save_name}_step{step}_{epoch}.pth')
        # torch.save(netD.state_dict(), model_save_path + f'{model_save_name}_dis_step{step}_{epoch}.pth')
