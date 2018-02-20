import numpy as np
import torch
import yaml
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from txt2image_dataset import Text2ImageDataset
from models.gan_factory import gan_factory
from utils import Utils, Logger
from PIL import Image
import os

class Validation(object):
    def __init__(self, type, dataset, vis_screen, pre_trained_gen, batch_size, num_workers, epochs):
        with open('config.yaml', 'r') as f:
            config = yaml.load(f)

        self.generator = torch.nn.DataParallel(gan_factory.generator_factory(type).cuda())

        if pre_trained_gen:
            self.generator.load_state_dict(torch.load(pre_trained_gen))
        else:
            self.generator.apply(Utils.weights_init)

        if dataset == 'birds':
            self.dataset = Text2ImageDataset(config['birds_dataset_path'], split=split)
        elif dataset == 'flowers':
            self.dataset = Text2ImageDataset(config['flowers_val_dataset_path'], split=split)
        else:
            print('Dataset not supported, please select either birds or flowers.')
            exit()

        self.noise_dim = 100
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = epochs

        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers)

        self.logger = Logger(vis_screen)
        self.type = type

    def validate(self):
        for sample in data_loader:
            right_images = sample['right_images']
            right_embed = sample['right_embed']
            txt = sample['txt']


            if not os.path.exists('val_results/{0}'.format(self.save_path)):
                os.makedirs('val_results/{0}'.format(self.save_path))

            right_images = Variable(right_images.float()).cuda()
            right_embed = Variable(right_embed.float()).cuda()

            # Get generated images
            noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
            noise = noise.view(noise.size(0), 100, 1, 1)
            fake_images = self.generator(right_embed, noise)

            self.logger.draw(right_images, fake_images)

            for image, t in zip(fake_images, txt):
                im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                im.save('results/{0}/{1}.jpg'.format(self.save_path, t.replace("/", "")[:100]))
                print(t)

