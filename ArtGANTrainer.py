from ArtGANParser import ArtGanParser
from datasets import ArtDataset
from models import Generator, Discriminator
import os
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
import itertools
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import torch.nn.functional as F

class ArtGanTrainer(object):
    def __init__(self, _argv):
        self.init_parser(_argv)


        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # init dataloader
        self.ds = ArtDataset(self.parser.data_dir)
        self.data_loader = DataLoader(self.ds, batch_size=self.parser.batch_size, shuffle=True, num_workers=8)
    
        # Load and initialize the generator and discriminator models
        self.G = Generator(self.parser.lat_dim, self.parser.n_classes, self.parser.cont_dim, self.parser.img_sz)
        self.D = Discriminator(self.parser.lat_dim, self.parser.n_classes, self.parser.cont_dim, self.parser.img_sz)
        self.G.train()
        self.D.train()
        self.G.to(self.device)
        self.D.to(self.device)
        self.init_weights(self.G)
        self.init_weights(self.D)

        # optimizers 
        self.G_opt = optim.Adam(self.G.parameters(), lr=self.parser.G_lr, betas=[0.5, 0.999])
        self.D_opt = optim.Adam(self.D.parameters(), lr=self.parser.D_lr, betas=[0.5, 0.999])
        # self.G_scheduler = lr_scheduler.StepLR(self.G_opt, step_size=10, gamma=0.1, last_epoch=-1)
        # self.D_scheduler = lr_scheduler.StepLR(self.D_opt, step_size=10, gamma=0.1, last_epoch=-1)

        # loss functions and weights 
        self.adversarial_loss = torch.nn.BCELoss()
        self.categorical_loss = torch.nn.CrossEntropyLoss()
        self.continuous_loss = torch.nn.MSELoss()
        self.lambda_cat = 1
        self.lambda_cont = 0.1

        #static codes for sampling
        # Static sample
        self.static_noise = Variable(torch.randn(5**2, self.parser.lat_dim)).to(self.device)
        self.static_cont_code = Variable(torch.rand(5**2, self.parser.cont_dim)).to(self.device)
        self.static_class_code = self.gen_class_code(5**2, self.parser.n_classes).to(self.device)
                

    def init_parser(self, _argv):
        self.parser = ArtGanParser()
        self.parser.parse(_argv)

        if not os.path.exists(self.parser.save_dir): os.makedirs(self.parser.save_dir)
        jf = os.path.join(self.parser.save_dir, 'parameters.json')
        self.parser.to_json(file_name=jf)


    def init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def train(self):
        for epoch in range(self.parser.end_epoch - self.parser.start_epoch):
            for i, data in enumerate(self.data_loader):
                real_images = data.to(self.device).float() / 255.0
                real_images = real_images/real_images.sum(0).expand_as(real_images) 
                real_images[torch.isnan(real_images)]=0   #if an entire column is zero, division by 0 will cause NaNs
                real_images = 2*real_images - 1
                #======= Train D =======#
                noise = Variable(torch.ones(self.parser.batch_size, self.parser.lat_dim)).to(self.device) *0.1
                cont_code = Variable(torch.rand(self.parser.batch_size, self.parser.cont_dim)).to(self.device) 
                class_code = self.gen_class_code(self.parser.batch_size, self.parser.n_classes).to(self.device)

                fake_images = self.G.forward(noise, cont_code, class_code)
                
                d_output_real, _, _ = self.D.forward(real_images)
                d_output_real = torch.sigmoid(d_output_real)
                d_output_fake, d_output_class, d_output_cont = self.D.forward(fake_images.detach())
                d_output_fake = torch.sigmoid(d_output_fake)

                # print("d_output_real", d_output_real)
                # print("d_output_fake", d_output_fake)
                print( torch.mean(d_output_real))
                print(torch.mean(d_output_fake))
                d_loss_validity = self.adversarial_loss(d_output_real, torch.ones(self.parser.batch_size, 1)) + self.adversarial_loss(d_output_fake, torch.zeros(self.parser.batch_size, 1))
                # print(torch.log(d_output_real))
                # print(torch.log(1 - d_output_fake))
                # print(d_loss_validity.shape)
                # print(d_loss_validity)
                #information loss
                d_loss_cont = self.continuous_loss(d_output_cont, cont_code)
                d_loss_class = self.categorical_loss(d_output_class, class_code.argmax(dim=1))

                d_loss = d_loss_validity #+ d_loss_class * self.lambda_cat + d_loss_cont * self.lambda_cont

                self.D.zero_grad()
                d_loss.backward(retain_graph=True)
                self.D_opt.step()

                #======= Train G =======#
                d_output_fake, _, _= self.D.forward(fake_images)
                d_output_fake = torch.sigmoid(d_output_fake)
                g_loss = self.adversarial_loss(d_output_fake, torch.ones(self.parser.batch_size, 1)) #+ self.lambda_cont * d_loss_cont + self.lambda_cat * d_loss_class

                self.G.zero_grad()
                g_loss.backward()
                self.G_opt.step()

                print("G loss: {}, D loss: {}".format(g_loss, d_loss))
            self.sample_image(n_row=5, batches_done=epoch)

    def to_categorical(self,y, num_columns):
        """Returns one-hot encoded Variable"""
        y_cat = np.zeros((y.shape[0], num_columns))
        y_cat[range(y.shape[0]), y] = 1.0

        return Variable(torch.tensor(y_cat))

    def sample_image(self, n_row, batches_done):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        static_sample = self.G.forward(self.static_noise, self.static_cont_code, self.static_class_code)
        save_image(static_sample.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

        # Get varied c1 and c2
        # zeros = np.zeros((n_row ** 2, 1))
        # c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
        # c1 = Variable(torch.tensor(np.concatenate((c_varied, zeros), -1)))
        # c2 = Variable(torch.tensor(np.concatenate((zeros, c_varied), -1)))
        # sample1 = generator(static_z, static_label, c1)
        # sample2 = generator(static_z, static_label, c2)
        # save_image(sample1.data, "images/c1%d.png" % batches_done, nrow=n_row, normalize=True)
        # save_image(sample2.data, "images/c2%d.png" % batches_done, nrow=n_row, normalize=True)

    # Generate random class codes
    def gen_class_code(self, size, dim):
        codes=[]
        code = np.zeros((size, dim))
        random_cat = np.random.randint(0, dim, size)
        code[range(size), random_cat] = 1
        codes.append(code)
        codes = np.concatenate(codes,1)
        return torch.Tensor(codes)