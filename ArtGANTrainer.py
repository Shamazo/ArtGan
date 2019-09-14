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
from torch import autograd
import torchvision.transforms as transforms




class ArtGanTrainer(object):
    def __init__(self, _argv):
        self.init_parser(_argv)


        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # init dataloader
        trans = transforms.RandomHorizontalFlip(p=0.5)
        self.ds = ArtDataset(self.parser.data_dir, transform=trans)
        self.data_loader = DataLoader(self.ds, batch_size=self.parser.batch_size, shuffle=True, num_workers=8, drop_last=True)
        # self.data is an infinite iterator which will return the next batch and loop back to the start
        self.data = self.get_infinite_batches(self.data_loader)
    
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
        #value with wgan paper
        self.weight_cliping_limit = 0.02
        self.G_opt = optim.Adam(self.G.parameters(), lr=self.parser.G_lr, betas=(0.5, 0.9))
        self.D_opt = optim.Adam(self.D.parameters(), lr=self.parser.D_lr, betas=(0.5, 0.9))
        self.critic_iter = 10
        self.g_start_batch_num = 0
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

    def gradient_penalty(self, real_data, generated_data):
        # Calculate interpolation
        alpha = torch.rand(self.parser.batch_size, 1).to(self.device)
        alpha = alpha.expand_as(real_data)

        interpolated = alpha * real_data.data + (1-alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)

        D_interpolated, _ ,_ = self.D(interpolated)

        # Calculate gradients of of the discriminator wrt interpolated input
        gradients = autograd.grad(outputs=D_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(D_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # gradients have shape (batch_size, num_channels, img_width, img_height),
        # flatten for convenience 
        gradients = gradients.view(self.parser.batch_size, -1)

        # Very small gradients can be problematic so we use an epsilon hack
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-11)
        return self.parser.gradient_penalty_weight * ((gradients_norm - 1) ** 2).mean()

    def train(self):
        one = torch.FloatTensor([1]).to(self.device)
        mone = -1 * one

        running_g_loss = []
        running_d_loss_real = []
        running_d_loss_fake = []
        running_wasserstein_distance = []

        for g_batch_num in range(self.g_start_batch_num, self.parser.num_gen_batches):

            #train the discriminator/critic
            for p in self.D.parameters():
                p.requires_grad = True

            if g_batch_num <= 25 or g_batch_num % 500 == 0:
                crit_iters = 100
            else:
                crit_iters = self.critic_iter

            for d_iter in range(crit_iters):
                self.D.zero_grad()

                #train on real images
                real_images = Variable(next(self.data))
                # normalise to -1 to 1
                real_images = real_images.to(self.device).float() / 255.0
                real_images = 2*real_images - 1
                d_output_real, _, _ = self.D.forward(real_images)
                d_loss_real = torch.mean(d_output_real)
                d_loss_real.backward(one)

                #train on fake 
                noise = Variable(torch.ones(self.parser.batch_size, self.parser.lat_dim)).to(self.device)
                cont_code = Variable(torch.rand(self.parser.batch_size, self.parser.cont_dim)).to(self.device) 
                class_code = Variable(self.gen_class_code(self.parser.batch_size, self.parser.n_classes)).to(self.device)

                with torch.no_grad():
                    fake_images = self.G.forward(noise, cont_code, class_code).data
                d_output_fake, _, _ = self.D(fake_images)
                d_loss_fake = torch.mean(d_output_fake)
                d_loss_fake.backward(mone)

                #wgan-gp gradient penalty
                gradient_penalty = self.gradient_penalty(real_images.data, fake_images.data)
                gradient_penalty.backward()
                wasserstein_distance = d_loss_real - d_loss_fake
                # wasserstein_distance.backward()
                self.D_opt.step()

            #Train generator 
            #avoiding unnecessary computation 
            for p in self.D.parameters():
                p.requires_grad = False   

            self.G.zero_grad()

            # Train generator
            # Compute loss with fake images
            noise = Variable(torch.ones(self.parser.batch_size, self.parser.lat_dim)).to(self.device)
            cont_code = Variable(torch.rand(self.parser.batch_size, self.parser.cont_dim)).to(self.device) 
            class_code = self.gen_class_code(self.parser.batch_size, self.parser.n_classes).to(self.device)
            fake_images = self.G(noise, cont_code, class_code)

            g_loss, _, _ = self.D(fake_images)
            g_loss = torch.mean(g_loss)
            g_loss.backward(one)
            self.G_opt.step()

            running_g_loss.append(g_loss.to("cpu").detach().numpy())
            running_d_loss_real.append(d_loss_real.to("cpu").detach().numpy())
            running_d_loss_fake.append(d_loss_fake.to("cpu").detach().numpy())
            running_wasserstein_distance.append(wasserstein_distance.to("cpu").detach().numpy())

            if g_batch_num % 1000 == 0:
                # total_norm = 0
                # num_none = 0
                # parameters = list(filter(lambda p: p.grad is not None, self.D.parameters()))
                # for p in parameters:
                #     param_norm = p.grad.data.norm(2)
                #     total_norm += param_norm.item() ** 2

                #     num_none += 1
                # total_norm = total_norm ** (1. / 2)
                # print("Gradiant norm: {}, num not None: {}".format(total_norm, num_none))
                print("g batches: {} G loss: {}, D loss real: {}, D loss fake: {}, wasserstein_distance: {}, grad norm: {}"
                    .format(g_batch_num, np.mean(running_g_loss), np.mean(running_d_loss_real),
                            np.mean(running_d_loss_fake), np.mean(running_wasserstein_distance),
                            gradient_penalty / self.parser.gradient_penalty_weight))
                running_g_loss = []
                running_d_loss_real = []
                running_d_loss_fake = []
                running_wasserstein_distance = []
                self.sample_image(n_row=5, batches_done=g_batch_num)

            if g_batch_num % 1000 == 0:
                save_name = "ArtGan_weights_{}".format(g_batch_num)
                save_path = os.path.join(self.parser.save_dir, save_name)
                torch.save({
                    'g_batch_num': g_batch_num,
                    'D_state_dict': self.D.state_dict(),
                    'G_state_dict': self.G.state_dict(),
                    'D_optimizer_state_dict': self.D_opt.state_dict(),
                    'G_optimizer_state_dict': self.G_opt.state_dict(),
                }, save_path)

        # for epoch in range(self.parser.end_epoch - self.parser.start_epoch):
        #     for i, data in enumerate(self.data_loader):
        #         real_images = data.to(self.device).float() / 255.0
        #         real_images = real_images/real_images.sum(0).expand_as(real_images) 
        #         real_images[torch.isnan(real_images)]=0   #if an entire column is zero, division by 0 will cause NaNs
        #         real_images = 2*real_images - 1
        #         #======= Train D =======#
        #         noise = Variable(torch.ones(self.parser.batch_size, self.parser.lat_dim)).to(self.device) *0.1
        #         cont_code = Variable(torch.rand(self.parser.batch_size, self.parser.cont_dim)).to(self.device) 
        #         class_code = self.gen_class_code(self.parser.batch_size, self.parser.n_classes).to(self.device)

        #         fake_images = self.G.forward(noise, cont_code, class_code)
                
        #         d_output_real, _, _ = self.D.forward(real_images)
        #         d_output_real = torch.sigmoid(d_output_real)
        #         d_output_fake, d_output_class, d_output_cont = self.D.forward(fake_images.detach())
        #         d_output_fake = torch.sigmoid(d_output_fake)

        #         # print("d_output_real", d_output_real)
        #         # print("d_output_fake", d_output_fake)
        #         print( torch.mean(d_output_real))
        #         print(torch.mean(d_output_fake))
        #         d_loss_validity = self.adversarial_loss(d_output_real, torch.ones(self.parser.batch_size, 1)) + self.adversarial_loss(d_output_fake, torch.zeros(self.parser.batch_size, 1))
        #         # print(torch.log(d_output_real))
        #         # print(torch.log(1 - d_output_fake))
        #         # print(d_loss_validity.shape)
        #         # print(d_loss_validity)
        #         #information loss
        #         d_loss_cont = self.continuous_loss(d_output_cont, cont_code)
        #         d_loss_class = self.categorical_loss(d_output_class, class_code.argmax(dim=1))

        #         d_loss = d_loss_validity #+ d_loss_class * self.lambda_cat + d_loss_cont * self.lambda_cont

        #         self.D.zero_grad()
        #         d_loss.backward(retain_graph=True)
        #         self.D_opt.step()

                #======= Train G =======#
            #     d_output_fake, _, _= self.D.forward(fake_images)
            #     d_output_fake = torch.sigmoid(d_output_fake)
            #     g_loss = self.adversarial_loss(d_output_fake, torch.ones(self.parser.batch_size, 1)) #+ self.lambda_cont * d_loss_cont + self.lambda_cat * d_loss_class

            #     self.G.zero_grad()
            #     g_loss.backward()
            #     self.G_opt.step()

            #     print("G loss: {}, D loss: {}".format(g_loss, d_loss))
            # self.sample_image(n_row=5, batches_done=epoch)

    def to_categorical(self,y, num_columns):
        """Returns one-hot encoded Variable"""
        y_cat = np.zeros((y.shape[0], num_columns))
        y_cat[range(y.shape[0]), y] = 1.0

        return Variable(torch.tensor(y_cat))

    def sample_image(self, n_row, batches_done):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        static_sample = self.G.forward(self.static_noise, self.static_cont_code, self.static_class_code)
        save_image(static_sample.data, os.path.join(self.parser.save_dir, "images/%d.png" % batches_done), nrow=n_row, normalize=True)

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

    def get_infinite_batches(self, data_loader):
        while True:
            for i, data in enumerate(data_loader):
                yield data