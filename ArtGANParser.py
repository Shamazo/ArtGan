import argparse
import os
import inspect
import json
import copy

class ArtGanParser(object):
    def __init__(self):
            pass

    ##--------------------------------------------
    # DEF create_parser
    # - create argparse and return
    # --------------------------------------------
    def create_parser(self, desc=None):
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument('-pwp', '--pretrained_weights_path', default=None, type=str, help='pretrained weights path of the model, takes a folder if loading checkpoint, takes a *.pth string if specific weights')
        parser.add_argument('-sm', '--save_model', default=True, type=bool, help='save model [True/False]')
        parser.add_argument('-sd', '--save_dir', default="weights", help="Directory to save model to")
        parser.add_argument('-dd', '--data_dir', default="proc_moma", help='Date directory')
        parser.add_argument('-bs', '--batch_size', default=4, help="Training batch size")
        parser.add_argument('-Glr', '--gen_learning_rate', default=0.0002, help="Set initial learning rate for generator")
        parser.add_argument('-Dlr', '--disc_learning_rate', default=0.0002, help="Set initial learning rate for discriminator")
        parser.add_argument('-img_sz', '--image_size', default=128, help="Set image size of height/width, assuming square images")
        parser.add_argument('-lat_dim', '--latent_dimension', default=32, help="Set latent noise dimension")
        parser.add_argument('-cont_dim', '--cont_dimension', default=2, help="Set number of continuous variables")
        parser.add_argument('-n_classes', '--n_classes', default=2, help="Set number of classes")
        parser.add_argument('-starte', '--start_epoch', default=0, help="epoch to start training from")
        parser.add_argument('-ee', '--end_epoch', default=400, help="epoch to stop training on" )
        parser.add_argument('-mg', '--multi_gpu', default=False, help="Run on multiple GPUS")
        return parser

    def parse(self, argv, desc=None):
        parser = self.create_parser(desc)
        options = parser.parse_args(argv[1:])
        self.pretrained_weights_path = options.pretrained_weights_path
        self.data_dir = options.data_dir
        self.save_model = bool(options.save_model)
        self.save_dir = options.save_dir
        self.G_lr = options.gen_learning_rate
        self.D_lr = options.disc_learning_rate
        self.img_sz = options.image_size
        self.lat_dim = options.latent_dimension
        self.cont_dim = options.cont_dimension
        self.n_classes = options.n_classes
        self.batch_size = options.batch_size
        self.start_epoch = options.start_epoch
        self.end_epoch = options.end_epoch
        self.multi_gpu = options.multi_gpu

    def to_json(self, file_name):
        # make a copy
        new = copy.deepcopy(self)
        variables = [v for v in dir(new) if not callable(getattr(new,v))]
        attributes = inspect.getmembers(new, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if a[0] in variables]

        # write to disk
        with open(file_name, 'w') as f:
            json.dump(attributes, f)