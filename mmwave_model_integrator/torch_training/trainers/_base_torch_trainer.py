from mmwave_model_integrator.torch_training.datasets import _BaseTorchDataset
from torch.nn import Module
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os

import mmwave_model_integrator.torch_training.models as models
import mmwave_model_integrator.torch_training.optimizers as optimizers
import mmwave_model_integrator.torch_training.transforms as transforms
import mmwave_model_integrator.torch_training.loss_fns as loss_fns
import mmwave_model_integrator.torch_training.datasets as datasets
import mmwave_model_integrator.torch_training.data_loaders as data_loaders

class _BaseTorchTrainer:

    def __init__(self,
                 model:dict,
                 optimizer:dict,
                 dataset:dict,
                 data_loader:dict,
                 dataset_path:str,
                 input_directory:str,
                 output_directory:str,
                 val_split:float,
                 working_dir:str,
                 save_name:str,
                 loss_fn:dict,
                 epochs = 40,
                 pretrained_state_dict_path:str = None,
                 cuda_device:str = "cuda:0",
                 multiple_GPUs = False):

        #determine if device is cuda and/or multiple gpus
        self.cuda_device = None
        self.multiple_GPUs = multiple_GPUs
        self._init_cuda_device(cuda_device)

        #initialize the model
        self.model:Module = None
        self.pretrained_state_dict_path:str = None
        self.optimizer:torch.optim.Optimizer = None
        self.pretrained_state_dict_path = pretrained_state_dict_path
        self._init_model(model_config=model,optimizer_config=optimizer)
        
        #initialize the loss function
        self.loss_fn = None
        self._init_loss_fn(loss_fn_config=loss_fn)

        #initialize the dataset paths        
        self.dataset_path = dataset_path
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.dataset:_BaseTorchDataset = None
        self.val_split = val_split
        self.train_dataset:_BaseTorchDataset = None
        self.val_dataset:_BaseTorchDataset = None
        self._init_datasets(dataset)

        #initialize dataloaders
        self.train_data_loader:DataLoader = None
        self.val_data_loader:DataLoader = None
        self.pin_memory = True if self.cuda_device == "cuda" else False 
        #determine if pinning memory during training
        self._init_data_loaders(data_loader)

        #save the path to the working directory
        self.working_dir = working_dir
        self.save_name = save_name        
        self._check_for_directory(path=working_dir,clear_contents=False)
        
        #batch size, train/test steps, train/test loss history
        self.epochs = epochs
        self.train_steps = 0
        self.test_steps = 0
        self.history = {"train_loss":[],"val_loss":[]} #to store train/test loss history        

    def _init_cuda_device(self,cuda_device:str):
        """Initialize the cuda device

        Args:
            cuda_device (str): the cuda device to use for training
        """
        # determine the device to be used for training and evaluation
        if torch.cuda.is_available():

            self.cuda_device = cuda_device
            torch.cuda.set_device(self.cuda_device)
        else:
            self.cuda_device = "cpu"
        

    def _init_model(self,model_config:dict,optimizer_config:dict):
        """Initialize the model, pretrained weights, optimizer, and send it
        to the cuda device

        Args:
            model (dict): _description_
            optimizer (dict): _description_
        """
        #determine the model
        model_class = getattr(models,model_config['type'])
        model_config.pop('type')
        self.model = model_class(**model_config)

        #configure for multiple GPUs if set
        if self.multiple_GPUs and torch.cuda.is_available() and \
            (torch.cuda.device_count() > 1):

            self.model = nn.DataParallel(self.model)
            print("Trainer._init_model: using {} GPUs".format(
                torch.cuda.device_count()))
        
        #load pretrained weights if available
        if self.pretrained_state_dict_path:
            if self.cuda_device != 'cpu':
                self.model.load_state_dict(
                    torch.load(self.pretrained_state_dict_path,
                               weights_only=True)
                )
            else:
                self.model.load_state_dict(
                    torch.load(self.pretrained_state_dict_path,
                               weights_only=True,
                               map_location='cpu')
                )

        #send the model to the cuda device
        self.model.to(self.cuda_device)

        #set the optimizer
        optimizer_class = getattr(optimizers,optimizer_config['type'])
        optimizer_config.pop('type')
        optimizer_config["params"] = self.model.parameters()
        self.optimizer = optimizer_class(**optimizer_config)
    
    def _init_loss_fn(self,loss_fn_config:dict):

        loss_fn_class = getattr(loss_fns,loss_fn_config['type'])
        loss_fn_config.pop('type')
        self.loss_fn = loss_fn_class(**loss_fn_config)

    def _init_datasets(self,dataset:dict):

        #get a list of the input and output files
        input_files = sorted(os.listdir(os.path.join(self.dataset_path,self.input_directory)))
        output_files = sorted(os.listdir(os.path.join(self.dataset_path,self.output_directory)))

        #get the full paths
        input_paths = [os.path.join(
            self.dataset_path,self.input_directory,file) for \
                file in input_files]
        output_paths = [os.path.join(
            self.dataset_path,self.output_directory,file) for \
                file in output_files]
        
        #obtain the train/val split
        train_inputs,val_inputs,train_outputs,val_outputs = \
            train_test_split(input_paths,output_paths,
                             test_size=self.val_split,
                             shuffle=True)
        
        #get the dataset type
        dataset_class = getattr(datasets,dataset['type'])
        dataset.pop('type')

        #initialize the train/val dataset
        train_dataset_config = dataset.copy()
        train_dataset_config["input_paths"] = train_inputs
        train_dataset_config["output_paths"] = train_outputs
        self.train_dataset = dataset_class(**train_dataset_config)

        val_dataset_config = dataset.copy()
        val_dataset_config["input_paths"] = val_inputs
        val_dataset_config["output_paths"] = val_outputs
        self.val_dataset = dataset_class(**val_dataset_config)

        print("ModelTrainer: {} train, {} val samples loaded".format(
            len(self.train_dataset),len(self.val_dataset)
        ))
        
        return
    
    def _init_data_loaders(self,data_loader_config:dict):

        data_loader_class = getattr(data_loaders,data_loader_config['type'])
        data_loader_config.pop('type')
        data_loader_config['pin_memory'] = self.pin_memory

        train_data_loader_config = data_loader_config.copy()
        train_data_loader_config['dataset'] = self.train_dataset
        self.train_data_loader = data_loader_class(**train_data_loader_config)

        val_data_loader_config = data_loader_config.copy()
        val_data_loader_config['dataset'] = self.val_dataset
        self.val_data_loader = data_loader_class(**val_data_loader_config)

    def _check_for_directory(self,path, clear_contents = False):
        """Checks to see if a directory exists, 
        if the directory does not exist, attepts to create the directory.
        If it does exist, optionally removes all files

        Args:
            path (str): path to the directory to create
            clear_contents (bool, optional): removes all contents in the directory on True. Defaults to False.
        """

        if os.path.isdir(path):
            print("_BaseTorchTrainer._check_for_directory: found directory {}".format(path))

            if clear_contents:
                print("_BaseTorchTrainer._check_for_directory: clearing contents of {}".format(path))

                #clear the contents
                for file in os.listdir(path):
                    file_path = os.path.join(path,file)

                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print("Failed to delete {}".format(path))
        else:
            print("_BaseTorchTrainer._check_for_directory: creating directory {}".format(path))
            os.makedirs(path)
        return
    
    def train_model(self):

        print("ModelTrainer.train: training the network...")
        start_time = time.time()

        #initialize train and test steps
        self.train_steps = len(self.train_dataset) // self.train_data_loader.batch_size
        self.test_steps = len(self.val_dataset) // self.val_data_loader.batch_size

        for epoch in (tqdm(range(self.epochs))):
            
            #put model into training mode
            self.model.train()

            #initialize total training and validation loss
            total_train_loss = 0
            total_val_loss = 0
            
            for (x,y) in self.train_data_loader:

                #send the input to the device
                (x,y) = (x.to(self.cuda_device),y.to(self.cuda_device))

                #perform forward pass and calculate training loss
                pred = self.model(x)
                loss = self.loss_fn(pred,y)
                #zero out any previously accumulated gradients, perform back propagation, update model parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                #add the loss to the total loss
                total_train_loss += loss

            #switch off autograd
            with torch.no_grad():

                #set model in evaluation mode
                self.model.eval()

                #loop over validation set
                for (x,y) in self.val_data_loader:

                    (x,y) = (x.to(self.cuda_device),(y.to(self.cuda_device)))

                    #perform forward pass and calculate training loss
                    pred = self.model(x)
                    loss = self.loss_fn(pred,y)

                    total_val_loss += loss
            
            avg_train_loss = total_train_loss / self.train_steps
            avg_val_loss = total_val_loss / self.test_steps

            #update training history
            self.history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
            self.history["val_loss"].append(avg_val_loss.cpu().detach().numpy())

            print("EPOCH: {}/{}".format(epoch + 1, self.epochs))
            print("\t Train loss: {}, Val loss:{}".format(avg_train_loss,avg_val_loss))

            #save the model
            file_name = "{}.pth".format(self.save_name)

            #save the state dict
            if self.multiple_GPUs and torch.cuda.is_available() and (torch.cuda.device_count() > 1):
                torch.save(self.model.module.state_dict(),os.path.join(self.working_dir,file_name))
            else:
                torch.save(self.model.state_dict(), os.path.join(self.working_dir,file_name))

        end_time = time.time()
        print("ModelTrainer.train: total training time {:.2f}".format(end_time - start_time))
        
        #plot the results
        self.plot_results()


    def plot_results(self):

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(self.history["train_loss"], label="train_loss")
        plt.plot(self.history["val_loss"], label="val_loss")
        plt.title("Training Loss on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")
        file_name = "{}.png".format(self.save_name)
        plt.savefig(os.path.join(self.working_dir,file_name))