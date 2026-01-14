import os
import time
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from mmwave_model_integrator.torch_training.trainers._base_torch_trainer import _BaseTorchTrainer
import mmwave_model_integrator.torch_training.datasets as datasets



class GNNTorchTrainer(_BaseTorchTrainer):
    """Trainer class for Graph Neural Networks using PyTorch Geometric.

    This class extends _BaseTorchTrainer to handle GNN-specific data structures
    and training loops.
    """

    def __init__(self,
                 model: dict,
                 optimizer: dict,
                 dataset: dict,
                 data_loader: dict,
                 dataset_path: str,
                 node_directory: str,
                 label_directory: str,
                 val_split: float,
                 working_dir: str,
                 save_name: str,
                 loss_fn: dict,
                 epochs: int = 40,
                 pretrained_state_dict_path: str = None,
                 cuda_device: str = "cuda:0",
                 multiple_GPUs: bool = False) -> None:
        """Initializes the GNNTorchTrainer.

        Args:
            model (dict): Configuration dictionary for the GNN model.
            optimizer (dict): Configuration dictionary for the optimizer.
            dataset (dict): Configuration dictionary for the dataset.
            data_loader (dict): Configuration dictionary for the data loader.
            dataset_path (str): Path to the root directory of the dataset.
            node_directory (str): Subdirectory containing node features.
            label_directory (str): Subdirectory containing labels.
            val_split (float): Fraction of data to use for validation.
            working_dir (str): Directory to save model checkpoints and logs.
            save_name (str): Name used for saving the model.
            loss_fn (dict): Configuration dictionary for the loss function.
            epochs (int, optional): Number of training epochs. Defaults to 40.
            pretrained_state_dict_path (str, optional): Path to pretrained model weights. Defaults to None.
            cuda_device (str, optional): CUDA device to use. Defaults to "cuda:0".
            multiple_GPUs (bool, optional): Whether to use multiple GPUs if available. Defaults to False.
        """
        
        super().__init__(
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            data_loader=data_loader,
            dataset_path=dataset_path,
            input_directory=node_directory,
            output_directory=label_directory,
            val_split=val_split,
            working_dir=working_dir,
            save_name=save_name,
            loss_fn=loss_fn,
            epochs=epochs,
            pretrained_state_dict_path=pretrained_state_dict_path,
            cuda_device=cuda_device,
            multiple_GPUs=multiple_GPUs
        )
    
    def _init_datasets(self, dataset: dict) -> None:
        """Initializes the training and validation datasets.

        Args:
            dataset (dict): Configuration dictionary for the dataset.
        """

        #get a list of the input and output files
        input_files = sorted(os.listdir(os.path.join(self.dataset_path,self.input_directory)))
        output_files = sorted(os.listdir(os.path.join(self.dataset_path,self.output_directory)))

        #get the full paths
        node_paths = [os.path.join(
            self.dataset_path,self.input_directory,file) for \
                file in input_files]
        label_paths = [os.path.join(
            self.dataset_path,self.output_directory,file) for \
                file in output_files]
        
        #obtain the train/val split
        train_nodes,val_nodes,train_labels,val_labels = \
            train_test_split(node_paths,label_paths,
                             test_size=self.val_split,
                             shuffle=True)
        
        #get the dataset type
        dataset_class = getattr(datasets,dataset['type'])
        dataset.pop('type')

        #initialize the train/val dataset
        train_dataset_config = dataset.copy()
        train_dataset_config["node_paths"] = train_nodes
        train_dataset_config["label_paths"] = train_labels
        self.train_dataset = dataset_class(**train_dataset_config)

        val_dataset_config = dataset.copy()
        val_dataset_config["node_paths"] = val_nodes
        val_dataset_config["label_paths"] = val_labels
        self.val_dataset = dataset_class(**val_dataset_config)

        print("GNNModelTrainer: {} train, {} val samples loaded".format(
            len(self.train_dataset),len(self.val_dataset)
        ))
        
        return
    
    def train_model(self) -> None:
        """Trains the GNN model using the initialized datasets and parameters."""

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
            batch_num = 0

            for data in self.train_data_loader:
                batch_num += 1
                
                #make the prediction using the helper function
                pred, y = self.make_prediction(data)
                
                loss = self.loss_fn(pred.squeeze(),y) #may need to squeeze prediction

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
                for data in self.val_data_loader:
                    
                    #make the prediction using the helper function
                    pred, y = self.make_prediction(data)
                    
                    loss = self.loss_fn(pred.squeeze(),y) #may need to squeeze prediction

                    #add the loss to the total loss
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
        self.save_result_fig()
    
    def make_prediction(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Makes a prediction for a single batch of GNN data.

        Args:
            data (Data): PyTorch Geometric Data object containing node features, edge index, and labels.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing (prediction, ground_truth_labels).
        """

        #access the necessary data and move to device
        x = data.x.to(self.cuda_device)
        edge_index = data.edge_index.to(self.cuda_device)
        edge_attr = data.edge_attr.to(self.cuda_device)
        batch=data.batch.to(self.cuda_device)
        y = data.y.to(self.cuda_device)

        #make the prediction
        pred = self.model(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch)

        return pred, y


