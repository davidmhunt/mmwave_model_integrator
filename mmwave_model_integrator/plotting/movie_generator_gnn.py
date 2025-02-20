import tqdm
import numpy as np

from cpsl_datasets.gnn_node_ds import GnnNodeDS

from mmwave_model_integrator.plotting._movie_generator import _MovieGenerator
from mmwave_model_integrator.input_encoders._node_encoder import _NodeEncoder
from mmwave_model_integrator.ground_truth_encoders._gt_node_encoder import _GTNodeEncoder
from mmwave_model_integrator.plotting.plotter_gnn_pc_processing import PlotterGnnPCProcessing
from mmwave_model_integrator.model_runner.gnn_runner import GNNRunner
from mmwave_model_integrator.decoders._decoder import _Decoder

class MovieGeneratorGNN(_MovieGenerator):

    def __init__(
            self, 
            gnn_ds:GnnNodeDS,
            plotter:PlotterGnnPCProcessing,
            input_encoder:_NodeEncoder,
            runner:GNNRunner = None,
            decoder:_Decoder = None,
            ground_truth_encoder:_GTNodeEncoder = None,
            temp_dir_path="~/Downloads/odometry_temp"):
        
        self.dataset:GnnNodeDS = gnn_ds

        super().__init__(
            plotter,
            input_encoder,
            runner,
            decoder,
            ground_truth_encoder,
            temp_dir_path)
    
    def reset(self):
        """Reset the movie generator
        """
        
        super().reset()

    def generate_movie_frames(self):

        #prime the dataset to ensure that the encoders and decoders have sufficient data 
        #with an encoding ready to go
        start_idx=0
        self.input_encoder.reset()

        if self.ground_truth_encoder:
            while (not self.input_encoder.full_encoding_ready) or \
                (not self.ground_truth_encoder.full_encoding_ready):

                #get the node data
                nodes = self.dataset.get_node_data(idx=start_idx)
                nodes_encoded = self.input_encoder.encode(nodes)

                #get the labeled data
                labels = self.dataset.get_label_data(idx=start_idx)
                gt_encoding = self.ground_truth_encoder.encode(labels)


                start_idx += 1
        else:
            while (not self.input_encoder.full_encoding_ready):

                #get the node data
                nodes = self.dataset.get_node_data(idx=start_idx)
                nodes_encoded = self.input_encoder.encode(nodes)

                start_idx += 1
        
        for i in tqdm.tqdm(range(start_idx,self.dataset.num_frames)):

            #get the node data
            nodes = self.dataset.get_node_data(idx=i)

            if self.ground_truth_encoder:
                labels = self.dataset.get_label_data(idx=i)
            else:
                labels = np.empty(shape=(0))
            
            self.plotter.plot_compilation(
                input_data=nodes,
                input_encoder=self.input_encoder,
                runner=self.runner,
                decoder=self.decoder,
                gt_data=labels,
                ground_truth_encoder=self.ground_truth_encoder,
                axs=self.axs,
                show=False
            )

            #save the frame
            self.save_frame(clear_axs=True) 