import matplotlib.pyplot as plt
import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_model_integrator.input_encoders.hermes_encoder import HermesEncoder
from mmwave_model_integrator.decoders.heremes_decoder import HermesDecoder
from mmwave_model_integrator.model_runner._model_runner import _ModelRunner
from mmwave_model_integrator.ground_truth_encoders.hermes_gt_encoder import HermesGTEncoder
from mmwave_model_integrator.transforms.coordinate_transforms import polar_to_cartesian
from mmwave_model_integrator.plotting._plotter import _Plotter
from mmwave_radar_processing.supportFns.rotation_functions import apply_rot_trans


from mmwave_radar_processing.processors.synthetic_array_beamformer_processor_revA import SyntheticArrayBeamformerProcessor

class PlotterHermes(_Plotter):

    def __init__(self):

        super().__init__()
    
    ####################################################################
    #Plotting grid encodings
    ####################################################################
    def plot_input_encoding(
            self,
            encoding,
            hermes_encoder:HermesEncoder,
            cmap="viridis",
            ax:plt.Axes=None,
            show=False
    ):
        """Plot the input encoding from the HermesEncoder

        Args:
            encoding (np.ndarray): encoded input from the HermesEncoder
            hermes_encoder (HermesEncoder): HermesEncoder object
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()

        x_s = hermes_encoder.interp_x_s
        y_s = hermes_encoder.interp_y_s

        ax.pcolormesh(
            y_s,
            x_s,
            encoding,
            shading='gouraud',
            cmap=cmap
        )
        
        ax.set_xlabel("Y (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("X (m)",fontsize=self.font_size_axis_labels)
        ax.set_title("Model Input",fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)
        ax.invert_xaxis()

        if show:
            plt.show()
    
    def plot_gt_encoding(
            self,
            gt_encoding,
            hermes_gt_encoder:HermesGTEncoder,
            cmap="viridis",
            ax:plt.Axes=None,
            show=False
    ):
        """Plot the ground truth encoding from the HermesGTEncoder

        Args:
            gt_encoding (np.ndarray): encoded ground truth from the HermesGTEncoder
            hermes_gt_encoder (HermesGTEncoder): HermesGTEncoder object
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()

        x_s = hermes_gt_encoder.x_s
        y_s = hermes_gt_encoder.y_s

        ax.pcolormesh(
            y_s,
            x_s,
            gt_encoding,
            shading='gouraud',
            cmap=cmap
        )
        
        ax.set_xlabel("Y (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("X (m)",fontsize=self.font_size_axis_labels)
        ax.set_title("Ground Truth",fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)
        ax.invert_xaxis()

        if show:
            plt.show()

    def plot_model_prediction(
            self,
            pred,
            hermes_decoder:HermesDecoder,
            cmap="viridis",
            ax:plt.Axes=None,
            show=False
    ):
        """Plot the ground truth encoding from the HermesGTEncoder

        Args:
            pred (np.ndarray): encoded ground truth from the HermesGTEncoder
            hermes_decoder (HermesDecoder): HermesDecoder object
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()

        x_s = hermes_decoder.x_mesh
        y_s = hermes_decoder.y_mesh

        ax.pcolormesh(
            y_s,
            x_s,
            pred,
            shading='gouraud',
            cmap=cmap
        )
        
        ax.set_xlabel("Y (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("X (m)",fontsize=self.font_size_axis_labels)
        ax.set_title("Model Prediction",fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)
        ax.invert_xaxis()

        if show:
            plt.show()

    ####################################################################
    #Plotting point clouds
    ####################################################################
    def plot_gt_pc(
        self,
        point_cloud:np.ndarray,
        gt_encoder:HermesGTEncoder,
        ax:plt.Axes=None,
        title:str="Ground Truth Point Cloud",
        show=False
        ):
        """Plot the ground truth point cloud in cartesian coordinates

        Args:
            point_cloud (np.ndarray): Nx2 array of points generated from the decoder
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            title(str,optional): the title to be displayed on the plot. 
                Defaults to "Ground Truth Point Cloud".
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()
        
        ax.scatter(
            x=point_cloud[:,1],
            y=point_cloud[:,0],
            s=self.marker_size
        )

        max_y = np.max(np.abs(gt_encoder.y_s))
        ax.set_xlim(
            left=-1 * max_y,
            right=max_y
        )

        max_x = np.max(gt_encoder.x_s)
        ax.set_ylim(
            bottom=0,
            top=max_x
        )
        ax.set_xlabel("Y (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("X (m)",fontsize=self.font_size_axis_labels)
        ax.set_title(title,fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)

        ax.invert_xaxis()

        if show:
            plt.show()
    
    def plot_predicted_pc(
        self,
        point_cloud:np.ndarray,
        hermes_decoder:HermesDecoder,
        ax:plt.Axes=None,
        title:str="Predicted Point Cloud",
        show=False
        ):
        """Plot the ground truth point cloud in cartesian coordinates

        Args:
            point_cloud (np.ndarray): Nx2 array of points generated from the decoder
            hermes_decoder (HermesDecoder): HermesDecoder object
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            title(str,optional): the title to be displayed on the plot. 
                Defaults to "Ground Truth Point Cloud".
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()
        
        ax.scatter(
            x=point_cloud[:,1],
            y=point_cloud[:,0],
            s=5
        )

        max_y = np.max(np.abs(hermes_decoder.y_mesh))
        ax.set_xlim(
            left=-1 * max_y,
            right=max_y
        )

        max_x = np.max(hermes_decoder.x_mesh)
        ax.set_ylim(
            bottom=0,
            top=max_x
        )
        ax.set_xlabel("Y (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("X (m)",fontsize=self.font_size_axis_labels)
        ax.set_title(title,fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)

        ax.invert_xaxis()

        if show:
            plt.show()
    
    ####################################################################
    #Plotting compilations of data
    ####################################################################
    def plot_compilation(
            self,
            input_adc_cube:np.ndarray,
            input_vels:np.ndarray,
            input_encoder:HermesEncoder,
            gt_data:np.ndarray=np.empty(shape=(0)),
            ground_truth_encoder:HermesGTEncoder=None,
            runner:_ModelRunner=None,
            decoder:HermesDecoder=None,
            camera_view:np.ndarray=np.empty(shape=(0)),
            axs:plt.Axes=[],
            show=False
    ):
        """_summary_

        Args:
            input_adc_cube (np.ndarray): _description_
            input_vels (np.ndarray): _description_
            input_encoder (RadSAREncoder): _description_
            gt_data (np.ndarray, optional): _description_. Defaults to np.empty(shape=(0)).
            ground_truth_encoder (_lidarPCPolarDecoder, optional): _description_. Defaults to None.
            runner (_ModelRunner, optional): _description_. Defaults to None.
            decoder (_lidarPCPolarDecoder, optional): _description_. Defaults to None.
            axs (plt.Axes, optional): _description_. Defaults to [].
            show (bool, optional): _description_. Defaults to False.
        """
    

        if len(axs) == 0:
            fig,axs=plt.subplots(3,3, figsize=(15,15))
            fig.subplots_adjust(wspace=0.3,hspace=0.30)
        
        #plot range az response
        input_encoding = input_encoder.encode(
            adc_data_cube=input_adc_cube,
            vels=input_vels
        )


        if input_encoder.full_encoding_ready:

            self.plot_input_encoding(
                encoding=input_encoding,
                hermes_encoder=input_encoder,
                cmap="gray",
                ax=axs[0,0],
                show=False
            )
        
            if ground_truth_encoder:

                gt_encoding = ground_truth_encoder.encode(
                    lidar_pc=gt_data,
                    input_encoding=input_encoding
                )

                self.plot_gt_encoding(
                    gt_encoding=gt_encoding,
                    hermes_gt_encoder=ground_truth_encoder,
                    cmap="binary",
                    ax=axs[1,0],
                    show=False
                )
                axs[1,0].set_title("Ground Truth Encoding",fontsize=self.font_size_title)

                original_gt_pc = ground_truth_encoder.grid_to_points(
                    ground_truth_encoder.original_gt_encoding)
                self.plot_gt_pc(
                    point_cloud=original_gt_pc,
                    gt_encoder=ground_truth_encoder,
                    ax=axs[1,1],
                    title="LiDAR Point Cloud",
                    show=False
                )

            if runner and decoder:

                #plot the prediction
                pred = runner.make_prediction(input=input_encoding)

                self.plot_model_prediction(
                    pred=pred,
                    hermes_decoder=decoder,
                    cmap="gray",
                    ax=axs[0,1],
                    show=False
                )

                #plot the point cloud in cartesian
                pc = decoder.decode(pred)

                self.plot_predicted_pc(
                    point_cloud=pc,
                    hermes_decoder=decoder,
                    ax=axs[1,2],
                    title="Predicted Point Cloud",
                    show=False
                )
        
        # if ground_truth_encoder and gt_data.shape[0] > 0:

        #     gt_grid = ground_truth_encoder.encode(gt_data)
        #     quantized_pc = ground_truth_encoder.grid_to_polar_points(gt_grid)
        #     quantized_pc = polar_to_cartesian(quantized_pc)

        #     self.plot_model_output_polar_grid(
        #         output_grid=gt_grid,
        #         range_bins_m=ground_truth_encoder.range_bins_m,
        #         angle_bins_rad=ground_truth_encoder.angle_bins_rad,
        #         cmap="binary",
        #         title="Ground Truth (Polar)",
        #         ax=axs[1,2],
        #         show=False
        #     )

        #     self.plot_output_pc_cartesian(
        #         point_cloud=quantized_pc,
        #         range_bins_m=ground_truth_encoder.range_bins_m,
        #         ax=axs[0,2],
        #         title="Ground Truth Point Cloud \n (Cart.)",
        #         show=False
        #     )

        #     #plot the camera view
        # if camera_view.shape[0] > 0:
        #     axs[2,0].imshow(camera_view)
        #     axs[2,0].set_title("Frontal Camera View",fontsize=self.font_size_title)

        if show:
            plt.show()