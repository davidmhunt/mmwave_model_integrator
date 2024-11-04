import matplotlib.pyplot as plt
import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_model_integrator.input_encoders._radar_range_az_encoder import _RadarRangeAzEncoder
from mmwave_model_integrator.decoders._lidar_pc_polar_decoder import _lidarPCPolarDecoder
from mmwave_model_integrator.model_runner._model_runner import _ModelRunner
from mmwave_model_integrator.ground_truth_encoders._gt_encoder_lidar2D import _GTEncoderLidar2D
from mmwave_model_integrator.transforms.coordinate_transforms import polar_to_cartesian


class PlotterRngAzToPC:

    def __init__(self) -> None:
        
        #define default plot parameters:
        self.font_size_axis_labels = 12
        self.font_size_title = 15
        self.font_size_ticks = 12
        self.font_size_legend = 12
        self.plot_x_max = 10
        self.plot_y_max = 20
        self.marker_size = 10
        self.line_width = 3

    ####################################################################
    #Plotting radar images/responses
    ####################################################################
    def plot_range_az_resp_cart(
        self,
        resp:np.ndarray,
        range_az_encoder:_RadarRangeAzEncoder,
        cmap="viridis",
        ax:plt.Axes=None,
        show=False
        ):
        """Plot the range azimuth response in cartesian

        Args:
            resp (np.ndarray): range_bins x angle_bins np.ndarray of the already computed
                range azimuth response
            range_az_encoder (_RadarRangeAzEncoder): _RadarRangeAzEncoder object
                used to generate the response
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()
        
        #rotate the grids by 90 degrees so that the x-axis is displayed
        #along the y axis
        
        x_grid = -1 * range_az_encoder.y_s
        y_grid = range_az_encoder.x_s

        ax.pcolormesh(
            x_grid,
            y_grid,
            resp,
            cmap=cmap,
            shading='gouraud'
        )

        ax.set_xlabel("Y (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("X (m)",fontsize=self.font_size_axis_labels)
        
        ax.set_title("Range-Azimuth\nHeatmap (Cart.)",fontsize=self.font_size_title)
        
        ax.tick_params(labelsize=self.font_size_ticks)

        if show:
            plt.show()

    def plot_range_az_resp_polar(
        self,
        resp:np.ndarray,
        range_az_encoder:_RadarRangeAzEncoder,
        cmap="viridis",
        ax:plt.Axes=None,
        show=False
        ):
        """Plot the range azimuth response in polar

        Args:
            resp (np.ndarray): range_bins x angle_bins np.ndarray of the already computed
                range azimuth response
            range_az_encoder (_RadarRangeAzEncoder): _RadarRangeAzEncoder object
                used to generate the response
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()

        ax.imshow(
            np.flip(resp,axis=0),
            extent=[
                range_az_encoder.angle_bins[0],
                range_az_encoder.angle_bins[-1],
                range_az_encoder.range_bins[0],
                range_az_encoder.range_bins[-1]
            ],
            cmap=cmap,
            aspect='auto'
            )
        
        ax.set_xlabel("Angle (radians)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("Range (m)",fontsize=self.font_size_axis_labels)
        ax.set_title("Range-Azimuth\nHeatmap (Polar)",fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)

        if show:
            plt.show()

    ####################################################################
    #Plotting model predictions/ground truth information
    ####################################################################
    def plot_model_output_polar_grid(
        self,
        output_grid:np.ndarray,
        range_bins_m:np.ndarray,
        angle_bins_rad:np.ndarray,
        cmap="binary",
        title:str="Model Output (Polar)",
        ax:plt.Axes=None,
        show=False
        ):
        """Plot the range azimuth response in polar

        Args:
            output_grid (np.ndarray): np array of the quantized output grid
                in polar coordinates
            range_bins_m (np.ndarray): np array of the range bins for the
                corresponding output grid
            angle_bins_rad (np.ndarray): np array of angle bins for the 
                corresponding output grid in polar coordinates
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            title(str,optional): the title to be displayed on the plot. 
                Defaults to "Model Output (Polar)".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()

        #flip the image to be easier to read
        if range_bins_m[0] < range_bins_m[-1]:
            output_grid = np.flip(output_grid,axis=0)

        ax.imshow(
            output_grid,
            extent=[
                np.min(angle_bins_rad),
                np.max(angle_bins_rad),
                np.min(range_bins_m),
                np.max(range_bins_m)
            ],
            cmap=cmap,
            aspect='auto'
            )
        
        ax.set_xlabel("Angle (radians)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("Range (m)",fontsize=self.font_size_axis_labels)
        ax.set_title(title,fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)

        if show:
            plt.show()

    def plot_output_pc_cartesian(
        self,
        point_cloud:np.ndarray,
        range_bins_m:np.ndarray,
        ax:plt.Axes=None,
        title:str="Model Output (Polar)",
        show=False
        ):
        """Plot the output point cloud in cartesian coordinates

        Args:
            point_cloud (np.ndarray): Nx2 array of points generated from the decoder
            range_bins_m (np.ndarray): np array of the range bins for the
                corresponding output.
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            title(str,optional): the title to be displayed on the plot. 
                Defaults to "Model Output (Polar)".
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()

        #rotate the point cloud so that x is on the y axis
        rot_matrix = np.array([
            [0,1],
            [-1,0]
        ])

        point_cloud = np.dot(point_cloud,rot_matrix)

        ax.scatter(
            x=point_cloud[:,0],
            y=point_cloud[:,1],
            s=self.marker_size
        )
        
        max_rng = np.max(range_bins_m)
        ax.set_xlim(
            left=-1 * max_rng,
            right=max_rng
        )
        ax.set_ylim(
            bottom=0,
            top=max_rng
        )
        ax.set_xlabel("Y (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("X (m)",fontsize=self.font_size_axis_labels)
        ax.set_title(title,fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)

        if show:
            plt.show()

    ####################################################################
    #Plotting compilations of data
    ####################################################################
    def plot_compilation(
            self,
            adc_cube:np.ndarray,
            input_encoder:_RadarRangeAzEncoder,
            model_runner:_ModelRunner=None,
            prediction_decoder:_lidarPCPolarDecoder=None,
            lidar_pc:np.ndarray = np.empty(0),
            ground_truth_encoder:_GTEncoderLidar2D=None,
            axs:plt.Axes=[],
            show=False
    ):
        """Plot a compilation of the model input, prediction (if available),
            and desired model output (if available).

        Args:
            adc_cube (np.ndarray): _description_
            range_az_encoder (_RadarRangeAzEncoder): _description_
            model_runner (_ModelRunner,optional): _description_.
                Defaults to None
            lidar_pc_polar_decoder (_lidarPCPolarDecoder, optional): _description_.
                Defaults to None
            lidar_pc (np.ndarray,optional): _description_.
                Defaults to np.empty().
            lidar_pc_encoder(_Lidar2DPCEncoder,optional):_description_. Defaults to None
            axs (plt.Axes, optional): _description_. Defaults to [].
            show (bool, optional): _description_. Defaults to False.
        """

        if len(axs) == 0:
            fig,axs=plt.subplots(2,3, figsize=(15,10))
            fig.subplots_adjust(wspace=0.3,hspace=0.30)
        
        #plot range az response
        rng_az_resp_encoded = input_encoder.encode(adc_cube)

        rng_az_to_plot = input_encoder.get_rng_az_resp_from_encoding(
            rng_az_resp=rng_az_resp_encoded
        )

        self.plot_range_az_resp_cart(
            resp=rng_az_to_plot,
            range_az_encoder=input_encoder,
            cmap="gray",
            ax=axs[0,0],
            show=False
        )

        self.plot_range_az_resp_polar(
            resp=rng_az_to_plot,
            range_az_encoder=input_encoder,
            cmap="gray",
            ax=axs[1,0],
            show=False
        )

        if input_encoder.full_encoding_ready \
            and model_runner \
            and prediction_decoder:

            #plot the prediction
            pred = model_runner.make_prediction(input=rng_az_resp_encoded)

            self.plot_model_output_polar_grid(
                output_grid=pred,
                range_bins_m=prediction_decoder.range_bins,
                angle_bins_rad=prediction_decoder.angle_bins,
                cmap="binary",
                title="Model Prediction (Polar)",
                ax=axs[1,1],
                show=False
            )

            #plot the point cloud in cartesian
            pc = polar_to_cartesian(prediction_decoder.decode(pred))

            self.plot_output_pc_cartesian(
                point_cloud=pc,
                range_bins_m=prediction_decoder.range_bins,
                ax=axs[0,1],
                title="Generated Point Cloud \n (Cart.)",
                show=False
            )
        
        if ground_truth_encoder and lidar_pc.shape[0] > 0:

            gt_grid = ground_truth_encoder.encode(lidar_pc)
            quantized_pc = ground_truth_encoder.grid_to_polar_points(gt_grid)
            quantized_pc = polar_to_cartesian(quantized_pc)

            self.plot_model_output_polar_grid(
                output_grid=gt_grid,
                range_bins_m=ground_truth_encoder.range_bins_m,
                angle_bins_rad=ground_truth_encoder.angle_bins_rad,
                cmap="binary",
                title="Ground Truth (Polar)",
                ax=axs[1,2],
                show=False
            )

            self.plot_output_pc_cartesian(
                point_cloud=quantized_pc,
                range_bins_m=ground_truth_encoder.range_bins_m,
                ax=axs[0,2],
                title="Ground Truth Point Cloud \n (Cart.)",
                show=False
            )

        if ground_truth_encoder and (lidar_pc.shape[0]>0):
            pass

        if show:
            plt.show()
    
    ####################################################################
    #Plotting analysis results
    ####################################################################
    def _plot_cdf(
            self,
            distances:np.ndarray,
            label:str,
            show=True,
            percentile = 0.95,
            ax:plt.Axes = None):

        if not ax:
            fig = plt.figure(figsize=(3,3))
            ax = fig.add_subplot()

        sorted_data = np.sort(distances)
        p = 1. * np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

        #compute the index of the percentile
        idx = (np.abs(p - percentile)).argmin()

        plt.plot(sorted_data[:idx],p[:idx],
                 label=label,
                 linewidth=self.line_width)

        ax.set_xlabel('Error (m)',fontsize=self.font_size_axis_labels)
        ax.set_ylabel('CDF',fontsize=self.font_size_axis_labels)
        ax.set_title("Error Comparison",fontsize=self.font_size_title)
        ax.set_xlim((0,5))

        if show:
            plt.grid()
            plt.legend()
            plt.show()

    def plot_distance_metrics_cdfs(
            self,
            chamfer_distances:np.ndarray=np.empty(0),
            hausdorf_distances:np.ndarray=np.empty(0),
            chamfer_distances_radarHD:np.ndarray=np.empty(0),
            modified_hausdorf_distances_radarHD:np.ndarray=np.empty(0)
        ):
        
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot()

            if chamfer_distances.shape[0] > 0:
                self._plot_cdf(
                    distances=chamfer_distances,
                    label="Chamfer Distance",
                    show=False,
                    percentile=1.0,
                    ax = ax
                )

            if hausdorf_distances.shape[0] > 0:
                self._plot_cdf(
                distances=hausdorf_distances,
                label="Hausdorf Distance",
                show=False,
                percentile=1.0,
                ax = ax
            )

            if chamfer_distances_radarHD.shape[0] > 0:
                self._plot_cdf(
                    distances=chamfer_distances_radarHD,
                    label="Chamfer Distance (radarHD)",
                    show=False,
                    percentile=1.0,
                    ax = ax
                )
            
            if modified_hausdorf_distances_radarHD.shape[0] > 0:
                self._plot_cdf(
                    distances=modified_hausdorf_distances_radarHD,
                    label="Modified Hausdorff Distance (radarHD)",
                    show=False,
                    percentile=1.0,
                    ax = ax
                )

            plt.grid()
            plt.legend()
            plt.show()