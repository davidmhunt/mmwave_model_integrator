import matplotlib.pyplot as plt
import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager

from mmwave_model_integrator.encoders._radar_range_az_encoder import _RadarRangeAzEncoder
from mmwave_model_integrator.decoders._lidar_pc_polar_decoder import _lidarPCPolarDecoder
from mmwave_model_integrator.model_runner._model_runner import _ModelRunner

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

    def plot_model_pred_polar(
        self,
        pred:np.ndarray,
        lidar_pc_polar_decoder:_lidarPCPolarDecoder,
        cmap="binary",
        ax:plt.Axes=None,
        show=False
        ):
        """Plot the range azimuth response in polar

        Args:
            pred (np.ndarray): np array of the predicted model output
            lidar_pc_polar_decoder (_lidarPCPolarDecoder): _lidarPCPolarDecoder
                 object used to decode the prediction
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()

        #flip the image to be easier to read
        if lidar_pc_polar_decoder.range_bins[0] < lidar_pc_polar_decoder.range_bins[-1]:
            pred = np.flip(pred,axis=0)

        ax.imshow(
            pred,
            extent=[
                np.min(lidar_pc_polar_decoder.angle_bins),
                np.max(lidar_pc_polar_decoder.angle_bins),
                np.min(lidar_pc_polar_decoder.range_bins),
                np.max(lidar_pc_polar_decoder.range_bins)
            ],
            cmap=cmap,
            aspect='auto'
            )
        
        ax.set_xlabel("Angle (radians)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("Range (m)",fontsize=self.font_size_axis_labels)
        ax.set_title("Model Prediction (Polar)",fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)

        if show:
            plt.show()

    def plot_decoded_pc(
        self,
        point_cloud:np.ndarray,
        lidar_pc_polar_decoder:_lidarPCPolarDecoder,
        ax:plt.Axes=None,
        show=False
        ):
        """Plot the range azimuth response in polar

        Args:
            point_cloud (np.ndarray): Nx2 array of points generated from the decoder
            lidar_pc_polar_decoder (_lidarPCPolarDecoder): _lidarPCPolarDecoder
                 object used to decode the prediction
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
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
        
        max_rng = np.max(lidar_pc_polar_decoder.range_bins)
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
        ax.set_title("Generated Point Cloud\n (Cart.)",fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)

        if show:
            plt.show()

    def plot_compilation(
            self,
            adc_cube:np.ndarray,
            range_az_encoder:_RadarRangeAzEncoder,
            model_runner:_ModelRunner,
            lidar_pc_polar_decoder:_lidarPCPolarDecoder,
            axs:plt.Axes=[],
            show=False
    ):
        """_summary_

        Args:
            adc_cube (np.ndarray): _description_
            range_az_encoder (_RadarRangeAzEncoder): _description_
            model_runner (_ModelRunner): _description_
            lidar_pc_polar_decoder (_lidarPCPolarDecoder): _description_
            chirp_idx (int, optional): _description_. Defaults to 0.
            axs (plt.Axes, optional): _description_. Defaults to [].
            show (bool, optional): _description_. Defaults to False.
        """

        if len(axs) == 0:
            fig,axs=plt.subplots(2,2, figsize=(10,10))
            fig.subplots_adjust(wspace=0.3,hspace=0.30)
        
        #plot range az response
        rng_az_resp_encoded = range_az_encoder.encode(adc_cube)

        rng_az_to_plot = range_az_encoder.get_rng_az_resp_from_encoding(
            rng_az_resp=rng_az_resp_encoded
        )

        self.plot_range_az_resp_cart(
            resp=rng_az_to_plot,
            range_az_encoder=range_az_encoder,
            cmap="gray",
            ax=axs[0,0],
            show=False
        )

        self.plot_range_az_resp_polar(
            resp=rng_az_to_plot,
            range_az_encoder=range_az_encoder,
            cmap="gray",
            ax=axs[1,0],
            show=False
        )

        if range_az_encoder.full_encoding_ready:

            #plot the prediction
            pred = model_runner.make_prediction(input=rng_az_resp_encoded)

            self.plot_model_pred_polar(
                pred=pred,
                lidar_pc_polar_decoder=lidar_pc_polar_decoder,
                cmap="binary",
                ax=axs[1,1],
                show=False
            )

            #plot the point cloud in cartesian
            pc = lidar_pc_polar_decoder.convert_polar_to_cartesian(
                lidar_pc_polar_decoder.decode(pred)
            )

            self.plot_decoded_pc(
                point_cloud=pc,
                lidar_pc_polar_decoder=lidar_pc_polar_decoder,
                ax=axs[0,1],
                show=False
            )

        if show:
            plt.show()
