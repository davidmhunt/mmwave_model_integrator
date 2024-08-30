import matplotlib.pyplot as plt
import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager

from mmwave_model_integrator.encoders._radar_range_az_encoder import _RadarRangeAzEncoder

class PlotterRangeAzEncoder:

    def __init__(self,config_manager:ConfigManager) -> None:
        
        #define default plot parameters:
        self.font_size_axis_labels = 12
        self.font_size_title = 15
        self.font_size_ticks = 12
        self.font_size_legend = 12
        self.plot_x_max = 10
        self.plot_y_max = 20
        self.marker_size = 10

        #configuration manager
        self.config_manager:ConfigManager = config_manager

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
        
        ax.pcolormesh(
            range_az_encoder.x_s,
            range_az_encoder.y_s,
            resp,
            cmap=cmap,
            shading='gouraud'
        )

        ax.set_xlabel("X (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("Y (m)",fontsize=self.font_size_axis_labels)
        
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