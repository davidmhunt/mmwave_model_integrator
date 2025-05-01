import matplotlib.pyplot as plt
import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_model_integrator.input_encoders.radsar_encoder import RadSAREncoder
from mmwave_model_integrator.decoders._lidar_pc_polar_decoder import _lidarPCPolarDecoder
from mmwave_model_integrator.model_runner._model_runner import _ModelRunner
from mmwave_model_integrator.ground_truth_encoders._gt_encoder_lidar2D import _GTEncoderLidar2D
from mmwave_model_integrator.transforms.coordinate_transforms import polar_to_cartesian
from mmwave_model_integrator.plotting.plotter_rng_az_to_pc import PlotterRngAzToPC
from mmwave_radar_processing.supportFns.rotation_functions import apply_rot_trans


from mmwave_radar_processing.processors.synthetic_array_beamformer_processor_revA import SyntheticArrayBeamformerProcessor

class PlotterRadSar(PlotterRngAzToPC):

    def __init__(self):

        super().__init__()
    
    ####################################################################
    #Plotting radar images/responses
    ####################################################################
    def plot_range_az_resp_cart(
        self,
        resp:np.ndarray,
        range_az_encoder:RadSAREncoder,
        cmap="viridis",
        ax:plt.Axes=None,
        show=False
        ):
        """Plot the range azimuth response in cartesian

        Args:
            resp (np.ndarray): range_bins x angle_bins np.ndarray of the already computed
                range azimuth response
            range_az_encoder (RadSAREncoder): RadSAREncoder object
                used to generate the response
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if range_az_encoder.synthetic_array_processor.mode == SyntheticArrayBeamformerProcessor.ENDFIRE_MODE:
            self.plot_range_az_resp_cart_endfire(
                resp=resp,
                range_az_encoder=range_az_encoder,
                cmap=cmap,
                ax=ax,
                show=show
            )
        if range_az_encoder.synthetic_array_processor.mode == SyntheticArrayBeamformerProcessor.BROADSIDE_MODE:
            self.plot_range_az_resp_cart_broadside(
                resp=resp,
                range_az_encoder=range_az_encoder,
                cmap=cmap,
                ax=ax,
                show=show
            )

    def plot_range_az_resp_cart_endfire(
        self,
        resp:np.ndarray,
        range_az_encoder:RadSAREncoder,
        cmap="viridis",
        ax:plt.Axes=None,
        show=False
        ):
        """Plot the range azimuth response in cartesian

        Args:
            resp (np.ndarray): range_bins x angle_bins np.ndarray of the already computed
                range azimuth response
            range_az_encoder (RadSAREncoder): RadSAREncoder object
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
        
        x_s = range_az_encoder.synthetic_array_processor.x_s[:,:,0]
        y_s = range_az_encoder.synthetic_array_processor.y_s[:,:,0]

        resp = range_az_encoder.get_rng_az_resp_from_encoding(resp)

        ax.pcolormesh(
            y_s,
            x_s,
            resp,
            cmap=cmap,
            shading='gouraud'
        )

        ax.set_xlabel("Y (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("X (m)",fontsize=self.font_size_axis_labels)
        
        ax.set_title("Range-Azimuth\nHeatmap (Cart.)",fontsize=self.font_size_title)
        
        ax.tick_params(labelsize=self.font_size_ticks)

        #invert the axis to get it to display correctly
        ax.invert_xaxis()
        
        if show:
            plt.show()

    def plot_range_az_resp_cart_broadside(
        self,
        resp:np.ndarray,
        range_az_encoder:RadSAREncoder,
        cmap="viridis",
        ax:plt.Axes=None,
        show=False
        ):
        """Plot the range azimuth response in cartesian

        Args:
            resp (np.ndarray): range_bins x angle_bins np.ndarray of the already computed
                range azimuth response
            range_az_encoder (RadSAREncoder): RadSAREncoder object
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
        
        x_s = range_az_encoder.synthetic_array_processor.x_s[:,:,0]
        y_s = range_az_encoder.synthetic_array_processor.y_s[:,:,0]

        resp = range_az_encoder.get_rng_az_resp_from_encoding(resp)

        ax.pcolormesh(
            x_s,
            y_s,
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
        range_az_encoder:RadSAREncoder,
        cmap="viridis",
        ax:plt.Axes=None,
        show=False
        ):
        """Plot the range azimuth response in polar

        Args:
            resp (np.ndarray): range_bins x angle_bins np.ndarray of the already computed
                range azimuth response
            range_az_encoder (RadSAREncoder): _RadarRangeAzEncoder object
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

        if range_az_encoder.synthetic_array_processor.mode == SyntheticArrayBeamformerProcessor.ENDFIRE_MODE:
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
            ax.invert_xaxis()
        if range_az_encoder.synthetic_array_processor.mode == SyntheticArrayBeamformerProcessor.BROADSIDE_MODE:
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
        output_grid = np.flip(output_grid,axis=0)

        ax.imshow(
            output_grid,
            extent=[
                angle_bins_rad[0],
                angle_bins_rad[-1],
                range_bins_m[0],
                range_bins_m[-1]
            ],
        cmap=cmap,
            aspect='auto'
            )
        ax.invert_xaxis()
        
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

        #rotate the lidar pc into the radar's frame of view
        point_cloud = apply_rot_trans(
            point_cloud,
            rot_angle_rad=np.deg2rad(90),
            trans=np.array([0,0])
        )
        
        ax.scatter(
            x=-1 * point_cloud[:,0],
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
            input_encoder:RadSAREncoder,
            gt_data:np.ndarray=np.empty(shape=(0)),
            ground_truth_encoder:_lidarPCPolarDecoder=None,
            runner:_ModelRunner=None,
            decoder:_lidarPCPolarDecoder=None,
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
        rng_az_resp_encoded = input_encoder.encode(
            adc_data_cube=input_adc_cube,
            vels=input_vels
        )


        if input_encoder.full_encoding_ready:

            self.plot_range_az_resp_cart(
                resp=rng_az_resp_encoded,
                range_az_encoder=input_encoder,
                cmap="gray",
                ax=axs[0,0],
                show=False
            )

            self.plot_range_az_resp_polar(
                resp=rng_az_resp_encoded,
                range_az_encoder=input_encoder,
                cmap="gray",
                ax=axs[1,0],
                show=False
            )

            if runner and decoder:

                #plot the prediction
                pred = runner.make_prediction(input=rng_az_resp_encoded)

                self.plot_model_output_polar_grid(
                    output_grid=pred,
                    range_bins_m=decoder.range_bins,
                    angle_bins_rad=decoder.angle_bins,
                    cmap="binary",
                    title="Model Prediction (Polar)",
                    ax=axs[1,1],
                    show=False
                )

                #plot the point cloud in cartesian
                pc = polar_to_cartesian(decoder.decode(pred))

                self.plot_output_pc_cartesian(
                    point_cloud=pc,
                    range_bins_m=decoder.range_bins,
                    ax=axs[0,1],
                    title="Generated Point Cloud \n (Cart.)",
                    show=False
                )
        
        if ground_truth_encoder and gt_data.shape[0] > 0:

            gt_grid = ground_truth_encoder.encode(gt_data)
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

            #plot the camera view
        if camera_view.shape[0] > 0:
            axs[2,0].imshow(camera_view)
            axs[2,0].set_title("Frontal Camera View",fontsize=self.font_size_title)

        if show:
            plt.show()