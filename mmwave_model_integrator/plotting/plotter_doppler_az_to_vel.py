import matplotlib.pyplot as plt
import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_model_integrator.input_encoders._doppler_az_encoder import _DopplerAzEncoder
from mmwave_model_integrator.decoders._decoder import _Decoder
from mmwave_model_integrator.model_runner._model_runner import _ModelRunner
from mmwave_model_integrator.ground_truth_encoders._gt_encoder_odom_to_vel import _GTEncoderOdomToVel
from mmwave_model_integrator.plotting._plotter import _Plotter


class PlotterDopplerAzToVel(_Plotter):

    def __init__(self):
        super().__init__()

    ####################################################################
    #Doppler Azimuth Response
    ####################################################################
    def plot_doppler_az_resp(
        self,
        resp:np.ndarray,
        doppler_azimuth_encoder:_DopplerAzEncoder,
        cmap="viridis",
        ax:plt.Axes=None,
        title:str = "Doppler-Azimuth Heatmap",
        show=False
        ):
        """Plot the range doppler response

        Args:
            resp (np.ndarray): range_bins x velocity bins np.ndarray of the already computed
                range doppler response
            doppler_azimuth_processor (_DopplerAzEncoder): _DopplerAzEncoder object
                used to generate the encoding
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()
        
        im = ax.imshow(
            np.flip(resp,axis=0),
            extent=[
                doppler_azimuth_encoder.angle_bins[0],
                doppler_azimuth_encoder.angle_bins[-1],
                doppler_azimuth_encoder.vel_bins[0],
                doppler_azimuth_encoder.vel_bins[-1],
            ],
            cmap=cmap,
            aspect='auto',
            interpolation='bilinear'
            )
        ax.set_ylabel("Velocity (m/s)",fontsize=self.font_size_axis_labels)
        ax.set_xlabel("Angle (radians)",fontsize=self.font_size_axis_labels)
        ax.set_title(title,fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)

        if show:
            plt.show()

    ####################################################################
    #Plotting compilations of data
    ####################################################################

    def plot_compilation(
            self,
            input_data:np.ndarray,
            input_encoder:_DopplerAzEncoder,
            gt_data:np.ndarray=np.empty(shape=(0)),
            ground_truth_encoder:_GTEncoderOdomToVel=None,
            runner:_ModelRunner=None,
            decoder:_Decoder=None,
            camera_view:np.ndarray=np.empty(shape=(0)),
            axs:plt.Axes=[],
            show=False
    ):
        """_summary_

        Args:
            input_data (np.ndarray): _description_
            input_encoder (_DopplerAzEncoder): _description_
            gt_data (np.ndarray, optional): _description_. Defaults to np.empty(shape=(0)).
            ground_truth_encoder (_GTEncoderOdomToVel, optional): _description_. Defaults to None.
            runner (_ModelRunner, optional): _description_. Defaults to None.
            decoder (_Decoder,optional): _description_.Defaults to None.
            camera_view:(np.ndarray,optional): _description_. Defaults to np.empty(shape=(0)).
            axs (plt.Axes, optional): _description_. Defaults to [].
            show (bool, optional): _description_. Defaults to False.
        """
    

        if len(axs) == 0:
            fig,axs=plt.subplots(1,2, figsize=(10,5))
            fig.subplots_adjust(wspace=0.3,hspace=0.30)

        #get the doppler azimuth response
        doppler_az_resp = input_encoder.encode(input_data)

        gt_vel = ground_truth_encoder.encode(gt_data)

        formatted_vels = np.array2string(
            gt_vel,
            formatter={'float_kind': lambda x: f"{x:.2f}"},
            separator=', '
        )

        if input_encoder.full_encoding_ready \
            and runner \
            and decoder:

            pred = runner.make_prediction(doppler_az_resp)
            
            formatted_pred = np.array2string(
                pred,
                formatter={'float_kind': lambda x: f"{x:.2f}"},
                separator=', '
            )
            
            title_str = "Doppler-Azimuth Heatmap\n GT:{} m/s\n pred:{}".format(
                formatted_vels,
                formatted_pred
            )
        
        else:

            title_str = "Doppler-Azimuth Heatmap\n GT:{} m/s".format(
                formatted_vels
            )
        
        #plot doppler azimuth response
        self.plot_doppler_az_resp(
            resp=doppler_az_resp,
            doppler_azimuth_encoder=input_encoder,
            cmap="binary",
            title=title_str,
            ax=axs[0],
            show=False
        )

        #plot the camera view
        if camera_view.shape[0] > 0:
            axs[1].imshow(camera_view)
            axs[1].set_title("Frontal Camera View",fontsize=self.font_size_title)