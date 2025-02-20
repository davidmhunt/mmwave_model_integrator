import matplotlib.pyplot as plt
import numpy as np

from mmwave_model_integrator.input_encoders._input_encoder import _InputEncoder
from mmwave_model_integrator.ground_truth_encoders._gt_encoder import _GTEncoder
from mmwave_model_integrator.model_runner._model_runner import _ModelRunner
from mmwave_model_integrator.decoders._decoder import _Decoder
class _Plotter:

    def __init__(self):
        
        #define default plot parameters:
        self.font_size_axis_labels = 12
        self.font_size_title = 15
        self.font_size_ticks = 12
        self.font_size_legend = 12
        self.plot_x_max = 10
        self.plot_y_max = 10
        self.marker_size = 15
        self.line_width = 3
    
    def plot_compilation(
            self,
            input_data:np.ndarray,
            input_encoder:_InputEncoder,
            gt_data:np.ndarray=np.empty(shape=(0)),
            ground_truth_encoder:_GTEncoder=None,
            runner:_ModelRunner=None,
            decoder:_Decoder=None,
            axs:plt.Axes=[],
            show=False
    ):
        """_summary_

        Args:
            input_data (np.ndarray): _description_
            input_data (_InputEncoder): _description_
            gt_data (np.ndarray, optional): _description_. Defaults to np.empty(shape=(0)).
            ground_truth_encoder (_GTEncoder, optional): _description_. Defaults to None.
            runner (_ModelRunner, optional): _description_. Defaults to None.
            axs (plt.Axes, optional): _description_. Defaults to [].
            show (bool, optional): _description_. Defaults to False.
        """

        pass
    
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