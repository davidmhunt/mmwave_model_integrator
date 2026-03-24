import matplotlib.pyplot as plt
import numpy as np

from mmwave_model_integrator.input_encoders._input_encoder import _InputEncoder
from mmwave_model_integrator.ground_truth_encoders._gt_encoder import _GTEncoder
from mmwave_model_integrator.model_runner._model_runner import _ModelRunner
from mmwave_model_integrator.plotting._plotter import _Plotter
from mmwave_model_integrator.decoders._decoder import _Decoder


class PlotterGrids(_Plotter):

    def __init__(self, max_detection_range: float, grid_resolution_m: float) -> None:
        
        super().__init__()
        
        self.grid_max_distance_m: float = max_detection_range
        self.grid_resolution_m: float = grid_resolution_m
        self.grid_bins: np.ndarray = np.arange(
            start=-1 * self.grid_max_distance_m,
            stop=self.grid_max_distance_m + self.grid_resolution_m,
            step=self.grid_resolution_m
        )

    def plot_pc_grid(
        self,
        grid: np.ndarray,
        ax: plt.Axes = None,
        title: str = "Occupancy Probability",
        show: bool = False
    ):
        """Plot a point cloud grid.
        NOTE: Assumes that [0,0] is the bottom left coordinate of the grid

        Args:
            grid (np.ndarray): an NxN point cloud grid
            ax (plt.Axes, optional): A set of axes to plot on. 
                Defaults to None.
            title (str, optional): The title of the plot. Defaults to "Occupancy Probability".
            show (bool, optional): on True, shows the plot. 
                Defaults to False.
        """
        if not ax:
            fig, ax = plt.subplots()

        max_rng = self.grid_bins.max()
        min_rng = self.grid_bins.min()

        #plot the occupancy grid
        ax.imshow(
            grid,
            cmap='gray',
            interpolation='none',
            origin="lower",
            extent=(max_rng, min_rng, min_rng, max_rng),
            vmax=1,
            vmin=0
        )

        ax.set_title(title, fontsize=self.font_size_title)
        ax.set_xlim(min_rng, max_rng)
        ax.set_xlabel("Y (m)", fontsize=self.font_size_axis_labels)
        ax.set_ylim(min_rng, max_rng)
        ax.set_ylabel("X (m)", fontsize=self.font_size_axis_labels)
        ax.tick_params(labelsize=self.font_size_ticks)

        if show:
            plt.show()

        return

    def plot_compilation(
        self,
        input_data: np.ndarray,
        input_encoder: _InputEncoder,
        gt_data: np.ndarray = np.empty(shape=(0)),
        ground_truth_encoder: _GTEncoder = None,
        runner: _ModelRunner = None,
        decoder: _Decoder = None,
        axs: list = [],
        show: bool = False
    ):
        """Plot a compilation of the input grid, ground truth grid, and predicted grid.

        Args:
            input_data (np.ndarray): The raw input data to be encoded.
            input_encoder (_InputEncoder): The encoder for the input data.
            gt_data (np.ndarray, optional): The ground truth data. Defaults to np.empty(shape=(0)).
            ground_truth_encoder (_GTEncoder, optional): The encoder for the ground truth data. Defaults to None.
            runner (_ModelRunner, optional): The model runner to get predictions. Defaults to None.
            decoder (_Decoder, optional): The decoder. Defaults to None.
            axs (list, optional): A list of axes to plot on. Defaults to [].
            show (bool, optional): If True, shows the plot. Defaults to False.
        """
        if len(axs) == 0:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            fig.subplots_adjust(wspace=0.3, hspace=0.30)
        
        # Plot the input grid
        input_encoded = input_encoder.encode(input_data)
        
        self.plot_pc_grid(
            grid=input_encoded,
            ax=axs[0],
            title="Input Grid",
            show=False
        )
        
        # Plot the ground truth grid if available
        if gt_data.shape[0] > 0 and ground_truth_encoder is not None:
            gt_encoded = ground_truth_encoder.encode(gt_data)
            self.plot_pc_grid(
                grid=gt_encoded,
                ax=axs[1],
                title="Ground Truth Grid",
                show=False
            )
        
        # Plot the predicted grid if a runner is provided
        if runner is not None:
            prediction = runner.make_prediction(input_encoded)
            self.plot_pc_grid(
                grid=prediction,
                ax=axs[2],
                title="Predicted Grid",
                show=False
            )

        if show:
            plt.show()
