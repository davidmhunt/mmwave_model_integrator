import matplotlib.pyplot as plt
import numpy as np

from mmwave_model_integrator.input_encoders._node_encoder import _NodeEncoder
from mmwave_model_integrator.ground_truth_encoders._gt_node_encoder import _GTNodeEncoder
from mmwave_model_integrator.plotting.plotter_rng_az_to_pc import PlotterRngAzToPC
from mmwave_model_integrator.dataset_generators.rng_az_to_pc_dataset_generator import RngAzToPCDatasetGenerator
from mmwave_model_integrator.model_runner.gnn_runner import GNNRunner
from geometries.transforms.transformation import Transformation
from geometries.pose.orientation import Orientation

class PlotterGnnPCProcessing:

    def __init__(self) -> None:
        
        #define default plot parameters:
        self.font_size_axis_labels = 12
        self.font_size_title = 15
        self.font_size_ticks = 12
        self.font_size_legend = 12
        self.plot_x_max = 10
        self.plot_y_max = 10
        self.marker_size = 15
        self.line_width = 3

    ####################################################################
    #Plotting point clouds
    ####################################################################

    def plot_points(
        self,
        points: np.ndarray,
        ax: plt.Axes = None,
        color="red",
        label="points",
        title: str = "Detections",
        show: bool = False
    ):
        """Plot both nodes and ground truth valid nodes on the same plot.

        Args:
            points (np.ndarray): At least Nx2 array containing the (x,y) coordinates of each node.
            ax (plt.Axes, optional): The axes on which to display the plot. If none provided, a figure is automatically generated.
                Defaults to None.
            color (str,optional): the color of the points. Defaults to "red"
            label (str,optional): the label for the legend. Defaults to "points"
            title (str, optional): The title to be displayed on the plot. Defaults to "Nodes and Ground Truth".
            show (bool, optional): If True, shows the plot. Defaults to False.
        """
        if not ax:
            fig, ax = plt.subplots()
        
        #transform node points
        pts = np.hstack((points[:,0:2],np.zeros(shape=(points.shape[0],1))))
        rotation = Orientation.from_euler(
            yaw=90,
            degrees=True)
        transformation = Transformation(
            rotation=rotation._orientation
        )
        pts = transformation.apply_transformation(
            pts
        )

        # Plot all nodes in a light color
        ax.scatter(
            x=pts[:, 0],
            y=pts[:, 1],
            s=self.marker_size,
            color=color,
            alpha=0.5,
            label=label
        )
        
        ax.set_xlim(left=-1 * self.plot_x_max, right=self.plot_x_max)
        ax.set_ylim(bottom=-1 * self.plot_y_max, top=self.plot_y_max)
        ax.set_xlabel("Y (m)", fontsize=self.font_size_axis_labels)
        ax.set_ylabel("X (m)", fontsize=self.font_size_axis_labels)
        ax.set_title(title, fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)
        ax.legend()
        
        if show:
            plt.show()

    def plot_nodes(
        self,
        nodes: np.ndarray,
        labels: np.ndarray=np.empty(0),
        ax: plt.Axes = None,
        title: str = "Nodes vs Valid Nodes",
        show: bool = False
    ):
        """Plot both nodes and ground truth valid nodes on the same plot.

        Args:
            nodes (np.ndarray): At least Nx2 array containing the (x,y) coordinates of each node.
            labels (np.ndarray, optional): N-element array containing the label for each node
                (1 is valid, 0 is invalid). Defaults to np.empty(0).
            ax (plt.Axes, optional): The axes on which to display the plot. If none provided, a figure is automatically generated.
                Defaults to None.
            title (str, optional): The title to be displayed on the plot. Defaults to "Nodes and Ground Truth".
            show (bool, optional): If True, shows the plot. Defaults to False.
        """
        if not ax:
            fig, ax = plt.subplots()
        
        self.plot_points(
            points=nodes[:,0:2],
            ax=ax,
            color="blue",
            label="Nodes",
            show=False
        )
        
        # Plot ground truth valid nodes in a distinct color
        if labels.shape[0]>0:
            valid_pts = nodes[labels == 1.0, :]
            self.plot_points(
                points=valid_pts,
                ax=ax,
                color="red",
                label="Valid Nodes",
                show=False
            )
        
        ax.set_xlim(left=-1 * self.plot_x_max, right=self.plot_x_max)
        ax.set_ylim(bottom=-1 * self.plot_y_max, top=self.plot_y_max)
        ax.set_xlabel("Y (m)", fontsize=self.font_size_axis_labels)
        ax.set_ylabel("X (m)", fontsize=self.font_size_axis_labels)
        ax.set_title(title, fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)
        ax.legend()
        
        if show:
            plt.show()

    ####################################################################
    #Plotting compilations of data
    ####################################################################
    def plot_compilation(
            self,
            nodes:np.ndarray,
            input_encoder:_NodeEncoder,
            labels:np.ndarray=np.empty(shape=(0)),
            ground_truth_encoder:_GTNodeEncoder=None,
            runner:GNNRunner=None,
            axs:plt.Axes=[],
            show=False
    ):
        """_summary_

        Args:
            nodes (np.ndarray): _description_
            input_encoder (_NodeEncoder): _description_
            labels (np.ndarray, optional): _description_. Defaults to np.empty(shape=(0)).
            ground_truth_encoder (_GTNodeEncoder, optional): _description_. Defaults to None.
            runner (GNNRunner, optional): _description_. Defaults to None.
            axs (plt.Axes, optional): _description_. Defaults to [].
            show (bool, optional): _description_. Defaults to False.
        """

        if len(axs) == 0:
            fig,axs=plt.subplots(2,3, figsize=(15,10))
            fig.subplots_adjust(wspace=0.3,hspace=0.30)
        
        #plot the input point cloud
        nodes_encoded = input_encoder.encode(nodes)
        labels_encoded = ground_truth_encoder.encode(labels)

        self.plot_points(
            points=nodes[:,0:2],
            ax=axs[0,0],
            color="blue",
            title="Input Points: {} dets".format(nodes.shape[0]),
            show=False
        )
        
        # Plot ground truth valid nodes in a distinct color
        if labels.shape[0]>0:
            valid_pts = nodes[labels == 1.0, :]
            self.plot_points(
                points=valid_pts[:,0:2],
                ax=axs[0,1],
                color="red",
                title="GT Detections:{} dets".format(valid_pts.shape[0]),
                show=False
            )
        
        if runner:
            dets = runner.make_prediction(nodes_encoded)
            self.plot_points(
                points=dets[:,0:2],
                ax=axs[0,2],
                color="green",
                title="Predicted Points: {} dets".format(dets.shape[0]),
                show=False
            )

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