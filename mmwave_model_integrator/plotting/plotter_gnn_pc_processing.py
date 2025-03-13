import matplotlib.pyplot as plt
import numpy as np

from mmwave_model_integrator.input_encoders._node_encoder import _NodeEncoder
from mmwave_model_integrator.ground_truth_encoders._gt_node_encoder import _GTNodeEncoder
from mmwave_model_integrator.model_runner.gnn_runner import GNNRunner
from geometries.transforms.transformation import Transformation
from geometries.pose.orientation import Orientation

from mmwave_model_integrator.plotting._plotter import _Plotter
from mmwave_model_integrator.decoders._decoder import _Decoder


class PlotterGnnPCProcessing(_Plotter):

    def __init__(self) -> None:
        
        super().__init__()

        self.plot_x_max = 4
        self.plot_y_max = 6
        self.marker_size = 30

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
            input_data:np.ndarray,
            input_encoder:_NodeEncoder,
            gt_data:np.ndarray=np.empty(shape=(0)),
            ground_truth_encoder:_GTNodeEncoder=None,
            runner:GNNRunner=None,
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

        if len(axs) == 0:
            fig,axs=plt.subplots(1,3, figsize=(15,5))
            fig.subplots_adjust(wspace=0.3,hspace=0.30)
        
        #plot the input point cloud
        nodes_encoded = input_encoder.encode(input_data)
        labels_encoded = ground_truth_encoder.encode(gt_data)

        self.plot_points(
            points=input_data[:,0:2],
            ax=axs[0],
            color="blue",
            title="Input Points: {} dets".format(input_data.shape[0]),
            show=False
        )
        
        # Plot ground truth valid nodes in a distinct color
        if gt_data.shape[0]>0:
            valid_pts = input_data[gt_data == 1.0, :]
            self.plot_points(
                points=valid_pts[:,0:2],
                ax=axs[1],
                color="red",
                title="GT Detections:{} dets".format(valid_pts.shape[0]),
                show=False
            )
        
        if runner:
            dets = runner.make_prediction(nodes_encoded)
            self.plot_points(
                points=dets[:,0:2],
                ax=axs[2],
                color="green",
                title="Predicted Points: {} dets".format(dets.shape[0]),
                show=False
            )

        if show:
            plt.show()