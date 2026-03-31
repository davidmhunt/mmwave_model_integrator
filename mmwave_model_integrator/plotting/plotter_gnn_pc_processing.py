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

    def plot_multi_instance_compilation(
            self,
            instances_nodes: list,
            instances_labels: list,
            input_encoder: _NodeEncoder,
            ground_truth_encoder: _GTNodeEncoder,
            runner: GNNRunner,
            save_path: str = None,
            show: bool = False
    ):
        """Plot a compilation of multiple instances (e.g. 5x3 grids). 
        
        Args:
            instances_nodes (list of np.ndarray): List of input nodes.
            instances_labels (list of np.ndarray): List of gt labels.
            input_encoder (_NodeEncoder): Encoder for inputs.
            ground_truth_encoder (_GTNodeEncoder): Encoder for GT.
            runner (GNNRunner): Model runner.
            save_path (str, optional): Path to save the resulting .png file.
            show (bool, optional): Whether to display the plot.
        """
        num_instances = len(instances_nodes)
        fig, axs = plt.subplots(num_instances, 3, figsize=(15, 5 * num_instances))
        fig.subplots_adjust(wspace=0.3, hspace=0.30)
        
        # Ensure axs is always 2D
        if num_instances == 1:
            axs = np.expand_dims(axs, axis=0)

        for i in range(num_instances):
            nodes = instances_nodes[i]
            labels = instances_labels[i]
            self.plot_compilation(
                input_data=nodes,
                gt_data=labels,
                input_encoder=input_encoder,
                ground_truth_encoder=ground_truth_encoder,
                runner=runner,
                axs=axs[i, :],
                show=False
            )
            
            # Add a row title or label to the first axis in the row
            axs[i, 0].set_ylabel(f"Instance {i+1}\n\n" + axs[i, 0].get_ylabel(), fontsize=self.font_size_axis_labels, weight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Compilation saved to {save_path}")

        if show:
            plt.show()

    def plot_forward_pass_analysis(
            self,
            nodes: np.ndarray,
            model_outputs: dict,
            gt_labels: np.ndarray = None,
            save_path: str = None,
            show: bool = False
    ):
        """Plot detailed ablation data from a forward pass.
        
        Args:
            nodes (np.ndarray): Original (x,y,z,...) input nodes.
            model_outputs (dict): Dictionary of intermediate outputs from the model.
            gt_labels (np.ndarray, optional): Ground truth labels for the points.
            save_path (str, optional): Path to save the figure to.
            show (bool, optional): If True, shows the plot.
        """
        if model_outputs is None or "indices" not in model_outputs:
            print("No intermediate model outputs found. Cannot plot forward pass analysis.")
            return

        h_local = model_outputs.get("h_local") # [N, hidden]
        indices = model_outputs.get("indices") # [1, M]
        h_context = model_outputs.get("h_context") # [1, N, hidden]
        attn_weights = model_outputs.get("attn_weights") # [N, M]

        # Extract numpy arrays safely (detach if needed, but plotting might pass cpu detached tensors or numpy)
        # However, they usually come from PyTorch. Convert if tensor:
        import torch
        if isinstance(nodes, torch.Tensor):
            nodes = nodes.detach().cpu().numpy()
        
        pts = nodes[:, 0:2] # just (x, y)
        
        def to_npy(t):
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().numpy()
            return t
        
        h_local_npy = to_npy(h_local) if h_local is not None else None
        indices_npy = to_npy(indices) if indices is not None else None
        attn_weights_npy = to_npy(attn_weights) if attn_weights is not None else None

        fig, axs = plt.subplots(1, 5, figsize=(30, 6))
        fig.subplots_adjust(wspace=0.3, hspace=0.30)
        
        # Helper to plot rotated points 
        def transform_pts(points):
            p = np.hstack((points[:,0:2], np.zeros(shape=(points.shape[0],1))))
            r = Orientation.from_euler(yaw=90, degrees=True)
            t = Transformation(rotation=r._orientation)
            p = t.apply_transformation(p)
            return p[:, 0:2]

        pts_transformed = transform_pts(pts)

        # 1. Base Point Cloud
        axs[0].scatter(pts_transformed[:, 0], pts_transformed[:, 1], s=self.marker_size, color="blue", alpha=0.5, label="Input Points")
        axs[0].set_title("Original Input")
        
        # 2. Local Activations (Magnitude of h_local)
        if h_local_npy is not None:
            # norm along feature dim
            local_mags = np.linalg.norm(h_local_npy, axis=1)
            sc = axs[1].scatter(pts_transformed[:, 0], pts_transformed[:, 1], c=local_mags, cmap='viridis', s=self.marker_size, alpha=0.8)
            axs[1].set_title("Local Features (Magnitude)")
            plt.colorbar(sc, ax=axs[1], fraction=0.046, pad=0.04, label="L2 Norm")
        else:
            axs[1].set_title("No Local Features Available")

        # 3. Super-Nodes
        axs[2].scatter(pts_transformed[:, 0], pts_transformed[:, 1], s=self.marker_size, color="gray", alpha=0.3, label="Input Points")
        if indices_npy is not None:
            sn_pts = pts_transformed[indices_npy.flatten()]
            axs[2].scatter(sn_pts[:, 0], sn_pts[:, 1], s=self.marker_size * 3, marker='*', color="red", label="Super-nodes")
            axs[2].set_title(f"Selected Super-nodes (n={sn_pts.shape[0]})")
        else:
            axs[2].set_title("No Super-nodes Available")
        axs[2].legend()

        # 4. Attention Focus (Total Attention Weight per point)
        if attn_weights_npy is not None:
            # Sum attention across super-nodes to see overall focus
            total_attn = np.sum(attn_weights_npy, axis=1)
            sc = axs[3].scatter(pts_transformed[:, 0], pts_transformed[:, 1], c=total_attn, cmap='plasma', s=self.marker_size, alpha=0.8)
            axs[3].set_title("Cumulative Attention Focus")
            plt.colorbar(sc, ax=axs[3], fraction=0.046, pad=0.04, label="Total Attention")
        else:
            axs[3].set_title("No Attention Weights Available")

        # 5. Ground Truth
        if gt_labels is not None:
            self.plot_nodes(
                nodes=nodes,
                labels=gt_labels,
                ax=axs[4],
                title="Ground Truth",
                show=False
            )
        else:
            axs[4].set_title("No Ground Truth Available")

        for ax in axs:
            ax.set_xlim(left=-1 * self.plot_x_max, right=self.plot_x_max)
            ax.set_ylim(bottom=-1 * self.plot_y_max, top=self.plot_y_max)
            ax.set_xlabel("Y (m)", fontsize=self.font_size_axis_labels)
            ax.set_ylabel("X (m)", fontsize=self.font_size_axis_labels)
            ax.tick_params(labelsize=self.font_size_ticks)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Forward pass analysis saved to {save_path}")

        if show:
            plt.show()