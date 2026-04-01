import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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
        h_super_context_npy = to_npy(model_outputs.get("h_super_context"))
        super_edge_index_npy = to_npy(model_outputs.get("super_edge_index"))
        intra_patch_attn_npy = to_npy(model_outputs.get("intra_patch_attn"))
        attn_weights_npy = to_npy(attn_weights) if attn_weights is not None else None

        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        fig.subplots_adjust(wspace=0.3, hspace=0.30)
        
        # Helper to plot rotated points 
        def transform_pts(points):
            p = np.hstack((points[:,0:2], np.zeros(shape=(points.shape[0],1))))
            r = Orientation.from_euler(yaw=90, degrees=True)
            t = Transformation(rotation=r._orientation)
            p = t.apply_transformation(p)
            return p[:, 0:2]

        pts_transformed = transform_pts(pts)

        # [0, 0] 1. Base Point Cloud
        axs[0, 0].scatter(pts_transformed[:, 0], pts_transformed[:, 1], s=self.marker_size, color="blue", alpha=0.5, label="Input Points")
        axs[0, 0].set_title("1. Original Input")
        
        # [0, 1] 2. Local Activations (Magnitude of h_local)
        if h_local_npy is not None:
            local_mags = np.linalg.norm(h_local_npy, axis=1)
            sc = axs[0, 1].scatter(pts_transformed[:, 0], pts_transformed[:, 1], c=local_mags, cmap='viridis', s=self.marker_size, alpha=0.8)
            axs[0, 1].set_title("2. Local Features (Magnitude)")
            plt.colorbar(sc, ax=axs[0, 1], fraction=0.046, pad=0.04, label="L2 Norm")
        else:
            axs[0, 1].set_title("2. No Local Features")

        # [0, 2] 3. Ground Truth
        if gt_labels is not None:
            self.plot_nodes(
                nodes=nodes,
                labels=gt_labels,
                ax=axs[0, 2],
                title="3. Ground Truth",
                show=False
            )
        else:
            axs[0, 2].set_title("3. No Ground Truth")

        # [1, 0] 4. Selected Super-Nodes
        axs[1, 0].scatter(pts_transformed[:, 0], pts_transformed[:, 1], s=self.marker_size, color="gray", alpha=0.3, label="Input Points")
        if indices_npy is not None:
            sn_pts = pts_transformed[indices_npy.flatten()]
            axs[1, 0].scatter(sn_pts[:, 0], sn_pts[:, 1], s=self.marker_size * 3, marker='*', color="red", label="Super-nodes")
            axs[1, 0].set_title(f"4. Selected Super-nodes (n={sn_pts.shape[0]})")
        else:
            axs[1, 0].set_title("4. No Super-nodes")
        axs[1, 0].legend()

        # [1, 1] 5. Smart Super-Nodes (Macro-Reasoning / PTv3)
        axs[1, 1].scatter(pts_transformed[:, 0], pts_transformed[:, 1], s=self.marker_size, color="gray", alpha=0.1)
        if h_super_context_npy is not None and indices_npy is not None:
            sn_pts = pts_transformed[indices_npy.flatten()]
            
            if intra_patch_attn_npy is not None:
                # --- Plot PTv3 Intra-Patch Attention Focus ---
                sc = axs[1, 1].scatter(sn_pts[:, 0], sn_pts[:, 1], c=intra_patch_attn_npy, cmap='plasma', s=self.marker_size * 4, marker='*', alpha=1.0, zorder=2,
                                        vmin=intra_patch_attn_npy.min(), vmax=intra_patch_attn_npy.max())
                axs[1, 1].set_title("5. PTv3 Intra-Patch Focus")
                plt.colorbar(sc, ax=axs[1, 1], fraction=0.046, pad=0.04, label="Patch Focal Weight")
            else:
                # --- Plot Traditional Edges & Macro-Reasoning ---
                if super_edge_index_npy is not None:
                    # Map indices back to the coordinate set of super-nodes
                    src_pts = sn_pts[super_edge_index_npy[0]] # [E, 2]
                    dst_pts = sn_pts[super_edge_index_npy[1]] # [E, 2]
                    segments = np.stack([src_pts, dst_pts], axis=1) # [E, 2, 2]
                    
                    lc = LineCollection(segments, colors='gray', alpha=0.3, linewidths=0.5, zorder=1)
                    axs[1, 1].add_collection(lc)

                smart_mags = np.linalg.norm(h_super_context_npy, axis=1)
                sc = axs[1, 1].scatter(sn_pts[:, 0], sn_pts[:, 1], c=smart_mags, cmap='plasma', s=self.marker_size * 4, marker='*', alpha=1.0, zorder=2,
                                        vmin=smart_mags.min(), vmax=smart_mags.max())
                axs[1, 1].set_title("5. Smart Super-Nodes (Macro-GNN)")
                plt.colorbar(sc, ax=axs[1, 1], fraction=0.046, pad=0.04, label="GNN Activation")
        else:
            axs[1, 1].set_title("5. Super-Nodes Unavailable")

        # [1, 2] 6. Max Attention Focus
        if attn_weights_npy is not None:
            # Show the strongest attention link for each point to see specific regions focused on
            max_attn = np.max(attn_weights_npy, axis=1)
            # DYNAMIC SCALING
            sc = axs[1, 2].scatter(pts_transformed[:, 0], pts_transformed[:, 1], c=max_attn, cmap='magma', s=self.marker_size, alpha=0.8,
                                    vmin=max_attn.min(), vmax=max_attn.max())
            axs[1, 2].set_title("6. Max Attention Focus")
            plt.colorbar(sc, ax=axs[1, 2], fraction=0.046, pad=0.04, label="Max Attention")
        else:
            axs[1, 2].set_title("6. No Attention Weights")

        for ax in axs.flatten():
            ax.set_xlim(left=-1 * self.plot_x_max, right=self.plot_x_max)
            ax.set_ylim(bottom=-1 * self.plot_y_max, top=self.plot_y_max)
            ax.set_xlabel("Y (m)", fontsize=self.font_size_axis_labels)
            ax.set_ylabel("X (m)", fontsize=self.font_size_axis_labels)
            ax.tick_params(labelsize=self.font_size_ticks)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Deep EdgeConv analysis saved to {save_path}")

        if show:
            plt.show()

    def plot_deep_edge_conv_diagnostic(
            self,
            nodes: np.ndarray,
            gt_labels: np.ndarray,
            predictions: np.ndarray,
            save_path: str = None,
            show: bool = False
    ):
        """Figure 1: Diagnostic overview (Input, GT, Prediction)."""
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.subplots_adjust(wspace=0.3)

        def transform_pts(points):
            p = np.hstack((points[:,0:2], np.zeros(shape=(points.shape[0],1))))
            r = Orientation.from_euler(yaw=90, degrees=True)
            t = Transformation(rotation=r._orientation)
            p = t.apply_transformation(p)
            return p[:, 0:2]

        pts_transformed = transform_pts(nodes)

        # 1.1 Original Input
        axs[0].scatter(pts_transformed[:, 0], pts_transformed[:, 1], s=self.marker_size, color="blue", alpha=0.5)
        axs[0].set_title("1. Original Input")
        
        # 1.2 Ground Truth
        if gt_labels is not None:
            valid_mask = (gt_labels == 1.0)
            axs[1].scatter(pts_transformed[~valid_mask, 0], pts_transformed[~valid_mask, 1], s=self.marker_size, color="gray", alpha=0.3)
            axs[1].scatter(pts_transformed[valid_mask, 0], pts_transformed[valid_mask, 1], s=self.marker_size, color="red", alpha=0.8, label="GT Valid")
            axs[1].set_title("2. Ground Truth")
            axs[1].legend()

        # 1.3 Final Prediction
        import torch
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        sc = axs[2].scatter(pts_transformed[:, 0], pts_transformed[:, 1], c=predictions.flatten(), cmap='RdYlGn', s=self.marker_size, alpha=0.8)
        axs[2].set_title("3. Model Prediction")
        plt.colorbar(sc, ax=axs[2], label="Score")

        for ax in axs:
            ax.set_xlim(left=-1 * self.plot_x_max, right=self.plot_x_max)
            ax.set_ylim(bottom=-1 * self.plot_y_max, top=self.plot_y_max)
            ax.set_xlabel("Y (m)", fontsize=self.font_size_axis_labels)
            ax.set_ylabel("X (m)", fontsize=self.font_size_axis_labels)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()

    def plot_deep_edge_conv_layer_analysis(
            self,
            nodes: np.ndarray,
            model_outputs: dict,
            layer_predictions: list,
            super_node_indices: np.ndarray = None,
            save_path: str = None,
            show: bool = False
    ):
        """Figure 2: Layer-wise graph and ablation readout."""
        layer_indices = sorted(list(set([int(k.split('_')[1]) for k in model_outputs.keys() if k.startswith('layer_') and k.endswith('_features')])))
        num_layers = len(layer_indices)
        
        fig, axs = plt.subplots(2, num_layers, figsize=(6 * num_layers, 12))
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        
        # Ensure axs is always 2D
        if num_layers == 1:
            axs = np.expand_dims(axs, axis=1)

        def transform_pts(points):
            p = np.hstack((points[:,0:2], np.zeros(shape=(points.shape[0],1))))
            r = Orientation.from_euler(yaw=90, degrees=True)
            t = Transformation(rotation=r._orientation)
            p = t.apply_transformation(p)
            return p[:, 0:2]

        pts_transformed = transform_pts(nodes)
        
        # If super-nodes are a subset, we need their transformed coords for the graph
        if super_node_indices is not None:
            pts_sn = pts_transformed[super_node_indices]
        else:
            pts_sn = pts_transformed

        import torch
        for idx, layer_idx in enumerate(layer_indices):
            # --- TOP ROW: Global k-NN Graph (on super-nodes if indices provided) ---
            ax_top = axs[0, idx]
            feat_key = f"layer_{layer_idx}_features"
            edge_key = f"layer_{layer_idx}_edge_index"
            
            features = model_outputs.get(feat_key)
            if isinstance(features, torch.Tensor):
                features = features.detach().cpu().numpy()
            
            # Draw Edges BEHIND (zorder=1)
            edge_index = model_outputs.get(edge_key)
            if edge_index is not None:
                if isinstance(edge_index, torch.Tensor):
                    edge_index = edge_index.detach().cpu().numpy()
                
                # Plot edges using the appropriate point set
                src_pts = pts_sn[edge_index[0]]
                dst_pts = pts_sn[edge_index[1]]
                segments = np.stack([src_pts, dst_pts], axis=1)
                
                lc = LineCollection(segments, colors='blue', alpha=0.1, linewidths=0.3, zorder=1)
                ax_top.add_collection(lc)

            # Draw Nodes ON TOP (zorder=2)
            feats_mag = np.linalg.norm(features, axis=1) if features is not None else np.zeros(pts_sn.shape[0])
            sc = ax_top.scatter(pts_sn[:, 0], pts_sn[:, 1], c=feats_mag, cmap='viridis', s=self.marker_size * 1.5, alpha=0.9, zorder=2)
            ax_top.set_title(f"Layer {layer_idx}: Super-Node Graph\n(Color by Activation)")
            if idx == num_layers - 1:
                plt.colorbar(sc, ax=ax_top, label="Magnitude")

            # --- BOTTOM ROW: Ablation Readout (Full point cloud predictions) ---
            ax_bot = axs[1, idx]
            pred = layer_predictions[idx]
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
            
            sc_p = ax_bot.scatter(pts_transformed[:, 0], pts_transformed[:, 1], c=pred.flatten(), cmap='RdYlGn', s=self.marker_size, alpha=0.8)
            ax_bot.set_title(f"Layer {layer_idx}: Full Prediction\n(Layers 0 to {layer_idx})")
            if idx == num_layers - 1:
                plt.colorbar(sc_p, ax=ax_bot, label="Score")

        for ax in axs.flatten():
            ax.set_xlim(left=-1 * self.plot_x_max, right=self.plot_x_max)
            ax.set_ylim(bottom=-1 * self.plot_y_max, top=self.plot_y_max)
            ax.set_xlabel("Y (m)", fontsize=self.font_size_axis_labels)
            ax.set_ylabel("X (m)", fontsize=self.font_size_axis_labels)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()