import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig


class EHCDRSExplainer:
    def __init__(self, model, epochs=200, lr=0.01):
        self.model = model

        self.explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=epochs, lr=lr),

            explanation_type='model',

            node_mask_type='attributes',
            edge_mask_type='object',

            # ✅ NEW REQUIRED CONFIG
            model_config=ModelConfig(
                mode='multiclass_classification',
                task_level='node',   # since NodeRecommender
                return_type='raw'
            )
        )

    def explain(self, data, node_idx):
        explanation = self.explainer(
            x=data.x,
            edge_index=data.edge_index,
            index=node_idx
        )

        edge_mask = explanation.edge_mask
        node_mask = explanation.node_mask

        return edge_mask, node_mask