# mo_extensions/front/onnx/conv2d.py
import numpy as np

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph, Node

class Conv1dToConv2d(FrontReplacementSubgraph):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('conv', dict(op='Conv')),
                ('weights', dict(op='Const')),
            ],
            edges=[
                ('weights', 'conv', {'in': 1})
            ])

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict):
        conv = match['conv']
        conv['pad'] = np.insert(conv['pad'], 2, [0, 0], axis=0)
        conv['stride'] = np.insert(conv['stride'], 2, 1)
        conv['dilation'] = np.insert(conv['dilation'], 2, 1)
        conv['kernel_spatial'] = np.insert(conv['kernel_spatial'], 0, 1)

        weights = match['weights']
        weights['shape'] = np.insert(weights['shape'], 2, 1)
        weights['pb'].dims.insert(2, 1)
        weights['value'] = np.expand_dims(weights['value'], axis=2)
