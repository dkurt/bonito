import numpy as np
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node
from mo.ops.unsqueeze import Unsqueeze


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
        unsqueeze_node = create_op_with_const_inputs(graph, Unsqueeze, {1: int64_array([2])},
                                                     {'name': weights.soft_get('name', weights.id) + '/Unsqueeze'})
        weights.out_port(0).get_connection().insert_node(unsqueeze_node)
