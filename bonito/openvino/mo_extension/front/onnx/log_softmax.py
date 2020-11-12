# from extensions.ops.Log import LogOp
from extensions.ops.activation_ops import Log
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph, Node
from mo.ops.softmax import Softmax

class LogSoftmaxFrontReplacer(FrontReplacementOp):
    """
    Replace LogSoftmax operation by Softmax -> Log.
    """
    op = "LogSoftmax"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        axis = -1

        log = Log(graph, {'name': node.name + '/Log_'}).create_node()
        softmax = Softmax(graph, {'axis': axis, 'name': node.name + '/SoftMax_'}).create_node()

        # Connect nodes: input -> Softmax -> Log
        node.in_port(0).get_connection().set_destination(softmax.in_port(0))
        log.in_port(0).get_connection().set_source(softmax.out_port(0))

        # The "explicit" version of the return value is: [(out_node.id, 0)])
        return [log.id]
