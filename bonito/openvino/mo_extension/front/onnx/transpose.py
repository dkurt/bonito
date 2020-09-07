# mo_extensions/front/onnx/conv2d.py
import numpy as np

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph, Node
from extensions.ops.transpose import Transpose
from mo.front.extractor import FrontExtractorOp

class Transpose2d(FrontExtractorOp):
    op = 'Transpose'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Transpose.update_node_stat(node, {'order': [0, 3, 2, 1]})
        return cls.enabled
