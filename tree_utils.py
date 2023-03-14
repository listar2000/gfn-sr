from binarytree import Node, build
import torch

DEMO_LABELS = [
    None, "X0", "X1",  # features
    "+", "*", "/", "-",  # binary ops
    "sin", "cos", "exp", "log", "abs"  # unary ops
]

LABELS_MAPPING = {
    "X0": 0, "X1": 1,
    "+": torch.add, "*": torch.mul, "/": torch.div, "-": torch.sub,
    "sin": torch.sin, "cos": torch.cos, "log": torch.log, "abs": torch.abs
}


def encoding_to_tree(encoding: torch.Tensor):
    labeled = [DEMO_LABELS[i] for i in encoding.long()]
    return build(labeled)


def evaluate(tree: Node, X: torch.Tensor):
    if not tree.left or not tree.left.val:
        return X[:, LABELS_MAPPING[tree.val]]
    elif not tree.right or not tree.right.val:
        op = LABELS_MAPPING[tree.val]
        return op(evaluate(tree.left, X))
    else:
        op = LABELS_MAPPING[tree.val]
        return op(evaluate(tree.left, X), evaluate(tree.right, X))
