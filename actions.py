from typing import List

import torch
from torch import nn


class Action:
    OPERATORS = {
        'cos': torch.cos,
        'sin': torch.sin,
        'exp': torch.exp,
        'square': torch.square,
        'sqrt': torch.sqrt,
        'log': torch.log
    }

    FUNCTIONS = {
        '*': torch.mul,
        '+': torch.add,
        '/': torch.div,
        '-': torch.sub
    }

    def __init__(self, num_features: int):
        self.feat_num = num_features + 1  # constant is considered
        self.op_num, self.fn_num = len(Action.OPERATORS), len(Action.FUNCTIONS)

        self.actions_dict = {"c": None}  # constant symbol
        self.actions_dict.update({
            **{f'x{idx + 1}': idx for idx in range(num_features)},  # features
            **Action.OPERATORS,  # operators
            **Action.FUNCTIONS,  # functions
        })

    @property
    def operator_num(self):
        return self.op_num

    @property
    def function_num(self):
        return self.fn_num

    @property
    def feature_num(self):
        return self.feat_num

    @property
    def action_names(self):
        return list(self.actions_dict.keys())

    @property
    def action_fns(self):
        return list(self.actions_dict.values())

    @property
    def action_arities(self):
        return self.feat_num * [0] + self.op_num * [1] + self.fn_num * [2]


class ActionNode(object):
    def __init__(self, index, action_fns, action_arities, action_names, c_index=None):
        self.name = action_names[index]
        self.fn = action_fns[index]
        self.arity = action_arities[index]
        self.left, self.right = None, None
        self.c_index = c_index  # for constant

    def add_child(self, child=None):
        if not self.left:
            self.left = child
        elif not self.right:
            self.right = child
        else:
            raise RuntimeError("adding more than 2 children")

    def expr(self, constants=None):
        if self.arity == 2:
            left_expr, right_expr = self.left.expr(constants), self.right.expr(constants)
            return f'{self.name}({left_expr}, {right_expr})'
        elif self.arity == 1:
            left_expr = self.left.expr(constants)
            return f'{self.name}({left_expr})'
        elif self.name == 'c':
            if constants is not None:
                return round(constants[self.c_index].item(), 2)
            else:
                return 'c'
        else:
            return self.name

    def eval(self, X, constants):
        if self.arity == 2:
            return self.fn(self.left.eval(X, constants), self.right.eval(X, constants))
        elif self.arity == 1:
            return self.fn(self.left.eval(X, constants))
        elif self.name == 'c':  # constant
            return constants[self.c_index]
        else:
            return X[:, self.fn]

    def __str__(self):
        return self.expr()

    def __repr__(self):
        return self.expr()


class ExpressionTree(nn.Module):
    def __init__(self, encoding, action_fns, action_arities, action_names):
        super(ExpressionTree, self).__init__()
        c_counter = 0
        nodes = {}
        for i in torch.nonzero(encoding >= 0).flatten().tolist():
            if encoding[i] == 0:  # constant
                nodes[i] = ActionNode(encoding[i], action_fns, action_arities, action_names, c_index=c_counter)
                c_counter += 1
            else:
                nodes[i] = ActionNode(encoding[i], action_fns, action_arities, action_names)
            if i > 0:
                par = (i - 1) // 2
                nodes[par].add_child(nodes[i])

        self.encoding = encoding
        self.constants = nn.Parameter(torch.rand(c_counter), requires_grad=True)
        self.root = nodes[0]

    def forward(self, X):
        return self.root.eval(X, self.constants)

    def __str__(self):
        return self.root.expr(self.constants)


class ETEnsemble(nn.Module):
    """
    An ensemble of expression trees (with constant terms)
    Credits to https://github.com/dandip/DSRPytorch/blob/main/expression_utils.py
    """
    def __init__(self, expressions: List[ExpressionTree]):
        super().__init__()
        expressions = [expression for expression in expressions if len(expression.constants) > 0]
        self.expressions = torch.nn.ModuleList(expressions)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        futures = [torch.jit.fork(expression, X) for expression in self.expressions]
        results = [torch.jit.wait(fut) for fut in futures]
        return torch.stack(results, dim=0)


def optimize_constant(model, X: torch.Tensor, y: torch.Tensor, inner_loop_config: dict):
    if inner_loop_config["loss"] == "mse":
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError("only mse is supported")

    optim = inner_loop_config["optim"]

    if isinstance(model, ExpressionTree):
        parameters = [model.constants]
    elif isinstance(model, ETEnsemble):
        parameters = [m.constants for m in model.expressions]
        y = y.repeat(len(model.expressions), 1)  # make sure the dimension aligns
    else:
        raise ValueError("Invalid model type")

    if not len(parameters):
        return

    if optim == 'lbfgs':
        optimizer = torch.optim.LBFGS(parameters)
        def closure():
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            return loss

        for _ in range(inner_loop_config["iteration"]):
            curr_loss = optimizer.step(closure)
            if torch.isnan(curr_loss) or torch.isinf(curr_loss):
                break

    else:
        if optim == 'rmsprop':
            optimizer = torch.optim.RMSprop(parameters, lr=inner_loop_config['lr'])
        elif optim == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=inner_loop_config['lr'])
        else:
            raise ValueError("Invalid optimizer type")

        for _ in range(inner_loop_config["iteration"]):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            if torch.isnan(loss) or torch.isinf(loss):
                break
            loss.backward()
            optimizer.step()


def evaluate_encodings(encodings, X, action_fns, action_arities, constants: dict = None):
    """
    Evaluate the expression tree represented by the encoding. The expression tree
    might be invalid even with constraints applied (e.g. taking log over negative
    values); errors will be thrown and used to discard any invalid tree.
    Let N, M, D, T be data size, batch size, feature size, and max tree size
    Args:
        encodings: a (M * T) encoding matrix
        X: a (N * D) data matrix
        action_fns: a list of action functions
        action_arities: a list of action arities
        constants: a dictionary of nn.Parameter for the constants
    Returns:
        y: a (M, N) matrix of the evaluation results
    """
    action_arities = torch.tensor(action_arities)
    batch_size, tree_size = encodings.shape

    results = torch.zeros((batch_size, X.shape[0], tree_size))
    for i in reversed(range(tree_size)):
        non_empty_mask = encodings[:, i] >= 0
        if not non_empty_mask.any():
            continue
        non_empty_idx = non_empty_mask.nonzero().squeeze()
        non_empty_ecds = encodings[non_empty_mask, i]
        non_empty_arities = action_arities[non_empty_ecds]

        use_feat = non_empty_arities == 0
        for idx in non_empty_idx[use_feat]:
            action_idx = encodings[idx, i]
            if action_idx == 0:  # constant case
                results[idx, :, i] = constants[(idx, i)]
            else:
                # action_idx - 1 since constant takes 0 index
                results[idx, :, i] = X[:, action_idx - 1]

        use_op = non_empty_arities == 1
        for idx in non_empty_idx[use_op]:
            action_idx = encodings[idx, i]
            left_results = results[idx, :, 2 * i + 1]
            results[idx, :, i] = action_fns[action_idx](left_results)

        use_fn = non_empty_arities == 2
        for idx in non_empty_idx[use_fn]:
            action_idx = encodings[idx, i]
            left_results = results[idx, :, 2 * i + 1]
            right_results = results[idx, :, 2 * i + 2]
            results[idx, :, i] = action_fns[action_idx](left_results, right_results)

    y = results[:, :, 0]
    return y


def get_next_node_indices(encodings, placeholder: int = -2):
    """
    A placeholder in `encodings` takes value -2, while uninitialized values use -1

    Args:
        encodings: a (M * T) encoding matrix
        placeholder: the value identifier for an initialized (but not assigned) node
    Returns:
        nodes_to_assign: a (M, ) vector of node indices to apply next action
        siblings: a (M, ) vector of siblings of `nodes_to_assign`, 0 if not exist
        parents: a (M, ) vector of parents of `nodes_to_assign`, 0 if not exist
    """
    batch_size, _ = encodings.shape
    siblings = torch.ones(batch_size, dtype=torch.long) * placeholder

    # get the indices of most recent nodes (to be assigned value) for each sample
    # we assume that the tree cannot be fully initialized
    nodes_to_assign = (encodings == placeholder).long().argmax(axis=1)

    parent_idx = torch.div(nodes_to_assign - 1, 2, rounding_mode='floor').clamp(min=0)
    parents = encodings[torch.arange(batch_size), parent_idx]
    is_right_node = nodes_to_assign % 2 == 0
    sibling_idx = (nodes_to_assign[is_right_node] - 1).clamp(min=0)
    siblings[is_right_node] = encodings[is_right_node, sibling_idx]
    return nodes_to_assign, siblings, parents


# some local test on the above
if __name__ == "__main__":
    # X = torch.randn((5, 2))
    # print(X)
    # encodings = torch.tensor([[8, 0, 1], [9, 0, 1]])
    # actions = Action(2)
    # print(evaluate_encodings(encodings, X, actions.action_fns, actions.action_arities))
    # encodings = torch.tensor([[-2, -2, 2], [0, -2, 1], [3, 2, -2], [1, 1, 2]])
    X = torch.ones((5, 2))
    X[:, 1] = 2
    action = Action(2)
    encoding = torch.tensor([9, 0, 1])
    tree = ExpressionTree(encoding, action.action_fns, action.action_arities, action.action_names)
