# A simplified tensorflow like DNN engine

class Node:
    def __init__(self, inbound_nodes=[]):
        # Node(s) that transmit input values
        self.inbound_nodes = inbound_nodes
        # Node(s) that recieves output values
        self.outbound_nodes = []
        # append this node to the outbound_nodes of input
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
        # The calculated value
        self.value = None

    def forward(self):
        """
        Forward propagation.
        Compute value based on inbound_nodes
        """
        raise NotImplemented

class Input(Node):
    def __init__(self):
        # Input node does not have inbound_nodes
        Node.__init__(self)

    # Note: input node is the only type where value may be passed
    # All other node types should calculate values based on inbound_nodes
    def forward(self, value=None):
        if value is not None:
            self.value = value;

class Add(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x , y])

    def forward(self):
        """
        Forward propagation by summing the values of input nodes
        """
        self.value = 0
        for n in self.inbound_nodes:
            self.value += n.value

class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

    def forward(self):
        wightedInputs = sum([w*x for w,x in zip(self.inbound_nodes[1].value, self.inbound_nodes[0].value)])
        self.value = wightedInputs + self.inbound_nodes[2].value

def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value