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
        slef.value = None

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