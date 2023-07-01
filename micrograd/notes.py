from graphviz import Digraph


class Value:
    def __init__(self, data, _children=(), _op="") -> None:
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # The plus equal is to handle a + a
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        # Now out has a method that can correctly pass its
        # own grad to its children via chain rule.
        # Note how the _backward method is determined by the
        # operation that leads to out.
        # This function go modifies the attributes in self and
        # other, which are passed to __add__ as arguments - not
        # very functional programming!
        # Also, unclear how _backward store self and other, probably
        # some absolute addresses.
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            # serch for children, until no children, append
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            # we are thinking backward, the child actually comes first
            # in the mathematical operation
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root, format="svg", rankdir="LR"):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ["LR", "TB"]
    nodes, edges = trace(root)
    dot = Digraph(
        format=format, graph_attr={"rankdir": rankdir}
    )  # , node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(
            name=str(id(n)),
            label="{ data %.4f | grad %.4f }" % (n.data, n.grad),
            shape="record",
        )
        if n._op:
            # if there's and _op, create an op node and
            # link it to the parent
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        # Recall that edges are tupples linking parent with child
        # Now we link every child with the the operation node
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


if __name__ == "__main__":
    a = Value(3)
    b = Value(4)
    c = a * b
    d = Value(2)
    e = c + d

    draw_dot(e)

    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(e)
