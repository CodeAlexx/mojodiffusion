# autograd_v2/graph.mojo - Graph: the node tenure table + minimal recording
# surface. Port of flame-core src/autograd_v2/ recording.rs + meta.rs reduced
# per contract C3 (AUTOGRAD_V2_MOJO_DESIGN.md):
#
#  * ALL nodes live in Graph.nodes (List[Node]) - the tenure table. Tensors
#    carry only integer ids (Tensor.id, 0 = untracked); the tensor->grad_fn
#    link flame keeps in autograd_meta (engine.rs:52-64 grad_fn_of /
#    output_nr_of) is two Dicts here: tensor id -> producing node idx and
#    tensor id -> output_nr.
#  * Leaf accumulators are looked up by param id in leaf_of_param (the Weak
#    cycle-break of accumulator.rs:27-105 becomes a plain table; no
#    ArcPointer[Node] is constructible).
#  * Recording surface is MINIMAL for P1: one generic `record` covering the
#    four non-leaf kinds; the toy-gate harness composes graphs through it.
#    Per-op record_* wrappers (frozen-skip gate C7, recording.rs:87-95) are
#    P2 scope (ops_record.mojo).

from std.collections import Dict
from serenitymojo.autograd_v2.node import Edge, Node, TArc, OPK_LEAF


struct Graph(Movable):
    var nodes: List[Node]
    var seq_counter: Int
    # Tensor-id allocator for the recording surface. Starts at 1; 0 is the
    # untracked sentinel (tensor.mojo Tensor.id contract). P1 test tensors get
    # ids via t.set_id(graph.fresh_tensor_id()).
    var id_counter: Int
    # param tensor id -> leaf (OPK_LEAF) node idx (contract C3).
    var leaf_of_param: Dict[Int, Int]
    # tensor id -> producing node idx / output_nr on that node (the Mojo
    # stand-in for flame meta.grad_fn / meta.output_nr, engine.rs:52-64).
    var node_of_tensor: Dict[Int, Int]
    var outnr_of_tensor: Dict[Int, Int]

    def __init__(out self):
        self.nodes = List[Node]()
        self.seq_counter = 0
        self.id_counter = 1
        self.leaf_of_param = Dict[Int, Int]()
        self.node_of_tensor = Dict[Int, Int]()
        self.outnr_of_tensor = Dict[Int, Int]()

    def fresh_tensor_id(mut self) -> Int:
        var id = self.id_counter
        self.id_counter += 1
        return id

    def leaf(mut self, param_id: Int) raises -> Int:
        """Get-or-create the OPK_LEAF accumulator node for a parameter (flame
        AccumulateGrad construction, accumulator.rs:59-77): no outgoing edges
        (terminal sink), num_inputs = 1 (accumulator.rs:208-210),
        topological_nr = 0 (leaves anchor the numbering, accumulator.rs:69-72).
        Also registers the param tensor's gradient edge as (leaf, slot 0) -
        flame's gradient_edge() on a leaf yields Edge{AccumulateGrad, 0}."""
        if param_id <= 0:
            raise Error("Graph.leaf: param_id must be a tracked (nonzero) tensor id")
        if self.leaf_of_param.__contains__(param_id):
            return self.leaf_of_param[param_id]
        var idx = len(self.nodes)
        self.nodes.append(
            Node(
                OPK_LEAF,
                List[Edge](),
                List[TArc](),
                List[Int](),
                List[Float32](),
                1,                 # num_inputs (accumulator.rs:208-210)
                idx,               # node_id == tenure-table index
                self.seq_counter,
                0,                 # topological_nr (accumulator.rs:69-72)
                param_id,
            )
        )
        self.seq_counter += 1
        self.leaf_of_param[param_id] = idx
        self.node_of_tensor[param_id] = idx
        self.outnr_of_tensor[param_id] = 0
        return idx

    def edge_for(self, tensor_id: Int) raises -> Edge:
        """Gradient edge for a tensor id: null edge if untracked (id 0 or no
        producer recorded) - flame Edge::null() / grad drop, node.rs:66-71."""
        if tensor_id == 0 or not self.node_of_tensor.__contains__(tensor_id):
            return Edge.null()
        return Edge(self.node_of_tensor[tensor_id], self.outnr_of_tensor[tensor_id])

    def record(
        mut self,
        kind: Int,
        var input_edges: List[Edge],
        var saved: List[TArc],
        var saved_meta: List[Int],
        var scalars: List[Float32],
        out_ids: List[Int],
    ) raises -> Int:
        """Generic recording for the non-leaf P1 kinds: append a Node whose
        edges route grads back to the producers of its forward inputs, and
        register each forward-output tensor id -> (this node, output_nr).

        num_inputs = len(out_ids): the node's forward OUTPUTS are the backward
        node's INPUTS (engine.rs:290-303 comment block).
        topological_nr = 1 + max over valid input edges' target topo (0 when
        there are none) - flame stored-field numbering, node.rs:141-144.
        sequence_nr = seq_counter++ (recording order, node.rs:137-139).

        C15 (slot-ordered fan-in): every VALID input edge gets its
        contrib_slot assigned HERE, in registration order per target
        (child, input_nr) - read-then-increment of the target node's
        contrib_counts. Registration order is the recording wrappers' forward
        order, which is the hand-chain's fan-in fold order; the engine's
        InputBuffer reduces contributions in ascending contrib_slot order so
        bf16 fan-in sums reproduce the hand-chain bit-for-bit."""
        if kind == OPK_LEAF:
            raise Error("Graph.record: OPK_LEAF nodes are created via Graph.leaf")
        if len(out_ids) == 0:
            raise Error("Graph.record: node must have at least one output id")
        var max_topo = 0
        for i in range(len(input_edges)):
            var child = input_edges[i].node_idx
            if child >= 0:
                if child >= len(self.nodes):
                    raise Error("Graph.record: edge to unknown node idx")
                var slot = input_edges[i].input_nr
                if slot < 0 or slot >= self.nodes[child].num_inputs:
                    raise Error("Graph.record: edge input_nr out of bounds")
                # C15: assign this contribution's fan-in slot (registration
                # order per (child, input_nr) target).
                var cs = self.nodes[child].contrib_counts[slot]
                input_edges[i].contrib_slot = cs
                self.nodes[child].contrib_counts[slot] = cs + 1
                var t = self.nodes[child].topological_nr
                if t > max_topo:
                    max_topo = t
        var idx = len(self.nodes)
        var num_inputs = len(out_ids)
        self.nodes.append(
            Node(
                kind,
                input_edges^,
                saved^,
                saved_meta^,
                scalars^,
                num_inputs,
                idx,
                self.seq_counter,
                1 + max_topo,
            )
        )
        self.seq_counter += 1
        for i in range(len(out_ids)):
            self.node_of_tensor[out_ids[i]] = idx
            self.outnr_of_tensor[out_ids[i]] = i
        return idx
