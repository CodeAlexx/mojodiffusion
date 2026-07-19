# =============================================================================
# node_graph.mojo  --  NODE GRAPH EDITOR (LITE)
# Mojo 1.0.0b1 port of MediaEditor's blueprintsdk node editor, drawn DIRECTLY on
# the MojoGUI C backend (librender.so) via external_call ONLY. No other Mojo dep.
#
# BEHAVIOR REFERENCE (cited lines are in the MediaEditor tree under /tmp/Med):
#   /tmp/Med/blueprintsdk/include/Node.h
#     :111-279  struct Node -> m_ID(264), m_Name(265), GetInputPins()/GetOutputPins()
#               (243-244), OnDragStart/OnDragging/OnDragEnd (138-140) -> draggable.
#   /tmp/Med/blueprintsdk/include/Pin.h
#     :6-14     PIN_FLAG_IN(1<<0), PIN_FLAG_OUT(1<<1), PIN_FLAG_LINKED(1<<4).
#     :132-173  struct Pin -> m_ID(162), m_Node(163), m_Name(165), m_Link(166)
#               (receiver stores provider id), m_Flags(167), m_LinkFrom(169).
#     :150-153  IsInput()/IsOutput(); :145 LinkTo(provider->receiver).
#   /tmp/Med/blueprintsdk/src/UI.cpp
#     :1215-1224 "link id is same as pin id"; receiver pin's m_Link = provider id,
#               provider pushed into receiver.m_LinkFrom; PIN_FLAG_LINKED set.
#               Link is rendered output-pin -> input-pin (ed::Link, bezier).
#
# A Link here = (from_pin_id [an OUTPUT pin], to_pin_id [an INPUT pin]) created by
# dragging FROM an output pin TO an input pin -- exactly UI.cpp's provider->receiver.
#
# BEZIER NOTE: links are cubic beziers (immediate-mode-node-editor draws curved links). We
# evaluate B(t) with horizontal control handles and chain draw_line between samples
# -- only +,-,* used, no transcendentals (same safe-symbol rule as the exemplar).
# =============================================================================
from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from builtin.type_aliases import MutExternalOrigin


# ---------- backend FFI helpers (idioms copied from keyframe_editor.mojo) ----------
def to_cstr(s: String) -> UnsafePointer[Int8, MutExternalOrigin]:
    var b = s.as_bytes()
    var buf = alloc[Int8](len(b) + 1)
    for i in range(len(b)):
        buf[i] = Int8(b[i])
    buf[len(b)] = 0
    return buf


def col(r: Int, g: Int, b: Int):
    _ = external_call["set_color", Int32](
        Float32(r) / 255.0, Float32(g) / 255.0, Float32(b) / 255.0, Float32(1.0)
    )


def fill(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["draw_filled_rectangle", Int32](x, y, w, h)


def rect(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["draw_rectangle", Int32](x, y, w, h)


def line_t(x1: Float32, y1: Float32, x2: Float32, y2: Float32, thick: Float32):
    _ = external_call["draw_line", Int32](x1, y1, x2, y2, thick)


def fcircle(x: Float32, y: Float32, radius: Float32, segs: Int):
    _ = external_call["draw_filled_circle", Int32](x, y, radius, Int32(segs))


def txt(s: String, x: Float32, y: Float32, size: Float32):
    _ = external_call["draw_text", Int32](to_cstr(s), x, y, size)


# ---------- pin kind (Pin.h:6-14 PIN_FLAG_IN/OUT) ----------
comptime PIN_IN: Int = 0     # PIN_FLAG_IN  (1<<0) -> receiver side
comptime PIN_OUT: Int = 1    # PIN_FLAG_OUT (1<<1) -> provider side


# ---------- model ----------
# Pin: id, name, kind(in/out), and the parent node index it belongs to. Its screen
# connection point is computed by the owning Node at draw time. (Pin.h:132-173)
@fieldwise_init
struct Pin(ImplicitlyCopyable, Movable):
    var id: Int
    var name: String
    var kind: Int        # PIN_IN | PIN_OUT
    var node_idx: Int    # index of owning node (mirrors Pin::m_Node)

    fn copy(self) -> Self:
        return Pin(self.id, self.name, self.kind, self.node_idx)


# Node: id, title, rect (x,y,w,h), input pins, output pins. (Node.h:111-279)
struct Node(Copyable, Movable):
    var id: Int
    var title: String
    var x: Float32
    var y: Float32
    var w: Float32
    var h: Float32
    var inputs: List[Pin]
    var outputs: List[Pin]

    comptime TITLE_H: Float32 = 22.0     # title-bar height (drag handle)
    comptime PIN_DY: Float32 = 20.0      # vertical spacing between pin rows
    comptime PIN_TOP: Float32 = 14.0     # first pin offset below the title bar
    comptime PIN_R: Float32 = 5.0        # pin dot radius

    def __init__(out self, id: Int, var title: String, x: Float32, y: Float32, w: Float32, h: Float32):
        self.id = id
        self.title = title^
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.inputs = List[Pin]()
        self.outputs = List[Pin]()

    def add_input(mut self, pid: Int, var name: String, node_idx: Int):
        self.inputs.append(Pin(pid, name^, PIN_IN, node_idx))

    def add_output(mut self, pid: Int, var name: String, node_idx: Int):
        self.outputs.append(Pin(pid, name^, PIN_OUT, node_idx))

    # connection point of input pin i: on the LEFT edge, stacked top-down.
    def in_pin_x(self, i: Int) -> Float32:
        return self.x

    def in_pin_y(self, i: Int) -> Float32:
        return self.y + Self.TITLE_H + Self.PIN_TOP + Float32(i) * Self.PIN_DY

    # connection point of output pin i: on the RIGHT edge.
    def out_pin_x(self, i: Int) -> Float32:
        return self.x + self.w

    def out_pin_y(self, i: Int) -> Float32:
        return self.y + Self.TITLE_H + Self.PIN_TOP + Float32(i) * Self.PIN_DY

    # is (mx,my) on the title bar (the drag handle)? (Node.h:138 OnDragStart)
    def hit_title(self, mx: Float32, my: Float32) -> Bool:
        return (
            mx >= self.x
            and mx <= self.x + self.w
            and my >= self.y
            and my <= self.y + Self.TITLE_H
        )


# Link: provider OUTPUT pin -> receiver INPUT pin. (Pin.h:166 m_Link; UI.cpp:1215)
@fieldwise_init
struct Link(ImplicitlyCopyable, Movable):
    var from_pin_id: Int   # an OUTPUT pin id (provider)
    var to_pin_id: Int     # an INPUT pin id  (receiver)

    fn copy(self) -> Self:
        return Link(self.from_pin_id, self.to_pin_id)


# A 2D point (used as a small return value, avoids mut out-params).
@fieldwise_init
struct P2(ImplicitlyCopyable, Movable):
    var x: Float32
    var y: Float32


# Result of an OUTPUT-pin hit-test: pin id (-1 none) + its connection point.
@fieldwise_init
struct PinHit(ImplicitlyCopyable, Movable):
    var id: Int
    var x: Float32
    var y: Float32


# ---------- bezier helper (cubic) ----------
# Horizontal-handle cubic between (x0,y0) and (x1,y1), eval at t. Only +,-,*.
def bezier_xy(x0: Float32, y0: Float32, x1: Float32, y1: Float32, t: Float32) -> P2:
    var dx = x1 - x0
    var handle = dx * Float32(0.5)
    if handle < Float32(40.0):
        handle = Float32(40.0)
    var c0x = x0 + handle
    var c0y = y0
    var c1x = x1 - handle
    var c1y = y1
    var u = Float32(1.0) - t
    var b0 = u * u * u
    var b1 = Float32(3.0) * u * u * t
    var b2 = Float32(3.0) * u * t * t
    var b3 = t * t * t
    return P2(b0 * x0 + b1 * c0x + b2 * c1x + b3 * x1, b0 * y0 + b1 * c0y + b2 * c1y + b3 * y1)


# ---------- editor ----------
struct NodeGraph(Copyable, Movable):
    var nodes: List[Node]
    var links: List[Link]
    # drag state
    var drag_node: Int       # node index being dragged by title, -1 none
    var drag_dx: Float32     # grab offset (mouse - node.x)
    var drag_dy: Float32
    # link-creation state
    var link_from_pin: Int   # output pin id we started dragging a link from, -1 none
    var link_from_x: Float32 # cached start point for rubber-band line
    var link_from_y: Float32
    var prev_down: Bool

    comptime BEZ_SAMPLES: Int = 18   # samples per link bezier
    comptime PIN_HIT_R2: Float32 = 100.0   # pin grab radius^2 (10px)

    def __init__(out self):
        self.nodes = List[Node]()
        self.links = List[Link]()
        self.drag_node = -1
        self.drag_dx = 0.0
        self.drag_dy = 0.0
        self.link_from_pin = -1
        self.link_from_x = 0.0
        self.link_from_y = 0.0
        self.prev_down = False

    # find OUTPUT pin under cursor -> PinHit(id,x,y); id=-1 when none. SQUARED dist.
    def hit_output_pin(self, mx: Float32, my: Float32) -> PinHit:
        var best = Float32(Self.PIN_HIT_R2)
        var rid = -1
        var rx = Float32(0.0)
        var ry = Float32(0.0)
        for ni in range(len(self.nodes)):
            var no = self.nodes[ni].copy()
            for pi in range(len(no.outputs)):
                var hx = no.out_pin_x(pi)
                var hy = no.out_pin_y(pi)
                var dx = mx - hx
                var dy = my - hy
                var d2 = dx * dx + dy * dy
                if d2 <= best:
                    best = d2
                    rid = no.outputs[pi].id
                    rx = hx
                    ry = hy
        return PinHit(rid, rx, ry)

    # find INPUT pin under cursor -> pin id, -1 none. SQUARED dist.
    def hit_input_pin(self, mx: Float32, my: Float32) -> Int:
        var best = Float32(Self.PIN_HIT_R2)
        var rid = -1
        for ni in range(len(self.nodes)):
            var no = self.nodes[ni].copy()
            for pi in range(len(no.inputs)):
                var hx = no.in_pin_x(pi)
                var hy = no.in_pin_y(pi)
                var dx = mx - hx
                var dy = my - hy
                var d2 = dx * dx + dy * dy
                if d2 <= best:
                    best = d2
                    rid = no.inputs[pi].id
        return rid

    # find node-by-title under cursor (topmost = last drawn = highest index first)
    def hit_node_title(self, mx: Float32, my: Float32) -> Int:
        var i = len(self.nodes) - 1
        while i >= 0:
            if self.nodes[i].hit_title(mx, my):
                return i
            i -= 1
        return -1

    # does a link already exist between these two pins?
    def _link_exists(self, fid: Int, tid: Int) -> Bool:
        for li in range(len(self.links)):
            if self.links[li].from_pin_id == fid and self.links[li].to_pin_id == tid:
                return True
        return False

    # ==========================================================================
    # update(mx,my,down): SINGLE interaction step from injected mouse state.
    # NEVER reads get_mouse_* directly (headless verification requirement).
    #   - press on output pin -> begin link drag        (UI.cpp provider start)
    #   - press on title bar  -> begin node move         (Node.h:138 OnDragStart)
    #   - hold + dragging node-> move node               (Node.h:139 OnDragging)
    #   - release on input pin (while link-dragging) -> CREATE link  (Pin::LinkTo)
    #   - release             -> end any drag            (Node.h:140 OnDragEnd)
    # ==========================================================================
    def update(mut self, mx: Int, my: Int, down: Bool):
        var fmx = Float32(mx)
        var fmy = Float32(my)
        var pressed = down and (not self.prev_down)
        var released = (not down) and self.prev_down

        if pressed:
            # output pin has priority -> start a link
            var oh = self.hit_output_pin(fmx, fmy)
            if oh.id >= 0:
                self.link_from_pin = oh.id
                self.link_from_x = oh.x
                self.link_from_y = oh.y
            else:
                # else grab a title bar to move the node
                var ni = self.hit_node_title(fmx, fmy)
                if ni >= 0:
                    self.drag_node = ni
                    self.drag_dx = fmx - self.nodes[ni].x
                    self.drag_dy = fmy - self.nodes[ni].y
        elif down and self.drag_node >= 0:
            # move the dragged node, keeping the grab offset
            var n = self.nodes[self.drag_node].copy()
            n.x = fmx - self.drag_dx
            n.y = fmy - self.drag_dy
            self.nodes[self.drag_node] = n^
        elif released:
            # if we were dragging a link, try to drop on an input pin
            if self.link_from_pin >= 0:
                var ipid = self.hit_input_pin(fmx, fmy)
                if ipid >= 0 and not self._link_exists(self.link_from_pin, ipid):
                    self.links.append(Link(self.link_from_pin, ipid))
                self.link_from_pin = -1
            self.drag_node = -1

        self.prev_down = down

    # ==========================================================================
    # draw(self): issue real MojoGUI draw_* for EVERY element so the framebuffer
    # verifier sees links (bezier) + node bodies + title bars + pin dots + labels.
    # Links are drawn FIRST (under nodes), then nodes on top. (UI.cpp render order)
    # ==========================================================================
    def draw(self):
        # ---- links (cubic bezier, output pin -> input pin) ----
        for li in range(len(self.links)):
            var fid = self.links[li].from_pin_id
            var tid = self.links[li].to_pin_id
            var fx = Float32(0.0)
            var fy = Float32(0.0)
            var tx = Float32(0.0)
            var ty = Float32(0.0)
            var got_from = False
            var got_to = False
            # resolve pin ids -> screen connection points
            for ni in range(len(self.nodes)):
                var no = self.nodes[ni].copy()
                for pi in range(len(no.outputs)):
                    if no.outputs[pi].id == fid:
                        fx = no.out_pin_x(pi)
                        fy = no.out_pin_y(pi)
                        got_from = True
                for pi in range(len(no.inputs)):
                    if no.inputs[pi].id == tid:
                        tx = no.in_pin_x(pi)
                        ty = no.in_pin_y(pi)
                        got_to = True
            if got_from and got_to:
                col(180, 180, 70)
                var prevx = fx
                var prevy = fy
                var s = 1
                while s <= Self.BEZ_SAMPLES:
                    var t = Float32(s) / Float32(Self.BEZ_SAMPLES)
                    var sp = bezier_xy(fx, fy, tx, ty, t)
                    line_t(prevx, prevy, sp.x, sp.y, 2.0)
                    prevx = sp.x
                    prevy = sp.y
                    s += 1

        # ---- rubber-band line while creating a link ----
        if self.link_from_pin >= 0:
            col(220, 220, 120)
            line_t(self.link_from_x, self.link_from_y, self.link_from_x, self.link_from_y, 1.0)

        # ---- nodes ----
        for ni in range(len(self.nodes)):
            var no = self.nodes[ni].copy()
            # body
            col(48, 50, 58)
            fill(no.x, no.y, no.w, no.h)
            # border
            col(20, 20, 24)
            rect(no.x, no.y, no.w, no.h)
            # title bar
            col(70, 110, 160)
            fill(no.x, no.y, no.w, Node.TITLE_H)
            col(235, 235, 240)
            txt(no.title, no.x + 6.0, no.y + 5.0, 12.0)

            # input pins (left edge) -> dot + label to the right
            var ip = 0
            while ip < len(no.inputs):
                var hx = no.in_pin_x(ip)
                var hy = no.in_pin_y(ip)
                col(120, 200, 120)
                fcircle(hx, hy, Node.PIN_R, 12)
                col(30, 30, 34)
                rect(hx - Node.PIN_R, hy - Node.PIN_R, Node.PIN_R * 2.0, Node.PIN_R * 2.0)
                col(200, 200, 205)
                txt(no.inputs[ip].name, hx + 10.0, hy - 6.0, 11.0)
                ip += 1

            # output pins (right edge) -> dot + label to the left
            var op = 0
            while op < len(no.outputs):
                var hx = no.out_pin_x(op)
                var hy = no.out_pin_y(op)
                col(200, 160, 90)
                fcircle(hx, hy, Node.PIN_R, 12)
                col(30, 30, 34)
                rect(hx - Node.PIN_R, hy - Node.PIN_R, Node.PIN_R * 2.0, Node.PIN_R * 2.0)
                col(200, 200, 205)
                txt(no.outputs[op].name, hx - 44.0, hy - 6.0, 11.0)
                op += 1


# =============================================================================
# SELF-TEST (do NOT run; the single builder compiles & framebuffer-verifies).
# Builds 2 nodes; drags node A's title bar (+30,+15); asserts A moved by delta.
# Then drags from A's output pin to B's input pin; asserts a Link was created.
# =============================================================================
def main() raises:
    var g = NodeGraph()

    # Node A at (40,40) 120x80; output pin id=101 at right edge.
    var a = Node(1, String("Source"), Float32(40.0), Float32(40.0), Float32(120.0), Float32(80.0))
    a.add_output(101, String("out"), 0)
    g.nodes.append(a^)

    # Node B at (260,120) 120x80; input pin id=201 at left edge.
    var b = Node(2, String("Sink"), Float32(260.0), Float32(120.0), Float32(120.0), Float32(80.0))
    b.add_input(201, String("in"), 1)
    g.nodes.append(b^)

    # ---- DRAG NODE A by (+30,+15) via its title bar ----
    var ax0 = g.nodes[0].x   # 40
    var ay0 = g.nodes[0].y   # 40
    # title bar center: (40 + 60, 40 + 11) = (100, 51)
    var t_mx = Int(ax0) + 60
    var t_my = Int(ay0) + 11
    g.update(t_mx, t_my, True)               # press on title -> begin move
    g.update(t_mx + 30, t_my + 15, True)     # drag +30,+15
    g.update(t_mx + 30, t_my + 15, False)    # release
    var ax1 = g.nodes[0].x   # expect 70
    var ay1 = g.nodes[0].y   # expect 55
    var dx = ax1 - ax0       # expect +30
    var dy = ay1 - ay0       # expect +15
    var ok_move = (dx > Float32(29.5)) and (dx < Float32(30.5)) and (dy > Float32(14.5)) and (dy < Float32(15.5))

    # ---- DRAG A.out (id 101) -> B.in (id 201) to CREATE a link ----
    var links_before = len(g.links)   # expect 0
    # A moved to (70,55): output pin x = 70+120 = 190 ; y = 55 + 22 + 14 = 91
    var apx = g.nodes[0].out_pin_x(0)   # 190
    var apy = g.nodes[0].out_pin_y(0)   # 91
    # B input pin x = 260 ; y = 120 + 22 + 14 = 156
    var bpx = g.nodes[1].in_pin_x(0)    # 260
    var bpy = g.nodes[1].in_pin_y(0)    # 156
    g.update(Int(apx), Int(apy), True)    # press on output pin -> start link
    g.update(Int(bpx), Int(bpy), True)    # drag toward input pin
    g.update(Int(bpx), Int(bpy), False)   # release on input pin -> CREATE link
    var links_after = len(g.links)   # expect 1
    var ok_link = (links_after == links_before + 1)
    var link_from = -1
    var link_to = -1
    if links_after > 0:
        link_from = g.links[0].from_pin_id   # expect 101
        link_to = g.links[0].to_pin_id       # expect 201

    # issue draw calls once (framebuffer-verifiable)
    g.draw()

    print("NODEGRAPH_SELFTEST:",
        "ax0=", ax0, "ay0=", ay0, "ax1=", ax1, "ay1=", ay1, "dx=", dx, "dy=", dy, "move_ok=", ok_move,
        "links_before=", links_before, "links_after=", links_after, "link_ok=", ok_link,
        "link_from=", link_from, "link_to=", link_to)
    # EXPECTED self-test numbers (for the main-loop gate):
    #   ax0==40  ay0==40  ax1==70  ay1==55   dx==30  dy==15  -> move_ok == True
    #   links_before==0   links_after==1                     -> link_ok  == True
    #   link_from==101    link_to==201
