# Unit tests for WebSocket fragmentation reassembly (RFC 6455 §5.4).
# Builds frames with the server encoders, decodes them back, and feeds them
# through WsReassembler — pure Mojo, no sockets. Run from the repo root:
#   pixi run mojo run -I . http/tests/ws_reassembly_test.mojo

from http.websocket import (
    decode_frame, encode_text, encode_first, encode_cont, encode_pong,
    WsReassembler, OP_TEXT, OP_BINARY, OP_PING, OP_CONT,
)


struct TT(Copyable, Movable):
    var p: Int
    var f: Int
    def __init__(out self):
        self.p = 0
        self.f = 0
    def ck(mut self, cond: Bool, name: String):
        if cond: self.p += 1
        else:
            self.f += 1
            print("  FAIL:", name)


def _push_bytes(mut r: WsReassembler, frame: String) raises -> Int:
    """Decode one frame from `frame` and push it; returns 1 if a message became
    ready (and prints nothing), else 0. Used only where we don't need the payload."""
    var fr = decode_frame(frame)
    var m = r.push(fr)
    return 1 if m.ready else 0


def main() raises:
    var t = TT()

    # 1) a single unfragmented TEXT frame is ready immediately
    var r1 = WsReassembler()
    var m1 = r1.push(decode_frame(encode_text("hello")))
    t.ck(m1.ready and m1.opcode == OP_TEXT and m1.payload == "hello", "single frame ready")
    t.ck(not r1.in_progress(), "no fragment left in progress after a FIN frame")

    # 2) a 3-fragment TEXT message reassembles in order
    var r2 = WsReassembler()
    var a = r2.push(decode_frame(encode_first(OP_TEXT, "Hello ")))
    t.ck(not a.ready and r2.in_progress(), "first fragment buffered, in progress")
    var b = r2.push(decode_frame(encode_cont("beautiful ", False)))
    t.ck(not b.ready and r2.in_progress(), "middle fragment buffered")
    var c = r2.push(decode_frame(encode_cont("World", True)))
    t.ck(c.ready and c.opcode == OP_TEXT and c.payload == "Hello beautiful World",
         "final fragment -> full reassembled message")
    t.ck(not r2.in_progress(), "reassembler reset after delivery")

    # 3) a control frame (PING) interjected mid-message passes through and does
    #    NOT disturb the in-progress data message
    var r3 = WsReassembler()
    _ = r3.push(decode_frame(encode_first(OP_TEXT, "part1-")))
    var pong = r3.push(decode_frame(encode_pong("hb")))  # ping/pong both control
    t.ck(pong.ready and pong.opcode != OP_TEXT, "interjected control frame delivered")
    t.ck(r3.in_progress(), "data message still in progress after control frame")
    var fin3 = r3.push(decode_frame(encode_cont("part2", True)))
    t.ck(fin3.ready and fin3.payload == "part1-part2", "data resumes correctly after control")

    # 4) a BINARY fragmented message keeps its opcode
    var r4 = WsReassembler()
    _ = r4.push(decode_frame(encode_first(OP_BINARY, "AB")))
    var b4 = r4.push(decode_frame(encode_cont("CD", True)))
    t.ck(b4.ready and b4.opcode == OP_BINARY and b4.payload == "ABCD", "binary opcode preserved")

    # 5) a stray continuation (no message in progress) is ignored, not delivered
    var r5 = WsReassembler()
    var stray = r5.push(decode_frame(encode_cont("orphan", True)))
    t.ck(not stray.ready, "stray continuation ignored")

    # 6) many small fragments reassemble to the exact concatenation
    var r6 = WsReassembler()
    _ = r6.push(decode_frame(encode_first(OP_TEXT, "0")))
    var expect = String("0")
    var last_ready = False
    var last_payload = String("")
    for i in range(1, 50):
        var piece = String(i)
        expect += piece
        var fin = i == 49
        var mm = r6.push(decode_frame(encode_cont(piece, fin)))
        if mm.ready:
            last_ready = True
            last_payload = mm.payload
    t.ck(last_ready and last_payload == expect, "50-fragment message reassembled exactly")

    print("---")
    print("passed:", t.p, " failed:", t.f)
    if t.f == 0:
        print("ALL WS REASSEMBLY TESTS PASSED")
