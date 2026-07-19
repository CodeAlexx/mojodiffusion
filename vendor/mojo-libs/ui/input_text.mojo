# input_text.mojo — an editable text field, enabled by the Tier-3 char-input queue.
# Logic takes INJECTED codepoints (verifiable headless); pump_live() drains get_char().
from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from builtin.type_aliases import MutExternalOrigin


fn to_cstr_bytes(b: List[UInt8]) raises -> UnsafePointer[Int8, MutExternalOrigin]:
    var buf = alloc[Int8](len(b) + 1)
    for i in range(len(b)):
        buf[i] = Int8(Int(b[i]))
    buf[len(b)] = 0
    return buf


fn to_cstr(s: String) raises -> UnsafePointer[Int8, MutExternalOrigin]:
    var b = s.as_bytes()
    var buf = alloc[Int8](len(b) + 1)
    for i in range(len(b)):
        buf[i] = Int8(b[i])
    buf[len(b)] = 0
    return buf


struct TextInput(Copyable, Movable):
    var bytes: List[UInt8]      # ASCII buffer
    var focused: Bool

    def __init__(out self):
        self.bytes = List[UInt8]()
        self.focused = True

    # INJECTED edit: feed typed codepoints + control keys (verifiable headless).
    def edit(mut self, typed: List[Int], backspace: Bool):
        if not self.focused:
            return
        if backspace and len(self.bytes) > 0:
            _ = self.bytes.pop()
        for i in range(len(typed)):
            var cp = typed[i]
            if cp >= 32 and cp < 127:        # printable ASCII
                self.bytes.append(UInt8(cp))

    # LIVE edit: drain the backend char queue + read the backspace keystate.
    # (Thin live binding over edit(); the logic is gated via edit() directly.)
    def pump_live(mut self):
        var typed = List[Int]()
        while True:
            var c = Int(external_call["get_char", Int32]())
            if c == 0:
                break
            typed.append(c)
        var bsp = Int(external_call["get_key_state", Int32](Int32(259))) != 0  # GLFW_KEY_BACKSPACE
        self.edit(typed, bsp)

    def value(self) -> String:
        var s = String("")
        for i in range(len(self.bytes)):
            s += chr(Int(self.bytes[i]))
        return s

    def draw(self, x: Float32, y: Float32, w: Float32, h: Float32) raises:
        # field frame fill (38,40,48) + border + the text
        _ = external_call["set_color", Int32](Float32(38.0/255.0), Float32(40.0/255.0), Float32(48.0/255.0), Float32(1.0))
        _ = external_call["draw_filled_rectangle", Int32](x, y, w, h)
        _ = external_call["set_color", Int32](Float32(90.0/255.0), Float32(90.0/255.0), Float32(110.0/255.0), Float32(1.0))
        _ = external_call["draw_rectangle", Int32](x, y, w, h)
        _ = external_call["set_color", Int32](Float32(0.9), Float32(0.9), Float32(0.95), Float32(1.0))
        _ = external_call["draw_text", Int32](to_cstr_bytes(self.bytes), x + 6.0, y + h - Float32(8.0), Float32(14.0))


def main() raises:
    var ti = TextInput()
    # inject 'H'(72) 'i'(105)
    var t1 = List[Int](); t1.append(72); t1.append(105)
    ti.edit(t1, False)
    print("after 'Hi'   value =", ti.value(), "(expect Hi) len=", len(ti.bytes))
    # backspace -> 'H'
    var empty = List[Int]()
    ti.edit(empty, True)
    print("after bksp   value =", ti.value(), "(expect H) len=", len(ti.bytes))
    # type ' World!' -> 'H World!'
    var t2 = List[Int]()
    var msg = String(" World!")
    var mb = msg.as_bytes()
    for i in range(len(mb)): t2.append(Int(mb[i]))
    ti.edit(t2, False)
    print("after typing value =", ti.value(), "(expect 'H World!') len=", len(ti.bytes))
    var ok = ti.value() == String("H World!")
    print("INPUTTEXT_SELFTEST:", "PASS" if ok else "FAIL")
