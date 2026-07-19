# config/value.mojo — shared value + tree types for the config library.
#
# Every parser (ini, toml, ...) produces a ConfigTree; the layered Config wraps
# one and adds env overlay + typed access. ConfigValue is a tagged union with
# coercions (INI yields strings; TOML yields typed values — both coerce on read).

from std.collections import Dict

comptime CV_NULL: Int = 0
comptime CV_STR: Int = 1
comptime CV_INT: Int = 2
comptime CV_FLOAT: Int = 3
comptime CV_BOOL: Int = 4
comptime CV_ARRAY: Int = 5


struct ConfigValue(Copyable, Movable):
    var kind: Int
    var s: String
    var i: Int64
    var f: Float64
    var b: Bool
    var arr: List[ConfigValue]

    def __init__(out self):
        self.kind = CV_NULL
        self.s = String("")
        self.i = 0
        self.f = 0.0
        self.b = False
        self.arr = List[ConfigValue]()

    def __init__(out self, *, copy: Self):
        self.kind = copy.kind
        self.s = copy.s
        self.i = copy.i
        self.f = copy.f
        self.b = copy.b
        self.arr = copy.arr.copy()

    @staticmethod
    def str_(v: String) -> ConfigValue:
        var x = ConfigValue(); x.kind = CV_STR; x.s = v; return x^

    @staticmethod
    def int_(v: Int64) -> ConfigValue:
        var x = ConfigValue(); x.kind = CV_INT; x.i = v; return x^

    @staticmethod
    def float_(v: Float64) -> ConfigValue:
        var x = ConfigValue(); x.kind = CV_FLOAT; x.f = v; return x^

    @staticmethod
    def bool_(v: Bool) -> ConfigValue:
        var x = ConfigValue(); x.kind = CV_BOOL; x.b = v; return x^

    @staticmethod
    def array_(var v: List[ConfigValue]) -> ConfigValue:
        var x = ConfigValue(); x.kind = CV_ARRAY; x.arr = v^; return x^

    # ---- typed coercions (raise on impossible) ----
    def as_str(self) raises -> String:
        if self.kind == CV_STR:
            return self.s
        if self.kind == CV_INT:
            return String(self.i)
        if self.kind == CV_FLOAT:
            return String(self.f)
        if self.kind == CV_BOOL:
            return String("true") if self.b else String("false")
        raise Error("ConfigValue.as_str: not a scalar")

    def as_int(self) raises -> Int64:
        if self.kind == CV_INT:
            return self.i
        if self.kind == CV_FLOAT:
            return Int64(self.f)
        if self.kind == CV_BOOL:
            return Int64(1) if self.b else Int64(0)
        if self.kind == CV_STR:
            return _parse_int(self.s)
        raise Error("ConfigValue.as_int: not coercible")

    def as_float(self) raises -> Float64:
        if self.kind == CV_FLOAT:
            return self.f
        if self.kind == CV_INT:
            return Float64(self.i)
        if self.kind == CV_STR:
            return _parse_float(self.s)
        raise Error("ConfigValue.as_float: not coercible")

    def as_bool(self) raises -> Bool:
        if self.kind == CV_BOOL:
            return self.b
        if self.kind == CV_INT:
            return self.i != 0
        if self.kind == CV_STR:
            var l = _lower(self.s)
            if l == "true" or l == "1" or l == "yes" or l == "on":
                return True
            if l == "false" or l == "0" or l == "no" or l == "off":
                return False
            raise Error("ConfigValue.as_bool: not a bool string")
        raise Error("ConfigValue.as_bool: not coercible")

    def as_array(self) raises -> List[ConfigValue]:
        if self.kind == CV_ARRAY:
            return self.arr.copy()
        raise Error("ConfigValue.as_array: not an array")


def _lower(s: String) -> String:
    var out = String("")
    var b = s.as_bytes()
    for i in range(s.byte_length()):
        var c = Int(b[i])
        if c >= 65 and c <= 90:
            c += 32
        out += chr(c)
    return out^


def _parse_int(s: String) raises -> Int64:
    var b = s.as_bytes()
    var n = s.byte_length()
    var i = 0
    var neg = False
    if n > 0 and (Int(b[0]) == 45 or Int(b[0]) == 43):
        neg = Int(b[0]) == 45
        i = 1
    if i >= n:
        raise Error("not an int: " + s)
    var v: Int64 = 0
    while i < n:
        var c = Int(b[i])
        if c < 48 or c > 57:
            raise Error("not an int: " + s)
        v = v * 10 + Int64(c - 48)
        i += 1
    return -v if neg else v


def _parse_float(s: String) raises -> Float64:
    # delegate to Mojo's Float64 parse via atof-like manual parse (sign, int, frac, exp)
    var b = s.as_bytes()
    var n = s.byte_length()
    var i = 0
    var neg = False
    if n > 0 and (Int(b[0]) == 45 or Int(b[0]) == 43):
        neg = Int(b[0]) == 45
        i = 1
    var mant = 0.0
    var any = False
    while i < n and Int(b[i]) >= 48 and Int(b[i]) <= 57:
        mant = mant * 10.0 + Float64(Int(b[i]) - 48); i += 1; any = True
    if i < n and Int(b[i]) == 46:  # '.'
        i += 1
        var scale = 0.1
        while i < n and Int(b[i]) >= 48 and Int(b[i]) <= 57:
            mant += Float64(Int(b[i]) - 48) * scale; scale *= 0.1; i += 1; any = True
    if not any:
        raise Error("not a float: " + s)
    var exp = 0
    var eneg = False
    if i < n and (Int(b[i]) == 101 or Int(b[i]) == 69):  # e/E
        i += 1
        if i < n and (Int(b[i]) == 45 or Int(b[i]) == 43):
            eneg = Int(b[i]) == 45; i += 1
        while i < n and Int(b[i]) >= 48 and Int(b[i]) <= 57:
            exp = exp * 10 + (Int(b[i]) - 48); i += 1
    var p = 1.0
    for _ in range(exp):
        p *= 10.0
    var val = mant * (1.0 / p) if eneg else mant * p
    return -val if neg else val


struct ConfigTree(Movable):
    """Sections -> (key -> ConfigValue). Section "" is the global/default scope."""
    var data: Dict[String, Dict[String, ConfigValue]]

    def __init__(out self):
        self.data = Dict[String, Dict[String, ConfigValue]]()

    def __init__(out self, *, copy: Self):
        self.data = Dict[String, Dict[String, ConfigValue]]()
        for ref sec in copy.data.items():
            var inner = Dict[String, ConfigValue]()
            for ref kv in sec.value.items():
                inner[kv.key] = kv.value.copy()
            self.data[sec.key] = inner^

    def set(mut self, section: String, key: String, var val: ConfigValue) raises:
        if section not in self.data:
            self.data[section] = Dict[String, ConfigValue]()
        self.data[section][key] = val^

    def has(self, section: String, key: String) raises -> Bool:
        if section not in self.data:
            return False
        return key in self.data[section]

    def get(self, section: String, key: String) raises -> ConfigValue:
        if not self.has(section, key):
            raise Error("config: missing [" + section + "] " + key)
        return self.data[section][key].copy()

    def sections(self) raises -> List[String]:
        var out = List[String]()
        for ref e in self.data.items():
            out.append(e.key)
        return out^

    def keys(self, section: String) raises -> List[String]:
        var out = List[String]()
        if section in self.data:
            for ref e in self.data[section].items():
                out.append(e.key)
        return out^

    def merge_over(mut self, other: ConfigTree) raises:
        """Overlay `other` on top of self (other's values win)."""
        for ref sec in other.data.items():
            for ref kv in sec.value.items():
                self.set(sec.key, kv.key, kv.value.copy())
