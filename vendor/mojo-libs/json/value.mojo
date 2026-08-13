# json.value — the JSON value tree. 100% Mojo, no FFI.
#
# A JSONValue is a tagged union over the seven JSON types. Arrays and objects
# hold child JSONValues (the type is recursive — List[Self] / Dict[String, Self]
# work because the collections box their elements on the heap).

comptime JSON_NULL = 0
comptime JSON_BOOL = 1
comptime JSON_INT = 2
comptime JSON_FLOAT = 3
comptime JSON_STR = 4
comptime JSON_ARR = 5
comptime JSON_OBJ = 6


struct JSONValue(Copyable, Movable):
    var kind: Int
    var _b: Bool
    var _i: Int
    var _f: Float64
    var _s: String
    var _arr: List[JSONValue]
    var _obj: Dict[String, JSONValue]

    # Mojo 1.0 cannot decide `Deinitable` for a directly recursive value type
    # (`_arr: List[JSONValue]` inside JSONValue). Declaring the destructor
    # explicitly breaks the cycle; field destructors still run after it, so
    # this is equivalent to the synthesized one.
    def __del__(deinit self):
        pass

    def __init__(out self):
        """Default value is JSON null."""
        self.kind = JSON_NULL
        self._b = False
        self._i = 0
        self._f = 0.0
        self._s = String("")
        self._arr = List[JSONValue]()
        self._obj = Dict[String, JSONValue]()

    # ── constructors ─────────────────────────────────────────────────────────
    @staticmethod
    def null() -> JSONValue:
        return JSONValue()

    @staticmethod
    def from_bool(v: Bool) -> JSONValue:
        var x = JSONValue()
        x.kind = JSON_BOOL
        x._b = v
        return x^

    @staticmethod
    def from_int(v: Int) -> JSONValue:
        var x = JSONValue()
        x.kind = JSON_INT
        x._i = v
        return x^

    @staticmethod
    def from_float(v: Float64) -> JSONValue:
        var x = JSONValue()
        x.kind = JSON_FLOAT
        x._f = v
        return x^

    @staticmethod
    def from_string(v: String) -> JSONValue:
        var x = JSONValue()
        x.kind = JSON_STR
        x._s = v
        return x^

    @staticmethod
    def new_array() -> JSONValue:
        var x = JSONValue()
        x.kind = JSON_ARR
        return x^

    @staticmethod
    def new_object() -> JSONValue:
        var x = JSONValue()
        x.kind = JSON_OBJ
        return x^

    # ── type tests ───────────────────────────────────────────────────────────
    def is_null(self) -> Bool:
        return self.kind == JSON_NULL

    def is_bool(self) -> Bool:
        return self.kind == JSON_BOOL

    def is_int(self) -> Bool:
        return self.kind == JSON_INT

    def is_float(self) -> Bool:
        return self.kind == JSON_FLOAT

    def is_number(self) -> Bool:
        return self.kind == JSON_INT or self.kind == JSON_FLOAT

    def is_string(self) -> Bool:
        return self.kind == JSON_STR

    def is_array(self) -> Bool:
        return self.kind == JSON_ARR

    def is_object(self) -> Bool:
        return self.kind == JSON_OBJ

    # ── scalar accessors (lenient: return a default on type mismatch) ─────────
    def as_bool(self) -> Bool:
        return self._b if self.kind == JSON_BOOL else False

    def as_int(self) -> Int:
        if self.kind == JSON_INT:
            return self._i
        if self.kind == JSON_FLOAT:
            return Int(self._f)
        return 0

    def as_float(self) -> Float64:
        if self.kind == JSON_FLOAT:
            return self._f
        if self.kind == JSON_INT:
            return Float64(self._i)
        return 0.0

    def as_string(self) -> String:
        return self._s if self.kind == JSON_STR else String("")

    # ── container accessors ───────────────────────────────────────────────────
    def length(self) -> Int:
        if self.kind == JSON_ARR:
            return len(self._arr)
        if self.kind == JSON_OBJ:
            return len(self._obj)
        return 0

    def contains(self, key: String) -> Bool:
        return self.kind == JSON_OBJ and key in self._obj

    def __getitem__(self, key: String) raises -> JSONValue:
        """Object field access; returns null if absent or not an object."""
        if self.kind == JSON_OBJ and key in self._obj:
            return self._obj[key].copy()
        return JSONValue()

    def __getitem__(self, idx: Int) -> JSONValue:
        """Array element access; returns null if out of range or not an array."""
        if self.kind == JSON_ARR and idx >= 0 and idx < len(self._arr):
            return self._arr[idx].copy()
        return JSONValue()

    # ── builders ──────────────────────────────────────────────────────────────
    def append(mut self, var v: JSONValue):
        self._arr.append(v^)

    def set(mut self, key: String, var v: JSONValue):
        self._obj[key] = v^

    def keys(self) -> List[String]:
        var out = List[String]()
        if self.kind == JSON_OBJ:
            for e in self._obj.items():
                out.append(e.key)
        return out^
