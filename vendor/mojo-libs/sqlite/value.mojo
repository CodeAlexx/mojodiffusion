# sqlite/value.mojo — a SQLite dynamic value (the 5 storage classes).

comptime VT_NULL: Int = 0
comptime VT_INT: Int = 1
comptime VT_REAL: Int = 2
comptime VT_TEXT: Int = 3
comptime VT_BLOB: Int = 4


struct Value(Copyable, Movable):
    var kind: Int
    var i: Int64
    var r: Float64
    var s: String
    var b: List[UInt8]

    def __init__(out self):
        self.kind = VT_NULL
        self.i = 0
        self.r = 0.0
        self.s = String("")
        self.b = List[UInt8]()

    def __init__(out self, *, copy: Self):
        self.kind = copy.kind
        self.i = copy.i
        self.r = copy.r
        self.s = copy.s
        self.b = copy.b.copy()

    @staticmethod
    def null() -> Value:
        return Value()

    @staticmethod
    def integer(v: Int64) -> Value:
        var x = Value()
        x.kind = VT_INT
        x.i = v
        return x^

    @staticmethod
    def real(v: Float64) -> Value:
        var x = Value()
        x.kind = VT_REAL
        x.r = v
        return x^

    @staticmethod
    def text(v: String) -> Value:
        var x = Value()
        x.kind = VT_TEXT
        x.s = v
        return x^

    @staticmethod
    def blob(var v: List[UInt8]) -> Value:
        var x = Value()
        x.kind = VT_BLOB
        x.b = v^
        return x^

    def is_null(self) -> Bool:
        return self.kind == VT_NULL

    def as_int(self) -> Int64:
        if self.kind == VT_INT:
            return self.i
        if self.kind == VT_REAL:
            return Int64(self.r)
        return 0

    def as_real(self) -> Float64:
        if self.kind == VT_REAL:
            return self.r
        if self.kind == VT_INT:
            return Float64(self.i)
        return 0.0

    def as_text(self) -> String:
        return self.s

    def as_blob(self) raises -> List[UInt8]:
        return self.b.copy()

    def type_name(self) -> String:
        if self.kind == VT_NULL:
            return String("NULL")
        if self.kind == VT_INT:
            return String("INT")
        if self.kind == VT_REAL:
            return String("REAL")
        if self.kind == VT_TEXT:
            return String("TEXT")
        return String("BLOB")
