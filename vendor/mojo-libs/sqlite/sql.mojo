# sqlite/sql.mojo — a tokenizer + recursive-descent parser for read-only
# SELECT statements, producing an AST consumed by query.mojo.
#
# Supported grammar (read-only, single table):
#   SELECT <select-list> FROM <table> [WHERE <expr>] [ORDER BY <col> [ASC|DESC]] [LIMIT <n>]
#   select-list = '*' | col (',' col)*
#   WHERE expr  = or-expr
#   or-expr     = and-expr (OR and-expr)*
#   and-expr    = primary (AND primary)*
#   primary     = '(' or-expr ')' | comparison
#   comparison  = col OP operand        (operand = literal | col)
#   OP          = = == != <> < <= > >=
#
# Tokenizer rules:
#   - keywords are case-insensitive (matched on upper-cased identifier text)
#   - identifiers: [A-Za-z_][A-Za-z0-9_]*  (also "quoted" / `quoted` / [quoted])
#   - integer / real numeric literals (real => has '.' or exponent)
#   - string literals in single quotes with '' as an escaped quote
#   - operators: = == != <> < <= > >=
#   - punctuation: , ( ) * . ;
#
# Everything is byte-level ASCII; SQL text here is ASCII.

# ─── token kinds ─────────────────────────────────────────────────────────────
comptime TK_EOF: Int = 0
comptime TK_IDENT: Int = 1     # identifier (already unquoted)
comptime TK_KEYWORD: Int = 2   # reserved word (text is upper-cased)
comptime TK_INT: Int = 3       # integer literal
comptime TK_REAL: Int = 4      # real literal
comptime TK_STRING: Int = 5    # string literal (already unescaped)
comptime TK_OP: Int = 6        # comparison operator (text is canonical)
comptime TK_STAR: Int = 7      # *
comptime TK_COMMA: Int = 8     # ,
comptime TK_LPAREN: Int = 9    # (
comptime TK_RPAREN: Int = 10   # )
comptime TK_DOT: Int = 11      # .
comptime TK_SEMI: Int = 12     # ;

# ─── expression node kinds (WhereExpr.kind) ──────────────────────────────────
comptime EX_AND: Int = 0
comptime EX_OR: Int = 1
comptime EX_CMP: Int = 2

# ─── comparison operand kinds (cmp side) ─────────────────────────────────────
comptime OPND_COL: Int = 0
comptime OPND_INT: Int = 1
comptime OPND_REAL: Int = 2
comptime OPND_TEXT: Int = 3
comptime OPND_NULL: Int = 4

# byte constants
comptime B_SPACE = UInt8(32)
comptime B_TAB = UInt8(9)
comptime B_NL = UInt8(10)
comptime B_CR = UInt8(13)
comptime B_SQUOTE = UInt8(39)   # '
comptime B_DQUOTE = UInt8(34)   # "
comptime B_BTICK = UInt8(96)    # `
comptime B_LBRACK = UInt8(91)   # [
comptime B_RBRACK = UInt8(93)   # ]
comptime B_STAR = UInt8(42)     # *
comptime B_COMMA = UInt8(44)    # ,
comptime B_LPAREN = UInt8(40)   # (
comptime B_RPAREN = UInt8(41)   # )
comptime B_DOT = UInt8(46)      # .
comptime B_SEMI = UInt8(59)     # ;
comptime B_EQ = UInt8(61)       # =
comptime B_BANG = UInt8(33)     # !
comptime B_LT = UInt8(60)       # <
comptime B_GT = UInt8(62)       # >
comptime B_USC = UInt8(95)      # _
comptime B_PLUS = UInt8(43)     # +
comptime B_MINUS = UInt8(45)    # -


def _is_ws(b: UInt8) -> Bool:
    return b == B_SPACE or b == B_TAB or b == B_NL or b == B_CR


def _is_digit(b: UInt8) -> Bool:
    return b >= UInt8(48) and b <= UInt8(57)


def _is_alpha(b: UInt8) -> Bool:
    return (
        (b >= UInt8(65) and b <= UInt8(90))
        or (b >= UInt8(97) and b <= UInt8(122))
        or b == B_USC
    )


def _is_alnum(b: UInt8) -> Bool:
    return _is_alpha(b) or _is_digit(b)


def _up(b: UInt8) -> UInt8:
    if b >= UInt8(97) and b <= UInt8(122):
        return b - UInt8(32)
    return b


def _upper(s: String) raises -> String:
    var sb = s.as_bytes()
    var out = String("")
    for i in range(s.byte_length()):
        out += chr(Int(_up(sb[i])))
    return out^


# ─── Token ───────────────────────────────────────────────────────────────────
struct Token(Movable, Copyable):
    var kind: Int
    var text: String   # canonical text (keywords upper, idents unquoted, ops canonical)
    var ival: Int64    # for TK_INT
    var rval: Float64  # for TK_REAL

    def __init__(out self, kind: Int, text: String):
        self.kind = kind
        self.text = text
        self.ival = 0
        self.rval = 0.0

    def __init__(out self, *, copy: Self):
        self.kind = copy.kind
        self.text = copy.text
        self.ival = copy.ival
        self.rval = copy.rval


def _is_keyword(up: String) -> Bool:
    return (
        up == "SELECT" or up == "FROM" or up == "WHERE" or up == "AND"
        or up == "OR" or up == "ORDER" or up == "BY" or up == "ASC"
        or up == "DESC" or up == "LIMIT" or up == "NULL"
    )


def tokenize(sql: String) raises -> List[Token]:
    var sb = sql.as_bytes()
    var n = sql.byte_length()
    var toks = List[Token]()
    var i = 0
    while i < n:
        var c = sb[i]
        if _is_ws(c):
            i += 1
            continue
        # ── string literal '...' with '' escape ──
        if c == B_SQUOTE:
            var s = String("")
            i += 1
            while i < n:
                if sb[i] == B_SQUOTE:
                    if i + 1 < n and sb[i + 1] == B_SQUOTE:
                        s += chr(Int(B_SQUOTE))
                        i += 2
                        continue
                    i += 1
                    break
                s += chr(Int(sb[i]))
                i += 1
            else:
                raise Error("unterminated string literal")
            toks.append(Token(TK_STRING, s^))
            continue
        # ── quoted identifier "x" / `x` / [x] ──
        if c == B_DQUOTE or c == B_BTICK or c == B_LBRACK:
            var close = B_DQUOTE
            if c == B_BTICK:
                close = B_BTICK
            elif c == B_LBRACK:
                close = B_RBRACK
            var s = String("")
            i += 1
            while i < n and sb[i] != close:
                s += chr(Int(sb[i]))
                i += 1
            if i >= n:
                raise Error("unterminated quoted identifier")
            i += 1  # consume closer
            toks.append(Token(TK_IDENT, s^))
            continue
        # ── number (int or real) ──
        if _is_digit(c) or (c == B_DOT and i + 1 < n and _is_digit(sb[i + 1])):
            var start = i
            var is_real = False
            while i < n and _is_digit(sb[i]):
                i += 1
            if i < n and sb[i] == B_DOT:
                is_real = True
                i += 1
                while i < n and _is_digit(sb[i]):
                    i += 1
            # exponent
            if i < n and (_up(sb[i]) == UInt8(69)):  # 'E'
                is_real = True
                i += 1
                if i < n and (sb[i] == B_PLUS or sb[i] == B_MINUS):
                    i += 1
                while i < n and _is_digit(sb[i]):
                    i += 1
            var txt = String("")
            for k in range(start, i):
                txt += chr(Int(sb[k]))
            if is_real:
                var t = Token(TK_REAL, txt)
                t.rval = _parse_real(txt)
                toks.append(t^)
            else:
                var t = Token(TK_INT, txt)
                t.ival = _parse_int(txt)
                toks.append(t^)
            continue
        # ── identifier / keyword ──
        if _is_alpha(c):
            var start = i
            while i < n and _is_alnum(sb[i]):
                i += 1
            var txt = String("")
            for k in range(start, i):
                txt += chr(Int(sb[k]))
            var up = _upper(txt)
            if _is_keyword(up):
                toks.append(Token(TK_KEYWORD, up^))
            else:
                toks.append(Token(TK_IDENT, txt^))
            continue
        # ── operators ──
        if c == B_EQ:
            i += 1
            if i < n and sb[i] == B_EQ:
                i += 1
            toks.append(Token(TK_OP, String("=")))
            continue
        if c == B_BANG:
            i += 1
            if i < n and sb[i] == B_EQ:
                i += 1
                toks.append(Token(TK_OP, String("!=")))
                continue
            raise Error("unexpected '!' (expected '!=')")
        if c == B_LT:
            i += 1
            if i < n and sb[i] == B_EQ:
                i += 1
                toks.append(Token(TK_OP, String("<=")))
                continue
            if i < n and sb[i] == B_GT:
                i += 1
                toks.append(Token(TK_OP, String("!=")))  # <> normalizes to !=
                continue
            toks.append(Token(TK_OP, String("<")))
            continue
        if c == B_GT:
            i += 1
            if i < n and sb[i] == B_EQ:
                i += 1
                toks.append(Token(TK_OP, String(">=")))
                continue
            toks.append(Token(TK_OP, String(">")))
            continue
        # ── punctuation ──
        if c == B_STAR:
            i += 1
            toks.append(Token(TK_STAR, String("*")))
            continue
        if c == B_COMMA:
            i += 1
            toks.append(Token(TK_COMMA, String(",")))
            continue
        if c == B_LPAREN:
            i += 1
            toks.append(Token(TK_LPAREN, String("(")))
            continue
        if c == B_RPAREN:
            i += 1
            toks.append(Token(TK_RPAREN, String(")")))
            continue
        if c == B_DOT:
            i += 1
            toks.append(Token(TK_DOT, String(".")))
            continue
        if c == B_SEMI:
            i += 1
            toks.append(Token(TK_SEMI, String(";")))
            continue
        raise Error("unexpected character: " + chr(Int(c)))
    toks.append(Token(TK_EOF, String("")))
    return toks^


def _parse_int(s: String) raises -> Int64:
    return Int64(Int(s))


def _parse_real(s: String) raises -> Float64:
    return Float64(s)


# ─── WhereExpr — a flattened tree using child indices into a node arena ──────
# Each node is either a comparison (EX_CMP) carrying col + op + operand, or a
# boolean node (EX_AND / EX_OR) referencing two child node indices.
struct WhereNode(Movable, Copyable):
    var kind: Int          # EX_AND | EX_OR | EX_CMP
    var left: Int          # child index (bool nodes)
    var right: Int         # child index (bool nodes)
    # comparison fields:
    var col: String        # left column name
    var op: String         # canonical operator text
    var rhs_kind: Int      # OPND_COL | OPND_INT | OPND_REAL | OPND_TEXT | OPND_NULL
    var rhs_col: String    # when rhs_kind == OPND_COL
    var rhs_ival: Int64
    var rhs_rval: Float64
    var rhs_text: String

    def __init__(out self):
        self.kind = EX_CMP
        self.left = -1
        self.right = -1
        self.col = String("")
        self.op = String("")
        self.rhs_kind = OPND_NULL
        self.rhs_col = String("")
        self.rhs_ival = 0
        self.rhs_rval = 0.0
        self.rhs_text = String("")

    def __init__(out self, *, copy: Self):
        self.kind = copy.kind
        self.left = copy.left
        self.right = copy.right
        self.col = copy.col
        self.op = copy.op
        self.rhs_kind = copy.rhs_kind
        self.rhs_col = copy.rhs_col
        self.rhs_ival = copy.rhs_ival
        self.rhs_rval = copy.rhs_rval
        self.rhs_text = copy.rhs_text


struct WhereExpr(Movable, Copyable):
    var nodes: List[WhereNode]   # arena; root is the last appended top node
    var root: Int                # index of root node, -1 if no WHERE

    def __init__(out self):
        self.nodes = List[WhereNode]()
        self.root = -1

    def __init__(out self, var nodes: List[WhereNode], root: Int):
        self.nodes = nodes^
        self.root = root

    def __init__(out self, *, copy: Self):
        self.nodes = copy.nodes.copy()
        self.root = copy.root

    def has_where(self) -> Bool:
        return self.root >= 0


# ─── SelectStmt — parsed AST root ────────────────────────────────────────────
struct SelectStmt(Movable, Copyable):
    var columns: List[String]   # selected column names ([] when star)
    var star: Bool
    var table: String
    var where: WhereExpr
    var has_where: Bool
    var order_by: String        # "" when no ORDER BY
    var order_desc: Bool
    var limit: Int              # -1 = no limit

    def __init__(out self):
        self.columns = List[String]()
        self.star = False
        self.table = String("")
        self.where = WhereExpr()
        self.has_where = False
        self.order_by = String("")
        self.order_desc = False
        self.limit = -1

    def __init__(out self, *, copy: Self):
        self.columns = copy.columns.copy()
        self.star = copy.star
        self.table = copy.table
        self.where = WhereExpr(copy=copy.where)
        self.has_where = copy.has_where
        self.order_by = copy.order_by
        self.order_desc = copy.order_desc
        self.limit = copy.limit


# ─── Parser — recursive descent over the token list ─────────────────────────
struct Parser(Movable):
    var toks: List[Token]
    var pos: Int
    var nodes: List[WhereNode]   # arena built while parsing WHERE

    def __init__(out self, var toks: List[Token]):
        self.toks = toks^
        self.pos = 0
        self.nodes = List[WhereNode]()

    def _peek_kind(self) -> Int:
        return self.toks[self.pos].kind

    def _peek_text(self) -> String:
        return self.toks[self.pos].text

    def _advance(mut self) raises -> Token:
        var t = Token(copy=self.toks[self.pos])
        if self.pos < len(self.toks) - 1:
            self.pos += 1
        return t^

    def _is_kw(self, word: String) -> Bool:
        return (
            self.toks[self.pos].kind == TK_KEYWORD
            and self.toks[self.pos].text == word
        )

    def _expect_kw(mut self, word: String) raises:
        if not self._is_kw(word):
            raise Error("expected keyword " + word + " but found '" + self._peek_text() + "'")
        _ = self._advance()

    # Parse an identifier, allowing a dotted form table.col (we keep the col).
    def _parse_colname(mut self) raises -> String:
        if self._peek_kind() != TK_IDENT:
            raise Error("expected column name but found '" + self._peek_text() + "'")
        var name = self._advance().text
        # optional .col → use the part after the dot
        if self._peek_kind() == TK_DOT:
            _ = self._advance()
            if self._peek_kind() != TK_IDENT:
                raise Error("expected column name after '.'")
            name = self._advance().text
        return name^

    def parse(mut self) raises -> SelectStmt:
        var stmt = SelectStmt()
        self._expect_kw("SELECT")
        # ── select-list ──
        if self._peek_kind() == TK_STAR:
            _ = self._advance()
            stmt.star = True
        else:
            stmt.columns.append(self._parse_colname())
            while self._peek_kind() == TK_COMMA:
                _ = self._advance()
                stmt.columns.append(self._parse_colname())
        # ── FROM table ──
        self._expect_kw("FROM")
        if self._peek_kind() != TK_IDENT:
            raise Error("expected table name but found '" + self._peek_text() + "'")
        stmt.table = self._advance().text
        # ── optional WHERE ──
        if self._is_kw("WHERE"):
            _ = self._advance()
            var root = self._parse_or()
            stmt.where = WhereExpr(self.nodes.copy(), root)
            stmt.has_where = True
        # ── optional ORDER BY ──
        if self._is_kw("ORDER"):
            _ = self._advance()
            self._expect_kw("BY")
            stmt.order_by = self._parse_colname()
            if self._is_kw("ASC"):
                _ = self._advance()
                stmt.order_desc = False
            elif self._is_kw("DESC"):
                _ = self._advance()
                stmt.order_desc = True
        # ── optional LIMIT ──
        if self._is_kw("LIMIT"):
            _ = self._advance()
            if self._peek_kind() != TK_INT:
                raise Error("expected integer after LIMIT")
            stmt.limit = Int(self._advance().ival)
        # ── trailing ; and EOF ──
        if self._peek_kind() == TK_SEMI:
            _ = self._advance()
        if self._peek_kind() != TK_EOF:
            raise Error("unexpected trailing tokens after statement: '" + self._peek_text() + "'")
        return stmt^

    # or-expr = and-expr (OR and-expr)*
    def _parse_or(mut self) raises -> Int:
        var left = self._parse_and()
        while self._is_kw("OR"):
            _ = self._advance()
            var right = self._parse_and()
            var node = WhereNode()
            node.kind = EX_OR
            node.left = left
            node.right = right
            self.nodes.append(node^)
            left = len(self.nodes) - 1
        return left

    # and-expr = primary (AND primary)*
    def _parse_and(mut self) raises -> Int:
        var left = self._parse_primary()
        while self._is_kw("AND"):
            _ = self._advance()
            var right = self._parse_primary()
            var node = WhereNode()
            node.kind = EX_AND
            node.left = left
            node.right = right
            self.nodes.append(node^)
            left = len(self.nodes) - 1
        return left

    # primary = '(' or-expr ')' | comparison
    def _parse_primary(mut self) raises -> Int:
        if self._peek_kind() == TK_LPAREN:
            _ = self._advance()
            var inner = self._parse_or()
            if self._peek_kind() != TK_RPAREN:
                raise Error("expected ')' in WHERE expression")
            _ = self._advance()
            return inner
        return self._parse_comparison()

    # comparison = col OP operand
    def _parse_comparison(mut self) raises -> Int:
        var col = self._parse_colname()
        if self._peek_kind() != TK_OP:
            raise Error("expected comparison operator after column '" + col + "'")
        var op = self._advance().text
        var node = WhereNode()
        node.kind = EX_CMP
        node.col = col^
        node.op = op^
        # operand: literal or column
        var k = self._peek_kind()
        if k == TK_INT:
            node.rhs_kind = OPND_INT
            node.rhs_ival = self._advance().ival
        elif k == TK_REAL:
            node.rhs_kind = OPND_REAL
            node.rhs_rval = self._advance().rval
        elif k == TK_STRING:
            node.rhs_kind = OPND_TEXT
            node.rhs_text = self._advance().text
        elif self._is_kw("NULL"):
            _ = self._advance()
            node.rhs_kind = OPND_NULL
        elif k == TK_IDENT:
            node.rhs_kind = OPND_COL
            node.rhs_col = self._parse_colname()
        else:
            raise Error("expected literal or column on right of operator")
        self.nodes.append(node^)
        return len(self.nodes) - 1


def parse_select(sql: String) raises -> SelectStmt:
    var toks = tokenize(sql)
    var parser = Parser(toks^)
    return parser.parse()
