"""Small layout helpers shared by native trainer UI tabs."""


def row1(a: Int32) -> List[Int32]:
    var r = List[Int32]()
    r.append(a)
    return r^


def row2(a: Int32, b: Int32) -> List[Int32]:
    var r = List[Int32]()
    r.append(a)
    r.append(b)
    return r^


def row3(a: Int32, b: Int32, c: Int32) -> List[Int32]:
    var r = List[Int32]()
    r.append(a)
    r.append(b)
    r.append(c)
    return r^


def row4(a: Int32, b: Int32, c: Int32, d: Int32) -> List[Int32]:
    var r = List[Int32]()
    r.append(a)
    r.append(b)
    r.append(c)
    r.append(d)
    return r^


def row5(a: Int32, b: Int32, c: Int32, d: Int32, e: Int32) -> List[Int32]:
    var r = List[Int32]()
    r.append(a)
    r.append(b)
    r.append(c)
    r.append(d)
    r.append(e)
    return r^


def row6(a: Int32, b: Int32, c: Int32, d: Int32, e: Int32, f: Int32) -> List[Int32]:
    var r = List[Int32]()
    r.append(a)
    r.append(b)
    r.append(c)
    r.append(d)
    r.append(e)
    r.append(f)
    return r^


def two_col_w(content_w: Int32) -> Int32:
    var cw = (content_w - 16) // 2
    if cw < 320:
        return 320
    return cw


def value_w(content_w: Int32, label_w: Int32) -> Int32:
    var w = content_w - label_w - 16
    if w < 180:
        return 180
    return w
