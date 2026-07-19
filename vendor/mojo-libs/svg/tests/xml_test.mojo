from svg.xml import parse_xml, XmlNode, find_all

def main() raises:
    var doc = String("<?xml version='1.0'?>\n<!-- icon -->\n<svg viewBox=\"0 0 24 24\" width='24'>\n  <g fill=\"red\"><path d=\"M0 0L1 1\"/><rect x='2' y='3' width='4' height='5'/></g>\n  <circle cx='1' cy='2' r='3'/>\n</svg>")
    var root = parse_xml(doc)
    var p = 0
    var f = 0
    if root.name == "svg": p += 1
    else: print("FAIL root name:", root.name); f += 1
    if root.attr(String("viewBox"), String("")) == "0 0 24 24": p += 1
    else: print("FAIL viewBox:", root.attr(String("viewBox"), String(""))); f += 1
    if root.attr(String("width"), String("")) == "24": p += 1
    else: print("FAIL width"); f += 1
    if len(root.children) == 2: p += 1
    else: print("FAIL child count:", len(root.children)); f += 1
    # find_all paths
    var paths = List[XmlNode]()
    find_all(root, String("path"), paths)
    if len(paths) == 1 and paths[0].attr(String("d"), String("")) == "M0 0L1 1": p += 1
    else: print("FAIL paths"); f += 1
    var rects = List[XmlNode]()
    find_all(root, String("rect"), rects)
    if len(rects) == 1 and rects[0].attr(String("width"), String("")) == "4": p += 1
    else: print("FAIL rect"); f += 1
    var g = List[XmlNode]()
    find_all(root, String("g"), g)
    if len(g) == 1 and g[0].attr(String("fill"), String("")) == "red": p += 1
    else: print("FAIL g fill"); f += 1
    print("xml:", p, "passed,", f, "failed")
    if f != 0: raise Error("xml FAILED")
