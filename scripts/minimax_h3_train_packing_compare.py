#!/usr/bin/env python3
# minimax_h3_train_packing_compare.py — compare the Musubi training-layout dump
# (scripts/minimax_h3_train_packing_oracle.py) against the Mojo twin dump
# (parity/minimax_h3_train_packing_dump.mojo | grep -E '^(case |S |seg |pos$|rowts$|tags$|adaln$|endcase$|tv |ta |[0-9]+$)').
# PASS bars: model times + rowts + tags + adaln BIT-exact; segment boundaries
# equal; positions within 1e-9 (f64 summation-order lineage noise between the
# musubi grid and the diffusers-gated grid; measured 5.7e-14).
import struct
import sys

def parse(path):
    cases = {}; cur = None; sec = None; head = {}
    for ln in open(path):
        t = ln.split()
        if not t: continue
        if t[0] == 'case': cur = t[1]; cases[cur] = {'segs': [], 'pos': [], 'rowts': [], 'tags': [], 'adaln': []}; sec = None
        elif t[0] == 'endcase': cur = None; sec = None
        elif cur is None:
            if t[0] in ('tv', 'ta'): head[t[0]] = t[1]
        elif t[0] == 'S': cases[cur]['S'] = int(t[1])
        elif t[0] == 'seg': cases[cur]['segs'].append((t[1], int(t[2]), int(t[3])))
        elif t[0] in ('pos', 'rowts', 'tags', 'adaln'): sec = t[0]
        elif sec: cases[cur][sec].append(int(t[0]))
    return head, cases

def main():
    h1, a = parse(sys.argv[1]); h2, b = parse(sys.argv[2])
    assert len(a) >= 2 and len(a) == len(b), f"case counts {len(a)} vs {len(b)}"
    f64 = lambda i: struct.unpack('<d', struct.pack('<Q', i))[0]
    f32 = lambda i: struct.unpack('<f', struct.pack('<I', i))[0]
    ok = h1 == h2
    print('model times equal:', h1 == h2)
    for name in a:
        A, B = a[name], b[name]
        print(f'== {name}: S {A["S"]} vs {B["S"]}')
        ok = ok and A['S'] == B['S']
        ok = ok and [(s, e) for _, s, e in A['segs']] == [(s, e) for _, s, e in B['segs']]
        for sec, dec, tol in (('pos', f64, 1e-9), ('rowts', f32, 0.0)):
            assert A[sec] and B[sec], f"empty {sec}"
            if A[sec] == B[sec]:
                print(f'  {sec}: bit-exact ({len(A[sec])})'); continue
            mx = max(abs(dec(x) - dec(y)) for x, y in zip(A[sec], B[sec]) if x != y)
            print(f'  {sec}: max abs float delta {mx:.3e} (tol {tol})')
            ok = ok and mx <= tol
        for sec in ('tags', 'adaln'):
            eq = A[sec] == B[sec]
            print(f'  {sec}: {"bit-exact" if eq else "DIFF"}')
            ok = ok and eq
    print('TRAIN PACKING PARITY', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
