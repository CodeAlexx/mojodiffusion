# pdf/rc4.mojo
# Pure-Mojo RC4 stream cipher (KSA + PRGA).
# RC4 is symmetric: the same call encrypts and decrypts.
# Used by the PDF standard security handler (RC4 40/128-bit).


def rc4(key: List[UInt8], data: List[UInt8]) raises -> List[UInt8]:
    var klen = len(key)
    if klen == 0:
        raise Error("rc4: empty key")

    # --- KSA: initialize and key-schedule the S permutation ---
    var S = List[Int]()
    for i in range(256):
        S.append(i)

    var j = 0
    for i in range(256):
        j = (j + S[i] + Int(key[i % klen])) & 0xFF
        var tmp = S[i]
        S[i] = S[j]
        S[j] = tmp

    # --- PRGA: generate keystream and XOR with data ---
    var out = List[UInt8]()
    var ii = 0
    var jj = 0
    for n in range(len(data)):
        ii = (ii + 1) & 0xFF
        jj = (jj + S[ii]) & 0xFF
        var tmp = S[ii]
        S[ii] = S[jj]
        S[jj] = tmp
        var k = S[(S[ii] + S[jj]) & 0xFF]
        out.append(data[n] ^ UInt8(k))
    return out^
