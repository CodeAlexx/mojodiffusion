#!/usr/bin/env bash
# pdf/tests/sign_verify.sh
# Decisive oracle for the pure-Mojo PDF signer.
#   1. Generates an RSA key + self-signed cert (openssl).
#   2. Extracts n/e/d hex (python cryptography) + cert DER.
#   3. Runs the Mojo signer (sign_test.mojo) -> /tmp/pdf_signed.pdf.
#   4. Validates with `pdfsig` (poppler) AND `openssl cms -verify`.
#
# Run from /home/alex/MOJO-libs:
#   bash pdf/tests/sign_verify.sh
set -e

echo "=== [1] generate key + self-signed cert ==="
openssl req -x509 -newkey rsa:2048 -keyout /tmp/sk.pem -out /tmp/sc.pem \
    -days 3650 -nodes -subj "/CN=Mojo Test Signer" >/dev/null 2>&1
openssl x509 -in /tmp/sc.pem -outform DER -out /tmp/sc.der
echo "cert DER: $(wc -c < /tmp/sc.der) bytes"

echo "=== [2] extract n/e/d hex ==="
python3 - <<'PY'
from cryptography.hazmat.primitives import serialization
key = serialization.load_pem_private_key(open('/tmp/sk.pem','rb').read(), password=None)
nums = key.private_numbers(); pub = nums.public_numbers
open('/tmp/n.hex','w').write(format(pub.n,'x'))
open('/tmp/e.hex','w').write(format(pub.e,'x'))
open('/tmp/d.hex','w').write(format(nums.d,'x'))
print('n hex len', len(format(pub.n,'x')), 'e', format(pub.e,'x'))
PY

echo "=== [3] ensure an input PDF ==="
if [ ! -f /tmp/pdf_showcase.pdf ]; then
    echo "no /tmp/pdf_showcase.pdf — building one via demo"
    pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . pdf/tests/demo.mojo || true
fi
ls -l /tmp/pdf_showcase.pdf

echo "=== [4] run the Mojo signer ==="
pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . pdf/tests/sign_test.mojo

echo "=== [5] pdfsig (poppler) validation ==="
if command -v pdfsig >/dev/null 2>&1; then
    pdfsig /tmp/pdf_signed.pdf || true
else
    echo "pdfsig not found — skipping"
fi

echo "=== [6] extract PKCS#7 + ByteRange content, openssl cms -verify ==="
python3 - <<'PY'
import re
data = open('/tmp/pdf_signed.pdf','rb').read()
# Find the LAST /Contents <...> hex string and the ByteRange.
m = list(re.finditer(rb'/ByteRange\s*\[\s*([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s*\]', data))
br = m[-1]
a,b,c,d = (int(x) for x in br.groups())
print('ByteRange =', [a,b,c,d])
# segment1 = [a, a+b), segment2 = [c, c+d)
content = data[a:a+b] + data[c:c+d]
open('/tmp/sig_content.bin','wb').write(content)
# /Contents hex value lies between the '<' and '>' in the gap (b .. c).
gap = data[b:c]          # includes '<' ... '>'
lt = gap.index(b'<'); gt = gap.rindex(b'>')
hexv = gap[lt+1:gt].strip()
# strip trailing zero padding (hex). DER is self-delimiting; openssl reads it.
raw = bytes.fromhex(hexv.decode())
open('/tmp/sig.p7s','wb').write(raw)
print('content bytes =', len(content), ' p7 DER bytes =', len(raw))
PY

echo "--- openssl cms -verify (self-signed; skip chain trust; -binary for raw PDF bytes) ---"
openssl cms -verify -in /tmp/sig.p7s -inform DER \
    -content /tmp/sig_content.bin \
    -noverify -binary -out /dev/null && \
    echo "CMS-RESULT: success" || echo "CMS-RESULT: FAILED"
