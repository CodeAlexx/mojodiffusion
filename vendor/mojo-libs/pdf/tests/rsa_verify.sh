#!/usr/bin/env bash
# pdf/tests/rsa_verify.sh
# Decisive oracle for the pure-Mojo RSA signer: have OpenSSL verify the
# signature produced by rsa_sign_pkcs1v15_sha256.
#
# Prereqs (the Mojo test embeds n/e/d from this exact key):
#   /tmp/rsa_test.pem   (openssl genrsa -out /tmp/rsa_test.pem 2048)
#   /tmp/msg.txt        ("The quick brown fox jumps over the lazy dog")
#   /tmp/mojo_sig.bin   (written by pdf/tests/rsa_test.mojo)
#
# Run from /home/alex/MOJO-libs:
#   bash pdf/tests/rsa_verify.sh
set -euo pipefail

KEY=/tmp/rsa_test.pem
MSG=/tmp/rsa_msg.txt          # private to this oracle (avoid /tmp/msg.txt collisions)
SIG=/tmp/mojo_sig.bin

[ -f "$KEY" ] || { echo "missing $KEY"; exit 1; }
[ -f "$SIG" ] || { echo "missing $SIG (run the Mojo test first)"; exit 1; }

# Write the EXACT message rsa_test.mojo signs (no trailing newline), so the
# oracle does not depend on a shared /tmp/msg.txt that other tests overwrite.
printf '%s' "The quick brown fox jumps over the lazy dog" > "$MSG"

# Public key from the private key.
openssl rsa -in "$KEY" -pubout -out /tmp/rsa_test_pub.pem 2>/dev/null

echo "=== openssl dgst -sha256 -verify ==="
openssl dgst -sha256 -verify /tmp/rsa_test_pub.pem -signature "$SIG" "$MSG"
