#!/usr/bin/env bash
# UserPromptSubmit hook — fires on EVERY user message, independent of anything the
# model remembers or reads. Injects a standing self-verification directive so the
# model verifies its own work every turn instead of the user having to.
cat <<'DIRECTIVE'
[STANDING DIRECTIVE — self-verification is mandatory, fires every turn]
Before reporting ANY item as done / working / fixed / visible in this reply:
1. VERIFY IT YOURSELF with concrete evidence you gathered THIS turn — run it, read the file back, hit the endpoint, check the disk. Never from assumption or a headless test the user cannot see on their own screen.
2. Report ONLY what you actually verified. Anything not verified in the user's real environment must be labeled "not verified yet."
3. Do this automatically for every task, unprompted. The user must never be the one who discovers your work doesn't do what you claimed.
DIRECTIVE
