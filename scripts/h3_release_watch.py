"""Watch for the MiniMax-H3 release and fetch it if it is not gated.

Alex's instruction, 2026-08-02: "at that time find it on hf, if gated wait for
me .. if not dl it".

  * poll Hugging Face first, ModelScope as the secondary signal
  * if the repo is GATED (or access is refused in any way) -> DOWNLOAD NOTHING.
    Log loudly and stop. Accepting a licence on someone's behalf is theirs to do.
  * if it is open -> fetch, smallest-first, with a disk floor

NO HARD-CODED RELEASE TIME. MiniMax already moved the date once — 16:00Z became
22:00Z, measured on the ModelScope page at 11:35Z — so waiting for a specific
instant is exactly the wrong design. This polls steadily from launch until it
gives up, and separately reports every change to the announced date. Another
slip then costs nothing.

FETCH ORDER, smallest first so something useful lands even if space runs out:
  1. every config / index / json  (KB)
  2. audio_vae/                   (small; unit 10 is already gated against it)
  3. vae/                         (video VAE)
  4. transformer/                 (~61.7 GiB bf16 — the big one)

`text_encoder/` is NOT fetched: Qwen3-VL-32B-Instruct is already on disk at
~/.serenity/models/text_encoders/qwen3vl-32b-instruct-h3 (53.06 GiB, verified),
and whether H3's copy is the same weights is still unresolved. Pulling a second
~60 GiB conditioner on that guess is not a call to make unattended.

DISK FLOOR: stop before free space drops under 25 GiB — the drive is shared
with the running product.

Run:
    nohup python3 scripts/h3_release_watch.py > output/logs/h3_release_watch.log 2>&1 &
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

# ONLY MiniMax's own weights are fetched. Alex, 2026-08-02: "we are not using
# comfyui weights .. we will use regular weights from minimax". Comfy-Org's are
# `pruned_int8_convrot` anyway — a rotation-based int8 scheme that rewrites the
# tensors, not the original bf16 layout this port's loader targets.
HF_REPO = "MiniMaxAI/MiniMax-H3"

# Watched as a SIGNAL ONLY, never fetched: if Comfy-Org publishes it means the
# release is happening, which is worth knowing while we wait for MiniMax's.
HF_SIGNAL_REPO = "Comfy-Org/MiniMax-H3"
MS_REPO = "MiniMax/MiniMax-H3"
MS_PAGE = f"https://modelscope.cn/models/{MS_REPO}"
LOCAL_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3"
DISK_FLOOR_GIB = 25.0

GIVE_UP_AT = datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc)
POLL_SECONDS = 300


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    print(f"[{stamp}] {message}", flush=True)


def free_gib() -> float:
    stat = os.statvfs("/")
    return stat.f_bavail * stat.f_frsize / 1024**3


def announced_release_date():
    """`PreReleaseData.ReleaseDate` as the ModelScope page currently shows it.

    Reported, never trusted: it is MiniMax's stated intent and it has already
    moved once."""
    try:
        with urllib.request.urlopen(MS_PAGE, timeout=20) as response:
            html = response.read().decode("utf-8", "replace")
        match = re.search(r'ReleaseDate\\?"\s*:\s*\\?"([^"\\]+)', html)
        return match.group(1) if match else None
    except Exception:
        return None


def check_hf(repo: str):
    """(exists, gated) for `repo`. `gated` is HF's own field: False, 'auto' or 'manual'.
    Anything other than False means a human must accept terms.

    Classified on EXCEPTION TYPE and HTTP STATUS, never on substring matching.
    An earlier version of this function looked for "403" anywhere in the error
    text and reported the repo GATED at 15:52:51Z while it did not exist —
    every huggingface_hub error embeds a random hex request ID, so a substring
    test trips by chance. Verified against a nonsense control repo, which
    returns the identical `RepositoryNotFoundError` / 404."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=False)
        return True, getattr(info, "gated", False)
    except Exception as error:
        name = type(error).__name__
        status = getattr(getattr(error, "response", None), "status_code", None)
        if name == "GatedRepoError" or status == 403:
            return True, "gated"
        if name == "RepositoryNotFoundError" or status in (401, 404):
            return False, False
        log(f"unexpected HF error ({name}, status {status}); treating as absent: {str(error)[:120]}")
        return False, False


def check_github_creator():
    """A newly published MiniMax-AI repo that looks like H3.

    This matters more than the weights for correctness: the creator's own code
    is the PARITY AUTHORITY, outranking both the diffusers PR and the ComfyUI
    port, which are third-party readings of a checkpoint. It is also a few MB
    and ungated, so unlike weights there is no reason to wait or to ask before
    taking it.

    Known repos as of 2026-08-02: MSA, minimax-code, MiniMax-M3, cli,
    MiniMax-Provider-Verifier — none H3."""
    try:
        url = "https://api.github.com/orgs/MiniMax-AI/repos?sort=created&direction=desc&per_page=30"
        request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(request, timeout=20) as response:
            repos = json.loads(response.read())
        for repo in repos:
            name = repo.get("name", "")
            if re.search(r"h3|hailuo", name, re.I):
                return name, repo.get("clone_url")
    except Exception:
        pass
    return None, None


def clone_creator(name: str, clone_url: str) -> None:
    destination = f"/home/alex/minimax_h3_ref/creator-{name}"
    if os.path.exists(destination):
        return
    log("=" * 70)
    log(f"CREATOR CODE PUBLISHED: MiniMax-AI/{name}")
    log("This is the PARITY AUTHORITY. Every number in the intake that came from")
    log("the diffusers PR or ComfyUI is now a claim to re-check against this.")
    log("=" * 70)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", clone_url, destination],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        log(f"cloned to {destination}")
    else:
        log(f"clone FAILED: {result.stderr.strip()[-300:]}")


def check_modelscope() -> bool:
    try:
        url = f"https://modelscope.cn/api/v1/models/{MS_REPO}/repo/files?Revision=master&Recursive=True"
        with urllib.request.urlopen(url, timeout=20) as response:
            return response.status == 200
    except Exception:
        return False


def fetch(patterns, label: str) -> bool:
    free = free_gib()
    if free < DISK_FLOOR_GIB:
        log(f"STOP: {free:.1f} GiB free is under the {DISK_FLOOR_GIB} GiB floor; skipping {label}")
        return False
    log(f"fetching {label} ({free:.1f} GiB free)")
    script = (
        "import sys, json\n"
        "from huggingface_hub import snapshot_download\n"
        "snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2],\n"
        "                  allow_patterns=json.loads(sys.argv[3]), max_workers=4)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, HF_REPO, LOCAL_DIR, json.dumps(patterns)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log(f"FAILED {label}: {result.stderr.strip()[-500:]}")
        return False
    log(f"done {label} ({free_gib():.1f} GiB free)")
    return True


def preflight() -> None:
    """Header-only inventory the instant the transformer lands.

    Runs unattended because it reads nothing but each shard's JSON directory —
    a few hundred KB, about a second — and because its answer decides whether
    anything else is worth doing. The whole port is built against
    `transformer_key_plan.txt`, which came from the diffusers converter's
    --dry_run with NO WEIGHTS: a third party's reading of a model nobody
    outside MiniMax had seen. This is the first moment that reading can be
    checked against the real thing, and if it is wrong then gated units 12
    (loader) and 14 (fp8 policy) are both built on sand."""
    target = os.path.join(LOCAL_DIR, "transformer")
    if not os.path.isdir(target):
        log(f"preflight skipped: {target} is not there")
        return
    log("=" * 70)
    log("PREFLIGHT: header-only inventory against transformer_key_plan.txt")
    result = subprocess.run(
        [sys.executable, "/home/alex/mojodiffusion/scripts/h3_preflight.py", target],
        capture_output=True, text=True,
    )
    for line in (result.stdout or "").splitlines():
        log("  " + line)
    if result.returncode != 0:
        for line in (result.stderr or "").splitlines()[-10:]:
            log("  ! " + line)
        log("PREFLIGHT FAILED — do not load anything until this is understood")
    log("=" * 70)


def main() -> None:
    announced = announced_release_date()
    log("H3 release watcher armed")
    log(f"  hf fetch   {HF_REPO}")
    log(f"  hf signal  {HF_SIGNAL_REPO} (never fetched)")
    log(f"  github     MiniMax-AI/* — creator code IS cloned, it is the authority")
    log(f"  modelscope {MS_REPO}")
    log(f"  announced  {announced}")
    log(f"  give up    {GIVE_UP_AT.isoformat()}   poll every {POLL_SECONDS}s")
    log(f"  local dir  {LOCAL_DIR}")
    log(f"  disk now   {free_gib():.1f} GiB free, floor {DISK_FLOOR_GIB} GiB")
    log("polling now; announced-date changes are reported as they happen")

    polls = 0
    signalled = [False]
    while datetime.now(timezone.utc) < GIVE_UP_AT:
        polls += 1

        if polls % 6 == 1:
            current = announced_release_date()
            if current != announced:
                log(f"ANNOUNCED DATE CHANGED: {announced} -> {current}")
                announced = current

        # Creator code first: it is small, ungated, and the parity authority.
        # Cloning it does NOT end the watch — the weights are still wanted.
        creator_name, creator_url = check_github_creator()
        if creator_name:
            clone_creator(creator_name, creator_url)

        signal_exists, _ = check_hf(HF_SIGNAL_REPO)
        if signal_exists and not signalled[0]:
            signalled[0] = True
            log(f"SIGNAL: {HF_SIGNAL_REPO} is up — release is happening.")
            log("Not fetching it: Alex wants MiniMax's own weights, and Comfy's")
            log("are pruned int8 'convrot', not the layout our loader targets.")

        exists, gated = check_hf(HF_REPO)
        if exists:
            log(f"HUGGING FACE HAS IT: {HF_REPO}  gated={gated!r}")
            if gated:
                log("=" * 70)
                log("GATED — NOTHING DOWNLOADED. Alex has to accept the terms himself.")
                log("=" * 70)
                return
            log("open access — fetching, smallest first")
            fetch(["*.json", "*.txt", "*.md"], "configs and indexes")
            fetch(["audio_vae/*"], "audio_vae")
            fetch(["vae/*"], "vae")
            # MEASURED 2026-08-02 21:35Z: this link caps at ~14 MiB/s. Four
            # concurrent connections gave 1.07x over one, and hf_transfer gave
            # 0.97x — it is the link, not the client. So transformer/ is ~75
            # minutes of transfer and no local knob shortens it.
            #
            # transformer_ref/ (ref2va) is NOT fetched: it is another 61.73 GiB
            # and both do not fit — 123.46 GiB against 119.6 GiB free. Getting
            # it means freeing space first, which is Alex's call.
            fetch(["transformer/*"], "transformer")
            log("NOTE text_encoder/ deliberately skipped — see this script's docstring")
            log("NOTE transformer_ref/ skipped — both transformers do not fit (123.5 vs 119.6 GiB)")
            log("fetch sequence complete")
            preflight()
            return

        if check_modelscope():
            log("MODELSCOPE HAS IT but Hugging Face does not yet.")
            log("Instruction was HF; ModelScope needs a different client. Still polling HF.")

        if polls % 12 == 1:
            log(f"still unpublished (poll {polls}, announced {announced}, {free_gib():.1f} GiB free)")
        time.sleep(POLL_SECONDS)

    log("gave up: nothing published within the watch window")


if __name__ == "__main__":
    main()
