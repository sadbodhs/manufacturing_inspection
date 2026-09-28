#!/bin/bash
# Exclusive GPU lock for anything that measures on this card.
#
# The 3090 is shared with other workloads, and this study measured what sharing
# does to a number: ~24% error, larger than most of the effects it reports. So
# benchmarks are serialised, never "balanced": one holder at a time, everyone
# else waits or goes away.
#
# The lock is a directory, because mkdir is atomic - two callers racing for it
# cannot both succeed. Inside it, an `owner` file says who holds it and why, so
# a stuck lock can be diagnosed instead of guessed at. It lives OUTSIDE the repo
# so unrelated projects on the same box can honour it too.
#
# Usage:
#   scripts/gpu_lock.sh acquire <name> [purpose]   # fail immediately if held
#   scripts/gpu_lock.sh wait    <name> [purpose]   # block until free, then take it
#   scripts/gpu_lock.sh release <name>             # only the holder named <name>
#   scripts/gpu_lock.sh release --force            # break someone else's lock
#   scripts/gpu_lock.sh status
#   scripts/gpu_lock.sh run <name> -- <cmd...>     # wait, run, always release
#
# Env: GPU_LOCK (default /home/suchi/sadbodh/.gpu-lock), GPU_LOCK_POLL (s, default 30)
set -euo pipefail
LOCK="${GPU_LOCK:-/home/suchi/sadbodh/.gpu-lock}"
POLL="${GPU_LOCK_POLL:-30}"

die() { echo "gpu_lock: $*" >&2; exit 1; }

holder() { sed -n 's/^name=//p' "$LOCK/owner" 2>/dev/null || true; }

status() {
  if [ -d "$LOCK" ]; then
    echo "HELD  $LOCK"
    sed 's/^/  /' "$LOCK/owner" 2>/dev/null || echo "  (no owner file - lock is mid-acquire or was left broken)"
  else
    echo "FREE  $LOCK"
  fi
  # The lock is a convention; the GPU does not know about it. Show what is
  # actually running so a free lock with a busy GPU is visible.
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
    | sed 's/^/  gpu: /' || true
}

try_acquire() {  # name purpose
  mkdir "$LOCK" 2>/dev/null || return 1
  {
    echo "name=$1"
    echo "purpose=${2:-}"
    echo "user=$(id -un)"
    echo "host=$(hostname)"
    echo "pid=$$"
    echo "since=$(date -Is)"
  } > "$LOCK/owner"
}

release() {  # name | --force
  [ -d "$LOCK" ] || { echo "gpu_lock: already free"; return 0; }
  if [ "$1" != "--force" ]; then
    local h; h="$(holder)"
    [ "$h" = "$1" ] || die "held by '$h', not '$1' - use 'release --force' only if you know it is stale"
  fi
  rm -rf "$LOCK"
  echo "gpu_lock: released"
}

wait_acquire() {  # name purpose
  until try_acquire "$1" "${2:-}"; do
    echo "gpu_lock: held by '$(holder)', waiting ${POLL}s" >&2
    sleep "$POLL"
  done
  echo "gpu_lock: acquired by '$1'"
}

cmd="${1:-status}"; shift || true
case "$cmd" in
  status)  status ;;
  acquire) [ -n "${1:-}" ] || die "acquire needs a name"
           try_acquire "$1" "${2:-}" || { status >&2; die "held"; }
           echo "gpu_lock: acquired by '$1'" ;;
  wait)    [ -n "${1:-}" ] || die "wait needs a name"
           wait_acquire "$1" "${2:-}" ;;
  release) [ -n "${1:-}" ] || die "release needs a name (or --force)"
           release "$1" ;;
  run)     name="${1:-}"; shift || true
           [ -n "$name" ] && [ "${1:-}" = "--" ] || die "usage: run <name> -- <cmd...>"
           shift
           wait_acquire "$name" "$*"
           trap 'release "$name" >/dev/null' EXIT
           "$@" ;;
  *)       die "unknown command '$cmd' (acquire|wait|release|status|run)" ;;
esac
