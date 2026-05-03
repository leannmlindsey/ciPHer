#!/usr/bin/env bash
#
# DEPRECATED 2026-05-03 -- this script invoked the legacy
# `cipher.evaluation.runner` which produces the OLD per-pair /
# variable-denominator HR@k metric. Per agent 1+2's 2026-04-27
# broadcasts, the headline metric is now strict-denominator any-hit
# HR@k, produced by `scripts/analysis/per_head_strict_eval.py`.
#
# Use scripts/strict_eval_light_attention_binary.sh instead. Same
# CLI surface (EXP_NAME=... DRY_RUN=... etc).
#
# Why this is a stub instead of an immediate exit: the user may have
# muscle memory for the old script name; the redirect-with-warning
# pattern is friendlier than a "command not found" failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cat >&2 <<'WARN'
WARNING: evaluate_light_attention_binary.sh is DEPRECATED.
  The legacy `cipher.evaluation.runner` it called produced numbers
  that aren't comparable to the post-2026-04-27 headline metric.
  Redirecting to scripts/strict_eval_light_attention_binary.sh ...
WARN

exec "${SCRIPT_DIR}/strict_eval_light_attention_binary.sh" "$@"
