#!/usr/bin/env bash
# Compatibility entry point for the original experiment name.
set -e
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_lgr.sh" "$@"
