#!/bin/bash
# Build the RC-method report and publish it to OneDrive.
#
#     bash docs/rc_method_pi0/publish.sh
#
# The repo is the source of truth; OneDrive is a publication target.  Never edit
# the OneDrive copy.  Layout on the OneDrive side: the .tex, the .pdf, figs/,
# and a README with the commit it was built from.  No .git is ever copied:
# OneDrive syncing a git directory can corrupt it.
set -euo pipefail
export PATH=/Library/TeX/texbin:$PATH

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${ONEDRIVE_TOPIC:-$HOME/OneDrive/Work/2026_pi0_RC_method}"
NAME=rc_method_pi0_report
[ -d "$DEST" ] || mkdir -p "$DEST"
mkdir -p "$DEST/figs"

# build: twice for the table of contents
( cd "$SRC" && pdflatex -interaction=nonstopmode -halt-on-error "$NAME.tex" >/dev/null \
            && pdflatex -interaction=nonstopmode -halt-on-error "$NAME.tex" >/dev/null ) \
  || { echo "pdflatex failed, see $SRC/$NAME.log" >&2; exit 1; }

cp "$SRC/$NAME.tex" "$SRC/$NAME.pdf" "$DEST/"
n=0
for f in "$SRC"/*.png "$SRC"/*.pdf; do
  case "$(basename "$f")" in "$NAME.pdf") continue ;; esac
  cp "$f" "$DEST/figs/"; n=$((n + 1))
done

{
  echo "# pi0 radiative corrections: rad/Born MC against the EXCLURAD formula"
  echo
  echo "Published: $(date '+%Y-%m-%d %H:%M')"
  echo "Commit:    $(cd "$SRC" && git describe --tags --always --dirty 2>/dev/null || echo '?')"
  echo "Branch:    $(cd "$SRC" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  echo "Source:    github.com/vkubarovsky/exclurad_py, docs/rc_method_pi0/ (do not edit here)"
  echo
  echo "- $NAME.pdf  -- the report, all figures and conclusions so far"
  echo "- figs/      -- every figure and the per-study PDF books"
  echo
  echo "Status: statistics are NOT sufficient for a definite conclusion on the phi"
  echo "dependence of the correction; a larger MC sample is being generated."
} > "$DEST/README.md"

echo "published $NAME.pdf + $n figures -> $DEST"
