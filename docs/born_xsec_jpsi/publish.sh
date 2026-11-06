#!/bin/bash
# Build the Born cross-section extraction note and publish it to OneDrive.
#
#     bash docs/born_xsec_jpsi/publish.sh
#
# Override the destination with ONEDRIVE_TOPIC=<dir> if the topic moves.
#
# The repo is the source of truth; OneDrive is a publication target.  Never edit
# the OneDrive copy.  No .git is ever copied: OneDrive syncing a git directory
# can corrupt it.
set -euo pipefail
export PATH=/Library/TeX/texbin:$PATH

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS="$(cd "$SRC/.." && pwd)"
DEST="${ONEDRIVE_TOPIC:-$HOME/OneDrive/Work/2026_Jpsi_Born_xsec_extraction}"
NAME=born_xsec_jpsi
MD=BORN_XSEC_EXTRACTION_JPSI.md
[ -d "$DEST" ] || mkdir -p "$DEST"

# Byte-reproducible build.  Without this pdflatex stamps the current time into
# the PDF, so every publish run leaves the committed born_xsec_jpsi.pdf modified
# and the README below records a "-dirty" commit that is not really dirty.
# pdfTeX honours SOURCE_DATE_EPOCH when FORCE_SOURCE_DATE=1; anchoring it to the
# .tex's own commit date makes the output depend on the source alone.
export FORCE_SOURCE_DATE=1
SOURCE_DATE_EPOCH="$(cd "$SRC" && git log -1 --format=%ct -- "$NAME.tex" 2>/dev/null || true)"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-0}"

# build: twice, so the equation cross-references resolve
( cd "$SRC" && pdflatex -interaction=nonstopmode -halt-on-error "$NAME.tex" >/dev/null \
            && pdflatex -interaction=nonstopmode -halt-on-error "$NAME.tex" >/dev/null ) \
  || { echo "pdflatex failed, see $SRC/$NAME.log" >&2; exit 1; }
rm -f "$SRC"/$NAME.{aux,log,out,toc}

cp "$SRC/$NAME.tex" "$SRC/$NAME.pdf" "$DEST/"
cp "$DOCS/$MD" "$DEST/"

{
  echo "# Extracting the Born cross section for ep -> e'p'J/psi"
  echo
  echo "Published: $(date '+%Y-%m-%d %H:%M')"
  echo "Commit:    $(cd "$SRC" && git describe --tags --always --dirty 2>/dev/null || echo '?')"
  echo "Branch:    $(cd "$SRC" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  echo "Source:    github.com/vkubarovsky/exclurad_py, docs/born_xsec_jpsi/ (do not edit here)"
  echo
  echo "- $NAME.pdf  -- the note, 5 pages: master formula, 9 steps, traps, systematics"
  echo "- $NAME.tex  -- its source"
  echo "- $MD        -- the same content in Markdown"
  echo
  echo "Procedure for turning a yield binned in (W,Q2,t) plus a rad MC and a Born MC"
  echo "into a Born cross section.  Form A (data/rad-MC ratio times the model) is the"
  echo "one to compute: a wrong model normalisation cancels exactly in it."
  echo
  echo "Verified against campaign prod_jpsi_f18in_20261009 on 2026-11-06:"
  echo "  * BR(J/psi->e+e-) is NOT applied in the generator -- factor 16.7 if missed"
  echo "  * the rad/born cards differ in THREE fields: iborn, idecay, itarg"
  echo "  * sigma_nb counts pre-preset events, so C_rad cannot come from it"
  echo "  * the Born outbending arm is 28 % short of the rad arm"
  echo
  echo "Status: procedure complete; not yet applied to data."
} > "$DEST/README.md"

echo "published $NAME.pdf, $NAME.tex, $MD -> $DEST"
