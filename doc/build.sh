#!/bin/bash
# Сборка обоих документов.  Дважды pdflatex -- ради оглавления и ссылок.
set -eu
cd "$(dirname "$0")"
for d in manual note; do
  [ -f "$d"/*.tex ] || continue
  ( cd "$d" && for f in *.tex; do
      pdflatex -interaction=nonstopmode -halt-on-error "$f" >/dev/null 2>&1 || {
        echo "ОШИБКА сборки $d/$f -- смотри $d/${f%.tex}.log"; exit 1; }
      pdflatex -interaction=nonstopmode -halt-on-error "$f" >/dev/null 2>&1
      echo "  собрано: $d/${f%.tex}.pdf ($(du -h ${f%.tex}.pdf | cut -f1))"
    done )
done
