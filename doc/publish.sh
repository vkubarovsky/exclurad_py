#!/bin/bash
# Публикация картинок и документов в OneDrive.
#
#     bash doc/publish.sh            только картинки
#     bash doc/publish.sh --docs     ещё и PDF мануала с нотой
#
# Скриптом, а не руками: копирование в облако должно быть повторяемым и
# должно оставлять после себя запись о том, что и откуда положено.  Имена
# на той стороне русские -- так уже сложилось в этом каталоге.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${ONEDRIVE_TOPIC:-$HOME/OneDrive/Work/2026_exclurad_py_v1}"
[ -d "$DST" ] || { echo "нет каталога темы: $DST" >&2; exit 1; }

# исходник -> имя в OneDrive
FIGS=(
  "bw_look_rho.png|BW_сравнение_rho.png"
  "bw_look_phi.png|BW_сравнение_phi.png"
  "bw_closure_rho.png|BW_замыкание_rho.png"
  "bw_closure_phi.png|BW_замыкание_phi.png"
)
DOCS=(
  "manual/exclurad_py_manual.pdf|exclurad_py_мануал.pdf"
  "note/exclurad_py_note.pdf|exclurad_py_нота.pdf"
)

items=("${FIGS[@]}")
[ "${1:-}" = "--docs" ] && items+=("${DOCS[@]}")

n=0
for it in "${items[@]}"; do
  s="$SRC/${it%%|*}"; d="$DST/${it##*|}"
  if [ ! -f "$s" ]; then echo "  ПРОПУЩЕНО (нет файла): ${it%%|*}"; continue; fi
  cp "$s" "$d"
  printf "  %-28s -> %s (%sК)\n" "${it%%|*}" "${it##*|}" "$(( $(stat -f%z "$d") / 1024 ))"
  n=$((n + 1))
done

# Запись о происхождении: через полгода никто не вспомнит, из какого
# коммита взялась картинка, а спорить придётся именно об этом.
{
  echo "# exclurad_py -- картинки"
  echo
  echo "Положено: $(date '+%Y-%m-%d %H:%M')"
  echo "Коммит:   $(cd "$SRC/.." && git describe --tags --always --dirty 2>/dev/null || echo '?')"
  echo "Ветка:    $(cd "$SRC/.." && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  echo
  echo "## Брейт-Вигнер"
  echo
  echo '- `BW_сравнение_*.png` -- масса зафиксирована на полюсе против'
  echo '  разыгранной по Брейту-Вигнеру.  Одно зерно, одна карточка,'
  echo '  различие только в массе.  Смотреть надо не на пик, а на то, что'
  echo '  масса тянет за собой: у rho передача, импульс и угол пи+, угол'
  echo '  разлёта пары; у phi всё внутри полупроцента, как и должно быть'
  echo '  при ширине 4 МэВ.'
  echo '- `BW_замыкание_*.png` -- спектр масс в событиях против'
  echo '  предсказания BW/N(W^2) x интеграл sigma_Born, посчитанного'
  echo '  квадратурой по коробке без мажоранты и без accept-reject.'
  echo '  rho: chi2/N = 0.69 по 24 бинам, phi: 0.52 по 18.'
  echo
  echo 'Выборки, по которым это построено:'
  echo '`/Volumes/wd_14tb/exclurad_py_data/bw_check_v1/` (по 3000 событий,'
  echo 'Борн, 10.6 ГэВ).'
  echo
  echo "Воспроизведение: production/bw_look.py и production/bw_closure.py"
  echo "в репозитории exclurad_py."
} > "$DST/README_картинки.md"

echo "положено файлов: $n  ->  $DST"
