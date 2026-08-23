#!/bin/bash
# Развернуть exclurad_py на удалённую машину С ПРОВЕНАНСОМ.
#
#   ./production/deploy.sh <host> <каталог>
#
# rsync --exclude .git оставляет дерево без репозитория, и провенанс пишет
# в каждый LUND "git_commit: unknown".  Так вышло с обеими боевыми
# выборками 22.08: 200 тысяч событий, о которых нельзя доказать, каким
# кодом они сделаны.  Шапка провенанса существует ровно для того, чтобы
# этого не случалось, и она промолчала.
#
# Поэтому версия записывается файлом рядом с пакетом, а provenance.py
# читает его, когда git недоступен.
set -eu
H=$1; DIR=$2
SRC=$(cd "$(dirname "$0")/.." && pwd)
COMMIT=$(git -C "$SRC" rev-parse --short HEAD)
git -C "$SRC" diff --quiet || COMMIT="$COMMIT-dirty"
BRANCH=$(git -C "$SRC" rev-parse --abbrev-ref HEAD)

TMP=$(mktemp -d)
TAG=$(git -C "$SRC" describe --tags --always --dirty 2>/dev/null || echo unknown)
printf 'commit=%s\nbranch=%s\ntag=%s\ndeployed=%s\nfrom=%s\n' \
    "$COMMIT" "$BRANCH" "$TAG" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$(hostname)" \
    > "$TMP/DEPLOYED_VERSION"

rsync -a --delete --exclude '.git' --exclude '__pycache__' \
      --exclude '*.npz' --exclude '*.lund' --exclude '*.png' --exclude '*.log' \
      "$SRC/exclurad_py" "$SRC/cards" "$SRC/production" "$H:$DIR/"
rsync -a "$TMP/DEPLOYED_VERSION" "$H:$DIR/exclurad_py/"
rm -rf "$TMP"

echo "развёрнуто на $H:$DIR"
echo "  версия: $COMMIT ($BRANCH), тег $TAG"
ssh "$H" "cd $DIR && PYTHONPATH=\$PWD python3 -c \"
from exclurad_py.provenance import git_hash, git_branch, git_tag
print('  провенанс на месте:', git_hash(), git_branch(), git_tag())\""
