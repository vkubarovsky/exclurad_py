# Where a LUND run writes.  Source this; do not hard-code an output root.
#
#   LUND OUTPUT NEVER GOES TO $HOME OR TO THE INTERNAL SSD.
#
# On the mac mini it goes to the external 14 TB scratch, on ifarm to
# /volatile.  $HOME is the internal SSD on the mini and a quota'd filesystem
# on ifarm; neither is for generator output, and a production run is
# gigabytes.  The run cards say `./exclurad_out` RELATIVE on purpose and the
# production wrappers substitute it -- so anything that runs a card directly
# has to set the root itself, which is exactly where it gets forgotten.
#
#   . production/speed/lund_outroot.sh
#   echo "$LUND_OUTROOT"
if [ -d /Volumes/wd_14tb ]; then                       # mac mini
    LUND_OUTROOT=/Volumes/wd_14tb/exclurad_speed
elif [ -d /volatile/clas12/vpk ]; then                 # ifarm
    LUND_OUTROOT=/volatile/clas12/vpk/speed_work
else
    echo "lund_outroot.sh: no scratch filesystem found (wd_14tb / volatile)." >&2
    echo "Refusing to default to \$HOME.  Set LUND_OUTROOT yourself." >&2
    return 1 2>/dev/null || exit 1
fi
case "$LUND_OUTROOT" in
    "$HOME"/*) echo "lund_outroot.sh: refusing a root under \$HOME" >&2
               return 1 2>/dev/null || exit 1 ;;
esac
export LUND_OUTROOT
mkdir -p "$LUND_OUTROOT"
