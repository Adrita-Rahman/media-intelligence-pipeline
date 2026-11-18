#!/bin/zsh
# Turn the 9 pm nightly run on or off (scripts_content_v1/nightly.py).
#   deploy/launchd/install.sh on      copy the job to ~/Library/LaunchAgents and load it
#   deploy/launchd/install.sh off     unload and remove it
#   deploy/launchd/install.sh status  is it loaded, and when did it last run
#   deploy/launchd/install.sh now     start one run now (as launchd would at 21:00)
#   deploy/launchd/install.sh check   can a background job read the NAS and the drive (privacy)
#   deploy/launchd/install.sh viewer-on|viewer-off|viewer-status
#                                     the file viewer (open NAS files from the Content Desk)
set -e
LABEL=com.klinikdrinder.kdi-nightly
SRC=${0:A:h}/$LABEL.plist
DST=~/Library/LaunchAgents/$LABEL.plist
mkdir -p /Volumes/WorkDrive/repos/kdi-content-v1-pipeline/work/logs/nightly
case ${1:-status} in
  on)
    plutil -lint $SRC
    cp $SRC $DST
    launchctl bootout gui/$(id -u)/$LABEL 2>/dev/null || true
    launchctl bootstrap gui/$(id -u) $DST
    echo "on: runs every day at 21:00" ;;
  off)
    launchctl bootout gui/$(id -u)/$LABEL 2>/dev/null || true
    rm -f $DST
    echo "off" ;;
  now)
    launchctl kickstart gui/$(id -u)/$LABEL
    echo "started; log: work/logs/nightly/" ;;
  check)
    # can a background job (as at 21:00) read the NAS and the work drive? macOS privacy
    # blocks it until Python has Full Disk Access. Output goes to /tmp (always writable).
    OUT=/tmp/kdi-nightly-access.txt; rm -f $OUT
    launchctl submit -l $LABEL.check -- /Volumes/WorkDrive/repos/kdi-content-v1-pipeline/venv/bin/python -c "
import os
r = []
for p in ['/Volumes/MediaShare', '/Volumes/WorkDrive/batches', '/Volumes/WorkDrive/repos/kdi-content-v1-pipeline/work']:
    try: os.listdir(p); r.append('OK      ' + p)
    except Exception as e: r.append('BLOCKED ' + p + '  (' + type(e).__name__ + ')')
open('$OUT', 'w').write(chr(10).join(r) + chr(10))"
    for i in {1..30}; do [[ -s $OUT ]] && break; sleep 1; done
    launchctl remove $LABEL.check 2>/dev/null || true
    if [[ -s $OUT ]]; then cat $OUT; else echo "no answer in 30 s: Python could not even start from the work drive (blocked)"; fi ;;
  status)
    if launchctl print gui/$(id -u)/$LABEL >/dev/null 2>&1; then
      launchctl print gui/$(id -u)/$LABEL | grep -E "state|last exit|runs" | sed 's/^/  /'
    else
      echo "not installed"
    fi ;;
  viewer-on)
    VL=com.klinikdrinder.kdi-viewer
    plutil -lint ${0:A:h}/$VL.plist
    cp ${0:A:h}/$VL.plist ~/Library/LaunchAgents/$VL.plist
    launchctl bootout gui/$(id -u)/$VL 2>/dev/null || true
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/$VL.plist
    echo "viewer on: port 8790, starts at login; log: /tmp/kdi-viewer.log" ;;
  viewer-off)
    launchctl bootout gui/$(id -u)/com.klinikdrinder.kdi-viewer 2>/dev/null || true
    rm -f ~/Library/LaunchAgents/com.klinikdrinder.kdi-viewer.plist
    echo "viewer off" ;;
  viewer-status)
    launchctl print gui/$(id -u)/com.klinikdrinder.kdi-viewer 2>/dev/null | grep -E "state|last exit|runs" | sed 's/^/  /' || echo "not installed"
    curl -s -o /dev/null -w "  answers on port 8790: %{http_code}\n" -m 5 http://127.0.0.1:8790/health || echo "  not answering" ;;
  *) echo "usage: $0 on|off|status|now|check|viewer-on|viewer-off|viewer-status"; exit 1 ;;
esac
