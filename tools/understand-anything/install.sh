#!/bin/sh
set -eu
repo=https://github.com/Egonex-AI/Understand-Anything.git
revision=8ec75de1de8c9d2eef685d8d6cbc50782ad31fbb
checkout="$TOOL_HOME/source"

if [ ! -e "$checkout" ]; then
    staging=$(mktemp -d "$TOOL_HOME/checkout.XXXXXX")
    git -C "$staging" init -q
    git -C "$staging" remote add origin "$repo"
    git -C "$staging" fetch -q --depth 1 origin "$revision"
    git -C "$staging" checkout -q --detach "$revision"
    mv "$staging" "$checkout"
fi
if [ "$(git -C "$checkout" remote get-url origin)" != "$repo" ] ||
   [ "$(git -C "$checkout" rev-parse HEAD)" != "$revision" ]; then
    echo "The checkout at $checkout differs from the trial revision. Your copy has been preserved." >&2
    exit 1
fi

verify_archive() {
    python3 - "$1" <<'PY'
import hashlib, pathlib, sys
archive = pathlib.Path(sys.argv[1])
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
if digest != '37393fc364e7c71b13ea657b9b2bca91c8232474dd7e6efb9f627e72a87ef65c':
    sys.exit(f'Unexpected viewer checksum for {archive}')
PY
}
archive="$TOOL_HOME/viewer-v2.9.8.tgz"
if [ ! -s "$archive" ] || ! verify_archive "$archive"; then
    download=$(mktemp "$TOOL_HOME/download.XXXXXX")
    trap 'rm -f "$download"' 0
    curl -fsS -L https://github.com/Egonex-AI/Understand-Anything/releases/download/v2.9.8/understand-anything-viewer.tgz -o "$download"
    verify_archive "$download"
    mv "$download" "$archive"
fi
viewer="$TOOL_HOME/viewer-v2.9.8"
if [ ! -d "$viewer" ]; then
    staging=$(mktemp -d "$TOOL_HOME/viewer.XXXXXX")
    tar -xzf "$archive" -C "$staging" --strip-components=1
    mv "$staging" "$viewer"
fi
node "$viewer/bin/viewer.mjs" --help
