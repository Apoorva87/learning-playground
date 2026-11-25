#!/bin/sh
# Package the pinned visualizer for a static web host. Leave the development checkout intact.
set -eu
here=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
root=$(CDPATH= cd -- "$here/../.." && pwd)
tool="$root/.local/tools/llm-viz"
source="$tool/source"
"$root/install.sh" llm-viz
if [ ! -x "$tool/bundler/node_modules/.bin/vite" ] ||
   [ "$(node -p 'require(process.argv[1]).version' "$tool/bundler/node_modules/vite/package.json")" != '6.3.5' ]; then
    npm install --prefix "$tool/bundler" --no-save --no-package-lock --no-audit --no-fund --prefer-offline vite@6.3.5
fi

stage=$(mktemp -d "$tool/player-build.XXXXXX")
trap 'rm -rf "$stage"' 0
git -C "$source" archive -o "$stage/input.tar" HEAD -- src/llm src/utils styles \
    tsconfig.json tailwind.config.js postcss.config.mjs public/native.wasm \
    public/gpt-nano-sort-model.json public/gpt-nano-sort-t0-partials.json \
    public/fonts/font-atlas.png public/fonts/font-def.json
tar -xf "$stage/input.tar" -C "$stage"
ln -s "$source/node_modules" "$stage/node_modules"
cp "$here/player-source/index.html.in" "$stage/index.html"
cp "$here/player-source/entry.tsx" "$here/player-source/vite.config.mjs" "$stage/"
# Upstream serves this one file at the domain root. On Pages it sits next to the player.
sed "s|fetch('/native.wasm')|fetch('./native.wasm')|" "$stage/src/llm/NativeBindings.ts" > "$stage/native.ts"
mv "$stage/native.ts" "$stage/src/llm/NativeBindings.ts"
(cd "$stage" && "$tool/bundler/node_modules/.bin/vite" build)
cp "$here/media/LICENSE.txt" "$stage/dist/notices.txt"
cat "$here/player-source/font-notices.txt" >> "$stage/dist/notices.txt"
for package in react react-dom scheduler clsx d3-color @fortawesome/fontawesome-svg-core \
    @fortawesome/react-fontawesome @fortawesome/free-regular-svg-icons @fortawesome/free-solid-svg-icons; do
    printf '\n--- %s ---\n' "$package" >> "$stage/dist/notices.txt"
    cat "$source/node_modules/$package"/[Ll][Ii][Cc][Ee][Nn][Ss][Ee]* >> "$stage/dist/notices.txt"
done
# This directory contains only generated player assets.
mkdir -p "$here/play"
rsync -a --delete "$stage/dist/" "$here/play/"
