#!/bin/sh
set -eu

PLAYGROUND_ROOT=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
export PLAYGROUND_ROOT

install_tool() {
    tool_name=$1
    case "$tool_name" in
        ''|[!a-z0-9]*|*[!a-z0-9_-]*)
            echo "Invalid tool name: $tool_name" >&2
            return 1
            ;;
    esac

    recipe_dir="$PLAYGROUND_ROOT/tools/$tool_name"
    if [ ! -f "$recipe_dir/install.sh" ]; then
        echo "No install recipe at tools/$tool_name/install.sh" >&2
        return 1
    fi

    TOOL_HOME="$PLAYGROUND_ROOT/.local/tools/$tool_name"
    export TOOL_HOME
    mkdir -p "$TOOL_HOME"
    printf 'Installing %s\n' "$tool_name"
    (cd "$recipe_dir" && sh ./install.sh)
}

case $# in
    0)
        found=0
        for recipe in "$PLAYGROUND_ROOT"/tools/*/install.sh; do
            [ -f "$recipe" ] || continue
            recipe_dir=${recipe%/*}
            install_tool "${recipe_dir##*/}"
            found=1
        done
        if [ "$found" -eq 0 ]; then
            echo "No tool recipes yet. Register the first one in tools/."
        fi
        ;;
    1) install_tool "$1" ;;
    *)
        echo "Usage: ./install.sh [tool-name]" >&2
        exit 2
        ;;
esac
