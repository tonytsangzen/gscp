#!/usr/bin/env bash
# gscp 打包脚本（macOS 上执行，产出 macOS app/dmg）
# 用法: ./scripts/package-macos.sh [dmg|app|all]
set -euo pipefail
cd "$(dirname "$0")/.."

BUNDLE="${1:-app}"
echo "==> cargo tauri build (bundles: $BUNDLE)"
command cargo tauri build --bundles "$BUNDLE"

echo "==> 产物:"
ls -lh target/release/bundle/macos/ 2>/dev/null || true
ls -lh target/release/bundle/dmg/ 2>/dev/null || true
BIN=target/release/gscp
if [ -f "$BIN" ]; then
  echo "==> 二进制体积: $(du -h "$BIN" | cut -f1)"
fi
