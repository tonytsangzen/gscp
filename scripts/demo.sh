#!/usr/bin/env bash
# 启动 mock scrcpy 三路服务器 + 本地播放器，用于无眼镜环境的端到端演示。
# 用法: ./scripts/demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command cargo build -p gscp-player --example mock_server

echo "==> 启动 mock scrcpy server..."
./target/debug/examples/mock_server > /tmp/gscp-mock.log 2>&1 &
MOCK_PID=$!
trap 'kill $MOCK_PID 2>/dev/null || true' EXIT
sleep 2
LINE=$(head -1 /tmp/gscp-mock.log)
echo "    $LINE"
CAMERA=$(echo "$LINE" | sed -E 's/.*camera=([0-9]+).*/\1/')
OVERLAY=$(echo "$LINE" | sed -E 's/.*overlay=([0-9]+).*/\1/')
AUDIO=$(echo "$LINE" | sed -E 's/.*audio=([0-9]+).*/\1/')

echo "==> 启动播放器（Esc 退出，双击切换全屏）..."
command cargo run -p gscp-app -- --player \
  --remote-host 127.0.0.1 \
  --camera-port "$CAMERA" \
  --remote-overlay-port "$OVERLAY" \
  --audio-port "$AUDIO" \
  --canvas-width 960 --canvas-height 720 \
  --title "GSCP Mock 演示"
