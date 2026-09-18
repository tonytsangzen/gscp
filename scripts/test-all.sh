#!/usr/bin/env bash
# gscp 全量测试脚本：workspace 单测 + 集成测试 + 前端语法检查
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> [1/3] 前端语法检查"
node --check ui/app.js
node --check ui/bridge.js
echo "    OK"

echo "==> [2/3] workspace 单元/集成测试（含 VideoToolbox/OpenH264 真实码流、mock scrcpy 会话）"
command cargo test --workspace

echo "==> [3/3] 完成"
