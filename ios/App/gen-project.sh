#!/usr/bin/env bash
# 生成 Xcode 工程：需要 xcodegen（brew install xcodegen）。
# 若手写的 GSCP.xcodeproj 已能打开（Xcode 16+），无需运行本脚本。
set -euo pipefail
cd "$(dirname "$0")"
xcodegen generate
echo "✓ GSCP.xcodeproj 已生成（双击打开，Cmd+R 运行到 iPhone 模拟器/真机）"
