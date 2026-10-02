#!/usr/bin/env bash
# iOS 真机构建（免费 Apple ID 个人团队签名）。
#
# 前提（一次性）：
#   1. 完整 Xcode 已安装（免费证书只能在 Xcode 里自动生成，网页申请仅付费账号）
#   2. Xcode → Settings → Accounts 登录 Apple ID（个人团队 = "Your Name (Personal Team)"）
#   3. iPhone 用数据线连接，并在 手机端 设置 → 隐私与安全性 → 开发者模式 打开
#   4. 首次在 Xcode 里打开本项目、选中个人 Team 构建一次（Xcode 会自动生成
#      "Apple Development" 证书 + 免费描述文件，并注册设备）
#
# 之后即可用本脚本命令行出包（7 天有效期，过期重跑即可续签）：
#   ios/scripts/build-device.sh            # 构建 + 安装到已连接的 iPhone
#
# 免费签名的限制：profile 7 天过期、每设备每周 3 个应用、无法进 CI。
set -euo pipefail
cd "$(dirname "$0")/../App"

# ── 自检 ─────────────────────────────────────────────
command -v xcodebuild >/dev/null || { echo "✗ 未找到 xcodebuild：请先安装完整 Xcode 并 xcode-select -s 切换"; exit 1; }
xcodebuild -version

# 自动识别个人团队的开发证书（免费签名创建后即出现在钥匙串）
IDENTITY=$(security find-identity -p codesigning -v 2>/dev/null | awk -F'"' '/Apple Development/ {print $2; exit}')
if [[ -z "${IDENTITY:-}" ]]; then
  echo "✗ 钥匙串里没有 'Apple Development' 证书。"
  echo "  请先在 Xcode 里用个人 Team 构建一次（自动生成证书），再跑本脚本。"
  exit 1
fi
echo "✓ 签名证书: $IDENTITY"

# 从证书/profile 反查 TeamID（个人团队 10 位字符）
TEAM=$(security find-certificate -c "Apple Development" -p 2>/dev/null \
  | openssl x509 -noout -subject 2>/dev/null | sed -n 's/.*OU=\([A-Z0-9]*\).*/\1/p')
[[ -n "${TEAM:-}" ]] || { echo "✗ 未能从证书解析 TeamID"; exit 1; }
echo "✓ TeamID:   $TEAM"

# ── 构建（真机，开发签名）────────────────────────────
APP_BUNDLE_ID=$(sed -n 's/.*PRODUCT_BUNDLE_IDENTIFIER = \(.*\);/\1/p' \
  GSCP.xcodeproj/project.pbxproj | head -n1 | xargs)
echo "✓ App ID:   $APP_BUNDLE_ID"

xcodebuild \
  -project GSCP.xcodeproj \
  -target GSCP \
  -configuration Release \
  -sdk iphoneos \
  "CODE_SIGN_IDENTITY=$IDENTITY" \
  "DEVELOPMENT_TEAM=$TEAM" \
  "PRODUCT_BUNDLE_IDENTIFIER=$APP_BUNDLE_ID" \
  -allowProvisioningUpdates \
  SYMROOT="$PWD/build-device" \
  build

APP=$(ls -d build-device/Release-iphoneos/GSCP.app)
echo "✓ 构建: $APP"

# ── 打包 .ipa（Payload 结构）─────────────────────────
STAGE=$(mktemp -d)
mkdir -p "$STAGE/Payload"
cp -R "$APP" "$STAGE/Payload/"
cd "$STAGE"
zip -qr "$OLDPWD/GSCP-device.ipa" Payload
cd - >/dev/null
rm -rf "$STAGE"
echo "✓ IPA:  $(pwd)/GSCP-device.ipa"

# ── 安装到已连接设备（可选）──────────────────────────
if xcrun devicectl list devices 2>/dev/null | grep -q "Available"; then
  read -r -p "检测到已连接的 iPhone，是否安装？[y/N] " yn
  if [[ "$yn" =~ ^[Yy]$ ]]; then
    DEV=$(xcrun devicectl list devices | awk '/Available/{print $1; exit}')
    xcrun devicectl device install app --device "$DEV" "$(pwd)/GSCP-device.ipa"
    echo "✓ 已安装到设备 $DEV（首次安装需在手机上：设置 → 通用 → VPN与设备管理 → 信任开发者证书）"
  fi
else
  echo "（未检测到已连接设备；ipa 已产出，可手动 AirDrop/devicectl 安装）"
fi
