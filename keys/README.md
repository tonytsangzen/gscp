# Android Release 签名密钥

- `gscp-release.keystore`：正式发布签名密钥（RSA 2048，有效期 30 年，别名 `gscp`，DN: CN=GSCP Release, O=tonytsangzen, C=CN）。
- `gscp-release.properties`：keystore 密码与别名配置，`gen/android/app/build.gradle.kts` 的 `signingConfigs.release` 从这里读取。

## 约束

- keystore 与密码随本**私有**仓库分发，便于 CI 直接产出 release 签名 APK。**仓库若转为公开，必须立即新建 keystore 并更换密码**（签名变更后，已安装的旧包需卸载才能安装新包）。
- 密钥丢失或泄露时同理：换 keystore 即无法对同一 `applicationId` 直接升级安装，必须卸载重装。
- 本地没有 `keys/` 目录时，`assembleArm64Release` 回退 debug 签名（可直接安装调试，但不可作为正式发布包）。

## 证书指纹查看

```sh
keytool -list -v -keystore keys/gscp-release.keystore -storepass "$(grep storePassword keys/gscp-release.properties | cut -d= -f2)"
```
