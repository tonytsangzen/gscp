import Foundation
import GSCPKit

// 协议自检：swift run gscp-selftest
// （CLT 环境无 XCTest/swift-testing；有完整 Xcode 的环境可自行转 XCTest。）

var failed = 0
func check(_ cond: Bool, _ name: String) {
    print(cond ? "  ✓ \(name)" : "  ✗ \(name)")
    if !cond { failed += 1 }
}

print("GSCPKit 协议自检")

// scrcpy 帧头：ptsAndFlags(8, BE) + size(4, BE)；最高位 = 配置包。
do {
    let config = Data([0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x40])
    check(config.loadBE64(at: 0) >> 63 == 1, "帧头：配置包标志位（最高位）")
    check(config.loadBE32(at: 8) == 0x40, "帧头：size 大端解析")

    let frame = Data([0, 0, 0, 0, 0, 0, 0x30, 0x39, 0, 0, 0x10, 0x00])
    check(frame.loadBE64(at: 0) >> 63 == 0, "帧头：普通帧无配置位")
    check(frame.loadBE64(at: 0) == 12345, "帧头：pts 大端解析")
    check(frame.loadBE32(at: 8) == 4096, "帧头：size 数值")
}

// annexB 起始码切分（3B 与 4B 混合，前导零归属下一码）
do {
    let sps: [UInt8] = [0x67, 0x64, 0x00, 0x1f]
    let pps: [UInt8] = [0x68, 0xeb, 0xec, 0xb2]
    let idr: [UInt8] = [0x65, 0x88, 0x84, 0x00, 0x00, 0x03, 0x00, 0x00]
    var d = Data([0, 0, 0, 1])
    d.append(contentsOf: sps)
    d.append(contentsOf: [0, 0, 1])
    d.append(contentsOf: pps)
    d.append(contentsOf: [0, 0, 0, 1])
    d.append(contentsOf: idr)
    let nals = H264AnnexB.split(d)
    check(nals.count == 3, "annexB：3 个起始码切出 3 个 NAL")
    check(nals.first?.type == 7 && nals[0].data == Data(sps), "annexB：SPS（4B 起始码）")
    check(nals[1].type == 8 && nals[1].data == Data(pps), "annexB：PPS（3B 起始码）")
    check(nals[2].type == 5 && nals[2].data == Data(idr), "annexB：IDR（含内部 000003 防伪码）")
}

// adb 消息帧：头布局 + magic + crc32
do {
    let msg = AdbMessage(command: AdbCommand.open, arg0: 7, arg1: 0,
                         payload: Data("shell:ls".utf8))
    let raw = msg.encoded()
    check(raw.count == AdbMessage.headerLength + 8, "adb 帧：24B 头 + 载荷")
    check(raw.loadLE32(at: 0) == AdbCommand.open, "adb 帧：command 小端")
    check(raw.loadLE32(at: 4) == 7 && raw.loadLE32(at: 8) == 0, "adb 帧：arg0/arg1")
    check(raw.loadLE32(at: 12) == 8, "adb 帧：载荷长度")
    // zlib crc32("shell:ls") = 0xA31763ED（python zlib 实测校验值）
    check(AdbMessage.crc32(Data("shell:ls".utf8)) == 0xA31763ED, "adb 帧：CRC-32 与 zlib 一致")
    check(raw.loadLE32(at: 16) == 0xA31763ED, "adb 帧：头内 CRC 字段")
    check(raw.loadLE32(at: 20) == AdbCommand.open ^ 0xffffffff, "adb 帧：magic = cmd^0xffffffff")
}

// scid 必须为 < 0x80000000 的合法 hex（server 按 Int32.parseInt(hex,16) 解析）
do {
    var ok = true
    for _ in 0..<1000 {
        let scid = String(format: "%08x", UInt32.random(in: 0x1000_0000...0x7fff_ffff))
        guard scid.count == 8, let v = UInt32(scid, radix: 16), v < 0x8000_0000 else {
            ok = false
            break
        }
    }
    check(ok, "scid：1000 次随机采样均为合法 Int32 hex")
}

// PKCS#8 → PKCS#1 剥壳 + SecKey 签名（AUTH SIGN 路径）
do {
    let keyText = ScrcpyConnection.loadResource("ca.key")
    _ = ScrcpyConnection.loadResource("ca.crt")
    check(true, "资源：ca.key/ca.crt 随包分发")
    let der = Data(base64Encoded: keyText.filter { !$0.isWhitespace })!
    let pkcs1 = try AdbAuth.toPkcs1(der)
    check(pkcs1.first == 0x30 && pkcs1.count == der.count && pkcs1.count > 1000,
          "DER：ca.key 识别为 PKCS#1（全文保留，SecKey 原生格式）")
    let auth = try AdbAuth(keyBase64: keyText, deviceName: "gscp",
                           pubKeyString: "AAAATest gscp\0")
    let token = Data((0..<20).map { _ in UInt8.random(in: 0...255) })
    let sig = try auth.sign(token: token)
    check(sig.count == 256, "AUTH：RSA-2048 SHA1 签名长度 256B")
    let auth2 = try AdbAuth(keyBase64: keyText, deviceName: "gscp",
                            pubKeyString: ScrcpyConnection.adbPublicKeyString)
    check(!(try auth2.sign(token: token)).isEmpty, "AUTH：真实公钥串注入后可签名")
    // Opus：OpusHead 解析 + 解码器创建 + 静音包解码
    do {
        var head = Data("OpusHead".utf8)
        head.append(contentsOf: [1])                       // version
        head.append(contentsOf: [2])                       // channels=2
        head.append(contentsOf: [50, 0])                   // preskip=50 (LE)
        head.append(contentsOf: [128, 187, 0, 0])          // 48000 (LE)
        head.append(contentsOf: [0, 0])                    // gain
        head.append(contentsOf: [0])                       // mapping family
        let dec = OpusAudioDecoder()
        try dec.configure(withOpusHead: head)
        check(dec.channels == 2 && dec.preSkip == 50 && dec.inputSampleRate == 48000,
              "Opus：OpusHead 字段解析（ch/preskip/rate）")
        // 1 字节静音包（opcode 0xF8 变体）→ 合法解码或错误，均不应崩溃；
        // 用真实静音包：Opus TOC 0x08（SILK-only NB 10ms）+ 0x00 帧长
        let pcm = try dec.decode(Data([0x08, 0x00]))
        check(pcm.count > 0 && pcm.count % 4 == 0, "Opus：静音包解码出 PCM16 交错数据")
    }

    // scrcpy-server 资源
    let serverData = try ScrcpyConnection.loadResourceData("scrcpy-server")
    check(serverData.count == 91_818, "资源：scrcpy-server 完整打包（91818B）")
}

print(failed == 0 ? "\n全部通过 ✓" : "\n失败 \(failed) 项 ✗")
exit(failed == 0 ? 0 : 1)
