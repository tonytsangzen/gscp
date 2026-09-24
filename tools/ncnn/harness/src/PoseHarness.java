import com.gscp.desktop.NcnnEngine;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import javax.imageio.ImageIO;

/**
 * 离线对齐 harness：用 host(CPU fp32) 编译的 nativePose 跑 ht 的 CoreML 参考 CSV
 * 中同一批图片，比较 yaw/pitch/roll 是否一致。
 *
 * 用法: java -cp pose-harness.jar PoseHarness <dylib> <det.param> <det.bin> <lm.param> <lm.bin> <ref.csv> <out.csv> [stride]
 */
public class PoseHarness {
    static double deg(double r) { return Math.toDegrees(r); }

    public static void main(String[] args) throws Exception {
        if (args.length < 7) { System.err.println("need args"); System.exit(2); }
        System.load(new File(args[0]).getAbsolutePath());
        String detP = args[1], detB = args[2], lmP = args[3], lmB = args[4];
        Path csv = Paths.get(args[5]);
        PrintStream out = new PrintStream(new FileOutputStream(args[6]));
        int stride = args.length > 7 ? Integer.parseInt(args[7]) : 1;

        NcnnEngine eng = new NcnnEngine();
        long det = eng.nativeInit(detP, detB, false, false);
        long lm = eng.nativeInit(lmP, lmB, false, false);
        if (det == 0 || lm == 0) { System.err.println("model load failed"); System.exit(1); }

        out.println("ref,fid,scenario,src,native_ok,yaw,dyaw,dpitch,droll,coreml_yaw,coreml_pitch,coreml_roll");
        List<Double> dy = new ArrayList<>(), dp = new ArrayList<>(), dr = new ArrayList<>();
        Map<String, List<Double>> byScenario = new TreeMap<>();
        Map<String, List<Double>> byDataset = new TreeMap<>();
        Map<String, int[]> cover = new TreeMap<>();   // dataset -> {hit, miss}
        int total = 0, hit = 0, miss = 0, decoded = 0, skipped = 0;
        long t0 = System.currentTimeMillis();

        BufferedReader br = Files.newBufferedReader(csv);
        String line = br.readLine();  // header
        int rowNo = 0;
        while ((line = br.readLine()) != null) {
            rowNo++;
            if (stride > 1 && rowNo % stride != 0) continue;
            String[] c = line.split(",", -1);
            if (c.length < 12 || !c[7].equals("1")) { skipped++; continue; }  // 参考失败的行不参与对齐
            total++;
            String src = c[5];
            if (System.getenv("GSCP_ONLY_CROP") != null
                    && !src.contains("/biwi/") && !src.contains("/metahuman_tracking")) {
                total--; continue;
            }
            // 反向过滤：只跑非裁剪集（aflw/300w）。裁剪帧需要 GSCP_PREP_DIR 复现
            // GtCropSource 预处理，分两批跑再合并可省掉一次全量重跑。
            if (System.getenv("GSCP_SKIP_CROP") != null
                    && (src.contains("/biwi/") || src.contains("/metahuman_tracking"))) {
                total--; continue;
            }
            byte[] bgr = null;
            int w = 0, h = 0;
            // 裁剪集(biwi/metahuman)在参考跑里经 GtCropSource 的 pad=0.4 + upscale=4 预处理，
            // 这里用同名 .bgr（Python 侧同法预生成）复现，否则小图无上下文会检不出。
            String prep = System.getenv("GSCP_PREP_DIR");
            if (prep != null) {
                File pf = new File(prep, (rowNo - 1) + ".bgr");
                if (pf.exists()) {
                    try (InputStream is = new BufferedInputStream(new FileInputStream(pf))) {
                        byte[] hd = new byte[8];
                        int got = 0; while (got < 8) { int r = is.read(hd, got, 8 - got); if (r < 0) break; got += r; }
                        if (got < 8) throw new IOException("short header");
                        // struct.pack('<ii') —— 小端
                        w = (hd[0] & 255) | (hd[1] & 255) << 8 | (hd[2] & 255) << 16 | (hd[3] & 255) << 24;
                        h = (hd[4] & 255) | (hd[5] & 255) << 8 | (hd[6] & 255) << 16 | (hd[7] & 255) << 24;
                        bgr = new byte[w * h * 3];
                        int off = 0; while (off < bgr.length) { int r = is.read(bgr, off, bgr.length - off); if (r < 0) break; off += r; }
                        if (off < bgr.length) throw new IOException("short body");
                    } catch (Exception e) { bgr = null; }
                }
            }
            if (bgr == null) {
                BufferedImage img;
                try { img = ImageIO.read(new File(src)); } catch (Exception e) { img = null; }
                if (img == null) { skipped++; total--; continue; }
                w = img.getWidth(); h = img.getHeight();
                bgr = new byte[w * h * 3];
                for (int y = 0; y < h; y++) {
                    int rowOff = y * w * 3;
                    for (int x = 0; x < w; x++) {
                        int rgb = img.getRGB(x, y);
                        int o = rowOff + x * 3;
                        bgr[o] = (byte) rgb;            // B
                        bgr[o + 1] = (byte) (rgb >> 8); // G
                        bgr[o + 2] = (byte) (rgb >> 16);// R
                    }
                }
            }
            decoded++;
            // 默认逐帧独立（对齐 CoreML 参考）；GSCP_KEEP_TRACK=1 时保留跨帧锁定，
            // 用于喂连续帧序列验证 det 节流的外推质量。
            if (System.getenv("GSCP_KEEP_TRACK") == null) eng.nativeResetTrack();
            float[] p = new float[26];
            int ok = eng.nativePose(det, lm, bgr, w, h, p);
            String ds = datasetOf(src);
            if (ok == 0) {
                miss++; cover.computeIfAbsent(ds, k -> new int[2])[1]++;
                System.err.println("MISS " + src + " " + w + "x" + h);
                out.printf(Locale.US, "%d,%s,%s,%s,0,0,0,0,0,%s,%s,%s%n", rowNo, c[1], c[4], src, c[9], c[10], c[11]);
                continue;
            }
            hit++;
            cover.computeIfAbsent(ds, k -> new int[2])[0]++;
            if (System.getenv("GSCP_DIAG") != null) {
                System.err.printf(Locale.US, "%s bbox=(%.1f,%.1f,%.1f,%.1f) hw=%dx%d r=(%.2f,%.2f,%.2f) u=(%.2f,%.2f,%.2f) n=(%.2f,%.2f,%.2f)%n",
                        src, p[12], p[13], p[14], p[15], w, h, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]);
            }
            // rot_to_ypr（ht geom 同式）。nativePose out[0..8] = insightface P2sRt 的 r1,r2,r3（R 的行）：
            //   yaw_raw=atan2(-r3x,√(r1x²+r2x²)), pitch=atan2(r3y,r3z), roll=atan2(r2x,r1x)
            // 报告约定 = 校准 p012s-++（results/calibration.json, insight_procrustes_coreml）：yaw 取反。
            double yaw = -Math.atan2(-p[6], Math.sqrt(p[0] * p[0] + p[3] * p[3]));
            double pitch = Math.atan2(p[7], p[8]);
            double roll = Math.atan2(p[3], p[0]);
            double ry = Double.parseDouble(c[9]), rp = Double.parseDouble(c[10]), rr = Double.parseDouble(c[11]);
            double dYaw = angDeg(yaw, ry), dPitch = angDeg(pitch, rp), dRoll = angDeg(roll, rr);
            dy.add(dYaw); dp.add(dPitch); dr.add(dRoll);
            byScenario.computeIfAbsent(c[4], k -> new ArrayList<>()).add(Math.max(Math.abs(dYaw), Math.max(Math.abs(dPitch), Math.abs(dRoll))));
            byDataset.computeIfAbsent(ds, k -> new ArrayList<>()).add(Math.abs(dYaw));
            out.printf(Locale.US, "%d,%s,%s,%s,1,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f%n",
                    rowNo, c[1], c[4], src, deg(yaw), dYaw, dPitch, dRoll, ry, rp, rr);
            if (rowNo % 200 == 0) {
                long el = System.currentTimeMillis() - t0;
                System.err.printf("row %d, hit=%d, %.1f ms/frame%n", rowNo, hit, (double) el / decoded);
            }
        }
        br.close();
        out.close();
        eng.nativeRelease(det);
        eng.nativeRelease(lm);

        System.out.printf(Locale.US,
                "frames: total=%d hit=%d miss=%d skipped(ref-fail/decode)=%d  elapsed=%.0fs%n",
                total, hit, miss, skipped, (System.currentTimeMillis() - t0) / 1000.0);
        stat("|dyaw|", dy); stat("|dpitch|", dp); stat("|droll|", dr);
        System.out.println("by scenario (max|Δ| deg):");
        for (var e : byScenario.entrySet()) {
            List<Double> v = e.getValue();
            double p95 = pct(v, 0.95), mean = v.stream().mapToDouble(Double::doubleValue).average().orElse(0);
            System.out.printf(Locale.US, "  %-28s n=%-5d mean=%5.2f p95=%6.2f max=%6.2f%n",
                    e.getKey(), v.size(), mean, p95, Collections.max(v));
        }
        System.out.println("by dataset (|dyaw| vs CoreML, 检出覆盖):");
        for (var e : byDataset.entrySet()) {
            List<Double> v = e.getValue();
            int[] cv = cover.getOrDefault(e.getKey(), new int[2]);
            System.out.printf(Locale.US, "  %-28s hit=%-5d miss=%-5d cov=%5.1f%%  p50=%5.2f p95=%6.2f max=%6.2f%n",
                    e.getKey(), cv[0], cv[1], 100.0 * cv[0] / Math.max(1, cv[0] + cv[1]),
                    pct(v, 0.5), pct(v, 0.95), Collections.max(v));
        }
        double[] pr = eng.nativeProf();
        double detRatio = pr[7] > 0 ? pr[8] / pr[7] : 0;
        System.out.printf(Locale.US,
                "nativePose 阶段均值 µs/帧 (成功=%d 调用=%d 实际det=%d 每%.2f帧一次): det前处理 %.0f | det前向 %.0f | 解码+NMS %.0f | lm裁剪 %.0f | lm前向 %.0f | 解算+Procrustes %.0f%n",
                (long) pr[6], (long) pr[7], (long) pr[8], 1.0 / Math.max(detRatio, 1e-9),
                pr[0], pr[1], pr[2], pr[3], pr[4], pr[5]);
    }

    /** 从 src 路径取数据集名（metahuman_tracking 再带一层子目录）。 */
    static String datasetOf(String src) {
        String[] p = src.split("/");
        for (int i = 0; i + 1 < p.length; i++) {
            if (p[i].equals("datasets")) {
                if (p[i + 1].startsWith("metahuman")) return i + 2 < p.length ? p[i + 1] + "/" + p[i + 2] : p[i + 1];
                return p[i + 1];
            }
        }
        return p[p.length - 2];
    }

    static double angDeg(double a, double ref) {
        double d = deg(a) - ref;
        while (d > 180) d -= 360;
        while (d < -180) d += 360;
        return d;
    }

    static void stat(String name, List<Double> v) {
        if (v.isEmpty()) { System.out.println(name + ": empty"); return; }
        List<Double> abs = v.stream().map(Math::abs).sorted().toList();
        System.out.printf(Locale.US, "%s: n=%d p50=%.3f p95=%.3f max=%.3f%n",
                name, abs.size(), pct(abs, 0.5), pct(abs, 0.95), abs.get(abs.size() - 1));
    }

    static double pct(List<Double> vals, double q) {
        List<Double> s = vals.stream().map(Math::abs).sorted().toList();
        int i = (int) Math.min(s.size() - 1, Math.ceil(q * s.size()) - 1);
        return s.get(Math.max(0, i));
    }
}
