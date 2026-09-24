import com.gscp.desktop.NcnnEngine;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

/** 诊断：nativePose 得到 bbox 后，用同样 1.5× 仿射裁 192 crop，走 nativeRun 直接看 1k3d68 原始输出。 */
public class LmProbe {
    public static void main(String[] args) throws Exception {
        System.load(new File(args[0]).getAbsolutePath());
        NcnnEngine eng = new NcnnEngine();
        long det = eng.nativeInit(args[1], args[2], false, false);
        long lm = eng.nativeInit(args[3], args[4], false, false);
        for (int ai = 5; ai < args.length; ai++) {
            BufferedImage img = ImageIO.read(new File(args[ai]));
            int w = img.getWidth(), h = img.getHeight();
            byte[] bgr = new byte[w * h * 3];
            for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y);
                int o = (y * w + x) * 3;
                bgr[o] = (byte) rgb; bgr[o + 1] = (byte) (rgb >> 8); bgr[o + 2] = (byte) (rgb >> 16);
            }
            eng.nativeResetTrack();
            float[] p = new float[26];
            int ok = eng.nativePose(det, lm, bgr, w, h, p);
            System.out.printf("%s: nativePose ok=%d bbox=(%.1f,%.1f,%.1f,%.1f)%n",
                    args[ai], ok, p[12], p[13], p[14], p[15]);
            if (ok == 0) continue;
            // 复刻 lm_input：1.5× 居中 crop → 192
            float bx1 = p[12], by1 = p[13], bx2 = p[14], by2 = p[15];
            float sM = 192f / (Math.max(bx2 - bx1, by2 - by1) * 1.5f);
            float cx = (bx1 + bx2) / 2, cy = (by1 + by2) / 2;
            float[] planar = new float[3 * 192 * 192];
            for (int py = 0; py < 192; py++) for (int px = 0; px < 192; px++) {
                float u = (px - 96) / sM + cx, v = (py - 96) / sM + cy;
                int u0 = (int) Math.floor(u), v0 = (int) Math.floor(v);
                float fu = u - u0, fv = v - v0, r = 0, g = 0, b = 0;
                if (u0 >= 0 && v0 >= 0 && u0 + 1 < w && v0 + 1 < h) {
                    for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
                        float wt = (dx == 0 ? 1 - fu : fu) * (dy == 0 ? 1 - fv : fv);
                        int o = ((v0 + dy) * w + u0 + dx) * 3;
                        b += wt * (bgr[o] & 255); g += wt * (bgr[o + 1] & 255); r += wt * (bgr[o + 2] & 255);
                    }
                }
                planar[py * 192 + px] = r;
                planar[192 * 192 + py * 192 + px] = g;
                planar[2 * 192 * 192 + py * 192 + px] = b;
            }
            double cr = 0, cg = 0, cb = 0;
            for (int i = 0; i < 192 * 192; i++) { cr += planar[i]; cg += planar[36864 + i]; cb += planar[73728 + i]; }
            System.out.printf("  crop mean rgb = %.1f %.1f %.1f%n", cr / 36864, cg / 36864, cb / 36864);
            int[] dims = new int[]{0};
            float[] pred = eng.nativeRun(lm, "in0", planar, new String[]{"out0"}, dims);
            if (pred == null) { System.out.println("  nativeRun null"); continue; }
            System.out.printf("  out total=%d first6=", dims[0]);
            for (int i = 0; i < Math.min(6, pred.length); i++) System.out.printf(" %.4f", pred[i]);
            double s = 0; for (float f : pred) s += f;
            System.out.printf("  mean=%.4f  absmax=%.4f%n", s / pred.length, maxAbs(pred));
            // 最后 68 点 (base=npred/3-68) 的前 3 点 x,y,z
            int np = dims[0], base = np / 3 - 68;
            System.out.printf("  np=%d base=%d  pts:", np, base);
            for (int i = 0; i < 3; i++) System.out.printf(" (%.2f,%.2f,%.2f)",
                    pred[(base + i) * 3], pred[(base + i) * 3 + 1], pred[(base + i) * 3 + 2]);
            System.out.println();
        }
    }
    static double maxAbs(float[] a) { double m = 0; for (float f : a) m = Math.max(m, Math.abs(f)); return m; }
}
