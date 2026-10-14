//! Throwaway verification tool for the CRX adaptive-QP range (not for merging).
//!
//! `qpverify OUT_DIR FILE_OR_DIR...` decodes every CR3 and prints, per file: decode result, an FNV-1a hash of
//! the sensor samples, the QP range of its maps, how well the green mosaic follows the camera JPEG (Pearson r
//! on 24 × 16 cells), and, per QP value of the full-size map, the ratio of fine detail in the decode to fine
//! detail in the camera JPEG over the same 16 × 4 sensor cells. A wrong quantiser step at some QP shows up as
//! a jump in that ratio. It also writes side-by-side crops (decode | camera JPEG) of the highest-QP area.
use lightcraft_raw::{QP_LOG, RawData, RawImage, decode, embedded_preview};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn fnv(d: &[u16]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in d {
        for b in v.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

struct Jpeg {
    w: usize,
    h: usize,
    g: Vec<f64>,
}

fn jpeg(bytes: &[u8]) -> Option<Jpeg> {
    use zune_core::{bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions};
    let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut d = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(bytes), opts);
    let px = d.decode().ok()?;
    let info = d.info()?;
    let (w, h) = (info.width as usize, info.height as usize);
    Some(Jpeg { w, h, g: (0..w * h).map(|i| f64::from(px[i * 3 + 1]) / 255.0).collect() })
}

/// Gamma-encoded green plane on green sites, NaN elsewhere.
fn green(raw: &RawImage) -> Option<Vec<f64>> {
    let RawData::U16(s) = &raw.data else { return None };
    let cfa = raw.cfa.clone()?;
    let (black, white) = (f64::from(raw.black.mean()), f64::from(raw.white_at(0)));
    let mut g = vec![f64::NAN; raw.width * raw.height];
    for y in 0..raw.height {
        for x in 0..raw.width {
            if cfa.pattern[(y % 2) * 2 + x % 2] == 1 {
                g[y * raw.width + x] = ((f64::from(s[y * raw.width + x]) - black).max(0.0) / (white - black)).powf(1.0 / 2.2);
            }
        }
    }
    Some(g)
}

fn pearson(r: &[f64], j: &[f64]) -> f64 {
    let n = r.len() as f64;
    let (mr, mj) = (r.iter().sum::<f64>() / n, j.iter().sum::<f64>() / n);
    let cov: f64 = r.iter().zip(j).map(|(x, y)| (x - mr) * (y - mj)).sum();
    let vr: f64 = r.iter().map(|x| (x - mr).powi(2)).sum();
    let vj: f64 = j.iter().map(|y| (y - mj).powi(2)).sum();
    cov / (vr * vj).sqrt()
}

fn files(arg: &Path, out: &mut Vec<PathBuf>) {
    if arg.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(arg).map(|d| d.flatten().map(|e| e.path()).collect()).unwrap_or_default();
        entries.sort();
        for p in entries {
            files(&p, out);
        }
    } else if arg.extension().is_some_and(|e| e.eq_ignore_ascii_case("cr3")) {
        out.push(arg.to_path_buf());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_dir = PathBuf::from(&args[0]);
    std::fs::create_dir_all(&out_dir).expect("out dir");
    let mut list = Vec::new();
    for a in &args[1..] {
        files(Path::new(a), &mut list);
    }
    let mut pooled: BTreeMap<i32, (f64, f64, usize)> = BTreeMap::new();
    for path in &list {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                println!("{name}\tREAD-ERROR\t{e}");
                continue;
            }
        };
        QP_LOG.lock().unwrap().clear();
        let raw = match decode(&bytes) {
            Ok(r) => r,
            Err(e) => {
                println!("{name}\tDECODE-ERROR\t{e}");
                continue;
            }
        };
        let maps = std::mem::take(&mut *QP_LOG.lock().unwrap());
        let RawData::U16(samples) = &raw.data else {
            println!("{name}\tFLOAT");
            continue;
        };
        let hash = fnv(samples);
        let (qmin, qmax) = maps.iter().flat_map(|m| m.2.iter()).fold((i32::MAX, i32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        let over: usize = maps.iter().flat_map(|m| m.2.iter()).filter(|&&v| v > 168).count();
        let model = raw.metadata.model.clone().unwrap_or_default();
        let mut line = format!("{name}\tOK\t{model}\t{}x{}\tfnv={hash:016x}\tmaps={}", raw.width, raw.height, maps.len());
        if !maps.is_empty() {
            line += &format!("\tqp={qmin}..{qmax}\tcells>168={over}");
        }
        let (Some(g), Some(j)) = (green(&raw), embedded_preview(&bytes).as_deref().and_then(jpeg)) else {
            println!("{line}\tno-jpeg");
            continue;
        };
        let a = raw.active_area;
        // Block means for the Pearson r (24 x 16 cells of the active area).
        let (gw, gh) = (24, 16);
        let (mut rs, mut rn, mut js, mut jn) = (vec![0.0; gw * gh], vec![0.0; gw * gh], vec![0.0; gw * gh], vec![0.0; gw * gh]);
        for y in a.y..a.y + a.height {
            for x in a.x..a.x + a.width {
                let v = g[y * raw.width + x];
                if !v.is_nan() {
                    let i = ((y - a.y) * gh / a.height) * gw + (x - a.x) * gw / a.width;
                    rs[i] += v;
                    rn[i] += 1.0;
                }
            }
        }
        for y in 0..j.h {
            for x in 0..j.w {
                let i = (y * gh / j.h) * gw + x * gw / j.w;
                js[i] += j.g[y * j.w + x];
                jn[i] += 1.0;
            }
        }
        let rm: Vec<f64> = rs.iter().zip(&rn).map(|(s, n)| s / f64::max(*n, 1.0)).collect();
        let jm: Vec<f64> = js.iter().zip(&jn).map(|(s, n)| s / f64::max(*n, 1.0)).collect();
        line += &format!("\tr={:.3}", pearson(&rm, &jm));
        // Fine detail per QP value on the full-size map (16 x 4 sensor pixels per cell).
        let Some(map) = maps.iter().max_by_key(|m| m.0 * m.1) else {
            println!("{line}");
            continue;
        };
        let (sx, sy) = (j.w as f64 / a.width as f64, j.h as f64 / a.height as f64);
        let mut per_qp: BTreeMap<i32, (f64, f64, usize)> = BTreeMap::new();
        let mut best = (i32::MIN, 0usize, 0usize);
        for cy in 0..map.1 {
            for cx in 0..map.0 {
                let qp = map.2[cy * map.0 + cx];
                let (x0, y0) = (cx * 16, cy * 4);
                if x0 < a.x + 2 || y0 < a.y + 2 || x0 + 18 > a.x + a.width || y0 + 6 > a.y + a.height {
                    continue;
                }
                let (mut rd, mut rc) = (0.0, 0.0);
                for y in y0..y0 + 4 {
                    for x in x0..x0 + 16 {
                        let v = g[y * raw.width + x];
                        if v.is_nan() {
                            continue;
                        }
                        let d = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
                            .iter()
                            .map(|&(dx, dy)| g[(y as isize + dy) as usize * raw.width + (x as isize + dx) as usize])
                            .sum::<f64>()
                            / 4.0;
                        rd += (v - d).abs();
                        rc += 1.0;
                    }
                }
                let (jx0, jy0) = (((x0 - a.x) as f64 * sx) as usize, ((y0 - a.y) as f64 * sy) as usize);
                let (jx1, jy1) = ((((x0 + 16 - a.x) as f64 * sx) as usize).min(j.w - 1), (((y0 + 4 - a.y) as f64 * sy) as usize).min(j.h - 1));
                let (mut jd, mut jc) = (0.0, 0.0);
                for y in jy0.max(1)..jy1.max(jy0.max(1) + 1).min(j.h - 1) {
                    for x in jx0.max(1)..jx1.max(jx0.max(1) + 1).min(j.w - 1) {
                        let v = j.g[y * j.w + x];
                        let n = (j.g[(y - 1) * j.w + x] + j.g[(y + 1) * j.w + x] + j.g[y * j.w + x - 1] + j.g[y * j.w + x + 1]) / 4.0;
                        jd += (v - n).abs();
                        jc += 1.0;
                    }
                }
                if rc == 0.0 || jc == 0.0 {
                    continue;
                }
                let e = per_qp.entry(qp).or_default();
                e.0 += rd / rc;
                e.1 += jd / jc;
                e.2 += 1;
                let p = pooled.entry(qp).or_default();
                p.0 += rd / rc;
                p.1 += jd / jc;
                p.2 += 1;
                if qp > best.0 {
                    best = (qp, x0, y0);
                }
            }
        }
        println!("{line}");
        if qmax > 168 {
            for (qp, (rd, jd, n)) in &per_qp {
                println!("    qp {qp}: cells {n:7}  decode/jpeg detail {:.3}", rd / jd);
            }
            // Crop: 384 x 256 sensor pixels round the highest-QP cell; left green-site gray (2x2 binned), right JPEG.
            let (cw, ch) = (384usize, 256usize);
            let cx0 = best.1.saturating_sub(cw / 2).clamp(a.x, a.x + a.width - cw) & !1;
            let cy0 = best.2.saturating_sub(ch / 2).clamp(a.y, a.y + a.height - ch) & !1;
            let (ow, oh) = (cw, ch / 2);
            let mut px = vec![0u8; ow * 2 * oh * 3];
            for oy in 0..oh {
                for ox in 0..ow / 2 {
                    let (x, y) = (cx0 + ox * 2, cy0 + oy * 2);
                    let mut s = 0.0;
                    let mut n = 0.0;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let v = g[(y + dy) * raw.width + x + dx];
                        if !v.is_nan() {
                            s += v;
                            n += 1.0;
                        }
                    }
                    let v = (s / n * 255.0).clamp(0.0, 255.0) as u8;
                    for k in 0..3 {
                        px[(oy * ow * 2 + ox * 2) * 3 + k] = v;
                        px[(oy * ow * 2 + ox * 2 + 1) * 3 + k] = v;
                    }
                    let (jx, jy) = (((x - a.x) as f64 * sx) as usize, ((y - a.y) as f64 * sy) as usize);
                    let jv = (j.g[jy.min(j.h - 1) * j.w + jx.min(j.w - 1)] * 255.0) as u8;
                    for k in 0..3 {
                        px[(oy * ow * 2 + ow + ox * 2) * 3 + k] = jv;
                        px[(oy * ow * 2 + ow + ox * 2 + 1) * 3 + k] = jv;
                    }
                }
            }
            let out = out_dir.join(format!("{name}-qp{}-crop.jpg", best.0));
            let mut buf = Vec::new();
            jpeg_encoder::Encoder::new(&mut buf, 95).encode(&px, (ow * 2) as u16, oh as u16, jpeg_encoder::ColorType::Rgb).expect("encode");
            std::fs::write(&out, buf).expect("write crop");
            println!("    crop: {}", out.display());
        }
    }
    println!("POOLED decode/jpeg fine-detail ratio per QP (all files):");
    for (qp, (rd, jd, n)) in &pooled {
        println!("  qp {qp}: cells {n:9}  ratio {:.3}", rd / jd);
    }
}
