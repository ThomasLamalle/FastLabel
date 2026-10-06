//! Image clustering math: PCA, a UMAP 2D layout and HDBSCAN labels on that
//! layout. Pure functions (run on a task thread), deterministic for a seed.

use std::collections::HashMap;
use std::sync::atomic::Ordering;

use fastlabel_infer::Progress;

/// Dimensions kept by PCA before the neighbour search.
const PCA_DIMS: usize = 50;
/// UMAP neighbours.
const NEIGHBORS: usize = 15;
/// UMAP curve parameters for `min_dist = 0.1`, `spread = 1`.
const UMAP_A: f64 = 1.576_943_460_405_378;
const UMAP_B: f64 = 0.895_060_878_122_785_9;
const NEGATIVE_SAMPLES: usize = 5;
/// HDBSCAN density estimate: distance to this neighbour (at most the
/// minimum cluster size).
const MIN_SAMPLES: usize = 10;

/// Points of a layout, in the order of the input vectors.
#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub points: Vec<[f32; 2]>,
    /// Cluster per point, largest cluster first; -1 = noise.
    pub labels: Vec<i32>,
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(2, |n| n.get().saturating_sub(1).max(1))
}

fn cancelled(p: &Progress) -> bool {
    p.cancel.load(Ordering::Relaxed)
}

/// Full pipeline: PCA → k-nearest neighbours → UMAP → HDBSCAN.
/// `None` when cancelled.
pub fn layout(
    vectors: &[&[f32]],
    min_cluster_size: usize,
    seed: u64,
    progress: &Progress,
) -> Option<Layout> {
    let n = vectors.len();
    let epochs = if n <= 10_000 { 500 } else { 200 };
    progress.total.store(epochs + 3, Ordering::Relaxed);
    progress.done.store(0, Ordering::Relaxed);
    if n == 0 {
        return Some(Layout { points: Vec::new(), labels: Vec::new() });
    }
    let reduced = pca(vectors, PCA_DIMS);
    progress.done.fetch_add(1, Ordering::Relaxed);
    if cancelled(progress) {
        return None;
    }
    let knn = knn(&reduced, NEIGHBORS.min(n - 1));
    progress.done.fetch_add(1, Ordering::Relaxed);
    if cancelled(progress) {
        return None;
    }
    let points = umap(&reduced, &knn, epochs, seed, progress)?;
    let labels = hdbscan(&points, min_cluster_size);
    progress.done.fetch_add(1, Ordering::Relaxed);
    Some(Layout { points, labels })
}

/// Rows projected on the `k` main principal components (fewer when the data
/// has fewer dimensions or rows).
pub fn pca(rows: &[&[f32]], k: usize) -> Vec<Vec<f32>> {
    let n = rows.len();
    let d = rows.first().map_or(0, |r| r.len());
    let k = k.min(d).min(n.max(1));
    if n == 0 || d == 0 {
        return vec![Vec::new(); n];
    }
    let mut mean = vec![0.0f64; d];
    for r in rows {
        for (m, x) in mean.iter_mut().zip(r.iter()) {
            *m += f64::from(*x);
        }
    }
    mean.iter_mut().for_each(|m| *m /= n as f64);

    // Covariance (upper triangle), rows split across threads.
    let chunk = n.div_ceil(threads());
    let partials: Vec<Vec<f64>> = std::thread::scope(|s| {
        let handles: Vec<_> = rows
            .chunks(chunk)
            .map(|part| {
                let mean = &mean;
                s.spawn(move || {
                    let mut cov = vec![0.0f64; d * d];
                    let mut c = vec![0.0f64; d];
                    for r in part {
                        for (ci, (x, m)) in c.iter_mut().zip(r.iter().zip(mean)) {
                            *ci = f64::from(*x) - m;
                        }
                        for i in 0..d {
                            let ci = c[i];
                            let row = &mut cov[i * d..(i + 1) * d];
                            for j in i..d {
                                row[j] += ci * c[j];
                            }
                        }
                    }
                    cov
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("pca thread")).collect()
    });
    let mut cov = vec![0.0f64; d * d];
    for p in partials {
        cov.iter_mut().zip(p).for_each(|(a, b)| *a += b);
    }
    for i in 0..d {
        for j in 0..i {
            cov[i * d + j] = cov[j * d + i];
        }
    }
    let (values, vectors) = jacobi_eigen(cov, d);
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    let comps: Vec<usize> = order.into_iter().take(k).collect();
    rows.iter()
        .map(|r| {
            comps
                .iter()
                .map(|&c| {
                    (0..d).map(|i| (f64::from(r[i]) - mean[i]) * vectors[i * d + c]).sum::<f64>()
                        as f32
                })
                .collect()
        })
        .collect()
}

/// Eigenvalues and eigenvectors (columns of the returned row-major matrix)
/// of a symmetric matrix, cyclic Jacobi rotations.
fn jacobi_eigen(mut a: Vec<f64>, d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0f64; d * d];
    for i in 0..d {
        v[i * d + i] = 1.0;
    }
    let scale: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
    for _sweep in 0..50 {
        let off: f64 = (0..d)
            .flat_map(|i| (i + 1..d).map(move |j| (i, j)))
            .map(|(i, j)| a[i * d + j] * a[i * d + j])
            .sum::<f64>()
            .sqrt();
        if off <= 1e-12 * scale {
            break;
        }
        for p in 0..d {
            for q in p + 1..d {
                let apq = a[p * d + q];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let theta = (a[q * d + q] - a[p * d + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..d {
                    let (akp, akq) = (a[k * d + p], a[k * d + q]);
                    a[k * d + p] = c * akp - s * akq;
                    a[k * d + q] = s * akp + c * akq;
                }
                for k in 0..d {
                    let (apk, aqk) = (a[p * d + k], a[q * d + k]);
                    a[p * d + k] = c * apk - s * aqk;
                    a[q * d + k] = s * apk + c * aqk;
                }
                for k in 0..d {
                    let (vkp, vkq) = (v[k * d + p], v[k * d + q]);
                    v[k * d + p] = c * vkp - s * vkq;
                    v[k * d + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..d).map(|i| a[i * d + i]).collect(), v)
}

/// Exact k nearest neighbours (Euclidean, self excluded), sorted by
/// distance: `(index, distance)` per row.
pub fn knn(rows: &[Vec<f32>], k: usize) -> Vec<Vec<(u32, f32)>> {
    let n = rows.len();
    let chunk = n.div_ceil(threads()).max(1);
    let starts: Vec<usize> = (0..n).step_by(chunk).collect();
    std::thread::scope(|s| {
        let handles: Vec<_> = starts
            .iter()
            .map(|&start| {
                s.spawn(move || {
                    let mut dist: Vec<(u32, f32)> = Vec::with_capacity(n);
                    (start..(start + chunk).min(n))
                        .map(|i| {
                            dist.clear();
                            dist.extend(
                                (0..n)
                                    .filter(|&j| j != i)
                                    .map(|j| (j as u32, sq_dist(&rows[i], &rows[j]))),
                            );
                            let k = k.min(dist.len());
                            if k < dist.len() {
                                dist.select_nth_unstable_by(k, |a, b| a.1.total_cmp(&b.1));
                            }
                            let mut best: Vec<(u32, f32)> =
                                dist[..k].iter().map(|&(j, d)| (j, d.sqrt())).collect();
                            best.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
                            best
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().expect("knn thread")).collect()
    })
}

fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Small deterministic generator (splitmix64).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Fuzzy neighbour graph (UMAP): symmetric edges with weights in (0, 1].
fn fuzzy_graph(knn: &[Vec<(u32, f32)>]) -> Vec<(u32, u32, f64)> {
    let mut directed: HashMap<(u32, u32), (f64, f64)> = HashMap::new();
    for (i, nn) in knn.iter().enumerate() {
        if nn.is_empty() {
            continue;
        }
        let d: Vec<f64> = nn.iter().map(|&(_, d)| f64::from(d)).collect();
        let rho = d.iter().copied().find(|&x| x > 0.0).unwrap_or(0.0);
        let target = (nn.len() as f64).log2();
        let (mut lo, mut hi, mut sigma) = (0.0f64, f64::INFINITY, 1.0f64);
        for _ in 0..64 {
            let psum: f64 = d.iter().map(|&x| (-(x - rho).max(0.0) / sigma).exp()).sum();
            if (psum - target).abs() < 1e-5 {
                break;
            }
            if psum > target {
                hi = sigma;
                sigma = (lo + hi) / 2.0;
            } else {
                lo = sigma;
                sigma = if hi.is_infinite() { sigma * 2.0 } else { (lo + hi) / 2.0 };
            }
        }
        let mean = d.iter().sum::<f64>() / d.len() as f64;
        let sigma = sigma.max(1e-3 * mean).max(1e-12);
        for (&(j, _), &x) in nn.iter().zip(&d) {
            let w = (-(x - rho).max(0.0) / sigma).exp();
            let i = i as u32;
            let e = directed.entry((i.min(j), i.max(j))).or_default();
            if i < j {
                e.0 = w;
            } else {
                e.1 = w;
            }
        }
    }
    let mut edges: Vec<(u32, u32, f64)> = directed
        .into_iter()
        .map(|((i, j), (a, b))| (i, j, a + b - a * b))
        .filter(|e| e.2 > 0.0)
        .collect();
    edges.sort_by_key(|e| (e.0, e.1));
    edges
}

/// UMAP layout optimisation, initialised from the first two principal
/// components. `None` when cancelled.
fn umap(
    rows: &[Vec<f32>],
    knn: &[Vec<(u32, f32)>],
    epochs: usize,
    seed: u64,
    progress: &Progress,
) -> Option<Vec<[f32; 2]>> {
    let n = rows.len();
    let mut rng = Rng(seed);
    let init = |r: &Vec<f32>, c: usize| f64::from(r.get(c).copied().unwrap_or(0.0));
    let extent = rows
        .iter()
        .flat_map(|r| [init(r, 0).abs(), init(r, 1).abs()])
        .fold(0.0f64, f64::max)
        .max(1e-12);
    let mut y: Vec<[f64; 2]> = rows
        .iter()
        .map(|r| {
            let jitter = |rng: &mut Rng| (rng.unit() - 0.5) * 1e-4;
            [
                init(r, 0) / extent * 10.0 + jitter(&mut rng),
                init(r, 1) / extent * 10.0 + jitter(&mut rng),
            ]
        })
        .collect();
    if n < 3 {
        return Some(y.iter().map(|p| [p[0] as f32, p[1] as f32]).collect());
    }

    let graph = fuzzy_graph(knn);
    let max_w = graph.iter().map(|e| e.2).fold(0.0f64, f64::max);
    // Both directions of every edge, too-weak edges dropped.
    let edges: Vec<(usize, usize, f64)> = graph
        .iter()
        .filter(|e| e.2 >= max_w / epochs as f64)
        .flat_map(|&(i, j, w)| [(i as usize, j as usize, w), (j as usize, i as usize, w)])
        .collect();
    let per_sample: Vec<f64> = edges.iter().map(|e| max_w / e.2).collect();
    let per_negative: Vec<f64> = per_sample.iter().map(|e| e / NEGATIVE_SAMPLES as f64).collect();
    let mut next_sample = per_sample.clone();
    let mut next_negative = per_negative.clone();
    let clip = |g: f64| g.clamp(-4.0, 4.0);

    for epoch in 0..epochs {
        if cancelled(progress) {
            return None;
        }
        let alpha = 1.0 - epoch as f64 / epochs as f64;
        let now = epoch as f64;
        for (e, &(j, k, _)) in edges.iter().enumerate() {
            if next_sample[e] > now {
                continue;
            }
            let (cur, other) = (y[j], y[k]);
            let d2 = (cur[0] - other[0]).powi(2) + (cur[1] - other[1]).powi(2);
            if d2 > 0.0 {
                let coeff = -2.0 * UMAP_A * UMAP_B * d2.powf(UMAP_B - 1.0)
                    / (UMAP_A * d2.powf(UMAP_B) + 1.0);
                for dim in 0..2 {
                    let g = clip(coeff * (cur[dim] - other[dim])) * alpha;
                    y[j][dim] += g;
                    y[k][dim] -= g;
                }
            }
            next_sample[e] += per_sample[e];

            let negatives = ((now - next_negative[e]) / per_negative[e]).max(0.0) as usize;
            for _ in 0..negatives {
                let k = rng.below(n);
                if k == j {
                    continue;
                }
                let (cur, other) = (y[j], y[k]);
                let d2 = (cur[0] - other[0]).powi(2) + (cur[1] - other[1]).powi(2);
                let coeff = if d2 > 0.0 {
                    2.0 * UMAP_B / ((0.001 + d2) * (UMAP_A * d2.powf(UMAP_B) + 1.0))
                } else {
                    0.0
                };
                for dim in 0..2 {
                    let g = if coeff > 0.0 { clip(coeff * (cur[dim] - other[dim])) } else { 4.0 };
                    y[j][dim] += g * alpha;
                }
            }
            next_negative[e] += negatives as f64 * per_negative[e];
        }
        progress.done.fetch_add(1, Ordering::Relaxed);
    }
    Some(y.iter().map(|p| [p[0] as f32, p[1] as f32]).collect())
}

/// HDBSCAN (excess of mass) on 2D points; labels by cluster size, largest
/// first, -1 = noise.
pub fn hdbscan(points: &[[f32; 2]], min_cluster_size: usize) -> Vec<i32> {
    let n = points.len();
    let mcs = min_cluster_size.max(2);
    if n < mcs {
        return vec![-1; n];
    }
    let dist = |a: usize, b: usize| {
        let (p, q) = (points[a], points[b]);
        f64::from((p[0] - q[0]).hypot(p[1] - q[1]))
    };

    // Core distance: distance to the MIN_SAMPLES-th neighbour.
    let k = MIN_SAMPLES.min(mcs).min(n - 1).max(1);
    let rows: Vec<Vec<f32>> = points.iter().map(|p| p.to_vec()).collect();
    let core: Vec<f64> =
        knn(&rows, k).iter().map(|nn| nn.last().map_or(0.0, |x| f64::from(x.1))).collect();
    let mreach = |a: usize, b: usize| dist(a, b).max(core[a]).max(core[b]);

    // Minimum spanning tree of the mutual reachability graph (Prim, O(n²)).
    let mut in_tree = vec![false; n];
    let mut best = vec![f64::INFINITY; n];
    let mut from = vec![0usize; n];
    let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(n - 1);
    let mut cur = 0;
    in_tree[0] = true;
    for _ in 1..n {
        let mut next = usize::MAX;
        let mut next_d = f64::INFINITY;
        for j in 0..n {
            if in_tree[j] {
                continue;
            }
            let d = mreach(cur, j);
            if d < best[j] {
                best[j] = d;
                from[j] = cur;
            }
            if best[j] < next_d {
                next_d = best[j];
                next = j;
            }
        }
        in_tree[next] = true;
        edges.push((next_d, from[next], next));
        cur = next;
    }
    edges.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Single-linkage dendrogram: node n + i merges two subtrees at edge i.
    let mut uf: Vec<usize> = (0..2 * n - 1).collect();
    fn find(uf: &mut [usize], mut x: usize) -> usize {
        while uf[x] != x {
            uf[x] = uf[uf[x]];
            x = uf[x];
        }
        x
    }
    let mut children = Vec::with_capacity(n - 1);
    let mut size = vec![1usize; 2 * n - 1];
    for (i, &(d, a, b)) in edges.iter().enumerate() {
        let (ra, rb) = (find(&mut uf, a), find(&mut uf, b));
        let node = n + i;
        uf[ra] = node;
        uf[rb] = node;
        size[node] = size[ra] + size[rb];
        children.push((ra, rb, d));
    }
    let lambda = |d: f64| if d > 0.0 { 1.0 / d } else { f64::MAX };
    let leaves = |node: usize, out: &mut Vec<usize>| {
        let mut stack = vec![node];
        while let Some(x) = stack.pop() {
            if x < n {
                out.push(x);
            } else {
                let (l, r, _) = children[x - n];
                stack.push(l);
                stack.push(r);
            }
        }
    };

    // Condensed tree: clusters (0 = root) with birth lambda and parent;
    // each point falls out of one cluster at some lambda.
    let mut birth = vec![0.0f64];
    let mut parent = vec![usize::MAX];
    let mut stability = vec![0.0f64];
    let mut fell_from = vec![0usize; n];
    let mut stack = vec![(2 * n - 2, 0usize)];
    let mut buf = Vec::new();
    while let Some((node, c)) = stack.pop() {
        if node < n {
            fell_from[node] = c;
            continue;
        }
        let (l, r, d) = children[node - n];
        let lam = lambda(d);
        let (ls, rs) = (size[l], size[r]);
        let mut fall = |sub: usize, stability: &mut Vec<f64>, fell_from: &mut Vec<usize>| {
            buf.clear();
            leaves(sub, &mut buf);
            for &p in &buf {
                fell_from[p] = c;
            }
            stability[c] += (lam.min(1e300) - birth[c]) * buf.len() as f64;
        };
        if ls >= mcs && rs >= mcs {
            for (sub, s) in [(l, ls), (r, rs)] {
                let id = birth.len();
                birth.push(lam);
                parent.push(c);
                stability.push(0.0);
                stability[c] += (lam.min(1e300) - birth[c]) * s as f64;
                stack.push((sub, id));
            }
        } else if ls < mcs && rs < mcs {
            fall(l, &mut stability, &mut fell_from);
            fall(r, &mut stability, &mut fell_from);
        } else if ls < mcs {
            fall(l, &mut stability, &mut fell_from);
            stack.push((r, c));
        } else {
            fall(r, &mut stability, &mut fell_from);
            stack.push((l, c));
        }
    }

    // Excess of mass selection, children (higher ids) before parents.
    let m = birth.len();
    let mut selected = vec![false; m];
    let mut value = vec![0.0f64; m];
    let mut kids: Vec<Vec<usize>> = vec![Vec::new(); m];
    for c in 1..m {
        kids[parent[c]].push(c);
    }
    for c in (1..m).rev() {
        let child_sum: f64 = kids[c].iter().map(|&k| value[k]).sum();
        if kids[c].is_empty() || stability[c] >= child_sum {
            selected[c] = true;
            value[c] = stability[c];
            let mut st: Vec<usize> = kids[c].clone();
            while let Some(x) = st.pop() {
                selected[x] = false;
                st.extend(&kids[x]);
            }
        } else {
            value[c] = child_sum;
        }
    }

    let label_of = |mut c: usize| {
        while c != 0 {
            if selected[c] {
                return Some(c);
            }
            c = parent[c];
        }
        None
    };
    let raw: Vec<Option<usize>> = fell_from.iter().map(|&c| label_of(c)).collect();
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for c in raw.iter().flatten() {
        *counts.entry(*c).or_default() += 1;
    }
    let mut order: Vec<(usize, usize)> = counts.into_iter().collect();
    order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let rank: HashMap<usize, i32> =
        order.iter().enumerate().map(|(i, (c, _))| (*c, i as i32)).collect();
    raw.iter().map(|c| c.map_or(-1, |c| rank[&c])).collect()
}

/// Targets of an automatic split by cluster.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct AutoSplit {
    /// Shares of frames for val and test (train gets the rest).
    pub val: f64,
    pub test: f64,
    /// Largest share of a split's target one group may fill, so val / test
    /// span several scenes; bigger groups stay in train.
    pub max_group_share: f64,
    pub seed: u64,
}

impl Default for AutoSplit {
    fn default() -> Self {
        Self { val: 0.1, test: 0.1, max_group_share: 0.25, seed: 1 }
    }
}

/// Assigns whole groups (clusters; noise frames are groups of one) to
/// splits so near-identical frames never straddle two splits. Groups are
/// visited in a seeded random order, small ones (fillers) last and largest
/// first; each goes to val or test when it fits there (at most
/// `max_group_share` of the split's target, and getting the split closer to
/// its target), otherwise to train.
pub fn auto_split(sizes: &[usize], p: &AutoSplit) -> Vec<fastlabel_core::Split> {
    use fastlabel_core::Split;
    let total: usize = sizes.iter().sum();
    let targets = [p.val.clamp(0.0, 1.0) * total as f64, p.test.clamp(0.0, 1.0) * total as f64];
    let mut counts = [0.0f64; 2];
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    let mut rng = Rng(p.seed);
    for i in (1..order.len()).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let filler = (p.max_group_share * targets[0].min(targets[1]) / 4.0).max(1.0);
    order.sort_by_key(|&g| {
        let small = sizes[g] as f64 <= filler;
        (small, if small { usize::MAX - sizes[g] } else { 0 })
    });
    let mut out = vec![Split::Train; sizes.len()];
    for g in order {
        let s = sizes[g] as f64;
        let best = (0..2)
            .filter(|&k| {
                let cap = (p.max_group_share * targets[k]).max(1.0);
                s <= cap && (targets[k] - (counts[k] + s)).abs() < (targets[k] - counts[k]).abs()
            })
            .max_by(|&a, &b| {
                let need = |k: usize| (targets[k] - counts[k]) / targets[k].max(1.0);
                need(a).total_cmp(&need(b))
            });
        if let Some(k) = best {
            counts[k] += s;
            out[g] = [Split::Val, Split::Test][k];
        }
    }
    out
}

/// Ray casting: is `p` inside the polygon?
pub fn in_polygon(p: [f32; 2], poly: &[[f32; 2]]) -> bool {
    let mut inside = false;
    let mut j = poly.len().wrapping_sub(1);
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[j]);
        if (a[1] > p[1]) != (b[1] > p[1])
            && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `per` points around each center, deterministic jitter.
    fn blobs(centers: &[Vec<f32>], per: usize, spread: f32) -> Vec<Vec<f32>> {
        let mut rng = Rng(7);
        centers
            .iter()
            .flat_map(|c| {
                (0..per)
                    .map(|_| c.iter().map(|x| x + (rng.unit() as f32 - 0.5) * spread).collect())
                    .collect::<Vec<Vec<f32>>>()
            })
            .collect()
    }

    #[test]
    fn eigen_decomposition_of_a_known_matrix() {
        let (values, vectors) = jacobi_eigen(vec![2.0, 1.0, 1.0, 2.0], 2);
        let mut sorted = values.clone();
        sorted.sort_by(f64::total_cmp);
        assert!((sorted[0] - 1.0).abs() < 1e-9 && (sorted[1] - 3.0).abs() < 1e-9, "{values:?}");
        let top = if values[0] > values[1] { 0 } else { 1 };
        assert!((vectors[top].abs() - vectors[2 + top].abs()).abs() < 1e-9);
    }

    #[test]
    fn pca_keeps_the_direction_of_largest_variance() {
        // Points along (1, 1, 0) with a little noise elsewhere.
        let rows: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![i as f32, i as f32 + 0.1 * (i % 3) as f32, 0.05 * (i % 2) as f32])
            .collect();
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        let out = pca(&refs, 1);
        assert_eq!(out[0].len(), 1);
        let spread = out.iter().map(|r| r[0]).fold(f32::MIN, f32::max)
            - out.iter().map(|r| r[0]).fold(f32::MAX, f32::min);
        assert!((spread - 49.0 * 2f32.sqrt()).abs() < 0.5, "{spread}");
        assert_eq!(pca(&refs, 10)[0].len(), 3, "never more than the input dimensions");
    }

    #[test]
    fn knn_finds_closest_points_without_self() {
        let rows = vec![vec![0.0], vec![1.0], vec![3.0], vec![10.0]];
        let nn = knn(&rows, 2);
        assert_eq!(nn[0], vec![(1, 1.0), (2, 3.0)]);
        assert_eq!(nn[3], vec![(2, 7.0), (1, 9.0)]);
    }

    #[test]
    fn hdbscan_separates_blobs_and_marks_outliers_as_noise() {
        let mut pts: Vec<[f32; 2]> =
            blobs(&[vec![0.0, 0.0], vec![10.0, 0.0], vec![0.0, 10.0]], 40, 1.0)
                .into_iter()
                .map(|v| [v[0], v[1]])
                .collect();
        pts.push([50.0, 50.0]);
        let labels = hdbscan(&pts, 10);
        for b in 0..3 {
            let l = labels[b * 40];
            assert!(l >= 0);
            assert!(labels[b * 40..(b + 1) * 40].iter().all(|&x| x == l), "blob {b}: {labels:?}");
        }
        let distinct: std::collections::HashSet<i32> = labels[..120].iter().copied().collect();
        assert_eq!(distinct.len(), 3);
        assert_eq!(labels[120], -1);
        assert_eq!(hdbscan(&pts[..5], 10), vec![-1; 5], "too few points");
    }

    #[test]
    fn layout_keeps_high_dimensional_clusters_apart() {
        let centers: Vec<Vec<f32>> =
            (0..3).map(|c| (0..64).map(|d| if d % 3 == c { 5.0 } else { 0.0 }).collect()).collect();
        let rows = blobs(&centers, 60, 1.0);
        let refs: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
        let progress = Progress::default();
        let out = layout(&refs, 15, 1, &progress).unwrap();
        assert_eq!(out.points.len(), 180);
        assert_eq!(progress.done.load(Ordering::Relaxed), progress.total.load(Ordering::Relaxed));
        // Each blob is one cluster, all different.
        let mut seen = std::collections::HashSet::new();
        for b in 0..3 {
            let l = out.labels[b * 60];
            let same = out.labels[b * 60..(b + 1) * 60].iter().filter(|&&x| x == l).count();
            assert!(l >= 0 && same >= 55, "blob {b}: {:?}", &out.labels[b * 60..(b + 1) * 60]);
            seen.insert(l);
        }
        assert_eq!(seen.len(), 3);
        assert_eq!(layout(&refs, 15, 1, &progress), Some(out), "deterministic");

        progress.cancel.store(true, Ordering::Relaxed);
        assert!(layout(&refs, 15, 1, &progress).is_none());
    }

    #[test]
    fn auto_split_hits_targets_with_whole_groups() {
        use fastlabel_core::Split;
        // One huge scene, a few medium ones and many singletons (noise).
        let mut sizes = vec![500, 40, 35, 30, 30, 25, 20, 20, 15, 10];
        sizes.extend(std::iter::repeat_n(1, 75));
        let total: usize = sizes.iter().sum(); // 800
        let p = AutoSplit::default();
        let out = auto_split(&sizes, &p);
        let count = |s: Split| {
            sizes.iter().zip(&out).filter(|(_, o)| **o == s).map(|(n, _)| n).sum::<usize>()
        };
        for s in [Split::Val, Split::Test] {
            let share = count(s) as f64 / total as f64;
            assert!((share - 0.1).abs() < 0.02, "{s:?}: {share}");
        }
        assert_eq!(out[0], Split::Train, "a group above 25% of the val target stays in train");
        assert_eq!(auto_split(&sizes, &p), out, "deterministic");
        assert_ne!(
            auto_split(&sizes, &AutoSplit { seed: 2, ..p }),
            out,
            "another seed, another split"
        );
        let none = AutoSplit { val: 0.0, test: 0.0, ..p };
        assert!(auto_split(&sizes, &none).iter().all(|s| *s == Split::Train));
    }

    #[test]
    fn polygon_hit_test() {
        let square = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        assert!(in_polygon([5.0, 5.0], &square));
        assert!(!in_polygon([15.0, 5.0], &square));
        assert!(!in_polygon([5.0, 5.0], &square[..2]));
        assert!(!in_polygon([5.0, 5.0], &[]));
    }
}
