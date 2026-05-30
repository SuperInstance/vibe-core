//! vibe-core: 16-dimensional vibe embeddings with blending, qualitative descriptions,
//! groove locking, and energy scaling.

use std::fmt;

/// A 16-dimensional vibe embedding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VibeEmbedding {
    pub dims: [f32; 16],
}

impl Default for VibeEmbedding {
    fn default() -> Self {
        Self::zero()
    }
}

impl VibeEmbedding {
    /// Create a zero embedding.
    pub const fn zero() -> Self {
        Self { dims: [0.0; 16] }
    }

    /// Create an embedding from an array.
    pub const fn new(dims: [f32; 16]) -> Self {
        Self { dims }
    }

    /// Create a random-ish embedding from a seed (deterministic).
    pub fn from_seed(seed: u64) -> Self {
        let mut dims = [0.0f32; 16];
        let mut s = seed;
        for dim in &mut dims {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *dim = ((s >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
        Self { dims }
    }

    /// Euclidean norm of the embedding.
    pub fn norm(&self) -> f32 {
        self.dims.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize to unit length.
    pub fn normalize(&mut self) {
        let n = self.norm();
        if n > 1e-8 {
            for x in &mut self.dims {
                *x /= n;
            }
        }
    }

    /// Cosine similarity between two embeddings.
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot = self.dot(other);
        let denom = self.norm() * other.norm();
        if denom > 1e-8 {
            dot / denom
        } else {
            0.0
        }
    }

    /// Dot product.
    pub fn dot(&self, other: &Self) -> f32 {
        self.dims.iter().zip(other.dims.iter()).map(|(a, b)| a * b).sum()
    }

    /// Euclidean distance.
    pub fn distance(&self, other: &Self) -> f32 {
        self.dims
            .iter()
            .zip(other.dims.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Blend two embeddings with a weight `t` (0.0 = self, 1.0 = other).
    pub fn blend(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let mut out = [0.0f32; 16];
        for (o, (a, b)) in out.iter_mut().zip(self.dims.iter().zip(other.dims.iter())) {
            *o = *a * (1.0 - t) + *b * t;
        }
        Self { dims: out }
    }

    /// Multi-way blend across many embeddings with weights.
    pub fn blend_weighted(embeddings: &[Self], weights: &[f32]) -> Option<Self> {
        if embeddings.is_empty() || embeddings.len() != weights.len() {
            return None;
        }
        let sum: f32 = weights.iter().sum();
        if sum.abs() < 1e-8 {
            return None;
        }
        let mut out = [0.0f32; 16];
        for (emb, w) in embeddings.iter().zip(weights.iter()) {
            for (o, d) in out.iter_mut().zip(emb.dims.iter()) {
                *o += *d * w;
            }
        }
        for x in &mut out {
            *x /= sum;
        }
        Some(Self { dims: out })
    }

    /// Scale the "energy" (magnitude) of the embedding.
    pub fn energy_scale(&mut self, target_energy: f32) {
        let n = self.norm();
        if n > 1e-8 {
            let scale = target_energy / n;
            for x in &mut self.dims {
                *x *= scale;
            }
        }
    }

    /// Return the current energy (norm).
    pub fn energy(&self) -> f32 {
        self.norm()
    }

    /// Lock the embedding to the nearest groove template.
    pub fn groove_lock(&mut self, grooves: &[Self]) -> usize {
        assert!(!grooves.is_empty(), "groove list must not be empty");
        let mut best = 0usize;
        let mut best_dist = f32::MAX;
        for (i, g) in grooves.iter().enumerate() {
            let d = self.distance(g);
            if d < best_dist {
                best_dist = d;
                best = i;
            }
        }
        self.dims = grooves[best].dims;
        best
    }

    /// Return a qualitative description based on dominant dimensions.
    pub fn qualitative_description(&self) -> QualitativeVibe {
        // Sum positive and negative energies across halves.
        let first_half: f32 = self.dims[..8].iter().sum();
        let second_half: f32 = self.dims[8..].iter().sum();
        let pos_count = self.dims.iter().filter(|&&x| x > 0.0).count();
        let max_idx = self
            .dims
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut intensity = "subtle";
        if self.energy() > 2.0 {
            intensity = "intense";
        } else if self.energy() > 1.0 {
            intensity = "moderate";
        }

        let mut warmth = "cool";
        if first_half > second_half {
            warmth = "warm";
        } else if first_half < second_half {
            warmth = "cold";
        }

        let direction = match max_idx {
            0..=3 => "grounded",
            4..=7 => "expansive",
            8..=11 => "focused",
            _ => "ethereal",
        };

        let polarity = if pos_count > 8 {
            "positive"
        } else if pos_count < 8 {
            "negative"
        } else {
            "balanced"
        };

        QualitativeVibe {
            intensity: intensity.to_string(),
            warmth: warmth.to_string(),
            direction: direction.to_string(),
            polarity: polarity.to_string(),
            dominant_dimension: max_idx,
        }
    }
}

impl fmt::Display for VibeEmbedding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vibe[{:?}]", &self.dims[..])
    }
}

/// Qualitative description of a vibe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualitativeVibe {
    pub intensity: String,
    pub warmth: String,
    pub direction: String,
    pub polarity: String,
    pub dominant_dimension: usize,
}

/// Groove template bank for locking.
#[derive(Debug, Clone, Default)]
pub struct GrooveBank {
    pub grooves: Vec<VibeEmbedding>,
}

impl GrooveBank {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_grooves(grooves: Vec<VibeEmbedding>) -> Self {
        Self { grooves }
    }

    pub fn add(&mut self, groove: VibeEmbedding) {
        self.grooves.push(groove);
    }

    pub fn len(&self) -> usize {
        self.grooves.len()
    }

    pub fn is_empty(&self) -> bool {
        self.grooves.is_empty()
    }

    pub fn lock_embedding(&self, emb: &mut VibeEmbedding) -> usize {
        emb.groove_lock(&self.grooves)
    }
}

/// Energy scaler that maps embeddings to a target energy range.
#[derive(Debug, Clone, Copy)]
pub struct EnergyScaler {
    pub target: f32,
}

impl EnergyScaler {
    pub fn new(target: f32) -> Self {
        Self { target }
    }

    pub fn scale(&self, emb: &mut VibeEmbedding) {
        emb.energy_scale(self.target);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_embedding() {
        let v = VibeEmbedding::zero();
        assert_eq!(v.dims, [0.0; 16]);
        assert_eq!(v.energy(), 0.0);
    }

    #[test]
    fn test_new() {
        let arr: [f32; 16] = (0..16).map(|i| i as f32).collect::<Vec<_>>().try_into().unwrap();
        let v = VibeEmbedding::new(arr);
        assert_eq!(v.dims[0], 0.0);
        assert_eq!(v.dims[15], 15.0);
    }

    #[test]
    fn test_from_seed_deterministic() {
        let a = VibeEmbedding::from_seed(42);
        let b = VibeEmbedding::from_seed(42);
        assert_eq!(a.dims, b.dims);
    }

    #[test]
    fn test_norm() {
        let mut v = VibeEmbedding::zero();
        v.dims[0] = 3.0;
        v.dims[1] = 4.0;
        assert!((v.norm() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize() {
        let mut v = VibeEmbedding::from_seed(7);
        v.normalize();
        assert!((v.norm() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot() {
        let mut a = VibeEmbedding::zero();
        a.dims[0] = 1.0;
        a.dims[1] = 2.0;
        let mut b = VibeEmbedding::zero();
        b.dims[0] = 3.0;
        b.dims[1] = 4.0;
        assert_eq!(a.dot(&b), 11.0);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = VibeEmbedding::from_seed(99);
        assert!((a.cosine_similarity(&a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let mut a = VibeEmbedding::zero();
        a.dims[0] = 1.0;
        let mut b = VibeEmbedding::zero();
        b.dims[1] = 1.0;
        assert!(a.cosine_similarity(&b).abs() < 1e-5);
    }

    #[test]
    fn test_distance() {
        let mut a = VibeEmbedding::zero();
        a.dims[0] = 1.0;
        let mut b = VibeEmbedding::zero();
        b.dims[0] = 4.0;
        assert!((a.distance(&b) - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_blend_t_zero() {
        let a = VibeEmbedding::from_seed(1);
        let b = VibeEmbedding::from_seed(2);
        let c = a.blend(&b, 0.0);
        assert_eq!(c.dims, a.dims);
    }

    #[test]
    fn test_blend_t_one() {
        let a = VibeEmbedding::from_seed(1);
        let b = VibeEmbedding::from_seed(2);
        let c = a.blend(&b, 1.0);
        assert_eq!(c.dims, b.dims);
    }

    #[test]
    fn test_blend_midpoint() {
        let mut a = VibeEmbedding::zero();
        a.dims[0] = 0.0;
        let mut b = VibeEmbedding::zero();
        b.dims[0] = 2.0;
        let c = a.blend(&b, 0.5);
        assert!((c.dims[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_blend_weighted() {
        let a = VibeEmbedding::from_seed(1);
        let b = VibeEmbedding::from_seed(2);
        let c = VibeEmbedding::blend_weighted(&[a, b], &[1.0, 1.0]).unwrap();
        for i in 0..16 {
            assert!((c.dims[i] - (a.dims[i] + b.dims[i]) / 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_blend_weighted_none_on_mismatch() {
        let a = VibeEmbedding::zero();
        assert!(VibeEmbedding::blend_weighted(&[a], &[1.0, 1.0]).is_none());
    }

    #[test]
    fn test_energy_scale() {
        let mut v = VibeEmbedding::from_seed(5);
        v.energy_scale(10.0);
        assert!((v.energy() - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_energy_scale_zero() {
        let mut v = VibeEmbedding::zero();
        v.energy_scale(5.0); // should not panic
        assert_eq!(v.energy(), 0.0);
    }

    #[test]
    fn test_groove_lock() {
        let g0 = VibeEmbedding::new([0.0; 16]);
        let mut g1 = VibeEmbedding::new([0.0; 16]);
        g1.dims[0] = 1.0;
        let bank = GrooveBank::with_grooves(vec![g0, g1]);
        let mut v = VibeEmbedding::new([0.0; 16]);
        v.dims[0] = 0.9;
        let idx = bank.lock_embedding(&mut v);
        assert_eq!(idx, 1);
        assert_eq!(v.dims, g1.dims);
    }

    #[test]
    fn test_qualitative_description() {
        let mut v = VibeEmbedding::zero();
        for i in 0..9 {
            v.dims[i] = (i as f32 + 1.0) * 0.5;
        }
        let q = v.qualitative_description();
        assert_eq!(q.dominant_dimension, 8);
        assert_eq!(q.direction, "focused");
        assert_eq!(q.warmth, "warm");
        assert_eq!(q.polarity, "positive");
    }

    #[test]
    fn test_qualitative_description_cool() {
        let mut v = VibeEmbedding::zero();
        for i in 8..16 {
            v.dims[i] = 1.0;
        }
        let q = v.qualitative_description();
        assert_eq!(q.warmth, "cold");
    }

    #[test]
    fn test_groove_bank_add() {
        let mut bank = GrooveBank::new();
        assert!(bank.is_empty());
        bank.add(VibeEmbedding::zero());
        assert_eq!(bank.len(), 1);
    }

    #[test]
    fn test_energy_scaler() {
        let scaler = EnergyScaler::new(7.0);
        let mut v = VibeEmbedding::from_seed(3);
        scaler.scale(&mut v);
        assert!((v.energy() - 7.0).abs() < 1e-4);
    }

    #[test]
    fn test_display() {
        let v = VibeEmbedding::zero();
        let s = format!("{}", v);
        assert!(s.starts_with("Vibe["));
    }
}
