//! Native TurboVec quantization, with parallel preparation of loaded row blocks.
//! Each chunk starts on a SIMD block boundary; concatenation is byte-identical
//! to the dependency's serial packer. All scoring still uses its public kernel.
use std::{io, path::Path};
use rayon::prelude::*;
use turbovec::{TurboQuantIndex, SearchResults};

pub(crate) enum SearchIndex {
    Native(TurboQuantIndex),
    Loaded(LoadedIndex),
}

pub(crate) struct LoadedIndex {
    bits: usize,
    dim: usize,
    n: usize,
    packed: Vec<u8>,
    scales: Vec<f32>,
    shift: Vec<f32>,
    scale: Vec<f32>,
    rotation: Vec<f32>,
    centroids: Vec<f32>,
    blocked: Vec<u8>,
    n_blocks: usize,
}

fn parallel_repack(packed: &[u8], n: usize, bits: usize, dim: usize) -> (Vec<u8>, usize) {
    // BLOCK is 32 in TurboVec 0.9. Keep complete blocks together, including
    // exactly one zero-padded tail, and collect in indexed iterator order.
    const BLOCK: usize = 32;
    const CHUNK_ROWS: usize = BLOCK * 128;
    let row_bytes = bits * (dim / 8);
    let chunks: Vec<Vec<u8>> = packed.par_chunks(CHUNK_ROWS * row_bytes)
        .map(|chunk| turbovec::pack::repack(chunk, chunk.len() / row_bytes, bits, dim).0)
        .collect();
    (chunks.concat(), n.div_ceil(BLOCK))
}

impl SearchIndex {
    /// Caller runs this in the existing bounded preparation pool.
    pub(crate) fn load(path: &Path) -> io::Result<Self> {
        let (bits, dim, n, packed, scales, shift, scale) = turbovec::io::load(path)?;
        if dim != crate::ops::EMBED_DIM || bits != 4 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "unexpected Turbo dimensions/bits"));
        }
        let rotation = turbovec::rotation::make_rotation_matrix(dim);
        let centroids = turbovec::codebook::codebook(bits, dim).1;
        let (blocked, n_blocks) = parallel_repack(&packed, n, bits, dim);
        Ok(Self::Loaded(LoadedIndex { bits, dim, n, packed, scales, shift, scale,
            rotation, centroids, blocked, n_blocks }))
    }

    pub(crate) fn dim(&self) -> usize { match self { Self::Native(i) => i.dim(), Self::Loaded(i) => i.dim } }
    pub(crate) fn bit_width(&self) -> usize { match self { Self::Native(i) => i.bit_width(), Self::Loaded(i) => i.bits } }
    pub(crate) fn len(&self) -> usize { match self { Self::Native(i) => i.len(), Self::Loaded(i) => i.n } }
    pub(crate) fn prepare(&self) { if let Self::Native(i) = self { i.prepare(); } }
    pub(crate) fn write(&self, path: &Path) -> io::Result<()> {
        match self {
            Self::Native(i) => i.write(path),
            Self::Loaded(i) => turbovec::io::write(path, i.bits, i.dim, i.n, &i.packed,
                &i.scales, &i.shift, &i.scale),
        }
    }
    pub(crate) fn search(&self, queries: &[f32], k: usize) -> SearchResults {
        match self {
            Self::Native(i) => i.search(queries, k),
            Self::Loaded(i) => {
                let nq = queries.len() / i.dim;
                assert_eq!(queries.len(), nq * i.dim);
                assert!(turbovec::first_invalid_coord(queries, i.dim).is_none());
                let (scores, indices) = turbovec::search::search(queries, nq, &i.rotation,
                    &i.blocked, &i.centroids, &i.scales, &i.shift, &i.scale,
                    i.bits, i.dim, i.n, i.n_blocks, k, None);
                SearchResults { scores, indices, nq, k: k.min(i.n) }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parallel_block_preparation_matches_native_including_partial_tail() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for bits in [2, 3, 4] {
            for n in [0, 1, 31, 32, 33, 4096, 4097, 8201] {
                let dim = 24;
                let packed: Vec<u8> = (0..n * bits * (dim / 8)).map(|j| (j * 137 + j / 3) as u8).collect();
                assert_eq!(pool.install(|| parallel_repack(&packed, n, bits, dim)),
                    turbovec::pack::repack(&packed, n, bits, dim));
            }
        }
    }
}
