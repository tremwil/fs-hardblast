use std::{
    ops::Range,
    simd::{
        LaneCount, Mask, Simd, SupportedLaneCount,
        cmp::{SimdPartialEq, SimdPartialOrd},
    },
};

use crate::const_vec::ConstVec;

/// Simple insertion sort
const fn sort_bytes<const N: usize>(mut bytes: [u8; N]) -> [u8; N] {
    let mut i = 1;
    while i < bytes.len() {
        let mut j = i;
        while j > 0 && bytes[j - 1] > bytes[j] {
            bytes.swap(j, j - 1);
            j -= 1;
        }
        i += 1;
    }

    bytes
}

/// Compile-time preprocessed alphabet.
///
/// Stores the sorted bytes as well as the the contiguous ranges making up this alphabet.
///
/// Because this is all computed at compile-time, the optimizer can generate highly optimized code
/// for [`Self::contains`], [`Self::simd_prefilter`] and unroll loops that iterate on
/// [`Self::simd_groups`].
#[derive(Debug, Clone)]
pub struct Alphabet<const N: usize> {
    bytes: [u8; N],
    ranges: ConstVec<Range<u32>, N>,
}

impl<const N: usize> Alphabet<N> {
    pub const fn new(bytes: &'static [u8; N]) -> Self {
        // avoid sorting if not necessary
        if bytes.len() > u8::MAX as usize + 1 {
            panic!("duplicate character in alphabet");
        }

        let sorted = sort_bytes(*bytes);

        let mut i = 1;
        while i < bytes.len() {
            if bytes[i] == bytes[i - 1] {
                panic!("duplicate character in alphabet");
            }
            i += 1;
        }

        Self {
            ranges: Self::compute_ranges(&sorted),
            bytes: sorted,
        }
    }

    const fn compute_ranges(sorted: &[u8; N]) -> ConstVec<Range<u32>, N> {
        const U8_SIZE: u32 = u8::MAX as u32 + 1;

        let mut ranges = ConstVec::new();

        if sorted.is_empty() {
            return ranges;
        }

        ranges.push(sorted[0] as u32..U8_SIZE);

        let mut i = 1;
        while i < sorted.len() {
            if sorted[i] as u32 != sorted[i - 1] as u32 + 1 {
                ranges.index_mut(ranges.len() - 1).end = sorted[i - 1] as u32 + 1;
                ranges.push(sorted[i] as u32..U8_SIZE)
            }
            i += 1;
        }

        ranges.index_mut(ranges.len() - 1).end = sorted[sorted.len() - 1] as u32 + 1;
        ranges
    }

    pub const fn bytes(&self) -> &[u8; N] {
        &self.bytes
    }

    #[inline(always)]
    pub const fn contains(&self, char: u32) -> bool {
        if self.ranges.is_empty() {
            return false;
        }

        let mut i = self.ranges.len() - 1;
        loop {
            if char >= self.ranges.index(i).end {
                return false;
            }
            if char >= self.ranges.index(i).start {
                return true;
            }
            if i == 0 {
                return false;
            }
            i -= 1;
        }
    }

    /// Quickly eliminate vectors for which none of the elements are in this alphabet.
    #[inline(always)]
    pub fn simd_prefilter<const L: usize>(&self, chars: Simd<u32, L>) -> bool
    where
        LaneCount<L>: SupportedLaneCount,
        Simd<u32, L>: SimdPartialEq<Mask = Mask<i32, L>>,
    {
        if self.ranges.is_empty() {
            return false;
        }

        let alphabet_end = self.ranges[self.ranges.len() - 1].end;
        chars.simd_lt(Simd::splat(alphabet_end)).any()
    }

    /// Split the characters of the alphabet into `floor(N/L)` SIMD vectors and a remainder
    /// array of less `N % L` elements.
    pub const fn simd_chunks<const L: usize>(&self) -> (ConstVec<Simd<u32, L>, N>, ConstVec<u32, L>)
    where
        LaneCount<L>: SupportedLaneCount,
    {
        let mut simd = ConstVec::new();

        let mut i = 0;
        loop {
            let mut chunk = [0u32; L];
            let mut j = 0;

            while j < L && i < self.bytes.len() {
                chunk[j] = self.bytes[i] as u32;
                j += 1;
                i += 1;
            }

            if j < L {
                return (simd, ConstVec::from_slice_range(&chunk, 0..j));
            } else {
                simd.push(Simd::from_array(chunk));
            }
        }
    }
}
