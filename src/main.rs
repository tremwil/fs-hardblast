#![feature(portable_simd)]
#![feature(likely_unlikely)]

use std::{
    hint::unlikely,
    simd::{LaneCount, Mask, Simd, SupportedLaneCount, cmp::SimdPartialEq},
    time::Instant,
};

mod alphabet;
mod const_vec;

use alphabet::Alphabet;

const PREFIX: &[u8] = b"/other/";
const SUFFIX: &[u8] = b".dcx";

const ALPHABET: Alphabet<38> = Alphabet::new(b"_.abcdefghijklmnopqrstuvwxyz0123456789");

const START: &[u8] = b"mnopqrs";
const TARGET: u32 = 0xd7255946;
const SEARCH: usize = 7;

/// Note that this isn't the real FNV prime, but what FromSoft uses.
const FNV_PRIME: u32 = 37;

/// Precomputed information about the hash of a suffix.
///
/// Used to efficiently compute the combined hash of `base|suffix` given `hash(base)`
/// as well as efficiently finding a single character `x` such that
/// `hash(base|x|suffix) == target_hash`.
#[derive(Debug, Clone, Copy)]
#[allow(unused)]
struct PrecomputedSuffix {
    hash: u32,
    mult: u32,
    target_shift: u32,
}

impl PrecomputedSuffix {
    pub const fn new(suffix: &[u8], target_hash: u32) -> Self {
        // 32-bit modular inverse using 3 Newton-Raphson iterations :)
        // From https://arxiv.org/abs/2204.04342
        const fn minv32(a: u32) -> u32 {
            assert!(!a.is_multiple_of(2));

            let mut x = 3u32.wrapping_mul(a) ^ 2;
            let mut y = 1u32.wrapping_sub(a.wrapping_mul(x));

            x = x.wrapping_mul(y.wrapping_add(1));
            y = y.wrapping_mul(y);
            x = x.wrapping_mul(y.wrapping_add(1));
            y = y.wrapping_mul(y);
            x.wrapping_mul(y.wrapping_add(1))
        }

        let hash = fnv_hash(suffix);
        let mult = FNV_PRIME.wrapping_pow(suffix.len() as u32);
        let target_shift = target_hash.wrapping_sub(hash).wrapping_mul(minv32(mult));

        Self {
            hash,
            mult,
            target_shift,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Match {
    bytes_be: u64,
    len: usize,
}

impl Match {
    pub fn bytes(&self) -> [u8; 8] {
        self.bytes_be
            .rotate_right(8 * self.len as u32)
            .to_be_bytes()
    }
}

fn main() {
    let now = Instant::now();

    let mut prefix = PREFIX.to_owned();
    prefix.push(0);

    for &start_char in START {
        *prefix.last_mut().unwrap() = start_char;

        for m in find_collisions_simd::<4>(&prefix, SUFFIX, SEARCH, TARGET) {
            let match_bytes = &m.bytes()[..m.len];

            let mut collision = prefix.clone();
            collision.extend_from_slice(match_bytes);
            collision.extend_from_slice(SUFFIX);

            println!("{}", String::from_utf8_lossy(&collision));

            // for validation purposes
            assert_eq!(fnv_hash(&collision), TARGET)
        }
    }

    println!("{:?}", now.elapsed());
}

const fn fnv_hash(data: &[u8]) -> u32 {
    let mut hash: u32 = 0;
    let mut i = 0;
    while i < data.len() {
        hash = hash.wrapping_mul(FNV_PRIME).wrapping_add(data[i] as u32);
        i += 1;
    }
    hash
}

/// Find bytes strings `m` of length at most `max_len` such that
///
/// ```text
/// fnv_hash(prefix|m|suffix) == target_hash
/// ```
///
/// The maximum value of `max_len` is 8.
///
/// The search is optimized by using iterative DFS to avoid recomputing
/// hashes, mathematically solving for the possible value of the last
/// character and parallelizing the above over second-to-last characters
/// using `L`-lane SIMD.
fn find_collisions_simd<const L: usize>(
    prefix: &[u8],
    suffix: &[u8],
    max_len: usize,
    target_hash: u32,
) -> Vec<Match>
where
    LaneCount<L>: SupportedLaneCount,
    Simd<u32, L>: SimdPartialEq<Mask = Mask<i32, L>>,
{
    let suffix = PrecomputedSuffix::new(suffix, target_hash);
    let prefix_hash = fnv_hash(prefix);
    let mut matches = Vec::with_capacity(8);

    // check the empty string (matches if prefix|suffix matches)
    if prefix_hash == target_hash {
        matches.push(Match {
            bytes_be: 0,
            len: 0,
        })
    }

    // check one-character strings by directly solving for the possible value
    let prefix_hash_base = prefix_hash.wrapping_mul(FNV_PRIME);
    let one_length_collision = suffix.target_shift.wrapping_sub(prefix_hash_base);
    if ALPHABET.contains(one_length_collision) {
        matches.push(Match {
            bytes_be: one_length_collision as u64,
            len: 1,
        })
    }

    // having 2 vecs means that we can copy the next_hash_base vectors straight into
    // the DFS stack
    let init_cap = max_len * ALPHABET.bytes().len();
    let mut hash_base_stack = Vec::with_capacity(init_cap);
    let mut match_stack = Vec::with_capacity(init_cap);

    hash_base_stack.push(prefix_hash_base);
    match_stack.push(Match {
        bytes_be: 0,
        len: 2,
    });

    let target_shift_splat = Simd::splat(suffix.target_shift);

    while let (Some(hash_base), Some(seq)) = (hash_base_stack.pop(), match_stack.pop()) {
        let hash_base_splat = Simd::splat(hash_base);

        // use simd to process second-to-last characters in parallel
        //
        // because these chunks are known at compile-time the loops below can be unrolled
        // and bounds checks can be removed
        let (alphabet_chunks, alphabet_remainder) = const { ALPHABET.simd_chunks::<L>() };

        for chunk in alphabet_chunks.as_slice() {
            let next_hash_base = (hash_base_splat + chunk) * Simd::splat(FNV_PRIME);
            let chunk_arr = chunk.as_array();

            // add len+1 strings to the DFS stack
            if seq.len != max_len {
                hash_base_stack.extend_from_slice(next_hash_base.as_array());
                match_stack.extend(chunk_arr.iter().map(|&c| Match {
                    bytes_be: (seq.bytes_be << 8) | (c as u64),
                    len: seq.len + 1,
                }));
            }
            // solve for the only last character that could collide and report matches
            let solutions = target_shift_splat - next_hash_base;
            if unlikely(ALPHABET.simd_prefilter(solutions)) {
                matches.extend(
                    solutions
                        .as_array()
                        .iter()
                        .zip(chunk_arr)
                        .filter(|(s, _)| ALPHABET.contains(**s))
                        .map(|(&s, &c)| Match {
                            bytes_be: (seq.bytes_be << 16 | (c as u64) << 8 | s as u64),
                            len: seq.len,
                        }),
                )
            }
        }
        for &c in alphabet_remainder.as_slice() {
            let next_hash_base = (hash_base + c).wrapping_mul(FNV_PRIME);

            // add len+1 strings to the DFS stack
            if seq.len != max_len {
                hash_base_stack.push(next_hash_base);
                match_stack.push(Match {
                    bytes_be: (seq.bytes_be << 8) | (c as u64),
                    len: seq.len + 1,
                });
            }
            // solve for the only last character that could collide and report matches
            let s = suffix.target_shift - next_hash_base;
            if unlikely(ALPHABET.contains(s)) {
                matches.push(Match {
                    bytes_be: (seq.bytes_be << 16 | (c as u64) << 8 | s as u64),
                    len: seq.len,
                })
            }
        }
    }

    matches
}
