#![feature(portable_simd)]
#![feature(likely_unlikely)]

use rayon::prelude::*;
use std::{
    hint::unlikely,
    simd::{cmp::SimdPartialEq, *},
    time::Instant,
};

mod alphabet;
mod const_vec;

// /other/m_q4ww2d.dcx

use alphabet::Alphabet;

const PREFIX: &[u8] = b"/other/";
const SUFFIX: &[u8] = b".dcx";

const ALPHABET: Alphabet<38> = Alphabet::new(b"_.abcdefghijklmnopqrstuvwxyz0123456789");
//const ALPHABET: &[u8] = b"._abcdefghijklmnopqrstuvwxyz0123456789";

const START: &[u8] = b"mnopqrs";
const TARGET: u32 = 0xd7255946;
const SEARCH: usize = 7;

const FNV_PRIME: u32 = 37;

#[derive(Debug, Clone, Copy)]
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

const fn fnv_hash(data: &[u8]) -> u32 {
    let mut hash: u32 = 0;
    let mut i = 0;
    while i < data.len() {
        hash = hash.wrapping_mul(FNV_PRIME).wrapping_add(data[i] as u32);
        i += 1;
    }
    hash
}

fn main() {
    let now = Instant::now();

    for m in find_collisions_simd::<4>(b"/other/m", SUFFIX, SEARCH, TARGET) {
        let slice = &m.bytes()[..m.len];
        println!(
            "/other/m{}{}",
            String::from_utf8_lossy(slice),
            String::from_utf8_lossy(SUFFIX)
        )
    }

    //blast_all();
    println!("{:?}", now.elapsed());
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
    let mut matches = Vec::with_capacity(8);

    let suffix = PrecomputedSuffix::new(suffix, target_hash);
    let prefix_hash = fnv_hash(prefix);

    println!("suffix: {suffix:x?}");

    // Manually check plain prefix (len 0)
    if prefix_hash == target_hash {
        matches.push(Match {
            bytes_be: 0,
            len: 0,
        })
    }

    // Manually check first character (len 1)
    let one_length_collision = suffix.target_shift.wrapping_sub(prefix_hash);
    if ALPHABET.contains(one_length_collision) {
        matches.push(Match {
            bytes_be: one_length_collision as u64,
            len: 1,
        })
    }

    // having 2 vecs means that we can copy the hash vectors straight into the queue
    let init_cap = max_len * ALPHABET.bytes().len();
    let mut hash_base_stack = Vec::with_capacity(init_cap);
    let mut match_stack = Vec::with_capacity(init_cap);

    hash_base_stack.push(prefix_hash.wrapping_mul(FNV_PRIME));
    match_stack.push(Match {
        bytes_be: 0,
        len: 2,
    });

    let target_shift_splat = Simd::splat(suffix.target_shift);

    while let (Some(hash_base), Some(seq)) = (hash_base_stack.pop(), match_stack.pop()) {
        let hash_base_splat = Simd::splat(hash_base);

        let (alphabet_chunks, alphabet_remainder) = const { ALPHABET.simd_groups::<L>() };

        for chunk in alphabet_chunks.as_slice() {
            let next_hash_base = (hash_base_splat + chunk) * Simd::splat(FNV_PRIME);
            let chunk_arr = chunk.as_array();

            if seq.len != max_len {
                hash_base_stack.extend_from_slice(next_hash_base.as_array());
                match_stack.extend(chunk_arr.iter().map(|&c| Match {
                    bytes_be: (seq.bytes_be << 8) | (c as u64),
                    len: seq.len + 1,
                }));
            }

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

            if seq.len != max_len {
                hash_base_stack.push(next_hash_base);
                match_stack.push(Match {
                    bytes_be: (seq.bytes_be << 8) | (c as u64),
                    len: seq.len + 1,
                });
            }

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

// fn blast_all() {
//     for &c in START {
//         let mut prefix = PREFIX.to_owned();
//         prefix.push(c);
//         blast(&prefix, SUFFIX);
//     }
// }

// fn blast(prefix: &[u8], suffix: &[u8]) {
//     let mut base_hash: u32 = 0;
//     for c in prefix {
//         base_hash = base_hash.wrapping_mul(37).wrapping_add(*c as u32);
//     }

//     let max = ALPHABET.len().pow(SEARCH as u32);
//     let par = (0..max)
//         .into_par_iter()
//         .filter(|i| test_hash(base_hash, suffix, *i));

//     let result: Vec<_> = par.collect();

//     for i in result {
//         print_path(prefix, suffix, i);
//     }
// }

// fn test_hash(base_hash: u32, suffix: &[u8], i: usize) -> bool {
//     let mut hash: u32 = base_hash;
//     let mut rem = i;
//     loop {
//         hash = hash
//             .wrapping_mul(37)
//             .wrapping_add(ALPHABET[rem % ALPHABET.len()] as u32);
//         rem /= ALPHABET.len();
//         if rem == 0 {
//             break;
//         }
//     }

//     for c in suffix {
//         hash = hash.wrapping_mul(37).wrapping_add(*c as u32);
//     }

//     hash == TARGET
// }

// fn print_path(prefix: &[u8], suffix: &[u8], i: usize) {
//     let mut output = prefix.to_owned();
//     let mut rem = i;
//     loop {
//         output.push(ALPHABET[rem % ALPHABET.len()]);
//         rem /= ALPHABET.len();
//         if rem == 0 {
//             break;
//         }
//     }
//     output.extend_from_slice(suffix);
//     println!("{}", String::from_utf8_lossy(&output));
// }
