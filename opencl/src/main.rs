use std::{cmp::Reverse, ffi::c_void, process::exit, ptr, time::Instant};

use cl3::{
    ext::{
        CL_BLOCKING, CL_DEVICE_AVAILABLE, CL_DEVICE_MAX_CLOCK_FREQUENCY,
        CL_DEVICE_MAX_COMPUTE_UNITS, CL_DEVICE_VERSION, CL_MEM_READ_WRITE, CL_MEM_USE_HOST_PTR,
        CL_MEM_WRITE_ONLY,
    },
    info_type::InfoType,
};
use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    device::{CL_DEVICE_NAME, CL_DEVICE_TYPE_GPU, Device, get_all_devices, get_device_info},
    error_codes::ClError,
    kernel::{ExecuteKernel, Kernel},
    memory::Buffer,
    program::Program,
};

type Hash = u32;

const FNV_PRIME: Hash = 37;
const ALPHABET: &[u8] = b".0123456789_abcdefghijklmnopqrstuvwxyz";

const PREFIX: &[u8] = b"/other/";
const SUFFIX: &[u8] = b".dcx";
const TARGET: Hash = 0xd7255946;

const PAR_LEN: usize = 4; // Assign a gpu thread to each prefix of this length
const SEQ_LEN: usize = 5; // Search for collisions of this many extra chars

const BLOCK_SIZE: usize = 512; // tune this for your GPU
const TOTAL_LEN: usize = PAR_LEN + SEQ_LEN;

fn main() -> Result<(), Err> {
    let suffix = PrecomputedSuffix::new(SUFFIX, TARGET);
    let prefix_hash = fnv_hash(PREFIX);

    let devices = get_all_devices(CL_DEVICE_TYPE_GPU)?;
    let mut usable: Vec<_> = devices
        .into_iter()
        .filter(|&dev| {
            match get_device_info(dev, CL_DEVICE_AVAILABLE) {
                Ok(InfoType::Uint(1..)) => (),
                _ => return false,
            }
            if let Ok(InfoType::VecUchar(ver)) = get_device_info(dev, CL_DEVICE_VERSION) {
                // for global int32 atomics support
                return ver.as_slice() >= b"1.1";
            }
            false
        })
        .filter_map(|dev| {
            let max_clock = get_device_info(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY).ok()?;
            let max_cu = get_device_info(dev, CL_DEVICE_MAX_COMPUTE_UNITS).ok()?;
            match (max_clock, max_cu) {
                (InfoType::Uint(clock), InfoType::Uint(cu)) => Some((dev, clock * cu)),
                _ => None,
            }
        })
        .collect();

    usable.sort_by_key(|&(_, compute)| Reverse(compute));

    if usable.is_empty() {
        println!("no usable OpenCL GPU device found.");
        exit(1);
    }

    println!("usable devices (OpenCL support >= 1.1):");
    for (i, &(dev, compute)) in usable.iter().enumerate() {
        let name = get_device_info(dev, CL_DEVICE_NAME)?;
        println!("{i}: {name}, effective compute {compute} MHz");
    }

    println!("\nusing device 0.");

    let device = Device::new(usable[0].0);
    let context = Context::from_device(&device)?;
    let queue = CommandQueue::create_default(&context, 0)?;

    let hash_type = if size_of::<Hash>() == 4 {
        "uint"
    } else {
        "ulong"
    };
    let program = Program::create_and_build_from_source(
        &context,
        include_str!("kernel.cl"),
        &format!("-D PAR_LEN={PAR_LEN} -D SEQ_LEN={SEQ_LEN} -D HASH_T={hash_type} -Werror"),
    )
    .expect("kernel failed to build");

    let kernel = Kernel::create(&program, "find_collisions")?;

    let work_items = ALPHABET.len().pow(PAR_LEN as u32);
    let work_size = work_items.next_multiple_of(BLOCK_SIZE);

    let expected_collisions =
        (ALPHABET.len() as f64).powi(TOTAL_LEN as i32) / 256f64.powi(size_of::<Hash>() as i32);
    let buf_len = (1.5 * expected_collisions) as usize + 100; // safety margin
    let buf_len_bytes = buf_len * TOTAL_LEN;
    if buf_len_bytes > u32::MAX as usize {
        panic!("results buffer too big")
    }

    println!("using {buf_len} element results buffer\n");

    let results_dev = unsafe {
        Buffer::<u8>::create(&context, CL_MEM_WRITE_ONLY, buf_len_bytes, ptr::null_mut())?
    };
    let results_count_dev = unsafe {
        static ZERO: &u32 = &0;
        Buffer::<u32>::create(
            &context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
            1,
            ZERO as *const u32 as *mut c_void,
        )?
    };

    let pre_kernel = Instant::now();

    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&(work_items as u32))
            .set_arg(&prefix_hash)
            .set_arg(&suffix.target_shift)
            .set_arg(&results_dev)
            .set_arg(&(buf_len as u32))
            .set_arg(&results_count_dev)
            .set_global_work_size(work_size)
            .set_local_work_size(BLOCK_SIZE)
            .enqueue_nd_range(&queue)?
    };

    // wait for kernel completion and read result count
    let mut results_count = 0;
    unsafe {
        queue.enqueue_read_buffer(
            &results_count_dev,
            CL_BLOCKING,
            0,
            std::slice::from_mut(&mut results_count),
            &[kernel_event.get()],
        )?
    };
    results_count = results_count.min(buf_len as u32);
    let kernel_time = pre_kernel.elapsed();

    // copy initialized portion of results buffer
    let mut results = vec![0; results_count as usize * TOTAL_LEN];
    unsafe {
        queue.enqueue_read_buffer(&results_dev, CL_BLOCKING, 0, results.as_mut_slice(), &[])?
    };

    // print matches
    for res in results.chunks_exact(TOTAL_LEN) {
        let len = res.iter().position(|&b| b == 0).unwrap_or(res.len());
        println!(
            "{}{}{}",
            String::from_utf8_lossy(PREFIX),
            String::from_utf8_lossy(&res[..len]),
            String::from_utf8_lossy(SUFFIX)
        );
    }

    println!("\nfound {} solutions in {:?}", results_count, kernel_time);

    Ok(())
}

const fn fnv_hash(bytes: &[u8]) -> Hash {
    let mut hash: Hash = 0;
    let mut i = 0;
    while i < bytes.len() {
        hash = hash.wrapping_mul(FNV_PRIME).wrapping_add(bytes[i] as Hash);
        i += 1;
    }
    hash
}

/// Precomputed information about the hash of a suffix.
///
/// Used to efficiently compute the combined hash of `base|suffix` given `hash(base)`
/// as well as efficiently finding a single character `x` such that
/// `hash(base|x|suffix) == target_hash`.
#[derive(Debug, Clone, Copy)]
#[allow(unused)]
struct PrecomputedSuffix {
    hash: Hash,
    mult: Hash,
    target_shift: Hash,
}

impl PrecomputedSuffix {
    pub const fn new(suffix: &[u8], target_hash: Hash) -> Self {
        // 64-bit modular inverse using 4 Newton-Raphson iterations
        // From https://arxiv.org/abs/2204.04342
        const fn minv32(a: Hash) -> Hash {
            assert!(!a.is_multiple_of(2));

            let mut x = 3u32.wrapping_mul(a) ^ 2;
            let mut y = 1u32.wrapping_sub(a.wrapping_mul(x));

            x = x.wrapping_mul(y.wrapping_add(1));
            y = y.wrapping_mul(y);
            x = x.wrapping_mul(y.wrapping_add(1));
            y = y.wrapping_mul(y);
            x = x.wrapping_mul(y.wrapping_add(1));
            y = y.wrapping_mul(y);

            x.wrapping_mul(y.wrapping_add(1))
        }

        let hash = fnv_hash(suffix);
        let mult = FNV_PRIME.wrapping_pow(suffix.len() as Hash);
        let target_shift = target_hash.wrapping_sub(hash).wrapping_mul(minv32(mult));

        Self {
            hash,
            mult,
            target_shift,
        }
    }
}

#[derive(Debug)]
pub struct Err(#[allow(unused)] ClError);

impl std::fmt::Display for Err {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self, f)
    }
}

impl From<ClError> for Err {
    fn from(value: ClError) -> Self {
        Self(value)
    }
}

impl From<i32> for Err {
    fn from(value: i32) -> Self {
        Self(ClError(value))
    }
}
