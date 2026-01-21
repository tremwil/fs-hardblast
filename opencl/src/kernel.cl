#ifndef PAR_LEN
#define PAR_LEN 4 // number of chars the work is split over
#endif
#ifndef SEQ_LEN
#define SEQ_LEN 4 // number of chars to check sequentially in a work item
#endif
#ifndef HASH_T
#define HASH_T uint // hash integer type
#endif
#ifndef FNV_PRIME
#define FNV_PRIME 37
#endif
#ifndef VEC_LEN
#define VEC_LEN 8 // SIMD vector size
#endif
#ifndef ALPHABET_LIT
#define ALPHABET_LIT ".0123456789_abcdefghijklmnopqrstuvwxyz"
#endif

#define CAT(a, b) a ## b
#define XCAT(a,b) CAT(a,b)
#define VEC(a) XCAT(a, VEC_LEN)
#define SEARCH_DEPTH (SEQ_LEN - 1)

typedef HASH_T hash_t;
typedef VEC(HASH_T) hashvec_t;

constant uchar ALPHABET[] = { ALPHABET_LIT, 0 };
#define ALPHABET_SIZE (sizeof(ALPHABET) - 1)

bool in_alphabet_prefilter(hashvec_t solutions) {
    // Compiler will unroll this loop and optimize into constant comparisons
    uchar max = 0;
    #pragma unroll
    for (int i = 0; i < ALPHABET_SIZE; i++) {
        if (ALPHABET[i] > max) {
            max = ALPHABET[i];
        }
    }
    return any(solutions < max + 1);
}

bool in_alphabet(hash_t solution) {
    // Compiler will unroll this loop and optimize into bitmasks
    ulong m0 = 0, m1 = 0, m2 = 0, m3 = 0;
    uchar max = 0;
    #pragma unroll
    for (int i = 0; i < ALPHABET_SIZE; i++) {
        uchar c = ALPHABET[i];
        if (c > max)      max = c;
        if (c < 64)       m0 |= (1UL << c);
        else if (c < 128) m1 |= (1UL << (c - 64));
        else if (c < 192) m2 |= (1UL << (c - 128));
        else              m3 |= (1UL << (c - 192));
    }

    if (solution > max) return false;

    ulong mask;
    if (solution < 64)       mask = m0;
    else if (solution < 128) mask = m1;
    else if (solution < 192) mask = m2;
    else                     mask = m3;

    return (mask >> solution) & 1;
}

typedef struct {
    uchar bytes[PAR_LEN];
} ItemBase;

typedef struct {
    ItemBase base;
    uchar bytes[SEQ_LEN];
} Match;

kernel void find_collisions(
    const ulong work_items,
    const hash_t prefix_hash,
    const hash_t suffix_shift,
    global Match* out_buffer,
    const uint out_buffer_size,
    volatile global int* out_buffer_written
) {
    // get global item index (encodes item-specific static prefix)
    const ulong item_index = VEC_LEN * (ulong)get_global_id(0);
    if (item_index >= work_items) {
        return;
    }

    // compute item-specific base hash and bytes for work item
    ulong encoded[VEC_LEN];
    hash_t nonvec_base_hashes[VEC_LEN] = {0};
    for (int i = 0; i < VEC_LEN; i++) {
        nonvec_base_hashes[i] = prefix_hash * FNV_PRIME;
        encoded[i] = item_index + i;
    }

    ItemBase item_base[VEC_LEN];
    for (int i = 0; i < PAR_LEN; i++) {
        for (int j = 0; j < VEC_LEN; j++) {
            uchar chr = ALPHABET[encoded[j] % ALPHABET_SIZE];
            item_base[j].bytes[i] = chr;
            nonvec_base_hashes[j] = (nonvec_base_hashes[j] + chr) * FNV_PRIME;
            encoded[j] /= ALPHABET_SIZE;
        }
    }

    // load item base hash into a vector
    hashvec_t item_base_hash = VEC(vload)(0, nonvec_base_hashes);

    // DFS state variables
    hashvec_t base_hashes[SEARCH_DEPTH] = { [0] = item_base_hash };
    char char_indices[SEARCH_DEPTH];
    char depth = 0;

    // Init char indices
    for (int i = 0; i < SEARCH_DEPTH; i++) {
        char_indices[i] = -1;
    }

    // DFS loop
    while (depth >= 0) {
        const char i = ++char_indices[depth];
        if (i >= ALPHABET_SIZE) {
            char_indices[depth--] = -1;
            continue;
        }

        const hashvec_t base_hash = (base_hashes[depth] + (hash_t)ALPHABET[i]) * FNV_PRIME;
        const hashvec_t solution = suffix_shift - base_hash;
        if (in_alphabet_prefilter(solution)) {
            hash_t solution_nonvvec[VEC_LEN];
            VEC(vstore)(solution, 0, solution_nonvvec);

            for (int k = 0; k < VEC_LEN; k++) {
                if (!in_alphabet(solution_nonvvec[k])) {
                    continue;
                }
                const uint slot = atomic_add(out_buffer_written, 1);
                if (slot < out_buffer_size) {
                    global Match* m = out_buffer + slot;
                    // write base (par) bytes
                    m->base = item_base[k];
                    // write seq bytes
                    for (int j = 0; j <= depth; j++) {
                        m->bytes[j] = ALPHABET[char_indices[j]];
                    }
                    m->bytes[depth+1] = solution_nonvvec[k];
                    // nul-terminate
                    if (depth + 2 < SEQ_LEN) {
                        m->bytes[depth+2] = 0;
                    }
                }
            }
        }

        if (depth < SEARCH_DEPTH - 1) {
            base_hashes[++depth] = base_hash;
        }
    }
}
