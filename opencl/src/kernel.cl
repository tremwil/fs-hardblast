#define CAT(a, b) a ## b
#define XCAT(a,b) CAT(a,b)

#ifndef PAR_LEN
#define PAR_LEN 4 // number of chars the work is split over
#endif
#ifndef SEQ_LEN
#define SEQ_LEN 4 // number of chars to check sequentially in a work item
#endif
#ifndef HASH_T
#define HASH_T uint // hash integer type
#endif
#ifndef VEC_LEN
#define VEC_LEN 8 // SIMD vector size
#endif

#define VEC(a) XCAT(a, VEC_LEN)
#define SEARCH_DEPTH (SEQ_LEN - 1)
#define FNV_PRIME 37

typedef HASH_T hash_t;
typedef VEC(HASH_T) hashvec_t;

constant uchar ALPHABET[] = ".0123456789_abcdefghijklmnopqrstuvwxyz";
#define ALPHABET_SIZE (sizeof(ALPHABET) - 1)

bool in_alphabet_prefilter(hashvec_t solutions) {
    return any(solutions <= 'z');
}

bool in_alphabet(hash_t solution) {
    if (solution > 'z') return false;
    if (solution < '.') return false;
    if (solution >= 'a') return true;
    if (solution > '_') return false;
    if (solution == '_') return true;
    if (solution > '9') return false;
    return solution != '/'; // only char between . and 0
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