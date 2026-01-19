#ifndef PAR_LEN
#define PAR_LEN 4 // number of chars the work is split over
#endif
#ifndef SEQ_LEN
#define SEQ_LEN 4 // number of chars to check sequentially in a work item
#endif
#ifndef HASH_T
#define HASH_T uint // hash integer type
#endif

#define SEARCH_DEPTH (SEQ_LEN - 1)
#define FNV_PRIME 37

typedef HASH_T hash_t;

constant uchar ALPHABET[] = ".0123456789_abcdefghijklmnopqrstuvwxyz";
#define ALPHABET_SIZE (sizeof(ALPHABET) - 1)

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
    const uint work_items,
    const hash_t prefix_hash,
    const hash_t suffix_shift,
    global Match* out_buffer,
    const uint out_buffer_size,
    volatile global int* out_buffer_written
) {
    // get global item index (encodes item-specific static prefix)
    const size_t item_index = get_global_id(0);
    if (item_index >= work_items) {
        return;
    }

    // compute item-specific base hash and bytes for work item
    hash_t item_base_hash = prefix_hash * FNV_PRIME;
    uint encoded = item_index;
    ItemBase item_base;
    
    for (int i = 0; i < PAR_LEN; i++) {
        uchar chr = ALPHABET[encoded % ALPHABET_SIZE];
        item_base.bytes[i] = chr;
        item_base_hash = (item_base_hash + chr) * FNV_PRIME;
        encoded /= ALPHABET_SIZE;
    }

    // DFS state variables
    hash_t base_hashes[SEARCH_DEPTH] = { [0] = item_base_hash };
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

        const hash_t base_hash = (base_hashes[depth] + (hash_t)ALPHABET[i]) * FNV_PRIME;
        const hash_t solution = suffix_shift - base_hash;
        if (in_alphabet(solution)) {
            const uint slot = atomic_add(out_buffer_written, 1);
            if (slot < out_buffer_size) {
                global Match* m = out_buffer + slot;
                // write base (par) bytes
                m->base = item_base;
                // write seq bytes
                for (int j = 0; j <= depth; j++) {
                    m->bytes[j] = ALPHABET[char_indices[j]];
                }
                m->bytes[depth+1] = solution;
                // nul-terminate
                if (depth + 2 < SEQ_LEN) {
                    m->bytes[depth+2] = 0;
                }
            }
        }

        if (depth < SEARCH_DEPTH - 1) {
            base_hashes[++depth] = base_hash; 
        }
    }
}