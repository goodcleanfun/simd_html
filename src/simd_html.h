#ifndef SIMD_HTML_PARSER_H
#define SIMD_HTML_PARSER_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdarg.h>

#include "bit_utils/bit_utils.h"
#include "simde_avx2/avx2.h"

typedef struct {
    const char *start;
    const char *end;
    size_t base_offset;
    size_t offset;
    uint64_t matches;
} simd_html_match_state_t;

static inline void simd_html_match_state_update(simd_html_match_state_t *state, const char *buffer) {
    if (state == NULL || buffer == NULL) return;

    static uint8_t low_nibble_mask_data[32] = {
        0, 0, 0, 0, 0, 0, 0x26, 0, 
        0, 0, 0, 0, 0x3c, 0xd, 0, 0,
        0, 0, 0, 0, 0, 0, 0x26, 0, 
        0, 0, 0, 0, 0x3c, 0xd, 0, 0,
    };

    static uint8_t bit_mask_data[32] = {
        0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
    };

    simde__m256i low_nibble_mask = simde_mm256_load_si256((const simde__m256i *)low_nibble_mask_data);
    simde__m256i v0f = simde_mm256_set1_epi8(0x0f);
    simde__m256i bit_mask = simde_mm256_load_si256((const simde__m256i *)bit_mask_data);

    simde__m256i data1 = simde_mm256_loadu_si256((const simde__m256i *)buffer);
    simde__m256i data2 = simde_mm256_loadu_si256((const simde__m256i *)(buffer + 32));

    simde__m256i lowpart1_idx1 = simde_mm256_and_si256(data1, v0f);
    simde__m256i lowpart2_idx2 = simde_mm256_and_si256(data2, v0f);
    simde__m256i lowpart_idx_max = simde_mm256_set1_epi8(15);

    simde__m256i lowpart1 = simde_mm256_shuffle_epi8(low_nibble_mask, 
        simde_mm256_or_si256(lowpart1_idx1, 
            simde_mm256_cmpgt_epi8(lowpart1_idx1, lowpart_idx_max)));
    simde__m256i lowpart2 = simde_mm256_shuffle_epi8(low_nibble_mask, 
        simde_mm256_or_si256(lowpart2_idx2, 
            simde_mm256_cmpgt_epi8(lowpart2_idx2, lowpart_idx_max)));

    simde__m256i matchesones1 = simde_mm256_cmpeq_epi8(lowpart1, data1);
    simde__m256i matchesones2 = simde_mm256_cmpeq_epi8(lowpart2, data2);

    static const uint8_t odds_data[32] = {
        1, 3, 5, 7, 9, 11, 13, 15,
        -1, -1, -1, -1, -1, -1, -1, -1,
        1, 3, 5, 7, 9, 11, 13, 15,
        -1, -1, -1, -1, -1, -1, -1, -1,
    };

    static const uint8_t evens_data[32] = {
        0, 2, 4, 6, 8, 10, 12, 14,
        -1, -1, -1, -1, -1, -1, -1, -1,
        0, 2, 4, 6, 8, 10, 12, 14,
        -1, -1, -1, -1, -1, -1, -1, -1
    };

    simde__m256i evens = simde_mm256_load_si256((const simde__m256i *)evens_data);
    simde__m256i odds = simde_mm256_load_si256((const simde__m256i *)odds_data);

    simde__m256i matchesones1_and_mask = simde_mm256_and_si256(matchesones1, bit_mask);
    simde__m256i matchesones2_and_mask = simde_mm256_and_si256(matchesones2, bit_mask);

    simde__m256i sums = simde_mm256_add_epi8(
        simde_mm256_permute4x64_epi64(
            simde_mm256_shuffle_epi8(matchesones1_and_mask, evens),
            SIMDE_MM_SHUFFLE(3, 1, 2, 0)
        ),
        simde_mm256_permute4x64_epi64(
            simde_mm256_shuffle_epi8(matchesones1_and_mask, odds),
            SIMDE_MM_SHUFFLE(3, 1, 2, 0)
        )
    );

    sums = simde_mm256_add_epi8(
        sums,
        simde_mm256_add_epi8(
            simde_mm256_permute4x64_epi64(
                simde_mm256_shuffle_epi8(matchesones2_and_mask, evens),
                SIMDE_MM_SHUFFLE(2, 0, 3, 1)
            ),
            simde_mm256_permute4x64_epi64(
                simde_mm256_shuffle_epi8(matchesones2_and_mask, odds),
                SIMDE_MM_SHUFFLE(2, 0, 3, 1)
            )
        )
    );

    sums = simde_mm256_add_epi8(
        simde_mm256_permute4x64_epi64(
            simde_mm256_shuffle_epi8(sums, evens),
            SIMDE_MM_SHUFFLE(3, 1, 2, 0)
        ),
        simde_mm256_permute4x64_epi64(
            simde_mm256_shuffle_epi8(sums, odds),
            SIMDE_MM_SHUFFLE(3, 1, 2, 0)
        )
    );

    sums = simde_mm256_add_epi8(
        simde_mm256_permute4x64_epi64(
            simde_mm256_shuffle_epi8(sums, evens),
            SIMDE_MM_SHUFFLE(3, 1, 2, 0)
        ),
        simde_mm256_permute4x64_epi64(
            simde_mm256_shuffle_epi8(sums, odds),
            SIMDE_MM_SHUFFLE(3, 1, 2, 0)
        )
    );

    state->matches = simde_mm256_extract_epi64(sums, 0);
}



static inline void simd_html_match_state_careful_update(simd_html_match_state_t *state) {
    uint8_t buffer[64] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    memcpy(buffer, state->start, state->end - state->start);
    simd_html_match_state_update(state, (const char *)buffer);
}


static simd_html_match_state_t simd_html_match_state_init(const char *start, const char *end) {
    simd_html_match_state_t state;
    state.start = start;
    state.end = end;
    state.base_offset = 0;
    state.offset = 0;
    state.matches = 0;

    if (start + 64 >= end) {
        simd_html_match_state_careful_update(&state);
    } else {
        simd_html_match_state_update(&state, start);
    }

    return state;
}

void simd_html_match_state_consume(simd_html_match_state_t *state) {
    state->offset++;
    state->matches >>= 1;
}

bool simd_html_match_state_advance(simd_html_match_state_t *state) {
    while (state->matches == 0) {
        state->start += 64;
        state->base_offset += 64;
        state->offset = state->base_offset;

        if (state->start >= state->end) {
            return false;
        }
        if (state->start + 64 >= state->end) {
            simd_html_match_state_careful_update(state);
            if (state->matches == 0) {
                return false;
            }
        } else {
            simd_html_match_state_update(state, (const char *)state->start);
        }
    }
    size_t off = ctz(state->matches);
    state->matches >>= off;
    state->offset += off;
    return true;
}

#endif