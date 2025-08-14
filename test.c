#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "greatest/greatest.h"
#include "aligned/aligned.h"
#include "vector_classification.h"

TEST test_html_scan(void) {
    const char *html = "<!doctype html><html><head><title>A Document</title><link rel=\"stylesheet\" href=\"style.css\"></head><body><h1>Hello, World!</h1></body></html>";

    size_t len = strlen((const char *)html);

    char *data = aligned_malloc(len + 1, 32);
    memcpy(data, html, len);
    data[len] = '\0';

    for (simd_html_match_state_t state = simd_html_match_state_init((const char *)data, (const char *)data + len); simd_html_match_state_advance(&state); simd_html_match_state_consume(&state)) {
        ASSERT_EQ(*(state.start + state.offset), '<');
    }

    aligned_free(data);
    PASS();
}


/* Add definitions that need to be in the test runner's main file. */
GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
    GREATEST_MAIN_BEGIN();      /* command-line options, initialization. */

    RUN_TEST(test_html_scan);

    GREATEST_MAIN_END();        /* display results */
}