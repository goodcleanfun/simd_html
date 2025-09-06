# simd_html
Fast streaming HTML tokenizer using SIMDe for token classification

In benchmarking (with RDTSC) on an Apple x64 laptop, vectorized SIMD token classification gets around 6.2 GB/s on standard HTML pages from Google/BBC, which is about 20x faster than a naive implementation (scanning through the string one character at a time, even when checking by most frequent to least frequent and compiling with `-O3` which would trigger auto-vectorization, which might be sufficient in the case of a single matched character but not for matching multiple characters).

In a larger streaming use case it can parse 1 GB of Wikipedia HTML at around 4 GB/s, beating the naive implementation by 8x.

On ARM NEON, uses the original version from Daniel Lemire: https://github.com/lemire/htmlscanning. Although the registers are smaller on ARM NEON, this version is likely faster overall since it has more efficient instructions (particularly PADDQ) and higher superscalarity (can execute more SIMD instructions per cycle).