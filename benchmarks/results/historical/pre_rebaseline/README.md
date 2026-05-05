# Historical Results Archive

These files were produced before the benchmark cleanup and re-baselining pass.

They are preserved only as historical reference:

- naming was inconsistent across suites
- metadata was incomplete or heterogeneous
- some files predate the unified result format
- the repository is expected to re-run all benchmarks after algorithm iteration

## Normalization Rule Used Here

- filenames were renamed to describe dataset, variant family, and context
- each archived filename ends with `_legacy`
- log files were separated from JSON benchmark artifacts

## Important Note

Do not treat these files as the benchmark baseline for future optimization work. The next meaningful baseline will come from re-running smoke, microbenchmark, and end-to-end suites after the algorithm review phase.
