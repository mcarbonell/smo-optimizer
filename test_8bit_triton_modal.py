"""Legacy compatibility wrapper. Canonical entrypoint: benchmarks.runners.modal.test_8bit_triton_modal."""

import runpy


if __name__ == "__main__":
    runpy.run_module("benchmarks.runners.modal.test_8bit_triton_modal", run_name="__main__")
