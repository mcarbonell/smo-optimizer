"""Legacy compatibility wrapper. Canonical entrypoint: benchmarks.runners.modal.test_modal_connection."""

import runpy


if __name__ == "__main__":
    runpy.run_module("benchmarks.runners.modal.test_modal_connection", run_name="__main__")
