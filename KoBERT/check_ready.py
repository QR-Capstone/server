"""Offline deployment check: python KoBERT/check_ready.py (no URL requests)."""
import os
import sys
import time


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    started = time.perf_counter()
    try:
        import koBERT as engine
        engine.warmup_engine(include_pw=False)
        first_model = engine.model
        first_tokenizer = engine.tokenizer
        engine.warmup_engine(include_pw=False)
        if (not engine.engine_initialized or engine.model is not first_model
                or engine.tokenizer is not first_tokenizer):
            raise RuntimeError("Model/tokenizer reuse check failed")
    except Exception as exc:
        print(f"[KoBERT NOT READY] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"[KoBERT READY] device={engine.device}; local assets verified; "
          f"512-token warmup and reuse passed ({time.perf_counter() - started:.2f}s)")
    print("This check exits here. The server prepares its own model at startup; "
          "check /ready after starting the server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
