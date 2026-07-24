"""Run nanoserve as an HTTP inference server.

    python serve.py --engine continuous --port 8000

    curl -s localhost:8000/healthz
    curl -s localhost:8000/metrics
    curl -s -XPOST localhost:8000/generate -d '{"prompt":"The capital of France is","max_tokens":16}'
    curl -N -XPOST localhost:8000/generate -d '{"prompt":"Tell me a story","max_tokens":64,"stream":true}'

SIGTERM / Ctrl-C drains in-flight requests before exiting (graceful shutdown).
On Apple Silicon add --device cpu (MPS matmul is broken for this model); on a GPU
box it picks CUDA automatically.
"""
import argparse
import signal
import threading
from http.server import ThreadingHTTPServer

from server.model import ModelRunner
from server.service import InferenceService, make_handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--engine", default="continuous",
                   choices=["continuous", "paged", "spec_cont"])
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--device", default=None)
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--max-queue", type=int, default=48,
                   help="shed load with 503 once pending+active reaches this")
    p.add_argument("--request-timeout", type=float, default=None,
                   help="per-request deadline in seconds; the engine evicts on breach")
    p.add_argument("--drain-timeout", type=float, default=30.0,
                   help="seconds to let in-flight requests finish on shutdown")
    p.add_argument("--log", action="store_true", help="structured per-request lifecycle logs")
    a = p.parse_args()

    print(f"loading {a.model} ...")
    model = ModelRunner(a.model, device=a.device)
    svc = InferenceService(model, engine=a.engine, max_batch=a.max_batch,
                           max_queue=a.max_queue, request_timeout=a.request_timeout,
                           log=a.log)
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(svc))
    httpd.daemon_threads = True

    print(f"nanoserve on http://{a.host}:{a.port}  "
          f"(engine={a.engine}, max_batch={a.max_batch}, max_queue={a.max_queue}"
          + (f", timeout={a.request_timeout}s" if a.request_timeout else "") + ")")
    print("  POST /generate   POST /generate {stream:true}   GET /metrics   GET /healthz")

    # serve in a background thread; the main thread waits for a shutdown signal
    # so it can drain gracefully (httpd.shutdown() must not run on the serving
    # thread).
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait()

    print("\nshutting down: draining in-flight requests ...")
    drained = svc.drain(timeout=a.drain_timeout)
    print(f"  drained cleanly: {drained}")
    print("final metrics:\n" + svc.prometheus())
    httpd.shutdown()
    svc.stop()
    print("bye.")


if __name__ == "__main__":
    main()
