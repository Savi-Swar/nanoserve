"""Run nanoserve as an HTTP inference server.

    python serve.py --engine continuous --port 8000

    curl -s localhost:8000/healthz
    curl -s localhost:8000/metrics
    curl -s -XPOST localhost:8000/generate -d '{"prompt":"The capital of France is","max_tokens":16}'

On Apple Silicon add --device cpu (MPS matmul is broken for this model); on a GPU
box it picks CUDA automatically.
"""
import argparse
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
    a = p.parse_args()

    print(f"loading {a.model} ...")
    model = ModelRunner(a.model, device=a.device)
    svc = InferenceService(model, engine=a.engine, max_batch=a.max_batch,
                           max_queue=a.max_queue)
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(svc))
    print(f"nanoserve on http://{a.host}:{a.port}  "
          f"(engine={a.engine}, max_batch={a.max_batch}, max_queue={a.max_queue})")
    print("  POST /generate   GET /metrics   GET /healthz")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        svc.stop()


if __name__ == "__main__":
    main()
