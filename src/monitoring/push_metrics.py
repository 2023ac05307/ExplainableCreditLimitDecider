import os, time
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

PUSHGATEWAY = os.getenv("PUSHGATEWAY_URL", "http://pushgateway:9091")

def push_job_status(job_name: str, ok: bool, duration_s: float, extra: dict | None = None):
    reg = CollectorRegistry()

    success = Gauge("ml_job_success", "Job success (1/0)", ["job"], registry=reg)
    dur = Gauge("ml_job_duration_seconds", "Job duration (seconds)", ["job"], registry=reg)
    ts = Gauge("ml_job_last_run_timestamp", "Last run unix timestamp", ["job"], registry=reg)

    success.labels(job=job_name).set(1 if ok else 0)
    dur.labels(job=job_name).set(float(duration_s))
    ts.labels(job=job_name).set(time.time())

    if extra:
        for k, v in extra.items():
            g = Gauge(f"ml_job_{k}", f"Extra metric {k}", ["job"], registry=reg)
            g.labels(job=job_name).set(float(v))

    push_to_gateway(PUSHGATEWAY, job=job_name, registry=reg)
