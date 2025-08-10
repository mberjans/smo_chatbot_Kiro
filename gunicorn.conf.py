#!/usr/bin/env python3
"""
Gunicorn configuration for Clinical Metabolomics Oracle
"""

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = min(4, multiprocessing.cpu_count())
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 300
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "clinical_metabolomics_oracle"

# Server mechanics
daemon = False
pidfile = "/tmp/gunicorn_cmo.pid"
user = None
group = None
tmp_upload_dir = None

# SSL (if needed in production)
# keyfile = None
# certfile = None

# Application
chdir = "."
pythonpath = "."

# Environment variables
raw_env = [
    f"DATABASE_URL={os.environ.get('DATABASE_URL', 'postgresql://test:test@localhost:5432/test_db')}",
    f"NEO4J_PASSWORD={os.environ.get('NEO4J_PASSWORD', 'test_password')}",
    f"PERPLEXITY_API={os.environ.get('PERPLEXITY_API', 'test_api_key_placeholder')}",
    f"OPENAI_API_KEY={os.environ.get('OPENAI_API_KEY', 'sk-test_key_placeholder')}",
]

# Preload application for better performance
preload_app = True

# Worker process lifecycle
def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Clinical Metabolomics Oracle starting...")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Clinical Metabolomics Oracle reloading...")

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("Clinical Metabolomics Oracle ready to serve requests")
    server.log.info(f"Listening on: http://{bind}")
    server.log.info(f"Chat interface: http://{bind}/chat")
    server.log.info(f"API docs: http://{bind}/docs")

def on_exit(server):
    """Called just before exiting."""
    server.log.info("Clinical Metabolomics Oracle shutting down...")

def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT."""
    worker.log.info(f"Worker {worker.pid} received INT or QUIT signal")

# Memory and performance tuning
worker_tmp_dir = "/dev/shm"  # Use shared memory for better performance
forwarded_allow_ips = "*"
secure_scheme_headers = {
    'X-FORWARDED-PROTOCOL': 'ssl',
    'X-FORWARDED-PROTO': 'https',
    'X-FORWARDED-SSL': 'on'
}