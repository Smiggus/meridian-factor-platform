# ─────────────────────────────────────────────────────────────────────────────
# meridian-factors code-location image (gRPC server).
#
# Consumed by dagster-platform. Exposes a gRPC code server on
# port 4000 that the shared webserver + daemon connect to. See
# dagster-platform/docs/CODE_LOCATIONS.md for the contract this image
# fulfils.
#
# Build context: Meridian repo root.
#   docker build -t meridian-factors-code -f dagster/Dockerfile .
#
# This image is independent of the root Dockerfile (which builds the
# FastAPI factor-scoring API, a separate service).
# ─────────────────────────────────────────────────────────────────────────────

FROM registry.example.com/python-torch-base:3.11-cu128

WORKDIR /app

# Code-location deps (distinct from the API's requirements.txt)
COPY dagster/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache the DuckDB httpfs extension into the image (Meridian #101). The
# parquet read path runs `LOAD httpfs` to read LDP's Parquet export from MinIO
# over S3. Without the extension baked in, run pods raise "Extension httpfs not
# found" on LOAD and silently fall back to PostgreSQL - exactly the failure the
# #101 read-path diagnostic surfaced (use_s3=True, probe failed: httpfs not
# found). The build has egress to DuckDB's extension repo, the run pods may
# not. Installs into /root/.duckdb so
# runtime LOAD finds it offline; version-matched to the duckdb in
# requirements.txt since build and runtime share this image.
RUN python -c "import duckdb; duckdb.connect().execute('INSTALL httpfs')"

# Tenant source - meridian_pipeline is the Dagster module.
COPY dagster/meridian_pipeline/ ./meridian_pipeline/

# Off-Dagster burst scorer + Garage relay loader. Lets this image double as the
# burst worker (python /app/scripts/burst/rescore_groups.py ...,
# python /app/scripts/burst/replay_asset.py ...) without a separate build.
# meridian_pipeline is importable via PYTHONPATH=/app, so the scripts' dev-only
# sys.path.insert(0,"dagster") is a harmless no-op here.
COPY scripts/ ./scripts/

ENV DAGSTER_HOME=/opt/dagster/dagster_home
ENV PYTHONPATH=/app

EXPOSE 4000

CMD ["dagster", "code-server", "start", \
     "-h", "0.0.0.0", "-p", "4000", \
     "-m", "meridian_pipeline.definitions"]
