FROM python:3.13-slim-bookworm

WORKDIR /app

# Pin to Debian 12 (bookworm) explicitly: the unpinned `python:3.13-slim`
# tag rolled to trixie (Debian 13) and brought CDN flake noise from a
# fresh release. bookworm is stable through June 2028.
#
# Switch the apt mirror from deb.debian.org (Fastly) to cloudfront.debian.net
# (AWS CloudFront). Fastly's edge for deb.debian.org is intermittently 451-ing
# `unused` on InRelease + .deb fetches today on BOTH trixie AND bookworm
# (reproduced in pipelines 1143, 1171, 1193 on packages like git, libcurl4,
# build-essential transitive deps). Same package contents, different CDN.
# `cloudfront.debian.net` is the official Debian-recommended AWS-backed
# mirror and is the standard fallback when Fastly hiccups.
#
# Acquire::Retries=3 covers any residual transient hiccup against the
# CloudFront edge.
RUN sed -i 's|deb.debian.org|cloudfront.debian.net|g' \
        /etc/apt/sources.list.d/debian.sources 2>/dev/null \
    || sed -i 's|deb.debian.org|cloudfront.debian.net|g' /etc/apt/sources.list \
    && echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

# Core deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directory (in-process score cache only - all data lives in the DB)
RUN mkdir -p data/cache

EXPOSE 8100

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8100/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8100"]
