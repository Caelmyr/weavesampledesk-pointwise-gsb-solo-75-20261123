ARG WEAVEDESK_RUNTIME_IMAGE=python:3.12-slim
FROM ${WEAVEDESK_RUNTIME_IMAGE}
USER root
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 WEAVEDESK_DATA=/var/lib/weavedesk
RUN if [ -d /app ]; then rm -rf /app; fi
WORKDIR /opt/weavedesk
COPY requirements.txt scripts/check_dependencies.py ./
RUN (python check_dependencies.py || python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt) \
    && python check_dependencies.py && python -m pip check
COPY weavedesk ./weavedesk
COPY image_lab ./image_lab
COPY static ./static
COPY templates ./templates
RUN useradd -U -u 11532 weavedesk && install -d -o weavedesk -g weavedesk /var/lib/weavedesk
USER weavedesk
EXPOSE 8000
HEALTHCHECK NONE
CMD ["python", "-m", "weavedesk"]
