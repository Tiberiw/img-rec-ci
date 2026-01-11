FROM python:3.11-slim AS builder
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt
FROM python:3.11-slim
RUN groupadd -g 9999 appgroup && \
    useradd -u 9999 -g appgroup --shell /bin/bash --create-home appuser && \
    mkdir -p /home/appuser/images_uploaded && \
    chown appuser:appgroup /home/appuser/images_uploaded
WORKDIR /home/appuser

COPY --chown=appuser:appgroup --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appgroup app app/
COPY --chown=appuser:appgroup .cvlib .cvlib/
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
EXPOSE 80
USER appuser
CMD ["fastapi", "run", "app/main.py", "--port", "80"]