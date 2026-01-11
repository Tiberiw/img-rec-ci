FROM python:3.11-slim AS builder
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt
FROM python:3.11-slim
COPY --from=builder /opt/venv /opt/venv
COPY app app/
# copy pretrained models
COPY .cvlib .cvlib/
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
EXPOSE 80
CMD ["fastapi", "run", "app/main.py", "--port", "80"]