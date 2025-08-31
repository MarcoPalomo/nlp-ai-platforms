FROM nvcr.io/nvidia/pytorch:24.04-py3 AS builder

RUN pip install --upgrade pip && \
    pip install --prefix=/install "vllm[all]" transformers accelerate

FROM nvcr.io/nvidia/pytorch:24.04-py3

RUN useradd -m mistral
USER mistral

WORKDIR /home/mistral

COPY --from=builder /install /usr/local

ENV MODEL_PATH=/models

CMD ["python", "-m", "vllm.entrypoints.api_server", \
     "--model", "/models", \
     "--port", "8000", \
     "--host", "0.0.0.0"]
    