FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
ARG KILN_REF=main
RUN pip install --no-cache-dir --user "kiln @ git+https://github.com/Mazennmetwallyy/kiln@${KILN_REF}"
RUN python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-0.5B-Instruct', allow_patterns=['*.json', '*.safetensors', '*.jinja'])"
COPY --chown=user app.py index.html ./
EXPOSE 7860
CMD ["python", "app.py"]
