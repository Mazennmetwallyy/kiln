"""Deploy the live demo (demo/space/app.py) to Modal, CPU only, scaling to zero when nobody is using it.

    modal deploy demo/modal_app.py
"""

from pathlib import Path

import modal

HERE = Path(__file__).parent / "space"
KILN_REF = "6bd502e"  # the kiln package version the image installs

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(f"kiln @ git+https://github.com/Mazennmetwallyy/kiln@{KILN_REF}")
    .run_commands("python -c \"from huggingface_hub import snapshot_download as d; "
                  "d('Qwen/Qwen2.5-0.5B-Instruct', allow_patterns=['*.json', '*.safetensors', '*.jinja'])\"")
    .add_local_file(HERE / "app.py", "/root/demo_app.py")
    .add_local_file(HERE / "index.html", "/root/index.html")
)
app = modal.App("kiln-demo", image=image)


# One container (the engine batches requests itself), shut down after 5 idle minutes.
@app.function(cpu=2.0, memory=3072, max_containers=1, scaledown_window=300, timeout=3600)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    from demo_app import build

    return build()
