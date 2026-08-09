# Container for the IELTS band scorer. Runs the same locally, on Cloud Run,
# or on any host that takes an image.
#
#   docker build -t ielts-scorer .
#   docker run -p 8080:8080 ielts-scorer      ->  http://localhost:8080/docs

FROM python:3.12-slim

WORKDIR /code

# Dependencies first, in their own layer. Docker caches this, so editing code
# does not re-download torch on every rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the base model's config.json into the image (a few hundred bytes).
# inference.py builds the architecture from it and restores every weight from
# our own checkpoint, so the container needs no network access at startup.
ENV HF_HOME=/code/.cache
RUN python -c "from transformers import AutoConfig; \
    AutoConfig.from_pretrained('microsoft/deberta-v3-base')"

COPY . .

ENV MODEL_DIR=/code/ielts-band-coral

# Cloud Run injects PORT and expects the app to listen on it. The shell form is
# deliberate -- the exec form would not expand the variable.
ENV PORT=8080
EXPOSE 8080

# One worker on purpose: each worker loads its own copy of a ~740MB model.
CMD exec uvicorn server:app --host 0.0.0.0 --port ${PORT} --workers 1
