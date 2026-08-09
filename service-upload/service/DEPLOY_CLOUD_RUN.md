# Deploying to Google Cloud Run

The scorer as a public HTTPS API, on Google Cloud.

---

## Step 0 — Test the container locally first

Debugging a build on your laptop takes seconds; debugging it in Cloud Build
takes minutes per attempt.

```bash
brew install --cask docker      # or: brew install colima docker (much lighter)
open -a Docker                  # wait for the whale icon to settle

cd ~/Desktop/essay-scorer-space
docker build -t ielts-scorer .
docker run -p 8080:8080 ielts-scorer
```

Open <http://localhost:8080/docs> and score an essay. If that works, Cloud Run
will work — that is the entire point of the container.

Expect the build to take several minutes and the image to land around 3–4 GB
(torch is most of it, the model is ~740 MB).

---

## Step 1 — Set up gcloud

```bash
brew install --cask google-cloud-sdk

gcloud auth login
gcloud projects create ielts-scorer-demo        # or reuse an existing project
gcloud config set project ielts-scorer-demo
```

Billing must be enabled on the project, even to stay inside the free tier.

Enable the two services the deploy needs:

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

---

## Step 2 — Deploy

```bash
gcloud run deploy ielts-scorer \
  --source . \
  --region europe-west2 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --allow-unauthenticated
```

`--source .` hands the Dockerfile to Cloud Build, which builds the image and
deploys it. No local Docker required for this step — though Step 0 is still
worth doing.

You get back a URL like `https://ielts-scorer-xxxxx.a.run.app`. Then:

| URL | What |
| --- | --- |
| `/docs` | interactive API documentation |
| `/demo` | the Gradio UI |
| `/health` | liveness check |
| `POST /score` | the API your website calls |

---

## Why those flags

| Flag | Reason |
| --- | --- |
| `--memory 4Gi` | torch plus a 740 MB fp32 model. 2 GiB gets OOM-killed on load. |
| `--cpu 2` | inference is CPU-bound; one vCPU roughly doubles latency. |
| `--timeout 300` | a long essay on a cold container can exceed the 60s default. |
| `--allow-unauthenticated` | it is a public demo. Drop this and the URL needs an auth token. |

---

## Cold starts

Cloud Run scales to zero, so an idle service must load a 740 MB model before it
answers — expect **15–30 seconds** on the first request, then fast.

For a demo that is usually fine, and being able to explain the tradeoff is worth
more than hiding it. If you need it always warm:

```bash
gcloud run services update ielts-scorer --min-instances 1
```

That keeps one container alive and **does cost money** — it is outside the
always-free tier. Turn it off when you are not demoing.

The image deliberately contains no Hugging Face download at startup: the
architecture is built from a baked-in `config.json` and every weight comes from
`ielts-band-coral/model.safetensors`. Boot time is disk plus CPU only, with no
network call that could fail.

---

## Cost

Cloud Run bills per request and per container-second, with a monthly free tier.
A portfolio demo scaling to zero costs approximately nothing. `--min-instances 1`
does not.

To take it down entirely:

```bash
gcloud run services delete ielts-scorer --region europe-west2
```

---

## Updating it

```bash
gcloud run deploy ielts-scorer --source . --region europe-west2
```

Same command. Cloud Run keeps the previous revision and can roll back to it.
