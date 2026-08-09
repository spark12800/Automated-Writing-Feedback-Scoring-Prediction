# Step 1 — Export the trained model from Colab

The training notebook's section 8 already writes everything and copies it to
Drive, so there is no extra export step. Just download that folder:

```
MyDrive/ielts-band-coral/
```

Put it inside this project so you have:

```
essay-scorer-space/
├── server.py                    REST API (FastAPI)
├── app.py                       Gradio UI
├── inference.py                 model loading + predict()
├── Dockerfile
├── requirements.txt
├── README.md
└── ielts-band-coral/
    ├── model.safetensors
    ├── tokenizer.json, tokenizer_config.json
    ├── coral_cutoffs.npy
    └── head_config.json         band mapping + head settings
```

`head_config.json` is the one people forget. Without it `inference.py` cannot
turn a predicted class index back into a band, and it refuses to start rather
than guess.

> Note: `inference.py` must define the same model head as the training
> notebook. It loads with `strict=True` on purpose — if the two ever drift
> apart you get a loud error instead of a randomly-initialised head quietly
> serving nonsense.

# Step 2 — Test locally

```bash
cd essay-scorer-space
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn server:app --reload   # API + demo, http://127.0.0.1:8000
```

Then:

| URL | What it is |
| --- | --- |
| `/docs` | clickable API documentation — try `/score` from the browser |
| `/demo` | the Gradio UI |
| `/health` | did the model load? |

From the command line:

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{"topic": "Some people think...", "essay": "In modern society, ..."}'
```

`python app.py` still works if you only want the Gradio UI.

# Step 2b — Test in Docker

Same thing, but in the sealed box that will run in production:

```bash
docker build -t ielts-scorer .
docker run -p 7860:7860 ielts-scorer      # http://localhost:7860/docs
```

If it works here, it will work on any host that runs containers.

# Step 3 — Deploy to Hugging Face Spaces

1. Create a free account at https://huggingface.co
2. Click **New → Space**, choose **Gradio** SDK, name it (e.g. `ielts-essay-scorer`).
3. Push these files. Easiest way:

```bash
# from inside essay-scorer-space/
git init
git remote add origin https://huggingface.co/spaces/<your-username>/ielts-essay-scorer
git lfs install
git lfs track "*.safetensors" "*.bin" "*.npy" "*.model"
git add .
git commit -m "Deploy IELTS essay scorer"
git push origin main
```

The Space builds automatically and gives you a public URL. Done — that's the
"production" v1 your users can visit.

> Free CPU Spaces score one essay in a few seconds. If that's too slow, upgrade
> the Space hardware to a GPU in its Settings (paid), no code change needed.

**To serve the REST API from the Space instead of just the Gradio UI**, switch
its README frontmatter from the Gradio SDK to Docker:

```yaml
sdk: docker
app_port: 7860
```

Spaces then builds the `Dockerfile` in this folder, and your website gets a real
`POST /score` endpoint with the demo still available at `/demo`.

# Step 4 — When the website needs more

The API is deliberately thin. Things to add when you actually need them, not
before:

- `POST /feedback` — the LLM explanation, once the prompt is proven in
  `Feedback_Generation.ipynb`. Keep it a separate endpoint: it is slow, costs
  money per call, and can fail, none of which should hold up a band score.
- API keys / rate limiting — the moment the URL is public.
- CORS — needed as soon as a browser on a different domain calls the API
  (`fastapi.middleware.cors.CORSMiddleware`).
