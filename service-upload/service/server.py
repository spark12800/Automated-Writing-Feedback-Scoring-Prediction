"""
FastAPI service for the IELTS essay band scorer.

Two front doors on one process:

    POST /score   JSON API for the website
    GET  /demo    the existing Gradio UI, unchanged
    GET  /docs    auto-generated, clickable API documentation
    GET  /health  did the model load?

Neither app.py nor inference.py is modified -- this only adds a new entry point.

Run locally:
    uvicorn server:app --reload
"""

import os

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import inference
from app import demo  # the Gradio UI object built in app.py

api = FastAPI(
    title="IELTS Essay Scorer",
    version="1.0.0",
    description=(
        "Estimates an IELTS Writing Task 2 band (4-8) from a prompt and an essay. "
        "An automated estimate, not an official IELTS score."
    ),
)

# Your website will be served from a different domain, and browsers block
# cross-domain requests unless the server allows them. Set ALLOWED_ORIGINS to
# your site (comma separated) before going public; "*" is fine while developing.
api.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --- the shape of a request and a response -------------------------------
# FastAPI checks incoming JSON against these. A request missing "essay" is
# rejected with a clear error instead of reaching your model and crashing.
class ScoreRequest(BaseModel):
    topic: str = Field("", description="The Task 2 question the essay answers.")
    essay: str = Field(..., description="The candidate's essay (~20+ words).")


class ScoreResponse(BaseModel):
    band: float
    pred_id: int
    cumulative_probs: list[float]
    n_words: int


@api.get("/health")
def health():
    """Cheap liveness check -- point uptime monitoring at this."""
    return {
        "status": "ok",
        "device": inference.DEVICE,
        "bands": sorted(inference.ID_TO_BAND.values()),
    }


@api.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """Score one essay."""
    try:
        return inference.predict(req.topic, req.essay)
    except ValueError as e:
        # e.g. "essay too short" -- the caller's fault, so 422 not 500
        raise HTTPException(status_code=422, detail=str(e))


# Mount the Gradio demo alongside the API. `app` is what uvicorn serves.
app = gr.mount_gradio_app(api, demo, path="/demo")
