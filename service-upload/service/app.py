"""
Gradio web UI for the IELTS essay band scorer.

This is the website. It calls inference.predict() -- it contains no model logic
itself, so you can swap the UI or add a REST client later without touching the
model. `demo.launch()` also exposes an automatic REST API at /run/predict.
"""

import gradio as gr

import inference


def score(topic, essay):
    try:
        result = inference.predict(topic, essay)
    except ValueError as e:
        return f"⚠️ {e}", ""

    band = result["band"]
    words = result["n_words"]
    probs = result["cumulative_probs"]

    headline = f"## Predicted IELTS Band: **{band:.1f}**"
    detail = (
        f"- Words analysed: {words}\n"
        f"- Model: DeBERTa-v3 (ordinal / CORAL)\n"
        f"- Cumulative band probabilities: {probs}\n\n"
        f"*This is an automated estimate (~±1 band typical error), "
        f"not an official IELTS score.*"
    )
    return headline, detail


EXAMPLE_TOPIC = (
    "Some people believe that in the future, most cities will become unlivable "
    "due to pollution and overcrowding. Others think technological innovation "
    "will solve these problems. Discuss both sides and give your own opinion."
)

with gr.Blocks(title="IELTS Essay Scorer") as demo:
    gr.Markdown(
        "# ✍️ IELTS Writing Task 2 — Automated Band Scorer\n"
        "Paste the essay prompt and your essay, then get an estimated band."
    )
    with gr.Row():
        with gr.Column():
            topic_in = gr.Textbox(label="Essay prompt / question", lines=3,
                                  placeholder="Paste the Task 2 question here...")
            essay_in = gr.Textbox(label="Your essay", lines=14,
                                  placeholder="Paste your full essay here...")
            btn = gr.Button("Score my essay", variant="primary")
        with gr.Column():
            band_out = gr.Markdown()
            detail_out = gr.Markdown()

    btn.click(score, inputs=[topic_in, essay_in], outputs=[band_out, detail_out])
    gr.Examples(examples=[[EXAMPLE_TOPIC, ""]], inputs=[topic_in, essay_in])

if __name__ == "__main__":
    demo.launch()
