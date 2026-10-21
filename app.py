"""T5 text checker & rewriter with a Gradio front-end.

The model is prompted to inspect the input for typos / grammatical errors
and return a corrected version. If the input is already correct the model is
instructed to return it unchanged (skip).
"""

import torch
import re
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import gradio as gr

MODEL_ID = "visheratin/t5-efficient-mini-grammar-correction"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)  # CPU (no device_map, no float16)
print("Model ready.")


def check_and_rewrite(text: str) -> str:
    """Check `text` for errors and return the corrected version.

    If no typo / grammatical errors are found the text is
    returned unchanged.
    """
    if not text or not text.strip():
        return ""
    # Correct each non-empty line independently so a list of sentences cannot
    # be merged into a summary by the small seq2seq model.
    lines = text.splitlines()
    corrected_lines = []
    for line in lines:
        if not line.strip():
            corrected_lines.append(line)
            continue

        marker_match = re.match(r"^(\s*(?:\d+[.)]|[-*])\s+)(.*)$", line)
        marker = marker_match.group(1) if marker_match else ""
        line_text = marker_match.group(2) if marker_match else line
        inputs = tokenizer(line_text, return_tensors="pt", truncation=True)
        with torch.inference_mode():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max(32, min(128, len(line.split()) * 4)),
                num_beams=4,
                do_sample=False,
            )
        corrected_text = tokenizer.decode(
            output_tokens[0], skip_special_tokens=True
        ).strip()
        corrected_lines.append(f"{marker}{corrected_text}")

    return "\n".join(corrected_lines)


def main():
    with gr.Blocks() as demo:
        gr.Markdown(
            "# T5 Text Checker & Rewriter\n"
            "Enter text. If it contains typos or grammatical errors it is "
            "rewritten correctly; otherwise it is returned unchanged."
        )
        with gr.Row():
            with gr.Column():
                input_box = gr.Textbox(
                    label="Input text",
                    placeholder="Type or paste text to check...",
                    lines=8,
                )
                with gr.Row():
                    submit_btn = gr.Button("Check & Rewrite")
                    clear_btn = gr.Button("Clear")
            with gr.Column():
                output_box = gr.Textbox(
                    label="Output text",
                    lines=8,
                    buttons=["copy"],
                )

        submit_btn.click(
            fn=check_and_rewrite,
            inputs=input_box,
            outputs=output_box,
        )
        clear_btn.click(
            fn=lambda: "",
            inputs=None,
            outputs=input_box,
        )
        gr.Examples(
            examples=[
                ["He go to the store yesterday."],
                ["The quick brown fox jumps over the lazy dog."],
                ["I are happy to seeing you."],
            ],
            inputs=input_box,
            outputs=output_box,
            fn=check_and_rewrite,
            cache_examples=False,
        )

    demo.launch()


if __name__ == "__main__":
    main()
