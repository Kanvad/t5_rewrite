"""T5 text checker & rewriter with a Gradio front-end.

The model is prompted to inspect the input for typos / grammatical errors
and return a corrected version. If the input is already correct the model is
instructed to return it unchanged (skip).
"""

import torch
import re
from difflib import SequenceMatcher
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


def check_and_rewrite(
    text: str,
) -> tuple[list[tuple[str, str | None]], str]:
    """Check `text` for errors and return the corrected version.

    Returns the highlighted spans for display plus the plain corrected
    text, so the result can be copied without any markup. If no typo /
    grammatical errors are found the text is returned unchanged.
    """
    if not text or not text.strip():
        return [], ""
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

    corrected_text = "\n".join(corrected_lines)
    # Keep whitespace as its own token so a highlighted span never includes
    # trailing spaces (which would render as an empty colored block).
    original_tokens = re.findall(r"\s+|\S+", text)
    corrected_tokens = re.findall(r"\s+|\S+", corrected_text)
    highlighted_text = []

    matcher = SequenceMatcher(None, original_tokens, corrected_tokens)
    for tag, _, _, corrected_start, corrected_end in matcher.get_opcodes():
        segment = "".join(corrected_tokens[corrected_start:corrected_end])
        if tag == "equal":
            if segment:
                highlighted_text.append((segment, None))
            continue
        # Never color whitespace: split it out as unhighlighted text.
        for piece in re.findall(r"\s+|\S+", segment):
            highlighted_text.append(
                (piece, None if piece.isspace() else "changed")
            )

    return highlighted_text, corrected_text


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
                    submit_btn = gr.Button("Check & Rewrite", variant="primary")
                    clear_btn = gr.Button("Clear")
            with gr.Column():
                output_box = gr.HighlightedText(
                    label="Output text",
                    combine_adjacent=True,
                    # A built-in palette name keeps the background fully opaque
                    color_map={"changed": "yellow"},
                    # Don't draw the category badge after each highlighted span.
                    show_inline_category=False,
                )
                corrected_box = gr.Textbox(
                    label="Corrected text",
                    # info="Plain text — use the copy button to copy it.",
                    lines=6,
                    interactive=False,
                    buttons=["copy"],
                )

        submit_btn.click(
            fn=check_and_rewrite,
            inputs=input_box,
            outputs=[output_box, corrected_box],
        )
        clear_btn.click(
            fn=lambda: ("", [], ""),
            inputs=None,
            outputs=[input_box, output_box, corrected_box],
        )
        gr.Examples(
            examples=[
                ["He go to the store yesterday."],
                ["The quick brown fox jumps over the lazy dog."],
                ["I are happy to seeing you."],
            ],
            inputs=input_box,
            outputs=[output_box, corrected_box],
            fn=check_and_rewrite,
            cache_examples=False,
        )

    demo.launch()


if __name__ == "__main__":
    main()
