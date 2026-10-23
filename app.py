"""T5 grammar checker and rewriter with a Gradio front-end.

Each non-empty line is corrected independently with a seq2seq model, and the
result is shown next to a token-level diff against the input. Whitespace-only
input produces no highlights and an empty result.
"""

import logging
import re
from difflib import SequenceMatcher
from functools import lru_cache

import gradio as gr
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)

MODEL_ID = "visheratin/t5-efficient-mini-grammar-correction"

CHANGED_LABEL = "changed"
CHANGED_COLOR = "yellow"

NUM_BEAMS = 4
MIN_NEW_TOKENS = 32
MAX_NEW_TOKENS = 128
TOKENS_PER_WORD = 4

# Used when the tokenizer reports no usable maximum input length.
FALLBACK_INPUT_TOKENS = 512
TOKEN_RE = re.compile(r"\s+|\S+")
LIST_MARKER_RE = re.compile(r"^(\s*(?:\d+[.)]|[-*])\s+)(.*)$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=1)
def load_model() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load the tokenizer and model once, on first use."""
    logger.info("Loading model %s...", MODEL_ID)
    # trust_remote_code executes code from the Hub. It is enabled because this
    # repository ships its own tokenizer/architecture code.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model ready.")
    return tokenizer, model


def _input_limit(tokenizer: PreTrainedTokenizerBase) -> int:
    """Return a sane maximum input length for `tokenizer`."""
    limit = tokenizer.model_max_length
    if not isinstance(limit, int) or not 0 < limit <= 100_000:
        return FALLBACK_INPUT_TOKENS
    return limit


def _fits(text: str, tokenizer: PreTrainedTokenizerBase, limit: int) -> bool:
    """Return whether `text` fits in the model's input window."""
    token_ids = tokenizer(text, truncation=True, max_length=limit)["input_ids"]
    return len(token_ids) < limit


def _split_for_model(text: str, tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Split `text` so no piece is silently truncated by the model.

    Text that fits is returned unchanged. Longer text is split on sentence
    boundaries first, then on words if a single sentence is still too long.
    """
    limit = _input_limit(tokenizer)
    if _fits(text, tokenizer, limit):
        return [text]

    chunks: list[str] = []
    for sentence in SENTENCE_SPLIT_RE.split(text):
        if not sentence.strip():
            continue
        if _fits(sentence, tokenizer, limit):
            chunks.append(sentence)
            continue

        current: list[str] = []
        for word in sentence.split():
            candidate = " ".join([*current, word])
            if current and not _fits(candidate, tokenizer, limit):
                chunks.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            chunks.append(" ".join(current))
    return chunks


def _generate(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
) -> str:
    """Run the model on one chunk of text."""
    model_inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=_input_limit(tokenizer),
    )
    token_budget = len(text.split()) * TOKENS_PER_WORD
    with torch.inference_mode():
        output_tokens = model.generate(
            **model_inputs,
            max_new_tokens=max(
                MIN_NEW_TOKENS,
                min(MAX_NEW_TOKENS, token_budget),
            ),
            num_beams=NUM_BEAMS,
        )
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()


def correct_line(
    line: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
) -> str:
    """Correct one line, preserving a leading list marker if present."""
    marker_match = LIST_MARKER_RE.match(line)
    marker = marker_match.group(1) if marker_match else ""
    line_text = marker_match.group(2) if marker_match else line
    corrected = [
        _generate(chunk, tokenizer, model)
        for chunk in _split_for_model(line_text, tokenizer)
    ]
    return f"{marker}{' '.join(corrected)}"


def correct_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
) -> str:
    """Correct every non-empty line, leaving blank lines untouched."""
    corrected_lines = []
    for line in text.splitlines():
        if line.strip():
            corrected_lines.append(correct_line(line, tokenizer, model))
        else:
            corrected_lines.append(line)
    return "\n".join(corrected_lines)


def highlight_changes(
    original: str,
    corrected: str,
) -> list[tuple[str, str | None]]:
    """Return a token-level diff of `corrected` against `original`.

    Only text present in `corrected` is returned, so deleted words are not
    shown. Whitespace is emitted unhighlighted so a changed span never
    renders as an empty colored block.
    """
    original_tokens = TOKEN_RE.findall(original)
    corrected_tokens = TOKEN_RE.findall(corrected)
    matcher = SequenceMatcher(None, original_tokens, corrected_tokens)

    spans: list[tuple[str, str | None]] = []
    for tag, _i1, _i2, corrected_start, corrected_end in matcher.get_opcodes():
        segment = "".join(corrected_tokens[corrected_start:corrected_end])
        if tag == "equal":
            if segment:
                spans.append((segment, None))
            continue
        for piece in TOKEN_RE.findall(segment):
            spans.append((piece, None if piece.isspace() else CHANGED_LABEL))
    return spans


def check_and_rewrite(
    text: str,
) -> tuple[list[tuple[str, str | None]], str]:
    """Check `text` and return highlighted spans plus the corrected text.

    Whitespace-only input returns no spans and an empty result. Text the model
    leaves alone is returned unchanged, with no highlights.
    """
    if not text or not text.strip():
        return [], ""
    tokenizer, model = load_model()
    corrected = correct_text(text, tokenizer, model)
    return highlight_changes(text, corrected), corrected


def clear_all() -> tuple[str, list[tuple[str, str | None]], str]:
    """Reset the input, highlighted output, and corrected text."""
    return "", [], ""


def build_ui() -> gr.Blocks:
    """Build the Gradio interface."""
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
                    # A built-in palette name keeps the background fully opaque.
                    color_map={CHANGED_LABEL: CHANGED_COLOR},
                    # Don't draw the category badge after each highlighted span.
                    show_inline_category=False,
                )
                corrected_box = gr.Textbox(
                    label="Corrected text",
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
            fn=clear_all,
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

    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_ui().launch()


if __name__ == "__main__":
    main()
