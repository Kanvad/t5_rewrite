# T5 Text Checker & Rewriter

A simple Gradio app that checks text for spelling and grammar errors using the Hugging Face `visheratin/t5-efficient-mini-grammar-correction` model.

## Setup

Requires Python 3.10 or newer. Install the required packages:

```bash
pip install -r requirements.txt
```

The model repository ships custom tokenizer and architecture code, so the app loads it with `trust_remote_code=True`.

## Run

```bash
python app.py
```

Open the local URL shown in the terminal. Enter text and select **Check & Rewrite** to receive a corrected version. Inference runs on CPU, and the model downloads automatically the first time the app starts.

The app processes non-empty lines independently and preserves blank lines and common list markers.

## License

MIT — see [LICENSE](LICENSE).
