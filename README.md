# T5 Text Checker & Rewriter

A simple Gradio app that checks text for spelling and grammar errors using the Hugging Face `visheratin/t5-efficient-mini-grammar-correction` model.

## Setup

Install the required packages:

```bash
pip install torch transformers gradio
```

## Run

```bash
python app.py
```

Open the local URL shown in the terminal. Enter text and select **Check & Rewrite** to receive a corrected version. The model downloads automatically the first time the app starts.

The app processes non-empty lines independently and preserves blank lines and common list markers.