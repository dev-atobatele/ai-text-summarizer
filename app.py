import os
import sqlite3
from flask import Flask, request, render_template, flash, url_for, redirect

# Import the summarization pipeline from transformers
from transformers import pipeline

app = Flask(__name__)
app.secret_key = "replace-with-a-random-secret-key"   # needed for flash messages

# Initialize the summarizer once (loads the model on startup)
SUMMARIZER = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",   # small, fast, works on CPU
    tokenizer="sshleifer/distilbart-cnn-12-6",
    device=-1,                    # -1 = CPU
    max_length=120,   # don't let it write novels
    min_length=30,    # make sure it's not too short
    do_sample=False   # deterministic, less random memory spikes
)

DB_PATH = "summaries.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return conn


@app.route("/", methods=["GET"])
def index():
    # Show the form and the 5 most recent summaries (optional)
    conn = get_db()
    recent = conn.execute(
        "SELECT original, summary FROM logs ORDER BY created_at DESC LIMIT 5"
    ).fetchall()
    conn.close()
    return render_template("index.html", recent=recent)


@app.route("/summarize", methods=["POST"])
def summarize():
    text = request.form.get("text", "").strip()
    if not text:
        flash("Please provide some text to summarize.")
        return redirect(url_for("index"))

    # Truncate to a reasonable length for the small model (e.g., 300 words)
    # Most small models have a max input length of 512 tokens.
    max_len = 500
    truncated = text[:max_len]

    # Generate summary
    result = SUMMARIZER(truncated, max_length=130, min_length=30, do_sample=False)
    summary = result[0]["summary_text"]

    # Store in DB (optional, for demo)
    conn = get_db()
    conn.execute(
        "INSERT INTO logs (original, summary) VALUES (?, ?)", (truncated, summary)
    )
    conn.commit()
    conn.close()

    flash("Summary generated!")
    return render_template("index.html", original=truncated, summary=summary)


if __name__ == "__main__":
    # For local testing only; Render uses gunicorn
    app.run(debug=True)




