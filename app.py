from flask import Flask, request, render_template, redirect, url_for, flash
from transformers import pipeline
import sqlite3

app = Flask(__name__)
app.secret_key = "replace-this-secret"

# ✅ Use t5-small (lightweight and guaranteed to fit Render free tier)
SUMMARIZER = pipeline("summarization", model="t5-small", tokenizer="t5-small", device=-1)

DB_PATH = "summaries.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    return conn

@app.route("/")
def index():
    conn = get_db()
    recent = conn.execute("SELECT original, summary FROM logs ORDER BY created_at DESC LIMIT 5").fetchall()
    conn.close()
    return render_template("index.html", recent=recent)

@app.route("/summarize", methods=["POST"])
def summarize():
    text = request.form.get("text", "").strip()
    if not text:
        flash("Please provide some text to summarize.")
        return redirect(url_for("index"))

    truncated = text[:1000]  # limit size for free-tier stability
    result = SUMMARIZER(truncated, max_length=100, min_length=25, do_sample=False)
    summary = result[0]["summary_text"]

    conn = get_db()
    conn.execute("INSERT INTO logs (original, summary) VALUES (?, ?)", (truncated, summary))
    conn.commit()
    conn.close()

    return render_template("index.html", original=truncated, summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
