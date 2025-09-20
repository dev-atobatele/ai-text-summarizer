from transformers import pipeline

def main():
    print("Downloading summarization model...")
    pipeline("summarization",
             model="sshleifer/distilbart-cnn-12-6",
             tokenizer="sshleifer/distilbart-cnn-12-6")

if __name__ == "__main__":
    main()
