from transformers import pipeline

def main():
    print("Downloading t5-small...")
    pipeline("summarization", model="t5-small", tokenizer="t5-small")

if __name__ == "__main__":
    main()
