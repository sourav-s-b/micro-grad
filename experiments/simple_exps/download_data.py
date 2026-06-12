from datasets import load_dataset

print("Downloading DailyDialog (Parquet format) from Hugging Face...")
# Pointing to DeepPavlov's safe Parquet version of the exact same dataset
dataset = load_dataset("DeepPavlov/daily_dialog", split="train")

print("Formatting into dialogs.txt...")
with open("dialogs.txt", "w", encoding="utf-8") as f:
    for dialog in dataset["dialog"]:
        # 'dialog' is a list of back-and-forth turns.
        for i in range(len(dialog) - 1):
            user_text = dialog[i].strip()
            bot_text = dialog[i + 1].strip()

            # Skip empty lines to keep our data clean
            if user_text and bot_text:
                f.write(f"User: {user_text}\n")
                f.write(f"Bot: {bot_text}\n\n")

print("Success! 'dialogs.txt' is ready for your Causal Transformer.")
