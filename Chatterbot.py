import os
import pandas as pd
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

# 1) Remove any existing DB so we always start fresh
DB_PATH = "mybot.sqlite3"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

# 2) Create & configure the bot
chatbot = ChatBot(
    "MyCustomBot",
    storage_adapter="chatterbot.storage.SQLStorageAdapter",
    database_uri=f"sqlite:///{DB_PATH}",
    logic_adapters=["chatterbot.logic.BestMatch"]
)
trainer = ListTrainer(chatbot)

# 3) Load your CSV and train
#    Assumes columns named exactly "Prompt" and "Response"
df = pd.read_csv("data.csv")
df = df.dropna(subset=["Prompt", "Response"])

print(f"Training on {len(df)} pairs…")
for prompt, response in zip(df["Prompt"], df["Response"]):
    trainer.train([str(prompt).strip(), str(response).strip()])

print("Training complete!\n")

# 4) Interactive query loop
print("You can now chat with the bot. Type 'exit' or 'quit' to stop.")
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    bot_reply = chatbot.get_response(user_input)
    print("Bot:", bot_reply)
