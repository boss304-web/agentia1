import logging
import os
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
import autogen

# 📦 1. Charger variables d'environnement (.env pour OpenRouter et Telegram)
load_dotenv()

# 🔐 2. Récupérer le TOKEN TELEGRAM depuis l'environnement
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# 📡 3. Configuration DeepSeek (via OpenRouter)
config_deepseek = {
    "config_list": [
        {
            "model": "deepseek-coder:1.3b-instruct-r1",
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "base_url": "https://openrouter.ai/api/v1",
            "headers": {
                "HTTP-Referer": "https://localhost",  # Obligatoire OpenRouter
                "X-Title": "BotTelegramCodeur"
            }
        }
    ],
    "temperature": 0.5,
    "timeout": 60,
    "seed": 42
}

# 🧠 4. Définir ton agent IA
agent_codeur = autogen.AssistantAgent(
    name="CodeurDeepSeek",
    system_message="""
    Tu es un assistant IA expert en programmation. 
    Tu génères du code (Python, JS, HTML...) et tu expliques tout clairement.
    Tu réponds toujours en français, avec des commentaires dans le code.
    """,
    llm_config=config_deepseek
)

# 👤 Agent utilisateur Telegram
user_proxy = autogen.UserProxyAgent(
    name="UtilisateurTelegram",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    code_execution_config=False
)

# 📩 Fonction de réponse aux messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    print(f"📩 Message reçu : {question}")

    try:
        loop = asyncio.get_running_loop()
        resultat = await loop.run_in_executor(
            None,
            lambda: user_proxy.initiate_chat(
                agent_codeur,
                message=question,
                clear_history=True
            )
        )
        contenu = resultat.chat_history[-1]["content"]

        # Répondre en morceaux si c'est trop long
        for i in range(0, len(contenu), 4000):
            await update.message.reply_text(contenu[i:i+4000])

    except Exception as e:
        await update.message.reply_text("❌ Erreur lors de la génération.")
        print("Erreur:", e)

# 📥 Commande /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Bienvenue ! Pose-moi ta question de code.")

# 🚀 Lancer le bot
if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Ton bot Telegram est en ligne !")
    app.run_polling()