
from transformers import pipeline
chatbot = pipeline("conversational", model="facebook/blenderbot-400M-distill")

chatbot("Hello, how are you?")