import os
import google.generativeai as genai
import os

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Explain transformers in simple words")

print(response.text)
