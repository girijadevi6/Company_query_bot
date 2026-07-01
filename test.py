import google.generativeai as genai

# Set API key
genai.configure(api_key="AIzaSyBTA3lUX_k5FVWpBeaDhS5ZuvT6DYwl_E8")

# Choose model
model = genai.GenerativeModel("gemini-2.5-flash")

# Send prompt
response = model.generate_content("Explain Bayes theorem in simple terms.")

# Print output
print(response.text)