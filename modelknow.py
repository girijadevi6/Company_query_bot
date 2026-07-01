import google.generativeai as genai

genai.configure(api_key="AIzaSyBTA3lUX_k5FVWpBeaDhS5ZuvT6DYwl_E8")

for m in genai.list_models():
    print(m.name)