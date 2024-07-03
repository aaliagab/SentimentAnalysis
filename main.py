from openai import OpenAI
from transformers import pipeline
import os

# Configuración de la API de OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Cargar el modelo preentrenado para análisis de sentimientos
sentiment_analysis = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    result = sentiment_analysis(text)
    return result[0]['label'], result[0]['score']

def get_custom_response(conversation_history):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation_history
    )
    return response.choices[0].message.content

def main():
    print("Bienvenido al asistente virtual de la tienda. ¿Cómo puedo ayudarte hoy?")
    
    # Mantener el historial de la conversación
    conversation_history = [
        {"role": "system", "content": "You are a helpful assistant for an online store. You must simulate that you have a wide variety of products and during the conversation with users feel free to generate links to products within the store based on the domain allshop.com"}
    ]
    
    while True:
        user_input = input("Tú: ")
        
        if user_input.lower() in ['salir', 'exit', 'quit']:
            print("Asistente: ¡Gracias por visitarnos! ¡Que tengas un buen día!")
            break
        
        sentiment, score = analyze_sentiment(user_input)
        
        # Agregar el input del usuario y el sentimiento al historial de la conversación
        conversation_history.append({"role": "user", "content": user_input})
        
        # Obtener la respuesta personalizada
        custom_response = get_custom_response(conversation_history)
        
        # Agregar la respuesta del asistente al historial de la conversación
        conversation_history.append({"role": "assistant", "content": custom_response})
        
        #print(f"Asistente (Sentimiento: {sentiment}, Confianza: {score:.2f}): {custom_response}")
        print(f"Asistente: {custom_response}")

if __name__ == "__main__":
    main()
