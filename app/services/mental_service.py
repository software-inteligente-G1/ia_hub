# app/services/mental_service.py
from transformers import T5Tokenizer, T5ForConditionalGeneration
from googletrans import Translator
from app.schemas.mental_schema import MentalHealthInput

# Inicializar el modelo y tokenizer
tokenizer = T5Tokenizer.from_pretrained("app/models/nlp")
model = T5ForConditionalGeneration.from_pretrained("app/models/nlp")
translator = Translator()


def get_mental_health_response(data: MentalHealthInput) -> dict:
    # Traducir al inglés
    input_es = data.message
    input_en = translator.translate(input_es, src='es', dest='en').text
    input_text = f"context: {input_en}"

    # Tokenizar e inferir
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=50,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9
    )
    response_en = tokenizer.decode(output[0], skip_special_tokens=True)

    # Traducir respuesta al español
    response_es = translator.translate(response_en, src='en', dest='es').text

    return {
        "input": input_es,
        "response": response_es
    }
