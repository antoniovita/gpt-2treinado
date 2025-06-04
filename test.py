from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_path = "./model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

result = pipe(
    "Quem descobriu o Brasil?",
    max_length=100,
    do_sample=True,
    temperature=0.7,
)

print(result[0]['generated_text'])
