from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


MODEL_PATH = "./results_v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)


special_tokens_dict = {'additional_special_tokens': ['<END>']}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

if num_added_tokens > 0:
    model.resize_token_embeddings(len(tokenizer))

end_token_id = tokenizer.convert_tokens_to_ids("<END>")


def gerar_treino(prompt, max_new_tokens=300, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=end_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    texto_gerado = tokenizer.decode(outputs[0], skip_special_tokens=True)

    texto_final = texto_gerado[len(prompt):].strip()

    if "<END>" in texto_final:
        texto_final = texto_final.split("<END>")[0].strip()

    return texto_final

prompt = "Me dê um treino para pernas."
resultado = gerar_treino(prompt)

print("📝 Resultado:")
print(resultado)
