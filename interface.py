from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr


MODEL_PATH = "./results_v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)

special_tokens_dict = {'additional_special_tokens': ['<END>']}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

if num_added_tokens > 0:
    model.resize_token_embeddings(len(tokenizer))

end_token_id = tokenizer.convert_tokens_to_ids("<END>")


def gerar_treino(grupo_muscular, objetivo, max_new_tokens=300, temperature=0.7, top_p=0.9):
    prompt = f"Me dê um treino para {grupo_muscular} focado em {objetivo}."

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

gr_interface = gr.Interface(
    fn=gerar_treino,
    inputs=[
        gr.Dropdown(
            ["peito e tríceps", "costas e bíceps", "pernas", "ombro e abdômen", "funcional"],
            label="Selecione o grupo muscular"
        ),
        gr.Dropdown(
            ["hipertrofia", "força", "emagrecimento", "resistência", "iniciante", "avançado"],
            label="Selecione o objetivo do treino"
        )
    ],
    outputs=gr.Textbox(label="Treino Gerado"),
    title="Gerador de Treinos Inteligente 💪🧠",
    description="Escolha o grupo muscular e o objetivo (hipertrofia, força, emagrecimento, resistência, iniciante, avançado) e gere um treino completo com IA."
)
0,
gr_interface.launch(share=True)
