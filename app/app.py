"""Frontend (Gradio): o usuário cola uma linha de log e o modelo devolve o diagnóstico.

Uso:
    python app/app.py
    (abre em http://127.0.0.1:7860)
"""
import glob
import json
import os
import sys
import time

import gradio as gr
import torch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))
from comum import carregar_modelo, extrair_json, gerar  # noqa: E402

EXEMPLOS = [
    "[ERROR] Motor - Timeout ao buscar transferencias Hilti",
    "[WARNING] Porteiro - Obra 1123 fora do mapa - ignorando",
    "[ERROR] Motor - Sienge recusou CM: CC=1049 lote=1 (HTTP 422): {\"status\":422,\"developerMessage\":\"Não há custo médio disponível para a data do movimento.\"}",
    "[WARNING] Motor - [AVISO] Item 25715 | Faltou baixar 3.0 de 5.0 (saldo insuficiente na familia da obra 1058)",
    "[ERROR] Auditor - Erro ao ler alocacao item 3321: 429 Client Error: Too Many Requests for url: <URL_HILTI>",
    "[ERROR] Porteiro - HTTP 502 ao ler movimento 455091",
]

_cache = {}


def opcoes_modelo():
    ops = ["base (sem treino)"]
    ops += sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(RAIZ, "runs", "*", "adaptador")))
    return ops


def obter(nome):
    if nome not in _cache:
        _cache.clear()  # mantém só um modelo na memória
        adapt = None if nome.startswith("base") else os.path.join(RAIZ, "runs", nome, "adaptador")
        _cache[nome] = carregar_modelo(adapt)
    return _cache[nome]


def info_gpu():
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return f"GPU: {p.name} ({p.total_memory / 1e9:.1f} GB) | CUDA {torch.version.cuda}"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return f"GPU Intel: {torch.xpu.get_device_name(0)}"
    return "Sem GPU - rodando na CPU"


def diagnosticar(log, nome, temperatura, max_tokens):
    if not log.strip():
        return "", {}, ""
    modelo, tok, disp = obter(nome)
    t0 = time.time()
    bruto = gerar(modelo, tok, log.strip(), disp, max_novos=int(max_tokens), temperatura=temperatura)
    dt = time.time() - t0
    js = extrair_json(bruto)
    return bruto, js or {"erro": "a resposta nao e um JSON valido"}, f"{dt:.2f}s em {disp}"


with gr.Blocks(title="Diagnostico de logs - LLM local") as demo:
    gr.Markdown("# Diagnostico de logs da integracao Sienge <-> Hilti\n"
                "LLM local (Qwen2.5-0.5B + LoRA) treinada com o historico real da integracao.")
    gr.Markdown(f"**{info_gpu()}**")
    with gr.Row():
        with gr.Column():
            log = gr.Textbox(label="Linha de log (prompt)", lines=4,
                             placeholder="[ERROR] Motor - Timeout ao buscar transferencias Hilti")
            modelo = gr.Dropdown(opcoes_modelo(), value=opcoes_modelo()[-1], label="Modelo")
            with gr.Accordion("Parametros de geracao", open=False):
                temp = gr.Slider(0, 1.5, value=0, step=0.1, label="Temperatura (0 = deterministico)")
                maxt = gr.Slider(32, 400, value=160, step=8, label="Max. tokens de resposta")
            btn = gr.Button("Diagnosticar", variant="primary")
            gr.Examples(EXEMPLOS, inputs=log)
        with gr.Column():
            js = gr.JSON(label="Diagnostico (JSON)")
            bruto = gr.Textbox(label="Resposta bruta do modelo", lines=6)
            tempo = gr.Markdown()
    btn.click(diagnosticar, [log, modelo, temp, maxt], [bruto, js, tempo])
    log.submit(diagnosticar, [log, modelo, temp, maxt], [bruto, js, tempo])

if __name__ == "__main__":
    demo.launch(share="--share" in sys.argv)  # --share gera link público (útil no Colab)
