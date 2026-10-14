"""Funções compartilhadas: escolha de dispositivo (GPU), prompt e carregamento do modelo."""
import json
import re

import torch

MODELO_BASE = "Qwen/Qwen2.5-0.5B-Instruct"

SISTEMA = (
    "Voce diagnostica linhas de log da integracao Sienge <-> Hilti ON!Track. "
    "Responda APENAS um JSON com as chaves: origem, categoria, sistema_afetado "
    "(sienge|hilti|rede|interno), severidade (baixa|media|alta), http_status "
    "(numero ou null), transitorio (true|false) e acao_sugerida."
)

CAMPOS = ["origem", "categoria", "sistema_afetado", "severidade", "http_status", "transitorio", "acao_sugerida"]


def escolher_dispositivo():
    """GPU NVIDIA (cuda) > GPU Intel (xpu) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def dtype_para(dispositivo, treino=False):
    """Precisão numérica por placa.

    GPUs Ampere+ (compute 8.x, ex.: RTX 30xx, A100) têm bfloat16 nativo.
    GPUs mais antigas (ex.: Tesla T4, compute 7.5) só EMULAM bf16, o que é lento:
    nelas usamos float16 para gerar e float32 para treinar (fp16 puro no AdamW
    pode estourar/zerar gradientes).
    """
    if dispositivo == "cuda":
        if torch.cuda.get_device_capability(0)[0] >= 8:
            return torch.bfloat16
        return torch.float32 if treino else torch.float16
    if dispositivo == "xpu":
        return torch.bfloat16
    return torch.float32


def mensagens(entrada, resposta=None):
    msgs = [{"role": "system", "content": SISTEMA}, {"role": "user", "content": entrada}]
    if resposta is not None:
        msgs.append({"role": "assistant", "content": json.dumps(resposta, ensure_ascii=False)})
    return msgs


def carregar_modelo(adaptador=None, dispositivo=None):
    """Carrega o modelo base e, se informado, o adaptador LoRA treinado."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dispositivo = dispositivo or escolher_dispositivo()
    tok = AutoTokenizer.from_pretrained(MODELO_BASE)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO_BASE, dtype=dtype_para(dispositivo))
    if adaptador:
        from peft import PeftModel
        modelo = PeftModel.from_pretrained(modelo, adaptador)
        modelo = modelo.merge_and_unload()
    modelo.to(dispositivo).eval()
    return modelo, tok, dispositivo


@torch.no_grad()
def gerar(modelo, tok, entrada, dispositivo, max_novos=160, temperatura=0.0):
    texto = tok.apply_chat_template(mensagens(entrada), tokenize=False, add_generation_prompt=True)
    ids = tok(texto, return_tensors="pt").to(dispositivo)
    kwargs = dict(max_new_tokens=max_novos, pad_token_id=tok.eos_token_id)
    if temperatura > 0:
        kwargs.update(do_sample=True, temperature=temperatura, top_p=0.9)
    else:
        kwargs.update(do_sample=False)
    out = modelo.generate(**ids, **kwargs)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def gerar_lote(modelo, tok, entradas, dispositivo, max_novos=160):
    """Gera várias respostas de uma vez (muito mais rápido na GPU). Decodificação gulosa."""
    tok.padding_side = "left"  # modelos decoder-only precisam de padding à esquerda para gerar
    textos = [tok.apply_chat_template(mensagens(e), tokenize=False, add_generation_prompt=True) for e in entradas]
    ids = tok(textos, return_tensors="pt", padding=True).to(dispositivo)
    out = modelo.generate(**ids, max_new_tokens=max_novos, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    n = ids["input_ids"].shape[1]
    return [tok.decode(o[n:], skip_special_tokens=True).strip() for o in out]


def extrair_json(texto):
    """Tenta ler o JSON da resposta (aceita texto em volta ou bloco ```json)."""
    m = re.search(r"\{.*\}", texto, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
