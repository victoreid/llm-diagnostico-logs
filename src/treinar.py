"""Etapa 2 - Fine-tuning com LoRA.

O modelo base fica congelado; treinamos só pequenas matrizes LoRA (poucos MB).
A loss é calculada apenas sobre a resposta (JSON), não sobre o prompt.

Uso:
    python src/treinar.py --nome lr2e-4_r16 --lr 2e-4 --rank 16 --epocas 2
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from comum import MODELO_BASE, dtype_para, escolher_dispositivo, mensagens  # noqa: E402


def ler(caminho):
    with open(caminho, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def tokenizar(tok, ex, max_len):
    """input_ids do diálogo completo; labels = -100 no prompt (não entra na loss)."""
    prompt = tok.apply_chat_template(mensagens(ex["entrada"]), tokenize=False, add_generation_prompt=True)
    completo = tok.apply_chat_template(mensagens(ex["entrada"], ex["saida"]), tokenize=False)
    ids_prompt = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(completo, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * len(ids_prompt) + ids[len(ids_prompt):]
    return ids, labels[: len(ids)]


def lote(itens, pad_id, dispositivo):
    n = max(len(i) for i, _ in itens)
    ids = torch.full((len(itens), n), pad_id)
    lab = torch.full((len(itens), n), -100)
    att = torch.zeros((len(itens), n), dtype=torch.long)
    for k, (i, l) in enumerate(itens):
        ids[k, : len(i)] = torch.tensor(i)
        lab[k, : len(l)] = torch.tensor(l)
        att[k, : len(i)] = 1
    return ids.to(dispositivo), att.to(dispositivo), lab.to(dispositivo)


@torch.no_grad()
def loss_validacao(modelo, dados, pad_id, dispositivo, bs):
    modelo.eval()
    total, n = 0.0, 0
    for k in range(0, len(dados), bs):
        ids, att, lab = lote(dados[k: k + bs], pad_id, dispositivo)
        total += modelo(input_ids=ids, attention_mask=att, labels=lab).loss.item()
        n += 1
    modelo.train()
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", required=True)
    ap.add_argument("--dados", default="data")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=None, help="padrão: 2 x rank")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--epocas", type=float, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--max-exemplos", type=int, default=0, help="0 = todos")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    disp = escolher_dispositivo()
    print(f"Dispositivo: {disp}")
    if disp == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tok = AutoTokenizer.from_pretrained(MODELO_BASE)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO_BASE, dtype=dtype_para(disp)).to(disp)
    cfg = LoraConfig(
        r=a.rank, lora_alpha=a.alpha or 2 * a.rank, lora_dropout=a.dropout, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    modelo = get_peft_model(modelo, cfg)
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modelo.parameters())
    print(f"Parametros treinaveis: {treinaveis:,} de {total:,} ({100 * treinaveis / total:.2f}%)")

    treino = ler(os.path.join(a.dados, "treino.jsonl"))
    if a.max_exemplos:
        treino = treino[: a.max_exemplos]
    val = ler(os.path.join(a.dados, "validacao.jsonl"))
    treino_t = [tokenizar(tok, e, a.max_len) for e in treino]
    val_t = [tokenizar(tok, e, a.max_len) for e in val]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    passos_epoca = math.ceil(len(treino_t) / a.batch)
    passos = int(passos_epoca * a.epocas)
    opt = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, passos // 20), passos)

    hist = {"config": vars(a), "dispositivo": disp, "treinaveis": treinaveis, "passos": [], "validacao": []}
    hist["validacao"].append({"passo": 0, "loss": loss_validacao(modelo, val_t, pad, disp, a.batch)})
    print(f"passo 0 | loss validacao {hist['validacao'][-1]['loss']:.4f}")

    modelo.train()
    t0 = time.time()
    passo = 0
    while passo < passos:
        random.shuffle(treino_t)
        for k in range(0, len(treino_t), a.batch):
            if passo >= passos:
                break
            ids, att, lab = lote(treino_t[k: k + a.batch], pad, disp)
            loss = modelo(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            passo += 1
            hist["passos"].append({"passo": passo, "loss": loss.item(), "lr": sched.get_last_lr()[0]})
            if passo % 10 == 0 or passo == passos:
                print(f"passo {passo}/{passos} | loss {loss.item():.4f} | {time.time() - t0:.0f}s", flush=True)
            if passo % passos_epoca == 0 or passo == passos:
                v = loss_validacao(modelo, val_t, pad, disp, a.batch)
                hist["validacao"].append({"passo": passo, "loss": v})
                print(f"  -> loss validacao {v:.4f}")

    hist["tempo_treino_s"] = round(time.time() - t0, 1)
    if disp == "cuda":
        hist["vram_pico_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    saida = os.path.join("runs", a.nome)
    os.makedirs(saida, exist_ok=True)
    modelo.save_pretrained(os.path.join(saida, "adaptador"))
    with open(os.path.join(saida, "historico.json"), "w", encoding="utf-8") as fh:
        json.dump(hist, fh, ensure_ascii=False, indent=2)
    print(f"Salvo em {saida} ({hist['tempo_treino_s']}s)")


if __name__ == "__main__":
    main()
