"""Etapa 4 - Roda a grade de experimentos (diferentes parâmetros) e avalia cada um.

Para cada configuração: treina o LoRA, avalia no teste e no fim gera a tabela
comparativa resultados/comparacao.md e os gráficos em resultados/*.png.

Uso:
    python src/experimentos.py                 (roda tudo)
    python src/experimentos.py --so exp1 exp3  (só alguns)
    python src/experimentos.py --so-tabela     (só refaz tabela/gráficos)
"""
import argparse
import json
import os
import subprocess
import sys

PY = sys.executable
EXPERIMENTOS = {
    # nome: parâmetros do treinar.py  -> o que cada um testa
    "exp1_lr2e-4_r16_ep2": ["--lr", "2e-4", "--rank", "16", "--epocas", "2"],   # referência
    "exp2_lr5e-5_r16_ep2": ["--lr", "5e-5", "--rank", "16", "--epocas", "2"],   # learning rate menor
    "exp3_lr2e-4_r4_ep2":  ["--lr", "2e-4", "--rank", "4", "--epocas", "2"],    # LoRA menor
    "exp4_lr2e-4_r16_ep1": ["--lr", "2e-4", "--rank", "16", "--epocas", "1"],   # menos épocas
    "exp5_lr1e-3_r16_ep2": ["--lr", "1e-3", "--rank", "16", "--epocas", "2"],   # learning rate alto
}


def rodar(cmd):
    print("\n>>> " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        # mostra só o fim do erro (o VSCode corta saídas longas no meio)
        print("\n".join(r.stderr.strip().splitlines()[-15:]))
        sys.exit(f"Falhou: {' '.join(cmd[1:4])}")


def tabela():
    linhas = []
    nomes = ["base"] + list(EXPERIMENTOS)
    for n in nomes:
        f = f"resultados/avaliacao_{n}.json"
        if not os.path.exists(f):
            continue
        r = json.load(open(f, encoding="utf-8"))
        h = {}
        if n != "base" and os.path.exists(f"runs/{n}/historico.json"):
            h = json.load(open(f"runs/{n}/historico.json", encoding="utf-8"))
        c = h.get("config", {})
        linhas.append({
            "nome": n, "lr": c.get("lr", "-"), "rank": c.get("rank", "-"), "epocas": c.get("epocas", "-"),
            "loss_val": round(h["validacao"][-1]["loss"], 4) if h else "-",
            "tempo_treino_s": h.get("tempo_treino_s", "-"),
            "json_valido": r["json_valido"], "categoria": r["acerto_por_campo"]["categoria"],
            "f1_macro": r["f1_macro_categoria"], "acerto_total": r["acerto_total"],
        })
    cab = list(linhas[0].keys()) if linhas else []
    md = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    md += ["| " + " | ".join(str(l[k]) for k in cab) + " |" for l in linhas]
    os.makedirs("resultados", exist_ok=True)
    open("resultados/comparacao.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))
    graficos(linhas)


def graficos(linhas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for n in EXPERIMENTOS:
        f = f"runs/{n}/historico.json"
        if os.path.exists(f):
            h = json.load(open(f, encoding="utf-8"))
            ax.plot([v["passo"] for v in h["validacao"]], [v["loss"] for v in h["validacao"]], marker="o", label=n)
    ax.set_xlabel("passo")
    ax.set_ylabel("loss de validacao")
    ax.set_title("Loss de validacao por experimento")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("resultados/loss_validacao.png", dpi=120)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    nomes = [l["nome"] for l in linhas]
    ax.bar(nomes, [l["acerto_total"] for l in linhas], color="#4C72B0")
    for i, l in enumerate(linhas):
        ax.text(i, l["acerto_total"] + 1, f"{l['acerto_total']}%", ha="center", fontsize=8)
    ax.set_ylabel("% respostas 100% corretas")
    ax.set_ylim(0, 110)
    ax.set_title("Acerto total no conjunto de teste")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig("resultados/acerto_total.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", nargs="*", default=None)
    ap.add_argument("--so-tabela", action="store_true")
    ap.add_argument("--pular-base", action="store_true")
    a = ap.parse_args()
    if not a.so_tabela:
        if not a.pular_base and not os.path.exists("resultados/avaliacao_base.json"):
            rodar([PY, "src/avaliar.py", "--nome", "base"])
        for nome, params in EXPERIMENTOS.items():
            if a.so and nome not in a.so:
                continue
            rodar([PY, "src/treinar.py", "--nome", nome] + params)
            rodar([PY, "src/avaliar.py", "--nome", nome, "--adaptador", f"runs/{nome}/adaptador"])
    tabela()


if __name__ == "__main__":
    main()
