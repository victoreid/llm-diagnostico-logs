"""Etapa 1 - Monta o dataset a partir dos logs reais da integração.

Lê motor.log*, porteiro.log* e auditor.log*, fica só com WARNING/ERROR,
anonimiza (URLs, tenant, caminhos), rotula com src/rotulos.py e divide em
treino / validação / teste.

Divisão temporal: o teste usa só linhas a partir de DATA_CORTE e cujo texto
não apareceu antes, para medir se o modelo generaliza para logs "novos".

Uso:
    python src/preparar_dados.py --logs "C:/caminho/Integração sienge new"
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from rotulos import rotular  # noqa: E402

LINHA = re.compile(r"^\[(\d{4}-\d{2}-\d{2}) [\d:]+\] \[(WARNING|ERROR) *\] (Motor|Porteiro|Auditor) - (.*)$")
MAX_CHARS = 320


def anonimizar(msg: str) -> str:
    def url(m):
        u = m.group(0)
        if "sienge" in u:
            return "<URL_SIENGE>"
        if "hilti" in u:
            return "<URL_HILTI>"
        return "<URL>"
    msg = re.sub(r"https?://\S+", url, msg)
    msg = re.sub(r"/frinstal/\S*", "/<tenant>/...", msg, flags=re.I)
    msg = re.sub(r"frinstal|fr instala\w*|exkalla", "<empresa>", msg, flags=re.I)
    msg = re.sub(r"[A-Za-z]:\\Users\\[^\s'\"]+", "<CAMINHO>", msg)
    msg = re.sub(r"\s+", " ", msg).strip()
    return msg[:MAX_CHARS]


def ler_logs(pasta):
    arquivos = [f for f in glob.glob(os.path.join(pasta, "*.log*"))
                if re.search(r"(motor|porteiro|auditor)\.log", os.path.basename(f))]
    for arq in sorted(arquivos):
        with open(arq, encoding="utf-8", errors="replace") as fh:
            for linha in fh:
                m = LINHA.match(linha.rstrip("\n"))
                if m:
                    data, nivel, origem, msg = m.groups()
                    yield data, nivel, origem, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="pasta com motor.log, porteiro.log, auditor.log")
    ap.add_argument("--saida", default="data")
    ap.add_argument("--data-corte", default="2026-09-15")
    ap.add_argument("--max-treino-por-cat", type=int, default=45)
    ap.add_argument("--max-teste-por-cat", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed)

    vistos = {}  # texto -> (data mais antiga, rotulo)
    sem_regra = collections.Counter()
    for data, nivel, origem, msg in ler_logs(a.logs):
        texto = f"[{nivel}] {origem} - {anonimizar(msg)}"
        if texto in vistos:
            vistos[texto] = (min(data, vistos[texto][0]), vistos[texto][1])
            continue
        rot = rotular(origem, msg)
        if rot is None:
            sem_regra[re.sub(r"\d+", "N", texto)[:90]] += 1
            continue
        vistos[texto] = (data, rot)

    antes = collections.defaultdict(list)
    depois = collections.defaultdict(list)
    for texto, (data, rot) in vistos.items():
        ex = {"entrada": texto, "saida": rot, "data": data}
        (depois if data >= a.data_corte else antes)[rot["categoria"]].append(ex)

    treino, val, teste = [], [], []
    for cat in sorted(set(antes) | set(depois)):
        a_lst, d_lst = antes[cat], depois[cat]
        random.shuffle(a_lst)
        random.shuffle(d_lst)
        # categorias raras sem exemplos "depois": separa um pedaço do "antes" para teste
        if not d_lst and len(a_lst) >= 4:
            d_lst, a_lst = a_lst[: max(1, len(a_lst) // 5)], a_lst[max(1, len(a_lst) // 5):]
        teste += d_lst[: a.max_teste_por_cat]
        n_val = max(1, min(5, len(a_lst) // 10)) if len(a_lst) >= 5 else 0
        val += a_lst[:n_val]
        treino += a_lst[n_val: n_val + a.max_treino_por_cat]

    os.makedirs(a.saida, exist_ok=True)
    for nome, lst in [("treino", treino), ("validacao", val), ("teste", teste)]:
        random.shuffle(lst)
        with open(os.path.join(a.saida, f"{nome}.jsonl"), "w", encoding="utf-8") as fh:
            for ex in lst:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    cont = {n: collections.Counter(e["saida"]["categoria"] for e in l)
            for n, l in [("treino", treino), ("validacao", val), ("teste", teste)]}
    resumo = {
        "linhas_unicas_rotuladas": len(vistos),
        "tamanho": {k: sum(v.values()) for k, v in cont.items()},
        "por_categoria": {c: {k: cont[k][c] for k in cont} for c in sorted(set().union(*cont.values()))},
        "sem_regra_top": sem_regra.most_common(15),
    }
    with open(os.path.join(a.saida, "resumo_dataset.json"), "w", encoding="utf-8") as fh:
        json.dump(resumo, fh, ensure_ascii=False, indent=2)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
