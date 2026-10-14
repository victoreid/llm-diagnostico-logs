"""Etapa 3 - Avaliação: compara a resposta do modelo com o gabarito do conjunto de teste.

Métricas:
  - json_valido: % de respostas que são um JSON legível
  - acerto por campo: % de acerto em cada chave (categoria, severidade, ...)
  - acerto_total: % de respostas com TODOS os campos corretos
  - f1_macro_categoria: F1 médio entre as categorias (penaliza errar as raras)

Uso:
    python src/avaliar.py --nome base                       (modelo sem treino)
    python src/avaliar.py --nome lr2e-4_r16 --adaptador runs/lr2e-4_r16/adaptador
"""
import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from comum import CAMPOS, carregar_modelo, extrair_json, gerar  # noqa: E402

CAMPOS_EXATOS = [c for c in CAMPOS if c != "acao_sugerida"]


def f1_macro(reais, previstos):
    f1s = []
    for c in set(reais):
        tp = sum(r == c and p == c for r, p in zip(reais, previstos))
        fp = sum(r != c and p == c for r, p in zip(reais, previstos))
        fn = sum(r == c and p != c for r, p in zip(reais, previstos))
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return sum(f1s) / len(f1s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", required=True)
    ap.add_argument("--adaptador", default=None)
    ap.add_argument("--teste", default="data/teste.jsonl")
    ap.add_argument("--limite", type=int, default=0)
    a = ap.parse_args()

    with open(a.teste, encoding="utf-8") as fh:
        teste = [json.loads(l) for l in fh]
    if a.limite:
        teste = teste[: a.limite]

    modelo, tok, disp = carregar_modelo(a.adaptador)
    print(f"Avaliando '{a.nome}' em {len(teste)} exemplos no dispositivo {disp}")

    acertos = collections.Counter()
    validos = 0
    todos = 0
    reais, previstos, registros = [], [], []
    t0 = time.time()
    for i, ex in enumerate(teste, 1):
        bruto = gerar(modelo, tok, ex["entrada"], disp)
        pred = extrair_json(bruto) or {}
        validos += bool(pred)
        ok = {c: pred.get(c) == ex["saida"][c] for c in CAMPOS}
        acertos.update(c for c, v in ok.items() if v)
        todos += all(ok[c] for c in CAMPOS_EXATOS)
        reais.append(ex["saida"]["categoria"])
        previstos.append(pred.get("categoria"))
        registros.append({"entrada": ex["entrada"], "esperado": ex["saida"], "resposta": bruto, "acertos": ok})
        if i % 10 == 0:
            print(f"{i}/{len(teste)} | {time.time() - t0:.0f}s", flush=True)

    n = len(teste)
    res = {
        "nome": a.nome,
        "adaptador": a.adaptador,
        "dispositivo": disp,
        "n": n,
        "json_valido": round(100 * validos / n, 1),
        "acerto_por_campo": {c: round(100 * acertos[c] / n, 1) for c in CAMPOS},
        "acerto_total": round(100 * todos / n, 1),
        "f1_macro_categoria": round(100 * f1_macro(reais, previstos), 1),
        "tempo_s": round(time.time() - t0, 1),
        "seg_por_resposta": round((time.time() - t0) / n, 2),
    }
    os.makedirs("resultados", exist_ok=True)
    with open(f"resultados/avaliacao_{a.nome}.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    with open(f"resultados/predicoes_{a.nome}.jsonl", "w", encoding="utf-8") as fh:
        for r in registros:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
