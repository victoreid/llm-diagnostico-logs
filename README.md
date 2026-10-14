# LLM local para diagnóstico de logs de integração (Sienge ↔ Hilti)

Fine-tuning de uma LLM pequena (**Qwen2.5-0.5B-Instruct**) com **LoRA**, treinada em **GPU NVIDIA (CUDA)**,
para ler uma linha de log de erro de uma integração real entre ERP e sistema de ferramentas e responder
um **diagnóstico estruturado em JSON**: categoria do erro, sistema afetado, severidade, código HTTP,
se o erro é transitório e qual ação tomar.

**Integrantes**

| Nome | RM |
|---|---|
| Victor Eid Carbutti Nicolas | RM98668 |
| Tiago Rafael Paulino Ferreira | RM551169 |

---

## 1. O problema

Uma empresa de instalações prediais mantém uma integração em Python que sincroniza o estoque entre o
ERP **Sienge** e o **Hilti ON!Track** em cerca de 70 obras. A integração tem três serviços:

- **Porteiro** recebe webhooks do Sienge e atualiza a Hilti.
- **Motor** lê as transferências da Hilti e lança saídas no Sienge.
- **Auditor** roda toda noite e corrige as divergências de estoque.

Entre junho e outubro de 2026 esses serviços geraram mais de **50 mil linhas de WARNING/ERROR**:
timeouts, erros HTTP 4xx/5xx, regras de negócio do Sienge (apropriação inválida, falta de custo médio),
falta de saldo e outros. Ler esses logs exige conhecer a integração. A ideia é que a LLM faça a triagem:

**Entrada (prompt)**
```
[ERROR] Motor - Sienge recusou CM: CC=1049 lote=1 (HTTP 422): {"status":422,"developerMessage":"Não há custo médio disponível para a data do movimento."}
```
**Saída esperada**
```json
{"origem": "motor", "categoria": "sem_custo_medio", "sistema_afetado": "sienge", "severidade": "alta",
 "http_status": 422, "transitorio": false,
 "acao_sugerida": "Lancar inicializacao ou entrada com custo antes da data do movimento."}
```

Por que esse problema é bom para uma LLM pequena:
- É **real**: os dados são o histórico de produção da integração.
- É **mensurável**: existe uma resposta certa para cada linha, então dá para calcular a % de acerto.
- O texto é **ruidoso**: há IDs variados, mensagens em JSON aninhado, stack traces e português misturado
  com inglês. Uma regex só funciona para padrões conhecidos, enquanto o modelo aprende a generalizar.

## 2. Escolha do modelo

| Critério | Qwen2.5-0.5B-Instruct |
|---|---|
| Tamanho | 0,5 bilhão de parâmetros (cerca de 1 GB em bf16) |
| Cabe em | GPU T4 de 16 GB com folga. Também roda na CPU para inferência. |
| Idioma | Multilíngue, entende português |
| Licença | Apache 2.0 |
| Por quê | Pequeno o bastante para treinar em minutos e servir localmente, e já é um modelo *instruct*, que segue formato de chat |

**Técnica: LoRA (Low-Rank Adaptation).** O modelo base fica congelado e só pequenas matrizes
adicionadas às camadas de atenção e MLP são treinadas: cerca de **1,75% dos parâmetros** com rank 16 (8,8 milhões de 503 milhões).
O resultado é um *adaptador* de poucos MB.

## 3. Dataset

O script é [`src/preparar_dados.py`](src/preparar_dados.py) e as regras ficam em [`src/rotulos.py`](src/rotulos.py).

1. Lê todos os `motor.log*`, `porteiro.log*` e `auditor.log*` e fica só com WARNING e ERROR.
2. **Anonimiza** os dados: troca URLs por `<URL_SIENGE>`/`<URL_HILTI>`, remove o nome do tenant e
   da empresa e tira os caminhos de usuário.
3. **Rotula automaticamente** com cerca de 25 regras regex. As regras definem a categoria (23 no total)
   e, a partir dela, a severidade, o sistema afetado, se é transitório e a ação.
4. Remove duplicatas: são 6.637 linhas únicas. Limita os exemplos por categoria para equilibrar as classes,
   porque só "obra fora do mapa" tem mais de 31 mil ocorrências.
5. **Divisão temporal:** o teste usa logs a partir de 15/09/2026 que não apareceram antes. Assim, o
   teste mede se o modelo generaliza para logs novos.

| Conjunto | Exemplos |
|---|---|
| Treino | 499 |
| Validação | 53 |
| Teste | 118 |

A distribuição por categoria está em [`data/resumo_dataset.json`](data/resumo_dataset.json).

## 4. Treino e experimentos

Os scripts são [`src/treinar.py`](src/treinar.py) (loop de treino em PyTorch + PEFT) e
[`src/experimentos.py`](src/experimentos.py). A loss é calculada **apenas sobre a resposta JSON**: os
tokens do prompt recebem label `-100`. Otimizador AdamW, schedule cosseno com warmup, batch 4 e bf16 na GPU.

| Experimento | lr | rank LoRA | épocas | O que testa |
|---|---|---|---|---|
| exp1 | 2e-4 | 16 | 2 | Configuração de referência |
| exp2 | 5e-5 | 16 | 2 | Learning rate menor |
| exp3 | 2e-4 | 4 | 2 | Adaptador menor (menos parâmetros) |
| exp4 | 2e-4 | 16 | 1 | Menos épocas |
| exp5 | 1e-3 | 16 | 2 | Learning rate alto |

## 5. Avaliação

O script é [`src/avaliar.py`](src/avaliar.py). Ele gera as respostas com decodificação gulosa
(temperatura 0) e compara com o gabarito.

- **json_valido:** % de respostas que são JSON legível.
- **acerto por campo:** % de acerto em categoria, severidade, sistema, http_status, transitório e ação.
- **acerto_total:** % de respostas com todos os campos corretos.
- **f1_macro_categoria:** F1 médio por categoria, que penaliza errar as classes raras.

### Resultados

Resultados no conjunto de teste (118 logs a partir de 15/09/2026 que o modelo nunca viu), medidos
na GPU Tesla T4. A tabela também está em [`resultados/comparacao.md`](resultados/comparacao.md).

| Modelo | lr | rank | épocas | loss val. | treino (s) | JSON válido | categoria | F1 macro | **acerto total** |
|---|---|---|---|---|---|---|---|---|---|
| base (sem treino) | - | - | - | - | - | 100% | 0,0% | 0,0% | **0,0%** |
| exp1 (referência) | 2e-4 | 16 | 2 | 0,0006 | 214 | 100% | 94,1% | 86,8% | **94,1%** |
| exp2 (lr menor) | 5e-5 | 16 | 2 | 0,0038 | 219 | 100% | 93,2% | 87,9% | **93,2%** |
| exp3 (LoRA menor) | 2e-4 | 4 | 2 | 0,0024 | 217 | 100% | 93,2% | 87,9% | **93,2%** |
| exp4 (1 época) | 2e-4 | 16 | 1 | 0,0023 | 110 | 100% | 94,1% | 87,1% | **93,2%** |
| **exp5 (lr alto)** | 1e-3 | 16 | 2 | 0,0003 | 219 | 100% | **99,2%** | **99,5%** | **97,5%** 🏆 |

![tabela e loss](docs/print_resultados_tabela.png)
![acerto total](docs/print_resultados_grafico.png)

### Análise

- **O fine-tuning é o que resolve o problema.** O modelo base já devolve JSON, mas inventa
  categorias (ex.: "Item"), escreve a origem com maiúscula ("Porteiro") e erra o campo
  `transitorio` em 98% dos casos. Por isso acerta 0% das respostas completas. Com cerca de **3,5 minutos de treino na T4**
  e só **1,75% dos parâmetros** treinados, o acerto total sobe para 93–97%.
- **Learning rate foi o parâmetro de maior impacto.** O lr 1e-3 (exp5) deu o melhor resultado
  (97,5%, F1 99,5%). Durante o treino a loss oscilou bastante, porque um lr alto "passa do ponto"
  a cada passo, mas como o LoRA mexe em poucos parâmetros e o schedule cosseno reduz o lr no final,
  o modelo convergiu melhor. O lr 5e-5 (exp2) aprendeu mais devagar: a loss de validação ficou 6x
  maior que no exp1 depois de 2 épocas.
- **Rank 4 contra rank 16:** com 4x menos parâmetros treináveis, o adaptador perdeu menos de 1 ponto
  de acerto. Para uma tarefa de formato fixo, um LoRA pequeno já basta.
- **1 contra 2 épocas:** com 1 época (metade do tempo, 110 s) o resultado foi praticamente igual.
  O dataset é pequeno e repetitivo, e a maior parte do aprendizado acontece nos primeiros 50 passos.
- **Onde os modelos erram:** nos exp1–4 o F1 macro (cerca de 87%) fica abaixo da acurácia de categoria
  (cerca de 94%). Isso indica que os erros se concentram nas categorias com poucos exemplos (no teste,
  algumas têm só 1 caso, como `sem_custo_medio` e `item_sem_scancode`), já que o F1 macro dá o mesmo
  peso a cada categoria. O exp5 chegou a 99,5% de F1 macro, ou seja, acertou também as raras.

Exemplos de resposta do melhor modelo (exp5):

![exemplos](docs/print_exemplos_respostas.png)

### Limitações
- Os rótulos vêm de regras regex, então o modelo aprende a imitar essas regras. A vantagem sobre a
  regex é lidar com variações de texto, mas um log de um tipo realmente novo não terá categoria certa.
- O dataset é pequeno (499 exemplos de treino) e desbalanceado nas categorias raras.
- O teste é temporal (logs posteriores a 15/09), mas muitas mensagens seguem o mesmo template e mudam
  só os números, o que deixa a tarefa mais fácil que um cenário aberto.

## 6. GPU

Todo o treino e a avaliação rodaram em **GPU NVIDIA Tesla T4 (16 GB, CUDA 13.0)** no Google Colab,
**conectado ao VSCode** pela extensão oficial *Google Colab*, conforme autorizado pelo professor.

**`nvidia-smi` dentro do VSCode:**

![GPU no VSCode](docs/print_gpu_vscode.png)

**PyTorch usando CUDA:** a T4 multiplicou matrizes **mais de 20x mais rápido** que a CPU do
Colab (3,2 contra 0,12 TFLOPS):

![verificar gpu](docs/print_verificar_gpu.png)

**Treino rodando na GPU** (cerca de 0,8 s por passo; na CPU do notebook de desenvolvimento eram cerca de 40 s por passo):

![treino](docs/print_treino_gpu.png)

**Avaliação do modelo base na GPU** (118 respostas em 24 s, geradas em lotes de 16):

![base](docs/print_avaliacao_base.png)

A T4 (compute capability 7.5) **não tem bfloat16 nativo**, só emulado, o que é lento. Por isso o código
usa float16 na inferência e float32 no treino nessa placa, e bfloat16 em GPUs Ampere ou mais novas
(veja `dtype_para` em [`src/comum.py`](src/comum.py)).

O script [`src/verificar_gpu.py`](src/verificar_gpu.py) mostra a versão de PyTorch/CUDA, o nome e a
memória da GPU e compara a velocidade de multiplicação de matrizes na GPU e na CPU.

## 7. Frontend

O frontend fica em [`app/app.py`](app/app.py) e usa **Gradio**. O usuário cola ou digita uma linha de log,
escolhe o modelo (base ou um dos adaptadores treinados) e ajusta a temperatura e o máximo de tokens.
A tela mostra o diagnóstico em JSON, a resposta bruta e o tempo de geração. No topo aparece a GPU em uso.

No print abaixo o frontend roda na **Tesla T4 (CUDA)** pelo Colab, com o melhor modelo (exp5). Ele
diagnosticou corretamente um log que **não estava no dataset** (CC 1072, lote 2) em 3,5 s:

![frontend](docs/print_frontend.jpg)

## 8. Como rodar

### Na GPU (Colab dentro do VSCode)
1. No VSCode, instale as extensões **Jupyter** e **Google Colab**.
2. Abra [`notebooks/treino_colab.ipynb`](notebooks/treino_colab.ipynb), clique em *Select Kernel → Colab →
   New Colab Server → GPU T4* e faça login com uma conta Google.
3. Rode as células em ordem. A célula `!nvidia-smi` mostra a GPU, e o fim do notebook baixa o
   `resultados_colab.zip`.

### Localmente
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU NVIDIA
# ou: pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python src/verificar_gpu.py
python src/avaliar.py --nome base
python src/experimentos.py
python app/app.py               # http://127.0.0.1:7860
```

> **CUDA:** a versão do CUDA no PyTorch precisa ser compatível com o driver da placa. Rode `nvidia-smi`
> e veja "CUDA Version" no canto superior direito: é a versão máxima suportada pelo driver. Escolha uma
> roda do PyTorch igual ou menor (cu118, cu121, cu124...).

## 9. Estrutura

```
├── app/app.py                 frontend Gradio
├── data/                      dataset anonimizado (treino/validacao/teste)
├── notebooks/treino_colab.ipynb
├── resultados/                métricas, predições, tabela e gráficos
├── runs/<experimento>/        adaptador LoRA + histórico de loss
└── src/
    ├── verificar_gpu.py       etapa 0 - GPU
    ├── preparar_dados.py      etapa 1 - dataset
    ├── rotulos.py             regras de rotulagem
    ├── treinar.py             etapa 2 - fine-tuning LoRA
    ├── avaliar.py             etapa 3 - métricas
    ├── experimentos.py        etapa 4 - grade de parâmetros
    └── comum.py               prompt, dispositivo, carregamento
```
