"""Regras que geram o rótulo (resposta esperada) de cada linha de log.

Cada linha de WARNING/ERROR da integração Sienge <-> Hilti é classificada por
regex em uma categoria. A partir da categoria saem severidade, se o erro é
transitório e a ação sugerida. Esse é o "gabarito" usado para treinar e avaliar.
"""
import re

# categoria -> (sistema_afetado, severidade, transitorio, acao_sugerida)
CATEGORIAS = {
    "obra_fora_do_mapa": ("interno", "baixa", False,
        "Cadastrar a obra no obras.json se ela usar Hilti; caso contrario, ignorar."),
    "movimento_deletado": ("sienge", "baixa", False,
        "Nenhuma acao: o movimento foi excluido no Sienge antes de ser lido."),
    "saldo_insuficiente": ("sienge", "media", False,
        "Conferir o saldo do insumo nas CCs da obra no Sienge e lancar entrada ou transferencia."),
    "timeout": ("rede", "media", True,
        "Aguardar o proximo ciclo; se persistir, verificar a disponibilidade da API."),
    "token_hilti": ("hilti", "alta", True,
        "Verificar a conexao com o SSO da Hilti e as credenciais da integracao."),
    "acesso_negado": ("hilti", "alta", False,
        "Verificar permissoes e credenciais da API (HTTP 401/403)."),
    "servidor_indisponivel": ("rede", "media", True,
        "Erro no servidor da API; aguardar e reprocessar no proximo ciclo."),
    "limite_requisicoes": ("rede", "media", True,
        "Limite de requisicoes atingido (HTTP 429); reduzir o ritmo e reprocessar."),
    "conexao_falhou": ("rede", "media", True,
        "Falha de conexao com a API; verificar internet/DNS e reprocessar."),
    "apropriacao_invalida": ("sienge", "alta", False,
        "Corrigir o nivel de apropriacao do insumo no orcamento da obra no Sienge."),
    "apropriacao_percentual": ("sienge", "alta", False,
        "Ajustar as apropriacoes do insumo para somarem 100%."),
    "sem_custo_medio": ("sienge", "alta", False,
        "Lancar inicializacao ou entrada com custo antes da data do movimento."),
    "preco_unitario_ausente": ("sienge", "alta", False,
        "Informar o preco unitario no documento de entrada."),
    "cadastro_invalido": ("sienge", "alta", False,
        "Corrigir o cadastro do insumo (tipo ou unidade) no Sienge."),
    "sem_apropriacao_orcamento": ("sienge", "media", False,
        "Incluir o insumo no orcamento da obra ou apropria-lo manualmente."),
    "item_sem_mapa": ("interno", "baixa", False,
        "Adicionar o item na planilha de mapeamento SIENGE.xlsx."),
    "falha_item": ("sienge", "alta", False,
        "Verificar o erro anterior do lote e o cadastro do insumo no Sienge."),
    "retentativa_lote": ("interno", "baixa", True,
        "Nenhuma acao: o sistema reenvia o lote item a item."),
    "item_ja_existe": ("hilti", "baixa", False,
        "Nenhuma acao: o sistema busca o ID do item ja existente."),
    "falha_hilti_item": ("hilti", "alta", False,
        "Verificar o item no catalogo da Hilti (scanCode e unidade)."),
    "watchdog": ("interno", "media", True,
        "Ciclo travado ou pulado; confirmar que o proximo ciclo executa normalmente."),
    "requisicao_rejeitada": ("hilti", "media", False,
        "A API recusou os parametros da consulta (HTTP 422); revisar o filtro ou o ultimo ID processado."),
    "item_sem_scancode": ("hilti", "baixa", False,
        "Cadastrar o scanCode do item na Hilti ou ignorar."),
}

# Ordem importa: a primeira regra que casar define a categoria.
REGRAS = [
    (r"fora do mapa", "obra_fora_do_mapa"),
    (r"deletado sem backup", "movimento_deletado"),
    (r"saldo insuficiente", "saldo_insuficiente"),
    (r"Nível de apropriação inválido", "apropriacao_invalida"),
    (r"Porcentagem total de apropriação", "apropriacao_percentual"),
    (r"custo médio", "sem_custo_medio"),
    (r"preço unitário deve ser informado", "preco_unitario_ausente"),
    (r"Unidade de movimento não encontrada|tipo inv[aá]lido", "cadastro_invalido"),
    (r"WATCHDOG", "watchdog"),
    (r"token", "token_hilti"),
    (r"Too Many Requests|HTTP 429|\(429\)", "limite_requisicoes"),
    (r"HTTP 40[13]\b|\b40[13] Client Error", "acesso_negado"),
    (r"Timeout|timed out", "timeout"),
    (r"HTTP 422 ao buscar", "requisicao_rejeitada"),
    (r"HTTP 50[0234]|50[0234] Server Error|Bad Gateway|Service Unavailable|Internal Server Error", "servidor_indisponivel"),
    (r"Max retries exceeded|Connection aborted|ConnectionReset", "conexao_falhou"),
    (r"sem apropriacoes", "sem_apropriacao_orcamento"),
    (r"SEM MAPA", "item_sem_mapa"),
    (r"fallback per-item|retentando item a item", "retentativa_lote"),
    (r"\[FALHOU\]|\[ERRO\] \d+ \||lote\(s\) falharam", "falha_item"),
    (r"ja existe no catalogo", "item_ja_existe"),
    (r"Hilti recusou|obter ID do item", "falha_hilti_item"),
    (r"sem scanCode", "item_sem_scancode"),
]
_REGRAS = [(re.compile(p, re.I), c) for p, c in REGRAS]

_HTTP = re.compile(r"HTTP (\d{3})|\b(\d{3}) (?:Client|Server) Error|\((\d{3})\)")


def http_status(msg: str):
    m = _HTTP.search(msg)
    if not m:
        return None
    return int(next(g for g in m.groups() if g))


def rotular(origem: str, msg: str):
    """Retorna o dicionário-resposta para uma linha, ou None se nenhuma regra casar."""
    cat = next((c for rx, c in _REGRAS if rx.search(msg)), None)
    if cat is None:
        return None
    sistema, sev, trans, acao = CATEGORIAS[cat]
    return {
        "origem": origem.lower(),
        "categoria": cat,
        "sistema_afetado": sistema,
        "severidade": sev,
        "http_status": http_status(msg),
        "transitorio": trans,
        "acao_sugerida": acao,
    }
