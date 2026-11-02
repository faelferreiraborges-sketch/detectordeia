#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detector_ia — detector de texto gerado ou revisado por IA, pensado para o português.

Reúne num só programa os métodos por trás das ferramentas mais usadas:

  Método                          Ferramenta de origem          Modo
  ------------------------------  ----------------------------  -----------
  Perplexidade + burstiness       GPTZero                       neural
  Posição do token (top-k)        GLTR                          neural
  Curvatura da probabilidade      DetectGPT / Fast-DetectGPT    neural
  Perplexidade cruzada            Binoculars                    neural
  Compressão contra referências   ZipPy                         estatístico
  Classificador estilométrico     Ghostbuster, Pangram e cia.*  estatístico
  (* classificadores supervisionados sobre características do texto)

Modo estatístico (padrão): roda sem internet e sem GPU, em segundos.
Modo neural (--neural): baixa um par de modelos Qwen2.5 do Hugging Face (base + instruct),
  autocalibra os limiares com os textos de referência e acrescenta as medidas neurais.

Uso:
  python detector_ia.py analisar meu_texto.pdf
  python detector_ia.py analisar meu_texto.docx --neural
  python detector_ia.py texto < arquivo.txt
  python detector_ia.py treinar --humano humanos.jsonl --ia ia.jsonl

Nenhum detector prova autoria. O resultado é um indício estatístico, com taxa de erro
medida e informada no relatório.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import math
import random
import re
import statistics
import sys
import unicodedata
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

VERSAO = "1.0"
PASTA = Path(__file__).resolve().parent
MODELO_PADRAO = PASTA / "modelo_estatistico.json"
REFERENCIAS = PASTA / "referencias"
CACHE_NEURAL = Path.home() / ".cache" / "detector_ia"

# =============================================================================
# 1. Leitura de documentos (.pdf, .docx, .txt, .md)
# =============================================================================


@dataclass
class Secao:
    titulo: str
    paragrafos: list
    analisar: bool = True
    motivo: str = ""


def ler_documento(caminho) -> list:
    caminho = Path(caminho)
    ext = caminho.suffix.lower()
    if ext == ".pdf":
        secoes = _ler_pdf(caminho)
    elif ext == ".docx":
        secoes = _ler_docx(caminho)
    else:
        secoes = ler_texto_simples(caminho.read_text(encoding="utf-8", errors="replace"))
    for s in secoes:
        _marcar_secao(s)
    return secoes


def ler_texto_simples(texto: str) -> list:
    """Parágrafos separados por linha em branco; títulos markdown (#) viram seções."""
    secoes, titulo, paras, buf = [], "Texto", [], []
    for linha in texto.splitlines() + [""]:
        l = linha.strip()
        m = re.match(r"^#{1,6}\s+(.*)", l)
        if m or not l:
            if buf:
                paras.append(" ".join(buf))
                buf = []
            if m:
                if paras:
                    secoes.append(Secao(titulo, paras))
                    paras = []
                titulo = m.group(1).strip() or "Texto"
            continue
        buf.append(l)
    if paras:
        secoes.append(Secao(titulo, paras))
    return secoes


def _juntar_linhas(linhas):
    out = ""
    for l in linhas:
        l = l.strip()
        if not l:
            continue
        if out.endswith("-") and l[:1].islower():
            out += l
        else:
            out = (out + " " + l).strip()
    return out


def _paragrafos_de_linhas(linhas):
    """Agrupa linhas de PDF em parágrafos pelos espaços verticais e recuos."""
    linhas = [l for l in linhas if (l.get("text") or "").strip()]
    if not linhas:
        return []
    alturas = [max(1.0, l["bottom"] - l["top"]) for l in linhas]
    h = statistics.median(alturas)
    x_min = min(l["x0"] for l in linhas)
    paras, atual = [], [linhas[0]["text"]]
    for a, b in zip(linhas, linhas[1:]):
        vao = b["top"] - a["bottom"]
        recuo = (b["x0"] - x_min) > 1.5 * h and re.search(r"[.!?:”\"]$", a["text"].strip())
        if vao > 0.65 * h or recuo:
            paras.append(_juntar_linhas(atual))
            atual = [b["text"]]
        else:
            atual.append(b["text"])
    paras.append(_juntar_linhas(atual))
    return [p for p in paras if p]


def _parece_cabecalho(texto):
    letras = [c for c in texto if c.isalpha()]
    if not letras:
        return True
    maiusc = sum(c.isupper() for c in letras) / len(letras)
    return len(texto.split()) <= 12 and maiusc > 0.6


def _anexar_continuacao(secao, paras):
    paras = list(paras)
    if secao.paragrafos and paras and not re.search(r"[.!?:;”\"»)]$", secao.paragrafos[-1].strip()):
        secao.paragrafos[-1] = secao.paragrafos[-1].rstrip() + " " + paras.pop(0)
    secao.paragrafos.extend(paras)


def _secoes_de_formulario(linhas_tabela):
    """Linhas de tabela no formato [rótulo | conteúdo] viram seções (ex.: fichas e formulários)."""
    secoes = []
    for celulas in linhas_tabela:
        cheias = [c for c in celulas if c and " ".join(c).strip()]
        if not cheias or _parece_cabecalho(" ".join(" ".join(c) for c in cheias)):
            continue
        if len(cheias) >= 2:
            rotulo = " ".join(cheias[0]).strip()
            if len(rotulo.split()) <= 12:
                secoes.append(Secao(rotulo.rstrip(": "), list(cheias[-1])))
                continue
        unico = cheias[-1]
        if secoes and not _parece_cabecalho(" ".join(unico)):
            _anexar_continuacao(secoes[-1], unico)
    return secoes


def _ler_pdf(caminho):
    import pdfplumber

    with pdfplumber.open(str(caminho)) as pdf:
        total = 0
        linhas_tabela = []
        for pg in pdf.pages:
            total += len((pg.extract_text() or "").split())
            for tab in pg.find_tables():
                for row in tab.rows:
                    celulas = []
                    for bb in row.cells:
                        if bb is None:
                            continue
                        try:
                            ls = pg.crop(bb).extract_text_lines(strip=True, return_chars=False)
                        except Exception:
                            ls = []
                        celulas.append(_paragrafos_de_linhas(ls))
                    linhas_tabela.append(celulas)
        secoes = _secoes_de_formulario(linhas_tabela)
        conteudo = sum(len(" ".join(s.paragrafos).split()) for s in secoes)
        if secoes and conteudo >= 0.5 * max(1, total):
            return secoes
        # PDF corrido: remove cabeçalhos/rodapés repetidos e números de página
        paginas = [pg.extract_text_lines(strip=True, return_chars=False) for pg in pdf.pages]
    cont = Counter(l["text"] for ls in paginas for l in ls)
    n = len(paginas)
    repetidas = {t for t, c in cont.items() if n >= 3 and c >= max(2, 0.5 * n)}
    sec = Secao("Texto", [])
    for ls in paginas:
        ls = [l for l in ls if l["text"] not in repetidas and not re.fullmatch(r"\d{1,4}", l["text"].strip())]
        _anexar_continuacao(sec, _paragrafos_de_linhas(ls))
    return [sec]


def _ler_docx(caminho):
    import docx

    d = docx.Document(str(caminho))
    linhas_tabela = []
    for tab in d.tables:
        for row in tab.rows:
            vistos, celulas = set(), []
            for c in row.cells:
                if id(c._tc) in vistos:
                    continue
                vistos.add(id(c._tc))
                celulas.append([p.text.strip() for p in c.paragraphs if p.text.strip()])
            linhas_tabela.append(celulas)
    secoes_tab = _secoes_de_formulario(linhas_tabela)
    secoes, titulo, paras = [], "Texto", []
    for p in d.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        estilo = (p.style.name or "").lower() if p.style is not None else ""
        if estilo.startswith(("heading", "título", "titulo")) or estilo == "title":
            if paras:
                secoes.append(Secao(titulo, paras))
                paras = []
            titulo = t
            continue
        paras.append(t)
    if paras:
        secoes.append(Secao(titulo, paras))
    palavras_tab = sum(len(" ".join(s.paragrafos).split()) for s in secoes_tab)
    palavras_corpo = sum(len(" ".join(s.paragrafos).split()) for s in secoes)
    return secoes_tab if palavras_tab > palavras_corpo else (secoes + secoes_tab)


_MARCACAO = re.compile(r"\[\s*[xX]?\s*\]|\(\s*[xX]?\s*\)")


def _marcar_secao(s: Secao):
    texto = " ".join(s.paragrafos)
    n = len(texto.split())
    if re.search(r"refer[êe]ncias|bibliografia", s.titulo, re.I):
        s.analisar, s.motivo = False, "lista de referências"
    elif len(_MARCACAO.findall(texto)) >= 2 or re.search(r"marque uma opção|\b1ª opção", texto, re.I):
        s.analisar, s.motivo = False, "campo de formulário (marcação ou lista de nomes)"
    elif n < 25:
        s.analisar, s.motivo = False, f"trecho curto demais ({n} palavras)"


# =============================================================================
# 2. Limpeza, frases e palavras
# =============================================================================

_PAL = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'’][A-Za-zÀ-ÖØ-öø-ÿ]+)*")
_CIT = re.compile(r"\((?=[^()]*\b(?:1[5-9]\d\d|20\d\d)[a-z]?\b)[^()]{0,160}\)")
_ASPAS = [("“", "”"), ("«", "»"), ('"', '"')]

ABREVIATURAS = set(
    """p pp v n nº ed eds org orgs coord sr sra srs dr dra drs prof profa etc cf ex vol cap fig art inc op cit al
    séc sec jan fev mar abr jun jul ago set out nov dez av ltda trad rev apud ibid id obs pág pag tel aprox
    núm num tab eq figs caps vols""".split()
)


def remover_citacoes_diretas(t, min_palavras=4):
    """Trechos entre aspas com 4+ palavras são citações de terceiros: não entram na análise."""
    for a, f in _ASPAS:
        padrao = re.compile(re.escape(a) + "([^" + re.escape(f) + "]{1,900}?)" + re.escape(f))
        t = padrao.sub(lambda m: " " if len(m.group(1).split()) >= min_palavras else m.group(0), t)
    return t


def limpar(texto: str) -> str:
    t = unicodedata.normalize("NFC", texto)
    t = t.replace("­", "").replace(" ", " ").replace(" ", " ").replace(" ", " ")
    t = re.sub(r"\s+", " ", t)
    t = remover_citacoes_diretas(t)
    t = _CIT.sub("(CIT)", t)
    t = t.replace("[...]", "").replace("[…]", "")
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def palavras(texto: str) -> list:
    return [w for w in _PAL.findall(texto) if w != "CIT"]


_FIM = re.compile(r"([.!?…]+)([\"”»’)\]]*)(\s+)")


def dividir_frases(texto: str) -> list:
    texto = texto.strip()
    frases, inicio = [], 0
    for m in _FIM.finditer(texto):
        prox = texto[m.end(): m.end() + 1]
        antes = texto[inicio: m.start()]
        ult = re.findall(r"[\wÀ-ÿº]+$", antes)
        ult = ult[0].lower() if ult else ""
        if m.group(1) == "." and (ult in ABREVIATURAS or (len(ult) == 1 and ult.isalpha())):
            continue
        if prox and (prox.islower() or prox in ",;:"):
            continue
        frase = texto[inicio: m.end(2)].strip()
        if frase:
            frases.append(frase)
        inicio = m.end()
    resto = texto[inicio:].strip()
    if resto:
        frases.append(resto)
    return frases


def segmentar(paragrafos, minimo=90, maximo=240) -> list:
    """Corta uma seção em segmentos de ~90 a 240 palavras, sempre em fim de frase."""
    frases = [f for p in paragrafos for f in dividir_frases(p)]
    segs, atual, n = [], [], 0
    for f in frases:
        k = len(f.split())
        if atual and n + k > maximo and n >= minimo:
            segs.append(" ".join(atual))
            atual, n = [], 0
        atual.append(f)
        n += k
    if atual:
        if segs and n < 0.6 * minimo:
            segs[-1] = segs[-1] + " " + " ".join(atual)
        else:
            segs.append(" ".join(atual))
    return segs


# =============================================================================
# 3. Léxico de marcadores (português)
# =============================================================================

CONECTIVOS = [
    "além disso", "ademais", "outrossim", "nesse sentido", "neste sentido", "nesse contexto", "neste contexto",
    "nesse cenário", "neste cenário", "diante disso", "diante desse cenário", "diante desse contexto",
    "dessa forma", "desta forma", "desse modo", "deste modo", "dessa maneira", "desta maneira", "assim",
    "portanto", "por conseguinte", "consequentemente", "em suma", "em síntese", "em resumo", "por fim",
    "por outro lado", "no entanto", "contudo", "entretanto", "todavia", "por sua vez", "ou seja", "isto é",
    "em outras palavras", "vale ressaltar", "vale destacar", "vale lembrar", "vale notar", "cabe destacar",
    "cabe ressaltar", "cabe reconhecer", "convém destacar", "é importante destacar", "é importante ressaltar",
    "é importante notar", "é importante lembrar", "a partir disso", "com isso", "em última análise",
    "no fim das contas", "sobretudo", "ainda assim", "complementarmente", "paralelamente", "em primeiro lugar",
    "em segundo lugar", "por um lado", "de um lado", "de outro", "nesse caso", "logo", "então",
]

AVALIATIVOS = [
    "fundamental", "fundamentais", "essencial", "essenciais", "crucial", "cruciais", "central", "centrais",
    "relevante", "relevantes", "relevância", "significativo", "significativa", "significativos",
    "significativas", "significativamente", "expressivo", "expressiva", "expressivos", "expressivas",
    "importante", "importantes", "indispensável", "indispensáveis", "decisivo", "decisiva", "profundo",
    "profunda", "profundamente", "particularmente", "especialmente", "notadamente", "efetivamente", "efetiva",
    "efetivo", "efetivas", "efetivos", "consistente", "consistentes", "robusto", "robusta", "rigoroso",
    "rigorosa", "valioso", "valiosa", "inovador", "inovadora", "amplamente", "intrinsecamente", "genuíno",
    "genuína", "genuinamente", "sólido", "sólida", "estratégico", "estratégica",
]

CONTRASTIVAS = [
    r"não (?:apenas|só|somente|necessariamente)",
    r"(?:mas|como) também",
    r"não se trata de",
    r"não é (?:apenas|só|somente)",
    r"menos [^.;:]{1,40}? (?:e|do que) mais",
    r"tanto [^.;:]{1,60}? quanto",
    r"seja [^.;:]{1,40}? seja",
    r"em vez de",
    r"ao invés de",
    r"ao contrário",
    r"longe de",
    r"deixa(?:m)? de [^.;:]{1,40}? e passa(?:m)? a",
    r"não [^.;:,]{1,30}?: (?:é|são)",
]

FORMULAS = [
    "busca-se", "parte-se", "pretende-se", "propõe-se", "trata-se", "observa-se", "destaca-se", "ressalta-se",
    "entende-se", "considera-se", "verifica-se", "a fim de", "de modo a", "de forma a", "uma vez que",
    "à medida que", "ao passo que", "no que se refere", "no que diz respeito", "diz respeito", "no âmbito",
    "à luz", "lacuna", "lacunas", "dimensão", "dimensões", "perspectiva", "perspectivas", "abordagem",
    "abordagens", "dinâmica", "dinâmicas", "articula", "articular", "articulação", "mobiliza", "mobilizar",
    "mobilizam", "mobilização", "condiciona", "condicionam", "contribui", "contribuir", "contribuindo",
    "possibilita", "permite", "permitem", "permitindo", "evidencia", "evidenciam", "evidenciando", "reforça",
    "reforçam", "ampliar", "aprofundar", "complexidade", "potencial", "desafio", "desafios", "cenário",
    "tensões", "tensão", "lente", "eixo", "camadas", "nuances",
]

INICIO_CONECTIVO = CONECTIVOS + ["além de", "enquanto isso", "por isso", "desse ponto", "a partir daí"]

FUNCIONAIS = set(
    """a o as os um uma uns umas de do da dos das em no na nos nas num numa por pelo pela pelos pelas para pra
    com sem sob sobre entre até após ante contra desde perante e ou mas nem que se porque pois porém como quando
    onde enquanto embora caso conforme segundo já ainda também só apenas mais menos muito muitos muita muitas pouco
    pouca poucos poucas tão tanto tanta tantos tantas todo toda todos todas outro outra outros outras mesmo mesma
    mesmos mesmas próprio própria qual quais cujo cuja cujos cujas quem este esta estes estas esse essa esses essas
    aquele aquela aqueles aquelas isto isso aquilo eu tu ele ela nós vós eles elas me te lhe lhes mim ti si meu
    minha meus minhas teu tua seu sua seus suas nosso nossa nossos nossas é são foi foram era eram ser sido sendo
    está estão estava estar tem têm tinha ter há havia não sim aqui ali lá então assim bem cada qualquer alguns
    algumas algum alguma nenhum nenhuma ao aos à às dele dela deles delas nele nela neles nelas desse dessa desses
    dessas deste desta destes destas daquele daquela nesse nessa neste nesta naquele naquela""".split()
)

PRIMEIRA_PESSOA = re.compile(
    r"\b(?:eu|meu|minha|meus|minhas|comigo|nós|nosso|nossa|nossos|nossas|pretendo|busco|considero|optei|"
    r"escolhi|proponho|pressuponho|questiono|apresento|acredito|entendo|percebi|passei|sou|estou|tenho|atuo|"
    r"trabalhei|vim|quero|pesquisei|observei|achei|farei|realizarei|entrevistarei|adotarei|utilizarei|"
    r"analisarei|buscarei|pesquisarei|concentrarei|submeterei|cursarei|aprofundarei)\b",
    re.I,
)


def _regex_lista(itens, ja_regex=False):
    corpo = "|".join(itens if ja_regex else (re.escape(i) for i in sorted(itens, key=len, reverse=True)))
    return re.compile(r"(?<![\wÀ-ÿ-])(?:" + corpo + r")(?![\wÀ-ÿ-])", re.I)


RX_CONECTIVOS = _regex_lista(CONECTIVOS)
RX_AVALIATIVOS = _regex_lista(AVALIATIVOS)
RX_CONTRASTIVAS = _regex_lista(CONTRASTIVAS, ja_regex=True)
RX_FORMULAS = _regex_lista(FORMULAS)
RX_TRIADE = re.compile(
    r"[\wÀ-ÿ-]+(?:\s+[\wÀ-ÿ-]+){0,3},\s+[\wÀ-ÿ-]+(?:\s+[\wÀ-ÿ-]+){0,3},?\s+e\s+[\wÀ-ÿ-]+", re.I
)
RX_TRAVESSAO = re.compile(r"[—–]|\s--\s")
RX_HIFEN_DASH = re.compile(
    r"(?<=[A-Za-zÀ-ÿ,])\s-\s(?=[A-Za-zÀ-ÿ])(?!(?:jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)\w*\b)"
)
_MESES = r"(?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)"
RX_FAIXA_MESES = re.compile(_MESES + r"\s+-\s+" + _MESES, re.I)

CATEGORIAS_MARCADORES = {
    "conectivo": RX_CONECTIVOS,
    "avaliativo": RX_AVALIATIVOS,
    "contraste": RX_CONTRASTIVAS,
    "fórmula": RX_FORMULAS,
}


@lru_cache(maxsize=200_000)
def _zipf(w):
    try:
        from wordfreq import zipf_frequency
    except ImportError:  # sem wordfreq, a característica fica neutra
        return 4.0
    return zipf_frequency(w, "pt")


def _mattr(ws, janela=50):
    ws = [w.lower() for w in ws]
    if len(ws) <= janela:
        return len(set(ws)) / max(1, len(ws))
    cont = Counter(ws[:janela])
    tot = len(cont)
    soma = tot
    for i in range(janela, len(ws)):
        sai, entra = ws[i - janela], ws[i]
        cont[sai] -= 1
        if cont[sai] == 0:
            del cont[sai]
        cont[entra] += 1
        soma += len(cont)
    return soma / (len(ws) - janela + 1) / janela


# =============================================================================
# 4. Características estilométricas (modo estatístico)
# =============================================================================

ROTULOS = {
    "frase_media": "tamanho médio das frases",
    "frase_dp": "desvio-padrão do tamanho das frases",
    "frase_cv": "variação do tamanho das frases (burstiness)",
    "frac_curtas": "fração de frases curtas (<12 palavras)",
    "frac_longas": "fração de frases longas (>35 palavras)",
    "virgulas_frase": "vírgulas por frase",
    "ponto_virgula": "ponto e vírgula por 100 palavras",
    "dois_pontos": "dois-pontos por 100 palavras",
    "parenteses": "parênteses (fora citações) por 100 palavras",
    "travessao": "travessões por 100 palavras",
    "interrog_exclam": "interrogações/exclamações por 100 palavras",
    "palavra_media": "tamanho médio das palavras",
    "mattr": "diversidade lexical (MATTR)",
    "zipf_medio": "frequência média das palavras (Zipf)",
    "frac_raras": "fração de palavras raras",
    "funcionais": "densidade de palavras funcionais",
    "conectivos": "conectivos formais por 100 palavras",
    "avaliativos": "palavras avaliativas por 100 palavras",
    "contrastivas": "construções de contraste por 100 palavras",
    "formulas": "fórmulas acadêmicas por 100 palavras",
    "triades": "enumerações em tríade por frase",
    "inicio_conectivo": "frases que abrem com conectivo",
    "pronominal_se": "construções com '-se' por 100 palavras",
    "adverbios_mente": "advérbios em '-mente' por 100 palavras",
    "compressao_delta": "semelhança por compressão com textos de IA (ZipPy)",
}


def caracteristicas(texto_bruto: str, ja_limpo=False) -> dict:
    t = texto_bruto if ja_limpo else limpar(texto_bruto)
    frases = [f for f in dividir_frases(t) if palavras(f)]
    ws = palavras(t)
    n = max(1, len(ws))
    tam = [len(palavras(f)) for f in frases] or [n]
    media = statistics.fmean(tam)
    dp = statistics.pstdev(tam) if len(tam) > 1 else 0.0
    nf = max(1, len(frases))
    minus = t.lower()
    minusculas = [w for w in ws if w[:1].islower()]
    zipfs = [_zipf(w) for w in minusculas] or [4.0]
    cent = 100.0 / n
    traco = len(RX_TRAVESSAO.findall(t)) + len(RX_HIFEN_DASH.findall(t)) - len(RX_FAIXA_MESES.findall(t))
    return {
        "frase_media": media,
        "frase_dp": dp,
        "frase_cv": dp / media if media else 0.0,
        "frac_curtas": sum(k < 12 for k in tam) / len(tam),
        "frac_longas": sum(k > 35 for k in tam) / len(tam),
        "virgulas_frase": t.count(",") / nf,
        "ponto_virgula": t.count(";") * cent,
        "dois_pontos": t.count(":") * cent,
        "parenteses": max(0, t.count("(") - t.count("(CIT)")) * cent,
        "travessao": max(0, traco) * cent,
        "interrog_exclam": (t.count("?") + t.count("!")) * cent,
        "palavra_media": statistics.fmean(len(w) for w in ws) if ws else 0.0,
        "mattr": _mattr(ws),
        "zipf_medio": statistics.fmean(zipfs),
        "frac_raras": sum(z < 3.0 for z in zipfs) / len(zipfs),
        "funcionais": sum(w.lower() in FUNCIONAIS for w in ws) / n,
        "conectivos": len(RX_CONECTIVOS.findall(minus)) * cent,
        "avaliativos": len(RX_AVALIATIVOS.findall(minus)) * cent,
        "contrastivas": len(RX_CONTRASTIVAS.findall(minus)) * cent,
        "formulas": len(RX_FORMULAS.findall(minus)) * cent,
        "triades": sum(len(RX_TRIADE.findall(f)) for f in frases) / nf,
        "inicio_conectivo": sum(
            any(f.lower().startswith(c + " ") or f.lower().startswith(c + ",") for c in INICIO_CONECTIVO)
            for f in frases
        ) / nf,
        "pronominal_se": len(re.findall(r"\b[\wÀ-ÿ]+-se\b", minus)) * cent,
        "adverbios_mente": len(re.findall(r"\b[\wÀ-ÿ]{3,}mente\b", minus)) * cent,
    }


# =============================================================================
# 5. Compressão (método do ZipPy) sobre o "esqueleto" do texto
# =============================================================================


_PESSOA = {"eu": "ele", "me": "se", "mim": "si", "comigo": "consigo", "meu": "seu", "minha": "sua", "meus": "seus",
           "minhas": "suas", "nós": "eles", "nosso": "seu", "nossa": "sua", "nossos": "seus", "nossas": "suas"}


def neutralizar_pessoa(w):
    """1ª pessoa vira 3ª: o tom pessoal é questão de gênero textual (motivação, ensaio), não indício de IA."""
    return _PESSOA.get(w, w)


def esqueleto(texto: str) -> str:
    """Troca palavras de conteúdo por classes (W, M, N) e mantém palavras funcionais e pontuação.
    Assim a compressão e os n-gramas medem estilo e sintaxe, não o assunto."""
    t = limpar(texto)
    out = []
    for tok in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:-[A-Za-zÀ-ÖØ-öø-ÿ]+)*|\d+|[^\w\s]", t):
        lw = neutralizar_pessoa(tok.lower())
        if tok == "CIT":
            out.append("C")
        elif lw in FUNCIONAIS:
            out.append(lw)
        elif tok[0].isdigit():
            out.append("N")
        elif tok[0].isalpha():
            if lw.endswith("-se"):
                out.append("V-se")
            elif lw.endswith("mente"):
                out.append("Wmente")
            else:
                out.append("M" if tok[0].isupper() else "W")
        else:
            out.append(tok)
    return " ".join(out)


class Compressor:
    """Mede quantos bytes o texto custa quando comprimido usando um corpus de referência como dicionário."""

    def __init__(self, textos_ia, textos_humanos, max_bytes=32768):
        def dic(textos):
            b = ("\n".join(esqueleto(t) for t in textos)).encode("utf-8")
            return b[-max_bytes:]

        self.zd_ia = dic(textos_ia)
        self.zd_hum = dic(textos_humanos)

    @staticmethod
    def _custo(dados: bytes, zdict: bytes) -> int:
        c = zlib.compressobj(level=9, zdict=zdict)
        return len(c.compress(dados) + c.flush())

    def delta(self, texto: str) -> float:
        b = esqueleto(texto).encode("utf-8")
        if not b:
            return 0.0
        return (self._custo(b, self.zd_hum) - self._custo(b, self.zd_ia)) / len(b)


# =============================================================================
# 6. Classificadores (estilo + sintaxe), validação cruzada por documento, limiares
# =============================================================================


def _sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def treinar_logistica(X, y, l2=1.0, pesos=None, iters=100):
    """Regressão logística com penalização L2, resolvida por Newton (IRLS)."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    R = l2 * np.eye(d + 1)
    R[-1, -1] = 0.0
    sw = np.ones(n) if pesos is None else np.asarray(pesos, float)
    for _ in range(iters):
        p = _sigmoide(Xb @ w)
        g = Xb.T @ (sw * (p - y)) + R @ w
        H = (Xb * (sw * p * (1 - p))[:, None]).T @ Xb + R
        passo = np.linalg.solve(H + 1e-9 * np.eye(d + 1), g)
        w -= passo
        if np.linalg.norm(passo) < 1e-9:
            break
    return w[:-1], float(w[-1])


def pesos_balanceados(y):
    y = np.asarray(y)
    n1, n0 = max(1, y.sum()), max(1, (1 - y).sum())
    return np.where(y == 1, len(y) / (2 * n1), len(y) / (2 * n0))


def auroc(y, s):
    """Área sob a curva ROC (estatística de Mann-Whitney, com empates)."""
    y, s = np.asarray(y), np.asarray(s, float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ordem = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[ordem]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[ordem[i: j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def dobras_por_grupo(y, grupos, k=5, semente=0):
    """Distribui documentos (grupos) entre k dobras, por classe. Trechos do mesmo documento — e uma
    reescrita por IA junto do original humano — nunca ficam em dobras diferentes."""
    rng = random.Random(semente)
    por_classe = defaultdict(set)
    for yi, g in zip(y, grupos):
        por_classe[int(yi)].add(g)
    destino = {}
    for c in sorted(por_classe, reverse=True):
        gs = sorted(por_classe[c])
        rng.shuffle(gs)
        for i, g in enumerate(gs):
            destino.setdefault(g, i % k)
    return np.array([destino[g] for g in grupos])


def validar(X, y, grupos, k=5, repeticoes=5, l2=1.0):
    """Probabilidades fora da dobra (média das repetições) de uma regressão logística sobre X."""
    oof = np.zeros(len(y))
    for r in range(repeticoes):
        dob = dobras_por_grupo(y, grupos, k=k, semente=r)
        for f in range(k):
            te, tr = dob == f, dob != f
            if tr.sum() == 0 or te.sum() == 0 or len(set(y[tr])) < 2:
                oof[te] += 0.5
                continue
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            coef, b = treinar_logistica((X[tr] - mu) / sd, y[tr], l2=l2, pesos=pesos_balanceados(y[tr]))
            oof[te] += _sigmoide(((X[te] - mu) / sd) @ coef + b)
    return oof / repeticoes


def escolher_limiares(y, s, fpr_alvo=0.05, fnr_alvo=0.10):
    """Limiar de IA: só 5% dos trechos humanos o ultrapassam. Limiar humano: 90% dos trechos de IA ficam acima."""
    y, s = np.asarray(y), np.asarray(s)
    limiar_ia = float(np.quantile(s[y == 0], 1 - fpr_alvo))
    limiar_hum = min(float(np.quantile(s[y == 1], fnr_alvo)), limiar_ia)
    return limiar_ia, limiar_hum


def perfil_funcional(texto_limpo, lista):
    ws = [neutralizar_pessoa(w.lower()) for w in palavras(texto_limpo)]
    n = max(1, len(ws))
    c = Counter(ws)
    return {f"fw_{w}": c[w] / n for w in lista}


def _ngramas(tokens, nmin, nmax):
    for n in range(nmin, nmax + 1):
        for i in range(len(tokens) - n + 1):
            yield " ".join(tokens[i: i + n])


class Sintaxe:
    """TF-IDF de n-gramas do esqueleto sintático + regressão logística (inferência sem scikit-learn)."""

    def __init__(self, d):
        self.lista = d["vocabulario"]
        self.vocab = {g: i for i, g in enumerate(self.lista)}
        self.idf = np.array(d["idf"])
        self.coef = np.array(d["coef"])
        self.b = float(d["intercepto"])
        self.nmin, self.nmax = d["ngramas"]

    def _vetor(self, esq):
        cont = Counter(g for g in _ngramas(esq.split(), self.nmin, self.nmax) if g in self.vocab)
        if not cont:
            return np.array([], int), np.array([])
        idx = np.array([self.vocab[g] for g in cont], int)
        val = np.array([1.0 + math.log(c) for c in cont.values()]) * self.idf[idx]
        return idx, val / (np.linalg.norm(val) or 1.0)

    def prob(self, esq):
        idx, val = self._vetor(esq)
        return float(_sigmoide((val @ self.coef[idx] if len(idx) else 0.0) + self.b))

    def padroes(self, esq, k=3):
        idx, val = self._vetor(esq)
        if not len(idx):
            return []
        contrib = val * self.coef[idx]
        ordem = np.argsort(contrib)[::-1][:k]
        return [(self.lista[idx[j]], float(contrib[j])) for j in ordem if contrib[j] > 0.02]


def rotulo(nome):
    if nome.startswith("fw_"):
        return f"frequência de “{nome[3:]}”"
    return ROTULOS.get(nome, nome)


class ModeloEstatistico:
    def __init__(self, dados: dict):
        self.d = dados
        e = dados["estilo"]
        self.fw = e["funcionais"]
        self.nomes = e["caracteristicas"]
        self.mu, self.sd = np.array(e["media"]), np.array(e["desvio"])
        self.coef, self.b = np.array(e["coef"]), float(e["intercepto"])
        self.sintaxe = Sintaxe(dados["sintaxe"])
        self.peso = float(dados["peso_estilo"])
        self.limiar_ia = float(dados["limiar_ia"])
        self.limiar_humano = float(dados["limiar_humano"])
        self.zippy = dados.get("zippy")
        self._comp = None

    @classmethod
    def carregar(cls, caminho=MODELO_PADRAO):
        return cls(json.loads(Path(caminho).read_text(encoding="utf-8")))

    def compressor(self):
        if self._comp is None and self.zippy:
            ia = [r["texto"] for r in carregar_jsonl(REFERENCIAS / "ia.jsonl")]
            hum = [r["texto"] for r in carregar_jsonl(REFERENCIAS / "humano.jsonl")]
            if ia and hum:
                self._comp = Compressor(ia, hum)
        return self._comp

    def pontuar(self, texto: str, k=3) -> dict:
        t = limpar(texto)
        f = caracteristicas(t, ja_limpo=True)
        f.update(perfil_funcional(t, self.fw))
        z = (np.array([f[n] for n in self.nomes]) - self.mu) / self.sd
        contrib = z * self.coef
        p_estilo = float(_sigmoide(z @ self.coef + self.b))
        esq = esqueleto(texto)
        p_sint = self.sintaxe.prob(esq)
        p = self.peso * p_estilo + (1 - self.peso) * p_sint
        ordem = np.argsort(contrib)
        pro_ia = [(self.nomes[i], float(contrib[i]), float(f[self.nomes[i]])) for i in ordem[::-1][:k] if contrib[i] > 0.1]
        pro_hum = [(self.nomes[i], float(contrib[i]), float(f[self.nomes[i]])) for i in ordem[:k] if contrib[i] < -0.1]
        out = {"escore": p, "estilo": p_estilo, "sintaxe": p_sint, "pro_ia": pro_ia, "pro_humano": pro_hum,
               "padroes": self.sintaxe.padroes(esq, k), "caracteristicas": {n: f[n] for n in ROTULOS if n in f}}
        comp = self.compressor()
        if comp is not None:
            out["zippy_delta"] = comp.delta(texto)
            out["zippy_ia"] = out["zippy_delta"] >= self.zippy["limiar"]
        return out

    def faixa(self, p):
        if p >= self.limiar_ia:
            return "ia"
        if p < self.limiar_humano:
            return "humano"
        return "incerto"


def carregar_jsonl(caminho):
    caminho = Path(caminho)
    if not caminho.exists():
        return []
    return [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()]


def carregar_corpus(fontes, rotulo_classe):
    """Aceita .jsonl ({"texto", "doc"?, "fonte"?, "grupo"?}), pastas com .txt/.md/.docx/.pdf ou arquivos."""
    registros = []
    for fonte in fontes:
        p = Path(fonte)
        if p.suffix == ".jsonl":
            for i, r in enumerate(carregar_jsonl(p)):
                doc = r.get("doc", f"{p.stem}_{i}")
                registros.append({"texto": r["texto"], "doc": doc, "grupo": r.get("grupo", doc),
                                  "fonte": r.get("fonte", f"{rotulo_classe}_{p.stem}")})
            continue
        arquivos = sorted(p.rglob("*")) if p.is_dir() else [p]
        for a in arquivos:
            if a.suffix.lower() not in (".txt", ".md", ".docx", ".pdf"):
                continue
            for s in ler_documento(a):
                for seg in segmentar(s.paragrafos, 110, 260):
                    registros.append({"texto": seg, "doc": str(a), "grupo": str(a), "fonte": f"{rotulo_classe}_{p.name}"})
    return registros


def _delta_aninhado(textos, y, grupos, indices, k=5, semente=99):
    """Delta do ZipPy para cada índice, com referências que nunca contêm o próprio documento."""
    indices = np.asarray(indices)
    dob = dobras_por_grupo(y[indices], [grupos[i] for i in indices], k=k, semente=semente)
    out = np.zeros(len(indices))
    for f in range(k):
        tr = indices[dob != f]
        rng = random.Random(1000 + f)
        hum = [i for i in tr if y[i] == 0]
        comp = Compressor([textos[i] for i in tr if y[i] == 1], [textos[i] for i in rng.sample(hum, min(400, len(hum)))])
        for j in np.where(dob == f)[0]:
            out[j] = comp.delta(textos[indices[j]])
    return out


def treinar_modelo(humanos, ias, saida=MODELO_PADRAO, peso_estilo=0.35, reps=3, k=5, n_funcionais=120, verbose=True):
    """Treina os dois classificadores, mede o erro com validação cruzada agrupada por documento,
    escolhe os limiares e salva o modelo. Requer scikit-learn (só para treinar)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    dados = [(r, 0) for r in humanos] + [(r, 1) for r in ias]
    y = np.array([c for _, c in dados])
    textos = [r["texto"] for r, _ in dados]
    grupos = [r.get("grupo", r.get("doc")) for r, _ in dados]
    fontes = [r.get("fonte", "?") for r, _ in dados]
    limpos = [limpar(t) for t in textos]
    cont = Counter(neutralizar_pessoa(w.lower()) for t in limpos for w in palavras(t) if w.lower() in FUNCIONAIS)
    fw = [w for w, _ in cont.most_common(n_funcionais)]
    base = [n for n in ROTULOS if n != "compressao_delta"]
    nomes = base + [f"fw_{w}" for w in fw]
    linhas = []
    for t in limpos:
        f = caracteristicas(t, ja_limpo=True)
        f.update(perfil_funcional(t, fw))
        linhas.append([f[n] for n in nomes])
    XA = np.array(linhas)
    esqs = [esqueleto(t) for t in textos]

    def ajustar_sintaxe(idx):
        v = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 3), min_df=3, token_pattern=r"\S+", lowercase=False)
        M = v.fit_transform([esqs[i] for i in idx])
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(M, y[idx])
        return v, lr

    oofA, oofB = np.zeros(len(y)), np.zeros(len(y))
    for r in range(reps):
        dob = dobras_por_grupo(y, grupos, k=k, semente=r)
        for f in range(k):
            tr, te = np.where(dob != f)[0], np.where(dob == f)[0]
            mu, sd = XA[tr].mean(0), XA[tr].std(0) + 1e-9
            coef, b = treinar_logistica((XA[tr] - mu) / sd, y[tr], l2=10.0, pesos=pesos_balanceados(y[tr]))
            oofA[te] += _sigmoide(((XA[te] - mu) / sd) @ coef + b)
            v, lr = ajustar_sintaxe(tr)
            oofB[te] += lr.predict_proba(v.transform([esqs[i] for i in te]))[:, 1]
        if verbose:
            print(f"  validação cruzada: repetição {r + 1}/{reps}")
    oofA, oofB = oofA / reps, oofB / reps
    oof = peso_estilo * oofA + (1 - peso_estilo) * oofB
    limiar_ia, limiar_hum = escolher_limiares(y, oof)

    mu, sd = XA.mean(0), XA.std(0) + 1e-9
    coef, b = treinar_logistica((XA - mu) / sd, y, l2=10.0, pesos=pesos_balanceados(y))
    v, lr = ajustar_sintaxe(np.arange(len(y)))
    vocab = sorted(v.vocabulary_, key=v.vocabulary_.get)

    if verbose:
        print("  ZipPy (compressão): medindo fora da amostra…")
    deltas = _delta_aninhado(textos, y, grupos, np.arange(len(y)))
    z_lim = float(np.quantile(deltas[y == 0], 0.95))

    por_fonte = {}
    for fonte in sorted(set(fontes)):
        idx = [i for i, fz in enumerate(fontes) if fz == fonte]
        s = oof[idx]
        por_fonte[fonte] = {"classe": "ia" if y[idx[0]] else "humano", "n": len(idx),
                            "acima_limiar_ia": float((s >= limiar_ia).mean()),
                            "abaixo_limiar_humano": float((s < limiar_hum).mean())}
    metricas = {
        "auroc": auroc(y, oof), "auroc_estilo": auroc(y, oofA), "auroc_sintaxe": auroc(y, oofB),
        "auroc_zippy": auroc(y, deltas),
        "fpr_no_limiar": float((oof[y == 0] >= limiar_ia).mean()),
        "tpr_no_limiar": float((oof[y == 1] >= limiar_ia).mean()),
        "ia_abaixo_limiar_humano": float((oof[y == 1] < limiar_hum).mean()),
        "humano_abaixo_limiar_humano": float((oof[y == 0] < limiar_hum).mean()),
        "n_humano": int((y == 0).sum()), "n_ia": int((y == 1).sum()), "por_fonte": por_fonte,
        "metodo": f"média ponderada de 2 regressões logísticas (estilo {peso_estilo:.0%}, sintaxe {1 - peso_estilo:.0%}); "
                  f"validação cruzada {k} dobras × {reps} repetições, agrupada por documento",
    }
    modelo = {
        "versao": VERSAO, "criado_em": _dt.datetime.now().isoformat(timespec="seconds"),
        "estilo": {"funcionais": fw, "caracteristicas": nomes, "media": mu.tolist(), "desvio": sd.tolist(),
                   "coef": coef.tolist(), "intercepto": b},
        "sintaxe": {"ngramas": [1, 3], "vocabulario": vocab, "idf": v.idf_.tolist(),
                    "coef": lr.coef_[0].tolist(), "intercepto": float(lr.intercept_[0])},
        "peso_estilo": peso_estilo, "limiar_ia": limiar_ia, "limiar_humano": limiar_hum,
        "zippy": {"limiar": z_lim, "auroc": metricas["auroc_zippy"]},
        "metricas": metricas,
    }
    if saida:
        Path(saida).write_text(json.dumps(modelo, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"AUROC: conjunto {metricas['auroc']:.3f} | estilo {metricas['auroc_estilo']:.3f} | "
              f"sintaxe {metricas['auroc_sintaxe']:.3f} | ZipPy {metricas['auroc_zippy']:.3f}")
        print(f"Limiar IA {limiar_ia:.3f}: falso positivo {metricas['fpr_no_limiar']:.1%} | IA detectada {metricas['tpr_no_limiar']:.1%}")
        print(f"Limiar humano {limiar_hum:.3f}: IA abaixo dele {metricas['ia_abaixo_limiar_humano']:.1%} | "
              f"humano abaixo dele {metricas['humano_abaixo_limiar_humano']:.1%}")
        for fz, m in por_fonte.items():
            print(f"  {fz:32s} n={m['n']:5d}  acima do limiar IA: {m['acima_limiar_ia']:6.1%}  abaixo do limiar humano: {m['abaixo_limiar_humano']:6.1%}")
    return modelo, oof, y


# =============================================================================
# 7. Modo neural: perplexidade/burstiness (GPTZero), GLTR, Fast-DetectGPT, Binoculars
# =============================================================================

MEDIDAS_NEURAIS = ["log_ppl", "burstiness", "gltr_top10", "fast_detectgpt", "binoculars"]
ROTULOS_NEURAIS = {
    "perplexidade": "perplexidade (observador)",
    "log_ppl": "log-perplexidade (observador)",
    "burstiness": "burstiness: variação da perplexidade entre frases (GPTZero)",
    "gltr_top10": "fração de tokens no top-10 do modelo (GLTR)",
    "gltr_top100": "fração de tokens no top-100 (GLTR)",
    "gltr_top1000": "fração de tokens no top-1000 (GLTR)",
    "fast_detectgpt": "discrepância de amostragem (Fast-DetectGPT)",
    "binoculars": "escore Binoculars (menor = mais típico de IA)",
}


def _carregar_lm(nome, dtype, dispositivo):
    from transformers import AutoModelForCausalLM

    try:
        m = AutoModelForCausalLM.from_pretrained(nome, dtype=dtype)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(nome, torch_dtype=dtype)
    return m.to(dispositivo).eval()


class AvaliadorNeural:
    def __init__(self, observador="Qwen/Qwen2.5-0.5B", executor="Qwen/Qwen2.5-0.5B-Instruct",
                 dispositivo=None, max_tokens=768, bloco=128):
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        if dispositivo is None:
            if torch.cuda.is_available():
                dispositivo = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                dispositivo = "mps"
            else:
                dispositivo = "cpu"
        self.disp = dispositivo
        dtype = {"cuda": torch.bfloat16, "mps": torch.float16}.get(dispositivo, torch.float32)
        self.nome_obs, self.nome_exe = observador, executor
        self.tok = AutoTokenizer.from_pretrained(observador)
        tok_exe = AutoTokenizer.from_pretrained(executor)
        if tok_exe.get_vocab() != self.tok.get_vocab():
            raise ValueError("Observador e executor precisam do mesmo tokenizador (ex.: Qwen2.5-0.5B e Qwen2.5-0.5B-Instruct).")
        self.obs = _carregar_lm(observador, dtype, dispositivo)
        self.exe = _carregar_lm(executor, dtype, dispositivo)
        self.max_tokens, self.bloco = max_tokens, bloco

    def pontuar(self, texto: str) -> dict:
        torch = self.torch
        t = limpar(texto)
        enc = self.tok(t, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=self.max_tokens)
        ids = enc["input_ids"].to(self.disp)
        if ids.shape[1] < 8:
            return {}
        with torch.no_grad():
            lo_all = self.obs(input_ids=ids).logits[0, :-1]
            le_all = self.exe(input_ids=ids).logits[0, :-1]
            y_all = ids[0, 1:]
            V = min(lo_all.shape[-1], le_all.shape[-1])
            ll_o, ll_e, mu_fd, var_fd, xent, rank = [], [], [], [], [], []
            for i in range(0, y_all.shape[0], self.bloco):  # em blocos, para caber na memória
                lo = lo_all[i: i + self.bloco, :V].float()
                le = le_all[i: i + self.bloco, :V].float()
                y = y_all[i: i + self.bloco]
                lp_o, lp_e = torch.log_softmax(lo, -1), torch.log_softmax(le, -1)
                a_o = lp_o.gather(-1, y[:, None]).squeeze(-1)
                ll_o.append(a_o)
                ll_e.append(lp_e.gather(-1, y[:, None]).squeeze(-1))
                p_ref = torch.softmax(le, -1)  # Fast-DetectGPT: amostrador = executor, pontuador = observador
                m = (p_ref * lp_o).sum(-1)
                mu_fd.append(m)
                var_fd.append((p_ref * lp_o.pow(2)).sum(-1) - m.pow(2))
                xent.append(-(torch.softmax(lo, -1) * lp_e).sum(-1))  # Binoculars: entropia cruzada
                rank.append((lo > lo.gather(-1, y[:, None])).sum(-1))
            ll_o, ll_e = torch.cat(ll_o), torch.cat(ll_e)
            mu_fd, var_fd, xent, rank = torch.cat(mu_fd), torch.cat(var_fd), torch.cat(xent), torch.cat(rank)
        nll = (-ll_o).cpu().numpy()
        fast = float((ll_o.sum() - mu_fd.sum()) / var_fd.sum().clamp_min(1e-6).sqrt())
        bino = float((-ll_e.mean()) / xent.mean().clamp_min(1e-6))
        rank = rank.cpu().numpy()
        # burstiness à moda do GPTZero: quanto a perplexidade varia de uma frase para outra
        offs = enc["offset_mapping"][0, 1:].tolist()
        limites, pos = [], 0
        for f in dividir_frases(t):
            ini = t.find(f, pos)
            limites.append((ini, ini + len(f)))
            pos = ini + len(f)
        por_frase = defaultdict(list)
        for j, (a, _) in enumerate(offs):
            for k, (ini, fim) in enumerate(limites):
                if ini <= a < fim:
                    por_frase[k].append(nll[j])
                    break
        medias = [float(np.mean(v)) for v in por_frase.values() if len(v) >= 3]
        burst = float(np.std(medias)) if len(medias) >= 2 else 0.0
        return {
            "perplexidade": float(np.exp(nll.mean())),
            "log_ppl": float(nll.mean()),
            "burstiness": burst,
            "gltr_top10": float((rank < 10).mean()),
            "gltr_top100": float((rank < 100).mean()),
            "gltr_top1000": float((rank < 1000).mean()),
            "fast_detectgpt": fast,
            "binoculars": bino,
            "tokens": int(len(nll)),
        }


def _slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def calibrar_neural(av: AvaliadorNeural, n_por_classe=40, forcar=False, verbose=True):
    """Pontua textos de referência (humanos e de IA) com o par de modelos escolhido e ajusta
    limiares e pesos para ESTE par de modelos. Fica em cache em ~/.cache/detector_ia."""
    CACHE_NEURAL.mkdir(parents=True, exist_ok=True)
    arq = CACHE_NEURAL / f"calibracao_{_slug(av.nome_obs)}__{_slug(av.nome_exe)}.json"
    if arq.exists() and not forcar:
        return json.loads(arq.read_text(encoding="utf-8"))
    rng = random.Random(7)
    hum = carregar_jsonl(REFERENCIAS / "humano.jsonl")
    ia = carregar_jsonl(REFERENCIAS / "ia.jsonl")
    hum = rng.sample(hum, min(n_por_classe, len(hum)))
    ia = rng.sample(ia, min(n_por_classe, len(ia)))
    linhas = []
    for rotulo, conj in ((0, hum), (1, ia)):
        for i, r in enumerate(conj):
            if verbose:
                print(f"\r  calibrando ({'humano' if rotulo == 0 else 'IA'}) {i + 1}/{len(conj)}", end="", flush=True)
            s = av.pontuar(r["texto"])
            if s:
                linhas.append((rotulo, r.get("grupo", r.get("doc", str(i))), s))
    if verbose:
        print()
    y = np.array([l[0] for l in linhas])
    grupos = [l[1] for l in linhas]
    X = np.array([[l[2][m] for m in MEDIDAS_NEURAIS] for l in linhas])
    individuais = {}
    for j, m in enumerate(MEDIDAS_NEURAIS):
        a = auroc(y, X[:, j])
        individuais[m] = {"auroc": a, "direcao": 1 if a >= 0.5 else -1}
    oof = validar(X, y, grupos, k=5, repeticoes=5, l2=2.0)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    coef, b = treinar_logistica((X - mu) / sd, y, l2=2.0, pesos=pesos_balanceados(y))
    limiar_ia, limiar_hum = escolher_limiares(y, oof)
    cal = {
        "observador": av.nome_obs, "executor": av.nome_exe, "medidas": MEDIDAS_NEURAIS,
        "media": mu.tolist(), "desvio": sd.tolist(), "coef": coef.tolist(), "intercepto": b,
        "limiar_ia": limiar_ia, "limiar_humano": limiar_hum, "auroc_cv": auroc(y, oof),
        "fpr_no_limiar": float((oof[y == 0] >= limiar_ia).mean()),
        "tpr_no_limiar": float((oof[y == 1] >= limiar_ia).mean()),
        "individuais": individuais, "n": int(len(y)),
    }
    arq.write_text(json.dumps(cal, ensure_ascii=False, indent=1), encoding="utf-8")
    return cal


def prob_neural(cal, medidas):
    x = np.array([medidas[m] for m in cal["medidas"]], float)
    z = (x - np.array(cal["media"])) / np.array(cal["desvio"])
    return float(_sigmoide(z @ np.array(cal["coef"]) + cal["intercepto"]))


# =============================================================================
# 8. Sinais típicos de escrita humana (informativos: não entram no escore)
# =============================================================================

_NUMERAIS = set("um uma dois duas três quatro cinco seis sete oito nove dez onze doze treze vinte trinta quarenta cinquenta cem mil".split())
_NAO_PLURAL = set("que quais quanto quantos mais menos demais mesmos seus suas meus minhas nossos nossas dos das".split())


def sinais_humanos(texto_bruto: str) -> list:
    """Deslizes de digitação e concordância. IA quase nunca os produz; revisão humana também os remove."""
    sinais = []
    t = re.sub(r"\s+", " ", unicodedata.normalize("NFC", texto_bruto))
    for m in re.finditer(r"\b(\w+)\s+\1\b", t, re.I):
        if m.group(1).lower() not in ("que", "se", "a", "o"):
            sinais.append(("palavra repetida", m.group(0)))
    rx = r"(\b\w+\s+)?\b(dos|das|os|as|nos|nas|pelos|pelas|aos|às|destes|desses|estes|esses)\s+([a-zà-ú]{4,})\b"
    for m in re.finditer(rx, t):
        antes, w = (m.group(1) or "").strip().lower(), m.group(3)
        if antes in ("que", "se", "não", "já", "também", "nunca", "lhe", "me"):
            continue  # "os" pronome oblíquo antes de verbo ("que os sustenta")
        if not w.endswith(("s", "z")) and w not in _NUMERAIS and w not in _NAO_PLURAL and _zipf(w) >= 3.0:
            sinais.append(("possível erro de concordância", m.group(0).strip()))
    for m in re.finditer(r"\((?:a|o) para\b", t):
        sinais.append(("construção truncada", m.group(0)))
    abre, fecha = t.count("“"), t.count("”")
    if abre != fecha:
        sinais.append(("aspas sem par", f"{abre} abertas, {fecha} fechadas"))
    vistos = set()
    for w in palavras(t):
        lw = w.lower()
        if lw in vistos or not w[:1].islower() or len(lw) < 6 or "-" in lw:
            continue
        vistos.add(lw)
        zw = _zipf(lw)
        if zw >= 2.0:
            continue
        melhor = max(_edicoes1(lw), key=_zipf, default=None)
        if melhor and _zipf(melhor) >= 3.5 and _zipf(melhor) - zw >= 2.0:
            sinais.append(("possível erro de digitação", f"{w} → {melhor}?"))
    return sinais


def _edicoes1(w):
    letras = "abcdefghijklmnopqrstuvwxyzáàâãéêíóôõúç"
    partes = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    for a, b in partes:
        if b:
            yield a + b[1:]
        if len(b) > 1:
            yield a + b[1] + b[0] + b[2:]
        for c in letras:
            if b:
                yield a + c + b[1:]
            yield a + c + b


# =============================================================================
# 9. Análise de um documento
# =============================================================================


def analisar_secoes(secoes, modelo: ModeloEstatistico, neural=None, cal=None, incluir_tudo=False):
    resultado = []
    for s in secoes:
        rs = {"titulo": s.titulo, "analisada": bool(s.analisar or incluir_tudo), "motivo": s.motivo,
              "texto": "\n\n".join(s.paragrafos), "segmentos": []}
        if rs["analisada"]:
            for seg in segmentar(s.paragrafos):
                r = modelo.pontuar(seg)
                item = {"texto": seg, "palavras": len(seg.split()), "palavras_autor": len(palavras(limpar(seg))),
                        **r, "faixa": modelo.faixa(r["escore"])}
                if neural is not None and cal is not None:
                    med = neural.pontuar(seg)
                    if med:
                        pn = prob_neural(cal, med)
                        item.update(neural=med, escore_neural=pn,
                                    faixa_neural="ia" if pn >= cal["limiar_ia"] else ("humano" if pn < cal["limiar_humano"] else "incerto"))
                if item["palavras_autor"] > 0:
                    rs["segmentos"].append(item)
        resultado.append(rs)
    return resultado


def resumir(resultado_secoes, chave="escore", chave_faixa="faixa"):
    segs = [g for s in resultado_secoes for g in s["segmentos"] if chave in g]
    total = sum(g["palavras_autor"] for g in segs) or 1
    pct = {f: 100 * sum(g["palavras_autor"] for g in segs if g[chave_faixa] == f) / total for f in ("ia", "incerto", "humano")}
    indice = 100 * sum(g[chave] * g["palavras_autor"] for g in segs) / total
    if pct["ia"] >= 50:
        veredito = "Predominam sinais de texto gerado ou reescrito por IA"
    elif pct["ia"] >= 15:
        veredito = "Texto misto: há trechos com sinais de IA"
    elif pct["humano"] >= 60:
        veredito = "Predominam sinais de escrita humana"
    else:
        veredito = "Inconclusivo: sinais fracos nas duas direções"
    return {"palavras_analisadas": total, "pct_ia": pct["ia"], "pct_incerto": pct["incerto"],
            "pct_humano": pct["humano"], "escore_medio": indice, "veredito": veredito, "segmentos": len(segs)}


def analisar_arquivo(caminho, modelo_path=MODELO_PADRAO, usar_neural=False, observador="Qwen/Qwen2.5-0.5B",
                     executor="Qwen/Qwen2.5-0.5B-Instruct", dispositivo=None, incluir_tudo=False, secoes=None):
    modelo = ModeloEstatistico.carregar(modelo_path)
    secoes = secoes if secoes is not None else ler_documento(caminho)
    neural = cal = None
    if usar_neural:
        print(f"Carregando modelos neurais: {observador} + {executor} …")
        neural = AvaliadorNeural(observador, executor, dispositivo)
        cal = calibrar_neural(neural)
    rs = analisar_secoes(secoes, modelo, neural, cal, incluir_tudo)
    texto_todo = "\n".join(s["texto"] for s in rs if s["analisada"])
    met = modelo.d["metricas"]
    res = {
        "arquivo": Path(caminho).name if caminho else "texto",
        "data": _dt.datetime.now().isoformat(timespec="seconds"),
        "versao": VERSAO,
        "modelo": {k: met.get(k) for k in ("auroc", "auroc_estilo", "auroc_sintaxe", "auroc_zippy", "fpr_no_limiar",
                                          "tpr_no_limiar", "n_humano", "n_ia", "por_fonte", "metodo")}
        | {"limiar_ia": modelo.limiar_ia, "limiar_humano": modelo.limiar_humano},
        "resumo": resumir(rs),
        "secoes": rs,
        "sinais_humanos": sinais_humanos(texto_todo),
        "neural": None,
    }
    if cal is not None:
        res["neural"] = {"calibracao": cal, "resumo": resumir(rs, "escore_neural", "faixa_neural")}
    return res


# =============================================================================
# 10. Relatório HTML
# =============================================================================


def _spans_citacao(texto: str):
    spans = []
    for a, f in _ASPAS:
        rx = re.compile(re.escape(a) + "([^" + re.escape(f) + "]{1,900}?)" + re.escape(f))
        spans += [(m.start(), m.end()) for m in rx.finditer(texto) if len(m.group(1).split()) >= 4]
    return spans


def _marcar_html(texto: str) -> str:
    """Grifa os marcadores e esmaece as citações diretas (que ficam fora da análise)."""
    cits = _spans_citacao(texto)
    dentro = lambda a, b: any(c0 <= a and b <= c1 for c0, c1 in cits)
    spans = [(a, b, "cit") for a, b in cits]
    for cat, rx in CATEGORIAS_MARCADORES.items():
        spans += [(m.start(), m.end(), cat) for m in rx.finditer(texto) if not dentro(m.start(), m.end())]
    spans += [(m.start(), m.end(), "travessão") for m in RX_TRAVESSAO.finditer(texto) if not dentro(m.start(), m.end())]
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    out, pos = [], 0
    for a, b, cat in spans:
        if a < pos:
            continue
        out.append(html.escape(texto[pos:a]))
        if cat == "cit":
            out.append(f'<span class="cit" title="citação direta: fora da análise">{html.escape(texto[a:b])}</span>')
        else:
            out.append(f'<mark class="m" title="{html.escape(cat)}">{html.escape(texto[a:b])}</mark>')
        pos = b
    out.append(html.escape(texto[pos:]))
    return "".join(out)


def _pct(x):
    return f"{x:.0f}%"


def _num(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def _valor(nome, v):
    return f"{_num(100 * v, 1)}% das palavras" if nome.startswith("fw_") else _num(v)


def _lim100(p):
    v = f"{100 * p:.1f}".replace(".", ",")
    return v[:-2] if v.endswith(",0") else v


def _esc100(p):
    return f"{100 * p:.0f}"


def gerar_html(res: dict) -> str:
    r, m = res["resumo"], res["modelo"]
    esc = html.escape
    cor = {"ia": "var(--ia)", "incerto": "var(--inc)", "humano": "var(--hum)"}
    nome_faixa = {"ia": "sinais de IA", "incerto": "incerto", "humano": "sinais humanos"}
    tem_neural = bool(res.get("neural"))

    linhas, blocos = [], []
    for i, s in enumerate(res["secoes"]):
        if not s["analisada"]:
            linhas.append(f'<tr class="off"><td>{esc(s["titulo"])}</td><td colspan="{4 if tem_neural else 3}">não analisada — {esc(s["motivo"])}</td></tr>')
            continue
        segs = s["segmentos"]
        if not segs:
            continue
        pal = sum(g["palavras_autor"] for g in segs) or 1
        media = sum(g["escore"] * g["palavras_autor"] for g in segs) / pal
        fx = Counter()
        for g in segs:
            fx[g["faixa"]] += g["palavras_autor"]
        barra = "".join(f'<span style="width:{100 * fx[f] / pal:.1f}%;background:{cor[f]}"></span>'
                        for f in ("humano", "incerto", "ia") if fx[f])
        col_n = ""
        if tem_neural:
            pn = [g["escore_neural"] for g in segs if "escore_neural" in g]
            col_n = f'<td class="num">{_esc100(statistics.fmean(pn)) if pn else "—"}</td>'
        linhas.append(f'<tr><td><a href="#s{i}">{esc(s["titulo"])}</a></td><td class="num">{pal}</td>'
                      f'<td><div class="bar">{barra}</div></td><td class="num">{_esc100(media)}</td>{col_n}</tr>')
        segs_html = []
        for j, g in enumerate(segs):
            exp = "".join(f"<li>↑ IA: {esc(rotulo(n))} ({_valor(n, v)})</li>" for n, c, v in g["pro_ia"])
            exp += "".join(f"<li>↓ humano: {esc(rotulo(n))} ({_valor(n, v)})</li>" for n, c, v in g["pro_humano"])
            if g.get("padroes"):
                exp += "<li>padrões sintáticos típicos de IA: " + ", ".join(f"<code>{esc(p)}</code>" for p, _ in g["padroes"]) + "</li>"
            zp = ""
            if "zippy_delta" in g:
                zp = f' · ZipPy: {"parecido com IA" if g["zippy_ia"] else "dentro do humano"}'
            neu = ""
            if "neural" in g:
                nm = g["neural"]
                neu = (f'<p class="neu">Neural {_esc100(g["escore_neural"])} · perplexidade {_num(nm["perplexidade"], 1)} · '
                       f'burstiness {_num(nm["burstiness"])} · GLTR top-10 {_pct(100 * nm["gltr_top10"])} · '
                       f'Fast-DetectGPT {_num(nm["fast_detectgpt"])} · Binoculars {_num(nm["binoculars"], 3)}</p>')
            segs_html.append(
                f'<div class="seg {g["faixa"]}"><div class="seghead"><span class="pill {g["faixa"]}">{nome_faixa[g["faixa"]]} · {_esc100(g["escore"])}</span>'
                f'<span class="muted">trecho {j + 1} · {g["palavras"]} palavras · estilo {_esc100(g["estilo"])} · sintaxe {_esc100(g["sintaxe"])}{zp}</span></div>'
                f'<p class="txt">{_marcar_html(g["texto"])}</p>{neu}'
                f'<details><summary>Por que esse escore</summary><ul class="exp">{exp or "<li>nenhuma medida se destacou</li>"}</ul></details></div>')
        blocos.append(f'<section id="s{i}"><h3>{esc(s["titulo"])}</h3>{"".join(segs_html)}</section>')

    sinais = res.get("sinais_humanos") or []
    sinais_html = "".join(f"<li><b>{esc(a)}</b>: {esc(b)}</li>" for a, b in sinais) or "<li>Nenhum encontrado.</li>"
    bloco_neural = ""
    if tem_neural:
        nr, cal = res["neural"]["resumo"], res["neural"]["calibracao"]
        ind = "".join(f"<li>{esc(ROTULOS_NEURAIS.get(k, k))}: AUROC {_num(v['auroc'])}</li>" for k, v in cal["individuais"].items())
        bloco_neural = f"""<section class="card"><h2>Modo neural</h2>
<p>Modelos <code>{esc(cal['observador'])}</code> (observador) e <code>{esc(cal['executor'])}</code> (executor), calibrados com {cal['n']} textos de referência:
AUROC {_num(cal['auroc_cv'])}, falso positivo {_pct(100 * cal['fpr_no_limiar'])}, IA detectada {_pct(100 * cal['tpr_no_limiar'])}.</p>
<p><b>{_pct(nr['pct_ia'])}</b> do texto com sinais de IA · {_pct(nr['pct_humano'])} com sinais humanos · {_pct(nr['pct_incerto'])} incerto.</p>
<ul>{ind}</ul></section>"""
    fontes = "".join(f"<li>{esc(k)} ({v['classe']}, {v['n']} trechos): {_pct(100 * v['acima_limiar_ia'])} acima do limiar de IA</li>"
                     for k, v in (m.get("por_fonte") or {}).items())
    th_n = '<th class="num">Neural</th>' if tem_neural else ""
    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Relatório de detecção de IA</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--card:#fff;--line:#e6e3dc;--ia:#c4532f;--inc:#c9b98f;--hum:#2f7f6b;--mark:#f4e2a0}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151513;--fg:#ecebe6;--muted:#a3a19a;--card:#1f1f1c;--line:#34332e;--ia:#e0774f;--inc:#8f8566;--hum:#5fb39c;--mark:#5c4f22}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
main{{max-width:960px;margin:0 auto;padding:32px 16px 64px}}h1{{font-size:26px;margin:0 0 4px}}h2{{font-size:19px;margin:0 0 12px}}h3{{font-size:17px;margin:28px 0 10px}}
.muted{{color:var(--muted);font-size:14px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin:16px 0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-top:16px}}.kpi{{border:1px solid var(--line);border-radius:10px;padding:12px 14px;background:var(--card)}}
.kpi b{{display:block;font-size:30px;line-height:1.1;font-variant-numeric:tabular-nums}}.verd{{font-size:20px;font-weight:600;margin:6px 0 0}}
table{{width:100%;border-collapse:collapse;font-size:15px}}td,th{{padding:8px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}}
th{{font-weight:600;color:var(--muted);font-size:13px}}.num{{text-align:right;font-variant-numeric:tabular-nums}}tr.off td{{color:var(--muted);font-size:14px}}
.bar{{display:flex;height:10px;border-radius:5px;overflow:hidden;background:var(--line);min-width:110px}}.bar span{{display:block;height:100%}}
.seg{{border-left:4px solid var(--line);padding:10px 14px;margin:10px 0;background:var(--card);border-radius:0 10px 10px 0}}
.seg.ia{{border-color:var(--ia)}}.seg.incerto{{border-color:var(--inc)}}.seg.humano{{border-color:var(--hum)}}
.seghead{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:6px}}.pill{{font-size:13px;font-weight:600;padding:2px 10px;border-radius:999px;color:#fff}}
.pill.ia{{background:var(--ia)}}.pill.incerto{{background:var(--inc);color:#1d1d1b}}.pill.humano{{background:var(--hum)}}
.txt{{margin:6px 0}}.cit{{color:var(--muted);font-style:italic}}mark.m{{background:var(--mark);color:inherit;border-radius:3px;padding:0 2px}}details{{font-size:14px;color:var(--muted)}}
.exp{{margin:6px 0 0;padding-left:18px}}.neu{{font-size:13px;color:var(--muted)}}.leg span{{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 10px}}
code{{font-size:12.5px}}a{{color:inherit}}@media (max-width:600px){{.kpi b{{font-size:24px}}td,th{{padding:6px 4px}}}}
</style></head><body><main>
<p class="muted">detector_ia {esc(res['versao'])} · {esc(res['data'])}</p>
<h1>Detecção de IA — {esc(res['arquivo'])}</h1>
<p class="verd">{esc(r['veredito'])}</p>
<div class="kpis">
<div class="kpi"><b>{_pct(r['pct_ia'])}</b><span class="muted">do texto com sinais de IA</span></div>
<div class="kpi"><b>{_pct(r['pct_humano'])}</b><span class="muted">com sinais de escrita humana</span></div>
<div class="kpi"><b>{_pct(r['pct_incerto'])}</b><span class="muted">incerto</span></div>
<div class="kpi"><b>{r['escore_medio']:.0f}</b><span class="muted">escore médio de IA (0–100)</span></div>
</div>
<p class="muted">{r['palavras_analisadas']} palavras do autor em {r['segmentos']} trechos. Citações diretas entre aspas, referências e campos de formulário ficam de fora.
O escore é uma pontuação de 0 a 100, não uma probabilidade: a partir de {_lim100(m['limiar_ia'])} o trecho é marcado como “sinais de IA”; abaixo de {_lim100(m['limiar_humano'])}, como “sinais humanos”.</p>

<section class="card"><h2>Por seção</h2>
<table><thead><tr><th>Seção</th><th class="num">Palavras</th><th>Composição<span class="leg"><span style="background:var(--hum)"></span>humano<span style="background:var(--inc)"></span>incerto<span style="background:var(--ia)"></span>IA</span></th><th class="num">Escore</th>{th_n}</tr></thead>
<tbody>{''.join(linhas)}</tbody></table></section>
{bloco_neural}
<section class="card"><h2>Sinais típicos de escrita humana</h2>
<p class="muted">Deslizes que modelos de linguagem quase nunca cometem. Não entram no escore; ajudam a ler o resultado.</p>
<ul>{sinais_html}</ul></section>

<section><h2>Trecho a trecho</h2>
<p class="muted">Grifos: conectivos, palavras avaliativas, construções de contraste, fórmulas acadêmicas e travessões. São marcas frequentes em texto de IA e também em boa escrita humana; o grifo mostra o que o detector vê, não é prova. Em itálico cinza, citações diretas, que ficam fora da análise.</p>
{''.join(blocos)}</section>

<section class="card"><h2>Confiabilidade</h2>
<p>{esc(m.get('metodo') or '')}. Treino: {m.get('n_humano')} trechos humanos escritos antes de 2022 e {m.get('n_ia')} trechos de IA.
Fora da amostra: AUROC {_num(m.get('auroc') or 0)} (estilo {_num(m.get('auroc_estilo') or 0)}, sintaxe {_num(m.get('auroc_sintaxe') or 0)}, ZipPy {_num(m.get('auroc_zippy') or 0)}).
No limiar usado, {_pct(100 * (m.get('fpr_no_limiar') or 0))} dos trechos humanos são marcados como IA (falso positivo) e {_pct(100 * (m.get('tpr_no_limiar') or 0))} dos trechos de IA são detectados.</p>
<ul>{fontes}</ul>
<p>Limites: os textos de IA de referência vêm de um único modelo; texto humano muito formal, revisado ou escrito por não nativos tende a gerar falso positivo; texto de IA reescrito por uma pessoa — ou texto humano reescrito por IA — costuma passar como humano. Trechos com menos de 80 palavras são pouco confiáveis. É um indício, nunca uma prova.</p>
</section>
</main></body></html>"""


# =============================================================================
# 11. Linha de comando
# =============================================================================


def _resumo_console(res):
    r = res["resumo"]
    print(f"\n{res['arquivo']}: {r['veredito']}")
    print(f"  sinais de IA: {r['pct_ia']:.0f}% | humano: {r['pct_humano']:.0f}% | incerto: {r['pct_incerto']:.0f}% | escore médio: {r['escore_medio']:.0f}/100")
    for s in res["secoes"]:
        if s["analisada"] and s["segmentos"]:
            ps = " ".join(f"{100 * g['escore']:.0f}" for g in s["segmentos"])
            print(f"  - {s['titulo'][:48]:48s} {ps}")
    if res.get("neural"):
        n = res["neural"]["resumo"]
        print(f"  neural → sinais de IA: {n['pct_ia']:.0f}% | humano: {n['pct_humano']:.0f}%")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="detector_ia", description="Detector de texto gerado por IA (português).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analisar", help="analisa um .pdf, .docx, .txt ou .md")
    a.add_argument("arquivo")
    a.add_argument("--saida", help="relatório HTML (padrão: <arquivo>_relatorio_ia.html)")
    a.add_argument("--json", help="salva o resultado completo em JSON")
    a.add_argument("--neural", action="store_true", help="acrescenta GPTZero/GLTR/Fast-DetectGPT/Binoculars (baixa modelos)")
    a.add_argument("--observador", default="Qwen/Qwen2.5-0.5B")
    a.add_argument("--executor", default="Qwen/Qwen2.5-0.5B-Instruct")
    a.add_argument("--dispositivo", default=None, help="cpu, cuda ou mps (padrão: automático)")
    a.add_argument("--modelo", default=str(MODELO_PADRAO), help="modelo estatístico em JSON")
    a.add_argument("--incluir-tudo", action="store_true", help="analisa também campos curtos, listas e referências")

    x = sub.add_parser("texto", help="analisa texto lido da entrada padrão")
    x.add_argument("--saida")
    x.add_argument("--modelo", default=str(MODELO_PADRAO))

    t = sub.add_parser("treinar", help="retreina o classificador estatístico com seus próprios corpora")
    t.add_argument("--humano", nargs="+", required=True, help=".jsonl ou pastas com textos humanos")
    t.add_argument("--ia", nargs="+", required=True, help=".jsonl ou pastas com textos de IA")
    t.add_argument("--saida", default=str(MODELO_PADRAO))

    c = sub.add_parser("calibrar-neural", help="calibra os limiares do modo neural para um par de modelos")
    c.add_argument("--observador", default="Qwen/Qwen2.5-0.5B")
    c.add_argument("--executor", default="Qwen/Qwen2.5-0.5B-Instruct")
    c.add_argument("--dispositivo", default=None)
    c.add_argument("--n", type=int, default=40, help="textos de referência por classe")

    args = ap.parse_args(argv)
    if args.cmd == "analisar":
        res = analisar_arquivo(args.arquivo, args.modelo, args.neural, args.observador, args.executor,
                               args.dispositivo, args.incluir_tudo)
        saida = Path(args.saida) if args.saida else Path(args.arquivo).with_name(Path(args.arquivo).stem + "_relatorio_ia.html")
        saida.write_text(gerar_html(res), encoding="utf-8")
        if args.json:
            Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        _resumo_console(res)
        print(f"\nRelatório: {saida}")
    elif args.cmd == "texto":
        texto = sys.stdin.read()
        res = analisar_arquivo(None, args.modelo, secoes=[_marcado(Secao("Texto", [p for p in re.split(r"\n\s*\n", texto) if p.strip()]))])
        _resumo_console(res)
        if args.saida:
            Path(args.saida).write_text(gerar_html(res), encoding="utf-8")
    elif args.cmd == "treinar":
        hum = carregar_corpus(args.humano, "humano")
        ia = carregar_corpus(args.ia, "ia")
        print(f"{len(hum)} trechos humanos, {len(ia)} trechos de IA")
        treinar_modelo(hum, ia, saida=args.saida)
        print(f"Modelo salvo em {args.saida}")
    elif args.cmd == "calibrar-neural":
        av = AvaliadorNeural(args.observador, args.executor, args.dispositivo)
        cal = calibrar_neural(av, n_por_classe=args.n, forcar=True)
        print(json.dumps({k: cal[k] for k in ("auroc_cv", "fpr_no_limiar", "tpr_no_limiar", "limiar_ia")}, indent=1))


def _marcado(s):
    _marcar_secao(s)
    return s


if __name__ == "__main__":
    main()
