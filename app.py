"""Interface web do detector_ia para a Vercel (Flask).

A Vercel encontra a variável `app` neste arquivo e a publica como função Python.
Roda só o modo estatístico: o modo neural precisa de ~2 GB de modelos e não cabe numa função.
"""

from __future__ import annotations

import html
import os
import tempfile
from pathlib import Path

from flask import Flask, request

import detector_ia as d

app = Flask(__name__)
# a Vercel recusa corpos de requisição acima de 4,5 MB; o limite aqui fica um pouco abaixo
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

EXTENSOES = {".pdf", ".docx", ".txt", ".md"}
MIN_PALAVRAS = 80


ESTILO = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--card:#fff;--line:#e6e3dc;--ia:#c4532f;--hum:#2f7f6b;--erro-bg:#fbeae4}
@media (prefers-color-scheme:dark){:root{--bg:#151513;--fg:#ecebe6;--muted:#a3a19a;--card:#1f1f1c;--line:#34332e;--ia:#e0774f;--hum:#5fb39c;--erro-bg:#3a2219}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:760px;margin:0 auto;padding:40px 16px 64px}
h1{font-size:28px;line-height:1.2;margin:0 0 8px}
.lead{color:var(--muted);margin:0 0 24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin:16px 0}
label{display:block;font-weight:600;margin:0 0 8px}
textarea{width:100%;min-height:260px;padding:12px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);font:inherit;resize:vertical}
textarea:focus,input[type=file]:focus-visible,button:focus-visible{outline:2px solid var(--hum);outline-offset:2px}
.ou{display:flex;align-items:center;gap:12px;color:var(--muted);font-size:14px;margin:20px 0}
.ou::before,.ou::after{content:"";flex:1;height:1px;background:var(--line)}
input[type=file]{font:inherit;color:var(--muted);max-width:100%}
.dica{color:var(--muted);font-size:14px;margin:8px 0 0}
button{margin-top:20px;font:inherit;font-weight:600;padding:12px 22px;border:0;border-radius:8px;background:var(--fg);color:var(--bg);cursor:pointer}
button[disabled]{opacity:.6;cursor:progress}
.erro{background:var(--erro-bg);border:1px solid var(--ia);border-radius:8px;padding:12px 14px;margin:0 0 16px}
.aviso{font-size:14px;color:var(--muted)}
.aviso li{margin:4px 0}
"""

PAGINA = """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Detector de IA</title><style>{estilo}</style></head><body><main>
<h1>Detector de texto gerado por IA</h1>
<p class="lead">Estima, trecho a trecho, se um texto em português tem sinais de ter sido gerado ou reescrito por IA, e mostra por quê.</p>
{erro}
<form class="card" method="post" action="/analisar" enctype="multipart/form-data" id="f">
  <label for="texto">Cole o texto</label>
  <textarea id="texto" name="texto" placeholder="Pelo menos {minimo} palavras. Parágrafos separados por linha em branco.">{texto}</textarea>
  <div class="ou">ou</div>
  <label for="arquivo">Envie um arquivo</label>
  <input type="file" id="arquivo" name="arquivo" accept=".pdf,.docx,.txt,.md">
  <p class="dica">.pdf, .docx, .txt ou .md, até 4 MB. Se você enviar um arquivo, o texto colado é ignorado.</p>
  <button type="submit" id="b">Analisar</button>
</form>
<section class="aviso">
<ul>
<li>Nenhum detector prova autoria. O resultado é um indício estatístico, com taxa de erro medida e mostrada no relatório.</li>
<li>Texto humano muito formal ou revisado tende a gerar falso positivo; texto de IA reescrito por uma pessoa tende a passar como humano.</li>
<li>O texto é analisado na hora e não fica guardado.</li>
</ul>
</section>
</main>
<script>
document.getElementById("f").addEventListener("submit",function(){{var b=document.getElementById("b");b.disabled=true;b.textContent="Analisando…";}});
window.addEventListener("pageshow",function(){{var b=document.getElementById("b");b.disabled=false;b.textContent="Analisar";}});
</script>
</body></html>"""


def pagina(erro: str = "", texto: str = "", status: int = 200):
    bloco = f'<p class="erro" role="alert">{html.escape(erro)}</p>' if erro else ""
    corpo = PAGINA.format(estilo=ESTILO, erro=bloco, texto=html.escape(texto), minimo=MIN_PALAVRAS)
    return corpo, status, {"Content-Type": "text/html; charset=utf-8"}


def secoes_do_arquivo(arq):
    nome = Path(arq.filename or "").name
    ext = Path(nome).suffix.lower()
    if ext not in EXTENSOES:
        raise ValueError("Formato não aceito. Envie .pdf, .docx, .txt ou .md.")
    fd, caminho = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as f:
            arq.save(f)
        return nome, d.ler_documento(caminho)
    finally:
        try:
            os.remove(caminho)
        except OSError:
            pass


def secoes_do_texto(texto: str):
    secoes = d.ler_texto_simples(texto)
    for s in secoes:
        d._marcar_secao(s)
    return secoes


@app.get("/")
def inicio():
    return pagina()


@app.post("/analisar")
def analisar():
    texto = (request.form.get("texto") or "").strip()
    arq = request.files.get("arquivo")
    tem_arquivo = bool(arq and arq.filename)

    if not tem_arquivo and not texto:
        return pagina("Cole um texto ou envie um arquivo.", status=400)
    if not tem_arquivo and len(texto.split()) < MIN_PALAVRAS:
        return pagina(f"O texto tem {len(texto.split())} palavras. Cole pelo menos {MIN_PALAVRAS} para um resultado minimamente confiável.",
                      texto, status=400)

    try:
        if tem_arquivo:
            nome, secoes = secoes_do_arquivo(arq)
        else:
            nome, secoes = "texto colado", secoes_do_texto(texto)
        res = d.analisar_arquivo(None, secoes=secoes)
    except ValueError as e:
        return pagina(str(e), texto, status=400)
    except Exception:
        app.logger.exception("falha ao analisar")
        return pagina("Não consegui ler esse arquivo. Se for PDF escaneado (imagem), ele não tem texto para analisar; "
                      "tente copiar e colar o texto.", texto, status=422)

    if res["resumo"]["segmentos"] == 0:
        return pagina("Não encontrei texto corrido do autor para analisar. Listas, referências, citações e campos curtos "
                      "ficam de fora.", texto, status=422)

    res["arquivo"] = nome
    relatorio = d.gerar_html(res)
    voltar = '<main><p class="muted"><a href="/">← Nova análise</a></p>'
    relatorio = relatorio.replace("<main>", voltar, 1)
    return relatorio, 200, {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store"}


@app.errorhandler(413)
def grande_demais(_e):
    return pagina("Arquivo grande demais. O limite é 4 MB; se for PDF, tente salvar sem imagens ou cole o texto.", status=413)


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
