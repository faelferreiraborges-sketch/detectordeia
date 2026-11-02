# detector_ia — detector de texto gerado por IA, em português

Programa em Python que estima, trecho a trecho, se um texto tem sinais de ter sido gerado ou reescrito por IA. Lê `.pdf` (inclusive fichas em tabela, como a do NPGA), `.docx`, `.txt` e `.md`, e gera um relatório em HTML.

Nenhum detector prova autoria. O resultado é um indício estatístico com taxa de erro medida, e o relatório mostra essa taxa.

## Instalação

Precisa de Python 3.9 ou mais novo.

```bash
pip install -r requirements.txt
```

## Uso

```bash
# modo estatístico: offline, sem GPU, alguns segundos
python detector_ia.py analisar meu_texto.pdf

# relatório em outro lugar e resultado completo em JSON
python detector_ia.py analisar meu_texto.docx --saida relatorio.html --json resultado.json

# acrescenta os métodos neurais (baixa ~2 GB de modelos do Hugging Face na primeira vez)
pip install torch transformers
python detector_ia.py analisar meu_texto.pdf --neural
python detector_ia.py analisar meu_texto.pdf --neural --observador Qwen/Qwen2.5-1.5B --executor Qwen/Qwen2.5-1.5B-Instruct

# texto colado
python detector_ia.py texto < texto.txt

# retreinar com os seus próprios exemplos humanos e de IA (precisa de scikit-learn)
python detector_ia.py treinar --humano pasta_textos_humanos/ --ia pasta_textos_ia/
```

Citações diretas entre aspas (4 palavras ou mais), listas de referências e campos de formulário ficam fora da análise: não são texto do autor.

## Como ler o relatório

- **Escore (0 a 100)** por trecho de 90 a 240 palavras. Não é probabilidade.
- **Sinais de IA**: escore acima do limiar em que só 5% dos trechos humanos de teste ficam. **Sinais humanos**: abaixo do limiar em que 90% dos trechos de IA ficam acima. Entre os dois: **incerto**.
- **% do texto**: soma das palavras dos trechos em cada faixa.
- **Por que esse escore**: as medidas que mais empurraram o trecho para um lado ou para o outro.
- **Sinais típicos de escrita humana**: deslizes de digitação e concordância. Não entram no escore.

## Métodos

| Método | Ferramenta de origem | O que mede | Modo |
|---|---|---|---|
| Perplexidade e burstiness | GPTZero | quão previsível é o texto e quanto isso varia entre frases | neural |
| Posição do token (top-10/100/1000) | GLTR | se as palavras escolhidas são sempre as mais prováveis | neural |
| Discrepância de amostragem | Fast-DetectGPT | se o texto fica no "pico" de probabilidade do modelo | neural |
| Perplexidade cruzada | Binoculars | perplexidade normalizada pela de um segundo modelo | neural |
| Compressão contra referências | ZipPy | se o texto comprime melhor junto de textos de IA ou de humanos | estatístico |
| Estilometria (25 medidas + 120 palavras funcionais) | classificadores como Ghostbuster e Pangram | ritmo das frases, pontuação, conectivos, vocabulário | estatístico |
| N-gramas do esqueleto sintático | idem | padrões de sintaxe com o assunto apagado | estatístico |

O escore estatístico é a média ponderada de dois classificadores (estilo 35%, sintaxe 65%). O esqueleto troca palavras de conteúdo por classes e neutraliza a 1ª pessoa, para o detector medir estilo e não assunto ou gênero textual. No modo neural, os limiares são recalibrados para o par de modelos escolhido com os textos de `referencias/`.

## Desempenho medido

Validação cruzada agrupada por documento (5 dobras × 3 repetições), com 2.979 trechos humanos escritos antes de 2022 e 57 trechos de IA:

- AUROC 0,92 (estilo 0,89; sintaxe 0,92; ZipPy 0,78)
- no limiar usado, 5% dos trechos humanos viram falso positivo e 77% dos trechos de IA são detectados
- texto gerado do zero por IA em registro acadêmico: 100% detectado; texto humano **reescrito** por IA: 36%
- falso positivo por gênero humano: jurídico 1%, acadêmico 3%, jornal 3% a 8%, redação ENEM 9%

**Ponto fraco: gênero não visto no treino.** Quando um gênero humano inteiro fica fora do treino, o falso positivo nele sobe para 4% a 54% (redações ENEM: 54%). Por isso, num gênero que o detector não conhece, um escore alto merece desconfiança. Retreinar com exemplos humanos do seu gênero (`treinar`) é o que mais melhora o resultado.

## Limites

- Os textos de IA de referência foram escritos por um único modelo (Claude, Anthropic). Outros modelos podem ter estilo diferente.
- Texto humano muito formal, revisado ou escrito por não nativos tende a gerar falso positivo. Texto de IA reescrito por uma pessoa, ou texto humano reescrito por IA, tende a passar como humano.
- Trechos com menos de 80 palavras são pouco confiáveis.
- Nunca use o resultado como prova contra ninguém.

## Dados de referência e licenças

`referencias/humano.jsonl` reúne 200 trechos (40 por fonte) de: UD Portuguese-PetroGold (CC BY-SA 4.0), UD Portuguese-Porttinari (CC BY 4.0), UD Portuguese-Bosque/CETENFolha (CC BY-SA 4.0), Essay-BR estendido (MIT, Marinho, Anchiêta e Moura, 2022) e LeNER-Br (MIT, Luz de Araujo et al., 2018). Esse arquivo segue a licença CC BY-SA 4.0. `referencias/ia.jsonl` reúne os 57 trechos de IA, escritos para este projeto.

## Arquivos

- `detector_ia.py`: o programa inteiro
- `modelo_estatistico.json`: classificador treinado, limiares e métricas
- `referencias/`: textos de referência (ZipPy e calibração do modo neural)
- `testes/teste_basico.py`: testes rápidos do modo estatístico
- `testes/teste_neural_modelo_minusculo.py`: testa o modo neural com modelos minúsculos, sem internet
