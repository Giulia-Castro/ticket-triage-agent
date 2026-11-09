# Ticket Triage Agent

![CI](https://github.com/<seu-usuario>/ticket-triage-agent/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Agente de IA que **classifica, prioriza e encaminha chamados de suporte** em segundos,
com **métricas de acerto, custo e latência medidas** contra um dataset rotulado.

> Um chamado chega em texto livre → o agente consulta a base de conhecimento e chamados
> parecidos → devolve categoria, severidade, fila, SLA, resumo e uma sugestão de resposta.
> Regras de negócio determinísticas decidem a fila final; casos incertos vão para revisão humana.

---

## Por que existe

Na triagem manual, um analista lê cada chamado, decide a categoria e a urgência, procura o
artigo certo e encaminha para a fila. É repetitivo, inconsistente entre pessoas e atrasa
justamente os chamados críticos. Este projeto automatiza essa primeira leitura e **mede**
quanto acerta, quanto custa e quanto tempo leva.

## Resultados medidos

Dataset: [`evals/dataset.jsonl`](evals/dataset.jsonl), com 48 chamados sintéticos em pt-BR
(8 por categoria, incluindo casos ambíguos, dados pessoais e uma tentativa de prompt injection).

| Abordagem | Acerto categoria | Macro F1 | Severidade exata | Severidade ±1 nível | Custo/chamado | Latência p50 |
|---|---|---|---|---|---|---|
| Baseline por palavras-chave | **79,2%** | 0,792 | **50,0%** | 95,8% | US$ 0 | 1 ms |
| Claude + prompt v1 (simples) | _rode o eval_ | | | | | |
| Claude + prompt v2 (estruturado) | _rode o eval_ | | | | | |

Para preencher as linhas do Claude, rode `python -m evals.run_evals --agent claude --prompt-version v1`
e depois `v2` (veja [Como medir](#como-medir)). Os relatórios completos ficam em
[`evals/results/`](evals/results/).

> **Leitura honesta da baseline:** as palavras-chave foram escritas pela mesma pessoa que
> escreveu o dataset, então 79% é um teto otimista para regras simples. O ponto fraco
> aparece em **BUG** (recall 0,38) e na **severidade** (50%), que dependem de interpretar
> impacto, não de palavras soltas. É aí que o LLM precisa provar valor.

### Ganho de tempo (como estimar)

| Variável | Como obter |
|---|---|
| Chamados por dia (`N`) | Volume real da sua operação |
| Minutos de triagem manual por chamado (`t`) | Cronometre 20 triagens reais |
| Taxa de revisão humana (`r`) | Métrica `Sent to human review` do eval ou `needsReview` da API |

Horas economizadas por mês ≈ `N × t × (1 − r) × 22 / 60`. Exemplo hipotético: 200 chamados/dia
× 2 min × 80% automatizados ≈ **117 h/mês**. Substitua pelos seus números medidos antes de
publicar qualquer resultado.

---

## Arquitetura

```mermaid
flowchart LR
    C[Cliente / Service Desk] -->|POST /v1/triages| API[FastAPI]
    API --> UC[TriageTicketUseCase]
    UC --> M[Mascaramento LGPD]
    M --> AG[Agente Claude]
    AG -->|tool| KB[(Base de conhecimento)]
    AG -->|tool| H[(Chamados similares)]
    AG -->|JSON validado| UC
    UC --> RP[RoutingPolicy<br/>regras determinísticas]
    RP --> DB[(PostgreSQL)]
    API -->|GET /v1/metrics/summary| D[Dashboard / Projeto 3]
```

Clean Architecture + DDD:

```
src/triage_agent/
├── domain/          # Entidades, regras, mascaramento. Sem framework, banco ou HTTP.
├── application/     # Casos de uso e portas (interfaces).
├── infrastructure/  # Claude, SQLAlchemy, base de conhecimento, logs, circuit breaker.
└── api/             # FastAPI, DTOs, RFC 7807, i18n (en / pt-BR), autenticação.
prompts/triage/      # Prompts versionados (v1, v2) + CHANGELOG.
evals/               # Dataset rotulado, métricas e runner.
migrations/          # Alembic (portável PostgreSQL / Oracle / SQL Server).
```

### Decisões de projeto

| Decisão | Por quê |
|---|---|
| LLM interpreta, **código decide** | Fila, SLA e escalonamento ficam em `RoutingPolicy`, testável e auditável. O modelo não define a fila sozinho. |
| **Structured outputs** + validação Pydantic | A resposta segue um JSON Schema; mesmo assim é revalidada antes de virar objeto de domínio. |
| Ferramentas **somente leitura** | O agente pode consultar, mas nunca alterar nada. Limite de rodadas de ferramenta (`AGENT_MAX_TOOL_ROUNDS`). |
| IDs de artigo filtrados | O agente só pode citar artigos que existem: ids inventados são descartados. |
| Confiança < 0,6 → **revisão humana** | Erro caro vai para uma pessoa em vez de ser encaminhado errado. |
| Baseline sem LLM | Toda versão de prompt é comparada com uma referência barata. Também permite rodar sem chave de API. |
| Feedback humano + `rowversion` | `PATCH /feedback` mede acurácia **em produção**, com lock otimista contra edição concorrente. |
| Erros de negócio via `Result`, HTTP via **RFC 7807** | Exceção só para falha técnica (LLM fora do ar → 503 com `Retry-After`). |

---

## Engenharia de prompt

Os prompts ficam em [`prompts/triage/`](prompts/triage/), um arquivo por versão. Uma versão
nunca é editada depois de publicada, o que mantém os evals reproduzíveis. O
[`CHANGELOG`](prompts/CHANGELOG.md) registra hipótese, mudança e resultado.

O **v2** é estruturado em:

1. **Papel e contexto**: o que é o produto e por que a triagem importa.
2. **Fronteira de dados**: o texto dentro de `<ticket>` é dado, nunca instrução (defesa contra prompt injection; o chamado `T046` do dataset testa isso).
3. **Definições de categoria + 5 regras de fronteira** (ex.: erro da API é INTEGRATION, não BUG).
4. **Rubrica de severidade** baseada em impacto e existência de contorno.
5. **Política de ferramentas**: quando buscar e a proibição de citar artigo não retornado.
6. **Regras da resposta sugerida**: idioma do chamado, sem promessas, nunca pedir senha.
7. **Calibração de confiança** ligada ao limiar de revisão humana.
8. **Exemplos** curtos.

---

## Segurança e LGPD

- **Mascaramento antes de tudo**: e-mail, CPF, CNPJ, telefone, cartão e placa de veículo são substituídos (`[EMAIL]`, `[CPF]`...) **antes** de ir para o LLM, o banco e os logs.
- **Minimização**: a API nunca devolve o texto do chamado, só a classificação.
- **Auditoria**: `createdby`/`updatedby` vêm do rótulo da chave de API; leituras geram log `triage.read`.
- **Autenticação negando por padrão**: sem `API_KEYS` configurado, ninguém entra. Comparação em tempo constante.
- **Segredos só em variável de ambiente** (`.env` fora do Git; GitHub Secrets no CI).
- **Logs JSON** com `correlationId` (cabeçalho `X-Correlation-ID`) e mascaramento aplicado no formatter.
- **Resiliência**: timeout, retry do SDK (classificação é idempotente) e circuit breaker.

---

## Como rodar

### Pré-requisitos
Python 3.12+ e Git. Docker é opcional.

### Local (sem custo, modo baseline)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # Windows: copy .env.example .env
```

Edite o `.env` e defina `API_KEYS=local-dev:<uma-chave-longa>`. Para gerar uma chave:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

```bash
alembic upgrade head
uvicorn triage_agent.api.main:app --reload
```

Abra a documentação interativa em http://localhost:8000/docs.

### Com o Claude
No `.env`, defina `TRIAGE_AGENT=claude` e `ANTHROPIC_API_KEY=<sua-chave>`
(crie em https://platform.claude.com). Reinicie a API.

### Docker (API + PostgreSQL)

```bash
docker compose up --build
```

### Exemplo de chamada

```bash
curl -X POST http://localhost:8000/v1/triages \
  -H "X-API-Key: <sua-chave>" -H "Content-Type: application/json" \
  -H "Accept-Language: pt-BR" \
  -d '{"subject": "Webhook com 401", "body": "Depois que trocamos o token o webhook de ordem concluída recebe 401."}'
```

Resposta (resumida):

```json
{
  "id": "6f1c...",
  "status": "ROUTED",
  "queue": "INTEGRATIONS",
  "category": "INTEGRATION",
  "severity": "MEDIUM",
  "slaHours": 24,
  "confidence": 0.92,
  "kbArticleIds": ["KB-008"],
  "usage": { "model": "claude-opus-5-5", "promptVersion": "v2", "costUsd": "0.0042", "latencyMs": 3100 },
  "version": 1
}
```

| Método | Rota | Descrição |
|---|---|---|
| `POST` | `/v1/triages` | Tria um chamado |
| `GET` | `/v1/triages/{id}` | Consulta uma triagem |
| `GET` | `/v1/triages?limit=20` | Lista as mais recentes |
| `PATCH` | `/v1/triages/{id}/feedback` | Correção humana (`category`, `severity`, `version`) |
| `GET` | `/v1/metrics/summary` | Volume, acurácia por feedback, custo, latência p95 |
| `GET` | `/health/live`, `/health/ready` | Saúde (sem autenticação) |

---

## Como medir

```bash
# Baseline (grátis, offline)
python -m evals.run_evals --agent keyword

# Claude: compare versões de prompt (custa tokens; peça --limit 10 para testar antes)
python -m evals.run_evals --agent claude --prompt-version v1 --yes
python -m evals.run_evals --agent claude --prompt-version v2 --yes
```

Métricas geradas: acurácia e macro F1 de categoria, precisão/recall por categoria, matriz de
confusão, severidade exata e ±1 nível, taxa de revisão humana, latência p50/p95, custo total
e por chamado. Cada execução salva um JSON em `evals/results/`.

## Testes e qualidade

```bash
pytest            # unitários + integração, cobertura mínima de 80% (atual: ~97%)
ruff check .      # lint (inclui regras de segurança Bandit)
ruff format .     # formatação
```

O CI no GitHub Actions roda lint, testes, migrations contra PostgreSQL real e o eval baseline
a cada push.

## Limitações e próximos passos

- Dataset sintético com 48 chamados: amplie para 200+ com chamados reais **anonimizados** antes de tirar conclusões.
- Similaridade lexical simples; para volume alto, trocar por embeddings (pgvector).
- Sem rate limiting na API; em produção, colocar atrás de um API gateway.
- Próximo: dashboard (projeto 3) consumindo `/v1/metrics/summary`.

## Licença

[MIT](LICENSE)
