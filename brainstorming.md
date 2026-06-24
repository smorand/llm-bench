# brainstorming : llm-bench

Outil de benchmark de performance et de qualité pour endpoints LLM compatibles OpenAI.

Date : 2026-06-24
Statut : document de cadrage, entrée pour `project:spec`.

---

## 1. Objectif

Construire un outil en ligne de commande qui mesure la performance (latence, débit, fiabilité, coût) et optionnellement la qualité de sortie de modèles LLM exposés via une API compatible OpenAI. L'outil doit produire des mesures déterministes dans le temps (runs à durée fixe), reproductibles, et restituées sous forme de rapport partageable.

Le bench couvre aussi bien des modèles auto-hébergés (vLLM, TGI, SGLang, llama.cpp) que des API hébergées, du moment qu'elles parlent le protocole OpenAI (`/v1/chat/completions` en streaming).

### Décisions structurantes actées

| Sujet | Choix |
|---|---|
| Langage / runtime | Python, asyncio + httpx streaming |
| Outils pour prompts avec tooling | Outils mock déterministes (MCP/tools factices) |
| Restitution | Rapport HTML statique Plotly + résumé terminal rich |
| Stockage brut | JSONL par requête, roll-up Parquet, requêtable via DuckDB |
| Stratégie de build | Phasé, mais toutes les phases implémentées en autonomie |
| Modèles de concurrence | closed-loop (défaut) ET open-loop Poisson, les deux livrés |
| Goodput | Oui, profils SLO interactif + relaxé, surchargeables |
| Statistiques | p50/p90/p95/p99 + mean/min/max/std (p99.9 si échantillons suffisants) |
| Coût | Oui, pricing optionnel par modèle dans le registre |
| Évaluation qualité | Embedding cosine par défaut, LLM-judge optionnel échantillonné, async |

---

## 2. Synthèse de la recherche (fondements)

Quatre axes de recherche ont convergé. Les principes directeurs retenus :

### 2.1 Modèle mental prefill / decode

La génération LLM a deux phases, et presque toute métrique se rattache à l'une d'elles :

| Phase | Ce qui se passe | Goulot | Varie avec | Métrique |
|---|---|---|---|---|
| Prefill | traitement du prompt en une passe, construction du KV cache, 1er token | compute-bound | longueur entrée (ISL) | TTFT |
| Decode | génération autorégressive token par token | memory-bandwidth-bound | longueur sortie (OSL) | TPOT / ITL |

C'est pourquoi le batching fonctionne : le decode laisse du compute GPU inutilisé, donc ajouter des requêtes concurrentes augmente le débit agrégé à bas coût, jusqu'à saturation. Le prefill, lui, est déjà compute-bound : gros prompts + forte concurrence gonflent directement le TTFT.

### 2.2 Outils de référence (ce qu'on imite)

- **vLLM `bench serve`** : TTFT, TPOT, ITL, E2EL (mean/median/p99), débits requêtes/tokens, goodput. Compte les tokens réels via `usage`. Modes closed (`--max-concurrency`) et open (`--request-rate` + `--burstiness`).
- **LLMPerf (Ray)** : TTFT, ITL, E2E, output tok/s, requêtes/min sur p25/p50/p90/p95/p99. Deux fichiers de sortie (résumé + détail par requête).
- **NVIDIA GenAI-Perf** : ajoute time-to-second-token, scatter ISL/OSL, sliding window pour exclure warmup/cooldown, goodput.
- **MLPerf Inference** : scénarios Offline / Server (SLO TTFT+TPOT au p99). Sert d'ancrage pour nos seuils SLO par défaut.

### 2.3 Pièges méthodologiques à encoder

1. **ITL ≠ TPOT selon les outils** (pondération et inclusion ou non du TTFT diffèrent). On fige notre convention et on la documente : TPOT request-weighted excluant le 1er token ; ITL token-weighted, distribution des écarts inter-tokens hors TTFT.
2. **Jamais calculer le débit depuis `max_tokens`** : toujours les `completion_tokens` réels renvoyés par le serveur.
3. **tiktoken ≠ tokenizer servi** : faire confiance au champ `usage` du serveur en source de vérité, ou charger le tokenizer exact du modèle si comptage client nécessaire.
4. **Closed-loop masque les queues de latence** (coordinated omission) : proposer aussi l'open-loop.
5. **Réutilisation de prompt = biais de cache préfixe** : randomiser, préfixe unique par requête, vérifier `cached_tokens == 0` sur les runs à froid.
6. **Le débit n'a aucun sens sans ISL/OSL + concurrence + état du cache reportés.**

---

## 3. Métriques

### 3.1 Latence (reportées en distribution : mean, min, max, std, p50, p90, p95, p99 ; p99.9 si assez d'échantillons)

| Métrique | Définition | Note |
|---|---|---|
| TTFT (Time To First Token) | `t_premier_token_contenu − t_envoi` | inclut file d'attente + prefill + réseau + 1er token. On ignore le 1er chunk vide ne portant que le rôle. |
| Time To Second Token | `t_2e_token − t_1er_token` | détecte un 1er token rapide suivi d'un stall |
| TPOT (Time Per Output Token) | `(E2E − TTFT) / (output_tokens − 1)` | vitesse de décodage, request-weighted, exclut le 1er token |
| ITL (Inter-Token Latency) | distribution des écarts entre tokens consécutifs | token-weighted, hors TTFT ; le p99 révèle les à-coups |
| E2E (Time To Complete) | `t_dernier_token − t_envoi` | latence totale |
| Latence normalisée | `E2E / output_tokens` | comparable entre runs de longueurs différentes |

### 3.2 Débit (reportés en valeur agrégée par run/niveau)

| Métrique | Définition |
|---|---|
| tok/s par user (output) | `output_tokens / (E2E − TTFT)` ; vitesse ressentie par un utilisateur |
| tok/s système (output) | `Σ output_tokens / fenêtre_steady` ; débit agrégé, métrique de dimensionnement/coût |
| RPS | `requêtes_complétées / fenêtre_steady` |
| tok/s total | `Σ (input + output) / fenêtre_steady` |

Note clé : tok/s par user *baisse* quand la concurrence monte, tok/s système *monte puis sature*. Le croisement des deux courbes est un livrable central.

### 3.3 Goodput (SLO-aware)

Débit ne comptant que les requêtes respectant *tous* les seuils SLO simultanément. Deux profils par défaut (surchargeables dans `config.yaml`) :

- **interactif** : TTFT < 500 ms ET TPOT < 50 ms ET E2E < 5 s
- **relaxé / batch** : TTFT < 2 s ET TPOT < 200 ms (ancré sur MLPerf Llama2-70B)

Reporté comme : requêtes conformes / s, et taux d'attainment (% de requêtes conformes).

### 3.4 Fiabilité (métriques de tête, première classe)

- taux de succès, taux d'erreur global
- décomposition par type : `429_rate_limited`, `timeout`, `http_error` (par code), `connection_error`, `malformed_stream`
- compte complété / échoué par niveau de concurrence

Règle : les latences (TTFT/ITL/etc.) ne sont calculées que sur les requêtes réussies ; les échecs sont reportés séparément. Aucune latence p99 n'est présentée sans le taux d'erreur à côté.

### 3.5 Comptabilité des tokens

- `input_tokens`, `output_tokens` (réels, depuis `usage`, jamais `max_tokens`)
- `cached_tokens` (depuis `prompt_tokens_details`) : tracké, assertion `== 0` sur runs à froid
- `reasoning_tokens` (depuis `completion_tokens_details`) : tracké en dimension séparée pour les modèles à raisonnement
- `stream_options: {include_usage: true}` obligatoire pour récupérer `usage` en streaming ; fallback comptage des deltas + flag `usage-incomplete` si stream interrompu

### 3.6 Coût (si pricing présent dans le registre)

Champs `price_input` / `price_output` optionnels par modèle ($/1M tokens). Dérive : coût par requête, coût pour 1000 requêtes, $/token sortie, coût par unité de throughput, et (si éval activée) coût par point de qualité. Ignoré silencieusement si pricing absent.

### 3.7 Qualité (si sortie attendue fournie, voir section 7)

- score de similarité cosine (distribution + seuil calibré)
- verdict LLM-judge (binaire ou 3 niveaux) si activé
- taux de réussite qualité, couverture d'évaluation (`judged / total`)
- corrélations qualité × concurrence et qualité × latence (la valeur unique de co-localiser éval et load test)

---

## 4. Dimensions (variables contrôlées, taggées sur chaque requête)

Toute métrique est filtrable/segmentable selon ces axes :

| Dimension | Rôle | Contrôle |
|---|---|---|
| Niveau de concurrence | axe principal du sweep | `concurrency_levels: [1,2,5,10,...]` |
| Output sequence length (OSL) | **domine la latence E2E** (~100x l'effet d'un token d'entrée) | `max_tokens` + `ignore_eos: true` pour forcer la longueur |
| Input sequence length (ISL) | pilote le TTFT (prefill) | buckets de longueur d'entrée |
| Catégorie de tâche | profils de tokens différents | coding / synthèse / tool-use / vision |
| État du cache | biais de TTFT | mode cache-busting (préfixe unique) vs cache-friendly ; `cached_tokens` observé |
| Modèle d'arrivée | honnêteté des queues | closed-loop N vs open-loop req/s + burstiness |
| Streaming | mode de mesure | on (défaut, pour TTFT/ITL) / off |
| Reasoning tokens | explose la latence des modèles à raisonnement | dimension séparée |
| Modèle | comparaison | entrée du registre |

Point d'attention majeur : **verrouiller l'OSL en priorité** (via `ignore_eos` + `max_tokens` fixe), faute de quoi les chiffres de latence sont du bruit. L'attention initiale portée à la taille d'entrée est secondaire face à la longueur de sortie.

---

## 5. Configuration

### 5.1 Registre de modèles (`config.yaml`)

Interpolation de variables d'environnement supportée partout via la syntaxe `$ENV:NOM_VARIABLE` (résolue au chargement, jamais de secret en clair dans le fichier).

```yaml
# Registre des endpoints sous test (SUT)
models:
  - name: gpt-oss-120b-local
    base_url: http://localhost:8000/v1
    model: openai/gpt-oss-120b
    api_key: $ENV:VLLM_API_KEY        # interpolation env ; peut être omis si non requis
    tokenizer: openai/gpt-oss-120b    # tokenizer HF exact (optionnel, sinon usage serveur)
    supports_vision: false
    supports_tools: true
    price_input: 0.0                  # $/1M tokens (optionnel)
    price_output: 0.0
    extra_headers: {}                 # auth custom éventuelle

  - name: claude-haiku-ibm
    base_url: $ENV:IBM_ICA_BASE_URL
    model: ibm/claude-haiku-4-5
    api_key: $ENV:IBM_ICA_API_KEY
    supports_vision: true
    supports_tools: true

# Évaluation de la qualité (voir section 7). Juge LLM + embedding.
evaluation:
  judge:
    model:
      url: $ENV:IBM_ICA_BASE_URL
      api_key: $ENV:IBM_ICA_API_KEY
      model: ibm/claude-haiku-4-5     # défaut juge via la gateway IBM ICA
      prompt: "Prompt de jugement par défaut, surchargeable ici."
  embedding:
    url: https://something_somewhere  # si omis => inférence locale (transformer local)
    model: text-embedding-3-small     # nom du modèle ; si url locale => transformer HF local
    api_key: $ENV:EMBEDDING_API_KEY   # optionnel selon endpoint
    threshold: 0.80                   # OBLIGATOIRE : seuil d'acceptation cosine (pas de défaut codé)

# Profils SLO pour le goodput (surchargeables)
slo_profiles:
  interactive: { ttft_ms: 500, tpot_ms: 50, e2e_ms: 5000 }
  relaxed:     { ttft_ms: 2000, tpot_ms: 200, e2e_ms: 30000 }
```

Règles de résolution :
- `$ENV:VAR` est remplacé par la valeur de la variable d'environnement au chargement ; erreur explicite si une variable référencée est absente.
- `evaluation.embedding.url` **omis** => inférence locale (transformer HF chargé en mémoire, zéro appel API).
- `evaluation.embedding.url` **présent** => appel à un endpoint compatible OpenAI.
- `evaluation.embedding.threshold` est **obligatoire** dès que l'évaluation par embedding est utilisée : aucun seuil par défaut n'est codé en dur (non portable entre modèles d'embedding).
- Le juge utilise par défaut `ibm/claude-haiku-4-5` via `IBM_ICA_BASE_URL` / `IBM_ICA_API_KEY`. Rester sur une famille de modèle différente du SUT (self-preference bias).

### 5.2 Paramètres de run (`config.yaml` ou CLI)

```yaml
run:
  mode: closed                  # closed | open
  duration: 180s                # durée par niveau de concurrence
  warmup: 30s                   # phase non mesurée, données taggées warmup
  cooldown: 10s                 # drain en fin de niveau
  min_samples: 30               # avertissement si un niveau collecte moins
  concurrency_levels: [1, 2, 5, 10]   # mode closed
  request_rates: [5, 10, 20]          # mode open (req/s)
  burstiness: 1.0               # 1.0 = Poisson pur
  max_outstanding: 500          # garde-fou open-loop
  max_tokens: 512
  ignore_eos: true              # force OSL déterministe
  temperature: 0.0              # sampling fixe et reporté
  cache_busting: true           # préfixe UUID unique par requête
  retries: 0                    # runs de perf : pas de retry silencieux
  timeout: 120s
  seed: 42
  slo_profile: interactive
```

CLI surcharge toute clé de config. Flag `--eval-method {embedding|judge|none}`, défaut `embedding` si une sortie attendue est présente dans les prompts.

---

## 6. Prompts

### 6.1 Format YAML

Un prompt par entrée, sélection aléatoire (seedée) pour éviter le biais de cache KV.

```yaml
# prompts/coding.yaml
- id: code-001
  category: coding
  messages:
    - role: user
      content: "Implémente une LRU cache thread-safe en Go avec tests."
  expected_output: null          # optionnel ; active l'éval si présent
  max_tokens_hint: 800           # override possible du max_tokens global
  isl_bucket: medium

- id: code-002
  category: coding
  messages:
    - role: user
      content: "Refactore cette fonction récursive en itératif : ..."
  expected_output: "..."         # déclenche l'évaluation qualité
```

### 6.2 Anti-cache

- pool de prompts + tirage aléatoire seedé par requête (pas de réutilisation séquentielle)
- préfixe unique (UUID/tokens aléatoires) injecté en tête quand `cache_busting: true`, placé au-delà de la fenêtre de hash (~256 tokens) pour forcer le cache miss
- préférer du texte réel (prompts variés, style ShareGPT/sonnet) ; éviter les séquences de tokens purement aléatoires qui faussent la latence (cassent le speculative decoding)
- vérification `cached_tokens == 0` en mode à froid, sinon warning

### 6.3 Catégories de prompts à fournir (jeu varié)

1. **coding** : génération, refactoring, debug, explication de code (sorties longues, tool-use occasionnel).
2. **synthèse de document avec tooling** : le prompt déclenche un appel à un outil mock (voir 6.4) de recherche web qui renvoie des résultats fixes, et le modèle doit synthétiser. Isole la capacité de synthèse et de tool-use.
3. **génération de document / PPTX avec tooling** : outil mock de génération de slides ; mesure le tool-calling structuré.
4. **vision / image** : prompt multimodal (image encodée base64 ou URL) pour les modèles `supports_vision`. Mesure TTFT/latence sur entrée image.
5. **prompts généraux** de longueurs variées pour couvrir les buckets ISL/OSL.

### 6.4 Outils mock déterministes

MCP/tools factices renvoyant toujours les mêmes résultats, pour un bench reproductible sans variance réseau ni rate-limit :

- `mock_web_search(query) -> résultats fixes` : simule une recherche internet ; le modèle synthétise.
- `mock_generate_pptx(outline) -> ack` : simule la génération de slides ; mesure le tool-calling.
- (extensible : `mock_fetch_url`, `mock_code_exec`, etc.)

Exposés soit comme outils inline dans la requête (`tools` du protocole OpenAI), soit via un petit serveur MCP local embarqué. On reste sur des définitions d'outils OpenAI passées dans la requête pour le MVP, le serveur MCP local étant une extension.

---

## 7. Évaluation de la qualité (asynchrone, découplée)

### 7.1 Principe

Quand un prompt porte une `expected_output`, on évalue la sortie réelle contre l'attendue. L'évaluation tourne **en parallèle du load test, jamais sur son chemin critique**, via une queue.

### 7.2 Architecture async

1. Les workers de load test mesurent uniquement l'appel au SUT (latence, tokens, statut), enregistrent les métriques perf immédiatement, puis **enfilent** un enregistrement léger `{request_id, prompt, expected, actual, concurrency_level, ts}` de façon non bloquante.
2. Un **pool de workers juges séparé** (concurrence propre, rate-limiter token-bucket dimensionné sur le *provider du juge*, pas du SUT) consomme la queue et écrit `{request_id, sim_score, judge_verdict, judge_reason}`.
3. Jointure ultérieure sur `request_id`. Le bench peut publier ses chiffres de perf avant la fin du jugement ; les scores qualité arrivent ensuite (backfill).
4. Backpressure : si la queue sature, drop-with-counter ou spill disque, jamais bloquer le générateur de charge. Drain de la queue au shutdown.

### 7.3 Méthodes

- **Défaut : embedding cosine** (similarité sémantique). 100 % des requêtes, batchable, déterministe. On reporte la **distribution brute** de similarité ET on applique le `threshold` **obligatoire** défini dans `evaluation.embedding` (aucun seuil par défaut codé, car dépendant du modèle d'embedding). Embedding **local** (transformer HF, `url` omise) ou **via endpoint compatible OpenAI** (`url` présente). Fast-path : normalized exact-match avant embedding.
- **Optionnel `--eval-method judge` : LLM-as-judge** reference-based, configuré dans `evaluation.judge` (défaut `ibm/claude-haiku-4-5` via `IBM_ICA_*`). Prompt de jugement par défaut surchargeable dans la config. Rubrique **binaire ou 3 niveaux** (pas de 1-10, non fiable), Chain-of-Thought + sortie JSON, température basse, **modèle d'une famille différente du SUT**. Échantillonné (N % ou seulement la bande ambiguë de cosine) pour borner coût et latence. Gardes anti-biais : ignorer la verbosité, swap d'ordre si pairwise.

### 7.4 Limites à documenter

- Le cosine mesure le *sujet*, pas la *correction* (rate les négations, nombres, inversions factuelles). D'où le juge en surcouche pour la correction stricte.
- Seuils à calibrer sur un petit jeu labellisé humain avant de leur faire confiance.

---

## 8. Méthodologie d'exécution

### 8.1 Sweep de concurrence à durée fixe (mode closed par défaut)

Niveaux de concurrence joués **séquentiellement** : chaque niveau maintient N users steady pendant `duration`, puis teardown, puis niveau suivant. Pour chaque niveau : ramp-up court → warmup (non mesuré) → steady (mesuré) → cooldown/drain.

Avantage : wall-clock déterministe par niveau, fenêtre d'observation constante pour les calculs de débit, poids statistique égal par niveau. Garde-fou : si un niveau collecte moins de `min_samples`, warning (un modèle lent à concurrence 1 sur 3 min peut ne donner que ~20 requêtes, rendant le p99 peu fiable).

### 8.2 Mode open-loop (Poisson)

Arrivées indépendantes des complétions, à taux fixe (`request_rates`), inter-arrivées Gamma (`burstiness: 1.0` = Poisson pur). Évite la coordinated omission, donne des queues p99 honnêtes sous saturation, modélise le trafic réel. Garde-fou `max_outstanding` contre la croissance non bornée de la file.

### 8.3 Phases taggées

Chaque requête porte `phase ∈ {warmup, steady, cooldown}` selon son horodatage de début vs les bornes de la fenêtre du niveau. Métriques reportées sur `steady` uniquement, mais tout est conservé dans le log brut pour re-fenêtrage offline.

### 8.4 Implémentation async

- `asyncio` + `httpx.AsyncClient` streaming, un client partagé (réutilisation des connexions), `httpx.Limits` dimensionné au pic de concurrence.
- timestamps via `time.monotonic()`, jamais `time.time()`.
- TTFT = 1er chunk *avec contenu* (on jette le chunk rôle vide).
- `stream_options.include_usage=true` pour récupérer `usage` au dernier chunk.

### 8.5 Reproductibilité

Seed unique propageant l'échantillonnage des prompts, le choix de bucket, et les inter-arrivées Poisson (un `random.Random` seedé par worker dérivé du master seed). Snapshot config résolue + environnement (version outil, modèle, horodatage) écrit à côté des résultats.

---

## 9. Persistance et restitution

### 9.1 Données brutes

- **JSONL par requête pendant le run** (crash-safe, append). Champs minimaux :
  `{run_id, model, mode, level_or_rate, phase, seed, prompt_id, category, isl_bucket, prompt_tokens, output_tokens, cached_tokens, reasoning_tokens, t_start, ttft, tt2t, e2e, tpot, itl_summary{mean,p50,p95,p99,max}, itl_list[] (si --raw-itl), outcome, status_code, retry_count, error, sim_score?, judge_verdict?}`
- **Roll-up Parquet** en fin de run (archival, colonne, compressé).
- **Résumé JSON** par run (config + agrégats percentiles) pour diff rapide.
- Analyse offline via **DuckDB** (lit JSONL/Parquet sans ETL). Toutes les agrégations recalculables a posteriori.

### 9.2 Sortie terminal

`rich` : table des percentiles par niveau, taux d'erreur/429 en évidence, éventuellement `plotext` pour une courbe latence-vs-concurrence inline. Feedback immédiat dans le shell.

### 9.3 Rapport HTML statique

Fichier unique autonome (Plotly `include_plotlyjs=True` offline, ou `cdn` pour alléger) généré par template Jinja2. Graphes standards :

- débit système vs concurrence (plateau = max soutenable)
- latence (p50/p99 TTFT) vs concurrence (le coude = saturation)
- tok/s par user vs concurrence (monotone descendante)
- **courbe de tradeoff signature** : TTFT (x) vs débit système (y), chaque point annoté du niveau de concurrence (révèle le coude)
- CDF / histogramme de TTFT et de tok/s
- barres de percentiles (p50/p90/p95/p99) pour TTFT, E2E, TPOT, ITL
- scatter latence vs ISL
- si éval : distributions de score qualité, qualité vs concurrence, qualité vs latence
- bandeau fiabilité : taux succès/erreur/429/timeout par niveau

### 9.4 Extensions différées

- commande `dashboard` Streamlit optionnelle au-dessus du Parquet (ré-exploration interactive).
- Grafana écarté (overkill pour de l'ad-hoc, pertinent seulement en monitoring continu).

---

## 10. CLI (esquisse, Typer)

```
llm-bench run --config config.yaml --model gpt-oss-120b-local \
  --prompts prompts/ --duration 180s --concurrency 1,2,5,10 \
  --max-tokens 512 --mode closed --eval-method embedding \
  --slo-profile interactive --out runs/2026-06-24-xyz/

llm-bench run ... --mode open --request-rate 5,10,20 --burstiness 1.0

llm-bench report runs/2026-06-24-xyz/          # (re)génère le HTML depuis le brut
llm-bench analyze runs/.../raw.jsonl           # requêtes DuckDB ad-hoc
llm-bench dashboard runs/...                   # Streamlit (extension)
```

---

## 11. Découpage en phases (toutes implémentées en autonomie, séquentiellement)

Chaque phase est livrable et testée avant de passer à la suivante.

- **Phase 1 — Cœur perf (closed-loop)**
  Registre de modèles + config, runner closed-loop sweep à durée fixe, streaming async avec timing monotonic, métriques TTFT/TT2T/TPOT/ITL/E2E + débits + fiabilité, comptabilité tokens via `usage`, warmup/steady/cooldown, JSONL brut, résumé terminal rich. Prompts coding + généraux. Anti-cache (préfixe unique). Tests unitaires sur le calcul des métriques (données synthétiques).

- **Phase 2 — Open-loop + goodput + reporting HTML**
  Mode open-loop Poisson avec garde-fou `max_outstanding`. Goodput avec profils SLO. Roll-up Parquet, analyse DuckDB, rapport HTML Plotly complet (tous les graphes). Dimension coût.

- **Phase 3 — Tooling + multimodal**
  Outils mock déterministes (web search, pptx), prompts tool-use et synthèse, prompts vision/image, dimension reasoning tokens.

- **Phase 4 — Évaluation qualité async**
  Queue + pool de workers juges découplé, embedding cosine (défaut), LLM-judge optionnel échantillonné, jointure sur `request_id`, métriques et graphes qualité (qualité vs concurrence/latence), couverture d'éval.

- **Phase 5 — Finitions**
  Commande `report` / `analyze` séparées, snapshot repro complet, `dashboard` Streamlit optionnel, doc (README + CLAUDE.md + `.agent_docs/`).

---

## 12. Pièges à encoder (checklist de correction)

1. ITL ≠ TPOT : convention figée et documentée (pondération + exclusion TTFT).
2. Débit depuis `completion_tokens` réels, jamais `max_tokens`.
3. Tokenizer : `usage` serveur en source de vérité, sinon tokenizer exact du modèle (pas tiktoken).
4. Closed-loop : caveat documenté sur les p99 sous saturation ; open-loop pour des queues honnêtes.
5. Réutilisation de prompt = biais cache : préfixe unique, `cached_tokens == 0` vérifié à froid.
6. Tout débit/latence tagué ISL + OSL + concurrence + état cache + tokenizer + sampling.
7. Échecs exclus des distributions de latence, reportés en taux séparé.
8. Éval async strictement hors chemin critique du load test ; backpressure = drop, jamais bloquer.
9. Juge d'une famille de modèle différente du SUT (self-preference bias).
10. Seuils cosine et prompt de jugement calibrés, pas codés en dur.
11. OSL verrouillé via `ignore_eos` + `max_tokens` ; sinon latences ininterprétables.
12. `min_samples` : warning si une fenêtre à durée fixe ne collecte pas assez pour un p99 fiable.

---

## 13. Stack technique

- **Python 3.12+**, gestion via `uv`, qualité via Ruff + mypy, CLI via Typer.
- `httpx` (async streaming), `httpx-sse` éventuel.
- `pydantic` (validation config), `pyyaml`.
- `numpy` (percentiles, RNG seedé), `duckdb`, `pyarrow` (Parquet).
- `plotly` + `jinja2` (rapport HTML), `rich` (+ `plotext` optionnel) terminal.
- `tokenizers` / `transformers` (tokenizer exact si besoin).
- embeddings et juge via endpoints compatibles OpenAI (mêmes clients httpx).

---

## 14. Décisions sur les points mineurs (tranchées)

- **Outils mock** : définitions d'outils **inline** du protocole OpenAI au MVP (handler Python local déterministe) ; serveur MCP local en extension.
- **Seuil cosine** : **pas de défaut codé**. `threshold` est une option **obligatoire** dans `evaluation.embedding`, liée au modèle d'embedding choisi. On reporte toujours la distribution brute à côté.
- **Stockage ITL** : **résumé par requête** par défaut (mean/p50/p95/p99/max des écarts inter-tokens) pour garder le JSONL compact ; **flag `--raw-itl`** pour conserver la liste complète des écarts quand on veut analyser finement les stalls.
- **Embedding / juge** : via **API compatible OpenAI**. Juge par défaut `ibm/claude-haiku-4-5` sur la gateway IBM ICA (`IBM_ICA_BASE_URL` / `IBM_ICA_API_KEY`). Embedding **local** si `url` omise, sinon endpoint. Config supporte l'interpolation `$ENV:VAR`.

---

## 15. Références principales

- vLLM Benchmark CLI : https://docs.vllm.ai/en/latest/benchmarking/cli/
- NVIDIA LLM Benchmarking Fundamentals : https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/
- LLMPerf : https://github.com/ray-project/llmperf
- Anyscale, métriques reproductibles : https://www.anyscale.com/blog/reproducible-performance-metrics-for-llm-inference
- DistServe (goodput) : https://hao-ai-lab.github.io/blogs/distserve
- Coordinated omission (ScyllaDB) : https://www.scylladb.com/2021/04/22/on-coordinated-omission/
- k6 open vs closed : https://grafana.com/docs/k6/latest/using-k6/scenarios/concepts/open-vs-closed/
- MT-Bench / LLM-as-judge (Zheng et al.) : https://arxiv.org/abs/2306.05685
- Evidently, guide LLM-as-judge : https://www.evidentlyai.com/llm-guide/llm-as-a-judge
- Ragas Answer Correctness : https://docs.ragas.io/en/v0.1.21/concepts/metrics/answer_correctness.html
- Databricks, inference performance : https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices
