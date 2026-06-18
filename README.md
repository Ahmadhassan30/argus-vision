# Argus Vision

> Adversarial multi-agent visual debate for uncertainty-aware medical image classification.

Argus Vision classifies dermoscopic skin-lesion images into the 8 ISIC categories
(MEL, NV, BCC, AK, BKL, DF, VASC, SCC) using two independent vision agents. When the
agents disagree or are uncertain, they enter a structured, evidence-grounded debate
and a calibrated consensus model produces the final, uncertainty-aware prediction.

## Architecture

```
                        ┌─────────────┐
                        │   Browser   │
                        └──────┬──────┘
                               │  http / ws
                        ┌──────▼──────┐
                        │  nginx :80  │
                        └──┬───────┬──┘
                  /  (ui)  │       │  /api  /ws
                  ┌────────▼─┐   ┌─▼──────────────┐
                  │ frontend │   │   backend       │
                  │ Next.js  │   │   FastAPI       │
                  │  :3000   │   │   :8000         │
                  └──────────┘   └──┬──────────────┘
                                    │
                              ┌─────▼─────┐
                              │  redis    │
                              │  :6379    │
                              │ (jobs +   │
                              │  pub/sub) │
                              └───────────┘

  Backend ML pipeline (per job):

  ┌────────────────────┐   ┌────────────────────┐
  │ Agent A             │   │ Agent B             │
  │ EfficientNet-B4     │   │ ViT-B/16            │
  └─────────┬──────────┘   └──────────┬─────────┘
            └──────────────┬──────────┘
                           ▼
                   ┌───────────────┐
                   │ trigger        │  (JS divergence + entropy)
                   └───────┬───────┘
                           ▼  (if fired)
                   ┌──────────────────────────┐
                   │ attention + disagreement  │  (Grad-CAM / attn maps,
                   └───────────┬──────────────┘   contested bbox + stats)
                               ▼
                   ┌──────────────────────────┐
                   │ Groq debate rounds 1 & 2  │  (llama-3.3-70b-versatile)
                   └───────────┬──────────────┘
                               ▼
                   ┌──────────────────────────┐
                   │ consensus MLP (calibrated)│  (788 -> 512 -> 256 -> 8,
                   └───────────┬──────────────┘   temperature scaling, ECE)
                               ▼
                        final ConsensusResult
```

## Quick start

**Requirements:** Docker Engine with the Compose plugin (`docker compose`,
v2.24+), and an internet connection on the first run (the backend downloads the
pretrained model weights once and caches them in the `hf-cache` volume).

1. Build and start the full stack — this works out of the box with no
   configuration:

   ```bash
   docker compose up --build
   ```

2. Open the app in your browser:

   ```
   http://localhost
   ```

That's it. With no API key the agents run on ImageNet-pretrained weights and the
debate uses deterministic fallback arguments, so the full pipeline (upload →
agents → trigger → attention → debate → consensus) runs end-to-end.

### Enable the live LLM debate (optional)

To get real Groq-generated argument text during the debate, provide a Groq API
key (free key at <https://console.groq.com/keys>) in any one of these ways:

```bash
# Option A — shell environment (picked up by docker compose interpolation)
export GROQ_API_KEY=gsk_...
docker compose up --build

# Option B — a .env file next to docker-compose.yml
cp .env.example .env          # then edit .env and set GROQ_API_KEY=...
docker compose up --build
```

> **Note:** The first `docker compose up` takes a few minutes while the backend
> downloads ~1 GB of pretrained weights. Subsequent runs reuse the `hf-cache`
> volume and start quickly. The backend health check has a 180 s start period to
> accommodate the initial download; the UI at `http://localhost` is reachable
> as soon as the frontend is up.

### Local (non-Docker) development

To run the backend directly with `uvicorn`, copy the backend template instead:

```bash
cp backend/.env.example backend/.env   # edit and set GROQ_API_KEY=...
```

## Environment variables

### Backend (`backend/.env`)

| Variable | Description | Default |
| --- | --- | --- |
| `GROQ_API_KEY` | API key for the Groq LLM used in debate rounds | `""` |
| `REDIS_URL` | Redis connection URL (jobs + pub/sub) | `redis://redis:6379` |
| `MODEL_CHECKPOINT_DIR` | Directory containing model checkpoints | `./checkpoints` |
| `AGENT_A_CHECKPOINT` | Agent A (EfficientNet-B4) checkpoint filename | `agent_a_best.pth` |
| `AGENT_B_CHECKPOINT` | Agent B (ViT-B/16) checkpoint filename | `agent_b_best.pth` |
| `CONSENSUS_CHECKPOINT` | Consensus MLP checkpoint filename | `consensus_best.pth` |
| `PRETRAINED_FALLBACK` | Fall back to ImageNet-pretrained weights if no checkpoint | `True` |
| `DEBATE_JS_THRESHOLD` | Jensen-Shannon divergence threshold to trigger debate | `0.25` |
| `DEBATE_ENTROPY_THRESHOLD` | Predictive entropy threshold to trigger debate | `0.8` |
| `GROQ_MODEL` | Groq model used for debate arguments | `llama-3.3-70b-versatile` |
| `MAX_IMAGE_SIZE_MB` | Maximum accepted upload size in megabytes | `10` |
| `ALLOWED_ORIGINS` | Comma-separated list of allowed CORS origins | `http://localhost:3000,http://localhost` |

### Frontend (build args / runtime env)

| Variable | Description | Default |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | Base URL for the backend HTTP API (via nginx) | `http://localhost/api` |
| `NEXT_PUBLIC_WS_URL` | Base URL for the debate WebSocket (via nginx) | `ws://localhost/ws` |

## How it works

1. **Upload.** A dermoscopic image is uploaded to `POST /classify`. The backend
   stores the image under `/tmp/argus/{job_id}.jpg`, queues a job in Redis, and
   returns a `job_id`. The browser opens `WS /ws/debate/{job_id}` to stream
   progress events.
2. **Independent agents.** Agent A (EfficientNet-B4) and Agent B (ViT-B/16) each
   classify the image into the 8 ISIC classes, producing per-class probabilities.
3. **Trigger.** The pipeline measures inter-agent Jensen-Shannon divergence and the
   predictive entropy of each agent. If either crosses its threshold, the adversarial
   debate is triggered; otherwise the fast path runs the consensus model directly.
4. **Attention.** When debate fires, attention/saliency maps for both agents are
   computed, a contested bounding box is localized, and region statistics describe
   where and how strongly the agents disagree.
5. **Debate.** Two rounds of Groq-powered arguments let each agent justify and then
   revise its prediction in light of the other's evidence; tokens stream live to the
   client over the WebSocket.
6. **Consensus.** A calibrated MLP fuses both probability vectors, the contested
   spatial statistics, and the argument embeddings into a temperature-scaled final
   prediction with an expected-calibration-error estimate.

## License

MIT.
