# Argus Vision Project Context

This file serves as the definitive reference for the **Argus Vision** project, providing a comprehensive, deep-dive context of its architecture, modules, machine learning pipeline, data contracts, and folder structure. It is designed to give any AI agent or developer the complete context required to understand, develop, debug, and maintain the repository.

---

## 1. Project Overview & Clinical Goal

**Argus Vision** is a medical image classification application designed to diagnose dermoscopic skin-lesion images into one of the 8 canonical **ISIC (International Skin Imaging Collaboration)** categories:

| Code | Diagnosis Name | Clinical Summary / Dermoscopic Criteria |
| :--- | :--- | :--- |
| **MEL** | Melanoma | Atypical, broadened pigment network; irregular streaks; structural/colour asymmetry; regression areas; blue-white veil; irregular dots/globules; chaotic vessels. |
| **NV** | Melanocytic Nevus | Symmetric, regular reticular or globular pattern; uniform colouration; smooth transition to surrounding skin; center-symmetric structure; no chaotic borders. |
| **BCC** | Basal Cell Carcinoma | Arborising (tree-like branching) vessels; blue-grey ovoid nests; pigment-network-free background; leaf-like areas; spoke-wheel structures; shiny white-red structureless zones. |
| **AK** | Actinic Keratosis | 'Strawberry' pattern on facial skin; red pseudo-network of dilated vessels surrounding keratin-plugged follicular openings; white rosettes under polarised light; scaly background. |
| **BKL** | Benign Keratosis | Seborrhoeic/lichenoid keratosis; cerebriform 'brain-like' surface; milia-like cysts; comedo-like openings; sharply demarcated borders; stuck-on appearance; fat-finger structures. |
| **DF** | Dermatofibroma | Central white scar-like patch; peripheral delicate pigment network; firm tan-brown ring; central dimpling on lateral compression. |
| **VASC** | Vascular Lesion | Haemangiomas/angiokeratomas; sharply demarcated red, purple, or maroon lacunae separated by pale septa; absence of melanocytic pigment network. |
| **SCC** | Squamous Cell Carcinoma | Central keratin masses; white circles around follicular openings; surface scale/ulceration; peripheral hairpin/looped and glomerular/coiled vessels. |

Rather than relying on a single vision classifier (which might be overconfident or poorly calibrated), Argus Vision uses an **adversarial multi-agent visual debate**:
1. **Two distinct model backbones** (a CNN and a Vision Transformer) analyze the image independently.
2. If they disagree or are uncertain, they enter a structured **2-round natural language debate** powered by an LLM (via Groq).
3. The LLM arguments are grounded in **spatial attention saliency maps** (Grad-CAM++ and attention rollout) over the contested region.
4. A **calibrated MLP consensus head** fuses the agents' probability vectors, spatial region statistics, and the sentence embeddings of their arguments to yield the final, temperature-scaled prediction with an Expected Calibration Error (ECE) estimate.

---

## 2. Architecture & System Topography

### Network Topology
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
```

*   **Nginx (Port 80):** Operates as the entry reverse proxy. It routes `/api/*` to the FastAPI backend, `/ws/*` to the FastAPI WebSocket endpoints, and all other traffic `/` to the Next.js frontend.
*   **Frontend (Port 3000):** A Next.js application that provides the image upload dropzone interface and renders the real-time websocket debate.
*   **Backend (Port 8000):** A FastAPI app that exposes endpoints for image upload, polling, and the WebSocket connection.
*   **Redis (Port 6379):** Acts as the database and real-time message broker:
    *   Saves job metadata (`JobResult` JSON string) under the key `argus:job:{job_id}`.
    *   Saves the raw uploaded image path under `argus:img:{job_id}`.
    *   Standard job TTL (Time-To-Live) is **3600 seconds** (1 hour).
    *   Streams real-time debate steps over pub/sub channels named `argus:debate:{job_id}`.

---

## 3. The End-to-End Machine Learning Pipeline

The backend orchestrates the ML pipeline in a single flow defined in [pipeline.py](file:///c:/Users/ahmad/Desktop/argus-vision/backend/ml/pipeline.py). Blocking CPU/GPU operations (Inference, Groq APIs, Sentence-Transformers) are run in background threads using `asyncio.to_thread` to ensure the FastAPI event loop remains responsive.

```
┌────────────────────────────────────────────────────────┐
│ Preprocess uploaded image -> PyTorch Tensor [1,3,224,224]│
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Run Classifier Agents                         │
│ - Agent A (EfficientNet-B4) -> Probability Dist pA    │
│ - Agent B (ViT-B/16)        -> Probability Dist pB    │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Evaluate Trigger                              │
│ - JS Divergence = jensenshannon(pA, pB)^2              │
│ - Entropy A = ShannonEntropy(pA), Entropy B = ...      │
│ - Trigger Fired = (JS > Threshold) OR (max_ent > Thresh)│
└───────────────────────────┬────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │ Trigger Fired?            │
              ├─────────────┬─────────────┤
              │ YES         │ NO          │
              ▼             ▼
┌─────────────────────────┐ ┌────────────────────────────┐
│ Stage 3: Attention      │ │ Fast Path (No Debate)      │
│ - Grad-CAM++ (Agent A)  │ │ - Spatial stats = [0,0,0,0]│
│ - Rollout (Agent B)     │ │ - Embedding A = [0]*384    │
│ - Disagreement Map      │ │ - Embedding B = [0]*384    │
│ - Bounding Box (top 20%)│ └─────────────┬──────────────┘
│ - Spatial stats         │               │
└─────────────┬───────────┘               │
              ▼                           │
┌─────────────────────────┐               │
│ Stage 4: LLM Debate     │               │
│ - Round 1: Defend       │               │
│ - Round 2: Rebut + Delta│               │
│ - Encode R2 args to     │               │
│   384-d sentence embeds │               │
│ - Nudge & renorm probs  │               │
└─────────────┬───────────┘               │
              ▼                           │
              └─────────────┬─────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage 5: Consensus Fusion MLP                          │
│ Input: Concatenated vector (788 dimensions)            │
│ Vector: [pA (8) + pB (8) + Spatial (4) + eA(384) + eB(384)]│
│ Architecture: 788 -> 512 -> 256 -> 8                   │
│ Scaling: logits / temperature                          │
│ Output: Calibrated Consensus Prediction                │
└────────────────────────────────────────────────────────┘
```

### 3.1. Classifier Agents
*   **Agent A (`backend/ml/agents/agent_a.py`):** Wraps `timm` model `efficientnet_b4`. It processes a pre-processed `(1, 3, 224, 224)` image tensor.
*   **Agent B (`backend/ml/agents/agent_b.py`):** Wraps `timm` model `vit_base_patch16_224.augreg_in21k_ft_in1k`.
*   **Loading Behavior:** The models load checkpoints (`agent_a_best.pth` and `agent_b_best.pth`) from `MODEL_CHECKPOINT_DIR`. If checkpoints are missing and `PRETRAINED_FALLBACK` is `True`, they fall back to ImageNet-pretrained weights (which yields random, clinically non-meaningful predictions, allowing testing without weights).

### 3.2. Debate Trigger
*   **Jensen-Shannon Divergence ($D_{JS}$):** The square of the JS distance (obtained using `scipy.spatial.distance.jensenshannon` with base-2 logarithm).
*   **Shannon Entropy ($H$):** Measures uncertainty of a probability distribution $p$ in bits:
    $$H(p) = - \sum_{i} p_i \log_2 (p_i + \epsilon)$$
*   **Thresholds:** Trigger fires if:
    *   $D_{JS}(p_A, p_B) > \text{DEBATE\_JS\_THRESHOLD}$ (default: `0.25`)
    *   $\max(H(p_A), H(p_B)) > \text{DEBATE\_ENTROPY\_THRESHOLD}$ (default: `0.8` bits)

### 3.3. Spatial Attention & Disagreement Map
When the debate trigger fires, the pipeline analyzes where the agents are focusing their visual attention:
*   **Agent A Saliency Map:** Grad-CAM++ is computed against its predicted class.
*   **Agent B Saliency Map:** Attention Rollout is computed by tracing self-attention weights across transformer layers.
*   **Disagreement Map ($M_{\delta}$):** Absolute difference of independently min-max normalized maps:
    $$M_{\delta} = | \bar{H}_A - \bar{H}_B |$$
*   **Contested Region Mask:** Formed by the top 20% highest pixels of the combined normalized activation ($\bar{H}_A + \bar{H}_B$).
*   **Spatial Stats:** Extracts the mean and standard deviation of each agent's attention map inside the contested region mask, returning a 4-element list:
    $$\text{spatial\_stats} = [\text{mean}_A, \text{mean}_B, \text{std}_A, \text{std}_B]$$
*   **Bounding Box:** Axis-aligned bounding box enclosing the top 20% pixels of the disagreement map $M_{\delta}$.

### 3.4. LLM Debate
*   **LLM Choice:** Groq API running `llama-3.3-70b-versatile` (or specified via `GROQ_MODEL`).
*   **Prompt Construction:** The prompts inject class profiles, predicted class confidence, the bounding box coordinates, and the statistical attention indicators for the agent.
*   **Debate Rounds:**
    *   **Round 1:** The agent constructs a concise single paragraph defending its diagnosis, citing features inside the bounding box.
    *   **Round 2:** The agent receives its round-1 argument and the opponent's round-1 argument. It drafts a counter-argument and must output a confidence adjustment:
        `CONFIDENCE_DELTA: <float>` where the float is bounded between `[-0.3, 0.3]`.
*   **Websocket streaming:** As the argument is generated, it is split into tokens and pushed over the websocket with a delay of `0.05`s to produce a typewriter effect.
*   **Offline Fallback:** If the Groq key is absent or a call fails, a deterministic fallback argument is generated using the static class descriptions and the region statistics.
*   **Sentence Encoding:** The final Round-2 rebuttals are encoded into 384-dimensional sentence embeddings using Hugging Face's `sentence-transformers/all-MiniLM-L6-v2`. Any encoding failure defaults to a 384-dimensional zero vector.
*   **Probability Nudging:** The original predicted class probability is shifted by the parsed `CONFIDENCE_DELTA` (clamped to `[0, 1]` constraints) and the probability vector is renormalized to sum to 1.

### 3.5. Consensus MLP & Temperature Calibration
The final decision is made by the calibrated consensus head:
*   **Inputs:** A concatenated 788-dimensional vector:
    $$\text{Feature Vector} = [p_A \;(8) \parallel p_B \;(8) \parallel \text{spatial\_stats} \;(4) \parallel e_A \;(384) \parallel e_B \;(384)]$$
*   **MLP Architecture:**
    `Linear(788, 512) -> BatchNorm1d(512) -> ReLU -> Dropout(0.3) -> Linear(512, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.3) -> Linear(256, 8)`
*   **Logits Scaling:** Divided by a learnable temperature parameter $\sigma$ (initialized to `1.0`, clamped with a floor of `1e-2` to prevent division-by-zero):
    $$p_i = \text{Softmax}\left(\frac{\text{logits}_i}{\max(\sigma, 10^{-2})}\right)$$
*   **Expected Calibration Error (ECE):** A train-time calibration error that flows into the final response to indicate reliability.

---

## 4. File Structure & Module Directory Map

```
argus-vision/
├── backend/                  # FastAPI Application Core
│   ├── api/
│   │   ├── routes/
│   │   │   ├── classify.py   # POST /classify (image upload handler)
│   │   │   ├── health.py     # GET /health (healthcheck endpoint)
│   │   │   └── jobs.py       # GET /jobs/{jobId} (results retrieval)
│   │   └── websocket/
│   │       └── debate_stream.py # WS /ws/debate/{jobId} (WS event loop handler)
│   ├── checkpoints/          # Model checkpoint directory (weights)
│   ├── core/
│   │   ├── config.py         # BaseSettings configuration with pydantic-settings
│   │   ├── exceptions.py     # Custom exceptions (ArgusError, JobNotFoundError, etc.)
│   │   └── models.py         # Unified Pydantic models & WS Event schemas
│   ├── ml/
│   │   ├── agents/
│   │   │   ├── agent_a.py    # Agent A (EfficientNet-B4 timm loader)
│   │   │   └── agent_b.py    # Agent B (ViT-B/16 timm loader)
│   │   ├── attention/
│   │   │   ├── disagreement.py# Combined heatmap delta & contested region bbox
│   │   │   ├── gradcam.py    # CNN Grad-CAM++ computation
│   │   │   └── rollout.py    # ViT attention rollout computation
│   │   ├── consensus/
│   │   │   └── classifier.py # 788-d input consensus MLP & Temp calibration
│   │   ├── debate/
│   │   │   ├── argument_gen.py# Prompt composition & Groq LLM debate loop
│   │   │   ├── encoder.py    # Sentence Transformer 384-d vector embeddings
│   │   │   └── trigger.py    # JS divergence & Shannon entropy calculation
│   │   └── pipeline.py       # End-to-end background job pipeline orchestrator
│   ├── services/
│   │   ├── image_service.py  # Image resize, normalisation, and base64 PNG renderers
│   │   └── job_service.py    # Redis job store persistence and WS event publisher
│   ├── main.py               # FastAPI entrypoint, lifespan and global middlewares
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/                 # Next.js Application Core
│   ├── src/
│   │   ├── app/
│   │   │   ├── debate/
│   │   │   │   └── [jobId]/
│   │   │   │       └── page.tsx # Subscribes to websocket & renders debate
│   │   │   ├── globals.css   # Main stylesheet
│   │   │   ├── layout.tsx
│   │   │   └── page.tsx      # Landing page with DropZone component
│   │   ├── components/
│   │   │   ├── debate/
│   │   │   │   ├── AgentCard.tsx       # Renders agent confidence and heatmaps
│   │   │   │   ├── ArgumentStream.tsx  # Handles streamed text with typewriter logic
│   │   │   │   ├── ConsensusVerdict.tsx# Displays final consensus output
│   │   │   │   ├── DisagreementMap.tsx # Displays visual disagreement map overlay
│   │   │   │   ├── HeatmapCanvas.tsx   # Base64 heatmap and bbox canvas overlay
│   │   │   │   └── TriggerIndicator.tsx# Shows JS divergence/entropies gauge
│   │   │   ├── ui/
│   │   │   │   ├── ClassBadge.tsx      # Colored badges for ISIC-8 categories
│   │   │   │   ├── ConfidenceBar.tsx   # Progress bar helper
│   │   │   │   └── LoadingOrbit.tsx    # Interactive orbit spinner
│   │   │   └── upload/
│   │   │       ├── DropZone.tsx        # File drag-and-drop handler
│   │   │       └── ImagePreview.tsx
│   │   ├── hooks/
│   │   ├── lib/
│   │   └── types/
│   ├── Dockerfile
│   ├── tailwind.config.ts
│   └── package.json
│
├── ml_training/              # Training Notebooks and Calibration Scripts
│   ├── 01_train_agent_a.ipynb   # EfficientNet-B4 training
│   ├── 02_train_agent_b.ipynb   # ViT-B/16 training
│   ├── 03_build_hard_subset.ipynb# Formulate difficult boundary classification datasets
│   ├── 04_train_consensus.ipynb # Train the consensus classifier MLP and calibrate
│   ├── 05_evaluation.ipynb      # Global evaluations, reliability, and ECE plots
│   ├── config.py
│   ├── dataset.py            # PyTorch datasets for training on ISIC
│   ├── losses.py             # Custom losses
│   └── requirements_training.txt
│
├── nginx/
│   └── nginx.conf            # Proxy routing configuration
│
└── docker-compose.yml        # Orchestrates Redis, Backend, Frontend, and Nginx
```

---

## 5. API Contracts & WebSocket Event Protocols

The data schemas are maintained using Pydantic on the backend ([models.py](file:///c:/Users/ahmad/Desktop/argus-vision/backend/core/models.py)) and mirrored on the TypeScript frontend.

### 5.1. REST Endpoints

#### `POST /api/classify`
*   **Request:** `multipart/form-data` with a `file` field holding the raw JPEG or PNG file.
*   **Response:**
    ```json
    {
      "job_id": "8bc5a0e0-47b2-4d2d-8068-07e15bf9ec2d",
      "status": "queued",
      "estimated_seconds": 10
    }
    ```

#### `GET /api/jobs/{job_id}`
*   **Response:** Full `JobResult` object structure:
    ```json
    {
      "job_id": "8bc5a0e0-47b2-4d2d-8068-07e15bf9ec2d",
      "status": "consensus_done",
      "created_at": "2026-06-18T14:40:00Z",
      "updated_at": "2026-06-18T14:40:10Z",
      "agent_a": {
        "agent_id": "A",
        "result": {
          "pred_class": "MEL",
          "confidence": 0.72,
          "probabilities": { "MEL": 0.72, "NV": 0.10, ... }
        },
        "heatmap_b64": "data:image/png;base64,..."
      },
      "agent_b": {
        "agent_id": "B",
        "result": {
          "pred_class": "BKL",
          "confidence": 0.61,
          "probabilities": { "MEL": 0.05, "NV": 0.12, "BKL": 0.61, ... }
        },
        "heatmap_b64": "data:image/png;base64,..."
      },
      "trigger": {
        "fired": true,
        "js_divergence": 0.38,
        "entropy_a": 0.95,
        "entropy_b": 1.12,
        "threshold_js": 0.25,
        "threshold_entropy": 0.8
      },
      "attention": {
        "heatmap_a_b64": "data:image/png;base64,...",
        "heatmap_b_b64": "data:image/png;base64,...",
        "disagreement_b64": "data:image/png;base64,...",
        "bbox": { "x1": 42, "y1": 30, "x2": 150, "y2": 180 },
        "region_stats_a": { "mean": 0.58, "std": 0.12, "max": 0.82 },
        "region_stats_b": { "mean": 0.41, "std": 0.19, "max": 0.75 }
      },
      "debate": {
        "argument_a": {
          "agent_id": "A",
          "argument": "The lesion displays a broadened pigment network...",
          "embedding": [0.012, -0.054, ...],
          "updated_probs": { "MEL": 0.81, "NV": 0.07, ... }
        },
        "argument_b": {
          "agent_id": "B",
          "argument": "Although there is a network, the borders are sharply demarcated...",
          "embedding": [-0.023, 0.091, ...],
          "updated_probs": { "MEL": 0.04, "NV": 0.10, "BKL": 0.68, ... }
        }
      },
      "consensus": {
        "pred_class": "MEL",
        "confidence": 0.79,
        "probabilities": { "MEL": 0.79, "NV": 0.05, ... },
        "temperature": 1.15,
        "ece": 0.042
      },
      "error": null
    }
    ```

---

### 5.2. WebSocket Event Streams (`WS /ws/debate/{job_id}`)

Clients receive a sequence of JSON messages categorized by `type` (Discriminated Union):

1.  **`{"type": "ping"}`:** Emitted every 30s to prevent proxy timeouts.
2.  **`{"type": "agents_running"}`:** Emitted when Agent A and B start classification.
3.  **`{"type": "agents_done", "agent_a": {...}, "agent_b": {...}}`:** Emitted when both agents complete their initial forward pass (without heatmaps).
4.  **`{"type": "trigger_evaluated", "result": {...}}`:** Emitted after evaluating the debate trigger conditions.
5.  **`{"type": "attention_computed", "result": {...}}`:** Emitted after heatmaps, disagreement map, and bounding boxes are calculated. This also re-publishes `agents_done` with base64 heatmap strings.
6.  **`{"type": "argument_token", "agent": "A"|"B", "token": "...", "round": 1|2}`:** Pushes single words/tokens with whitespace sequentially for typewriter rendering.
7.  **`{"type": "argument_done", "agent": "A"|"B", "argument": "...", "round": 1|2}`:** Sent when an agent completes its full argument.
8.  **`{"type": "consensus_done", "result": {...}}`:** Fired when the final calibrated consensus result is calculated.
9.  **`{"type": "error", "message": "..."}`:** Emitted if any pipeline stage fails, moving the job status to `failed`.

---

## 6. Environment Configurations

### 6.1. Backend Environment Settings (`backend/.env` / Docker Env)

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `GROQ_API_KEY` | string | `""` | Groq API key for LLM arguments. If empty, offline fallbacks are used. |
| `REDIS_URL` | string | `redis://redis:6379` | Redis server address for job queue and pub/sub. |
| `MODEL_CHECKPOINT_DIR`| string | `./checkpoints` | Path to weights checkpoints. |
| `AGENT_A_CHECKPOINT` | string | `agent_a_best.pth` | Agent A weight file. |
| `AGENT_B_CHECKPOINT` | string | `agent_b_best.pth` | Agent B weight file. |
| `CONSENSUS_CHECKPOINT`| string | `consensus_best.pth`| Consensus MLP weight file. |
| `PRETRAINED_FALLBACK` | boolean| `True` | Fall back to ImageNet weights if checkpoints are missing. |
| `DEBATE_JS_THRESHOLD` | float | `0.25` | JS divergence trigger threshold. |
| `DEBATE_ENTROPY_THRESHOLD`| float | `0.8` | Shannon entropy trigger threshold. |
| `GROQ_MODEL` | string | `llama-3.3-70b-versatile`| Groq model for debate arguments. |
| `MAX_IMAGE_SIZE_MB` | integer| `10` | Maximum file size allowed in uploads. |
| `ALLOWED_ORIGINS` | string | `http://localhost:3000,http://localhost` | Permitted CORS origins. |

### 6.2. Frontend Settings

*   `NEXT_PUBLIC_API_URL`: Base URL for REST endpoints (typically `http://localhost/api` when routed through Nginx, or `http://localhost:8000` for direct dev backend).
*   `NEXT_PUBLIC_WS_URL`: Base URL for WS endpoints (typically `ws://localhost/ws` via Nginx, or `ws://localhost:8000/ws` for direct dev backend).

---

## 7. Running & Local Setup

### 7.1. Via Docker Compose (Recommended)
Docker Compose spins up the entire stack (Redis, Backend, Frontend, and Nginx reverse proxy). 

1.  **Start Stack:**
    ```bash
    docker compose up --build
    ```
2.  **Access Page:** Open `http://localhost` in the browser.
3.  **HuggingFace / model weights cache:** Pretrained backbones and Sentence-Transformers weights (~1GB) are downloaded during first run and cached in the docker volume `hf-cache` so subsequent startups are fast.

### 7.2. Running Backend Locally for Development
1.  **Prerequisites:** Python 3.10+, Redis running locally on `localhost:6379`.
2.  **Install requirements:**
    ```bash
    cd backend
    pip install -r requirements.txt
    ```
3.  **Setup Environment:**
    ```bash
    cp .env.example .env
    # Edit .env and configure GROQ_API_KEY if desired
    ```
4.  **Run FastAPI:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```

### 7.3. Running Frontend Locally for Development
1.  **Install dependencies:**
    ```bash
    cd frontend
    npm install
    ```
2.  **Start Dev server:**
    ```bash
    npm run dev
    ```
    The frontend is accessible at `http://localhost:3000`, communicating directly with `http://localhost:8000` (or as configured in environment settings).
