---
title: RL Bot Detector
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
---

# RL Bot Detector API (Hugging Face)

This folder contains the FastAPI backend deployed on Hugging Face Spaces.
It is designed to be called by the Vercel proxy endpoint (`/api/analyze`) and
no longer serves static frontend files.

## Required Secrets and Variables (HF Space)

Configure these in Space Settings > Variables and secrets.

- `HF_API_KEY` (Secret): shared secret used by the Vercel proxy via `X-API-Key`
- `ALLOWED_ORIGIN` (Variable): your Vercel domain (for example `https://your-app.vercel.app`)
- `MAX_CONCURRENT_JOBS` (Variable, optional): max analyses in parallel (recommended `2`)
- `QUEUE_WAIT_SECONDS` (Variable, optional): wait time before returning busy error (recommended `8`)
- `PROCESSING_TIMEOUT_SECONDS` (Variable, optional): max analysis duration (recommended `180`)

## Endpoint

- `POST /analyze`
- multipart form-data with one field named `file`
- accepted extension: `.replay`
- backend max size: 50 MB

## Security model

- API key check on every request (`X-API-Key`)
- strict CORS origin using `ALLOWED_ORIGIN`
- bounded concurrency to protect CPU/RAM
- per-request processing timeout
- upload streamed to disk with enforced max size
- intended production caller: Vercel serverless proxy only

## Deploy to Hugging Face Spaces

1. Create a new Space (SDK: Docker).
2. Add `HF_API_KEY` and `ALLOWED_ORIGIN` in Space settings.
3. Push all files from this `website/` directory to your Space repository.
4. Confirm API health:
	- `GET /docs` should load
	- `POST /analyze` should require `X-API-Key`
