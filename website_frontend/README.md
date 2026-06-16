# RL Bot Detector Frontend (Vercel)

This folder contains the public website and a Vercel serverless proxy.

## Architecture

- Browser uploads `.replay` to `POST /api/analyze` (same Vercel origin)
- Vercel proxy forwards request to Hugging Face backend `/analyze`
- Vercel injects secret `HF_API_KEY` server-side
- API key never appears in browser JavaScript
- proxy enforces a 4.5 MB request limit
- proxy applies a lightweight per-IP rate limit

## Files

- `public/`: static frontend (HTML/CSS/JS)
- `api/analyze.js`: secure proxy route
- `vercel.json`: routing configuration

## Vercel Environment Variables

Set these in Project Settings > Environment Variables:

- `HF_API_URL`: complete backend URL, for example `https://your-space.hf.space/analyze`
- `HF_API_KEY`: same secret configured as `HF_API_KEY` in Hugging Face Space

## Deploy to Vercel

1. Push this `website_frontend/` folder to a GitHub repository.
2. Import that repository into Vercel.
3. Add `HF_API_URL` and `HF_API_KEY` variables.
4. Deploy.

## Upload size note

This setup uses Vercel serverless proxy on Hobby plan.
Frontend is limited to about 4.5 MB per upload to avoid request rejection.

## Abuse protection note

Current proxy rate-limit is in-memory (per runtime instance).
For stronger global rate limiting, place Cloudflare in front of Vercel or use a shared store rate limiter.
