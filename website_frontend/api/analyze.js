const config = {
  api: {
    bodyParser: false,
  },
};

const MAX_PROXY_UPLOAD_BYTES = Math.floor(4.5 * 1024 * 1024);
const RATE_LIMIT_WINDOW_MS = 60 * 1000;
const RATE_LIMIT_MAX_REQUESTS = 6;
const rateLimitStore = new Map();

function isRateLimited(clientIp) {
  const now = Date.now();
  const key = clientIp || "unknown";
  const timestamps = rateLimitStore.get(key) || [];
  const recent = timestamps.filter((ts) => now - ts < RATE_LIMIT_WINDOW_MS);

  if (recent.length >= RATE_LIMIT_MAX_REQUESTS) {
    rateLimitStore.set(key, recent);
    return true;
  }

  recent.push(now);
  rateLimitStore.set(key, recent);
  return false;
}

async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).send('Method Not Allowed');
  }

  const forwardedFor = req.headers["x-forwarded-for"] || "";
  const clientIp = forwardedFor.split(",")[0].trim();

  if (isRateLimited(clientIp)) {
    return res.status(429).json({ detail: "Too many requests. Please wait one minute and retry." });
  }
  
  try {
    const HF_URL = process.env.HF_API_URL;
    const API_KEY = process.env.HF_API_KEY;
    const HF_SPACE_TOKEN = process.env.HF_SPACE_TOKEN;

    if (!HF_URL || !API_KEY) {
      return res.status(500).json({ detail: "Server misconfiguration. HF_API_URL or HF_API_KEY missing." });
    }

    // Guardrail: avoid common misconfiguration like huggingface.co/spaces/... instead of *.hf.space/analyze
    if (!/^https:\/\/[a-z0-9-]+\.hf\.space\/analyze$/i.test(HF_URL.trim())) {
      return res.status(500).json({
        detail: "Invalid HF_API_URL format.",
        expected: "https://<space-subdomain>.hf.space/analyze",
      });
    }

    const inboundContentType = req.headers["content-type"];
    if (!inboundContentType || !inboundContentType.includes("multipart/form-data")) {
      return res.status(400).json({ detail: "Expected multipart/form-data upload." });
    }

    const rawContentLength = req.headers["content-length"];
    const inboundContentLength = Number(rawContentLength || "0");
    if (rawContentLength && Number.isFinite(inboundContentLength) && inboundContentLength > MAX_PROXY_UPLOAD_BYTES) {
      return res.status(413).json({ detail: "File too large for this deployment. Maximum size is 4.5 MB." });
    }

    const forwardHeaders = {
      "x-api-key": API_KEY,
      "content-type": inboundContentType,
      "x-forwarded-for": clientIp,
    };

    if (HF_SPACE_TOKEN) {
      forwardHeaders["authorization"] = `Bearer ${HF_SPACE_TOKEN}`;
    }

    if (req.headers["content-length"]) {
      forwardHeaders["content-length"] = req.headers["content-length"];
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 55000);

    const response = await fetch(HF_URL, {
      method: "POST",
      headers: forwardHeaders,
      body: req,
      duplex: "half",
      signal: controller.signal,
    });
    clearTimeout(timeout);

    const responseBody = await response.text();
    const responseType = response.headers.get("content-type") || "application/json";
    res.status(response.status).setHeader("content-type", responseType).send(responseBody);

  } catch (error) {
    if (error.name === "AbortError") {
      return res.status(504).json({ detail: "Analysis timeout on proxy. Please retry." });
    }
    res.status(502).json({ detail: "Proxy error while contacting analysis backend.", reason: String(error.message || error) });
  }
}

module.exports = handler;
module.exports.config = config;
