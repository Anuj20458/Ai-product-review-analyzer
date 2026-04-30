# AI Product Review Analyzer — Vercel Deployment Guide

## Project Structure

```
/
├── api/
│   └── index.py          ← Flask app (Vercel serverless entry point)
├── templates/
│   └── index.html        ← Jinja2 frontend template
├── requirements.txt      ← Python dependencies
└── vercel.json           ← Vercel routing + build config
```

---

## Deploy to Vercel (3 steps)

### Option A — Vercel CLI (recommended)

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Log in**
   ```bash
   vercel login
   ```

3. **Deploy from project root**
   ```bash
   cd /path/to/this/folder
   vercel --prod
   ```
   Follow the prompts — accept all defaults. Your live URL is printed at the end.

---

### Option B — GitHub + Vercel Dashboard (no CLI needed)

1. Push this folder to a **new GitHub repo** (public or private).
2. Go to [vercel.com](https://vercel.com) → **Add New Project** → import your repo.
3. Vercel auto-detects `vercel.json` and `requirements.txt`. Click **Deploy**.
4. Done — your URL is shown on the dashboard.

---

## How it works on Vercel

| File | Role |
|---|---|
| `vercel.json` | Tells Vercel to build `api/index.py` with `@vercel/python` and route all traffic there |
| `api/index.py` | The Flask WSGI app — Vercel wraps it in a serverless function automatically |
| `requirements.txt` | Vercel installs these Python packages in the serverless environment |
| `templates/index.html` | Served by Flask's `render_template()` |

### NLTK data
On first cold start, the app downloads NLTK corpora into `/tmp/nltk_data` (the only writable path in Vercel's serverless runtime). Subsequent warm invocations skip the download. Cold start takes ~5–10 seconds; normal requests are fast.

---

## Local development

```bash
pip install -r requirements.txt
cd api
python index.py
# → open http://localhost:5000
```

---

## Notes
- CSV analysis is capped at **500 rows** to stay within Vercel's 10-second serverless timeout.
- The `reviews.csv` from the original project is **not included** — it's only needed for local data exploration (`project.py`), not for the web app.
- Free Vercel tier (Hobby) supports this app without any paid plan.
