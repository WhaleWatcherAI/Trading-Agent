# Qwen TopstepX Bundle

Local-LLM snapshot of the TopstepX trading agent, risk managers, and dashboard using Qwen2.5 via Ollama (no DeepSeek/OpenAI API required). Structure matches the upstream bundle so it can run standalone with the same config files.

## What's here
- Agents: `live-fabio-agent-playbook*.ts` plus `live-topstepx-*.ts` variants and `restart-agents.sh`.
- Core logic: `lib/` (Qwen trading agent + risk managers wired to Ollama, execution manager, playbook, cache) and shared `types/`.
- Dashboard: Next.js app under `app/` with TopstepX pages (`app/topstepx/**`) and assets in `public/`.
- Config: `package*.json`, `tsconfig.json`, `next.config.js`, `tailwind.config.js`, `postcss.config.js`, `vitest.config.ts`, `jest.config.js`, `ecosystem.config.js`, and `projectx-rest.ts`.

## Quick start
```bash
cd qwen-topstepx-bundle
npm install
# ensure the model is available locally
ollama pull qwen2.5:7b
```

Set environment (example):
```bash
export OLLAMA_HOST=http://127.0.0.1:11434
export OLLAMA_MODEL=qwen2.5:7b
export TOPSTEPX_SYMBOL=NQZ5
export TOPSTEPX_ACCOUNT_ID=12345678
export LIVE_TRADING=true
export TOPSTEPX_ENABLE_NATIVE_BRACKETS=true
# optional
export DASHBOARD_PORT=3337
```

Run the main agent:
```bash
npx tsx live-fabio-agent-playbook.ts
```
Other variants (gold, micro, SMA, etc.) are in the same folder (e.g., `live-topstepx-mgc-winner.ts`).

Run the dashboard:
```bash
npm run dev
# open http://localhost:3000/topstepx
```

## Notes
- No DeepSeek/OpenAI keys are needed for the Qwen/Ollama path; keep any legacy env vars only if you still call the older OpenAI/deepseek agents.
- This is a copy; keep it in sync with root if upstream files change.
- Logs were intentionally excluded to keep the bundle lean.
