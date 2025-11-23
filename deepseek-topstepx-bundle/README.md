# DeepSeek TopstepX Bundle

Snapshot of the DeepSeek-powered TopstepX trading agent, risk managers, and Next.js dashboard collected into a single directory for handoff. Copied from the current repo so it can run standalone with the same config files.

## What's here
- Agents: `live-fabio-agent-playbook*.ts` plus `live-topstepx-*.ts` variants and `restart-agents.sh`.
- Core logic: `lib/` (DeepSeek integration, execution manager, risk managers, playbook, cache) and shared `types/`.
- Dashboard: Next.js app under `app/` with TopstepX pages (`app/topstepx/**`) and assets in `public/`.
- Config: `package*.json`, `tsconfig.json`, `next.config.js`, `tailwind.config.js`, `postcss.config.js`, `vitest.config.ts`, `jest.config.js`, `ecosystem.config.js`, and `projectx-rest.ts`.

## Quick start
```bash
cd deepseek-topstepx-bundle
npm install
```

Set environment (example):
```bash
export OPENAI_API_KEY=your_deepseek_key
export OPENAI_BASE_URL=https://api.deepseek.com
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
- This is a copy; keep it in sync with root if upstream files change.
- Logs and `node_modules` were intentionally excluded to keep the bundle lean.
