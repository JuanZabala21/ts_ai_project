# ts_ai_project

A small TypeScript study project for working with the Claude API through the official
[`@anthropic-ai/sdk`](https://github.com/anthropics/anthropic-sdk-typescript).

## Setup

```bash
npm install
cp .env.example .env   # then paste your API key from https://platform.claude.com
```

Requires Node 22+. Scripts run TypeScript directly with `tsx`, so there's no build step.

## Examples

Work through them in order. Each one is a single file in `src/` that you can run by itself.

| Script | File | What you learn |
|---|---|---|
| `npm run basic` | `01-basic.ts` | One request and one response, system prompts, content blocks, `stop_reason`, token usage |
| `npm run chat` | `02-chat-stream.ts` | Multi-turn chat with streaming, keeping your own history, adaptive thinking summaries |
| `npm run tools` | `03-tools.ts` | Tool use: Zod-typed tools plus the SDK's tool runner, which drives the call loop |
| `npm run structured` | `04-structured-output.ts` | Structured outputs: getting back a validated, typed object |

Pass your own prompt as an argument, for example:

```bash
npm run basic -- "What is the difference between interface and type in TS?"
```

`npm run typecheck` runs `tsc` over everything.

## Notes

- **Model:** every example uses `claude-opus-5-5`, set in `src/client.ts`. For cheaper
  experiments, change it to `claude-sonnet-5-5` or `claude-haiku-5-5`.
- **Refusal fallback:** examples 01 and 02 use the beta `fallbacks: "default"` option. If a
  safety classifier declines a request, the API retries it on a recommended fallback model
  instead of returning a refusal. Remove the `...FALLBACK` spread if you don't want this.
- **Thinking:** Opus 5.5 always thinks. You control how much with
  `output_config: { effort: "low" | "medium" | "high" | "xhigh" | "max" }`, and the default is `medium`.

## Ideas for next steps

- Add prompt caching (`cache_control`) to a large system prompt, then check `usage.cache_read_input_tokens`.
- Send an image or PDF to Claude (vision and document input).
- Give the tool runner a real tool, such as fetching a URL or reading a local file.
- Wrap the chat in a small HTTP API with Hono or Express.
