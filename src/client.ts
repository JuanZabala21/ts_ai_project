import Anthropic from "@anthropic-ai/sdk";

// Reads ANTHROPIC_API_KEY (or an `ant auth login` profile) from the environment.
export const client = new Anthropic();

export const MODEL = "claude-opus-5-5";

// Server-side refusal fallback: if Claude's safety classifiers decline a
// request, the API re-runs it on a recommended fallback model in the same call.
// Spread this into `client.beta.messages.*` calls.
export const FALLBACK = {
  betas: ["server-side-fallback-2026-07-01"],
  fallbacks: "default",
} satisfies Partial<Anthropic.Beta.MessageCreateParams>;
