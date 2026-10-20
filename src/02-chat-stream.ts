// 02 - Multi-turn chat in the terminal, streaming tokens as they arrive.
// The API is stateless: we keep the history ourselves and resend it each turn.
import * as readline from "node:readline/promises";
import Anthropic from "@anthropic-ai/sdk";
import { client, MODEL, FALLBACK } from "./client.js";

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
const history: Anthropic.Beta.BetaMessageParam[] = [];

console.log('Chat with Claude. Type "exit" to quit.\n');

while (true) {
  const input = (await rl.question("you > ")).trim();
  if (!input) continue;
  if (input === "exit") break;

  history.push({ role: "user", content: input });

  const stream = client.beta.messages.stream({
    ...FALLBACK,
    model: MODEL,
    max_tokens: 64000,
    // Show a readable summary of Claude's reasoning (hidden by default).
    thinking: { type: "adaptive", display: "summarized" },
    messages: history,
  });

  process.stdout.write("claude > ");
  for await (const event of stream) {
    if (event.type === "content_block_start" && event.content_block.type === "thinking") {
      process.stdout.write("\x1b[2m[thinking] ");
    } else if (event.type === "content_block_delta") {
      if (event.delta.type === "thinking_delta") process.stdout.write(event.delta.thinking);
      if (event.delta.type === "text_delta") process.stdout.write(event.delta.text);
    } else if (event.type === "content_block_stop") {
      process.stdout.write("\x1b[0m");
    }
  }
  process.stdout.write("\n\n");

  const message = await stream.finalMessage();
  // Append the full content (including thinking blocks), not just the text.
  history.push({ role: "assistant", content: message.content });
}

rl.close();
