// 01 - The simplest possible request: one prompt in, one response out.
import { client, MODEL, FALLBACK } from "./client.js";

const prompt = process.argv[2] ?? "Explain what a TypeScript generic is in 3 sentences.";

const response = await client.beta.messages.create({
  ...FALLBACK,
  model: MODEL,
  max_tokens: 16000,
  system: "You are a patient programming tutor.",
  messages: [{ role: "user", content: prompt }],
});

// Always check why the model stopped before trusting the content.
if (response.stop_reason === "refusal") {
  console.error("Request was declined:", response.stop_details);
  process.exit(1);
}

// `content` is a list of blocks (text, thinking, tool_use, ...). Narrow by type.
for (const block of response.content) {
  if (block.type === "text") console.log(block.text);
}

console.log("\n---");
console.log(`model: ${response.model} | stop: ${response.stop_reason}`);
console.log(`tokens in/out: ${response.usage.input_tokens}/${response.usage.output_tokens}`);
