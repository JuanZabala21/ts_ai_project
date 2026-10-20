// 03 - Tool use: give Claude functions it can call. The SDK's tool runner
// handles the loop (Claude asks for a tool -> we run it -> send the result back).
import { betaZodTool } from "@anthropic-ai/sdk/helpers/beta/zod";
import { z } from "zod";
import { client, MODEL } from "./client.js";

const calculator = betaZodTool({
  name: "calculator",
  description: "Perform a basic arithmetic operation on two numbers.",
  inputSchema: z.object({
    a: z.number(),
    b: z.number(),
    op: z.enum(["add", "subtract", "multiply", "divide"]),
  }),
  run: async ({ a, b, op }) => {
    console.log(`  [tool] calculator(${a} ${op} ${b})`);
    if (op === "divide" && b === 0) return "Error: division by zero";
    const result = { add: a + b, subtract: a - b, multiply: a * b, divide: a / b }[op];
    return String(result);
  },
});

const getTime = betaZodTool({
  name: "get_current_time",
  description: "Get the current date and time in a given IANA time zone.",
  inputSchema: z.object({
    timeZone: z.string().describe("IANA time zone, e.g. America/Bogota"),
  }),
  run: async ({ timeZone }) => {
    console.log(`  [tool] get_current_time(${timeZone})`);
    try {
      return new Date().toLocaleString("en-US", { timeZone });
    } catch {
      return `Error: unknown time zone "${timeZone}"`;
    }
  },
});

const prompt =
  process.argv[2] ??
  "What time is it in Tokyo and in Madrid right now? Also, what's 1234 * 5678 divided by 7?";

const finalMessage = await client.beta.messages.toolRunner({
  model: MODEL,
  max_tokens: 16000,
  tools: [calculator, getTime],
  messages: [{ role: "user", content: prompt }],
});

for (const block of finalMessage.content) {
  if (block.type === "text") console.log(block.text);
}
