// 04 - Structured outputs: constrain Claude's reply to a Zod schema and get
// back a typed, validated object instead of free text.
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import { z } from "zod";
import { client, MODEL } from "./client.js";

const Recipe = z.object({
  title: z.string(),
  servings: z.number(),
  prepMinutes: z.number(),
  ingredients: z.array(z.object({ item: z.string(), quantity: z.string() })),
  steps: z.array(z.string()),
});

const text =
  process.argv[2] ??
  "My grandma's arepas: mix 2 cups of precooked corn flour with 2.5 cups of warm water and a teaspoon of salt. " +
    "Knead 5 minutes, shape into 6 discs, cook on a hot griddle about 5 minutes per side. Takes about 25 minutes total.";

const response = await client.messages.parse({
  model: MODEL,
  max_tokens: 16000,
  messages: [{ role: "user", content: `Extract a structured recipe from this text:\n\n${text}` }],
  output_config: { format: zodOutputFormat(Recipe) },
});

const recipe = response.parsed_output; // typed as z.infer<typeof Recipe> | null
if (!recipe) {
  console.error("Could not parse a recipe. stop_reason:", response.stop_reason);
  process.exit(1);
}

console.log(`${recipe.title} (serves ${recipe.servings}, ${recipe.prepMinutes} min)`);
for (const { item, quantity } of recipe.ingredients) console.log(`  - ${quantity} ${item}`);
recipe.steps.forEach((s, i) => console.log(`  ${i + 1}. ${s}`));
