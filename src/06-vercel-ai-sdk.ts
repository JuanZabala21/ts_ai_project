// 06 - Vercel AI SDK (paquete `ai`, versión 7).
//
// ¿Qué es y en qué se diferencia de lo que hicimos en 01-05?
// En 01-05 usamos el SDK OFICIAL de Anthropic: acceso a todas las funciones de
// Claude, pero el código queda atado a Anthropic. El AI SDK de Vercel es una
// capa UNIFICADA: la misma función `generateText` sirve para Claude, GPT,
// Gemini, Mistral… y cambiar de proveedor es cambiar UNA línea (`model`).
// Además trae integración lista con React/Next.js (hooks como `useChat`).
// Por eso es de lo más usado en apps web con IA.
//
// Este archivo tiene 4 demos, las mismas ideas de 01-04 pero con el AI SDK:
//   1. generateText   -> pedir texto y esperar la respuesta completa
//   2. streamText     -> recibir el texto palabra por palabra (como ChatGPT)
//   3. Output.object  -> obtener un objeto tipado y validado con Zod
//   4. tool + stopWhen -> darle herramientas al modelo (un mini agente)
//
// Ejecuta todas:   npm run vercel
// Ejecuta una:     npm run vercel -- 3
import { generateText, streamText, Output, tool, isStepCount } from "ai";
import { anthropic, type AnthropicLanguageModelOptions } from "@ai-sdk/anthropic";
import { z } from "zod";

// ─────────────────────────────────────────────────────────────────────────────
// El "modelo". `anthropic(...)` crea el modelo usando ANTHROPIC_API_KEY del
// entorno (la misma variable de tu .env). Para usar otro proveedor bastaría con:
//   import { openai } from "@ai-sdk/openai";   const model = openai("...");
// y el resto del archivo quedaría IGUAL. Esa es la gracia del AI SDK.
// ─────────────────────────────────────────────────────────────────────────────
const model = anthropic("claude-opus-5-5");

// Opciones propias de Anthropic que no existen en otros proveedores van en
// `providerOptions.anthropic`. `satisfies` hace que TypeScript las valide.
// `effort: "low"` = pensar menos -> respuestas más rápidas y baratas.
const claudeOptions = {
  anthropic: { effort: "low" } satisfies AnthropicLanguageModelOptions,
};

// ═════════════════════════════════════════════════════════════════════════════
// DEMO 1: generateText — una pregunta, una respuesta (equivale a 01-basic.ts).
// ═════════════════════════════════════════════════════════════════════════════
async function demoGenerateText() {
  const result = await generateText({
    model,
    // En la v7, el "system prompt" se llama `instructions` (antes `system`).
    instructions: "Eres un tutor de programación paciente. Responde en español.",
    prompt: "Explica qué es un genérico en TypeScript en 2 frases.",
    providerOptions: claudeOptions,
  });

  console.log(result.text); // el texto final, ya unido (sin recorrer bloques)
  // `finishReason` es el equivalente unificado de `stop_reason`:
  // "stop" (terminó bien), "length" (se cortó por límite), "tool-calls", etc.
  console.log(`\n[fin: ${result.finishReason} | tokens in/out: ${result.usage.inputTokens}/${result.usage.outputTokens}]`);
}

// ═════════════════════════════════════════════════════════════════════════════
// DEMO 2: streamText — el texto llega en trozos mientras se genera
// (equivale a 02-chat-stream.ts). Fíjate que NO lleva `await` delante:
// streamText devuelve el objeto al instante y el texto va llegando después.
// ═════════════════════════════════════════════════════════════════════════════
async function demoStreamText() {
  const result = streamText({
    model,
    instructions: "Eres un narrador breve.",
    prompt: "Cuenta una historia de 4 frases sobre un robot que aprende a cocinar arepas.",
    providerOptions: claudeOptions,
  });

  // `textStream` es un iterable asíncrono: cada vuelta trae un trozo de texto.
  for await (const chunk of result.textStream) {
    process.stdout.write(chunk); // write (no log) para no meter saltos de línea
  }

  // Las propiedades del resultado son promesas que se resuelven al terminar.
  const usage = await result.usage;
  console.log(`\n\n[tokens in/out: ${usage.inputTokens}/${usage.outputTokens}]`);
}

// ═════════════════════════════════════════════════════════════════════════════
// DEMO 3: salida estructurada — el modelo devuelve un objeto que cumple un
// esquema Zod (equivale a 04-structured-output.ts). Caso real muy común:
// clasificar tickets de soporte para enrutarlos automáticamente.
// ═════════════════════════════════════════════════════════════════════════════
const Ticket = z.object({
  categoria: z.enum(["facturacion", "envio", "producto_defectuoso", "otro"]),
  urgencia: z.enum(["baja", "media", "alta"]),
  sentimiento: z.enum(["positivo", "neutral", "negativo"]),
  resumen: z.string().describe("Resumen del problema en una frase"),
  // `.describe()` le explica al modelo qué poner en cada campo.
  accionSugerida: z.string().describe("Siguiente paso para el agente de soporte"),
});

async function demoStructured() {
  const email =
    "Hola, pedí unos audífonos hace 10 días y todavía no llegan. " +
    "El número de seguimiento no funciona y ya me cobraron dos veces. Estoy muy molesto.";

  const { output } = await generateText({
    model,
    // `Output.object` le dice al SDK: "la respuesta debe ser este esquema".
    // El SDK valida el resultado y, si no cumple, lanza un error.
    output: Output.object({ schema: Ticket }),
    prompt: `Clasifica este ticket de soporte:\n\n${email}`,
    providerOptions: claudeOptions,
  });

  // `output` está TIPADO: escribe `output.` en el editor y verás los campos.
  console.log(output);
  if (output.urgencia === "alta") console.log("\n-> Escalar a un supervisor");
}

// ═════════════════════════════════════════════════════════════════════════════
// DEMO 4: herramientas + varios pasos — un mini agente (equivale a 03-tools.ts).
// El modelo decide qué herramienta llamar, el SDK la ejecuta, le devuelve el
// resultado al modelo y repite hasta que el modelo responde con texto.
// ═════════════════════════════════════════════════════════════════════════════

// Simulamos una base de datos de pedidos.
const PEDIDOS: Record<string, { estado: string; transportadora: string; diasRetraso: number }> = {
  "A-1001": { estado: "en tránsito", transportadora: "Servientrega", diasRetraso: 3 },
  "A-1002": { estado: "entregado", transportadora: "Coordinadora", diasRetraso: 0 },
};

const tools = {
  // `tool()` define una herramienta: descripción (para que el modelo sepa cuándo
  // usarla), esquema de entrada (Zod) y la función que se ejecuta de verdad.
  buscarPedido: tool({
    description: "Busca el estado de un pedido por su ID (formato A-XXXX).",
    inputSchema: z.object({ pedidoId: z.string() }),
    execute: async ({ pedidoId }) => PEDIDOS[pedidoId] ?? { error: `No existe el pedido ${pedidoId}` },
  }),
  calcularCompensacion: tool({
    description: "Calcula el cupón de compensación en COP según los días de retraso.",
    inputSchema: z.object({ diasRetraso: z.number().int().min(0) }),
    execute: async ({ diasRetraso }) => ({ cuponCOP: Math.min(diasRetraso * 5000, 30000) }),
  }),
};

async function demoTools() {
  const result = streamText({
    model,
    instructions: "Eres un agente de soporte. Usa las herramientas para verificar datos antes de responder.",
    prompt: "Mi pedido A-1001 está retrasado, ¿dónde está y me dan alguna compensación?",
    tools,
    // Sin `stopWhen` el SDK se detiene tras la primera llamada a herramienta.
    // Con isStepCount(5) permite hasta 5 rondas modelo -> herramienta -> modelo.
    // Es un límite de seguridad para que un agente no entre en bucle infinito.
    stopWhen: isStepCount(5),
    providerOptions: claudeOptions,
  });

  // `result.stream` trae TODOS los eventos (no solo texto): así podemos mostrar
  // en pantalla cuándo el agente usa cada herramienta.
  for await (const part of result.stream) {
    switch (part.type) {
      case "tool-call":
        console.log(`  [herramienta] ${part.toolName}(${JSON.stringify(part.input)})`);
        break;
      case "tool-result":
        console.log(`  [resultado]   ${JSON.stringify(part.output)}`);
        break;
      case "text-delta":
        process.stdout.write(part.text);
        break;
      case "error":
        console.error("\n[error]", part.error);
        break;
    }
  }

  const steps = await result.steps; // cada ronda del bucle queda registrada
  console.log(`\n\n[pasos: ${steps.length}]`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Ejecutar: todas las demos, o solo la que pases como argumento.
// ─────────────────────────────────────────────────────────────────────────────
const demos = [
  ["generateText", demoGenerateText],
  ["streamText", demoStreamText],
  ["Salida estructurada", demoStructured],
  ["Herramientas (agente)", demoTools],
] as const;

const elegida = Number(process.argv[2]); // NaN si no se pasa nada
for (const [i, [nombre, demo]] of demos.entries()) {
  if (elegida && elegida !== i + 1) continue;
  console.log(`\n═══ DEMO ${i + 1}: ${nombre} ═══\n`);
  await demo();
}
