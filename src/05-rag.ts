// 05 - RAG (Retrieval-Augmented Generation): "chatea con tus documentos".
//
// Es el patrón de IA más usado en empresas: chatbots de soporte, buscadores
// internos, asistentes sobre manuales, contratos, wikis... El flujo siempre es:
//
//   1. INDEXAR   -> partir los documentos en fragmentos ("chunks").
//   2. RECUPERAR -> dada una pregunta, encontrar los fragmentos más relevantes.
//   3. GENERAR   -> enviar SOLO esos fragmentos a Claude y pedirle que responda
//                   basándose en ellos, citando de dónde sacó cada dato.
//
// ¿Por qué no mandar todos los documentos siempre? Porque en producción hay
// miles de páginas: sería caro, lento y el modelo se distrae con ruido.
//
// Ejecuta:  npm run rag -- "¿Puedo devolver un producto abierto?"
import Anthropic from "@anthropic-ai/sdk";
import { client, MODEL, FALLBACK } from "./client.js";

// ─────────────────────────────────────────────────────────────────────────────
// PASO 0: La "base de conocimiento".
// En la vida real esto vendría de PDFs, Notion, Confluence, una base de datos...
// Aquí usamos una empresa ficticia para que el ejemplo sea autocontenido.
// ─────────────────────────────────────────────────────────────────────────────
type SourceDoc = { id: string; title: string; text: string };

const DOCS: SourceDoc[] = [
  {
    id: "politica-devoluciones",
    title: "Política de devoluciones",
    text: `Los clientes pueden devolver cualquier producto dentro de los 30 días siguientes a la entrega.
Los productos deben estar sin abrir y en su empaque original para un reembolso completo.
Los productos abiertos solo se aceptan si presentan un defecto de fábrica; en ese caso se cambia por uno nuevo.
El reembolso se procesa en un máximo de 5 días hábiles al método de pago original.
Los productos en oferta de liquidación no admiten devolución.`,
  },
  {
    id: "envios",
    title: "Envíos y entregas",
    text: `Los envíos nacionales tardan entre 2 y 4 días hábiles.
El envío es gratis en compras superiores a 150.000 COP; por debajo de ese monto cuesta 12.000 COP.
Los envíos internacionales están disponibles solo para México, Chile y Perú, y tardan de 7 a 12 días hábiles.
Cada pedido incluye un número de seguimiento que se envía por correo electrónico.`,
  },
  {
    id: "garantia",
    title: "Garantía de productos",
    text: `Todos los dispositivos electrónicos tienen una garantía de 12 meses contra defectos de fábrica.
La garantía no cubre daños por agua, caídas ni modificaciones hechas por el usuario.
Para hacer válida la garantía se necesita la factura de compra y el número de serie del producto.`,
  },
  {
    id: "soporte",
    title: "Atención al cliente",
    text: `El soporte está disponible de lunes a viernes de 8:00 a 18:00 y los sábados de 9:00 a 13:00 (hora de Colombia).
Los canales oficiales son el chat de la web, el correo soporte@nubia.example y la línea 601 555 0100.
El tiempo de respuesta promedio por correo es de 24 horas.`,
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// PASO 1: INDEXAR — partir cada documento en fragmentos pequeños.
// Un fragmento = una línea aquí. En producción se suele cortar cada ~300-800
// tokens con algo de solapamiento, para no partir una idea por la mitad.
// ─────────────────────────────────────────────────────────────────────────────
type Chunk = { docId: string; title: string; text: string; tokens: string[] };

// Normaliza texto: minúsculas, sin tildes, y lo separa en palabras.
// Así "Devolución" y "devolucion" cuentan como la misma palabra.
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .normalize("NFD") // separa "ó" en "o" + tilde...
    .replace(/[̀-ͯ]/g, "") // ...y aquí borramos la tilde
    .split(/[^a-z0-9]+/) // corta por todo lo que no sea letra o número
    .filter((w) => w.length > 2); // descarta palabras muy cortas ("de", "el", "y")
}

const CHUNKS: Chunk[] = DOCS.flatMap((doc) =>
  doc.text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => ({ docId: doc.id, title: doc.title, text: line, tokens: tokenize(line) })),
);

// ─────────────────────────────────────────────────────────────────────────────
// PASO 2: RECUPERAR — puntuar cada fragmento contra la pregunta.
// Usamos una versión simple de BM25 (el algoritmo de búsqueda por palabras
// clave que usan Elasticsearch y OpenSearch). En producción normalmente se
// combina con "embeddings" + una base de datos vectorial (pgvector, Pinecone…)
// para encontrar también sinónimos; a eso se le llama "búsqueda híbrida".
// ─────────────────────────────────────────────────────────────────────────────
const avgLen = CHUNKS.reduce((sum, c) => sum + c.tokens.length, 0) / CHUNKS.length;

// IDF: una palabra que aparece en pocos fragmentos vale más que una muy común.
function idf(term: string): number {
  const n = CHUNKS.filter((c) => c.tokens.includes(term)).length;
  return Math.log(1 + (CHUNKS.length - n + 0.5) / (n + 0.5));
}

function bm25(query: string[], chunk: Chunk, k1 = 1.5, b = 0.75): number {
  let score = 0;
  for (const term of new Set(query)) {
    const tf = chunk.tokens.filter((t) => t === term).length; // veces que aparece
    if (tf === 0) continue;
    // Fórmula BM25: premia la frecuencia pero con rendimientos decrecientes,
    // y penaliza fragmentos muy largos (b controla cuánto).
    const norm = tf + k1 * (1 - b + (b * chunk.tokens.length) / avgLen);
    score += idf(term) * ((tf * (k1 + 1)) / norm);
  }
  return score;
}

// Devuelve los `k` fragmentos con mejor puntuación (descartando los de 0).
function retrieve(question: string, k = 4): Chunk[] {
  const q = tokenize(question);
  return CHUNKS.map((chunk) => ({ chunk, score: bm25(q, chunk) }))
    .filter((r) => r.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map((r) => r.chunk);
}

// ─────────────────────────────────────────────────────────────────────────────
// PASO 3: GENERAR — enviar la pregunta + fragmentos a Claude.
// ─────────────────────────────────────────────────────────────────────────────
const question = process.argv[2] ?? "Abrí un audífono y no me gustó, ¿lo puedo devolver? ¿Cuánto tarda el reembolso?";
const hits = retrieve(question);

console.log(`Pregunta: ${question}\n`);
console.log(`Fragmentos recuperados (${hits.length}):`);
hits.forEach((h, i) => console.log(`  [${i}] (${h.title}) ${h.text}`));
console.log();

// Convertimos cada fragmento en un bloque `search_result`. Es el formato que
// la API de Claude entiende como "resultado de búsqueda" y le permite CITAR
// el origen exacto de cada afirmación (como las notas al pie de un artículo).
const searchResults: Anthropic.Beta.BetaSearchResultBlockParam[] = hits.map((h) => ({
  type: "search_result",
  source: h.docId, // identificador de dónde vino (URL, id de documento...)
  title: h.title, // título legible que se muestra en las citas
  content: [{ type: "text", text: h.text }], // el texto del fragmento
  citations: { enabled: true }, // activa las citas automáticas
}));

// El system prompt define el comportamiento. Es FIJO entre peticiones, así que
// lo marcamos con `cache_control`: Anthropic guarda ese prefijo y las siguientes
// peticiones lo leen de caché (~90% más barato y más rápido).
// Ojo: solo se cachea si el prefijo supera un tamaño mínimo (~1-4k tokens según
// el modelo). Este prompt es corto, así que verás 0 en caché; en un sistema
// real (instrucciones largas, ejemplos, catálogos) sí se nota mucho.
const SYSTEM_PROMPT = `Eres el asistente de soporte de Nubia, una tienda de tecnología.
Responde SOLO con la información de los resultados de búsqueda que se te entregan.
Si la respuesta no está en ellos, dilo claramente y sugiere contactar a soporte; no inventes datos.
Responde en español, de forma breve y amable.`;

const response = await client.beta.messages.create({
  ...FALLBACK, // si los filtros de seguridad rechazan la petición, reintenta con otro modelo
  model: MODEL,
  max_tokens: 16000,
  // `effort` controla cuánto "piensa" el modelo. Para Q&A sobre documentos,
  // "low" o "medium" suele bastar y ahorra tokens y latencia.
  output_config: { effort: "low" },
  system: [{ type: "text", text: SYSTEM_PROMPT, cache_control: { type: "ephemeral" } }],
  messages: [
    {
      role: "user",
      // Primero el contexto (los resultados), luego la pregunta: a los modelos
      // les va mejor cuando el material de referencia va antes de la tarea.
      content: [...searchResults, { type: "text", text: question }],
    },
  ],
});

// Siempre revisa por qué se detuvo el modelo antes de usar la respuesta.
if (response.stop_reason === "refusal") {
  console.error("La petición fue rechazada:", response.stop_details);
  process.exit(1);
}

// ─────────────────────────────────────────────────────────────────────────────
// PASO 4: Mostrar la respuesta con sus citas.
// Con citas activadas, la respuesta llega partida en varios bloques de texto;
// los que están respaldados por una fuente traen un arreglo `citations`.
// ─────────────────────────────────────────────────────────────────────────────
const sources: string[] = []; // lista de notas al pie, sin duplicados
let answer = "";

for (const block of response.content) {
  if (block.type !== "text") continue; // ignoramos bloques de "thinking"
  answer += block.text;

  for (const c of block.citations ?? []) {
    if (c.type !== "search_result_location") continue;
    const note = `${c.title} (${c.source}): "${c.cited_text.trim()}"`;
    let n = sources.indexOf(note);
    if (n === -1) n = sources.push(note) - 1;
    answer += ` [${n + 1}]`; // marcador tipo nota al pie justo tras el texto citado
  }
}

console.log("Respuesta:\n" + answer.trim());
if (sources.length) {
  console.log("\nFuentes:");
  sources.forEach((s, i) => console.log(`  [${i + 1}] ${s}`));
}

// Métricas de uso: en producción esto se registra para controlar costos.
const u = response.usage;
console.log("\n---");
console.log(`tokens entrada/salida: ${u.input_tokens}/${u.output_tokens}`);
console.log(`caché escrita/leída: ${u.cache_creation_input_tokens ?? 0}/${u.cache_read_input_tokens ?? 0}`);
