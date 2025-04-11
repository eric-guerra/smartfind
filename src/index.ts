import Anthropic from '@anthropic-ai/sdk';
import { Client as Elastic } from '@elastic/elasticsearch';
import { startStandaloneServer } from '@apollo/server/standalone';
import pg from 'pg';
import { createServer } from './graphql.js';
import type { ParsedQuery } from './query.js';
import { Doc, esRequest, pgvectorQuery, Retriever } from './retrieval.js';

const es = new Elastic({ node: process.env.ES_URL ?? 'http://localhost:9200' });
const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });

/** Embeddings come from the embedding service (same model used at index time). */
async function embed(text: string): Promise<number[]> {
  const r = await fetch(`${process.env.EMBED_URL ?? 'http://localhost:8001'}/embed`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text }) });
  return (await r.json()).embedding;
}

const lexical: Retriever = {
  name: 'lexical',
  async search(q: ParsedQuery, scope, k) {
    const r = await es.search({ index: 'documents', ...(esRequest(q, scope, k) as object) });
    return r.hits.hits.map((h) => ({ id: h._id!, score: h._score ?? 0 }));
  },
};

const semantic: Retriever = {
  name: 'semantic',
  async search(q, scope, k) {
    if (!q.text) return [];
    const { text, values } = pgvectorQuery(q, scope, k, await embed(q.text));
    return (await pool.query(text, values)).rows.map((r) => ({ id: r.id, score: Number(r.score) }));
  },
};

async function loadDocs(ids: string[]): Promise<Map<string, Doc>> {
  const r = await pool.query('SELECT id, type, account_id, title, body, to_char(date, \'YYYY-MM-DD\') AS date, amount_cents, status FROM documents WHERE id = ANY($1)', [ids]);
  return new Map(r.rows.map((x) => [x.id, { id: x.id, type: x.type, accountId: x.account_id, title: x.title, body: x.body, date: x.date, amountCents: x.amount_cents, status: x.status }]));
}

const llm = new Anthropic();
const { url } = await startStandaloneServer(createServer(), {
  listen: { port: Number(process.env.PORT ?? 4100) },
  context: async ({ req }) => {
    const accountId = (req.headers['x-account-id'] as string) ?? null; // set by the gateway after auth
    // Docs are hydrated lazily per hit set; a proxy Map keeps the resolver code backend-agnostic.
    const cache = new Map<string, Doc>();
    const docs = new Proxy(cache, { get: (t, p) => (p === 'get' ? (id: string) => t.get(id) : Reflect.get(t, p)) });
    const retrievers = [lexical, semantic].map((r) => ({
      ...r,
      search: async (q: ParsedQuery, scope: { accountId: string }, k: number) => {
        const hits = await r.search(q, scope, k);
        for (const [id, d] of await loadDocs(hits.map((h) => h.id))) cache.set(id, d);
        return hits;
      },
    }));
    return { accountId, retrievers, docs, llm, today: new Date() };
  },
});
console.log(`SmartFind GraphQL at ${url}`);
