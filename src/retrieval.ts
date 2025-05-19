// Federated hybrid retrieval: lexical (Elasticsearch) + semantic (pgvector) per document type, fused
// with RRF. Filters and the caller's account scope are pushed into *both* retrievers.

import type { DocType, ParsedQuery } from './query.js';

export interface Doc {
  id: string;
  type: DocType;
  accountId: string; // ACL boundary: buyers only see their own company's orders and invoices
  title: string;
  body: string;
  date: string; // YYYY-MM-DD
  amountCents: number | null;
  status: string | null;
}

export interface Retriever {
  name: string;
  search(q: ParsedQuery, scope: { accountId: string }, k: number): Promise<{ id: string; score: number }[]>;
}

export function matchesFilters(d: Doc, q: ParsedQuery, scope: { accountId: string }): boolean {
  if (d.type !== 'product' && d.accountId !== scope.accountId) return false; // products are a shared catalog
  if (q.types.length && !q.types.includes(d.type)) return false;
  if (q.dateFrom && d.type !== 'product' && d.date < q.dateFrom) return false;
  if (q.dateTo && d.type !== 'product' && d.date > q.dateTo) return false;
  if (q.minAmountCents != null && (d.amountCents == null || d.amountCents < q.minAmountCents)) return false;
  if (q.maxAmountCents != null && (d.amountCents == null || d.amountCents > q.maxAmountCents)) return false;
  if (q.status && d.status !== q.status && !(q.status === 'unpaid' && (d.status === 'open' || d.status === 'overdue'))) return false;
  return true;
}

export function rrf(lists: { id: string }[][], k = 60): { id: string; score: number }[] {
  const s = new Map<string, number>();
  for (const list of lists) list.forEach((h, r) => s.set(h.id, (s.get(h.id) ?? 0) + 1 / (k + r + 1)));
  return [...s].map(([id, score]) => ({ id, score })).sort((a, b) => b.score - a.score);
}

/** Elasticsearch request for the lexical side (filters as `filter` clauses: cached, no scoring cost). */
export function esRequest(q: ParsedQuery, scope: { accountId: string }, k: number) {
  const filter: unknown[] = [{ bool: { should: [{ term: { type: 'product' } }, { term: { accountId: scope.accountId } }], minimum_should_match: 1 } }];
  if (q.types.length) filter.push({ terms: { type: q.types } });
  if (q.dateFrom || q.dateTo) filter.push({ bool: { should: [{ term: { type: 'product' } }, { range: { date: { gte: q.dateFrom ?? undefined, lte: q.dateTo ?? undefined } } }], minimum_should_match: 1 } });
  if (q.minAmountCents != null || q.maxAmountCents != null) filter.push({ range: { amountCents: { gte: q.minAmountCents ?? undefined, lte: q.maxAmountCents ?? undefined } } });
  if (q.status) filter.push({ terms: { status: q.status === 'unpaid' ? ['open', 'overdue'] : [q.status] } });
  return {
    size: k,
    query: { bool: {
      must: q.text ? { multi_match: { query: q.text, fields: ['title^3', 'body', 'id^5'], fuzziness: 'AUTO' } } : { match_all: {} },
      filter,
    } },
  };
}

/** pgvector SQL for the semantic side; same scope and filters as SQL predicates. */
export function pgvectorQuery(q: ParsedQuery, scope: { accountId: string }, k: number, embedding: number[]) {
  const where = ["(type = 'product' OR account_id = $2)"];
  const params: unknown[] = [`[${embedding.join(',')}]`, scope.accountId];
  const add = (sql: string, v: unknown) => { params.push(v); where.push(sql.replace('?', `$${params.length}`)); };
  if (q.types.length) add('type = ANY(?)', q.types);
  if (q.dateFrom) add("(type = 'product' OR date >= ?)", q.dateFrom);
  if (q.dateTo) add("(type = 'product' OR date <= ?)", q.dateTo);
  if (q.minAmountCents != null) add('amount_cents >= ?', q.minAmountCents);
  if (q.maxAmountCents != null) add('amount_cents <= ?', q.maxAmountCents);
  if (q.status) add('status = ANY(?)', q.status === 'unpaid' ? ['open', 'overdue'] : [q.status]);
  params.push(k);
  return {
    text: `SELECT id, 1 - (embedding <=> $1::vector) AS score FROM documents WHERE ${where.join(' AND ')} ORDER BY embedding <=> $1::vector LIMIT $${params.length}`,
    values: params,
  };
}

// ----------------------------------------------------------------- in-memory twins (tests / local dev)

const tok = (s: string): string[] => s.toLowerCase().match(/[a-z0-9#-]+/g) ?? [];

export class MemoryLexical implements Retriever {
  name = 'lexical';
  constructor(private docs: Doc[]) {}
  async search(q: ParsedQuery, scope: { accountId: string }, k: number) {
    const terms = tok(q.text);
    return this.docs.filter((d) => matchesFilters(d, q, scope))
      .map((d) => {
        const title = tok(d.title), body = tok(d.body);
        const score = terms.length ? terms.reduce((s, t) => s + (d.id.toLowerCase() === t ? 5 : 0) + (title.includes(t) ? 3 : 0) + (body.includes(t) ? 1 : 0), 0) : 1;
        return { id: d.id, score };
      })
      .filter((h) => h.score > 0).sort((a, b) => b.score - a.score || (a.id < b.id ? -1 : 1)).slice(0, k);
  }
}

/** Semantic twin: a tiny concept map stands in for an embedding model in tests. */
export class MemorySemantic implements Retriever {
  name = 'semantic';
  constructor(private docs: Doc[], private concepts: Record<string, string[]>) {}
  private expand(text: string) {
    const out = new Set<string>(tok(text));
    for (const t of [...out]) for (const [c, syns] of Object.entries(this.concepts)) if (c === t || syns.includes(t)) [c, ...syns].forEach((x) => out.add(x));
    return out;
  }
  async search(q: ParsedQuery, scope: { accountId: string }, k: number) {
    const qs = this.expand(q.text);
    return this.docs.filter((d) => matchesFilters(d, q, scope))
      .map((d) => {
        const ds = this.expand(`${d.title} ${d.body}`);
        const inter = [...qs].filter((t) => ds.has(t)).length;
        return { id: d.id, score: qs.size ? inter / Math.sqrt(qs.size * ds.size) : 0 };
      })
      .filter((h) => h.score > 0).sort((a, b) => b.score - a.score).slice(0, k);
  }
}

export async function hybridSearch(retrievers: Retriever[], q: ParsedQuery, scope: { accountId: string }, k = 10) {
  const lists = await Promise.all(retrievers.map((r) => r.search(q, scope, 50).catch(() => []))); // one backend down ≠ no results
  return rrf(lists).slice(0, k);
}
