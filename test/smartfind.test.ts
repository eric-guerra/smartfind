import { describe, it, expect } from 'vitest';
import { needsModel, parseQuery, parseRules } from '../src/query.js';
import { Doc, esRequest, hybridSearch, matchesFilters, MemoryLexical, MemorySemantic, pgvectorQuery, rrf } from '../src/retrieval.js';
import { createServer } from '../src/graphql.js';

const today = new Date('2024-06-15T12:00:00Z');

const docs: Doc[] = [
  { id: 'INV-1042', type: 'invoice', accountId: 'acme', title: 'Invoice INV-1042', body: 'Pallet jack, 5500 lb capacity, blue. Freight included.', date: '2024-03-12', amountCents: 64900, status: 'open' },
  { id: 'INV-1043', type: 'invoice', accountId: 'acme', title: 'Invoice INV-1043', body: 'Hand truck convertible, steel.', date: '2024-03-20', amountCents: 21900, status: 'paid' },
  { id: 'INV-0999', type: 'invoice', accountId: 'acme', title: 'Invoice INV-0999', body: 'Pallet jack wheels replacement kit.', date: '2024-01-08', amountCents: 8900, status: 'paid' },
  { id: 'SO-5501', type: 'order', accountId: 'acme', title: 'Order SO-5501', body: 'Manual pallet truck, blue, 27 x 48 forks.', date: '2024-03-05', amountCents: 64900, status: 'delivered' },
  { id: 'INV-2001', type: 'invoice', accountId: 'globex', title: 'Invoice INV-2001', body: 'Pallet jack, blue, 5500 lb.', date: '2024-03-14', amountCents: 64900, status: 'open' },
  { id: 'PJ-55', type: 'product', accountId: '*', title: 'Pallet Jack 5500 lb', body: 'Manual hydraulic pallet jack, blue powder coat.', date: '2023-01-01', amountCents: 64900, status: null },
  { id: 'HT-10', type: 'product', accountId: '*', title: 'Convertible Hand Truck', body: 'Two-in-one dolly and cart.', date: '2023-01-01', amountCents: 21900, status: null },
];
const concepts = { 'pallet-jack': ['pallet', 'jack', 'truck', 'hydraulic'], dolly: ['hand', 'truck', 'cart', 'dolly'] };
const retrievers = [new MemoryLexical(docs), new MemorySemantic(docs, concepts)];
const byId = new Map(docs.map((d) => [d.id, d]));

describe('query understanding (rules)', () => {
  it('extracts type, month, amount and leaves the thing', () => {
    const q = parseRules('the invoice for the blue pallet jack we bought in March over $500', today);
    expect(q).toMatchObject({ types: ['invoice'], dateFrom: '2024-03-01', dateTo: '2024-03-31', minAmountCents: 50000, text: 'blue pallet jack' });
  });
  it('a future bare month means last year', () => {
    expect(parseRules('orders in september', today)).toMatchObject({ types: ['order'], dateFrom: '2023-09-01', dateTo: '2023-09-30' });
  });
  it('relative ranges, k amounts, status', () => {
    expect(parseRules('unpaid bills last month under 1.5k', today)).toMatchObject({ status: 'unpaid', types: ['invoice'], dateFrom: '2024-05-01', dateTo: '2024-05-31', maxAmountCents: 150000 });
    expect(parseRules('orders this year', today)).toMatchObject({ dateFrom: '2024-01-01', dateTo: '2024-12-31' });
  });
  it('only ambiguous queries go to the model', () => {
    expect(needsModel(parseRules('invoice pallet jack march', today), 'invoice pallet jack march')).toBe(false);
    expect(needsModel(parseRules('stuff from two quarters ago', today), 'stuff from two quarters ago')).toBe(true);
  });
});

describe('query understanding (model fallback)', () => {
  const llm = (impl: () => Promise<any>) => ({ beta: { messages: { create: impl } } }) as any;
  it('uses the structured output when the model answers', async () => {
    let req: any;
    const out = await parseQuery('biggest orders since Q4', today, llm(async (...args: any[]) => { req = args[0]; return {
      stop_reason: 'end_turn', content: [{ type: 'text', text: JSON.stringify({ text: '', types: ['order'], dateFrom: '2023-10-01', dateTo: null, minAmountCents: null, maxAmountCents: null, status: null }) }] }; }));
    expect(out).toMatchObject({ source: 'llm', types: ['order'], dateFrom: '2023-10-01' });
    expect(req.model).toBe('claude-opus-5-5');
    expect(req.output_config.format.type).toBe('json_schema');
    expect(req.fallbacks).toBe('default');
  });
  it('falls back to rules on error or refusal', async () => {
    expect((await parseQuery('orders since march', today, llm(async () => { throw new Error('timeout'); }))).source).toBe('rules');
    expect((await parseQuery('orders since march', today, llm(async () => ({ stop_reason: 'refusal', content: [] })))).source).toBe('rules');
  });
});

describe('retrieval', () => {
  const scope = { accountId: 'acme' };
  it('finds the right invoice: ACL, filters and synonyms all apply', async () => {
    const q = parseRules('the invoice for the blue pallet jack we bought in March over $500', today);
    const hits = await hybridSearch(retrievers, q, scope);
    expect(hits[0].id).toBe('INV-1042');
    expect(hits.map((h) => h.id)).not.toContain('INV-2001'); // another company's invoice
    expect(hits.map((h) => h.id)).not.toContain('INV-0999'); // January, $89
  });
  it('semantic side bridges vocabulary ("pallet truck" order for a "pallet jack" query)', async () => {
    const hits = await hybridSearch(retrievers, parseRules('pallet jack orders', today), scope);
    expect(hits[0].id).toBe('SO-5501');
  });
  it('products are visible to everyone; orders and invoices only to their account', () => {
    const q = parseRules('pallet', today);
    expect(matchesFilters(byId.get('PJ-55')!, q, { accountId: 'globex' })).toBe(true);
    expect(matchesFilters(byId.get('INV-1042')!, q, { accountId: 'globex' })).toBe(false);
  });
  it('one backend failing degrades instead of failing the search', async () => {
    const broken = { name: 'semantic', search: async () => { throw new Error('pg down'); } };
    const hits = await hybridSearch([retrievers[0], broken], parseRules('INV-1043', today), scope);
    expect(hits[0].id).toBe('INV-1043');
  });
  it('RRF rewards agreement', () => {
    expect(rrf([[{ id: 'a' }, { id: 'b' }], [{ id: 'b' }, { id: 'c' }]])[0].id).toBe('b');
  });
  it('backend queries carry scope and filters', () => {
    const q = parseRules('unpaid invoices in march over $500 pallet', today);
    const es = esRequest(q, { accountId: 'acme' }, 10) as any;
    const f = JSON.stringify(es.query.bool.filter);
    expect(f).toContain('"accountId":"acme"');
    expect(f).toContain('"status":["open","overdue"]');
    const pg = pgvectorQuery(q, { accountId: 'acme' }, 10, [0.1, 0.2]);
    expect(pg.text).toContain("(type = 'product' OR account_id = $2)");
    expect(pg.values).toContain(50000);
    expect(pg.text).toMatch(/LIMIT \$\d+$/);
  });
});

describe('GraphQL', () => {
  const server = createServer();
  const exec = async (q: string, accountId: string | null) => {
    const r = await server.executeOperation({ query: `query($q: String!) { search(q: $q) { interpretation { text types dateFrom source } hits { id type score } facets { type count } } }`, variables: { q } },
      { contextValue: { accountId, retrievers, docs: byId, today } });
    if (r.body.kind !== 'single') throw new Error();
    return r.body.singleResult;
  };
  it('returns interpretation, hits and facets', async () => {
    const r = await exec('blue pallet jack in March', 'acme');
    const s = (r.data as any).search;
    expect(s.interpretation).toMatchObject({ text: 'blue pallet jack', dateFrom: '2024-03-01', source: 'rules' });
    expect(s.hits.length).toBeGreaterThan(0);
    expect(s.facets.reduce((n: number, f: any) => n + f.count, 0)).toBe(s.hits.length);
  });
  it('requires auth', async () => {
    expect((await exec('x', null)).errors?.[0].extensions?.code).toBe('UNAUTHENTICATED');
  });
});
