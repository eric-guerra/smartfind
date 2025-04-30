// Query understanding: turn "the invoice for the blue pallet jack we bought in March over $500" into
// { types: [invoice], text: "blue pallet jack", dateFrom/dateTo: March, minAmount: 50000 }.
// A deterministic parser handles the common shapes; Claude (structured output) handles the long tail.
// The rule parser is always the fallback, so search never depends on the model being up.

import type Anthropic from '@anthropic-ai/sdk';

export type DocType = 'order' | 'invoice' | 'product';

export interface ParsedQuery {
  text: string;
  types: DocType[];
  dateFrom: string | null;
  dateTo: string | null;
  minAmountCents: number | null;
  maxAmountCents: number | null;
  status: string | null;
  source: 'rules' | 'llm';
}

const MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'];
const TYPE_WORDS: Record<string, DocType> = { invoice: 'invoice', invoices: 'invoice', bill: 'invoice', bills: 'invoice', order: 'order', orders: 'order', po: 'order', purchase: 'order', product: 'product', products: 'product', item: 'product', items: 'product', sku: 'product' };
const STATUS_WORDS = ['unpaid', 'paid', 'overdue', 'shipped', 'delivered', 'cancelled', 'open'];
const FILLER = /\b(the|a|an|for|we|i|our|my|that|which|bought|ordered|purchased|from|in|on|of|show|me|find|with|last|this|over|under|above|below|more|less|than|and|to)\b/g;

const pad = (n: number) => String(n).padStart(2, '0');
const lastDay = (y: number, m: number) => new Date(Date.UTC(y, m + 1, 0)).getUTCDate();

export function parseRules(q: string, today: Date): ParsedQuery {
  let s = ` ${q.toLowerCase()} `;
  const out: ParsedQuery = { text: '', types: [], dateFrom: null, dateTo: null, minAmountCents: null, maxAmountCents: null, status: null, source: 'rules' };

  const take = (re: RegExp, fn: (m: RegExpMatchArray) => void) => {
    const m = s.match(re);
    if (m) { fn(m); s = s.replace(m[0], ' '); }
  };

  take(/\b(over|above|more than|at least|>)\s*\$?\s*([\d,]+(?:\.\d{1,2})?)\s*(k)?\b/, (m) => { out.minAmountCents = Math.round(parseFloat(m[2].replace(/,/g, '')) * (m[3] ? 1000 : 1) * 100); });
  take(/\b(under|below|less than|at most|<)\s*\$?\s*([\d,]+(?:\.\d{1,2})?)\s*(k)?\b/, (m) => { out.maxAmountCents = Math.round(parseFloat(m[2].replace(/,/g, '')) * (m[3] ? 1000 : 1) * 100); });

  take(/\b(last|this) (week|month|year)\b/, (m) => {
    const y = today.getUTCFullYear(), mo = today.getUTCMonth();
    if (m[2] === 'month') {
      const [yy, mm] = m[1] === 'last' ? (mo === 0 ? [y - 1, 11] : [y, mo - 1]) : [y, mo];
      out.dateFrom = `${yy}-${pad(mm + 1)}-01`; out.dateTo = `${yy}-${pad(mm + 1)}-${pad(lastDay(yy, mm))}`;
    } else if (m[2] === 'year') {
      const yy = m[1] === 'last' ? y - 1 : y;
      out.dateFrom = `${yy}-01-01`; out.dateTo = `${yy}-12-31`;
    } else {
      const d = new Date(today); d.setUTCDate(d.getUTCDate() - (m[1] === 'last' ? 7 : 0) - ((d.getUTCDay() + 6) % 7));
      const e = new Date(d); e.setUTCDate(d.getUTCDate() + 6);
      out.dateFrom = d.toISOString().slice(0, 10); out.dateTo = e.toISOString().slice(0, 10);
    }
  });
  take(new RegExp(`\\b(${MONTHS.join('|')})(?:\\s+(\\d{4}))?\\b`), (m) => {
    const mm = MONTHS.indexOf(m[1]);
    // A bare month means the most recent one that isn't in the future.
    let yy = m[2] ? Number(m[2]) : today.getUTCFullYear();
    if (!m[2] && mm > today.getUTCMonth()) yy -= 1;
    out.dateFrom = `${yy}-${pad(mm + 1)}-01`; out.dateTo = `${yy}-${pad(mm + 1)}-${pad(lastDay(yy, mm))}`;
  });
  take(/\b(20\d{2})\b/, (m) => { if (!out.dateFrom) { out.dateFrom = `${m[1]}-01-01`; out.dateTo = `${m[1]}-12-31`; } });

  for (const w of STATUS_WORDS) take(new RegExp(`\\b${w}\\b`), () => { out.status = w; });
  for (const [w, t] of Object.entries(TYPE_WORDS)) {
    take(new RegExp(`\\b${w}\\b`), () => { if (!out.types.includes(t)) out.types.push(t); });
  }
  out.text = s.replace(FILLER, ' ').replace(/[^\w\s#-]/g, ' ').replace(/\s+/g, ' ').trim();
  return out;
}

const SCHEMA = {
  type: 'object',
  properties: {
    text: { type: 'string', description: 'What the thing is, without dates, amounts, types or status (e.g. "blue pallet jack")' },
    types: { type: 'array', items: { type: 'string', enum: ['order', 'invoice', 'product'] } },
    dateFrom: { type: ['string', 'null'], description: 'YYYY-MM-DD or null' },
    dateTo: { type: ['string', 'null'], description: 'YYYY-MM-DD or null' },
    minAmountCents: { type: ['integer', 'null'] },
    maxAmountCents: { type: ['integer', 'null'] },
    status: { type: ['string', 'null'], enum: [...STATUS_WORDS, null] },
  },
  required: ['text', 'types', 'dateFrom', 'dateTo', 'minAmountCents', 'maxAmountCents', 'status'],
  additionalProperties: false,
};

/** Rules first; only queries the rules leave ambiguous go to the model (keeps latency and cost low). */
export function needsModel(r: ParsedQuery, raw: string): boolean {
  const words = raw.trim().split(/\s+/).length;
  const relative = /\b(ago|since|before|after|between|quarter|q[1-4]|yesterday|recent|biggest|largest)\b/i.test(raw);
  return relative || (words >= 8 && r.types.length === 0);
}

export async function parseQuery(raw: string, today: Date, client?: Pick<Anthropic, 'beta'>): Promise<ParsedQuery> {
  const rules = parseRules(raw, today);
  if (!client || !needsModel(rules, raw)) return rules;
  try {
    const res = await client.beta.messages.create({
      model: 'claude-opus-5-5',
      max_tokens: 1000,
      system: `Extract search filters from a B2B buyer's search query. Today is ${today.toISOString().slice(0, 10)}. Amounts are in US cents. Leave fields null when the query doesn't say.`,
      output_config: { effort: 'low', format: { type: 'json_schema', schema: SCHEMA } },
      betas: ['server-side-fallback-2026-07-01'],
      fallbacks: 'default',
      messages: [{ role: 'user', content: raw }],
    }, { timeout: 4000, maxRetries: 0 }); // search is interactive: fall back fast rather than wait
    if (res.stop_reason !== 'end_turn') return rules;
    const j = JSON.parse(res.content.filter((b) => b.type === 'text').map((b: any) => b.text).join(''));
    return { ...j, source: 'llm' };
  } catch {
    return rules;
  }
}
