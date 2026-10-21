# SmartFind

Portfolio reference implementation of a 2025 project, built to show one search box for B2B buyers across **orders, invoices and the product catalog**. Buyers type the way they think, for example "the invoice for the blue pallet jack we bought in March over $500", and get the right document.

**Stack:** TypeScript, GraphQL (Apollo Server 4), Elasticsearch 8 (lexical), PostgreSQL + pgvector (semantic), Anthropic TypeScript SDK (Claude Opus 5.5, structured outputs) for long-tail query parsing.

## How a query is answered
1. **Understand** (`src/query.ts`):
   - A deterministic parser pulls out the document type, dates (months, "last month", "this year"; a bare future month means last year's), amount bounds ("over $500", "under 1.5k") and status ("unpaid" maps to open or overdue). What's left is the thing being searched for.
   - Only ambiguous queries go to Claude ("two quarters ago", "biggest", "since Q4", or long queries without a type). Claude returns a JSON-schema structured output with `effort: "low"`, a 4-second timeout and no retries.
   - On any error, refusal or timeout it **falls back to the rule parser**, so search never depends on the model being up.
2. **Retrieve:** Elasticsearch BM25 (title^3, body, id^5, fuzzy) and pgvector cosine similarity run in parallel, then **RRF** fuses the two lists.
   - The account scope and every filter are pushed into **both** backends: ES `filter` clauses and SQL predicates.
   - Products are a shared catalog. Orders and invoices are visible only to their own account.
   - If one backend fails, results degrade rather than the search failing.
3. **Respond:** GraphQL returns hits, per-type facets, and the **interpretation**, so the UI can show "invoices · March 2024 · over $500" as removable chips.

## Test
```bash
npm install && npm test && npm run typecheck
```
14 tests, using in-memory twins of both retrievers and a fake Claude client. They cover:
- Rule parsing: month, year rollover, relative ranges, "k" amounts, status.
- When the model is used, and the fallback on error or refusal.
- The target invoice ranking first while another account's identical invoice is excluded.
- Synonym bridging ("pallet jack" matching a "pallet truck" order).
- ACL rules, degraded mode, RRF.
- That the ES and pgvector queries carry the scope and filters.
- The GraphQL interpretation, facets and auth.
