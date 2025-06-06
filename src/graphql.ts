import { ApolloServer } from '@apollo/server';
import { GraphQLError } from 'graphql';
import type Anthropic from '@anthropic-ai/sdk';
import { parseQuery } from './query.js';
import { Doc, hybridSearch, Retriever } from './retrieval.js';

export const typeDefs = /* GraphQL */ `
  enum DocType { order invoice product }

  type Interpretation {
    text: String!
    types: [DocType!]!
    dateFrom: String
    dateTo: String
    minAmountCents: Int
    maxAmountCents: Int
    status: String
    source: String!
  }

  type SearchHit { id: ID!, type: DocType!, title: String!, date: String!, amountCents: Int, status: String, score: Float! }

  type SearchResult {
    interpretation: Interpretation!
    hits: [SearchHit!]!
    facets: [Facet!]!
  }

  type Facet { type: DocType!, count: Int! }

  type Query { search(q: String!, first: Int = 10): SearchResult! }
`;

export interface Ctx {
  accountId: string | null;
  retrievers: Retriever[];
  docs: Map<string, Doc>;
  llm?: Pick<Anthropic, 'beta'>;
  today: Date;
}

export const resolvers = {
  Query: {
    async search(_: unknown, { q, first }: { q: string; first: number }, ctx: Ctx) {
      if (!ctx.accountId) throw new GraphQLError('sign in required', { extensions: { code: 'UNAUTHENTICATED' } });
      if (q.length > 300) throw new GraphQLError('query too long', { extensions: { code: 'BAD_USER_INPUT' } });
      const interpretation = await parseQuery(q, ctx.today, ctx.llm);
      const fused = await hybridSearch(ctx.retrievers, interpretation, { accountId: ctx.accountId }, Math.min(first, 50));
      const hits = fused.map((h) => ({ ...ctx.docs.get(h.id)!, score: Math.round(h.score * 10000) / 10000 })).filter((h) => h.id);
      const facets = (['order', 'invoice', 'product'] as const).map((type) => ({ type, count: hits.filter((h) => h.type === type).length })).filter((f) => f.count);
      return { interpretation, hits, facets };
    },
  },
};

export const createServer = () => new ApolloServer<Ctx>({ typeDefs, resolvers, includeStacktraceInErrorResponses: false });
