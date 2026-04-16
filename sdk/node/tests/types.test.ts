import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { WeavClient, WeavError, contextToPrompt, contextToMessages, scopeToGraph } from '../src/index.js';
import type {
  ContextResult,
  ContextParams,
  GraphInfo,
  NodeInfo,
  AddNodeParams,
  AddEdgeParams,
  ScopeRef,
} from '../src/index.js';

// ── WeavClient construction ─────────────────────────────────────────────────

describe('WeavClient construction', () => {
  it('uses default host and port when no config is provided', () => {
    const client = new WeavClient();
    assert.equal(client.getBaseUrl(), 'http://localhost:6382');
  });

  it('uses custom host and port when provided', () => {
    const client = new WeavClient({ host: '10.0.0.1', port: 9999 });
    assert.equal(client.getBaseUrl(), 'http://10.0.0.1:9999');
  });

  it('uses default port when only host is provided', () => {
    const client = new WeavClient({ host: 'myhost' });
    assert.equal(client.getBaseUrl(), 'http://myhost:6382');
  });

  it('uses default host when only port is provided', () => {
    const client = new WeavClient({ port: 8080 });
    assert.equal(client.getBaseUrl(), 'http://localhost:8080');
  });
});

describe('WeavClient HTTP paths', () => {
  it('info() requests /v1/info', async () => {
    const client = new WeavClient({ host: '127.0.0.1', port: 9000 });
    const originalFetch = globalThis.fetch;
    const calls: string[] = [];

    globalThis.fetch = (async (input: string | URL | Request) => {
      calls.push(String(input));
      return {
        ok: true,
        json: async () => ({ success: true, data: 'weav-server 0.1.0' }),
      } as Response;
    }) as typeof fetch;

    try {
      const info = await client.info();
      assert.equal(info, 'weav-server 0.1.0');
      assert.equal(calls[0], 'http://127.0.0.1:9000/v1/info');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('context() serializes retrieval_mode', async () => {
    const client = new WeavClient({ host: '127.0.0.1', port: 9000 });
    const originalFetch = globalThis.fetch;
    let requestBody: Record<string, unknown> | undefined;

    globalThis.fetch = (async (_input: string | URL | Request, init?: RequestInit) => {
      requestBody = JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>;
      return {
        ok: true,
        json: async () => ({
          success: true,
          data: {
            chunks: [],
            total_tokens: 0,
            budget_used: 0,
            nodes_considered: 0,
            nodes_included: 0,
            query_time_us: 0,
          },
        }),
      } as Response;
    }) as typeof fetch;

    try {
      await client.context({ graph: 'g', query: 'rust', retrievalMode: 'hybrid' });
      assert.equal(requestBody?.retrieval_mode, 'hybrid');
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('context() serializes rerank config', async () => {
    const client = new WeavClient({ host: '127.0.0.1', port: 9000 });
    const originalFetch = globalThis.fetch;
    let requestBody: Record<string, unknown> | undefined;

    globalThis.fetch = (async (_input: string | URL | Request, init?: RequestInit) => {
      requestBody = JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>;
      return {
        ok: true,
        json: async () => ({
          success: true,
          data: {
            chunks: [],
            total_tokens: 0,
            budget_used: 0,
            nodes_considered: 0,
            nodes_included: 0,
            query_time_us: 0,
          },
        }),
      } as Response;
    }) as typeof fetch;

    try {
      await client.context({
        graph: 'g',
        query: 'systems programming',
        rerank: {
          provider: 'cross_encoder',
          candidateLimit: 25,
          scoreWeight: 0.5,
        },
      });
      assert.deepEqual(requestBody?.rerank, {
        provider: 'cross_encoder',
        candidate_limit: 25,
        score_weight: 0.5,
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('context() serializes scope when graph is omitted', async () => {
    const client = new WeavClient({ host: '127.0.0.1', port: 9000 });
    const originalFetch = globalThis.fetch;
    let requestBody: Record<string, unknown> | undefined;

    globalThis.fetch = (async (_input: string | URL | Request, init?: RequestInit) => {
      requestBody = JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>;
      return {
        ok: true,
        json: async () => ({
          success: true,
          data: {
            chunks: [],
            total_tokens: 0,
            budget_used: 0,
            nodes_considered: 0,
            nodes_included: 0,
            query_time_us: 0,
          },
        }),
      } as Response;
    }) as typeof fetch;

    try {
      await client.context({
        scope: { workspaceId: 'acme', userId: 'u_123' },
        query: 'rust',
      });
      assert.deepEqual(requestBody?.scope, {
        workspace_id: 'acme',
        user_id: 'u_123',
      });
      assert.equal(requestBody?.graph, undefined);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('context() parses snake_case response fields into SDK types', async () => {
    const client = new WeavClient({ host: '127.0.0.1', port: 9000 });
    const originalFetch = globalThis.fetch;

    globalThis.fetch = (async () => {
      return {
        ok: true,
        json: async () => ({
          success: true,
          data: {
            chunks: [
              {
                node_id: 7,
                content: 'Alice works on retrieval quality.',
                label: 'person',
                relevance_score: 0.91,
                depth: 1,
                token_count: 6,
                provenance: { source: 'doc-1', confidence: 0.8 },
                relationships: [
                  {
                    edge_label: 'works_on',
                    target_node_id: 9,
                    target_name: 'retrieval',
                    direction: 'outgoing',
                    weight: 1.0,
                  },
                ],
              },
            ],
            total_tokens: 6,
            budget_used: 0.25,
            nodes_considered: 12,
            nodes_included: 1,
            query_time_us: 345,
          },
        }),
      } as Response;
    }) as typeof fetch;

    try {
      const result = await client.context({ graph: 'g', query: 'retrieval quality' });
      assert.equal(result.totalTokens, 6);
      assert.equal(result.budgetUtilization, 0.25);
      assert.equal(result.nodesConsidered, 12);
      assert.equal(result.nodesIncluded, 1);
      assert.equal(result.queryTimeUs, 345);
      assert.equal(result.chunks[0]?.nodeId, 7);
      assert.equal(result.chunks[0]?.relevanceScore, 0.91);
      assert.equal(result.chunks[0]?.tokenCount, 6);
      assert.deepEqual(result.chunks[0]?.provenance, {
        source: 'doc-1',
        confidence: 0.8,
      });
      assert.deepEqual(result.chunks[0]?.relationships, [
        {
          edge_label: 'works_on',
          target_node_id: 9,
          target_name: 'retrieval',
          direction: 'outgoing',
          weight: 1.0,
        },
      ]);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('ingest() serializes the current ingest options', async () => {
    const client = new WeavClient({ host: '127.0.0.1', port: 9000 });
    const originalFetch = globalThis.fetch;
    let requestBody: Record<string, unknown> | undefined;

    globalThis.fetch = (async (_input: string | URL | Request, init?: RequestInit) => {
      requestBody = JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>;
      return {
        ok: true,
        json: async () => ({
          success: true,
          data: {
            document_id: 'doc-1',
            chunks_created: 2,
            entities_created: 3,
            entities_merged: 1,
            entities_resolved: 4,
            entities_linked_existing: 2,
            relationships_created: 5,
            pipeline_duration_ms: 123,
          },
        }),
      } as Response;
    }) as typeof fetch;

    try {
      await client.ingest('g', {
        content: 'Alice works at Acme.',
        documentId: 'doc-1',
        skipExtraction: true,
        skipDedup: false,
        chunkSize: 64,
        entityTypes: ['person', 'organization'],
        resolutionMode: 'semantic',
        linkExistingEntities: true,
        resolutionCandidateLimit: 12,
        customResolutionPrompt: 'Prefer canonical company names.',
      });
      assert.deepEqual(requestBody, {
        content: 'Alice works at Acme.',
        document_id: 'doc-1',
        skip_extraction: true,
        skip_dedup: false,
        chunk_size: 64,
        entity_types: ['person', 'organization'],
        resolution_mode: 'semantic',
        link_existing_entities: true,
        resolution_candidate_limit: 12,
        custom_resolution_prompt: 'Prefer canonical company names.',
      });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('ingest() parses the response fields into SDK result types', async () => {
    const client = new WeavClient({ host: '127.0.0.1', port: 9000 });
    const originalFetch = globalThis.fetch;

    globalThis.fetch = (async () => {
      return {
        ok: true,
        json: async () => ({
          success: true,
          data: {
            document_id: 'doc-42',
            chunks_created: 8,
            entities_created: 3,
            entities_merged: 1,
            entities_resolved: 6,
            entities_linked_existing: 4,
            relationships_created: 9,
            pipeline_duration_ms: 456,
          },
        }),
      } as Response;
    }) as typeof fetch;

    try {
      const result = await client.ingest('g', { content: 'Hello world' });
      assert.equal(result.documentId, 'doc-42');
      assert.equal(result.chunksCreated, 8);
      assert.equal(result.entitiesCreated, 3);
      assert.equal(result.entitiesMerged, 1);
      assert.equal(result.entitiesResolved, 6);
      assert.equal(result.entitiesLinkedExisting, 4);
      assert.equal(result.relationshipsCreated, 9);
      assert.equal(result.pipelineDurationMs, 456);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});

describe('scopeToGraph', () => {
  it('resolves the canonical graph hierarchy', () => {
    const scope: ScopeRef = {
      workspaceId: 'acme',
      userId: 'u_123',
      agentId: 'a_support',
      sessionId: 's_456',
    };
    assert.equal(
      scopeToGraph(scope),
      'ws:acme:user:u_123:agent:a_support:session:s_456',
    );
  });
});

// ── WeavError ───────────────────────────────────────────────────────────────

describe('WeavError', () => {
  it('has correct name and message', () => {
    const err = new WeavError('something failed', 404);
    assert.equal(err.name, 'WeavError');
    assert.equal(err.message, 'something failed');
    assert.equal(err.statusCode, 404);
    assert.ok(err instanceof Error);
  });

  it('works without a status code', () => {
    const err = new WeavError('no status');
    assert.equal(err.statusCode, undefined);
    assert.equal(err.message, 'no status');
  });
});

// ── contextToPrompt ─────────────────────────────────────────────────────────

describe('contextToPrompt', () => {
  it('formats an empty result', () => {
    const result: ContextResult = {
      chunks: [],
      totalTokens: 0,
      budgetUtilization: 0,
      nodesConsidered: 0,
      nodesIncluded: 0,
      queryTimeUs: 0,
    };
    const text = contextToPrompt(result);
    assert.equal(text, '');
  });

  it('formats a single chunk without relationships', () => {
    const result: ContextResult = {
      chunks: [
        {
          nodeId: 1,
          content: 'Alice is an engineer.',
          label: 'person',
          relevanceScore: 0.95,
          depth: 0,
          tokenCount: 5,
          relationships: [],
        },
      ],
      totalTokens: 5,
      budgetUtilization: 0.1,
      nodesConsidered: 10,
      nodesIncluded: 1,
      queryTimeUs: 123,
    };
    const text = contextToPrompt(result);
    const lines = text.split('\n');
    assert.equal(lines[0], '[person] (score: 0.95)');
    assert.equal(lines[1], 'Alice is an engineer.');
    assert.equal(lines[2], '');
  });

  it('formats chunks with relationships', () => {
    const result: ContextResult = {
      chunks: [
        {
          nodeId: 1,
          content: 'Alice is an engineer.',
          label: 'person',
          relevanceScore: 0.95,
          depth: 0,
          tokenCount: 5,
          relationships: [
            {
              edge_label: 'works_at',
              target_node_id: 2,
              target_name: 'Acme Corp',
              direction: 'outgoing',
              weight: 1.0,
            },
          ],
        },
        {
          nodeId: 2,
          content: 'Acme Corp is a company.',
          label: 'company',
          relevanceScore: 0.72,
          depth: 1,
          tokenCount: 6,
          relationships: [],
        },
      ],
      totalTokens: 11,
      budgetUtilization: 0.22,
      nodesConsidered: 15,
      nodesIncluded: 2,
      queryTimeUs: 456,
    };
    const text = contextToPrompt(result);
    const lines = text.split('\n');
    assert.equal(lines[0], '[person] (score: 0.95)');
    assert.equal(lines[1], 'Alice is an engineer.');
    assert.equal(lines[2], '  -> works_at -> Acme Corp');
    assert.equal(lines[3], '');
    assert.equal(lines[4], '[company] (score: 0.72)');
    assert.equal(lines[5], 'Acme Corp is a company.');
    assert.equal(lines[6], '');
  });

  it('uses target_node_id as fallback when target_name is absent', () => {
    const result: ContextResult = {
      chunks: [
        {
          nodeId: 1,
          content: 'Some content.',
          label: 'entity',
          relevanceScore: 0.5,
          depth: 0,
          tokenCount: 3,
          relationships: [
            {
              edge_label: 'related_to',
              target_node_id: 99,
              direction: 'outgoing',
              weight: 0.5,
            },
          ],
        },
      ],
      totalTokens: 3,
      budgetUtilization: 0.05,
      nodesConsidered: 5,
      nodesIncluded: 1,
      queryTimeUs: 50,
    };
    const text = contextToPrompt(result);
    assert.ok(text.includes('  -> related_to -> 99'));
  });

  it('formats relevance score to two decimal places', () => {
    const result: ContextResult = {
      chunks: [
        {
          nodeId: 1,
          content: 'Content.',
          label: 'node',
          relevanceScore: 0.1,
          depth: 0,
          tokenCount: 1,
          relationships: [],
        },
      ],
      totalTokens: 1,
      budgetUtilization: 0.01,
      nodesConsidered: 1,
      nodesIncluded: 1,
      queryTimeUs: 10,
    };
    const text = contextToPrompt(result);
    assert.ok(text.includes('(score: 0.10)'));
  });

  it('formats chunk with provenance data', () => {
    const result: ContextResult = {
      chunks: [
        {
          nodeId: 1,
          content: 'Data from research paper.',
          label: 'fact',
          relevanceScore: 0.88,
          depth: 0,
          tokenCount: 5,
          provenance: { source: 'arxiv:2024.1234', confidence: 0.95 },
          relationships: [],
        },
      ],
      totalTokens: 5,
      budgetUtilization: 0.1,
      nodesConsidered: 3,
      nodesIncluded: 1,
      queryTimeUs: 80,
    };
    const text = contextToPrompt(result);
    // contextToPrompt does not render provenance, but the chunk should still format
    assert.ok(text.includes('[fact] (score: 0.88)'));
    assert.ok(text.includes('Data from research paper.'));
  });

  it('formats chunk with empty relationships array', () => {
    const result: ContextResult = {
      chunks: [
        {
          nodeId: 42,
          content: 'Isolated node.',
          label: 'orphan',
          relevanceScore: 0.33,
          depth: 2,
          tokenCount: 2,
          relationships: [],
        },
      ],
      totalTokens: 2,
      budgetUtilization: 0.04,
      nodesConsidered: 8,
      nodesIncluded: 1,
      queryTimeUs: 25,
    };
    const text = contextToPrompt(result);
    const lines = text.split('\n');
    assert.equal(lines[0], '[orphan] (score: 0.33)');
    assert.equal(lines[1], 'Isolated node.');
    assert.equal(lines[2], '');
    // No relationship lines between content and blank line
    assert.equal(lines.length, 3);
  });
});

// ── contextToMessages ──────────────────────────────────────────────────────

describe('contextToMessages', () => {
  it('returns a single user message with formatted context', () => {
    const result: ContextResult = {
      chunks: [
        {
          nodeId: 1,
          content: 'Alice is an engineer.',
          label: 'person',
          relevanceScore: 0.95,
          depth: 0,
          tokenCount: 5,
          relationships: [],
        },
        {
          nodeId: 2,
          content: 'Acme Corp is a company.',
          label: 'company',
          relevanceScore: 0.72,
          depth: 1,
          tokenCount: 6,
          relationships: [],
        },
      ],
      totalTokens: 11,
      budgetUtilization: 0.22,
      nodesConsidered: 15,
      nodesIncluded: 2,
      queryTimeUs: 456,
    };
    const messages = contextToMessages(result);
    assert.equal(messages.length, 1);
    assert.equal(messages[0].role, 'user');
    assert.ok(messages[0].content.startsWith('Context:\n'));
    assert.ok(messages[0].content.includes('[person] Alice is an engineer.'));
    assert.ok(messages[0].content.includes('[company] Acme Corp is a company.'));
  });

  it('returns a message with empty content for no chunks', () => {
    const result: ContextResult = {
      chunks: [],
      totalTokens: 0,
      budgetUtilization: 0,
      nodesConsidered: 0,
      nodesIncluded: 0,
      queryTimeUs: 0,
    };
    const messages = contextToMessages(result);
    assert.equal(messages.length, 1);
    assert.equal(messages[0].role, 'user');
    assert.equal(messages[0].content, 'Context:\n');
  });
});

// ── ContextParams ──────────────────────────────────────────────────────────

describe('ContextParams', () => {
  it('allows construction with only required graph field', () => {
    const params: ContextParams = { graph: 'my_graph' };
    assert.equal(params.graph, 'my_graph');
    assert.equal(params.query, undefined);
    assert.equal(params.embedding, undefined);
    assert.equal(params.seedNodes, undefined);
    assert.equal(params.budget, undefined);
    assert.equal(params.maxDepth, undefined);
    assert.equal(params.decay, undefined);
    assert.equal(params.edgeLabels, undefined);
    assert.equal(params.temporalAt, undefined);
    assert.equal(params.includeProvenance, undefined);
    assert.equal(params.retrievalMode, undefined);
  });

  it('allows construction with all optional fields', () => {
    const params: ContextParams = {
      graph: 'g',
      query: 'tell me about Alice',
      retrievalMode: 'hybrid',
      embedding: [0.1, 0.2, 0.3],
      seedNodes: ['node1', 'node2'],
      budget: 4096,
      maxDepth: 3,
      decay: { type: 'exponential', halfLifeMs: 3600000 },
      edgeLabels: ['knows', 'works_at'],
      temporalAt: 1704067200000,
      includeProvenance: true,
      limit: 10,
      sortField: 'relevance',
      sortDirection: 'desc',
      direction: 'both',
    };
    assert.equal(params.graph, 'g');
    assert.equal(params.query, 'tell me about Alice');
    assert.equal(params.retrievalMode, 'hybrid');
    assert.deepEqual(params.embedding, [0.1, 0.2, 0.3]);
    assert.deepEqual(params.seedNodes, ['node1', 'node2']);
    assert.equal(params.budget, 4096);
    assert.equal(params.maxDepth, 3);
    assert.deepEqual(params.decay, { type: 'exponential', halfLifeMs: 3600000 });
    assert.deepEqual(params.edgeLabels, ['knows', 'works_at']);
    assert.equal(params.temporalAt, 1704067200000);
    assert.equal(params.includeProvenance, true);
    assert.equal(params.limit, 10);
    assert.equal(params.sortField, 'relevance');
    assert.equal(params.sortDirection, 'desc');
    assert.equal(params.direction, 'both');
  });
});

// ── GraphInfo ──────────────────────────────────────────────────────────────

describe('GraphInfo', () => {
  it('can be constructed with all fields', () => {
    const info: GraphInfo = {
      name: 'knowledge_base',
      node_count: 1500,
      edge_count: 4200,
    };
    assert.equal(info.name, 'knowledge_base');
    assert.equal(info.node_count, 1500);
    assert.equal(info.edge_count, 4200);
  });

  it('supports zero counts', () => {
    const info: GraphInfo = {
      name: 'empty_graph',
      node_count: 0,
      edge_count: 0,
    };
    assert.equal(info.node_count, 0);
    assert.equal(info.edge_count, 0);
  });
});

// ── NodeInfo ───────────────────────────────────────────────────────────────

describe('NodeInfo', () => {
  it('can be constructed with properties', () => {
    const node: NodeInfo = {
      node_id: 42,
      label: 'person',
      properties: { name: 'Alice', age: 30 },
    };
    assert.equal(node.node_id, 42);
    assert.equal(node.label, 'person');
    assert.equal(node.properties['name'], 'Alice');
    assert.equal(node.properties['age'], 30);
  });

  it('supports empty properties', () => {
    const node: NodeInfo = {
      node_id: 1,
      label: 'empty',
      properties: {},
    };
    assert.deepEqual(node.properties, {});
  });
});

// ── AddNodeParams ──────────────────────────────────────────────────────────

describe('AddNodeParams', () => {
  it('can be constructed with only required label', () => {
    const params: AddNodeParams = { label: 'concept' };
    assert.equal(params.label, 'concept');
    assert.equal(params.properties, undefined);
    assert.equal(params.embedding, undefined);
    assert.equal(params.entityKey, undefined);
  });

  it('can be constructed with all optional fields', () => {
    const params: AddNodeParams = {
      label: 'person',
      properties: { name: 'Bob', role: 'engineer' },
      embedding: [0.5, 0.5, 0.5],
      entityKey: 'bob-001',
    };
    assert.equal(params.label, 'person');
    assert.deepEqual(params.properties, { name: 'Bob', role: 'engineer' });
    assert.deepEqual(params.embedding, [0.5, 0.5, 0.5]);
    assert.equal(params.entityKey, 'bob-001');
  });
});

// ── AddEdgeParams ──────────────────────────────────────────────────────────

describe('AddEdgeParams', () => {
  it('can be constructed with only required fields', () => {
    const params: AddEdgeParams = {
      source: 1,
      target: 2,
      label: 'knows',
    };
    assert.equal(params.source, 1);
    assert.equal(params.target, 2);
    assert.equal(params.label, 'knows');
    assert.equal(params.weight, undefined);
    assert.equal(params.provenance, undefined);
  });

  it('can be constructed with weight and provenance', () => {
    const params: AddEdgeParams = {
      source: 10,
      target: 20,
      label: 'works_at',
      weight: 0.9,
      provenance: {
        source: 'linkedin',
        confidence: 0.85,
        extraction_method: 'scrape',
        source_document_id: 'doc-42',
      },
    };
    assert.equal(params.weight, 0.9);
    assert.equal(params.provenance?.source, 'linkedin');
    assert.equal(params.provenance?.confidence, 0.85);
    assert.equal(params.provenance?.extraction_method, 'scrape');
    assert.equal(params.provenance?.source_document_id, 'doc-42');
  });
});
