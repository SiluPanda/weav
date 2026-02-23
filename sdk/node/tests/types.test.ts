import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

import { WeavClient, WeavError, contextToPrompt } from '../src/index.js';
import type { ContextResult } from '../src/index.js';

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
          node_id: 1,
          content: 'Alice is an engineer.',
          label: 'person',
          relevance_score: 0.95,
          depth: 0,
          token_count: 5,
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
          node_id: 1,
          content: 'Alice is an engineer.',
          label: 'person',
          relevance_score: 0.95,
          depth: 0,
          token_count: 5,
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
          node_id: 2,
          content: 'Acme Corp is a company.',
          label: 'company',
          relevance_score: 0.72,
          depth: 1,
          token_count: 6,
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
          node_id: 1,
          content: 'Some content.',
          label: 'entity',
          relevance_score: 0.5,
          depth: 0,
          token_count: 3,
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
          node_id: 1,
          content: 'Content.',
          label: 'node',
          relevance_score: 0.1,
          depth: 0,
          token_count: 1,
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
});
