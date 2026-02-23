import {
  WeavConfig,
  ContextParams,
  ContextResult,
  GraphInfo,
  NodeInfo,
  AddNodeParams,
  AddEdgeParams,
  UpdateNodeParams,
  ApiResponse,
  ContextChunk,
} from './types.js';

export class WeavError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
  ) {
    super(message);
    this.name = 'WeavError';
  }
}

export class WeavClient {
  private baseUrl: string;

  constructor(config?: Partial<WeavConfig>) {
    const host = config?.host ?? 'localhost';
    const port = config?.port ?? 6382;
    this.baseUrl = `http://${host}:${port}`;
  }

  /** Visible for testing. */
  getBaseUrl(): string {
    return this.baseUrl;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const options: RequestInit = {
      method,
      headers: { 'Content-Type': 'application/json' },
    };
    if (body !== undefined) {
      options.body = JSON.stringify(body);
    }
    const response = await fetch(url, options);
    const json = (await response.json()) as ApiResponse<T>;
    if (!json.success) {
      throw new WeavError(json.error ?? 'Unknown error', response.status);
    }
    return json.data as T;
  }

  // ── Health ──────────────────────────────────────────────────────────────

  async ping(): Promise<boolean> {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.ok;
  }

  async info(): Promise<Record<string, unknown>> {
    const response = await this.request<Record<string, unknown>>('GET', '/health');
    return response;
  }

  // ── Graph management ──────────────────────────────────────────────────

  async createGraph(name: string): Promise<void> {
    await this.request('POST', '/v1/graphs', { name });
  }

  async dropGraph(name: string): Promise<void> {
    await this.request('DELETE', `/v1/graphs/${encodeURIComponent(name)}`);
  }

  async listGraphs(): Promise<string[]> {
    return this.request<string[]>('GET', '/v1/graphs');
  }

  async graphInfo(name: string): Promise<GraphInfo> {
    return this.request<GraphInfo>(
      'GET',
      `/v1/graphs/${encodeURIComponent(name)}`,
    );
  }

  // ── Node operations ───────────────────────────────────────────────────

  async addNode(graph: string, params: AddNodeParams): Promise<number> {
    const body = {
      label: params.label,
      properties: params.properties,
      embedding: params.embedding,
      entity_key: params.entityKey,
    };
    const result = await this.request<{ id: number }>(
      'POST',
      `/v1/graphs/${encodeURIComponent(graph)}/nodes`,
      body,
    );
    return result.id;
  }

  async getNode(graph: string, nodeId: number): Promise<NodeInfo> {
    return this.request<NodeInfo>(
      'GET',
      `/v1/graphs/${encodeURIComponent(graph)}/nodes/${nodeId}`,
    );
  }

  async deleteNode(graph: string, nodeId: number): Promise<void> {
    await this.request(
      'DELETE',
      `/v1/graphs/${encodeURIComponent(graph)}/nodes/${nodeId}`,
    );
  }

  async updateNode(graph: string, nodeId: number, params: UpdateNodeParams): Promise<void> {
    await this.request(
      'PUT',
      `/v1/graphs/${encodeURIComponent(graph)}/nodes/${nodeId}`,
      params,
    );
  }

  // ── Edge operations ───────────────────────────────────────────────────

  async addEdge(graph: string, params: AddEdgeParams): Promise<number> {
    const body: Record<string, unknown> = {
      source: params.source,
      target: params.target,
      label: params.label,
      weight: params.weight ?? 1.0,
    };
    if (params.provenance !== undefined) {
      body.provenance = params.provenance;
    }
    const result = await this.request<{ id: number }>(
      'POST',
      `/v1/graphs/${encodeURIComponent(graph)}/edges`,
      body,
    );
    return result.id;
  }

  async invalidateEdge(graph: string, edgeId: number): Promise<void> {
    await this.request(
      'POST',
      `/v1/graphs/${encodeURIComponent(graph)}/edges/${edgeId}/invalidate`,
    );
  }

  async bulkAddNodes(graph: string, nodes: AddNodeParams[]): Promise<number[]> {
    const body = {
      nodes: nodes.map(n => ({
        label: n.label,
        properties: n.properties,
        embedding: n.embedding,
        entity_key: n.entityKey,
      })),
    };
    const result = await this.request<{ ids: number[] }>(
      'POST',
      `/v1/graphs/${encodeURIComponent(graph)}/nodes/bulk`,
      body,
    );
    return result.ids;
  }

  async bulkAddEdges(graph: string, edges: AddEdgeParams[]): Promise<number[]> {
    const body = {
      edges: edges.map(e => {
        const edge: Record<string, unknown> = {
          source: e.source,
          target: e.target,
          label: e.label,
          weight: e.weight ?? 1.0,
        };
        if (e.provenance !== undefined) {
          edge.provenance = e.provenance;
        }
        return edge;
      }),
    };
    const result = await this.request<{ ids: number[] }>(
      'POST',
      `/v1/graphs/${encodeURIComponent(graph)}/edges/bulk`,
      body,
    );
    return result.ids;
  }

  // ── Context retrieval ─────────────────────────────────────────────────

  async context(params: ContextParams): Promise<ContextResult> {
    const body: Record<string, unknown> = {
      graph: params.graph,
      query: params.query,
      embedding: params.embedding,
      seed_nodes: params.seedNodes,
      budget: params.budget ?? 4096,
      max_depth: params.maxDepth ?? 3,
      include_provenance: params.includeProvenance ?? false,
    };
    if (params.decay !== undefined) {
      body.decay = params.decay;
    }
    if (params.edgeLabels !== undefined) {
      body.edge_labels = params.edgeLabels;
    }
    if (params.temporalAt !== undefined) {
      body.temporal_at = params.temporalAt;
    }
    const raw = await this.request<Record<string, unknown>>(
      'POST',
      '/v1/context',
      body,
    );
    return this.parseContextResult(raw);
  }

  private parseContextResult(raw: Record<string, unknown>): ContextResult {
    return {
      chunks: (
        (raw.chunks as Array<Record<string, unknown>>) ?? []
      ).map((c) => this.parseChunk(c)),
      totalTokens: (raw.total_tokens as number) ?? 0,
      budgetUtilization: (raw.budget_used as number) ?? 0,
      nodesConsidered: (raw.nodes_considered as number) ?? 0,
      nodesIncluded: (raw.nodes_included as number) ?? 0,
      queryTimeUs: (raw.query_time_us as number) ?? 0,
    };
  }

  private parseChunk(raw: Record<string, unknown>): ContextChunk {
    return {
      nodeId: raw.node_id as number,
      content: (raw.content as string) ?? '',
      label: (raw.label as string) ?? '',
      relevanceScore: (raw.relevance_score as number) ?? 0,
      depth: (raw.depth as number) ?? 0,
      tokenCount: (raw.token_count as number) ?? 0,
      provenance: raw.provenance as ContextChunk['provenance'],
      relationships:
        (raw.relationships as ContextChunk['relationships']) ?? [],
    };
  }
}

// ── Helper: convert context result to chat messages ───────────────────────

export function contextToMessages(
  result: ContextResult,
): Array<{ role: string; content: string }> {
  const content = result.chunks
    .map((chunk: ContextChunk) => `[${chunk.label}] ${chunk.content}`)
    .join('\n\n');
  return [{ role: 'user', content: `Context:\n${content}` }];
}

// ── Helper: format context result as prompt text ──────────────────────────

export function contextToPrompt(result: ContextResult): string {
  const lines: string[] = [];
  for (const chunk of result.chunks) {
    lines.push(`[${chunk.label}] (score: ${chunk.relevanceScore.toFixed(2)})`);
    lines.push(chunk.content);
    for (const rel of chunk.relationships) {
      lines.push(
        `  -> ${rel.edge_label} -> ${rel.target_name ?? rel.target_node_id}`,
      );
    }
    lines.push('');
  }
  return lines.join('\n');
}
