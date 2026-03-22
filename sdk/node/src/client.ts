import {
  WeavConfig,
  ContextParams,
  ContextResult,
  GraphInfo,
  NodeInfo,
  AddNodeParams,
  AddEdgeParams,
  UpdateNodeParams,
  IngestParams,
  IngestResult,
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
  private authHeader?: string;

  constructor(config?: Partial<WeavConfig>) {
    const host = config?.host ?? 'localhost';
    const port = config?.port ?? 6382;
    this.baseUrl = `http://${host}:${port}`;

    if (config?.apiKey) {
      this.authHeader = `Bearer ${config.apiKey}`;
    } else if (config?.username && config?.password) {
      const credentials = btoa(`${config.username}:${config.password}`);
      this.authHeader = `Basic ${credentials}`;
    }
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
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.authHeader) {
      headers['Authorization'] = this.authHeader;
    }
    const options: RequestInit = {
      method,
      headers,
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
    const result = await this.request<{ node_id: number }>(
      'POST',
      `/v1/graphs/${encodeURIComponent(graph)}/nodes`,
      body,
    );
    return result.node_id;
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
    const result = await this.request<{ edge_id: number }>(
      'POST',
      `/v1/graphs/${encodeURIComponent(graph)}/edges`,
      body,
    );
    return result.edge_id;
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
    const result = await this.request<{ node_ids: number[] }>(
      'POST',
      `/v1/graphs/${encodeURIComponent(graph)}/nodes/bulk`,
      body,
    );
    return result.node_ids;
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
    const result = await this.request<{ edge_ids: number[] }>(
      'POST',
      `/v1/graphs/${encodeURIComponent(graph)}/edges/bulk`,
      body,
    );
    return result.edge_ids;
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
      body.decay = {
        type: params.decay.type,
        half_life_ms: params.decay.halfLifeMs,
        max_age_ms: params.decay.maxAgeMs,
        cutoff_ms: params.decay.cutoffMs,
      };
    }
    if (params.edgeLabels !== undefined) {
      body.edge_labels = params.edgeLabels;
    }
    if (params.temporalAt !== undefined) {
      body.temporal_at = params.temporalAt;
    }
    if (params.limit !== undefined) {
      body.limit = params.limit;
    }
    if (params.sortField !== undefined) {
      body.sort_field = params.sortField;
    }
    if (params.sortDirection !== undefined) {
      body.sort_direction = params.sortDirection;
    }
    if (params.direction !== undefined) {
      body.direction = params.direction;
    }
    const raw = await this.request<Record<string, unknown>>(
      'POST',
      '/v1/context',
      body,
    );
    return this.parseContextResult(raw);
  }

  // ── Search ───────────────────────────────────────────────────────────

  async searchText(
    graph: string,
    query: string,
    limit: number = 20,
  ): Promise<Record<string, unknown>> {
    const params = new URLSearchParams({ q: query, limit: String(limit) });
    return this.request<Record<string, unknown>>(
      'GET',
      `/v1/graphs/${encodeURIComponent(graph)}/search/text?${params}`,
    );
  }

  // ── Node merge ──────────────────────────────────────────────────────

  async mergeNodes(
    graph: string,
    sourceId: number,
    targetId: number,
    conflictPolicy: string = 'keep_target',
  ): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>(
      'POST',
      `/v1/graphs/${encodeURIComponent(graph)}/nodes/merge`,
      {
        source_id: sourceId,
        target_id: targetId,
        conflict_policy: conflictPolicy,
      },
    );
  }

  // ── Vector search ───────────────────────────────────────────────────

  async searchVector(
    graph: string,
    embedding: number[],
    options?: {
      k?: number;
      labels?: string[];
      properties?: Record<string, unknown>;
    },
  ): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = { embedding };
    if (options?.k !== undefined) {
      body.k = options.k;
    }
    if (options?.labels !== undefined) {
      body.labels = options.labels;
    }
    if (options?.properties !== undefined) {
      body.properties = options.properties;
    }
    return this.request<Record<string, unknown>>(
      'POST',
      `/v1/graphs/${encodeURIComponent(graph)}/search/vector`,
      body,
    );
  }

  // ── Graph diff ─────────────────────────────────────────────────────

  async graphDiff(
    graph: string,
    fromTimestamp: number,
    toTimestamp: number,
  ): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>(
      'POST',
      `/v1/graphs/${encodeURIComponent(graph)}/diff`,
      { from_timestamp: fromTimestamp, to_timestamp: toTimestamp },
    );
  }

  // ── Community operations ───────────────────────────────────────────

  async communitySummarize(
    graph: string,
    options?: {
      algorithm?: string;
      resolution?: number;
    },
  ): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>(
      'POST',
      `/v1/graphs/${encodeURIComponent(graph)}/communities/summarize`,
      {
        algorithm: options?.algorithm ?? 'leiden',
        resolution: options?.resolution ?? 1.0,
      },
    );
  }

  async communitySummaries(graph: string): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>(
      'GET',
      `/v1/graphs/${encodeURIComponent(graph)}/communities/summaries`,
    );
  }

  async communitySearch(
    graph: string,
    query: string,
    limit?: number,
  ): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>(
      'POST',
      `/v1/graphs/${encodeURIComponent(graph)}/communities/search`,
      { query, limit: limit ?? 10 },
    );
  }

  // ── Algorithms ──────────────────────────────────────────────────────

  async runAlgorithm(
    graph: string,
    algorithm: string,
    params?: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>(
      'POST',
      `/v1/graphs/${encodeURIComponent(graph)}/algorithms/${encodeURIComponent(algorithm)}`,
      params ?? {},
    );
  }

  // ── CSV import/export ───────────────────────────────────────────────

  async importCsv(
    graph: string,
    csvContent: string,
  ): Promise<Record<string, unknown>> {
    const url = `${this.baseUrl}/v1/graphs/${encodeURIComponent(graph)}/import/csv`;
    const headers: Record<string, string> = { 'Content-Type': 'text/csv' };
    if (this.authHeader) {
      headers['Authorization'] = this.authHeader;
    }
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: csvContent,
    });
    const json = (await response.json()) as ApiResponse<Record<string, unknown>>;
    if (!json.success) {
      throw new WeavError(json.error ?? 'Unknown error', response.status);
    }
    return json.data as Record<string, unknown>;
  }

  async exportCsv(graph: string): Promise<string> {
    const url = `${this.baseUrl}/v1/graphs/${encodeURIComponent(graph)}/export/csv`;
    const headers: Record<string, string> = {};
    if (this.authHeader) {
      headers['Authorization'] = this.authHeader;
    }
    const response = await fetch(url, { method: 'GET', headers });
    if (!response.ok) {
      throw new WeavError(`Export failed: ${response.statusText}`, response.status);
    }
    return response.text();
  }

  // ── Ingest (extraction pipeline) ─────────────────────────────────────

  async ingest(graph: string, params: IngestParams): Promise<IngestResult> {
    const body: Record<string, unknown> = {};
    if (params.content !== undefined) {
      body.content = params.content;
    }
    if (params.contentBase64 !== undefined) {
      body.content_base64 = params.contentBase64;
    }
    if (params.format !== undefined) {
      body.format = params.format;
    }
    if (params.documentId !== undefined) {
      body.document_id = params.documentId;
    }
    body.skip_extraction = params.skipExtraction ?? false;
    body.skip_dedup = params.skipDedup ?? false;
    if (params.chunkSize !== undefined) {
      body.chunk_size = params.chunkSize;
    }
    if (params.entityTypes !== undefined) {
      body.entity_types = params.entityTypes;
    }

    const raw = await this.request<Record<string, unknown>>(
      'POST',
      `/v1/graphs/${encodeURIComponent(graph)}/ingest`,
      body,
    );
    return {
      documentId: raw.document_id as string,
      chunksCreated: raw.chunks_created as number,
      entitiesCreated: raw.entities_created as number,
      entitiesMerged: raw.entities_merged as number,
      relationshipsCreated: raw.relationships_created as number,
      pipelineDurationMs: raw.pipeline_duration_ms as number,
    };
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
