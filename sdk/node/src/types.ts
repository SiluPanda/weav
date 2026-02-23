export interface WeavConfig {
  host: string;
  port: number;
}

export interface GraphInfo {
  name: string;
  node_count: number;
  edge_count: number;
}

export interface NodeInfo {
  node_id: number;
  label: string;
  properties: Record<string, unknown>;
}

export interface Provenance {
  source: string;
  confidence: number;
  extraction_method?: string;
  source_document_id?: string;
}

export interface RelationshipSummary {
  edge_label: string;
  target_node_id: number;
  target_name?: string;
  direction: string;
  weight: number;
}

export interface ContextChunk {
  nodeId: number;
  content: string;
  label: string;
  relevanceScore: number;
  depth: number;
  tokenCount: number;
  provenance?: { source: string; confidence: number };
  relationships: Array<{
    edge_label: string;
    target_node_id: number;
    target_name?: string;
    direction: string;
    weight: number;
  }>;
}

export interface ContextResult {
  chunks: ContextChunk[];
  totalTokens: number;
  budgetUtilization: number;
  nodesConsidered: number;
  nodesIncluded: number;
  queryTimeUs: number;
}

export interface DecayParams {
  type: 'exponential' | 'linear' | 'step' | 'none';
  halfLifeMs?: number;
  maxAgeMs?: number;
  cutoffMs?: number;
}

export interface ContextParams {
  graph: string;
  query?: string;
  embedding?: number[];
  seedNodes?: string[];
  budget?: number;
  maxDepth?: number;
  decay?: DecayParams;
  edgeLabels?: string[];
  temporalAt?: number;
  includeProvenance?: boolean;
  limit?: number;
  sortField?: 'relevance' | 'recency' | 'confidence';
  sortDirection?: 'asc' | 'desc';
  direction?: 'outgoing' | 'incoming' | 'both';
}

export interface AddNodeParams {
  label: string;
  properties?: Record<string, unknown>;
  embedding?: number[];
  entityKey?: string;
}

export interface AddEdgeParams {
  source: number;
  target: number;
  label: string;
  weight?: number;
  provenance?: Provenance;
}

export interface UpdateNodeParams {
  properties?: Record<string, unknown>;
  embedding?: number[];
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}
