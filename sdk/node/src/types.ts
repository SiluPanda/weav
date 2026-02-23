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
  node_id: number;
  content: string;
  label: string;
  relevance_score: number;
  depth: number;
  token_count: number;
  provenance?: Provenance;
  relationships: RelationshipSummary[];
}

export interface ContextResult {
  chunks: ContextChunk[];
  totalTokens: number;
  budgetUtilization: number;
  nodesConsidered: number;
  nodesIncluded: number;
  queryTimeUs: number;
}

export interface ContextParams {
  graph: string;
  query?: string;
  embedding?: number[];
  seedNodes?: string[];
  budget?: number;
  maxDepth?: number;
  includeProvenance?: boolean;
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
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}
