import type { Glomeruli, SlideInfo, SlicesSummary, TileInfo } from '../services/api';

export type ImageVersionId = 'raw' | 'cleaned' | 'merged' | 'fibrosis' | 'length' | 'glomeruli';

export interface ImageVersion {
  id: ImageVersionId;
  label: string;
  url: string;
}

export interface Sample {
  id: string;
  name: string;
  folderPath: string;
  files: File[];
  jobId: string | null;
  processStage: 'initial' | 'folder_selected' | 'converted';
  slicesSummary?: SlicesSummary;
  analysisResult: {
    length?: number;
    fibrosis_ratio?: number;
    fibrosis_ratio_avg?: number;
    fibrosis_warning?: boolean;
    /** B-channel threshold (0-1) used for the current fibrosis_ratio. */
    fibrosis_threshold?: number;
    glomeruli_count?: number;
  };
  fibrosisCompleted: boolean;
  lengthCompleted: boolean;
  glomeruliCompleted: boolean;
  imageVersions: ImageVersion[];
  glomeruli?: Glomeruli[];
  glomeruliSlideInfo?: SlideInfo;
  glomeruliTiles?: TileInfo[];
}
