import type { Glomeruli, SlideInfo, TileInfo } from '../services/api';

export interface Sample {
  id: string;
  name: string;
  folderPath: string;
  files: File[];
  jobId: string | null;
  processStage: 'initial' | 'folder_selected' | 'converted';
  analysisResult: {
    length?: number;
    fibrosis_ratio?: number;
    fibrosis_ratio_avg?: number;
    fibrosis_warning?: boolean;
    glomeruli_count?: number;
  };
  fibrosisCompleted: boolean;
  lengthCompleted: boolean;
  glomerulesCompleted: boolean;
  imageVersions: Array<{
    id: 'original' | 'fibrosis' | 'length' | 'glomeruli' | 'glom_grid';
    label: string;
    url: string;
  }>;
  glomeruli?: Glomeruli[];
  glomeruliSlideInfo?: SlideInfo;
  glomeruliTiles?: TileInfo[];
}
