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
    glomeruli_count?: number;
  };
  fibrosisCompleted: boolean;
  lengthCompleted: boolean;
  glomerulesCompleted: boolean;
  imageVersions: Array<{
    id: 'original' | 'fibrosis' | 'length' | 'glomeruli';
    label: string;
    url: string;
  }>;
}