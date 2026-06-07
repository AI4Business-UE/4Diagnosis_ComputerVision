const API_BASE_URL = 'http://localhost:8000/api';

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface FibrosisResponse {
  job_id?: string;
  fibrosis_ratio?: number;
  fibrotic_pixels?: number;
  tissue_pixels?: number;
  image_path?: string;
  error?: string | null;
}

export interface LengthResponse {
  job_id?: string;
  length?: number;
  image_path?: string;
  error?: string | null;
}

export interface GlomeruliResponse {
  job_id?: string;
  count?: number;
  image_url?: string;
  error?: string | null;
}

export interface Glomerulus {
  x1: number; y1: number;
  x2: number; y2: number;
  cls: number;
  cls_name: string;
  conf: number;
}


async function readErrorMessage(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    if (typeof payload?.error === 'string' && payload.error.trim().length > 0) {
      return payload.error;
    }
    if (typeof payload?.message === 'string' && payload.message.trim().length > 0) {
      return payload.message;
    }
  } catch {
    // Ignore JSON parse issues and fall back to generic HTTP status.
  }

  return `HTTP ${response.status}`;
}

/**
 * Wysyła wybrany folder do backendu
 */
export async function selectFolder(folderName: string): Promise<ApiResponse<{ message: string }>> {
  try {
    const response = await fetch(`${API_BASE_URL}/select-folder`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ folderName }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Nieznany błąd';
    return { success: false, error: errorMessage };
  }
}

/**
 * Konwertuje obrazy na format TIFF
 */
export async function convertToTiff(files: File[]): Promise<ApiResponse<{ status: string; job_id: string; tiff: string; tiff_url: string; mask_preview_url?: string }>> {
  try {
    const formData = new FormData();

    for (const file of files) {
      const lower = file.name.toLowerCase();
      if (
        lower.endsWith('.mrxs') ||
        lower.endsWith('.dat') ||
        lower === 'slidedat.ini'
      ) {
        formData.append('files', file, file.name);
      }
    }

    const response = await fetch(`${API_BASE_URL}/convert/`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }

    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Nieznany błąd';
    return { success: false, error: errorMessage };
  }
}

/**
 * Analizuje włóknienia (fibrosis)
 */
export async function analyzeFibrosis(jobId: string): Promise<ApiResponse<FibrosisResponse>> {
    try {
      const response = await fetch(`${API_BASE_URL}/fibrosis/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: jobId }),
      });

      if (!response.ok) {
        throw new Error(await readErrorMessage(response));
      }

      const data = await response.json();
      return { success: true, data };

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Nieznany błąd';
      return { success: false, error: errorMessage };
    }
}

/**
 * Analizuje długość struktur
 */
export async function analyzeLength(jobId: string): Promise<ApiResponse<LengthResponse>> {
  try {
    const response = await fetch(`${API_BASE_URL}/length/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: jobId }),
    });

    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }

    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Nieznany błąd';
    return { success: false, error: errorMessage };
  }
}

export interface SlideInfo {
  w: number;
  h: number;
  /** Próg ufności modelu (conf w GlomeruliProcessor) — minimum suwaka na frontendzie. */
  conf: number;
}

export interface TileInfo {
  x: number;
  y: number;
  w: number;
  h: number;
  tissue: boolean;
}

/**
 * Streaming detekcji kłębuszków przez SSE — zwraca funkcję cleanup (zamknięcie połączenia)
 */
export function streamGlomeruli(
  jobId: string,
  onSlideInfo: (info: SlideInfo) => void,
  onBatch: (newBatch: Glomerulus[], totalSoFar: number) => void,
  onDone: (total: number) => void,
  onError: (msg: string) => void,
  onTile?: (tiles: TileInfo[]) => void,
  onFinalList?: (list: Glomerulus[]) => void,
): () => void {
  let total = 0;
  const es = new EventSource(`${API_BASE_URL}/glomeruli/stream/?job_id=${encodeURIComponent(jobId)}`);

  es.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.error) {
      onError(data.error);
      es.close();
    } else if (data.slide_info) {
      onSlideInfo(data.slide_info as SlideInfo);
    } else if (data.done) {
      if (data.final_glomeruli && onFinalList) {
        onFinalList(data.final_glomeruli as Glomerulus[]);
      }
      onDone(data.count ?? total);
      es.close();
    } else if (data.glomeruli?.length) {
      total += data.glomeruli.length;
      onBatch(data.glomeruli, total);
    } else if (data.tiles?.length && onTile) {
      onTile(data.tiles as TileInfo[]);
    }
  };

  es.onerror = () => {
    onError('Błąd połączenia ze strumieniem detekcji');
    es.close();
  };

  return () => es.close();
}

/**
 * Wykrywa kłębuszki (glomerule)
 */
export async function detectGlomerules(jobId: string): Promise<ApiResponse<GlomeruliResponse>> {
  try {
    const response = await fetch(`${API_BASE_URL}/glomeruli/count/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ job_id: jobId }),
    });

    const data = await response.json();

    if (!response.ok) {
      return {
        success: false,
        error: data?.error ?? `HTTP ${response.status}`,
      };
    }

    return {
      success: true,
      data: {
        job_id: data.job_id,
        count: data.count,
        image_url: data.image_url,
        error: data.error,
      },
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Nieznany błąd';
    return { success: false, error: errorMessage };
  }
}
