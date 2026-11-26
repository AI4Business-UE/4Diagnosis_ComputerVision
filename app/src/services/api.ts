import type { ConversionPreset } from '../types/Settings';
import { formatApiError } from './errors';

const API_BASE_URL = 'http://localhost:8000/api';

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface SlicesSummary {
  n_slices: number;
  representative_slice_id: number;
  grouping_enabled: boolean;
  is_grouped: boolean;
  fallback_used: boolean;
  fallback_prob: number;
  user_ungrouped: boolean;
  can_toggle_grouping: boolean;
}

export interface ConvertOptions {
  conversionPreset?: ConversionPreset;
  enableSliceGrouping?: boolean;
  cleaningLevel?: number;
}

export interface ConvertResponse {
  status: string;
  job_id: string;
  tiff: string;
  tiff_url: string;
  cleaned_url?: string | null;
  origin_detect_url?: string | null;
  conversion_level?: number;
  slices_summary?: SlicesSummary;
}

export interface ToggleGroupingResponse {
  status: string;
  job_id: string;
  cleaned_url: string;
  origin_detect_url: string;
  slices_summary: SlicesSummary;
}

export interface FibrosisResponse {
  job_id?: string;
  fibrosis_ratio?: number;
  fibrosis_ratio_avg?: number;
  fibrosis_ratio_per_slice?: number[];
  fibrosis_warning?: boolean;
  fibrotic_pixels?: number;
  tissue_pixels?: number;
  image_path?: string;
  /** B-channel threshold (0-1) actually used for this analysis. */
  threshold?: number;
  error?: string | null;
}

export interface LengthResponse {
  job_id?: string;
  length?: number;
  image_path?: string;
  error?: string | null;
}

export interface Glomeruli {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  cls: number;
  cls_name: string;
  conf: number;
}

export interface SlideInfo {
  w: number;
  h: number;
  conf: number;
  n_slices?: number;
}

export interface TileInfo {
  x: number;
  y: number;
  w: number;
  h: number;
  tissue: boolean;
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
    const response = await fetch(`${API_BASE_URL}/select-folder/`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ folderName }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    return { success: false, error: formatApiError(error) };
  }
}

export function isAbortError(error: unknown): boolean {
  return (
    (error instanceof DOMException && error.name === 'AbortError') ||
    (error instanceof Error && error.name === 'AbortError')
  );
}

/**
 * Konwertuje obrazy na format TIFF
 */
export async function convertToTiff(
  files: File[],
  options?: ConvertOptions,
  signal?: AbortSignal
): Promise<ApiResponse<ConvertResponse>> {
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

    if (options?.conversionPreset) {
      formData.append('conversion_preset', options.conversionPreset);
    }
    if (typeof options?.enableSliceGrouping === 'boolean') {
      formData.append('enable_slice_grouping', String(options.enableSliceGrouping));
    }
    if (typeof options?.cleaningLevel === 'number') {
      formData.append('cleaning_level', String(options.cleaningLevel));
    }

    const response = await fetch(`${API_BASE_URL}/convert/`, {
      method: 'POST',
      body: formData,
      signal,
    });

    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }

    const data = (await response.json()) as ConvertResponse;
    return { success: true, data };
  } catch (error) {
    if (isAbortError(error)) throw error;
    return { success: false, error: formatApiError(error, 'Błąd konwersji') };
  }
}

export async function setSlideGrouping(
  jobId: string,
  grouped: boolean
): Promise<ApiResponse<ToggleGroupingResponse>> {
  try {
    const response = await fetch(`${API_BASE_URL}/slides/${encodeURIComponent(jobId)}/grouping/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ grouped }),
    });

    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }

    const data = (await response.json()) as ToggleGroupingResponse;
    return { success: true, data };
  } catch (error) {
    return { success: false, error: formatApiError(error, 'Błąd zmiany grupowania') };
  }
}

/**
 * Analizuje włóknienia (fibrosis)
 *
 * @param threshold - opcjonalny próg klasyfikacji (kanał B, 0-1). Gdy pominięty,
 * backend używa settings.FIBROSIS_THRESHOLD.
 */
export async function analyzeFibrosis(
  jobId: string,
  threshold?: number,
  signal?: AbortSignal
): Promise<ApiResponse<FibrosisResponse>> {
  try {
    const response = await fetch(`${API_BASE_URL}/fibrosis/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: jobId, ...(threshold != null ? { threshold } : {}) }),
      signal,
    });

    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }

    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    if (isAbortError(error)) throw error;
    return { success: false, error: formatApiError(error, 'Błąd analizy zwłóknienia') };
  }
}

/**
 * Analizuje długość struktur
 */
export async function analyzeLength(
  jobId: string,
  signal?: AbortSignal
): Promise<ApiResponse<LengthResponse>> {
  try {
    const response = await fetch(`${API_BASE_URL}/length/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: jobId }),
      signal,
    });

    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }

    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    if (isAbortError(error)) throw error;
    return { success: false, error: formatApiError(error, 'Błąd analizy długości') };
  }
}
