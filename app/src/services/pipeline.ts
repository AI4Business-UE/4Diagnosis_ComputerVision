import {
    convertToTiff,
    setSlideGrouping,
    analyzeFibrosis,
    analyzeLength,
    isAbortError,
} from './api';
import { formatApiError } from './errors';
export { isAbortError, formatApiError };
import type {
    ConvertOptions,
    FibrosisResponse,
    Glomeruli,
    LengthResponse,
    SlideInfo,
    SlicesSummary,
    TileInfo,
} from './api';
import type { ImageVersion } from '../types/Sample';

export const API_ORIGIN = 'http://localhost:8000';

/**
 * Single-sample pipeline steps shared by the per-step buttons in ControlPanel
 * and the batch runner in App. Each step throws on failure so callers can
 * decide whether to abort the sample or continue.
 */

export function toResultImageUrl(imagePath: string, jobId: string | null): string | null {
    if (!jobId) return null;
    const fileName = imagePath.split(/[/\\]/).pop();
    if (!fileName) return null;
    return `${API_ORIGIN}/api/result-image/${encodeURIComponent(jobId)}/${encodeURIComponent(fileName)}/`;
}

export function buildBaseImageVersions(params: {
    rawUrl: string | null;
    cleanedUrl: string | null;
    mergedUrl: string | null;
    isGrouped: boolean;
}): ImageVersion[] {
    const versions: ImageVersion[] = [];
    if (params.rawUrl) {
        versions.push({ id: 'raw', label: 'Surowy skan', url: params.rawUrl });
    }
    if (params.cleanedUrl) {
        versions.push({ id: 'cleaned', label: 'Wyczyszczony', url: params.cleanedUrl });
    }
    if (params.isGrouped && params.mergedUrl) {
        versions.push({ id: 'merged', label: 'Zgrupowany', url: params.mergedUrl });
    }
    return versions;
}

export function getActiveBaseImageUrl(versions: ImageVersion[]): string | null {
    return (
        versions.find(v => v.id === 'merged')?.url ??
        versions.find(v => v.id === 'cleaned')?.url ??
        versions.find(v => v.id === 'raw')?.url ??
        null
    );
}

export interface ConvertStepResult {
    jobId: string;
    rawUrl: string | null;
    cleanedUrl: string | null;
    mergedUrl: string | null;
    activeBaseUrl: string | null;
    imageVersions: ImageVersion[];
    slicesSummary?: SlicesSummary;
    data: Record<string, unknown>;
}

export async function runConvert(
    files: File[],
    options?: ConvertOptions,
    signal?: AbortSignal
): Promise<ConvertStepResult> {
    const result = await convertToTiff(files, options, signal);

    if (!result.success || !result.data) {
        throw new Error(result.error || 'Nieznany błąd konwersji');
    }
    if (!result.data.job_id) {
        throw new Error('Backend nie zwrócił job_id');
    }

    const rawUrl = result.data.tiff_url ? `${API_ORIGIN}${result.data.tiff_url}` : null;
    const cleanedUrl = result.data.cleaned_url ? `${API_ORIGIN}${result.data.cleaned_url}` : null;
    const mergedUrl = result.data.origin_detect_url ? `${API_ORIGIN}${result.data.origin_detect_url}` : null;
    const slicesSummary = result.data.slices_summary;
    const isGrouped = Boolean(slicesSummary?.is_grouped);

    const imageVersions = buildBaseImageVersions({
        rawUrl,
        cleanedUrl: cleanedUrl ?? mergedUrl,
        mergedUrl,
        isGrouped,
    });

    return {
        jobId: result.data.job_id,
        rawUrl,
        cleanedUrl: cleanedUrl ?? mergedUrl,
        mergedUrl,
        activeBaseUrl: getActiveBaseImageUrl(imageVersions),
        imageVersions,
        slicesSummary,
        data: result.data as unknown as Record<string, unknown>,
    };
}

export interface ToggleGroupingStepResult {
    imageVersions: ImageVersion[];
    slicesSummary: SlicesSummary;
}

export async function runToggleGrouping(
    jobId: string,
    grouped: boolean,
    existingRawUrl: string | null
): Promise<ToggleGroupingStepResult> {
    const result = await setSlideGrouping(jobId, grouped);
    if (!result.success || !result.data) {
        throw new Error(result.error || 'Błąd zmiany grupowania');
    }

    const bust = `v=${Date.now()}`;
    const rawUrl = existingRawUrl ?? `${API_ORIGIN}/api/tiff/${encodeURIComponent(jobId)}/`;
    const cleanedUrl = `${API_ORIGIN}${result.data.cleaned_url}?${bust}`;
    const mergedUrl = `${API_ORIGIN}${result.data.origin_detect_url}?${bust}`;
    const slicesSummary = result.data.slices_summary;

    const imageVersions = buildBaseImageVersions({
        rawUrl,
        cleanedUrl,
        mergedUrl,
        isGrouped: slicesSummary.is_grouped,
    });

    return { imageVersions, slicesSummary };
}

export interface FibrosisStepResult {
    data: FibrosisResponse;
    overlayUrl: string | null;
}

export async function runFibrosis(
    jobId: string,
    threshold?: number,
    signal?: AbortSignal
): Promise<FibrosisStepResult> {
    const result = await analyzeFibrosis(jobId, threshold, signal);

    if (!result.success || !result.data) {
        throw new Error(result.error || 'Błąd analizy zwłóknienia');
    }
    if (result.data.error) {
        throw new Error(formatApiError(result.data.error, 'Błąd analizy zwłóknienia'));
    }

    const imagePath = result.data.image_path;
    const baseUrl = typeof imagePath === 'string' && imagePath.length > 0
        ? toResultImageUrl(imagePath, jobId)
        : null;

    return {
        data: result.data,
        overlayUrl: baseUrl ? `${baseUrl}?v=${Date.now()}` : null,
    };
}

export interface LengthStepResult {
    data: LengthResponse;
    overlayUrl: string | null;
}

export async function runLength(jobId: string, signal?: AbortSignal): Promise<LengthStepResult> {
    const result = await analyzeLength(jobId, signal);

    if (!result.success || !result.data) {
        throw new Error(result.error || 'Błąd analizy długości');
    }
    if (result.data.error) {
        throw new Error(formatApiError(result.data.error, 'Błąd analizy długości'));
    }

    const imagePath = result.data.image_path;
    const baseUrl = typeof imagePath === 'string' && imagePath.length > 0
        ? toResultImageUrl(imagePath, jobId)
        : null;

    return {
        data: result.data,
        overlayUrl: baseUrl ? `${baseUrl}?v=${Date.now()}` : null,
    };
}

export interface GlomeruliStepEvents {
    onSlideInfo?: (info: SlideInfo) => void;
    onGlomeruli?: (batch: Glomeruli[]) => void;
    onTiles?: (tiles: TileInfo[]) => void;
}

export interface GlomeruliStepResult {
    count: number;
    glomeruli: Glomeruli[];
}

interface GlomeruliStreamMessage {
    error?: string;
    slide_info?: SlideInfo;
    glomeruli?: Glomeruli[];
    tiles?: TileInfo[];
    done?: boolean;
    count?: number;
    final_glomeruli?: Glomeruli[];
}

/** Wraps the SSE detection stream in a promise that settles once the scan ends. */
export function runGlomeruli(
    jobId: string,
    events: GlomeruliStepEvents = {},
    signal?: AbortSignal
): Promise<GlomeruliStepResult> {
    return new Promise((resolve, reject) => {
        if (signal?.aborted) {
            reject(new DOMException('Anulowano', 'AbortError'));
            return;
        }

        const es = new EventSource(`${API_ORIGIN}/api/glomeruli/stream/?job_id=${encodeURIComponent(jobId)}`);
        let settled = false;

        const onAbort = () => {
            finish(() => reject(new DOMException('Anulowano', 'AbortError')));
        };

        const finish = (fn: () => void) => {
            if (settled) return;
            settled = true;
            signal?.removeEventListener('abort', onAbort);
            es.close();
            fn();
        };

        signal?.addEventListener('abort', onAbort, { once: true });

        es.onmessage = (e) => {
            let msg: GlomeruliStreamMessage;
            try {
                msg = JSON.parse(e.data) as GlomeruliStreamMessage;
            } catch {
                return;
            }

            if (msg.error) {
                finish(() => reject(new Error(formatApiError(msg.error))));
                return;
            }
            if (msg.slide_info) events.onSlideInfo?.(msg.slide_info);
            if (msg.glomeruli) events.onGlomeruli?.(msg.glomeruli);
            if (msg.tiles) events.onTiles?.(msg.tiles);
            if (msg.done) {
                finish(() => resolve({
                    count: msg.count ?? 0,
                    glomeruli: msg.final_glomeruli ?? [],
                }));
            }
        };

        es.onerror = () => {
            finish(() => reject(new Error(formatApiError('Błąd połączenia SSE'))));
        };
    });
}
