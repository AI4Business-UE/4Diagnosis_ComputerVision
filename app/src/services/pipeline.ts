import {
    convertToTiff,
    analyzeFibrosis,
    analyzeLength,
} from './api';
import type { FibrosisResponse, Glomeruli, LengthResponse, SlideInfo, TileInfo } from './api';

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

export interface ConvertStepResult {
    jobId: string;
    previewUrl: string | null;
    data: Record<string, unknown>;
}

export async function runConvert(files: File[]): Promise<ConvertStepResult> {
    const result = await convertToTiff(files);

    if (!result.success || !result.data) {
        throw new Error(result.error || 'Nieznany błąd konwersji');
    }
    if (!result.data.job_id) {
        throw new Error('Backend nie zwrócił job_id');
    }

    const previewPath = result.data.origin_detect_url || result.data.tiff_url;

    return {
        jobId: result.data.job_id,
        previewUrl: previewPath ? `${API_ORIGIN}${previewPath}` : null,
        data: result.data as unknown as Record<string, unknown>,
    };
}

export interface FibrosisStepResult {
    data: FibrosisResponse;
    overlayUrl: string | null;
}

export async function runFibrosis(jobId: string): Promise<FibrosisStepResult> {
    const result = await analyzeFibrosis(jobId);

    if (!result.success || !result.data) {
        throw new Error(result.error || 'Błąd analizy zwłóknienia');
    }

    const imagePath = result.data.image_path;
    return {
        data: result.data,
        overlayUrl: typeof imagePath === 'string' && imagePath.length > 0
            ? toResultImageUrl(imagePath, jobId)
            : null,
    };
}

export interface LengthStepResult {
    data: LengthResponse;
    overlayUrl: string | null;
}

export async function runLength(jobId: string): Promise<LengthStepResult> {
    const result = await analyzeLength(jobId);

    if (!result.success || !result.data) {
        throw new Error(result.error || 'Błąd analizy długości');
    }

    const imagePath = result.data.image_path;
    return {
        data: result.data,
        overlayUrl: typeof imagePath === 'string' && imagePath.length > 0
            ? toResultImageUrl(imagePath, jobId)
            : null,
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
    gridUrl: string | null;
}

interface GlomeruliStreamMessage {
    error?: string;
    slide_info?: SlideInfo;
    glomeruli?: Glomeruli[];
    tiles?: TileInfo[];
    done?: boolean;
    count?: number;
    final_glomeruli?: Glomeruli[];
    glom_grid_url?: string;
}

/** Wraps the SSE detection stream in a promise that settles once the scan ends. */
export function runGlomeruli(jobId: string, events: GlomeruliStepEvents = {}): Promise<GlomeruliStepResult> {
    return new Promise((resolve, reject) => {
        const es = new EventSource(`${API_ORIGIN}/api/glomeruli/stream/?job_id=${encodeURIComponent(jobId)}`);
        let settled = false;

        const finish = (fn: () => void) => {
            if (settled) return;
            settled = true;
            es.close();
            fn();
        };

        es.onmessage = (e) => {
            let msg: GlomeruliStreamMessage;
            try {
                msg = JSON.parse(e.data) as GlomeruliStreamMessage;
            } catch {
                return;
            }

            if (msg.error) {
                finish(() => reject(new Error(msg.error)));
                return;
            }
            if (msg.slide_info) events.onSlideInfo?.(msg.slide_info);
            if (msg.glomeruli) events.onGlomeruli?.(msg.glomeruli);
            if (msg.tiles) events.onTiles?.(msg.tiles);
            if (msg.done) {
                finish(() => resolve({
                    count: msg.count ?? 0,
                    glomeruli: msg.final_glomeruli ?? [],
                    gridUrl: msg.glom_grid_url ? `${API_ORIGIN}${msg.glom_grid_url}` : null,
                }));
            }
        };

        es.onerror = () => {
            finish(() => reject(new Error('Błąd połączenia SSE')));
        };
    });
}
