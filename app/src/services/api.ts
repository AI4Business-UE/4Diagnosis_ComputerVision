const API_BASE_URL = 'http://localhost:8000/api';

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
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
export async function convertToTiff(): Promise<ApiResponse<{ message: string; progress?: number }>> {
  try {
    const response = await fetch(`${API_BASE_URL}/convert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
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
 * Analizuje włóknienia (fibrosis)
 */
export async function analyzeFibrosis(jobId: string): Promise<ApiResponse<{ results: Record<string, unknown> }>> {
    try {
      const response = await fetch(`${API_BASE_URL}/fibrosis/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: jobId }),
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
 * Analizuje długość struktur
 */
export async function analyzeLength(jobId: string): Promise<ApiResponse<{ results: Record<string, unknown> }>> {
  try {
    const response = await fetch(`${API_BASE_URL}/length/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: jobId }),
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
 * Wykrywa kłębuszki (glomerule)
 */
export async function detectGlomerules(): Promise<ApiResponse<{ results: Record<string, unknown> }>> {
  try {
    const response = await fetch(`${API_BASE_URL}/glomerule`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
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
