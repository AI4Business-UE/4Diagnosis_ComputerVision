export type ConversionPreset = 'S' | 'M' | 'L' | 'XL';

export interface AppSettings {
  conversionPreset: ConversionPreset;
  cleaningLevel: number;
  enableSliceGrouping: boolean;
  includeGlomeruliInAnalyzeAll: boolean;
  defaultFibrosisThreshold: number;
  defaultGlomeruliConf: number;
}

export const DEFAULT_APP_SETTINGS: AppSettings = {
  conversionPreset: 'L',
  cleaningLevel: 5,
  enableSliceGrouping: true,
  includeGlomeruliInAnalyzeAll: true,
  defaultFibrosisThreshold: 0.4,
  defaultGlomeruliConf: 0.15,
};

const STORAGE_KEY = '4diagnosis_cv_settings_v1';

const VALID_PRESETS: ConversionPreset[] = ['S', 'M', 'L', 'XL'];

export function loadAppSettings(): AppSettings {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_APP_SETTINGS };
    const parsed = JSON.parse(raw) as Partial<AppSettings>;

    const preset = VALID_PRESETS.includes(parsed.conversionPreset as ConversionPreset)
      ? (parsed.conversionPreset as ConversionPreset)
      : DEFAULT_APP_SETTINGS.conversionPreset;

    const cleaning = typeof parsed.cleaningLevel === 'number' && Number.isFinite(parsed.cleaningLevel)
      ? Math.min(10, Math.max(0, Math.round(parsed.cleaningLevel)))
      : DEFAULT_APP_SETTINGS.cleaningLevel;

    const fib = typeof parsed.defaultFibrosisThreshold === 'number'
      ? Math.min(0.5, Math.max(0.3, parsed.defaultFibrosisThreshold))
      : DEFAULT_APP_SETTINGS.defaultFibrosisThreshold;

    const conf = typeof parsed.defaultGlomeruliConf === 'number'
      ? Math.min(0.95, Math.max(0.15, parsed.defaultGlomeruliConf))
      : DEFAULT_APP_SETTINGS.defaultGlomeruliConf;

    return {
      conversionPreset: preset,
      cleaningLevel: cleaning,
      enableSliceGrouping:
        typeof parsed.enableSliceGrouping === 'boolean'
          ? parsed.enableSliceGrouping
          : DEFAULT_APP_SETTINGS.enableSliceGrouping,
      includeGlomeruliInAnalyzeAll:
        typeof parsed.includeGlomeruliInAnalyzeAll === 'boolean'
          ? parsed.includeGlomeruliInAnalyzeAll
          : DEFAULT_APP_SETTINGS.includeGlomeruliInAnalyzeAll,
      defaultFibrosisThreshold: fib,
      defaultGlomeruliConf: conf,
    };
  } catch {
    return { ...DEFAULT_APP_SETTINGS };
  }
}

export function saveAppSettings(settings: AppSettings): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {}
}
