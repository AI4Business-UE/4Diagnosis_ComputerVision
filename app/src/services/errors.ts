export function formatApiError(error: unknown, fallback: string = 'Wystąpił nieoczekiwany błąd'): string {
  const raw =
    typeof error === 'string'
      ? error
      : error instanceof Error
        ? error.message
        : '';

  const msg = raw.trim();
  if (!msg) return fallback;

  const lower = msg.toLowerCase();

  if (
    lower.includes('failed to fetch') ||
    lower.includes('networkerror') ||
    lower.includes('load failed') ||
    lower.includes('err_connection_refused') ||
    lower.includes('network request failed')
  ) {
    return 'Brak połączenia z serwerem.';
  }

  if (lower.includes('no .mrxs file')) {
    return 'Brak pliku .mrxs w wybranym folderze próbki.';
  }
  if (lower.includes('index.dat missing') || lower.includes('data*.dat missing') || lower.includes('slidedat.ini missing')) {
    return `Niekompletne pliki skanu MRXS (${msg}).`;
  }

  if (lower.includes('tiff conversion failed') || lower.includes('nie wykryto tkanki')) {
    return 'Nie udało się przekonwertować skanu — brak wykrytej tkanki lub uszkodzony plik MRXS.';
  }
  if (lower.includes('tiff save failed')) {
    return 'Nie udało się zapisać pliku TIFF na serwerze (sprawdź miejsce na dysku).';
  }
  if (lower.includes('cannot read converted tiff') || lower.includes('cannot read tiff')) {
    return 'Nie udało się odczytać skonwertowanego pliku TIFF na serwerze.';
  }

  if (
    lower.includes('job not found') ||
    lower.includes('source .mrxs not found') ||
    lower.includes('tiff not found') ||
    lower.includes('result image not found') ||
    lower.includes('origin detect tiff not found') ||
    lower.includes('metadata json not found')
  ) {
    return 'Nie znaleziono próbki - kliknij ponownie „Konwertuj”.';
  }

  if (lower.includes('out of memory') || lower.includes('memoryerror') || lower.includes('cannot allocate')) {
    return 'Za mało pamięci dla wybranej jakości konwersji - zmniejsz jakość konwersji w ustawieniach.';
  }

  if (lower.includes('błąd połączenia sse')) {
    return 'Przerwano połączenie podczas detekcji kłębuszków (sprawdź, czy serwer backendu działa).';
  }

  if (/^http\s+404$/i.test(msg)) {
    return 'HTTP 404: Nie znaleziono zasobu na serwerze.';
  }
  if (/^http\s+500$/i.test(msg)) {
    return 'HTTP 500: Wewnętrzny błąd serwera backendu.';
  }
  if (/^http\s+50[234]$/i.test(msg)) {
    return `Serwer backendu jest chwilowo niedostępny (${msg}).`;
  }

  return msg;
}
