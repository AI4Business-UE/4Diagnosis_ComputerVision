interface IconProps {
    size?: number;
    className?: string;
}

function Svg({ size = 20, className, children }: IconProps & { children: React.ReactNode }) {
    return (
        <svg
            width={size}
            height={size}
            viewBox="0 0 24 24"
            className={className}
            aria-hidden="true"
            focusable="false"
            fill="none"
            stroke="currentColor"
            strokeWidth={1.7}
            strokeLinecap="round"
            strokeLinejoin="round"
        >
            {children}
        </svg>
    );
}

/**
 * Result icons — one solid family, all drawn to the same 24px box so they read
 * at equal weight, each tinted via currentColor. (The old ones were black
 * FontAwesome bitmapsy of differing glyph sizes, and the hue-rotate filters
 * meant to tint them did nothing on black.)
 */

/** Two-tone pie — fibrosis share of tissue. */
export function FibrosisPieIcon({ size = 28, className }: IconProps) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden="true" focusable="false" fill="currentColor">
            <path d="M11.1 2.9a9.4 9.4 0 1 0 10 10h-10V2.9Z" />
            <path d="M13.4 2.9a9.4 9.4 0 0 1 7.7 7.7h-7.7V2.9Z" opacity="0.42" />
        </svg>
    );
}

/** Ruler — tissue length. */
export function RulerIcon({ size = 28, className }: IconProps) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden="true" focusable="false">
            <g transform="rotate(-45 12 12)">
                <rect x="1.8" y="8.8" width="20.4" height="6.4" rx="1.7" fill="currentColor" />
                <path
                    d="M6.2 8.8v3.1M10.4 8.8v3.1M14.6 8.8v3.1M18.8 8.8v3.1"
                    stroke="#fff"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                />
            </g>
        </svg>
    );
}

/** Cluster of cells — total count. */
export function GlomeruliCountIcon({ size = 28, className }: IconProps) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden="true" focusable="false" fill="currentColor">
            <circle cx="8.3" cy="8.8" r="4.7" />
            <circle cx="15.9" cy="16" r="5.6" />
            <circle cx="17.8" cy="6.3" r="3.1" />
        </svg>
    );
}

/** Healthy glomerulus — solid disc with a check. */
export function GlomeruliHealthyIcon({ size = 28, className }: IconProps) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden="true" focusable="false">
            <circle cx="12" cy="12" r="9.2" fill="currentColor" />
            <path
                d="M7.9 12.4l2.7 2.7 5.5-5.7"
                fill="none"
                stroke="#fff"
                strokeWidth={2.1}
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    );
}

/** Sclerotic glomerulus — solid disc with a cross. */
export function GlomeruliScleroticIcon({ size = 28, className }: IconProps) {
    return (
        <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden="true" focusable="false">
            <circle cx="12" cy="12" r="9.2" fill="currentColor" />
            <path
                d="M9 9l6 6M15 9l-6 6"
                fill="none"
                stroke="#fff"
                strokeWidth={2.1}
                strokeLinecap="round"
            />
        </svg>
    );
}

/** Bolt — run the whole pipeline on every sample. */
export function BoltIcon(props: IconProps) {
    return (
        <Svg {...props}>
            <path d="M13 3 5.5 13.2H11l-.9 7.8 7.9-10.6H12L13 3Z" />
        </Svg>
    );
}

export function ChevronLeftIcon(props: IconProps) {
    return (
        <Svg {...props}>
            <path d="M14.5 5.5 8 12l6.5 6.5" />
        </Svg>
    );
}

export function ChevronRightIcon(props: IconProps) {
    return (
        <Svg {...props}>
            <path d="M9.5 5.5 16 12l-6.5 6.5" />
        </Svg>
    );
}
