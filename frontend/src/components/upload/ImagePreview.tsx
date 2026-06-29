"use client";

/**
 * ImagePreview shows a thumbnail of a selected image with its file name and a
 * clear button. The object URL backing `src` is owned by the parent. Styled for
 * the bright "Luminous Clinical Theatre" theme.
 */

export interface ImagePreviewProps {
  src: string;
  fileName: string;
  onClear: () => void;
}

export default function ImagePreview({
  src,
  fileName,
  onClear,
}: ImagePreviewProps): React.JSX.Element {
  return (
    <div className="flex flex-col gap-2">
      <div
        className="overflow-hidden rounded-2xl border border-hairline bg-surface shadow-panel"
        style={{ width: 220, height: 220 }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={fileName}
          width={220}
          height={220}
          className="h-full w-full object-cover"
        />
      </div>
      <div className="flex w-[220px] items-center justify-between gap-2 font-mono text-xs text-ink-faint">
        <span className="truncate text-ink-soft">{fileName}</span>
        <button
          type="button"
          onClick={onClear}
          className="shrink-0 uppercase tracking-wider text-ink-faint transition-colors hover:text-danger"
        >
          Clear
        </button>
      </div>
    </div>
  );
}
