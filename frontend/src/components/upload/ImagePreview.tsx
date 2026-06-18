"use client";

/**
 * ImagePreview shows a thumbnail of a selected image along with its file name
 * and a button to clear the current selection. The object URL backing the
 * preview (`src`) is owned by the parent, which is responsible for creating and
 * revoking it; this component only renders what it is given.
 */

/** Props for {@link ImagePreview}. */
export interface ImagePreviewProps {
  /** The object URL (or other source) backing the preview image. */
  src: string;
  /** The display name of the selected file. */
  fileName: string;
  /** Invoked when the user clears the current selection. */
  onClear: () => void;
}

/**
 * Renders a bordered thumbnail of the given image with its name and a clear
 * button below.
 *
 * @param props - The `src`, `fileName`, and `onClear` callback.
 * @returns The rendered preview element.
 */
export default function ImagePreview({
  src,
  fileName,
  onClear,
}: ImagePreviewProps): JSX.Element {
  return (
    <div className="flex flex-col gap-2">
      <div
        className="overflow-hidden rounded-xl border border-argus-border bg-argus-surface"
        style={{ width: 200, height: 200 }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={fileName}
          width={200}
          height={200}
          className="h-full w-full object-cover"
        />
      </div>
      <div className="flex w-[200px] items-center justify-between gap-2 font-mono text-xs text-argus-muted">
        <span className="truncate text-white">{fileName}</span>
        <button
          type="button"
          onClick={onClear}
          className="shrink-0 uppercase tracking-wider text-argus-muted transition-colors hover:text-argus-danger"
        >
          Clear
        </button>
      </div>
    </div>
  );
}
