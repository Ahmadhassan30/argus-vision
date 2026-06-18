"use client";

/**
 * DropZone provides a drag-and-drop / click-to-select area for a single
 * dermoscopic image (JPEG or PNG, up to 10 MB). On a valid drop it reads the
 * image dimensions, surfaces the file size and resolution, and notifies the
 * parent via `onFileSelected`. Rejected files (wrong type or too large) render
 * an inline error message. The border pulses between the Agent A and Agent B
 * colors while a drag is active.
 */

import { useCallback, useState } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { motion } from "framer-motion";
import clsx from "clsx";

/** The maximum accepted image size in bytes (10 MB). */
const MAX_SIZE_BYTES = 10 * 1024 * 1024;

/** Props for {@link DropZone}. */
export interface DropZoneProps {
  /** Invoked with the accepted image file once a valid file is provided. */
  onFileSelected: (file: File) => void;
}

/** The pixel dimensions of a read image. */
interface ImageDimensions {
  width: number;
  height: number;
}

/**
 * Formats a byte count into a human-readable KB/MB string.
 *
 * @param bytes - The size in bytes.
 * @returns A formatted size string (e.g. "842.0 KB" or "3.21 MB").
 */
function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/**
 * Translates the first react-dropzone rejection into a user-facing message.
 *
 * @param rejections - The rejected files reported by react-dropzone.
 * @returns A human-readable error message, or `null` when none apply.
 */
function rejectionMessage(rejections: FileRejection[]): string | null {
  if (rejections.length === 0) {
    return null;
  }
  const first = rejections[0];
  const error = first.errors[0];
  if (!error) {
    return "This file could not be accepted.";
  }
  switch (error.code) {
    case "file-too-large":
      return "File is too large. Maximum size is 10 MB.";
    case "file-invalid-type":
      return "Unsupported file type. Upload a JPG or PNG image.";
    case "too-many-files":
      return "Only a single image can be uploaded at a time.";
    default:
      return error.message;
  }
}

/**
 * A single-file image dropzone with validation and dimension readout.
 *
 * @param props - The `onFileSelected` callback.
 * @returns The rendered dropzone element.
 */
export default function DropZone({
  onFileSelected,
}: DropZoneProps): JSX.Element {
  const [acceptedFile, setAcceptedFile] = useState<File | null>(null);
  const [dimensions, setDimensions] = useState<ImageDimensions | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    (accepted: File[], rejections: FileRejection[]): void => {
      const message = rejectionMessage(rejections);
      if (message) {
        setError(message);
        setAcceptedFile(null);
        setDimensions(null);
        return;
      }

      const file = accepted[0];
      if (!file) {
        return;
      }

      setError(null);
      setAcceptedFile(file);

      const objectUrl = URL.createObjectURL(file);
      const img = new Image();
      img.onload = (): void => {
        setDimensions({ width: img.naturalWidth, height: img.naturalHeight });
        URL.revokeObjectURL(objectUrl);
      };
      img.onerror = (): void => {
        setDimensions(null);
        URL.revokeObjectURL(objectUrl);
      };
      img.src = objectUrl;

      onFileSelected(file);
    },
    [onFileSelected]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/jpeg": [".jpg", ".jpeg"],
      "image/png": [".png"],
    },
    maxFiles: 1,
    maxSize: MAX_SIZE_BYTES,
    multiple: false,
  });

  return (
    <div className="flex w-full flex-col gap-3">
      <motion.div
        {...getRootProps()}
        className={clsx(
          "flex cursor-pointer flex-col items-center justify-center gap-2",
          "rounded-xl border-2 border-dashed bg-argus-surface px-6 py-12 text-center",
          "transition-colors",
          isDragActive
            ? "animate-pulse-border"
            : "border-argus-border hover:border-argus-agent-a"
        )}
        animate={isDragActive ? { scale: 1.01 } : { scale: 1 }}
        transition={{ type: "spring", stiffness: 300, damping: 24 }}
      >
        <input {...getInputProps()} />
        <p className="font-display text-sm text-white">
          {isDragActive
            ? "Drop the image to analyze"
            : "Drag & drop a dermoscopic image here"}
        </p>
        <p className="font-mono text-xs text-argus-muted">
          JPG or PNG · up to 10 MB · click to browse
        </p>
      </motion.div>

      {error && (
        <p className="font-mono text-xs text-argus-danger" role="alert">
          {error}
        </p>
      )}

      {acceptedFile && !error && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-xs text-argus-muted">
          <span className="truncate text-white">{acceptedFile.name}</span>
          <span>{formatBytes(acceptedFile.size)}</span>
          {dimensions && (
            <span>
              {dimensions.width}&times;{dimensions.height}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
