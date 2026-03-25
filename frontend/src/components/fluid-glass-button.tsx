"use client";

import { useRef, useState, type MouseEvent, type ReactNode } from "react";
import { motion } from "framer-motion";

interface FluidGlassButtonProps {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "glass";
  className?: string;
  type?: "button" | "submit" | "reset";
}

export default function FluidGlassButton({
  children,
  onClick,
  disabled = false,
  variant = "glass",
  className = "",
  type = "button",
}: FluidGlassButtonProps) {
  const ref = useRef<HTMLButtonElement>(null);
  const [shimmerPos, setShimmerPos] = useState({ x: 50, y: 50 });
  const [hovered, setHovered] = useState(false);

  function handleMouseMove(e: MouseEvent<HTMLButtonElement>) {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    setShimmerPos({ x, y });
  }

  const isPrimary = variant === "primary";

  return (
    <motion.button
      ref={ref}
      type={type}
      disabled={disabled}
      onClick={onClick}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      whileTap={{ scale: disabled ? 1 : 0.97 }}
      className={`
        relative overflow-hidden rounded-lg px-4 py-2 text-sm font-medium
        transition-all duration-200 outline-none select-none
        disabled:opacity-50 disabled:cursor-not-allowed
        ${isPrimary
          ? "bg-indigo-600 text-white border border-indigo-500 shadow-lg shadow-indigo-600/25 hover:bg-indigo-500"
          : "bg-white/[0.06] text-zinc-300 border border-white/[0.1] hover:border-white/[0.2] hover:text-white"
        }
        ${className}
      `}
      style={
        !isPrimary
          ? { backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)" }
          : undefined
      }
    >
      {/* Shimmer layer */}
      <span
        aria-hidden
        className="pointer-events-none absolute inset-0 rounded-[inherit] opacity-0 transition-opacity duration-300"
        style={{
          opacity: hovered ? 1 : 0,
          background: `radial-gradient(circle at ${shimmerPos.x}% ${shimmerPos.y}%, ${
            isPrimary ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.08)"
          } 0%, transparent 60%)`,
        }}
      />
      {/* Top inset highlight */}
      <span
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 h-px rounded-t-lg"
        style={{
          background: isPrimary
            ? "linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)"
            : "linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent)",
        }}
      />
      <span className="relative flex items-center gap-2">{children}</span>
    </motion.button>
  );
}
