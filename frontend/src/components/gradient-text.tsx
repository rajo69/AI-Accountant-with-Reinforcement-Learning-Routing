"use client";

import { useEffect, useRef } from "react";

interface GradientTextProps {
  children: React.ReactNode;
  className?: string;
  colors?: string[];
  animationSpeed?: number;
}

export default function GradientText({
  children,
  className = "",
  colors = ["#818cf8", "#a78bfa", "#38bdf8", "#6366f1", "#818cf8"],
  animationSpeed = 6,
}: GradientTextProps) {
  const spanRef = useRef<HTMLSpanElement>(null);
  const rafRef = useRef<number>(0);
  const startRef = useRef<number | null>(null);

  useEffect(() => {
    const el = spanRef.current;
    if (!el) return;

    const gradient = `linear-gradient(135deg, ${colors.join(", ")})`;
    el.style.backgroundImage = gradient;
    el.style.backgroundSize = "300% 300%";

    function tick(ts: number) {
      if (!startRef.current) startRef.current = ts;
      const elapsed = ts - startRef.current;
      // cycle through 0–100 of background-position
      const pos = ((elapsed * animationSpeed * 0.002) % 100).toFixed(2);
      if (el) el.style.backgroundPosition = `${pos}% 50%`;
      rafRef.current = requestAnimationFrame(tick);
    }

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animationSpeed]);

  return (
    <span
      ref={spanRef}
      className={`bg-clip-text text-transparent ${className}`}
      style={{
        backgroundImage: `linear-gradient(135deg, ${colors.join(", ")})`,
        backgroundSize: "300% 300%",
      }}
    >
      {children}
    </span>
  );
}
