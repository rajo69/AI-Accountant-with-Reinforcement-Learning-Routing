"use client";

import { useState } from "react";
import { usePathname } from "next/navigation";
import { Menu } from "lucide-react";
import { Sheet, SheetContent } from "@/components/ui/sheet";
import Sidebar from "@/components/sidebar";

const APP_ROUTES = ["/dashboard", "/transactions", "/reconciliation", "/documents", "/settings"];

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);
  const showSidebar = APP_ROUTES.some((r) => pathname.startsWith(r));

  if (!showSidebar) return <>{children}</>;

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">
      {/* Desktop sidebar */}
      <div className="hidden md:flex">
        <Sidebar onNavigate={() => {}} />
      </div>

      {/* Mobile hamburger */}
      <button
        className="md:hidden fixed top-4 left-4 z-50 w-9 h-9 bg-white rounded-lg flex items-center justify-center shadow-sm border border-slate-200"
        onClick={() => setMobileOpen(true)}
      >
        <Menu className="w-4 h-4 text-slate-600" />
      </button>

      {/* Mobile sidebar drawer */}
      <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
        <SheetContent side="left" className="p-0 w-60 border-none">
          <Sidebar onNavigate={() => setMobileOpen(false)} />
        </SheetContent>
      </Sheet>

      <main className="flex-1 overflow-y-auto bg-slate-50">
        <div className="min-h-full p-4 md:p-8 pt-16 md:pt-8">{children}</div>
      </main>
    </div>
  );
}
