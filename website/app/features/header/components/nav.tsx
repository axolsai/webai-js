"use client";

import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { Menu, ChevronDown, ChevronRight } from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
  SheetTitle,
} from "@/components/ui/sheet";

export default function NavBar() {
  const navItems: Array<{
    name: string;
    href: string;
    badge: string | null;
    newTab: boolean;
    dropdown?: Array<{
      name: string;
      href: string;
      newTab: boolean;
      logoUrl?: string;
    }>;
  }> = [
    { name: "News", href: "https://www.webai-js.com/blogs", badge: null, newTab: false },
  ];

  const pathname = usePathname();
  const [activePath, setActivePath] = useState(pathname);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    const firstSegment = pathname.split("/")[1];
    setActivePath(`/${firstSegment}`);
  }, [pathname]);

  // Desktop Navigation
  const DesktopNav = () => (
    <ul className="hidden md:flex gap-x-4 lg:gap-x-6 text-sm">
      {navItems.map((item) => {
        const isActive =
          activePath === item.href ||
          (item.dropdown &&
            item.dropdown.some((subItem) => activePath === subItem.href));
        const hasDropdown = item.dropdown && item.dropdown.length > 0;

        return (
          <li key={item.name} className="relative group">
            {hasDropdown ? (
              <button
                className={`transition-colors duration-200 px-2 py-1 rounded-md flex items-center gap-1 cursor-default ${
                  isActive
                    ? "text-primary"
                    : "text-muted-foreground group-hover:text-primary"
                }`}
              >
                {item.badge && (
                  <span className="absolute -top-2 text-xs bg-orange-500/50 px-1 -right-5 rounded-md border border-orange-500">
                    {item.badge}
                  </span>
                )}
                {item.name}
                <ChevronDown className="h-3 w-3 transition-transform duration-200 group-hover:rotate-180" />
              </button>
            ) : (
              <Link
                href={item.href}
                target={item.newTab ? "_blank" : undefined}
                rel={item.newTab ? "noopener noreferrer" : undefined}
                className={`transition-colors duration-200 px-2 py-1 rounded-md flex items-center gap-1 ${
                  isActive
                    ? "text-primary"
                    : "text-muted-foreground hover:text-primary"
                }`}
              >
                {item.badge && (
                  <span className="absolute -top-2 text-xs bg-orange-500/50 px-1 -right-5 rounded-md border border-orange-500">
                    {item.badge}
                  </span>
                )}
                {item.name}
              </Link>
            )}

            {/* Dropdown */}
            {hasDropdown && (
              <div className="absolute top-full left-0 mt-2 w-64 bg-background border border-border rounded-md shadow-lg z-50 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 ease-out">
                {/* Invisible bridge to prevent flickering */}
                <div className="absolute -top-2 left-0 right-0 h-2 bg-transparent" />
                <div className="py-2">
                  {item.dropdown?.map((subItem) => {
                    return (
                      <Link
                        key={subItem.name}
                        href={subItem.href}
                        target={subItem.newTab ? "_blank" : undefined}
                        rel={subItem.newTab ? "noopener noreferrer" : undefined}
                        className="group/item block px-4 py-2.5 text-sm transition-all duration-200 hover:bg-muted/50 relative"
                      >
                        <div className="flex items-center gap-3">
                          {subItem.logoUrl && (
                            <div className="flex-shrink-0 w-7 h-7 rounded-lg flex items-center justify-center group-hover/item:bg-primary/20 transition-colors duration-200 overflow-hidden">
                              <img
                                src={subItem.logoUrl}
                                alt={`${subItem.name} logo`}
                                className="w-7 h-7 object-contain"
                              />
                            </div>
                          )}
                          <div className="flex-1">
                            <div className="font-medium text-foreground group-hover/item:text-primary transition-colors duration-200">
                              {subItem.name}
                            </div>
                          </div>
                          <ChevronRight className="h-4 w-4 text-muted-foreground opacity-0 group-hover/item:opacity-100 transform translate-x-[-4px] group-hover/item:translate-x-0 transition-all duration-200" />
                        </div>
                      </Link>
                    );
                  })}
                </div>
              </div>
            )}
          </li>
        );
      })}
    </ul>
  );

  // Mobile Navigation
  const MobileNav = () => (
    <Sheet open={isOpen} onOpenChange={setIsOpen}>
      <SheetTrigger className="md:hidden p-2" asChild>
        <button>
          <Menu className="h-6 w-6" />
        </button>
      </SheetTrigger>
      <SheetContent
        side="left"
        className="w-64 transform transition-transform duration-300 ease-in-out"
        onPointerDownOutside={(e) => e.preventDefault()}
        onOpenAutoFocus={(e) => e.preventDefault()}
      >
        <SheetTitle className="text-lg font-semibold mb-4 mt-4">
          Navigation
        </SheetTitle>
        <nav className="flex flex-col gap-2">
          {navItems.map((item) => {
            const isActive =
              activePath === item.href ||
              (item.dropdown &&
                item.dropdown.some((subItem) => activePath === subItem.href));
            const hasDropdown = item.dropdown && item.dropdown.length > 0;

            return (
              <div key={item.name}>
                {hasDropdown ? (
                  <div
                    className={`relative px-2 py-2 text-sm font-medium transition-colors rounded-sm flex items-center justify-between ${
                      isActive
                        ? "text-primary bg-muted"
                        : "text-muted-foreground"
                    }`}
                  >
                    <span>{item.name}</span>
                    {item.badge && (
                      <span className="ml-2 inline-block bg-orange-500/50 px-1 text-xs rounded-sm border border-orange-500">
                        {item.badge}
                      </span>
                    )}
                  </div>
                ) : (
                  <Link
                    href={item.href}
                    target={item.newTab ? "_blank" : undefined}
                    rel={item.newTab ? "noopener noreferrer" : undefined}
                    onClick={() => setIsOpen(false)}
                    className={`relative px-2 py-2 text-sm font-medium transition-colors rounded-sm flex items-center justify-between ${
                      isActive
                        ? "text-primary bg-muted"
                        : "text-muted-foreground hover:text-primary hover:bg-muted/50"
                    }`}
                  >
                    <span>{item.name}</span>
                    {item.badge && (
                      <span className="ml-2 inline-block bg-orange-500/50 px-1 text-xs rounded-sm border border-orange-500">
                        {item.badge}
                      </span>
                    )}
                  </Link>
                )}

                {/* Mobile Dropdown Items */}
                {hasDropdown && (
                  <div className="ml-4 mt-1">
                    {item.dropdown?.map((subItem) => (
                      <Link
                        key={subItem.name}
                        href={subItem.href}
                        target={subItem.newTab ? "_blank" : undefined}
                        rel={subItem.newTab ? "noopener noreferrer" : undefined}
                        onClick={() => setIsOpen(false)}
                        className="block px-2 py-1.5 text-xs text-muted-foreground hover:text-primary hover:bg-muted/30 transition-colors rounded-sm"
                      >
                        {subItem.name}
                      </Link>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </nav>
      </SheetContent>
    </Sheet>
  );

  return (
    <nav className="relative w-full">
      <div className="flex justify-between items-center">
        <MobileNav />
        <DesktopNav />
      </div>
    </nav>
  );
}
