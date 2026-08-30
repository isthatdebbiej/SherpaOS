import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Pemba's Field Journal",
  description: "A tiny robot's diary from the roof of the world.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
