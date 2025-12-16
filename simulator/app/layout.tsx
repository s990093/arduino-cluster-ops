import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GPU Trace Architect",
  description: "GPU Trace Visualization and Analysis Tool",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
