import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Churn Intelligence Platform",
  description:
    "AI-powered customer churn prediction and MLOps dashboard with Gemini integration",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
