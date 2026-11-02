import type { Config } from "tailwindcss";
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 950: "#070707", 900: "#0d0d0d", 850: "#141414", 800: "#1b1b1b", 700: "#262626", 600: "#343434", 500: "#4a4a4a" },
        accent: { DEFAULT: "#e50914", soft: "#b20710", glow: "#ff3b45" },
      },
      fontFamily: { display: ["var(--font-display)", "Georgia", "serif"], sans: ["var(--font-sans)", "system-ui", "sans-serif"] },
    },
  },
  plugins: [],
};
export default config;
