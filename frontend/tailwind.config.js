/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        base: {
          950: "#0b0d10",
          900: "#111417",
          850: "#161a1e",
          800: "#1c2126",
          700: "#2a3037",
          600: "#3a4148",
          500: "#5b636b",
          400: "#8a929a",
          300: "#b7bec4",
          200: "#dde1e4",
        },
        accent: {
          DEFAULT: "#4fd1a5",
          muted: "#2f6e57",
          dim: "#1d4536",
        },
        negative: {
          DEFAULT: "#e07a7a",
          muted: "#7a3f3f",
          dim: "#4a2828",
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
      },
    },
  },
  plugins: [],
};
