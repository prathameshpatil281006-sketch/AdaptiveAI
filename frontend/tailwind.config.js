/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        aa: {
          bg: '#060b14',
          sidebar: '#0a101d',
          card: '#0e1526',
          card2: '#111a2e',
          border: '#1c2740',
          borderSoft: '#162032',
          text: '#e6ebf4',
          muted: '#8b98b3',
          faint: '#5b6b8c',
          blue: '#2f6bff',
          blueSoft: '#1d2f5b',
          blueText: '#6ea8ff',
          green: '#22c55e',
          greenSoft: 'rgba(34,197,94,0.12)',
          amber: '#f59e0b',
          amberSoft: 'rgba(245,158,11,0.12)',
          red: '#ef4444',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
      fontSize: {
        '2xs': ['10px', { lineHeight: '14px' }],
      },
      boxShadow: {
        card: '0 1px 0 rgba(255,255,255,0.03) inset, 0 8px 24px rgba(0,0,0,0.35)',
      },
    },
  },
  plugins: [],
}
