import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Inline stringToSlug to avoid importing from src during config load
const stringToSlug = (str: string) => {
  return str
    .toLowerCase()
    .trim()
    .replace(/[^\w\s-]/g, '')
    .replace(/[\s_-]+/g, '-')
    .replace(/^-+|-+$/g, '');
};

// https://vitejs.dev/config/
export default () => {
  const env = loadEnv("dev", process.cwd());
  return defineConfig({
    base: `/${stringToSlug(env.VITE_TEAM_NAME)}/`,
    plugins: [react()],
    server: {
      port: 8080
    }
  });
};
