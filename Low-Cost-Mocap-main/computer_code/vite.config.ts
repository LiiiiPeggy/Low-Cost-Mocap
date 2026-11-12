import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 红外相机预览后端地址。默认指向独立预览服务 api/preview_server.py；
// 若改用 api/index.py（3001，同时提供 socket.io），可设置 MOCAP_PREVIEW_URL=http://127.0.0.1:3001
const PREVIEW_URL = process.env.MOCAP_PREVIEW_URL ?? 'http://127.0.0.1:3002'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // 相机预览 REST + MJPEG（CameraPreview.tsx 用相对路径访问）
      '/api': { target: PREVIEW_URL, changeOrigin: true },
      // 原项目的三维可视化数据链路
      '/socket.io': { target: 'http://127.0.0.1:3001', changeOrigin: true, ws: true },
    },
  },
})
