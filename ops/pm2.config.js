// SentiSense live UI — pm2 process (host process, NOT docker; runs the FastAPI app on :3000).
// The UI runs inside the /tf container as a plain uv process; pm2 keeps it alive across
// logout/reboot. Start:   pm2 start ops/pm2.config.js
//                         pm2 save && pm2 startup   (persist across reboot)
// Logs:   pm2 logs sentisense-ui   (also logs/ui_*.log)
// The built SPA must exist (cd ui/frontend && npm install && npm run build) — until then the
// backend serves an API-only placeholder at /.
module.exports = {
  apps: [
    {
      name: "sentisense-ui",
      cwd: "/tf/Data-Science-Final-Project",
      script: "uv",
      args: "run --extra ui --extra finance --extra ml python -m ui.app",
      interpreter: "none",
      autorestart: true,
      max_restarts: 10,
      env: { SENTISENSE_UI_PORT: "3000" },
      out_file: "logs/ui_out.log",
      error_file: "logs/ui_err.log",
      time: true,
    },
  ],
};
