// SentiSense live UI — pm2 process (plain process on :3000, NOT docker).
//
// Topology: the UI runs on the DB machine (it only READS Postgres + renders; predictions are
// written by the pipeline on the GPU container). It needs NO GPU/torch — only the `ui` extra
// (FastAPI/uvicorn) + `miro` (requests, for the sim-run websocket). Set `cwd` to the repo path
// on THIS host, and provide the env below (DB is local on the DB machine).
//
// Start:   pm2 start ops/pm2.config.js && pm2 save && pm2 startup   (persist across reboot)
// Logs:    pm2 logs sentisense-ui   (also logs/ui_*.log)
// Build the SPA once first: cd ui/frontend && npm install && npm run build
module.exports = {
  apps: [
    {
      name: "sentisense-ui",
      cwd: process.env.SENTISENSE_REPO || "/tf/Data-Science-Final-Project",  // override per host
      script: "uv",
      args: "run --extra ui --extra miro python -m ui.app",
      interpreter: "none",
      autorestart: true,
      max_restarts: 10,
      env: {
        SENTISENSE_UI_PORT: "3000",
        // Set on the DB machine (DB is local there):
        //   SENTISENSE_DATABASE_URL=postgresql://postgres:***@localhost:5432/sentisense
        //   SENTISENSE_ACTIVE_MODEL=mistral-small-4
        //   SENTISENSE_MIRO_BASE_URL=http://<container-ip>:5001   (only for "run new sim" from the UI)
      },
      out_file: "logs/ui_out.log",
      error_file: "logs/ui_err.log",
      time: true,
    },
  ],
};
