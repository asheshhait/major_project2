export function parseTrainingStatus(payload) {
  return {
    status: payload?.status || "idle",
    epoch: payload?.epoch || 0,
    loss: payload?.loss ?? null,
  };
}
