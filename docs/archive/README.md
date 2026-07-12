# docs/archive — closed investigations

Historical working papers, moved here 2026-07-13. Everything is preserved
verbatim (git history intact via `git mv`); file paths mentioned INSIDE these
docs describe the repo layout at the time of writing.

Contents, by saga:

- **WD-waste / body-norm-ramp investigation (2026-06)** — the Math-Agent
  relay: `MATH_AGENT_BRIEF_*` / `MATH_AGENT_Q7..Q10` / `RELAY_probe1_pass`,
  plus probe results (`FSDP_PROBE_MATRIX`, `REPLICATED_DATA_RESULT`,
  `PROBE_A_clip_replay_RESULTS`, `STAGE_A_ANCHOR_SETBACK`). CONCLUSION lives
  in the active `docs/WD_WASTE_ANALYSIS.md` (the −0.0129 lean is the
  post-Newton-Schulz update; fix = tangent projection, now shipped).
  Q11/Q12 remain ACTIVE (they govern the self-anchoring controller's
  12k-24k engagement window).
- **z-loss / CCE (2026-06)** — `ZLOSS_CCE_PROBLEM` (option-D reconstruction,
  shipped) and `ZLOSS_CENTERED_PLAN` (centered variant, shipped).
- **Superseded designs** — `kv2_ANNEAL_OPTIONS` (superseded by kv3 + the
  shadow-norm controller), `MATH_AGENT_Q_rowcenter_inloop_redundant`,
  `qk_clip_synopsis` (2026-01).

Rule of thumb for future archiving: a doc moves here when the investigation
it served is CLOSED and its conclusion is captured in an active doc, the
code, or the test suite.
