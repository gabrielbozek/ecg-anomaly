# Working agreement

- Discuss the work with the user in Polish. Write code, identifiers, comments,
  docstrings, notebook text, chart labels, and project documentation in English.
- Implement only the user's current request. Read relevant code and technical
  decisions first. Check local Git changes; remote files may be incomplete.
- Before implementing work that depends on an unresolved decision about the
  objective, data, method, interface, or evaluation, explain the uncertainty,
  give 2–3 options with consequences, and ask the user. A recommendation is
  not approval. Wait for the answer; continue only independent work meanwhile.
- Ask short, concrete questions whenever the answer materially changes the
  solution. Do not silently choose observation definitions, label mappings,
  exclusions, splits, preprocessing, scores, thresholds, or evaluation rules.
- Respect decisions already made. Resolve routine implementation details
  independently; do not turn every minor coding choice into a question.
- Challenge faulty assumptions and suggest alternatives with reasons. Do not
  expand the scope or implement an alternative without the user's decision.
- The user defines the solution; you may write its full implementation. Do not
  infer a roadmap, invent subsequent tasks, or generate teaching exercises.
- Keep solutions small. Preserve existing work and raw data. Use the project's
  uv environment and lockfile; add dependencies only when the task needs them.
- Verify relevant risks, such as units, indices, IDs, boundaries, leakage, and
  reproducibility. Report exactly what ran, what passed, and what was not verified.
- Document only what is needed to understand and reproduce the work. An existing
  request or technical note may be sufficient; avoid duplicate paperwork.
- Give each new function a short English docstring describing what it does,
  each parameter (including units where relevant), and its return value.
- If asked for review, review first and do not automatically rewrite the code.
  Commit and push only when requested. Never overwrite local changes or use
  destructive Git operations without explicit authorization.
