---
description: Open tasks from the last two weeks — the Dashboard task-shelf query.
query:
  has_task: true
  after: "-14d"
---

The deterministic shelf the Dashboard renders. `has_task` keeps blocks
with an open `- [ ]` checkbox or a `type/task` tag, and always excludes
`-14d` resolves
to fourteen days before the day the query runs.
