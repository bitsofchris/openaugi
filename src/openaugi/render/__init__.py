"""Render — static HTML surfaces generated from the DB (M6).

Same philosophy as views: derived, regenerable, file-based. `openaugi
render` writes a self-contained interactive HTML (data inlined as JSON,
client-side JS, no server). Read-only over the store.
"""
