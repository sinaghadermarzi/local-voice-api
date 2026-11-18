# Chatterbox Turbo voices

Drop reference clips here as `<name>.wav` (or `.mp3`, `.flac`, `.ogg`). Each must be longer than
5 seconds; 10–20 s of clean, single-speaker speech works best. Then request
`{"engine": "chatterbox-turbo", "voice": "<name>"}`. `GET /v1/voices` lists them.

Clips are picked up without a restart. Each voice's conditioning is computed on first use and cached.
Only clone voices you have the right to use.
