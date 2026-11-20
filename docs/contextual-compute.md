# Contextual compute and adapters

Status: `proposed`. The full argument is in [the Loom note](https://github.com/siliconcordat/loom/blob/main/docs/08-contextual-compute.md). This file states only what Layer 2 owns. It does not change the compiler, the certificates, or the adapter wire format.

Weft adapters are the parameter field for the local fusion. They open some behaviors and leave others unloaded. Hand-written weight-space programming is too crude to aim that field. Training the frontier model is the wrong target, because those weights stay the read-only base.

The proposed compiler input for a session field is a public data flow. The flow chooses which data moves, in what order, toward the fusion. The adapter that comes out stays re-derivable from that public flow. A hidden delta is still forbidden.

Compilation from public written rules remains valid. That path is not, by itself, the session taste field described in the Loom note.
