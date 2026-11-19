"""Weft: Siliconcordat Layer 2 deterministic weight compiler and verification harness."""

from .anchors import (
    AnchorCase,
    AnchorSuite,
    GateVerdict,
    VerifyResult,
    check_flake_stability,
    evaluate_adapter_gate,
    lexicase_select,
)
from .binding import (
    ADAPTER_CLASS_1,
    ADAPTER_CLASS_2,
    BindingManifest,
    RuntimeFlags,
    build_binding_manifest,
    verify_binding,
)
from .certificate import FixedRadixCertificate, GatePolicy, build_certificate
from .compiler import BetaEditConfig, BetaEditNullspaceProjector, StagedPatch, verify_rederivation
from .gguf_lora import GGUFLoraAdapter, read_gguf_lora, write_gguf_lora
from .quant import (
    dequantize_array,
    dequantize_int4,
    dequantize_int8,
    quantize_bf16,
    quantize_f16,
    quantize_int4,
    quantize_int8,
)
from .slab_wire import (
    WEFT_SLAB_FORMAT,
    SlabWireContainer,
    read_slab_wire,
    verify_slab_wire,
    write_slab_wire,
)

__all__ = [
    "ADAPTER_CLASS_1",
    "ADAPTER_CLASS_2",
    "AnchorCase",
    "AnchorSuite",
    "BetaEditConfig",
    "BetaEditNullspaceProjector",
    "BindingManifest",
    "FixedRadixCertificate",
    "GGUFLoraAdapter",
    "GatePolicy",
    "GateVerdict",
    "RuntimeFlags",
    "SlabWireContainer",
    "StagedPatch",
    "VerifyResult",
    "WEFT_SLAB_FORMAT",
    "build_binding_manifest",
    "build_certificate",
    "check_flake_stability",
    "dequantize_array",
    "dequantize_int4",
    "dequantize_int8",
    "evaluate_adapter_gate",
    "lexicase_select",
    "quantize_bf16",
    "quantize_f16",
    "quantize_int4",
    "quantize_int8",
    "read_gguf_lora",
    "read_slab_wire",
    "verify_binding",
    "verify_rederivation",
    "verify_slab_wire",
    "write_gguf_lora",
    "write_slab_wire",
]

__version__ = "0.1.0"
