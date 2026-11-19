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
from .cross_isa import (
    canonicalize_output_factors,
    compile_betaedit_cross_isa,
    det_cholesky_solve,
    det_householder_qr,
    det_jacobi_svd,
    ieee754_binary_sum,
    ozaki_exact_matmul,
)
from .gguf_lora import (
    GGUFLoraAdapter,
    build_llama_server_lora_flags,
    read_gguf_lora,
    read_gguf_model_geometry,
    write_gguf_lora,
)
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
    "build_llama_server_lora_flags",
    "canonicalize_output_factors",
    "check_flake_stability",
    "compile_betaedit_cross_isa",
    "dequantize_array",
    "dequantize_int4",
    "dequantize_int8",
    "det_cholesky_solve",
    "det_householder_qr",
    "det_jacobi_svd",
    "evaluate_adapter_gate",
    "ieee754_binary_sum",
    "lexicase_select",
    "ozaki_exact_matmul",
    "quantize_bf16",
    "quantize_f16",
    "quantize_int4",
    "quantize_int8",
    "read_gguf_lora",
    "read_gguf_model_geometry",
    "read_slab_wire",
    "verify_binding",
    "verify_rederivation",
    "verify_slab_wire",
    "write_gguf_lora",
    "write_slab_wire",
]

__version__ = "0.1.0"
