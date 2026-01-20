# Detailed Code Reviews - Tuner

**Repository:** [nod-ai/amd-shark-ai](https://github.com/nod-ai/amd-shark-ai)

**Total PRs: 64**

**Generated:** 2026-01-20 00:28:32

---


## [PR #2775](https://github.com/nod-ai/amd-shark-ai/pull/2775): [tuner] add gitignore patterns for boo tuner artifacts

### Review Summary

**APPROVED** (2026-01-16)

**COMMENTED** (2026-01-16)


### Code Comments

**File:** `amdsharktuner/.gitignore:3`

```diff
@@ -2,3 +2,5 @@
 
 # Tuning artifacts
```

**Comment:**
```suggestion
# Tuning artifacts.
```

---


---


## [PR #2771](https://github.com/nod-ai/amd-shark-ai/pull/2771): [Tuner] Refactor: Extract rocm_common.py from common.py (2/n)

### Review Summary

**COMMENTED** (2026-01-14)

**COMMENTED** (2026-01-14)

**APPROVED** (2026-01-14)


### Code Comments

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_common.py:121`

```diff
@@ -0,0 +1,239 @@
+# Copyright 2026 Advanced Micro Devices, Inc.
+#
+# Licensed under the Apache License v2.0 with LLVM Exceptions.
+# See https://llvm.org/LICENSE.txt for license information.
+# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+from dataclasses import dataclass
+from typing import Optional
+
+from iree.compiler import ir  # type: ignore
+from iree.compiler.dialects import iree_gpu  # type: ignore
+
+from .. import common
+
+
+# The Key name for the 'amdgpu-waves-per-eu' within the llvm_func_attrs attribute.
+WAVES_PER_EU_KEY = "amdgpu-waves-per-eu"
+
+
+@dataclass
+class LLVMGPUVectorDistributeContractionKnobs(common.KnobAssignment):
+    # Problem Size.
+    M: int
+    N: int
+    K: int
+
+    # Z3 numeric selections.
+    tile_m: int
+    tile_n: int
+    tile_k: int
+    wg_x: int
+    wg_y: int
+    wg_z: int
+    subgroup_m_cnt: int
+    subgroup_n_cnt: int
+    intrinsic_mn: int
+    intrinsic_k: int
+    subgroup_m: int
+    subgroup_n: int
+    subgroup_k: int
+
+
+@dataclass
+class ConvolutionKnobs(common.KnobAssignment):
+    pass
+
+
+@dataclass
+class AttentionKnobs(common.KnobAssignment):
+    pass
+
+
+def get_compatible_mma_intrinsics(
+    lhs_type: common.ShapedType,
+    rhs_type: common.ShapedType,
+    res_type: common.ShapedType,
+    mma_intrinsics: list[iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic],
+    allow_virtual_mma: bool = False,
+) -> list[iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic]:
+    def is_compatible(
+        mma: iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic,
+    ) -> bool:
+        # Filter out virtual intrinsics unless explicitly allowed (for attention ops).
+        is_virtual = isinstance(mma, iree_gpu.VirtualMMAIntrinsic)
+        if is_virtual and not allow_virtual_mma:
+            return False
+
+        mma_attr = (
+            iree_gpu.VirtualMMAAttr.get(mma)
+            if is_virtual
+            else iree_gpu.MMAAttr.get(mma)
+        )
+        a_type, b_type, c_type = mma_attr.abc_element_types
+        return (
+            lhs_type.element_type == a_type
+            and rhs_type.element_type == b_type
+            and res_type.element_type == c_type
+        )
+
+    return list(filter(is_compatible, mma_intrinsics))
+
+
+# Generate a config dictionary used in translation_info attribute.
+def get_translation_info_config(
+    pipeline_options: iree_gpu.PipelineOptionsAttr, waves_per_eu: int
+) -> ir.DictAttr:
+    """
+    Example IR
+    translation_info = #iree_codegen.translation_info<
+                    pipeline = LLVMGPUVectorDistribute workgroup_size = [512, 1, 1] subgroup_size = 64,
+                    {gpu_pipeline_options = #iree_gpu.pipeline_options<...>,
+                     llvm_func_attrs = {"amdgpu-waves-per-eu" = "3"}
+                    }
+                >
+    """
+    waves_per_eu_str = str(waves_per_eu)
+
+    # Create the waves_per_eu dictionary attribute.
+    waves_per_eu_dict = ir.DictAttr.get(
+        {WAVES_PER_EU_KEY: ir.StringAttr.get(waves_per_eu_str)}
+    )
+
+    config_dict = ir.DictAttr.get(
+        {
+            common.GPU_PIPELINE_OPTIONS_KEY: pipeline_options,
+            common.LLVM_FUNC_ATTRS_KEY: waves_per_eu_dict,
+        }
+    )
+
+    return config_dict
+
+
+def get_attention_decomposition_config(
+    tuner_ctx: common.TunerContext,
+    qk_lowering_config: iree_gpu.LoweringConfigAttr,
+    pv_lowering_config: iree_gpu.LoweringConfigAttr,
+) -> ir.DictAttr:
+    """
+    Constructs the decomposition config for an attention op, embedding
+    separate lowering configs for QK and PV matmuls.
+    """
```

**Comment:**
Isn't this code generic?

---

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_common.py:121`

```diff
@@ -0,0 +1,239 @@
+# Copyright 2026 Advanced Micro Devices, Inc.
+#
+# Licensed under the Apache License v2.0 with LLVM Exceptions.
+# See https://llvm.org/LICENSE.txt for license information.
+# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+from dataclasses import dataclass
+from typing import Optional
+
+from iree.compiler import ir  # type: ignore
+from iree.compiler.dialects import iree_gpu  # type: ignore
+
+from .. import common
+
+
+# The Key name for the 'amdgpu-waves-per-eu' within the llvm_func_attrs attribute.
+WAVES_PER_EU_KEY = "amdgpu-waves-per-eu"
+
+
+@dataclass
+class LLVMGPUVectorDistributeContractionKnobs(common.KnobAssignment):
+    # Problem Size.
+    M: int
+    N: int
+    K: int
+
+    # Z3 numeric selections.
+    tile_m: int
+    tile_n: int
+    tile_k: int
+    wg_x: int
+    wg_y: int
+    wg_z: int
+    subgroup_m_cnt: int
+    subgroup_n_cnt: int
+    intrinsic_mn: int
+    intrinsic_k: int
+    subgroup_m: int
+    subgroup_n: int
+    subgroup_k: int
+
+
+@dataclass
+class ConvolutionKnobs(common.KnobAssignment):
+    pass
+
+
+@dataclass
+class AttentionKnobs(common.KnobAssignment):
+    pass
+
+
+def get_compatible_mma_intrinsics(
+    lhs_type: common.ShapedType,
+    rhs_type: common.ShapedType,
+    res_type: common.ShapedType,
+    mma_intrinsics: list[iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic],
+    allow_virtual_mma: bool = False,
+) -> list[iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic]:
+    def is_compatible(
+        mma: iree_gpu.MMAIntrinsic | iree_gpu.VirtualMMAIntrinsic,
+    ) -> bool:
+        # Filter out virtual intrinsics unless explicitly allowed (for attention ops).
+        is_virtual = isinstance(mma, iree_gpu.VirtualMMAIntrinsic)
+        if is_virtual and not allow_virtual_mma:
+            return False
+
+        mma_attr = (
+            iree_gpu.VirtualMMAAttr.get(mma)
+            if is_virtual
+            else iree_gpu.MMAAttr.get(mma)
+        )
+        a_type, b_type, c_type = mma_attr.abc_element_types
+        return (
+            lhs_type.element_type == a_type
+            and rhs_type.element_type == b_type
+            and res_type.element_type == c_type
+        )
+
+    return list(filter(is_compatible, mma_intrinsics))
+
+
+# Generate a config dictionary used in translation_info attribute.
+def get_translation_info_config(
+    pipeline_options: iree_gpu.PipelineOptionsAttr, waves_per_eu: int
+) -> ir.DictAttr:
+    """
+    Example IR
+    translation_info = #iree_codegen.translation_info<
+                    pipeline = LLVMGPUVectorDistribute workgroup_size = [512, 1, 1] subgroup_size = 64,
+                    {gpu_pipeline_options = #iree_gpu.pipeline_options<...>,
+                     llvm_func_attrs = {"amdgpu-waves-per-eu" = "3"}
+                    }
+                >
+    """
+    waves_per_eu_str = str(waves_per_eu)
+
+    # Create the waves_per_eu dictionary attribute.
+    waves_per_eu_dict = ir.DictAttr.get(
+        {WAVES_PER_EU_KEY: ir.StringAttr.get(waves_per_eu_str)}
+    )
+
+    config_dict = ir.DictAttr.get(
+        {
+            common.GPU_PIPELINE_OPTIONS_KEY: pipeline_options,
+            common.LLVM_FUNC_ATTRS_KEY: waves_per_eu_dict,
+        }
+    )
+
+    return config_dict
+
+
+def get_attention_decomposition_config(
+    tuner_ctx: common.TunerContext,
+    qk_lowering_config: iree_gpu.LoweringConfigAttr,
+    pv_lowering_config: iree_gpu.LoweringConfigAttr,
+) -> ir.DictAttr:
+    """
+    Constructs the decomposition config for an attention op, embedding
+    separate lowering configs for QK and PV matmuls.
+    """
```

**Comment:**
Ah no, it uses iree_gpu lowering config, so it may be gpu-generic at best.

---


---


## [PR #2769](https://github.com/nod-ai/amd-shark-ai/pull/2769): [Tuner] Refactor: Move dispatch_constraints to rocm subdirectory (1/n)

### Review Summary

**COMMENTED** (2026-01-14)

**APPROVED** (2026-01-14)

Thanks


### Code Comments

**File:** `amdsharktuner/tests/rocm/rocm_dispatch_constraints_test.py:9`

```diff
@@ -1,11 +1,11 @@
-# Copyright 2024 Advanced Micro Devices, Inc.
+# Copyright 2026 Advanced Micro Devices, Inc.
 #
 # Licensed under the Apache License v2.0 with LLVM Exceptions.
 # See https://llvm.org/LICENSE.txt for license information.
 # SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 
 """
-Usage: python -m pytest dispatch_constraints_test.py
+Usage: python -m pytest tests/rocm/rocm_dispatch_constraints_test.py
 """
```

**Comment:**
We don't need this usage notes in tests -- all tests are supposed to be executed like this and the README explains it. We can drop it.

---


---


## [PR #2768](https://github.com/nod-ai/amd-shark-ai/pull/2768): [Tuner] Use isinstance() for AffineExpr type checking

### Review Summary

**APPROVED** (2026-01-13)

Thanks for cleaning this up


---


## [PR #2763](https://github.com/nod-ai/amd-shark-ai/pull/2763): [Tuner] Refactor to separate ROCM-specific code into rocm subdirectory

### Review Summary

**CHANGES_REQUESTED** (2026-01-12)

Can you split it up into a few smaller PRs? Nearly 3 kLOC is a lot to review, even if this is mostly code motion.


---


## [PR #2747](https://github.com/nod-ai/amd-shark-ai/pull/2747): [tuner] use virtual mma only for attention op

### Review Summary

**COMMENTED** (2025-12-29)

Why not handle this in `get_compatible_mfma_intrinsics`? It should be easier to test.

**COMMENTED** (2025-12-29)

**APPROVED** (2025-12-29)


### Code Comments

**File:** `amdsharktuner/amdsharktuner/dispatch_constraints.py:88`

```diff
@@ -74,10 +74,18 @@ def get_mfma_intrinsic_constraints(
     lhs_layout: MMASingleSubgroupLayout | None = None,
     rhs_layout: MMASingleSubgroupLayout | None = None,
     acc_layout: MMASingleSubgroupLayout | None = None,
+    allow_virtual_mma: bool = False,
 ) -> z3.BoolRef:
     compatible_intrinsics = common.get_compatible_mfma_intrinsics(
         lhs_type, rhs_type, res_type, mma_intrinsics
     )
+    # Filter out virtual intrinsics unless explicitly allowed (e.g., for attention ops).
+    if not allow_virtual_mma:
+        compatible_intrinsics = [
+            instr
+            for instr in compatible_intrinsics
+            if isinstance(instr, iree_gpu.MMAIntrinsic)
+        ]
```

**Comment:**
```suggestion
        compatible_intrinsics = filter(lambda x: isinstance(x, iree_gpu.MMAIntrinsic), compatible_intrinsics)
```

---

**File:** `amdsharktuner/tests/dispatch_constraints_test.py:426`

```diff
@@ -408,6 +408,34 @@ def test_get_mfma_intrinsic_constraints(
     k_val = model[intrinsic_k].as_long()
     assert (m_val, n_val, k_val) in [(16, 16, 16), (32, 32, 8)]
 
+    lhs_type = common.ShapedType([32, 16], tuner_ctx.type.f8E4M3FNUZ)
+    rhs_type = common.ShapedType([16, 32], tuner_ctx.type.f8E4M3FNUZ)
+    res_type = common.ShapedType([32, 32], tuner_ctx.type.f32)
+
+    constraints = dispatch_constraints.get_mma_intrinsic_constraints(
+        lhs_type=lhs_type,
+        rhs_type=rhs_type,
+        res_type=res_type,
+        intrinsic_m=intrinsic_m,
+        intrinsic_n=intrinsic_n,
+        intrinsic_k=intrinsic_k,
+        mma_intrinsics=[
+            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
+            iree_gpu.VirtualMMAIntrinsic.VMFMA_F32_32x32x16_F8E4M3FNUZ,
+        ],
+        allow_virtual_mma=True,
```

**Comment:**
Do we have any tests that exercises `allow_virtual_mma=False`? IE, does anything fail if you always pass `allow_virtual_mma=False`?

---


---


## [PR #2744](https://github.com/nod-ai/amd-shark-ai/pull/2744): [tuner] ensure correct number of TD specs are generated

### Review Summary

**COMMENTED** (2025-12-29)

**CHANGES_REQUESTED** (2025-12-29)

**COMMENTED** (2025-12-29)

**APPROVED** (2025-12-29)


### Code Comments

**File:** `amdsharktuner/amdsharktuner/libtuner.py:958`

```diff
@@ -939,7 +952,10 @@ def generate_candidate_specs(
             candidate_ordering.build_tuning_records_from_order(knobs, sorted_order)
         )
 
-        knob_assignments = [dispatch_tuner.get_knob_assignment(s) for s in solutions]
+        knob_assignments = build_knob_assignments_with_baseline(
+            dispatch_tuner, solutions
+        )
+        assert len(config_specs) == len(knob_assignments)
```

**Comment:**
I don't think we need a helper function for this -- this can be a s simple as prepending a `None`.

---

**File:** `amdsharktuner/tests/libtuner_test.py:450`

```diff
@@ -425,3 +425,30 @@ def test_compute_rocprof_avg_kernel_time(caplog):
     trace_rows = [drop_row] * 10 + [cal_row] * 5 + [cal_row_2] * 5
     avg_us = libtuner.compute_rocprof_avg_kernel_time(trace_rows)
     assert avg_us == pytest.approx(1.75)
+
+
+def test_build_knob_assignments_with_baseline():
+    """Test that knob assignments list is correctly constructed with baseline at index 0."""
+
+    knob1 = {"tile": 64}
+    knob2 = {"tile": 128}
+    knob3 = {"tile": 256}
+
+    class TestDispatchTuner:
+        def get_knob_assignment(self, solution):
+            return solution
+
+    dispatch_tuner = TestDispatchTuner()
+    solutions = [knob1, knob2, knob3]
+
+    result = libtuner.build_knob_assignments_with_baseline(dispatch_tuner, solutions)
+
+    assert len(result) == 4
+    assert result[0] is None
+    assert result[1] == knob1
+    assert result[2] == knob2
+    assert result[3] == knob3
```

**Comment:**
You can compare lists

---

**File:** `amdsharktuner/tests/libtuner_test.py:454`

```diff
@@ -425,3 +425,30 @@ def test_compute_rocprof_avg_kernel_time(caplog):
     trace_rows = [drop_row] * 10 + [cal_row] * 5 + [cal_row_2] * 5
     avg_us = libtuner.compute_rocprof_avg_kernel_time(trace_rows)
     assert avg_us == pytest.approx(1.75)
+
+
+def test_build_knob_assignments_with_baseline():
+    """Test that knob assignments list is correctly constructed with baseline at index 0."""
+
+    knob1 = {"tile": 64}
+    knob2 = {"tile": 128}
+    knob3 = {"tile": 256}
+
+    class TestDispatchTuner:
+        def get_knob_assignment(self, solution):
+            return solution
+
+    dispatch_tuner = TestDispatchTuner()
+    solutions = [knob1, knob2, knob3]
+
+    result = libtuner.build_knob_assignments_with_baseline(dispatch_tuner, solutions)
+
+    assert len(result) == 4
+    assert result[0] is None
+    assert result[1] == knob1
+    assert result[2] == knob2
+    assert result[3] == knob3
+
+    empty_result = libtuner.build_knob_assignments_with_baseline(dispatch_tuner, [])
+    assert len(empty_result) == 1
+    assert empty_result[0] is None
```

**Comment:**
also here

---

**File:** `amdsharktuner/amdsharktuner/libtuner.py:943`

```diff
@@ -939,7 +939,10 @@ def generate_candidate_specs(
             candidate_ordering.build_tuning_records_from_order(knobs, sorted_order)
         )
 
-        knob_assignments = [dispatch_tuner.get_knob_assignment(s) for s in solutions]
+        knob_assignments = [None] + [
```

**Comment:**
Can you add a one-line comment explaining the None element is for the baseline?

---

**File:** `amdsharktuner/tests/libtuner_test.py:452`

```diff
@@ -425,3 +425,28 @@ def test_compute_rocprof_avg_kernel_time(caplog):
     trace_rows = [drop_row] * 10 + [cal_row] * 5 + [cal_row_2] * 5
     avg_us = libtuner.compute_rocprof_avg_kernel_time(trace_rows)
     assert avg_us == pytest.approx(1.75)
+
+
+def test_knob_assignments_with_baseline():
+    """Test that knob assignments list is correctly constructed with baseline at index 0."""
+
+    knob1 = {"tile": 64}
+    knob2 = {"tile": 128}
+    knob3 = {"tile": 256}
+
+    class TestDispatchTuner:
+        def get_knob_assignment(self, solution):
+            return solution
+
+    dispatch_tuner = TestDispatchTuner()
+    solutions = [knob1, knob2, knob3]
+
+    knob_assignments = [None] + [
+        dispatch_tuner.get_knob_assignment(s) for s in solutions
+    ]
+    assert knob_assignments == [None, knob1, knob2, knob3]
+
+    empty_knob_assignments = [None] + [
+        dispatch_tuner.get_knob_assignment(s) for s in []
+    ]
+    assert empty_knob_assignments == [None]
```

**Comment:**
This doesn't test anything

---

**File:** `amdsharktuner/tests/libtuner_test.py:452`

```diff
@@ -425,3 +425,28 @@ def test_compute_rocprof_avg_kernel_time(caplog):
     trace_rows = [drop_row] * 10 + [cal_row] * 5 + [cal_row_2] * 5
     avg_us = libtuner.compute_rocprof_avg_kernel_time(trace_rows)
     assert avg_us == pytest.approx(1.75)
+
+
+def test_knob_assignments_with_baseline():
+    """Test that knob assignments list is correctly constructed with baseline at index 0."""
+
+    knob1 = {"tile": 64}
+    knob2 = {"tile": 128}
+    knob3 = {"tile": 256}
+
+    class TestDispatchTuner:
+        def get_knob_assignment(self, solution):
+            return solution
+
+    dispatch_tuner = TestDispatchTuner()
+    solutions = [knob1, knob2, knob3]
+
+    knob_assignments = [None] + [
+        dispatch_tuner.get_knob_assignment(s) for s in solutions
+    ]
+    assert knob_assignments == [None, knob1, knob2, knob3]
+
+    empty_knob_assignments = [None] + [
+        dispatch_tuner.get_knob_assignment(s) for s in []
+    ]
+    assert empty_knob_assignments == [None]
```

**Comment:**
This test doesn't exercise any of the tuner code. If you change the tuner code, the test won't catch anything.

---


---


## [PR #2736](https://github.com/nod-ai/amd-shark-ai/pull/2736): [tuner] enable igemm support for all conv layouts

### Review Summary

**CHANGES_REQUESTED** (2025-12-29)

**COMMENTED** (2025-12-30)

**COMMENTED** (2025-12-30)

**CHANGES_REQUESTED** (2026-01-09)


### Code Comments

**File:** `amdsharktuner/amdsharktuner/candidate_gen.py:230`

```diff
@@ -226,8 +222,12 @@ def set_dispatch_tuner(
     dispatch_tuner: Optional[DispatchTuner] = None
     for tuner_class in dispatch_tuners:
         if tuner_class.supports_root_op(root_op):
-            tuner = tuner_class(root_op, tuner_ctx)
-            dispatch_tuner = tuner
+            if tuner_class is ConvolutionOpInterfaceTuner:
+                dispatch_tuner = ConvolutionOpInterfaceTuner(
+                    root_op, tuner_ctx, conv_lowering_strategy=conv_lowering_strategy
+                )
+            else:
+                dispatch_tuner = tuner_class(root_op, tuner_ctx)
```

**Comment:**
This function shouldn't know about concrete tuner classes. Instead, could we have a property for the conv strategy and set it externally?

---

**File:** `amdsharktuner/amdsharktuner/dispatch_parser.py:198`

```diff
@@ -187,6 +187,16 @@ def get_iter_dim_size(
         tensor_dim = list(indexing_map.results).index(ir.AffineExpr.get_dim(iter_dim))
         return operand_type.shape[tensor_dim]
 
+    @property
+    def conv_lowering_strategy(self) -> Optional[common.ConvLoweringStrategy]:
+        """Convolution lowering strategy. Only meaningful for convolution tuners."""
+        return None
+
+    @conv_lowering_strategy.setter
+    def conv_lowering_strategy(self, value: common.ConvLoweringStrategy) -> None:
+        """Set convolution lowering strategy. No-op for non-convolution tuners."""
+        pass
```

**Comment:**
This doesn't belong in the abstract base -- not all dispatch parsers even know what a convolution is. I think the key design requirement should be that, in the future, we can tune across multiple conv lowering strategies within the same tuner context. This suggests to me that this should be stored somewhere outside of the parser.

---

**File:** `amdsharktuner/amdsharktuner/dispatch_parser.py:198`

```diff
@@ -187,6 +187,16 @@ def get_iter_dim_size(
         tensor_dim = list(indexing_map.results).index(ir.AffineExpr.get_dim(iter_dim))
         return operand_type.shape[tensor_dim]
 
+    @property
+    def conv_lowering_strategy(self) -> Optional[common.ConvLoweringStrategy]:
+        """Convolution lowering strategy. Only meaningful for convolution tuners."""
+        return None
+
+    @conv_lowering_strategy.setter
+    def conv_lowering_strategy(self, value: common.ConvLoweringStrategy) -> None:
+        """Set convolution lowering strategy. No-op for non-convolution tuners."""
+        pass
```

**Comment:**
Or alternatively, maybe we want to have two conv dispatch parsers, and allow for multiple parsers to matcha a single root op?

---

**File:** `amdsharktuner/amdsharktuner/candidate_gen.py:138`

```diff
@@ -122,28 +122,26 @@ def get_knob_assignment(
         return config_list[0].knob_assignment
 
 
-class ConvolutionOpInterfaceTuner(
-    DispatchTuner, dispatch_parser.ConvolutionOpInterfaceParser
-):
-    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
-        super().__init__(root_op, tuner_ctx)
+class ConvolutionOpInterfaceTunerBase(DispatchTuner):
+    """Base class for convolution tuners. Subclasses use specific lowering strategies."""
 
     @classmethod
-    def supports_root_op(cls, root_op: ir.Operation) -> bool:
-        if not linalg.isa_convolution_op(root_op):
-            return False
-        convolution_dims = linalg.infer_convolution_dimensions(root_op)
-        if not convolution_dims:
-            return False
-        # Only allow 'nhwc_hwcf' convs.
-        return (
-            list(convolution_dims.batch) == [0]
-            and list(convolution_dims.output_image) == [1, 2]
-            and list(convolution_dims.output_channel) == [3]
-            and list(convolution_dims.filter_loop) == [4, 5]
-            and list(convolution_dims.input_channel) == [6]
-            and list(convolution_dims.depth) == []
-        )
+    def get_tuner_for_strategy(
+        cls, strategy: common.ConvLoweringStrategy
+    ) -> type["ConvolutionOpInterfaceTunerBase"]:
+        strategy_to_tuner: dict[
+            common.ConvLoweringStrategy, type[ConvolutionOpInterfaceTunerBase]
+        ] = {
+            common.ConvLoweringStrategy.IGEMM: IGEMMConvolutionTuner,
+            common.ConvLoweringStrategy.INNER_MNK: InnerMNKConvolutionTuner,
+        }
+        return strategy_to_tuner[strategy]
```

**Comment:**
The base class shouldn't know about the derived classes -- can we make this a free function instead?

---

**File:** `amdsharktuner/amdsharktuner/candidate_gen.py:265`

```diff
@@ -201,12 +237,30 @@ def get_knob_assignment(
         return None
 
 
+def get_conv_lowering_strategy_for_pipeline(
+    codegen_pipeline: common.CodegenPipelines,
+) -> common.ConvLoweringStrategy:
+    """Get the appropriate convolution lowering strategy for the given pipeline.
+
+    IGEMM only works with TileAndFuse, INNER_MNK only works with VectorDistribute.
+
+    TODO(Bangtian): When direct conv support is added, expose this as a CLI arg
+    (e.g., --conv-lowering-strategy) to allow users to choose between igemm and direct.
+    """
+    if codegen_pipeline == common.CodegenPipelines.llvmgpu_tile_and_fuse:
+        return common.ConvLoweringStrategy.IGEMM
+    return common.ConvLoweringStrategy.INNER_MNK
+
+
 def set_dispatch_tuner(
-    input_module: ir.Module, tuner_ctx: common.TunerContext
+    input_module: ir.Module,
+    tuner_ctx: common.TunerContext,
+    codegen_pipeline: common.CodegenPipelines = common.CodegenPipelines.llvmgpu_tile_and_fuse,
 ) -> Optional[DispatchTuner]:
+    conv_lowering_strategy = get_conv_lowering_strategy_for_pipeline(codegen_pipeline)
     dispatch_tuners: list[type[DispatchTuner]] = [
         ContractionOpInterfaceTuner,
-        ConvolutionOpInterfaceTuner,
+        ConvolutionOpInterfaceTunerBase.get_tuner_for_strategy(conv_lowering_strategy),
         AttentionOpInterfaceTuner,
     ]
```

**Comment:**
I'm concerned we are leaking codegen pipeline and conv strategies to this generic code. The layering seems off to me.

---


---


## [PR #2713](https://github.com/nod-ai/amd-shark-ai/pull/2713): [tuner]: sync the change of using prefetch_num_stages to replace prefetch_shared_memory

### Review Summary

**COMMENTED** (2025-12-05)

**APPROVED** (2025-12-06)


### Code Comments

**File:** `amdsharktuner/amdsharktuner/libtuner.py:378`

```diff
@@ -372,10 +372,10 @@ def parse_arguments(
         "--tile-dims", help="Map of tile size matmul dims", type=str, default="mnk"
     )
     candidate_gen_args.add_argument(
-        "--prefetch-shared-memory-options",
-        type=lambda t: [s.strip().lower() == "true" for s in t.split(",")],
-        default=[True],
-        help="Comma-separated list of allowed values for the prefetch_shared_memory pipeline option. Possible values: [True, False]",
+        "--prefetch-num-stages-options",
+        type=lambda t: [int(s.strip()) for s in t.split(",")],
+        default=[2],
+        help="Comma-separated list of allowed values for prefetch_num_stages pipeline option. Values: 0/1 = don't prefetch, 2 = default prefetch, 3 = new option.",
```

**Comment:**
Why `3 = new option`? Can you describe what 3+ means instead?

---


---


## [PR #2701](https://github.com/nod-ai/amd-shark-ai/pull/2701): [tuner] add padding_conv attribute along IGEMM supprot for conv

### Review Summary

**COMMENTED** (2025-12-03)

**APPROVED** (2025-12-12)


### Code Comments

**File:** `amdsharktuner/amdsharktuner/common.py:255`

```diff
@@ -233,6 +250,24 @@ class AttentionKnobs(KnobAssignment):
     pass
 
 
+def is_affine_expr_function_of_dim(expr: ir.AffineExpr, position: int) -> bool:
+    """
+    Return True if the expression depends on the dimension at the given position.
```

**Comment:**
Can you add some example?

---

**File:** `amdsharktuner/tests/common_test.py:578`

```diff
@@ -567,3 +567,167 @@ def test_calculate_padded_dimensions(
         assert M_padded == [200], f"Expected M not padded, got {M_padded}"
         assert N_padded == [300], f"Expected N not padded, got {N_padded}"
         assert padding_applied == False
+
+
+def test_is_affine_expr_function_of_dim(tuner_ctx: common.TunerContext) -> None:
+    with tuner_ctx.mlir_ctx:
+        d0 = ir.AffineDimExpr.get(0)
+        d1 = ir.AffineDimExpr.get(1)
+
+        assert common.is_affine_expr_function_of_dim(d0, 0) == True
+        assert common.is_affine_expr_function_of_dim(d0, 1) == False
```

**Comment:**
```suggestion
        assert common.is_affine_expr_function_of_dim(d0, 0)
        assert not common.is_affine_expr_function_of_dim(d0, 1)
```
also below

---

**File:** `amdsharktuner/tests/common_test.py:598`

```diff
@@ -567,3 +567,167 @@ def test_calculate_padded_dimensions(
         assert M_padded == [200], f"Expected M not padded, got {M_padded}"
         assert N_padded == [300], f"Expected N not padded, got {N_padded}"
         assert padding_applied == False
+
+
+def test_is_affine_expr_function_of_dim(tuner_ctx: common.TunerContext) -> None:
+    with tuner_ctx.mlir_ctx:
+        d0 = ir.AffineDimExpr.get(0)
+        d1 = ir.AffineDimExpr.get(1)
+
+        assert common.is_affine_expr_function_of_dim(d0, 0) == True
+        assert common.is_affine_expr_function_of_dim(d0, 1) == False
+
+        c42 = ir.AffineConstantExpr.get(42)
+        assert common.is_affine_expr_function_of_dim(c42, 0) == False
+        assert common.is_affine_expr_function_of_dim(c42, 1) == False
+
+        add_expr = d0 + d1
+        assert common.is_affine_expr_function_of_dim(add_expr, 0) == True
+        assert common.is_affine_expr_function_of_dim(add_expr, 1) == True
+
+        mul_expr = d1 * 2
+        assert common.is_affine_expr_function_of_dim(mul_expr, 0) == False
+        assert common.is_affine_expr_function_of_dim(mul_expr, 1) == True
+
+        complex_expr = (d0 + d1) * 2
+        assert common.is_affine_expr_function_of_dim(complex_expr, 0) == True
+        assert common.is_affine_expr_function_of_dim(complex_expr, 1) == True
+
+
+def test_get_padding_conv_sizes(tuner_ctx: common.TunerContext) -> None:
+    from types import SimpleNamespace
```

**Comment:**
Can you add a comment explaining why we need this?

---

**File:** `amdsharktuner/tests/constraint_generator_test.py:301`

```diff
@@ -295,6 +295,10 @@ def test_generate_solutions_tile_and_fuse_contraction_padding(
             assert "padding =" in str(
                 lowering_config
             ), f"Missing padding in lowering config: {lowering_config}"
+            # padding_conv only for convolutions, not contractions.
+            assert "padding_conv =" not in str(
+                lowering_config
+            ), f"Unexpected padding_conv in non-convolution lowering config: {lowering_config}"
```

**Comment:**
Doesn't pytest print expected and actual values?

---

**File:** `amdsharktuner/tests/dispatch_parser_test.py:396`

```diff
@@ -388,3 +388,63 @@ def test_get_attention_operation(tuner_ctx: common.TunerContext) -> None:
     assert result.k1_dims == [2]
     assert result.k2_dims == [3]
     assert result.n_dims == [4]
+
+
+def test_build_conv_to_igemm_info(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    module_str = """
+        builtin.module{
```

**Comment:**
```suggestion
        builtin.module {
```

---

**File:** `amdsharktuner/tests/dispatch_parser_test.py:400`

```diff
@@ -388,3 +388,63 @@ def test_get_attention_operation(tuner_ctx: common.TunerContext) -> None:
     assert result.k1_dims == [2]
     assert result.k2_dims == [3]
     assert result.n_dims == [4]
+
+
+def test_build_conv_to_igemm_info(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    module_str = """
+        builtin.module{
+            func.func @test(%arg0: tensor<2x34x34x16xf16>, %arg1: tensor<3x3x16x32xf16>) -> tensor<2x32x32x32xf32> {
+                %cst = arith.constant 0.000000e+00 : f32
+                %0 = tensor.empty() : tensor<2x32x32x32xf32>
+                %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x32x32x32xf32>) -> tensor<2x32x32x32xf32>
```

**Comment:**
You can make the output be another function argument

---


---


## [PR #2698](https://github.com/nod-ai/amd-shark-ai/pull/2698): [tuner] update the calculation of shared memory usage

### Review Summary

**COMMENTED** (2025-11-30)

**APPROVED** (2025-12-01)

**COMMENTED** (2025-12-02)


### Code Comments

**File:** `amdsharktuner/amdsharktuner/dispatch_constraints.py:174`

```diff
@@ -161,17 +161,37 @@ def get_dispatch_constraints(
 def calculate_shared_memory_usage_in_bytes(
     lhs_type: common.ShapedType,
     rhs_type: common.ShapedType,
+    res_type: common.ShapedType,
     m: list[int] | list[z3.ArithRef],
     n: list[int] | list[z3.ArithRef],
     k: list[int] | list[z3.ArithRef],
+    promote_operands: list[int] = [0, 1],
 ) -> int | z3.ArithRef:
+    assert promote_operands == [0, 1] or promote_operands == [
+        0,
+        1,
+        2,
+    ], f"Got {promote_operands}"
```

**Comment:**
This formatting is really weird, isn't there any way to keep the second array on a single line?
For example, you could do something like:

```py
supported_promotions = ([0, 1], [0, 1, 2])
assert promote_operands in supported_promotions
```

---

**File:** `amdsharktuner/amdsharktuner/dispatch_constraints.py:192`

```diff
@@ -161,17 +161,37 @@ def get_dispatch_constraints(
 def calculate_shared_memory_usage_in_bytes(
     lhs_type: common.ShapedType,
     rhs_type: common.ShapedType,
+    res_type: common.ShapedType,
     m: list[int] | list[z3.ArithRef],
     n: list[int] | list[z3.ArithRef],
     k: list[int] | list[z3.ArithRef],
+    promote_operands: list[int] = [0, 1],
 ) -> int | z3.ArithRef:
+    assert promote_operands == [0, 1] or promote_operands == [
+        0,
+        1,
+        2,
+    ], f"Got {promote_operands}"
+
     lhs_memory = lhs_type.bitwidth // 8
     for size in m + k:
         lhs_memory *= size
+
     rhs_memory = rhs_type.bitwidth // 8
     for size in n + k:
         rhs_memory *= size
-    return lhs_memory + rhs_memory
+
+    output_memory = res_type.bitwidth // 8
+    for size in m + n:
+        output_memory *= size
+
+    total_memory = (
+        int(0 in promote_operands) * lhs_memory
+        + int(1 in promote_operands) * rhs_memory
+        + int(2 in promote_operands) * output_memory
+    )
```

**Comment:**
Can we replace this trick with a few if statements?

---

**File:** `amdsharktuner/amdsharktuner/dispatch_constraints.py:171`

```diff
@@ -161,17 +161,36 @@ def get_dispatch_constraints(
 def calculate_shared_memory_usage_in_bytes(
     lhs_type: common.ShapedType,
     rhs_type: common.ShapedType,
+    res_type: common.ShapedType,
     m: list[int] | list[z3.ArithRef],
     n: list[int] | list[z3.ArithRef],
     k: list[int] | list[z3.ArithRef],
+    promote_operands: list[int] = [0, 1],
 ) -> int | z3.ArithRef:
+    supported_promotions = ([0, 1], [0, 1, 2])
+    assert promote_operands in supported_promotions, f"Got {promote_operands}"
```

**Comment:**
+1, especially as we start looking at NN and TN variants

---


---


## [PR #2692](https://github.com/nod-ai/amd-shark-ai/pull/2692): [tuner] Sync Padding for TileAndFuse with IREE changes

### Review Summary

**COMMENTED** (2025-11-30)

Is this something that we would rather keep entirely in IREE and expose as new bindings?

**COMMENTED** (2025-12-01)

**APPROVED** (2025-12-01)

LGTM but consider updating the PR title: 'revisit' does not really convey what's changing, I'd call it something like 'Sync padding for TileAndFuse with IREE changes'


### Code Comments

**File:** `amdsharktuner/tests/constraint_generator_test.py:357`

```diff
@@ -353,27 +353,31 @@ def test_generate_solutions_tile_and_fuse_conv_padding(
             )
         )
 
-        assert len(solutions) > 0, "No solutions generated with TileAndFuse pipeline."
-        for solution in solutions:
-            assert len(solution) == 1, f"Expected a single-item list, got: {solution}"
-            config = solution[0]
-            assert isinstance(
-                config, common.TuningConfiguration
-            ), f"Expected TuningConfiguration, got: {type(config)}"
-
-            assert (
-                config.name == "compilation_info"
-            ), f"Expected key 'compilation_info', got: {config.name}"
-            assert isinstance(
-                config.configuration, iree_codegen.CompilationInfoAttr
-            ), f"Expected CompilationInfoAttr, got: {type(config.configuration)}"
-
-            lowering_config = config.configuration.lowering_config
-            assert "padding =" in str(
-                lowering_config
-            ), f"Missing padding in lowering config: {lowering_config}"
-            promote = [int(x) for x in lowering_config.attributes["promote_operands"]]
-            assert promote == [0, 1, 2]
+        if len(solutions) > 0:
+            for solution in solutions:
```

**Comment:**
why are you guarding a for loop with an if condition?

---

**File:** `amdsharktuner/amdsharktuner/common.py:575`

```diff
@@ -523,3 +523,62 @@ def get_target_info(input_module: ir.Module) -> iree_gpu.TargetInfo:
     target = executable_variant_op.target
 
     return iree_gpu.TargetInfo.get_gpu_target_info(target)
+
+
+# The following two functions are from IREE side for padding utility:
+# https://github.com/iree-org/iree/blob/8ae91ebb0e555e660b8a6898f6071476f7a1f20b/compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp#L631-L671
+def maybe_padded_bounds(original_bound: int, alignment: int) -> tuple[int, bool]:
+    remainder = original_bound % alignment
+    if remainder == 0:
+        return original_bound, False
+    return original_bound + alignment - remainder, True
+
+
+def get_dim_bounds(
+    dims: list[int],
+    padding_can_be_expensive: bool,
+) -> tuple[list[int], bool]:
+    result = []
+    any_padding_applied = False
+
+    for dim in dims:
+        if padding_can_be_expensive:
+            result.append(dim)
+            continue
+
+        # TODO: Make over-padding a tunable parameter. This logic allows over-padding to get larger
+        # tile sizes, which may result in better performance despite doing more padded computation.
+        if dim > 128:
+            padded, was_padded = maybe_padded_bounds(dim, 128)
+            result.append(padded)
+            any_padding_applied = any_padding_applied or was_padded
+        elif dim > 32:
+            padded, was_padded = maybe_padded_bounds(dim, 32)
+            result.append(padded)
+            any_padding_applied = any_padding_applied or was_padded
+        else:
+            result.append(dim)
+
+    return result, any_padding_applied
+
+
+# Use padding logic from IREE side:
+# https://github.com/iree-org/iree/blob/8ae91ebb0e555e660b8a6898f6071476f7a1f20b/compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp#L691-L703
+def calculate_padded_dimensions(
+    M: list[int],
+    N: list[int],
+    contraction_dims: ContractionDimensions,
+    contraction_maps: list[ir.AffineMap],
+) -> tuple[list[int], list[int], bool]:
```

**Comment:**
Can you explain what is returned?

---

**File:** `amdsharktuner/amdsharktuner/common.py:540`

```diff
@@ -523,3 +523,62 @@ def get_target_info(input_module: ir.Module) -> iree_gpu.TargetInfo:
     target = executable_variant_op.target
 
     return iree_gpu.TargetInfo.get_gpu_target_info(target)
+
+
+# The following two functions are from IREE side for padding utility:
+# https://github.com/iree-org/iree/blob/8ae91ebb0e555e660b8a6898f6071476f7a1f20b/compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp#L631-L671
+def maybe_padded_bounds(original_bound: int, alignment: int) -> tuple[int, bool]:
+    remainder = original_bound % alignment
+    if remainder == 0:
+        return original_bound, False
+    return original_bound + alignment - remainder, True
+
+
+def get_dim_bounds(
+    dims: list[int],
+    padding_can_be_expensive: bool,
+) -> tuple[list[int], bool]:
```

**Comment:**
Also here: what does this do?

---

**File:** `amdsharktuner/amdsharktuner/common.py:530`

```diff
@@ -523,3 +523,62 @@ def get_target_info(input_module: ir.Module) -> iree_gpu.TargetInfo:
     target = executable_variant_op.target
 
     return iree_gpu.TargetInfo.get_gpu_target_info(target)
+
+
+# The following two functions are from IREE side for padding utility:
+# https://github.com/iree-org/iree/blob/8ae91ebb0e555e660b8a6898f6071476f7a1f20b/compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp#L631-L671
+def maybe_padded_bounds(original_bound: int, alignment: int) -> tuple[int, bool]:
```

**Comment:**
also here -- could you explain what is being returned?

---

**File:** `amdsharktuner/amdsharktuner/common.py:530`

```diff
@@ -523,3 +523,62 @@ def get_target_info(input_module: ir.Module) -> iree_gpu.TargetInfo:
     target = executable_variant_op.target
 
     return iree_gpu.TargetInfo.get_gpu_target_info(target)
+
+
+# The following two functions are from IREE side for padding utility:
+# https://github.com/iree-org/iree/blob/8ae91ebb0e555e660b8a6898f6071476f7a1f20b/compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp#L631-L671
+def maybe_padded_bounds(original_bound: int, alignment: int) -> tuple[int, bool]:
```

**Comment:**
we should start function names with active verbs

---

**File:** `amdsharktuner/amdsharktuner/common.py:547`

```diff
@@ -523,3 +523,62 @@ def get_target_info(input_module: ir.Module) -> iree_gpu.TargetInfo:
     target = executable_variant_op.target
 
     return iree_gpu.TargetInfo.get_gpu_target_info(target)
+
+
+# The following two functions are from IREE side for padding utility:
+# https://github.com/iree-org/iree/blob/8ae91ebb0e555e660b8a6898f6071476f7a1f20b/compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp#L631-L671
+def maybe_padded_bounds(original_bound: int, alignment: int) -> tuple[int, bool]:
+    remainder = original_bound % alignment
+    if remainder == 0:
+        return original_bound, False
+    return original_bound + alignment - remainder, True
+
+
+def get_dim_bounds(
+    dims: list[int],
+    padding_can_be_expensive: bool,
+) -> tuple[list[int], bool]:
+    result = []
+    any_padding_applied = False
+
+    for dim in dims:
+        if padding_can_be_expensive:
+            result.append(dim)
+            continue
```

**Comment:**
Why not hoist this check outside of the loop?

---

**File:** `amdsharktuner/amdsharktuner/common.py:552`

```diff
@@ -523,3 +523,74 @@ def get_target_info(input_module: ir.Module) -> iree_gpu.TargetInfo:
     target = executable_variant_op.target
 
     return iree_gpu.TargetInfo.get_gpu_target_info(target)
+
+
+# The following two functions are from IREE side for padding utility:
+# https://github.com/iree-org/iree/blob/8ae91ebb0e555e660b8a6898f6071476f7a1f20b/compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp#L631-L671
+def compute_next_aligned_bound(original_bound: int, alignment: int) -> int:
+    """Pads a bound up to the next multiple of alignment if needed.
+
+    Returns:
+        The original bound if already aligned, or the next multiple of alignment.
+    """
+    remainder = original_bound % alignment
+    if remainder == 0:
+        return original_bound
+    return original_bound + alignment - remainder
+
+
+def get_dim_bounds(
+    dims: list[int],
+    padding_can_be_expensive: bool,
+) -> list[int]:
+    """Computes padded dimension bounds for better tile alignment.
+
+    Returns:
+        List of dimensions, potentially padded to alignment boundaries.
+    """
+    if padding_can_be_expensive:
+        return list(dims)
```

**Comment:**
dims is already a list

---


---


## [PR #2683](https://github.com/nod-ai/amd-shark-ai/pull/2683): [tuner] use igemm bindings

### Review Summary

**COMMENTED** (2025-11-21)

**COMMENTED** (2025-11-21)

**APPROVED** (2025-11-21)

LGTM % nit


### Code Comments

**File:** `amdsharktuner/amdsharktuner/constraint_generator.py:59`

```diff
@@ -31,7 +32,38 @@ def adjust_problem_size_for_pipeline(
 
     pipeline_options_search_space.use_igemm_convolution = [True]
 
-    # Flatten the K dimensions into a single dimension for IGEMM lowering.
+    # Use IGEMM binding details if available for accurate dimension mapping.
+    if igemm_details:
+        igemm_maps = [
+            map_attr.value for map_attr in igemm_details.igemm_contraction_maps
+        ]
+        igemm_contraction_dims = linalg.infer_contraction_dimensions_from_maps(
+            igemm_maps
+        )
+        assert (
+            igemm_contraction_dims
+        ), "Failed to infer contraction dimensions from IGEMM maps"
+
+        bounds = list(igemm_details.igemm_loop_bounds)
+
+        # Update contraction_dims with IGEMM structure.
+        contraction_dims.m = list(igemm_contraction_dims.m)
+        contraction_dims.n = list(igemm_contraction_dims.n)
+        contraction_dims.k = list(igemm_contraction_dims.k)
+        contraction_dims.batch = list(igemm_contraction_dims.batch)
+
+        # Update matmul_size with IGEMM loop bounds (K is already flattened!).
+        matmul_size.M = [bounds[i] for i in contraction_dims.m]
+        matmul_size.N = [bounds[i] for i in contraction_dims.n]
+        matmul_size.K = [bounds[i] for i in contraction_dims.k]
+        print(f"matmul_size.K: {matmul_size.K}")
```

**Comment:**
Drop debug prints

---

**File:** `amdsharktuner/amdsharktuner/dispatch_parser.py:76`

```diff
@@ -72,6 +72,9 @@ class ConvolutionOpInfo(OpInfo):
     strides: list[int]
     dilations: list[int]
 
+    # IGEMM details for TileAndFuse pipeline (None if not available).
+    igemm_details: Any = None
```

**Comment:**
What is the type? Using `Any` effectively sidesteps any type checking

---

**File:** `amdsharktuner/amdsharktuner/dispatch_parser.py:273`

```diff
@@ -267,6 +270,11 @@ def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
         rhs_type = root_op.operands[1].type
         res_type = root_op.operands[2].type
 
+        # Get IGEMM details for potential use with TileAndFuse pipeline.
```

**Comment:**
```suggestion
        # Get IGEMM details for potential use with the TileAndFuse pipeline.
```

---

**File:** `amdsharktuner/tests/constraint_generator_test.py:396`

```diff
@@ -391,7 +391,9 @@ def test_adjust_problem_size_for_pipeline(
         k=[3],
         batch=[0],
     )
-    taf_pipeline = iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse
+    tile_and_fuse_pipeline = (
+        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse
+    )
```

**Comment:**
Can we use this constant directly instead of creating a local variable for it? I think it only hurts readability here

---

**File:** `amdsharktuner/tests/constraint_generator_test.py:460`

```diff
@@ -438,8 +440,104 @@ def test_adjust_problem_size_for_pipeline(
         matmul_size=conv_size,
         dispatch_kind=common.DispatchKind.conv,
         pipeline_options_search_space=pipeline_options_space,
-        codegen_pipeline=taf_pipeline,
+        codegen_pipeline=tile_and_fuse_pipeline,
     )
     assert pipeline_options_space.use_igemm_convolution == [True]
     assert conv_size.K == [4608]
     assert conv_dims.k == [4]
+
+
+def test_adjust_problem_size_for_pipeline_with_igemm_details(
+    tuner_ctx: common.TunerContext,
+) -> None:
+    """Test adjust_problem_size_for_pipeline with IGEMM details from the binding."""
+    context = tuner_ctx.mlir_ctx
+    f16 = tuner_ctx.type.f16
+    f32 = tuner_ctx.type.f32
+
+    input_shape = (2, 32, 32, 128)
+    kernel_shape = (3, 3, 128, 256)
+    output_shape = (2, 30, 30, 256)
```

**Comment:**
Can we move these inside the `with` statement, since they are not used anywhere else?

---

**File:** `amdsharktuner/tests/constraint_generator_test.py:493`

```diff
@@ -438,8 +440,104 @@ def test_adjust_problem_size_for_pipeline(
         matmul_size=conv_size,
         dispatch_kind=common.DispatchKind.conv,
         pipeline_options_search_space=pipeline_options_space,
-        codegen_pipeline=taf_pipeline,
+        codegen_pipeline=tile_and_fuse_pipeline,
     )
     assert pipeline_options_space.use_igemm_convolution == [True]
     assert conv_size.K == [4608]
     assert conv_dims.k == [4]
+
+
+def test_adjust_problem_size_for_pipeline_with_igemm_details(
+    tuner_ctx: common.TunerContext,
+) -> None:
+    """Test adjust_problem_size_for_pipeline with IGEMM details from the binding."""
+    context = tuner_ctx.mlir_ctx
+    f16 = tuner_ctx.type.f16
+    f32 = tuner_ctx.type.f32
+
+    input_shape = (2, 32, 32, 128)
+    kernel_shape = (3, 3, 128, 256)
+    output_shape = (2, 30, 30, 256)
+
+    with ir.Location.unknown(context):
+        module = ir.Module.create()
+        build_func_with_conv2d_nhwc_hwcf(
+            module=module,
+            input_shape=input_shape,
+            kernel_shape=kernel_shape,
+            output_shape=output_shape,
+            input_type=f16,
+            kernel_type=f16,
+            output_type=f32,
+        )
+
+        root_op_list = iree_codegen.get_tuner_root_ops(module)
+        assert len(root_op_list) == 1
+        root_op = root_op_list[0]
+
+        parser = dispatch_parser.ConvolutionOpInterfaceParser(root_op, tuner_ctx)
+        conv_op_info = parser.get_op_info()
+        assert isinstance(conv_op_info, dispatch_parser.ConvolutionOpInfo)
+        assert (
+            conv_op_info.igemm_details is not None
+        ), "IGEMM details should be available for NHWC conv"
+
+        conv_dims = common.ContractionDimensions(
+            m=list(conv_op_info.dims.m),
+            n=list(conv_op_info.dims.n),
+            k=list(conv_op_info.dims.k),
+            batch=list(conv_op_info.dims.batch),
+        )
+        conv_size = common.ContractionSizes(
+            M=list(conv_op_info.matmul_size.M),
+            N=list(conv_op_info.matmul_size.N),
+            K=list(conv_op_info.matmul_size.K),
+            B=list(conv_op_info.matmul_size.B),
```

**Comment:**
Would it help to assert that what the returned dimensions and sizes are? I think it won't ever change, so we don't have to worry about these getting out of sync with the compiler.

---

**File:** `amdsharktuner/tests/constraint_generator_test.py:461`

```diff
@@ -438,8 +434,112 @@ def test_adjust_problem_size_for_pipeline(
         matmul_size=conv_size,
         dispatch_kind=common.DispatchKind.conv,
         pipeline_options_search_space=pipeline_options_space,
-        codegen_pipeline=taf_pipeline,
+        codegen_pipeline=iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse,
     )
     assert pipeline_options_space.use_igemm_convolution == [True]
     assert conv_size.K == [4608]
     assert conv_dims.k == [4]
+
+
+def test_adjust_problem_size_for_pipeline_with_igemm_details(
+    tuner_ctx: common.TunerContext,
+) -> None:
+    """Test adjust_problem_size_for_pipeline with IGEMM details from the binding."""
+    context = tuner_ctx.mlir_ctx
+
+    with ir.Location.unknown(context):
+        f16 = tuner_ctx.type.f16
+        f32 = tuner_ctx.type.f32
+
+        input_shape = (2, 32, 32, 128)
+        kernel_shape = (3, 3, 128, 256)
+        output_shape = (2, 30, 30, 256)
+
+        module = ir.Module.create()
+        build_func_with_conv2d_nhwc_hwcf(
+            module=module,
+            input_shape=input_shape,
+            kernel_shape=kernel_shape,
+            output_shape=output_shape,
+            input_type=f16,
+            kernel_type=f16,
+            output_type=f32,
+        )
+
```

**Comment:**
I'd inline these shapes here, if you are not going to use these variables anywhere else

---


---


## [PR #2656](https://github.com/nod-ai/amd-shark-ai/pull/2656): [tuner][nfc] clean up the code

### Review Summary

**APPROVED** (2025-11-10)


---


## [PR #2641](https://github.com/nod-ai/amd-shark-ai/pull/2641): [tuner] Create dedicated IREE requirements file for sharktuner

### Review Summary

**COMMENTED** (2025-11-04)

**COMMENTED** (2025-11-04)

**COMMENTED** (2025-11-04)

**COMMENTED** (2025-11-04)

We probably also have to update READMEs

**COMMENTED** (2025-11-04)

**APPROVED** (2025-11-04)


### Code Comments

**File:** `requirements-iree-unpinned.txt:2`

```diff
@@ -1,6 +1,4 @@
 # Unpinned versions of IREE dependencies.
-wave-lang
```

**Comment:**
I think we may want to have our own requirements for sharktuner independent of what the rest of shark-ai uses

---

**File:** `requirements-iree-unpinned.txt:2`

```diff
@@ -1,6 +1,4 @@
 # Unpinned versions of IREE dependencies.
-wave-lang
```

**Comment:**
https://github.com/nod-ai/shark-ai/pull/2641#discussion_r2491522614

---

**File:** `requirements-iree-sharktuner.txt:1`

```diff
@@ -0,0 +1,6 @@
+# Unpinned IREE dependencies for sharktuner.
```

**Comment:**
Should we move it inside the shartuner dir?

---

**File:** `requirements-iree-sharktuner.txt:1`

```diff
@@ -0,0 +1,6 @@
+# Unpinned IREE dependencies for sharktuner.
```

**Comment:**
Can we delete this file now?

---


---


## [PR #2627](https://github.com/nod-ai/amd-shark-ai/pull/2627): [tuner] exits gracefully for unsupported cases like mat-vec operations

### Review Summary

**COMMENTED** (2025-11-03)

**COMMENTED** (2025-11-04)

**APPROVED** (2025-11-04)

LGTM!


### Code Comments

**File:** `sharktuner/sharktuner/candidate_gen.py:110`

```diff
@@ -96,9 +96,25 @@ def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
 
     @classmethod
     def supports_root_op(cls, root_op: ir.Operation) -> bool:
-        return linalg.isa_contraction_op(root_op) and not linalg.isa_convolution_op(
-            root_op
-        )
+        if not linalg.isa_contraction_op(root_op):
+            return False
+
+        # Check if contraction has valid dimensions.
+        contraction_dims = linalg.infer_contraction_dimensions(root_op)
+        if not contraction_dims:
+            logging.warning("No contraction dimensions found for operation")
+            return False
+
+        if not contraction_dims.m or not contraction_dims.n or not contraction_dims.k:
+            logging.warning(
+                f"Contraction operation has missing or empty dimensions: "
```

**Comment:**
This warning makes it sounds like something is wrong with the input. Instead, we should print that this contraction type is not supported by the tuner yet.

---

**File:** `sharktuner/tests/candidate_gen_test.py:234`

```diff
@@ -215,3 +215,144 @@ def test_get_td_spec_convolution(tuner_ctx: common.TunerContext) -> None:
         "gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false>"
         in matcher_sequence_str
     )
+
+
+def test_set_dispatch_tuner_with_matvec(tuner_ctx: common.TunerContext) -> None:
+    """
+    This test is added to indicate that mat-vec can be recognized as a contraction,
+    but returns False due to empty dimensions, and then set_dispatch_tuner returns None.
+    The tuner will not crash and exits gracefully.
+    Once the tuner supports mat-vec, this test can be removed.
+    """
+    context = tuner_ctx.mlir_ctx
+    module_str = """
+        builtin.module{
+            func.func @test(%x: tensor<8xbf16>, %A: tensor<8x224xbf16>) -> tensor<224xf32> {
+                %zero = arith.constant 0.0 : f32
+                %init = tensor.empty() : tensor<224xf32>
+                %y0 = linalg.fill ins(%zero : f32) outs(%init : tensor<224xf32>) -> tensor<224xf32>
+                %y = linalg.generic {
```

**Comment:**
Can we use named linalg ops like `linalg.matmul` or `linalg.matvec` to simplify this?

---

**File:** `sharktuner/tests/candidate_gen_test.py:234`

```diff
@@ -215,3 +215,144 @@ def test_get_td_spec_convolution(tuner_ctx: common.TunerContext) -> None:
         "gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false>"
         in matcher_sequence_str
     )
+
+
+def test_set_dispatch_tuner_with_matvec(tuner_ctx: common.TunerContext) -> None:
+    """
+    This test is added to indicate that mat-vec can be recognized as a contraction,
+    but returns False due to empty dimensions, and then set_dispatch_tuner returns None.
+    The tuner will not crash and exits gracefully.
+    Once the tuner supports mat-vec, this test can be removed.
+    """
+    context = tuner_ctx.mlir_ctx
+    module_str = """
+        builtin.module{
+            func.func @test(%x: tensor<8xbf16>, %A: tensor<8x224xbf16>) -> tensor<224xf32> {
+                %zero = arith.constant 0.0 : f32
+                %init = tensor.empty() : tensor<224xf32>
+                %y0 = linalg.fill ins(%zero : f32) outs(%init : tensor<224xf32>) -> tensor<224xf32>
+                %y = linalg.generic {
```

**Comment:**
Also in the other tests.

---

**File:** `sharktuner/sharktuner/candidate_gen.py:258`

```diff
@@ -237,7 +252,12 @@ def set_dispatch_tuner(
             dispatch_tuner = tuner
             break
 
-    assert dispatch_tuner, "No suitable dispatch tuner found"
+    if not dispatch_tuner:
+        tune_logger.error(
+            "No suitable dispatch tuner found for the root operation. "
+            "The operation may have unsupported characteristics (e.g., missing dimensions)."
```

**Comment:**
```suggestion
            "The operation may not be supported by the tuner yet."
```

---

**File:** `sharktuner/tests/candidate_gen_test.py:226`

```diff
@@ -215,3 +215,101 @@ def test_get_td_spec_convolution(tuner_ctx: common.TunerContext) -> None:
         "gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false>"
         in matcher_sequence_str
     )
+
+
+def test_set_dispatch_tuner_with_matvec(tuner_ctx: common.TunerContext) -> None:
+    """
+    This test is added to indicate that mat-vec can be recognized as a contraction,
+    but returns False due to empty dimensions, and then set_dispatch_tuner returns None.
+    The tuner will not crash and exits gracefully.
+    Once the tuner supports mat-vec, this test can be removed.
+    """
```

**Comment:**
I would make this more concise:

```suggestion
    # Make sure we do not crash on unsupported root ops (matvec).
```

---

**File:** `sharktuner/tests/candidate_gen_test.py:255`

```diff
@@ -215,3 +215,101 @@ def test_get_td_spec_convolution(tuner_ctx: common.TunerContext) -> None:
         "gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false>"
         in matcher_sequence_str
     )
+
+
+def test_set_dispatch_tuner_with_matvec(tuner_ctx: common.TunerContext) -> None:
+    """
+    This test is added to indicate that mat-vec can be recognized as a contraction,
+    but returns False due to empty dimensions, and then set_dispatch_tuner returns None.
+    The tuner will not crash and exits gracefully.
+    Once the tuner supports mat-vec, this test can be removed.
+    """
+    context = tuner_ctx.mlir_ctx
+    module_str = """
+        builtin.module{
+            func.func @test(%A: tensor<8x224xf32>, %x: tensor<224xf32>) -> tensor<8xf32> {
+                %init = tensor.empty() : tensor<8xf32>
+                %y = linalg.matvec {root_op}
+                    ins(%A, %x : tensor<8x224xf32>, tensor<224xf32>)
+                    outs(%init : tensor<8xf32>) -> tensor<8xf32>
+                return %y : tensor<8xf32>
+            }
+        }"""
+
+    ir_module = ir.Module.parse(module_str, context)
+
+    # Should return None since mat-vec has invalid dimensions (M=[]).
+    result = candidate_gen.set_dispatch_tuner(ir_module, tuner_ctx)
+    assert result is None
+
+
+def test_set_dispatch_tuner_with_unsupported_conv(
+    tuner_ctx: common.TunerContext,
+) -> None:
+    """
+    Test that set_dispatch_tuner returns None for conv ops with unsupported layouts.
+
+    The tuner currently only supports conv with nhwc_hwcf layout. Conv ops with other
+    layouts (e.g., nchw_fchw) should be rejected gracefully without crashing.
+    Once the tuner supports additional layouts, this test can be updated or removed.
+    """
```

**Comment:**
similar here

---


---


## [PR #2621](https://github.com/nod-ai/amd-shark-ai/pull/2621): [tuner][nfc]: merge imports and update the readme

### Review Summary

**COMMENTED** (2025-10-31)

Thanks for the cleanup

**APPROVED** (2025-10-31)


### Code Comments

**File:** `sharktuner/tests/libtuner_test.py:283`

```diff
@@ -280,7 +280,7 @@ def test_baseline_result_handler_is_better_than_baseline():
         libtuner.BenchmarkResult(1, 2.5, "hip://slow1"),  # slower than 2.0.
         libtuner.BenchmarkResult(
             3, 2.0, "hip://slow2"
-        ),  # slower than fallback ≈ 1.1667
+        ),  # slower than fallback ≈ 1.1667.
```

**Comment:**
Can we also remove this non-ascii character?

---

**File:** `sharktuner/README.md:25`

```diff
@@ -22,7 +22,7 @@ source .venv/bin/activate
 **Development dependencies:**
 ```shell
 pip install -r requirements-dev.txt
-pip install -r requirements-test.txt
```

**Comment:**
We should rename the file instead -- the convention is to use `requirements-test.txt`

---


---


## [PR #2613](https://github.com/nod-ai/amd-shark-ai/pull/2613): [tuner] set prefetch shared memory option based on layout matching for attention ops

### Review Summary

**APPROVED** (2025-10-28)


---


## [PR #2596](https://github.com/nod-ai/amd-shark-ai/pull/2596): [tuner] use python binding to build td specs for attention

### Review Summary

**COMMENTED** (2025-10-27)

**COMMENTED** (2025-10-27)

**APPROVED** (2025-10-27)


### Code Comments

**File:** `sharktuner/sharktuner/constraint_generator.py:576`

```diff
@@ -583,88 +566,18 @@ class AttentionOpInterfaceConstraintGenerator(ConstraintGenerator):
     - O  : [B, M, N]
 
     Attributes:
-        transposed_q (bool): True if Q is logically transposed (k1 dim is not last in map).
-        transposed_k (bool): True if K is logically transposed (k1 dim is not last in map).
-        transposed_v (bool): True if V is logically transposed (k2 dim is not last in map).
-        qk_matmul (MatmulShapeType): Shape metadata for Q @ K^T.
-        pv_matmul (MatmulShapeType): Shape metadata for P @ V.
-        opinfo: dimensions info for attention op.
+        op_info (AttentionOpInfo): Contains all attention operation metadata including:
+            - transposed_q (bool): True if Q is logically transposed (k1 dim is not last in map).
+            - transposed_k (bool): True if K is logically transposed (k1 dim is not last in map).
+            - transposed_v (bool): True if V is logically transposed (k2 dim is not last in map).
+            - qk_matmul (MatmulShapeType): Shape metadata for Q @ K^T.
+            - pv_matmul (MatmulShapeType): Shape metadata for P @ V.
+            - domain_rank, batch_dims, m_dims, n_dims, k1_dims, k2_dims: Dimension indices.
+            - batch_sizes, m_sizes, n_sizes, k1_sizes, k2_sizes: Dimension sizes.
```

**Comment:**
Should we move this comment down to `class AttentionOpInfo(OpInfo):`?

---

**File:** `sharktuner/sharktuner/dispatch_parser.py:110`

```diff
@@ -73,6 +73,69 @@ class ConvolutionOpInfo(OpInfo):
     dilations: list[int]
 
 
+@dataclass
+class AttentionOpInfo(OpInfo):
+    """Information about an attention operation.
+
+    Attention is decomposed into two matrix multiplications:
+    - QK^T : Q @ K^T  (attention scores)
+    - PV   : P @ V    (projected output after softmax)
+
+    Assumed operand shapes:
+    - Q  : [B, M, K1]
+    - K  : [B, K2, K1]
+    - V  : [B, K2, N]
+    - O  : [B, M, N]
+
+    Attributes:
+        domain_rank: Total number of dimensions in the operation.
+        batch_dims: Indices of batch dimensions.
+        m_dims: Indices of M dimensions (query sequence length).
+        n_dims: Indices of N dimensions (output/value dimensions).
+        k1_dims: Indices of K1 dimensions (query/key feature dimensions).
+        k2_dims: Indices of K2 dimensions (key sequence length).
+        batch_sizes: Sizes of batch dimensions.
+        m_sizes: Sizes of M dimensions.
+        n_sizes: Sizes of N dimensions.
+        k1_sizes: Sizes of K1 dimensions.
+        k2_sizes: Sizes of K2 dimensions.
+        query_type: MLIR type of query tensor.
+        key_type: MLIR type of key tensor.
+        value_type: MLIR type of value tensor.
+        output_type: MLIR type of output tensor.
+        transposed_q: True if Q is logically transposed (k1 dim is not last in map).
+        transposed_k: True if K is logically transposed (k1 dim is not last in map).
+        transposed_v: True if V is logically transposed (k2 dim is not last in map).
+        qk_matmul: Shape metadata for Q @ K^T matmul.
+        pv_matmul: Shape metadata for P @ V matmul.
```

**Comment:**
Can we put them inline next to / on top of each property? This way it should be easier to keep it up to date and make sure the documentation matches the code. Probably also easier to read.

---


---


## [PR #2592](https://github.com/nod-ai/amd-shark-ai/pull/2592): [tuner] support phase-aware candidate pruning

### Review Summary

**COMMENTED** (2025-10-24)

**CHANGES_REQUESTED** (2025-10-24)

This doesn't look like a correct fix -- we can print a warning for sure, but it's still possible that even a slower dispatch candidate is faster than the baseline across the full model. We should still select top N best candidates and continue to model compilation and benchmarking next.

**COMMENTED** (2025-10-25)

Can we implement this in a way that allows us to write unit tests?

**COMMENTED** (2025-10-27)

**COMMENTED** (2025-10-28)

**COMMENTED** (2025-10-28)

**COMMENTED** (2025-10-28)

**APPROVED** (2025-10-28)


### Code Comments

**File:** `sharktuner/model_tuner/model_tuner.py:169`

```diff
@@ -164,6 +164,12 @@ def main() -> None:
             args.model_tuner_num_dispatch_candidates,
             args.dispatch_benchmark_timeout_mins,
         )
+        if not top_candidates:
+            logging.warning(
+                "No tuning specs to return: no dispatch candidates outperformed the baseline."
```

**Comment:**
```suggestion
                "No tuning specs to return: no dispatch candidate outperformed the baseline."
```

---

**File:** `sharktuner/sharktuner/libtuner.py:1269`

```diff

```

**Comment:**
Instead of relying on mocks for this test, could we add a function that takes `candidate_results` and the `baseline_handler` and decides which candidates to keep and which to drop?

I'd like to avoid mocks for things that can executed using the intended logic. The issue with mocks is that they require you to assume what a valid implementation may do and what its state is without exercising the surrounding logic, hence my preference is towards solutions that reduce the amount of state and don't have to mock. Some related articles that talks about this:
* https://the-dext.github.io/TDD_Mocking-vs-No-Mocking/
* https://www.robertopiva.pro/2017/02/15/the-mock-excuse.html

---

**File:** `sharktuner/sharktuner/libtuner.py:1219`

```diff
@@ -1191,6 +1201,27 @@ def compile(
     return compiled_candidates
 
 
+def filter_candidates_by_baseline(
+    candidate_results: list[BenchmarkResult],
+    baseline_handler: BaselineResultHandler,
+    num_candidates: int,
+    should_prune_slower_candidates: bool,
+) -> list[tuple[BenchmarkResult, float]]:
+    """
+    Filters and orders candidates based on baseline comparison.
+    """
+    if not baseline_handler.is_better_than_baseline(candidate_results):
+        if should_prune_slower_candidates:
+            return []
+
+    all_candidates_with_speedup = baseline_handler.get_candidates_ordered_by_speedup(
+        candidate_results
+    )
```

**Comment:**
Have you considered passing a second parameter to this function, `should_prune`? I'm wondering if we need a new function at all

---

**File:** `sharktuner/sharktuner/libtuner.py:1222`

```diff
@@ -1191,6 +1201,27 @@ def compile(
     return compiled_candidates
 
 
+def filter_candidates_by_baseline(
+    candidate_results: list[BenchmarkResult],
+    baseline_handler: BaselineResultHandler,
+    num_candidates: int,
+    should_prune_slower_candidates: bool,
+) -> list[tuple[BenchmarkResult, float]]:
+    """
+    Filters and orders candidates based on baseline comparison.
+    """
+    if not baseline_handler.is_better_than_baseline(candidate_results):
+        if should_prune_slower_candidates:
+            return []
+
+    all_candidates_with_speedup = baseline_handler.get_candidates_ordered_by_speedup(
+        candidate_results
+    )
+    top_candidates_with_speedup = all_candidates_with_speedup[:num_candidates]
+
+    return top_candidates_with_speedup
```

**Comment:**
Usually there's no need for local variables only used once
```suggestion
    return all_candidates_with_speedup[:num_candidates]
```

---

**File:** `sharktuner/sharktuner/libtuner.py:1085`

```diff
@@ -1070,7 +1080,9 @@ def is_better_than_baseline(self, candidate_results: list[BenchmarkResult]) -> b
         return False
 
     def get_candidates_ordered_by_speedup(
-        self, candidate_results: list[BenchmarkResult]
+        self,
+        candidate_results: list[BenchmarkResult],
+        should_prune: bool = False,
```

**Comment:**
We should update the docstring below and explain what this argument is responsible for

---

**File:** `sharktuner/sharktuner/libtuner.py:1085`

```diff
@@ -1070,7 +1080,9 @@ def is_better_than_baseline(self, candidate_results: list[BenchmarkResult]) -> b
         return False
 
     def get_candidates_ordered_by_speedup(
-        self, candidate_results: list[BenchmarkResult]
+        self,
+        candidate_results: list[BenchmarkResult],
+        should_prune: bool = False,
```

**Comment:**
I would also call it something more descriptive like `prune_slow_candidates`

---


---


## [PR #2584](https://github.com/nod-ai/amd-shark-ai/pull/2584): [tuner] use python binding to build td specs for convolution

### Review Summary

**APPROVED** (2025-10-23)


---


## [PR #2543](https://github.com/nod-ai/amd-shark-ai/pull/2543): [tuner] use python binding to build td specs for contraction

### Review Summary

**COMMENTED** (2025-10-17)

**COMMENTED** (2025-10-17)

**COMMENTED** (2025-10-17)

**COMMENTED** (2025-10-17)

**COMMENTED** (2025-10-17)

**COMMENTED** (2025-10-17)

**COMMENTED** (2025-10-17)

**COMMENTED** (2025-10-17)

**COMMENTED** (2025-10-20)

**APPROVED** (2025-10-20)


### Code Comments

**File:** `sharktuner/sharktuner/candidate_gen.py:21`

```diff
@@ -18,6 +18,7 @@
 from iree.compiler import ir  # type: ignore
 from iree.compiler.dialects import iree_codegen  # type: ignore
 from iree.compiler.dialects import iree_gpu  # type: ignore
+from iree.compiler.dialects import linalg  # type: ignore
```

**Comment:**
```suggestion
from iree.compiler.dialects import iree_codegen, iree_gpu, linalg  # type: ignore
```

---

**File:** `sharktuner/sharktuner/candidate_gen.py:139`

```diff
@@ -69,30 +76,51 @@ def find_handler(self, op_name: str) -> DispatchTuner:
 class ContractionOpInterfaceTuner(
     DispatchTuner, dispatch_parser.ContractionOpInterfaceParser
 ):
-    def __init__(self, root_op: ir.Operation):
-        super().__init__(root_op)
+    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
+        super().__init__(root_op, tuner_ctx)
+
+    @classmethod
+    def supports_root_op(cls, root_op: ir.Operation) -> bool:
+        return linalg.isa_contraction_op(root_op) and not linalg.isa_convolution_op(
+            root_op
+        )
 
     def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
         return constraint_generator.ContractionOpInterfaceConstraintGenerator(
-            self.get_root_op()
+            self.get_root_op(), self.get_op_info()
         )
 
     def get_td_spec(
         self,
         config_list: list[common.TuningConfiguration],
     ) -> ir.Module:
-        contraction_op = self.get_root_op()
-        func_name = self.get_root_op_func_name()
-        return spec_builder.build_td_spec(
-            contraction_op.context, contraction_op, config_list, func_name
+        return spec_builder.build_contraction_td_spec(
+            self._tuner_ctx, self.get_op_info(), config_list
         )
 
 
 class ConvolutionOpInterfaceTuner(
     DispatchTuner, dispatch_parser.ConvolutionOpInterfaceParser
 ):
-    def __init__(self, root_op: ir.Operation):
-        super().__init__(root_op)
+    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
+        super().__init__(root_op, tuner_ctx)
+
+    @classmethod
+    def supports_root_op(cls, root_op: ir.Operation) -> bool:
+        if not linalg.isa_convolution_op(root_op):
+            return False
+        convolution_dims = linalg.infer_convolution_dimensions(root_op)
+        if not convolution_dims:
+            return False
```

**Comment:**
When can this fail?

---

**File:** `sharktuner/sharktuner/constraint_generator.py:18`

```diff
@@ -15,6 +15,7 @@
 
 from . import common
 from . import dispatch_constraints
+from . import dispatch_parser
```

**Comment:**
```suggestion
from . import common, dispatch_constraints, dispatch_parser
```

---

**File:** `sharktuner/sharktuner/constraint_generator.py:492`

```diff
@@ -485,43 +486,14 @@ def generate_solutions(
 
 
 class ContractionOpInterfaceConstraintGenerator(ConstraintGenerator):
-    def __init__(self, root_op: ir.Operation):
+    def __init__(
+        self, root_op: ir.Operation, op_info: dispatch_parser.ContractionOpInfo
+    ):
+        # TODO (Bangtian): Both root_op and op_info are kept as a temporary solution.
```

**Comment:**
```suggestion
        # TODO(Bangtian): Both root_op and op_info are kept as a temporary solution.
```

---

**File:** `sharktuner/sharktuner/dispatch_parser.py:36`

```diff
@@ -28,40 +30,115 @@ def parse_mlir(mlir_text: str, ctx: common.TunerContext) -> ir.Module:
     return mlir_module
 
 
+@dataclass
+class OpInfo:
+    # Name of parent func.FuncOp with "match_" prefix.
+    parent_function_name: str
```

**Comment:**
Can we remove this member and query the nearest parent of type funcopinterface instead?

---

**File:** `sharktuner/sharktuner/dispatch_parser.py:59`

```diff
@@ -28,40 +30,115 @@ def parse_mlir(mlir_text: str, ctx: common.TunerContext) -> ir.Module:
     return mlir_module
 
 
+@dataclass
+class OpInfo:
+    # Name of parent func.FuncOp with "match_" prefix.
+    parent_function_name: str
+    indexing_maps: list[ir.AffineMap]
+
+
+@dataclass
+class ContractionOpInfo(OpInfo):
+    dims: common.ContractionDimensions
+    matmul_size: common.ContractionSizes
+    lhs_type: common.ShapedType
+    rhs_type: common.ShapedType
+    res_type: common.ShapedType
+
+
 class DispatchParser(metaclass=ABCMeta):
-    def __init__(self, root_op: ir.Operation):
+    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
         self._root_op = root_op
+        self._tuner_ctx = tuner_ctx
         func_op = self._root_op.parent.opview
         assert isinstance(
             func_op, func.FuncOp
         ), f"Expected func.func, got {func_op.name}"
         func_name_attr = func_op.name
-        self._func_name = f"match_{ir.StringAttr(func_name_attr).value}"
+        self._parent_function_name = f"match_{ir.StringAttr(func_name_attr).value}"
+        self._op_info: Optional[OpInfo] = None
```

**Comment:**
Why don't we store the op in op info and drop the parent function name?

---

**File:** `sharktuner/sharktuner/spec_builder.py:16`

```diff
@@ -7,8 +7,13 @@
 # Given an input dispatch, this code modifies the hyperparameters
 # in the code and runs it.
 
+import logging
+from abc import ABC, abstractmethod
+
 from iree.compiler import ir  # type: ignore
 from iree.compiler.dialects import iree_codegen  # type: ignore
+from iree.compiler.dialects import preprocessing_transform  # type: ignore
+from iree.compiler.dialects import transform  # type: ignore
```

**Comment:**
also here, can we combine these?

---

**File:** `sharktuner/sharktuner/candidate_gen.py:139`

```diff
@@ -69,30 +76,51 @@ def find_handler(self, op_name: str) -> DispatchTuner:
 class ContractionOpInterfaceTuner(
     DispatchTuner, dispatch_parser.ContractionOpInterfaceParser
 ):
-    def __init__(self, root_op: ir.Operation):
-        super().__init__(root_op)
+    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
+        super().__init__(root_op, tuner_ctx)
+
+    @classmethod
+    def supports_root_op(cls, root_op: ir.Operation) -> bool:
+        return linalg.isa_contraction_op(root_op) and not linalg.isa_convolution_op(
+            root_op
+        )
 
     def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
         return constraint_generator.ContractionOpInterfaceConstraintGenerator(
-            self.get_root_op()
+            self.get_root_op(), self.get_op_info()
         )
 
     def get_td_spec(
         self,
         config_list: list[common.TuningConfiguration],
     ) -> ir.Module:
-        contraction_op = self.get_root_op()
-        func_name = self.get_root_op_func_name()
-        return spec_builder.build_td_spec(
-            contraction_op.context, contraction_op, config_list, func_name
+        return spec_builder.build_contraction_td_spec(
+            self._tuner_ctx, self.get_op_info(), config_list
         )
 
 
 class ConvolutionOpInterfaceTuner(
     DispatchTuner, dispatch_parser.ConvolutionOpInterfaceParser
 ):
-    def __init__(self, root_op: ir.Operation):
-        super().__init__(root_op)
+    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
+        super().__init__(root_op, tuner_ctx)
+
+    @classmethod
+    def supports_root_op(cls, root_op: ir.Operation) -> bool:
+        if not linalg.isa_convolution_op(root_op):
+            return False
+        convolution_dims = linalg.infer_convolution_dimensions(root_op)
+        if not convolution_dims:
+            return False
```

**Comment:**
Right, it may fail when the output image is empty: https://github.com/llvm/llvm-project/blob/df2ff3a1b2c231f8ec78c244950687cdc54b507b/mlir/lib/Dialect/Linalg/IR/LinalgInterfaces.cpp#L784-L785.

---

**File:** `sharktuner/sharktuner/dispatch_parser.py:48`

```diff
@@ -28,40 +30,139 @@ def parse_mlir(mlir_text: str, ctx: common.TunerContext) -> ir.Module:
     return mlir_module
 
 
-class DispatchParser(metaclass=ABCMeta):
-    def __init__(self, root_op: ir.Operation):
-        self._root_op = root_op
-        func_op = self._root_op.parent.opview
+@dataclass
+class OpInfo:
+    root_op: ir.Operation
+    indexing_maps: list[ir.AffineMap]
+
+    @property
+    def parent_function_name(self) -> str:
+        """
+        Extracts the parent function name and prefixes it with "match_".
+        """
+        func_op = self.root_op.parent.opview
         assert isinstance(
             func_op, func.FuncOp
         ), f"Expected func.func, got {func_op.name}"
         func_name_attr = func_op.name
-        self._func_name = f"match_{ir.StringAttr(func_name_attr).value}"
+        return f"match_{ir.StringAttr(func_name_attr).value}"
```

**Comment:**
the `match_` prefix can be added by spec builders -- the op doesn't live in a named sequence and doesn't have to know about it

---

**File:** `sharktuner/sharktuner/dispatch_parser.py:46`

```diff
@@ -28,40 +30,139 @@ def parse_mlir(mlir_text: str, ctx: common.TunerContext) -> ir.Module:
     return mlir_module
 
 
-class DispatchParser(metaclass=ABCMeta):
-    def __init__(self, root_op: ir.Operation):
-        self._root_op = root_op
-        func_op = self._root_op.parent.opview
+@dataclass
+class OpInfo:
+    root_op: ir.Operation
+    indexing_maps: list[ir.AffineMap]
+
+    @property
+    def parent_function_name(self) -> str:
+        """
+        Extracts the parent function name and prefixes it with "match_".
+        """
+        func_op = self.root_op.parent.opview
         assert isinstance(
             func_op, func.FuncOp
         ), f"Expected func.func, got {func_op.name}"
```

**Comment:**
Can we use function interface like we do in the compiler? For example, this could also be `util.func`

---

**File:** `sharktuner/sharktuner/dispatch_parser.py:84`

```diff
@@ -28,40 +30,139 @@ def parse_mlir(mlir_text: str, ctx: common.TunerContext) -> ir.Module:
     return mlir_module
 
 
-class DispatchParser(metaclass=ABCMeta):
-    def __init__(self, root_op: ir.Operation):
-        self._root_op = root_op
-        func_op = self._root_op.parent.opview
+@dataclass
+class OpInfo:
+    root_op: ir.Operation
+    indexing_maps: list[ir.AffineMap]
+
+    @property
+    def parent_function_name(self) -> str:
+        """
+        Extracts the parent function name and prefixes it with "match_".
+        """
+        func_op = self.root_op.parent.opview
         assert isinstance(
             func_op, func.FuncOp
         ), f"Expected func.func, got {func_op.name}"
         func_name_attr = func_op.name
-        self._func_name = f"match_{ir.StringAttr(func_name_attr).value}"
+        return f"match_{ir.StringAttr(func_name_attr).value}"
+
+
+@dataclass
+class ContractionOpInfo(OpInfo):
+    dims: common.ContractionDimensions
+    matmul_size: common.ContractionSizes
+    lhs_type: common.ShapedType
+    rhs_type: common.ShapedType
+    res_type: common.ShapedType
+
+
+class DispatchParser(metaclass=ABCMeta):
+    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
+        self._root_op = root_op
+        self._tuner_ctx = tuner_ctx
+        self._op_info: Optional[OpInfo] = None
 
     def get_root_op(self) -> ir.Operation:
         return self._root_op
 
-    def get_root_op_func_name(self) -> str:
-        return self._func_name
+    def get_parent_function_name(self) -> str:
+        return (
+            self._op_info.parent_function_name
+            if self._op_info
+            else self._compute_parent_function_name()
+        )
+
+    # TODO(Bangtian): This is a temporary solution for convolution and attention ops,
+    # which don't yet implement get_op_info() and thus have _op_info set to None.
+    # Once ConvolutionOpInfo and AttentionOpInfo are implemented, this
+    # fallback method can be removed.
+    def _compute_parent_function_name(self) -> str:
+        """
+        Helper to compute parent function name before op_info is created.
+        """
+        func_op = self._root_op.parent.opview
```

**Comment:**
Can you make this a helper function that doesn't live in any specific class?  I don't see why this has to be duplicated across dispatch parser and op_info

---

**File:** `sharktuner/sharktuner/spec_builder.py:164`

```diff
@@ -184,3 +134,220 @@ def build_td_spec(
         }}
     }}"""
     return ir.Module.parse(spec_text, context)
+
+
+def get_readonly_arg_attr() -> dict[str, ir.Attribute]:
+    return {"transform.readonly": ir.UnitAttr.get()}
+
+
+def get_consumed_arg_attr() -> dict[str, ir.Attribute]:
+    return {"transform.consumed": ir.UnitAttr.get()}
+
+
+class SpecBuilder(ABC):
+    def __init__(self, op_info: OpInfo):
+        self.op_info = op_info
+
+    def create_config_params(
+        self, config_list: list[common.TuningConfiguration]
+    ) -> list[ir.Value]:
+        """
+        Creates a constant parameter for each configuration.
+        Parameters can contain #iree_codegen.compilation_info or other configuration attributes.
+        """
+        config_params = []
+        for config in config_list:
+            config_param = transform.ParamConstantOp(
+                transform.AnyParamType.get(),
+                config.configuration,
+            ).result
+            config_params.append(config_param)
```

**Comment:**
```suggestion
            config_params.append(ransform.ParamConstantOp(
                transform.AnyParamType.get(),
                config.configuration
            ).result)
```

---

**File:** `sharktuner/sharktuner/dispatch_parser.py:46`

```diff
@@ -28,40 +30,139 @@ def parse_mlir(mlir_text: str, ctx: common.TunerContext) -> ir.Module:
     return mlir_module
 
 
-class DispatchParser(metaclass=ABCMeta):
-    def __init__(self, root_op: ir.Operation):
-        self._root_op = root_op
-        func_op = self._root_op.parent.opview
+@dataclass
+class OpInfo:
+    root_op: ir.Operation
+    indexing_maps: list[ir.AffineMap]
+
+    @property
+    def parent_function_name(self) -> str:
+        """
+        Extracts the parent function name and prefixes it with "match_".
+        """
+        func_op = self.root_op.parent.opview
         assert isinstance(
             func_op, func.FuncOp
         ), f"Expected func.func, got {func_op.name}"
```

**Comment:**
ok fine to leave func.func for the time being

---

**File:** `sharktuner/sharktuner/spec_builder.py:167`

```diff
@@ -157,11 +164,12 @@ def create_config_params(
         """
         config_params = []
         for config in config_list:
-            config_param = transform.ParamConstantOp(
-                transform.AnyParamType.get(),
-                config.configuration,
-            ).result
-            config_params.append(config_param)
+            config_params.append(
```

**Comment:**
actually you can turn this whole loop into list comprehension

---

**File:** `sharktuner/sharktuner/dispatch_parser.py:27`

```diff
@@ -12,12 +12,21 @@
 from typing import Optional
 
 from iree.compiler import ir  # type: ignore
-from iree.compiler.dialects import linalg, func  # type: ignore
-from iree.compiler.dialects import iree_codegen  # type: ignore
+from iree.compiler.dialects import func, iree_codegen, linalg  # type: ignore
 
 from . import common
 
 
+def get_parent_function_name(root_op: ir.Operation) -> str:
+    """
+    Returns the parent function's symbol name from a root operation.
+    """
+    func_op = root_op.parent.opview
+    assert isinstance(func_op, func.FuncOp), f"Expected func.func, got {func_op.name}"
+    func_name_attr = func_op.name
+    return ir.StringAttr(func_name_attr).value
```

**Comment:**
nit: We don't really need local variables for stuff that's only being used once 
```suggestion
    return ir.StringAttr(func_op.name).value
```

---

**File:** `sharktuner/sharktuner/dispatch_parser.py:26`

```diff
@@ -8,14 +8,24 @@
 # in the code and runs it.
 
 from abc import ABCMeta, abstractmethod
+from dataclasses import dataclass
+from typing import Optional
 
 from iree.compiler import ir  # type: ignore
-from iree.compiler.dialects import linalg, func  # type: ignore
-from iree.compiler.dialects import iree_codegen  # type: ignore
+from iree.compiler.dialects import func, iree_codegen, linalg  # type: ignore
 
 from . import common
 
 
+def get_parent_function_name(root_op: ir.Operation) -> str:
+    """
+    Returns the parent function's symbol name from a root operation.
+    """
+    func_op = root_op.parent.opview
```

**Comment:**
How do you know the nearest parent is a function? The root op could be inside an `scr.if`, for example.  In C++ we'd check this with `op->getParentOfType<FunctionOpInterface>()`

---

**File:** `sharktuner/sharktuner/dispatch_parser.py:26`

```diff
@@ -8,14 +8,24 @@
 # in the code and runs it.
 
 from abc import ABCMeta, abstractmethod
+from dataclasses import dataclass
+from typing import Optional
 
 from iree.compiler import ir  # type: ignore
-from iree.compiler.dialects import linalg, func  # type: ignore
-from iree.compiler.dialects import iree_codegen  # type: ignore
+from iree.compiler.dialects import func, iree_codegen, linalg  # type: ignore
 
 from . import common
 
 
+def get_parent_function_name(root_op: ir.Operation) -> str:
+    """
+    Returns the parent function's symbol name from a root operation.
+    """
+    func_op = root_op.parent.opview
```

**Comment:**
We already have an assertion, so I'd leave it as `# FIXME: ...` if we can't easily query this from python

---

**File:** `sharktuner/sharktuner/candidate_gen.py:244`

```diff
@@ -207,8 +239,11 @@ def set_dispatch_tuner(input_module: ir.Module) -> DispatchTuner:
 
     dispatch_tuner: Optional[DispatchTuner] = None
     for tuner_class in dispatch_tuners:
-        tuner = tuner_class(root_op)
-        if tuner.has_valid_root_op():
+        # Check if the tuner class can handle this root op type (e.g., contraction,
+        # convolution, attention) before instantiation, since the constructor will
+        # extract op_info and will assert if the op is not supported.
```

**Comment:**
I'm not sure we need this comment -- it seems pretty clear what the code is doing

---

**File:** `sharktuner/sharktuner/spec_builder.py:27`

```diff
@@ -17,6 +20,13 @@
 ROOT_OP_ATTR_NAME = "root_op"
 
 
+def get_matcher_function_name(root_op: ir.Operation) -> str:
+    """
+    Adds the "match_" prefix to the parent function name.
+    """
+    return f"match_{get_parent_function_name(root_op)}"
```

**Comment:**
This is not a function name though, is it? A matcher is a named sequence.

---

**File:** `sharktuner/sharktuner/spec_builder.py:84`

```diff
@@ -57,82 +67,29 @@ def build_td_spec(
     if has_root_attr:
         op.opview.attributes[ROOT_OP_ATTR_NAME] = ir.UnitAttr.get(op.context)
 
-    if linalg.isa_contraction_op(op):
-        # Temporary solution using custom contraction transform ops for contraction operations.
-        inputs = op.opview.operands
-        outputs = op.opview.results
-        lhs_type = str(ir.RankedTensorType(inputs[0].type).element_type)
-        rhs_type = str(ir.RankedTensorType(inputs[1].type).element_type)
-        output_type = str(ir.RankedTensorType(outputs[0].type).element_type)
-
-        contraction_dims = linalg.infer_contraction_dimensions(op)
-        indexing_maps = linalg.get_indexing_maps(op)
-        maps = [map_attr.value for map_attr in indexing_maps]
-        lhs_dims = common.get_map_result_dim_positions(maps[0])
-        rhs_dims = common.get_map_result_dim_positions(maps[1])
-        assert lhs_dims, "no lhs dimensions"
-        assert rhs_dims, "no rhs dimensions"
-        lhs_type_shape = ir.RankedTensorType(op.operands[0].type)
-        rhs_type_shape = ir.RankedTensorType(op.operands[1].type)
-
-        m_dims = [
-            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.m
-        ]
-        n_dims = [
-            rhs_type_shape.shape[rhs_dims.index(dim)] for dim in contraction_dims.n
-        ]
-        k_dims = [
-            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.k
-        ]
-        batch_dims = [
-            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.batch
-        ]
-
-        dims_equal_checks = []
-        dims_equal_checks.append(
-            f"transform.iree.match.dims_equal %batch, {batch_dims} : !transform.param<i64>"
-        )
-        dims_equal_checks.append(
-            f"transform.iree.match.dims_equal %m, {m_dims} : !transform.param<i64>"
-        )
-        dims_equal_checks.append(
-            f"transform.iree.match.dims_equal %n, {n_dims} : !transform.param<i64>"
-        )
-        dims_equal_checks.append(
-            f"transform.iree.match.dims_equal %k, {k_dims} : !transform.param<i64>"
-        )
-        dims_equal_block = "\n            ".join(dims_equal_checks)
-
-        indexing_maps_str = ", ".join([str(map_attr) for map_attr in indexing_maps])
-        matcher_block = f"""%batch, %m, %n, %k = transform.iree.match.contraction %cont,
-                lhs_type = {lhs_type}, rhs_type = {rhs_type}, output_type = {output_type}
-                {{indexing_maps = [{indexing_maps_str}]}} :
-                !transform.any_op -> !transform.param<i64>
-            {dims_equal_block}"""
-    else:
-        # Get the names ssa names of operands to make sure they match in the
-        # template after string formatting.
-        captured_values: set[ir.Value] = set()
-        for operand in op.operands:
-            if operand in captured_values:
-                # TODO(Max191): Remove this warning when the transform for the
-                # `cast_compatible_dag_from_root` op fixes a bug in the matching
-                # logic that causes failure to match when the same operand is
-                # repeated. For now, still avoid adding duplicate SSA values to
-                # prevent parsing failure.
-                logging.warning(
-                    f"Root op has repeated operand. This can cause failure to match in the resulting TD spec at compile time."
-                )
-                continue
-            ssa_name = operand.get_name()
-            operand_type = operand.type
-            bbargs.append(f"{ssa_name}: {operand_type}")
-            captured_values.add(operand)
-        bbargs_str = ", ".join(bbargs)
-        matcher_block = f"""%ins, %outs = transform.iree.match.cast_compatible_dag_from_root %cont {{
-              ^bb0({bbargs_str}):
-              {root_operation}
-            }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)"""
+    # Get the names ssa names of operands to make sure they match in the
+    # template after string formatting.
+    captured_values: set[ir.Value] = set()
+    for operand in op.operands:
+        if operand in captured_values:
+            # TODO(Max191): Remove this warning when the transform for the
+            # `cast_compatible_dag_from_root` op fixes a bug in the matching
+            # logic that causes failure to match when the same operand is
+            # repeated. For now, still avoid adding duplicate SSA values to
+            # prevent parsing failure.
+            logging.warning(
+                f"Root op has repeated operand. This can cause failure to match in the resulting TD spec at compile time."
+            )
+            continue
+        ssa_name = operand.get_name()
```

**Comment:**
```suggestion
        operand_name = operand.get_name()
```

---

**File:** `sharktuner/sharktuner/spec_builder.py:196`

```diff
@@ -184,3 +141,219 @@ def build_td_spec(
         }}
     }}"""
     return ir.Module.parse(spec_text, context)
+
+
+def get_readonly_arg_attr() -> dict[str, ir.Attribute]:
+    return {"transform.readonly": ir.UnitAttr.get()}
+
+
+def get_consumed_arg_attr() -> dict[str, ir.Attribute]:
+    return {"transform.consumed": ir.UnitAttr.get()}
+
+
+class SpecBuilder(ABC):
+    def __init__(self, op_info: OpInfo):
+        self.op_info = op_info
+
+    def create_config_params(
+        self, config_list: list[common.TuningConfiguration]
+    ) -> list[ir.Value]:
+        """
+        Creates a constant parameter for each configuration.
+        Parameters can contain #iree_codegen.compilation_info or other configuration attributes.
+        """
+        return [
+            transform.ParamConstantOp(
+                transform.AnyParamType.get(),
+                config.configuration,
+            ).result
+            for config in config_list
+        ]
+
+    @abstractmethod
+    def build_matcher(
+        self,
+        entry_block: ir.Block,
+        cont_handle: ir.Value,
+        config_list: list[common.TuningConfiguration],
+    ) -> tuple[ir.OpResult, list[ir.OpResult]]:
+        pass
+
+    def create_matcher_sequence(
+        self,
+        config_list: list[common.TuningConfiguration],
+    ) -> transform.NamedSequenceOp:
+        """
+        Creates a transform.named_sequence that matches the operation and returns
+        the matched operation handle along with configuration parameters.
+        """
+        input_types = [transform.AnyOpType.get()]
+        output_types = [transform.AnyOpType.get()] + [
+            transform.AnyParamType.get()
+        ] * len(config_list)
+
+        named_seq = transform.NamedSequenceOp(
+            get_matcher_function_name(self.op_info.root_op),
```

**Comment:**
I would only add the `match_` prefix here, not in the helper

---


---


## [PR #2527](https://github.com/nod-ai/amd-shark-ai/pull/2527): [tuner][nfc] clean up the code

### Review Summary

**COMMENTED** (2025-10-16)

**APPROVED** (2025-10-16)

Thanks for cleaning this up


### Code Comments

**File:** `sharktuner/setup.py:10`

```diff
@@ -7,7 +7,7 @@
 import json
 import os
 
-from setuptools import setup
+from setuptools import setup  # type: ignore
```

**Comment:**
Could we add the typing info for these to our requirements? https://pypi.org/project/types-setuptools/

---


---


## [PR #2515](https://github.com/nod-ai/amd-shark-ai/pull/2515): [tuner] use python bindings to build td specs

### Review Summary

**COMMENTED** (2025-10-16)

Is there any way to split this into a few smaller PRs?

**COMMENTED** (2025-10-16)

**COMMENTED** (2025-10-16)

**COMMENTED** (2025-10-16)


### Code Comments

**File:** `sharktuner/sharktuner/candidate_gen.py:105`

```diff
@@ -69,67 +69,61 @@ def find_handler(self, op_name: str) -> DispatchTuner:
 class ContractionOpInterfaceTuner(
     DispatchTuner, dispatch_parser.ContractionOpInterfaceParser
 ):
-    def __init__(self, root_op: ir.Operation):
-        super().__init__(root_op)
+    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
+        super().__init__(root_op, tuner_ctx)
 
     def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
         return constraint_generator.ContractionOpInterfaceConstraintGenerator(
-            self.get_root_op()
+            self.get_op_info()
         )
 
     def get_td_spec(
         self,
         config_list: list[common.TuningConfiguration],
     ) -> ir.Module:
-        contraction_op = self.get_root_op()
-        func_name = self.get_root_op_func_name()
-        return spec_builder.build_td_spec(
-            contraction_op.context, contraction_op, config_list, func_name
-        )
+        opinfo = self.get_op_info()
+        builder = spec_builder.ContractionSpecBuilder(opinfo)
+        return builder.build_td_spec(config_list)
 
 
 class ConvolutionOpInterfaceTuner(
     DispatchTuner, dispatch_parser.ConvolutionOpInterfaceParser
 ):
-    def __init__(self, root_op: ir.Operation):
-        super().__init__(root_op)
+    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
+        super().__init__(root_op, tuner_ctx)
 
     def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
         return constraint_generator.ConvolutionOpInterfaceConstraintGenerator(
-            self.get_root_op()
+            self.get_op_info()
         )
 
     def get_td_spec(
         self,
         config_list: list[common.TuningConfiguration],
     ) -> ir.Module:
-        conv_op = self.get_root_op()
-        func_name = self.get_root_op_func_name()
-        return spec_builder.build_td_spec(
-            conv_op.context, conv_op, config_list, func_name
-        )
+        opinfo = self.get_op_info()
+        builder = spec_builder.ConvolutionSpecBuilder(opinfo)
```

**Comment:**
```suggestion
        op_info = self.get_op_info()
        builder = spec_builder.ConvolutionSpecBuilder(op_info)
```

---

**File:** `sharktuner/sharktuner/candidate_gen.py:125`

```diff
@@ -69,67 +69,61 @@ def find_handler(self, op_name: str) -> DispatchTuner:
 class ContractionOpInterfaceTuner(
     DispatchTuner, dispatch_parser.ContractionOpInterfaceParser
 ):
-    def __init__(self, root_op: ir.Operation):
-        super().__init__(root_op)
+    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
+        super().__init__(root_op, tuner_ctx)
 
     def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
         return constraint_generator.ContractionOpInterfaceConstraintGenerator(
-            self.get_root_op()
+            self.get_op_info()
         )
 
     def get_td_spec(
         self,
         config_list: list[common.TuningConfiguration],
     ) -> ir.Module:
-        contraction_op = self.get_root_op()
-        func_name = self.get_root_op_func_name()
-        return spec_builder.build_td_spec(
-            contraction_op.context, contraction_op, config_list, func_name
-        )
+        opinfo = self.get_op_info()
+        builder = spec_builder.ContractionSpecBuilder(opinfo)
+        return builder.build_td_spec(config_list)
 
 
 class ConvolutionOpInterfaceTuner(
     DispatchTuner, dispatch_parser.ConvolutionOpInterfaceParser
 ):
-    def __init__(self, root_op: ir.Operation):
-        super().__init__(root_op)
+    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
+        super().__init__(root_op, tuner_ctx)
 
     def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
         return constraint_generator.ConvolutionOpInterfaceConstraintGenerator(
-            self.get_root_op()
+            self.get_op_info()
         )
 
     def get_td_spec(
         self,
         config_list: list[common.TuningConfiguration],
     ) -> ir.Module:
-        conv_op = self.get_root_op()
-        func_name = self.get_root_op_func_name()
-        return spec_builder.build_td_spec(
-            conv_op.context, conv_op, config_list, func_name
-        )
+        opinfo = self.get_op_info()
+        builder = spec_builder.ConvolutionSpecBuilder(opinfo)
+        return builder.build_td_spec(config_list)
 
 
 class AttentionOpInterfaceTuner(
     DispatchTuner, dispatch_parser.AttentionOpInterfaceParser
 ):
-    def __init__(self, root_op: ir.Operation):
-        super().__init__(root_op)
+    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
+        super().__init__(root_op, tuner_ctx)
 
     def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
         return constraint_generator.AttentionOpInterfaceConstraintGenerator(
-            self.get_root_op()
+            self.get_op_info()
         )
 
     def get_td_spec(
         self,
         config_list: list[common.TuningConfiguration],
     ) -> ir.Module:
-        attention_op = self.get_root_op()
-        func_name = self.get_root_op_func_name()
-        return spec_builder.build_td_spec(
-            attention_op.context, attention_op, config_list, func_name
-        )
+        opinfo = self.get_op_info()
+        builder = spec_builder.AttentionSpecBuilder(opinfo)
```

**Comment:**
```suggestion
        op_info = self.get_op_info()
        builder = spec_builder.AttentionSpecBuilder(opinfo)
```

---

**File:** `sharktuner/sharktuner/candidate_gen.py:105`

```diff
@@ -69,67 +69,61 @@ def find_handler(self, op_name: str) -> DispatchTuner:
 class ContractionOpInterfaceTuner(
     DispatchTuner, dispatch_parser.ContractionOpInterfaceParser
 ):
-    def __init__(self, root_op: ir.Operation):
-        super().__init__(root_op)
+    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
+        super().__init__(root_op, tuner_ctx)
 
     def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
         return constraint_generator.ContractionOpInterfaceConstraintGenerator(
-            self.get_root_op()
+            self.get_op_info()
         )
 
     def get_td_spec(
         self,
         config_list: list[common.TuningConfiguration],
     ) -> ir.Module:
-        contraction_op = self.get_root_op()
-        func_name = self.get_root_op_func_name()
-        return spec_builder.build_td_spec(
-            contraction_op.context, contraction_op, config_list, func_name
-        )
+        opinfo = self.get_op_info()
+        builder = spec_builder.ContractionSpecBuilder(opinfo)
+        return builder.build_td_spec(config_list)
 
 
 class ConvolutionOpInterfaceTuner(
     DispatchTuner, dispatch_parser.ConvolutionOpInterfaceParser
 ):
-    def __init__(self, root_op: ir.Operation):
-        super().__init__(root_op)
+    def __init__(self, root_op: ir.Operation, tuner_ctx: common.TunerContext):
+        super().__init__(root_op, tuner_ctx)
 
     def get_constraint_generator(self) -> constraint_generator.ConstraintGenerator:
         return constraint_generator.ConvolutionOpInterfaceConstraintGenerator(
-            self.get_root_op()
+            self.get_op_info()
         )
 
     def get_td_spec(
         self,
         config_list: list[common.TuningConfiguration],
     ) -> ir.Module:
-        conv_op = self.get_root_op()
-        func_name = self.get_root_op_func_name()
-        return spec_builder.build_td_spec(
-            conv_op.context, conv_op, config_list, func_name
-        )
+        opinfo = self.get_op_info()
+        builder = spec_builder.ConvolutionSpecBuilder(opinfo)
```

**Comment:**
Can you also annotate op_info with a type?

---

**File:** `sharktuner/sharktuner/constraint_generator.py:492`

```diff
@@ -486,43 +488,8 @@ def generate_solutions(
 
 
 class ContractionOpInterfaceConstraintGenerator(ConstraintGenerator):
-    def __init__(self, root_op: ir.Operation):
-        self.root_op = root_op
-        contraction_dims = linalg.infer_contraction_dimensions(root_op)
-        assert contraction_dims, "no contraction dimensions"
-        dims = common.ContractionDimensions(
-            batch=list(contraction_dims.batch),
-            m=list(contraction_dims.m),
-            n=list(contraction_dims.n),
-            k=list(contraction_dims.k),
-        )
-
-        res_maps = linalg.get_indexing_maps(root_op)
-        maps = [map_attr.value for map_attr in res_maps]
-        lhs_dims = common.get_map_result_dim_positions(maps[0])
-        rhs_dims = common.get_map_result_dim_positions(maps[1])
-        res_dims = common.get_map_result_dim_positions(maps[2])
-
-        assert lhs_dims, "no lhs dimensions"
-        assert rhs_dims, "no rhs dimensions"
-        assert res_dims, "no result dimensions"
-
-        lhs_type = ir.RankedTensorType(root_op.operands[0].type)
-        rhs_type = ir.RankedTensorType(root_op.operands[1].type)
-        res_type = ir.RankedTensorType(root_op.operands[2].type)
-
-        matmul_size = common.ContractionSizes(
-            M=[lhs_type.shape[lhs_dims.index(dim)] for dim in contraction_dims.m],
-            N=[rhs_type.shape[rhs_dims.index(dim)] for dim in contraction_dims.n],
-            K=[lhs_type.shape[lhs_dims.index(dim)] for dim in contraction_dims.k],
-            B=[lhs_type.shape[lhs_dims.index(dim)] for dim in contraction_dims.batch],
-        )
-
-        self.dims = dims
-        self.matmul_size = matmul_size
-        self.lhs_type = common.ShapedType(lhs_type.shape, lhs_type.element_type)
-        self.rhs_type = common.ShapedType(rhs_type.shape, rhs_type.element_type)
-        self.res_type = common.ShapedType(res_type.shape, res_type.element_type)
+    def __init__(self, opinfo: dispatch_parser.ContractionOpInfo):
+        self.opinfo = opinfo
```

**Comment:**
what's the type?
```suggestion
        self.op_info = op_info
```

---

**File:** `sharktuner/sharktuner/dispatch_parser.py:35`

```diff
@@ -28,40 +30,252 @@ def parse_mlir(mlir_text: str, ctx: common.TunerContext) -> ir.Module:
     return mlir_module
 
 
+@dataclass
+class OpInfo:
+    func_name: str
```

**Comment:**
What does func_name refer to?

---

**File:** `sharktuner/sharktuner/dispatch_parser.py:41`

```diff
@@ -28,40 +30,252 @@ def parse_mlir(mlir_text: str, ctx: common.TunerContext) -> ir.Module:
     return mlir_module
 
 
+@dataclass
+class OpInfo:
+    func_name: str
+    indexing_maps: list[ir.AffineMap]
+    tuner_ctx: common.TunerContext
+
+    @property
+    def context(self) -> ir.Context:
+        return self.tuner_ctx.mlir_ctx
```

**Comment:**
Why does op info need context?

---

**File:** `sharktuner/sharktuner/spec_builder.py:25`

```diff
@@ -17,170 +20,330 @@
 ROOT_OP_ATTR_NAME = "root_op"
 
 
-def get_placeholder_spec(context: ir.Context) -> ir.Module:
-    spec_text = f"""
-        module attributes {{ transform.with_named_sequence }} {{
-            transform.named_sequence
-            @__kernel_config(%variant_op: !transform.any_op {{transform.readonly}}) -> !transform.any_op
-                attributes {{ iree_codegen.tuning_spec_entrypoint }} {{
-                transform.yield %variant_op : !transform.any_op
-            }}
-        }}
-        """
-    return ir.Module.parse(spec_text, context)
-
-
-# TODO(Max191): Use python bindings to build the transform dialect spec module
-# instead of using string formatting.
-def build_td_spec(
-    context: ir.Context,
-    op: ir.Operation,
-    config_list: list[common.TuningConfiguration],
-    func_name: str,
-) -> ir.Module:
-    bbargs = []
-    # The `root_op` attribute will prevent matching of ops without the attr in
-    # the resulting TD spec matcher if it is not removed, so we remove it here.
-    # After removing, we must add it back, since the op is connected to the
-    # input module, which gets used for all candidates.
-    # TODO(Max191): Find a cleaner way to do this without removing and adding
-    # back the attribute.
-    has_root_attr = ROOT_OP_ATTR_NAME in op.opview.attributes
-    if has_root_attr:
-        assert isinstance(
-            op.opview.attributes[ROOT_OP_ATTR_NAME], ir.UnitAttr
-        ), f"expected '{ROOT_OP_ATTR_NAME}' attr to be a unit attr"
-    if has_root_attr:
-        del op.opview.attributes[ROOT_OP_ATTR_NAME]
-    # Get the root op string for formatting the final spec.
-    root_operation = str(op)
-    if has_root_attr:
-        op.opview.attributes[ROOT_OP_ATTR_NAME] = ir.UnitAttr.get(op.context)
-
-    if linalg.isa_contraction_op(op):
-        # Temporary solution using custom contraction transform ops for contraction operations.
-        inputs = op.opview.operands
-        outputs = op.opview.results
-        lhs_type = str(ir.RankedTensorType(inputs[0].type).element_type)
-        rhs_type = str(ir.RankedTensorType(inputs[1].type).element_type)
-        output_type = str(ir.RankedTensorType(outputs[0].type).element_type)
-
-        contraction_dims = linalg.infer_contraction_dimensions(op)
-        indexing_maps = linalg.get_indexing_maps(op)
-        maps = [map_attr.value for map_attr in indexing_maps]
-        lhs_dims = common.get_map_result_dim_positions(maps[0])
-        rhs_dims = common.get_map_result_dim_positions(maps[1])
-        assert lhs_dims, "no lhs dimensions"
-        assert rhs_dims, "no rhs dimensions"
-        lhs_type_shape = ir.RankedTensorType(op.operands[0].type)
-        rhs_type_shape = ir.RankedTensorType(op.operands[1].type)
-
-        m_dims = [
-            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.m
-        ]
-        n_dims = [
-            rhs_type_shape.shape[rhs_dims.index(dim)] for dim in contraction_dims.n
-        ]
-        k_dims = [
-            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.k
-        ]
-        batch_dims = [
-            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.batch
-        ]
-
-        dims_equal_checks = []
-        dims_equal_checks.append(
-            f"transform.iree.match.dims_equal %batch, {batch_dims} : !transform.param<i64>"
-        )
-        dims_equal_checks.append(
-            f"transform.iree.match.dims_equal %m, {m_dims} : !transform.param<i64>"
-        )
-        dims_equal_checks.append(
-            f"transform.iree.match.dims_equal %n, {n_dims} : !transform.param<i64>"
+class SpecBuilder(ABC):
+    def __init__(self, opinfo: OpInfo):
+        self.opinfo = opinfo
```

**Comment:**
```suggestion
    def __init__(self, op_info: OpInfo):
        self.op_info = op_info
```

---

**File:** `sharktuner/sharktuner/spec_builder.py:52`

```diff
@@ -17,170 +20,330 @@
 ROOT_OP_ATTR_NAME = "root_op"
 
 
-def get_placeholder_spec(context: ir.Context) -> ir.Module:
-    spec_text = f"""
-        module attributes {{ transform.with_named_sequence }} {{
-            transform.named_sequence
-            @__kernel_config(%variant_op: !transform.any_op {{transform.readonly}}) -> !transform.any_op
-                attributes {{ iree_codegen.tuning_spec_entrypoint }} {{
-                transform.yield %variant_op : !transform.any_op
-            }}
-        }}
-        """
-    return ir.Module.parse(spec_text, context)
-
-
-# TODO(Max191): Use python bindings to build the transform dialect spec module
-# instead of using string formatting.
-def build_td_spec(
-    context: ir.Context,
-    op: ir.Operation,
-    config_list: list[common.TuningConfiguration],
-    func_name: str,
-) -> ir.Module:
-    bbargs = []
-    # The `root_op` attribute will prevent matching of ops without the attr in
-    # the resulting TD spec matcher if it is not removed, so we remove it here.
-    # After removing, we must add it back, since the op is connected to the
-    # input module, which gets used for all candidates.
-    # TODO(Max191): Find a cleaner way to do this without removing and adding
-    # back the attribute.
-    has_root_attr = ROOT_OP_ATTR_NAME in op.opview.attributes
-    if has_root_attr:
-        assert isinstance(
-            op.opview.attributes[ROOT_OP_ATTR_NAME], ir.UnitAttr
-        ), f"expected '{ROOT_OP_ATTR_NAME}' attr to be a unit attr"
-    if has_root_attr:
-        del op.opview.attributes[ROOT_OP_ATTR_NAME]
-    # Get the root op string for formatting the final spec.
-    root_operation = str(op)
-    if has_root_attr:
-        op.opview.attributes[ROOT_OP_ATTR_NAME] = ir.UnitAttr.get(op.context)
-
-    if linalg.isa_contraction_op(op):
-        # Temporary solution using custom contraction transform ops for contraction operations.
-        inputs = op.opview.operands
-        outputs = op.opview.results
-        lhs_type = str(ir.RankedTensorType(inputs[0].type).element_type)
-        rhs_type = str(ir.RankedTensorType(inputs[1].type).element_type)
-        output_type = str(ir.RankedTensorType(outputs[0].type).element_type)
-
-        contraction_dims = linalg.infer_contraction_dimensions(op)
-        indexing_maps = linalg.get_indexing_maps(op)
-        maps = [map_attr.value for map_attr in indexing_maps]
-        lhs_dims = common.get_map_result_dim_positions(maps[0])
-        rhs_dims = common.get_map_result_dim_positions(maps[1])
-        assert lhs_dims, "no lhs dimensions"
-        assert rhs_dims, "no rhs dimensions"
-        lhs_type_shape = ir.RankedTensorType(op.operands[0].type)
-        rhs_type_shape = ir.RankedTensorType(op.operands[1].type)
-
-        m_dims = [
-            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.m
-        ]
-        n_dims = [
-            rhs_type_shape.shape[rhs_dims.index(dim)] for dim in contraction_dims.n
-        ]
-        k_dims = [
-            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.k
-        ]
-        batch_dims = [
-            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.batch
-        ]
-
-        dims_equal_checks = []
-        dims_equal_checks.append(
-            f"transform.iree.match.dims_equal %batch, {batch_dims} : !transform.param<i64>"
-        )
-        dims_equal_checks.append(
-            f"transform.iree.match.dims_equal %m, {m_dims} : !transform.param<i64>"
-        )
-        dims_equal_checks.append(
-            f"transform.iree.match.dims_equal %n, {n_dims} : !transform.param<i64>"
+class SpecBuilder(ABC):
+    def __init__(self, opinfo: OpInfo):
+        self.opinfo = opinfo
+
+    def create_config_params(
+        self, config_list: list[common.TuningConfiguration]
+    ) -> list[ir.Value]:
+        """
+        Creates parameter constant operations for each configuration.
```

**Comment:**
I don't understand this comment

---

**File:** `sharktuner/sharktuner/spec_builder.py:43`

```diff
@@ -17,170 +20,330 @@
 ROOT_OP_ATTR_NAME = "root_op"
 
 
-def get_placeholder_spec(context: ir.Context) -> ir.Module:
-    spec_text = f"""
-        module attributes {{ transform.with_named_sequence }} {{
-            transform.named_sequence
-            @__kernel_config(%variant_op: !transform.any_op {{transform.readonly}}) -> !transform.any_op
-                attributes {{ iree_codegen.tuning_spec_entrypoint }} {{
-                transform.yield %variant_op : !transform.any_op
-            }}
-        }}
-        """
-    return ir.Module.parse(spec_text, context)
-
-
-# TODO(Max191): Use python bindings to build the transform dialect spec module
-# instead of using string formatting.
-def build_td_spec(
-    context: ir.Context,
-    op: ir.Operation,
-    config_list: list[common.TuningConfiguration],
-    func_name: str,
-) -> ir.Module:
-    bbargs = []
-    # The `root_op` attribute will prevent matching of ops without the attr in
-    # the resulting TD spec matcher if it is not removed, so we remove it here.
-    # After removing, we must add it back, since the op is connected to the
-    # input module, which gets used for all candidates.
-    # TODO(Max191): Find a cleaner way to do this without removing and adding
-    # back the attribute.
-    has_root_attr = ROOT_OP_ATTR_NAME in op.opview.attributes
-    if has_root_attr:
-        assert isinstance(
-            op.opview.attributes[ROOT_OP_ATTR_NAME], ir.UnitAttr
-        ), f"expected '{ROOT_OP_ATTR_NAME}' attr to be a unit attr"
-    if has_root_attr:
-        del op.opview.attributes[ROOT_OP_ATTR_NAME]
-    # Get the root op string for formatting the final spec.
-    root_operation = str(op)
-    if has_root_attr:
-        op.opview.attributes[ROOT_OP_ATTR_NAME] = ir.UnitAttr.get(op.context)
-
-    if linalg.isa_contraction_op(op):
-        # Temporary solution using custom contraction transform ops for contraction operations.
-        inputs = op.opview.operands
-        outputs = op.opview.results
-        lhs_type = str(ir.RankedTensorType(inputs[0].type).element_type)
-        rhs_type = str(ir.RankedTensorType(inputs[1].type).element_type)
-        output_type = str(ir.RankedTensorType(outputs[0].type).element_type)
-
-        contraction_dims = linalg.infer_contraction_dimensions(op)
-        indexing_maps = linalg.get_indexing_maps(op)
-        maps = [map_attr.value for map_attr in indexing_maps]
-        lhs_dims = common.get_map_result_dim_positions(maps[0])
-        rhs_dims = common.get_map_result_dim_positions(maps[1])
-        assert lhs_dims, "no lhs dimensions"
-        assert rhs_dims, "no rhs dimensions"
-        lhs_type_shape = ir.RankedTensorType(op.operands[0].type)
-        rhs_type_shape = ir.RankedTensorType(op.operands[1].type)
-
-        m_dims = [
-            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.m
-        ]
-        n_dims = [
-            rhs_type_shape.shape[rhs_dims.index(dim)] for dim in contraction_dims.n
-        ]
-        k_dims = [
-            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.k
-        ]
-        batch_dims = [
-            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.batch
-        ]
-
-        dims_equal_checks = []
-        dims_equal_checks.append(
-            f"transform.iree.match.dims_equal %batch, {batch_dims} : !transform.param<i64>"
-        )
-        dims_equal_checks.append(
-            f"transform.iree.match.dims_equal %m, {m_dims} : !transform.param<i64>"
-        )
-        dims_equal_checks.append(
-            f"transform.iree.match.dims_equal %n, {n_dims} : !transform.param<i64>"
+class SpecBuilder(ABC):
+    def __init__(self, opinfo: OpInfo):
+        self.opinfo = opinfo
+
+    def create_config_params(
+        self, config_list: list[common.TuningConfiguration]
+    ) -> list[ir.Value]:
+        """
+        Creates parameter constant operations for each configuration.
+        """
+        config_params = []
+        for config in config_list:
+            config_param = transform.ParamConstantOp(
+                transform.AnyParamType.get(),
+                config.configuration,
+            ).result
+            config_params.append(config_param)
+        return config_params
+
+    @staticmethod
+    def get_placeholder_spec(context: ir.Context) -> ir.Module:
```

**Comment:**
What's the benefit of building the base spec programmatically? If it's going to be the same for all inputs, having hardocded mlir sounds OK to me and is much more concise.

---

**File:** `sharktuner/sharktuner/spec_builder.py:109`

```diff
@@ -17,170 +20,330 @@
 ROOT_OP_ATTR_NAME = "root_op"
 
 
-def get_placeholder_spec(context: ir.Context) -> ir.Module:
-    spec_text = f"""
-        module attributes {{ transform.with_named_sequence }} {{
-            transform.named_sequence
-            @__kernel_config(%variant_op: !transform.any_op {{transform.readonly}}) -> !transform.any_op
-                attributes {{ iree_codegen.tuning_spec_entrypoint }} {{
-                transform.yield %variant_op : !transform.any_op
-            }}
-        }}
-        """
-    return ir.Module.parse(spec_text, context)
-
-
-# TODO(Max191): Use python bindings to build the transform dialect spec module
-# instead of using string formatting.
-def build_td_spec(
-    context: ir.Context,
-    op: ir.Operation,
-    config_list: list[common.TuningConfiguration],
-    func_name: str,
-) -> ir.Module:
-    bbargs = []
-    # The `root_op` attribute will prevent matching of ops without the attr in
-    # the resulting TD spec matcher if it is not removed, so we remove it here.
-    # After removing, we must add it back, since the op is connected to the
-    # input module, which gets used for all candidates.
-    # TODO(Max191): Find a cleaner way to do this without removing and adding
-    # back the attribute.
-    has_root_attr = ROOT_OP_ATTR_NAME in op.opview.attributes
-    if has_root_attr:
-        assert isinstance(
-            op.opview.attributes[ROOT_OP_ATTR_NAME], ir.UnitAttr
-        ), f"expected '{ROOT_OP_ATTR_NAME}' attr to be a unit attr"
-    if has_root_attr:
-        del op.opview.attributes[ROOT_OP_ATTR_NAME]
-    # Get the root op string for formatting the final spec.
-    root_operation = str(op)
-    if has_root_attr:
-        op.opview.attributes[ROOT_OP_ATTR_NAME] = ir.UnitAttr.get(op.context)
-
-    if linalg.isa_contraction_op(op):
-        # Temporary solution using custom contraction transform ops for contraction operations.
-        inputs = op.opview.operands
-        outputs = op.opview.results
-        lhs_type = str(ir.RankedTensorType(inputs[0].type).element_type)
-        rhs_type = str(ir.RankedTensorType(inputs[1].type).element_type)
-        output_type = str(ir.RankedTensorType(outputs[0].type).element_type)
-
-        contraction_dims = linalg.infer_contraction_dimensions(op)
-        indexing_maps = linalg.get_indexing_maps(op)
-        maps = [map_attr.value for map_attr in indexing_maps]
-        lhs_dims = common.get_map_result_dim_positions(maps[0])
-        rhs_dims = common.get_map_result_dim_positions(maps[1])
-        assert lhs_dims, "no lhs dimensions"
-        assert rhs_dims, "no rhs dimensions"
-        lhs_type_shape = ir.RankedTensorType(op.operands[0].type)
-        rhs_type_shape = ir.RankedTensorType(op.operands[1].type)
-
-        m_dims = [
-            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.m
-        ]
-        n_dims = [
-            rhs_type_shape.shape[rhs_dims.index(dim)] for dim in contraction_dims.n
-        ]
-        k_dims = [
-            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.k
-        ]
-        batch_dims = [
-            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.batch
-        ]
-
-        dims_equal_checks = []
-        dims_equal_checks.append(
-            f"transform.iree.match.dims_equal %batch, {batch_dims} : !transform.param<i64>"
-        )
-        dims_equal_checks.append(
-            f"transform.iree.match.dims_equal %m, {m_dims} : !transform.param<i64>"
-        )
-        dims_equal_checks.append(
-            f"transform.iree.match.dims_equal %n, {n_dims} : !transform.param<i64>"
+class SpecBuilder(ABC):
+    def __init__(self, opinfo: OpInfo):
+        self.opinfo = opinfo
+
+    def create_config_params(
+        self, config_list: list[common.TuningConfiguration]
+    ) -> list[ir.Value]:
+        """
+        Creates parameter constant operations for each configuration.
+        """
+        config_params = []
+        for config in config_list:
+            config_param = transform.ParamConstantOp(
+                transform.AnyParamType.get(),
+                config.configuration,
+            ).result
+            config_params.append(config_param)
+        return config_params
+
+    @staticmethod
+    def get_placeholder_spec(context: ir.Context) -> ir.Module:
+        """
+        Creates a placeholder Transform Dialect spec that does nothing.
+
+        This is used for the baseline (index 0) configuration where no
+        tuning spec is applied. It simply yields the input variant operation
+        without any modifications.
+
+        """
+        with context, ir.Location.unknown(context):
+            module = ir.Module.create()
+            module.operation.attributes[
+                "transform.with_named_sequence"
+            ] = ir.UnitAttr.get()
+
+            with ir.InsertionPoint(module.body):
+                input_types = [transform.AnyOpType.get()]
+                output_types = [transform.AnyOpType.get()]
+
+                arg_attrs = [
+                    {"transform.readonly": ir.UnitAttr.get()} for _ in input_types
+                ]
+
+                named_seq = transform.NamedSequenceOp(
+                    "__kernel_config",
+                    input_types,
+                    output_types,
+                    arg_attrs=arg_attrs,
+                )
+
+                named_seq.operation.attributes[
+                    "iree_codegen.tuning_spec_entrypoint"
+                ] = ir.UnitAttr.get()
+
+                with ir.InsertionPoint(named_seq.body):
+                    variant_op = named_seq.bodyTarget
+                    transform.YieldOp([variant_op])
+
+            return module
+
+    @abstractmethod
+    def build_matcher(
+        self,
+        entry_block: ir.Block,
+        cont_handle: ir.Value,
+        config_list: list[common.TuningConfiguration],
+    ) -> tuple[ir.OpResult, list[ir.OpResult]]:
+        pass
+
+    def create_matcher_sequence(
+        self,
+        config_list: list[common.TuningConfiguration],
+    ) -> transform.NamedSequenceOp:
+        """
+        Creates a transform.named_sequence that matches the operation and returns
+        the matched operation handle along with configuration parameters.
+        """
+        input_types = [transform.AnyOpType.get()]
+        output_types = [transform.AnyOpType.get()] + [
+            transform.AnyParamType.get()
+        ] * len(config_list)
+
+        named_seq = transform.NamedSequenceOp(
+            self.opinfo.func_name,
+            input_types,
+            output_types,
+            arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
```

**Comment:**
maybe create a helper for getting these readonly attributes?

---

**File:** `sharktuner/setup.py:10`

```diff
@@ -7,7 +7,7 @@
 import json
 import os
 
-from setuptools import setup
+from setuptools import setup  # type: ignore
```

**Comment:**
Can you move unrelated changes to a separate PR?

---

**File:** `sharktuner/tests/spec_builder_test.py:268`

```diff
@@ -248,21 +252,20 @@ def test_spec_builder_with_batch_dims(tuner_ctx: common.TunerContext) -> None:
         lowering_config, translation_info
     )
 
-    spec_module = spec_builder.build_td_spec(
-        tuner_ctx.mlir_ctx,
-        root_op,
+    builder = spec_builder.ContractionSpecBuilder(opinfo)
+    spec_module = builder.build_td_spec(
         [
             common.TuningConfiguration(
                 name="compilation_info", configuration=compilation_info
             )
-        ],
-        "match_batch_matmul",
+        ]
     )
+
     assert spec_module
     assert isinstance(spec_module, ir.Module)
     spec_str = str(spec_module)
 
-    assert "@match_batch_matmul -> @apply_op_config" in spec_str
+    assert "@match_batch_matmul_func -> @apply_op_config" in spec_str
```

**Comment:**
Why has the name changed?

---

**File:** `sharktuner/sharktuner/dispatch_parser.py:35`

```diff
@@ -28,40 +30,252 @@ def parse_mlir(mlir_text: str, ctx: common.TunerContext) -> ir.Module:
     return mlir_module
 
 
+@dataclass
+class OpInfo:
+    func_name: str
```

**Comment:**
Maybe call it `parent_function_name` instead? Or just keep the function op as a field and query its name. I thought that this was the name of the op itself.

---

**File:** `sharktuner/sharktuner/dispatch_parser.py:41`

```diff
@@ -28,40 +30,252 @@ def parse_mlir(mlir_text: str, ctx: common.TunerContext) -> ir.Module:
     return mlir_module
 
 
+@dataclass
+class OpInfo:
+    func_name: str
+    indexing_maps: list[ir.AffineMap]
+    tuner_ctx: common.TunerContext
+
+    @property
+    def context(self) -> ir.Context:
+        return self.tuner_ctx.mlir_ctx
```

**Comment:**
Can we pass context as a parameter to `build_td_spec` instead?

---

**File:** `sharktuner/sharktuner/spec_builder.py:52`

```diff
@@ -17,170 +20,330 @@
 ROOT_OP_ATTR_NAME = "root_op"
 
 
-def get_placeholder_spec(context: ir.Context) -> ir.Module:
-    spec_text = f"""
-        module attributes {{ transform.with_named_sequence }} {{
-            transform.named_sequence
-            @__kernel_config(%variant_op: !transform.any_op {{transform.readonly}}) -> !transform.any_op
-                attributes {{ iree_codegen.tuning_spec_entrypoint }} {{
-                transform.yield %variant_op : !transform.any_op
-            }}
-        }}
-        """
-    return ir.Module.parse(spec_text, context)
-
-
-# TODO(Max191): Use python bindings to build the transform dialect spec module
-# instead of using string formatting.
-def build_td_spec(
-    context: ir.Context,
-    op: ir.Operation,
-    config_list: list[common.TuningConfiguration],
-    func_name: str,
-) -> ir.Module:
-    bbargs = []
-    # The `root_op` attribute will prevent matching of ops without the attr in
-    # the resulting TD spec matcher if it is not removed, so we remove it here.
-    # After removing, we must add it back, since the op is connected to the
-    # input module, which gets used for all candidates.
-    # TODO(Max191): Find a cleaner way to do this without removing and adding
-    # back the attribute.
-    has_root_attr = ROOT_OP_ATTR_NAME in op.opview.attributes
-    if has_root_attr:
-        assert isinstance(
-            op.opview.attributes[ROOT_OP_ATTR_NAME], ir.UnitAttr
-        ), f"expected '{ROOT_OP_ATTR_NAME}' attr to be a unit attr"
-    if has_root_attr:
-        del op.opview.attributes[ROOT_OP_ATTR_NAME]
-    # Get the root op string for formatting the final spec.
-    root_operation = str(op)
-    if has_root_attr:
-        op.opview.attributes[ROOT_OP_ATTR_NAME] = ir.UnitAttr.get(op.context)
-
-    if linalg.isa_contraction_op(op):
-        # Temporary solution using custom contraction transform ops for contraction operations.
-        inputs = op.opview.operands
-        outputs = op.opview.results
-        lhs_type = str(ir.RankedTensorType(inputs[0].type).element_type)
-        rhs_type = str(ir.RankedTensorType(inputs[1].type).element_type)
-        output_type = str(ir.RankedTensorType(outputs[0].type).element_type)
-
-        contraction_dims = linalg.infer_contraction_dimensions(op)
-        indexing_maps = linalg.get_indexing_maps(op)
-        maps = [map_attr.value for map_attr in indexing_maps]
-        lhs_dims = common.get_map_result_dim_positions(maps[0])
-        rhs_dims = common.get_map_result_dim_positions(maps[1])
-        assert lhs_dims, "no lhs dimensions"
-        assert rhs_dims, "no rhs dimensions"
-        lhs_type_shape = ir.RankedTensorType(op.operands[0].type)
-        rhs_type_shape = ir.RankedTensorType(op.operands[1].type)
-
-        m_dims = [
-            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.m
-        ]
-        n_dims = [
-            rhs_type_shape.shape[rhs_dims.index(dim)] for dim in contraction_dims.n
-        ]
-        k_dims = [
-            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.k
-        ]
-        batch_dims = [
-            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.batch
-        ]
-
-        dims_equal_checks = []
-        dims_equal_checks.append(
-            f"transform.iree.match.dims_equal %batch, {batch_dims} : !transform.param<i64>"
-        )
-        dims_equal_checks.append(
-            f"transform.iree.match.dims_equal %m, {m_dims} : !transform.param<i64>"
-        )
-        dims_equal_checks.append(
-            f"transform.iree.match.dims_equal %n, {n_dims} : !transform.param<i64>"
+class SpecBuilder(ABC):
+    def __init__(self, opinfo: OpInfo):
+        self.opinfo = opinfo
+
+    def create_config_params(
+        self, config_list: list[common.TuningConfiguration]
+    ) -> list[ir.Value]:
+        """
+        Creates parameter constant operations for each configuration.
```

**Comment:**
Maybe `Creates a constant parameter with #iree_codegen.compilation_info for each configuration`?

---


---


## [PR #2489](https://github.com/nod-ai/amd-shark-ai/pull/2489): [tuner] update name for consistency in pypi readme

### Review Summary

**APPROVED** (2025-10-12)


---


## [PR #2482](https://github.com/nod-ai/amd-shark-ai/pull/2482): [tuner] add pypi readme

### Review Summary

**COMMENTED** (2025-10-10)

**APPROVED** (2025-10-11)


### Code Comments

**File:** `sharktuner/PYPI_README.md:9`

```diff
@@ -0,0 +1,50 @@
+# Sharktuner
+Sharktuner automates the dispatch-level and model-level tuning for the IREE (Intermediate Representation Execution Environment) ML Compiler on AMD GPUs.
+
+## Installation
+Install Sharktuner from PyPI:
+```shell
+pip install sharktuner
+```
+This will automatically install all required dependencies including IREE compiler and runtime.
```

**Comment:**
````suggestion
```

This will install all required dependencies including IREE compiler and runtime.
````

---

**File:** `sharktuner/PYPI_README.md:11`

```diff
@@ -0,0 +1,50 @@
+# Sharktuner
+Sharktuner automates the dispatch-level and model-level tuning for the IREE (Intermediate Representation Execution Environment) ML Compiler on AMD GPUs.
+
+## Installation
+Install Sharktuner from PyPI:
+```shell
+pip install sharktuner
+```
+This will automatically install all required dependencies including IREE compiler and runtime.
+
+You can use nightly IREE's python bindings for the latest version:
```

**Comment:**
```suggestion
You can use the latest nightly IREE python bindings:
```

---

**File:** `sharktuner/PYPI_README.md:25`

```diff
@@ -0,0 +1,50 @@
+# Sharktuner
+Sharktuner automates the dispatch-level and model-level tuning for the IREE (Intermediate Representation Execution Environment) ML Compiler on AMD GPUs.
+
+## Installation
+Install Sharktuner from PyPI:
+```shell
+pip install sharktuner
+```
+This will automatically install all required dependencies including IREE compiler and runtime.
+
+You can use nightly IREE's python bindings for the latest version:
+```shell
+pip install --upgrade --pre \
+    iree-base-compiler \
+    iree-base-runtime \
+    --find-links https://iree.dev/pip-release-links.html
+```
+
+Verify that the environment is set up correctly:
+```shell
+python -m model_tuner --help
+```
+or
```

**Comment:**
````suggestion
```

or
````

---

**File:** `sharktuner/PYPI_README.md:29`

```diff
@@ -0,0 +1,50 @@
+# Sharktuner
+Sharktuner automates the dispatch-level and model-level tuning for the IREE (Intermediate Representation Execution Environment) ML Compiler on AMD GPUs.
+
+## Installation
+Install Sharktuner from PyPI:
+```shell
+pip install sharktuner
+```
+This will automatically install all required dependencies including IREE compiler and runtime.
+
+You can use nightly IREE's python bindings for the latest version:
+```shell
+pip install --upgrade --pre \
+    iree-base-compiler \
+    iree-base-runtime \
+    --find-links https://iree.dev/pip-release-links.html
+```
+
+Verify that the environment is set up correctly:
+```shell
+python -m model_tuner --help
+```
+or
+
+```shell
+python -m dispatch_tuner --help
+```
+
+**Model Tuner**
```

**Comment:**
Use `###` for third-level headings 

---

**File:** `sharktuner/PYPI_README.md:31`

```diff
@@ -0,0 +1,50 @@
+# Sharktuner
+Sharktuner automates the dispatch-level and model-level tuning for the IREE (Intermediate Representation Execution Environment) ML Compiler on AMD GPUs.
+
+## Installation
+Install Sharktuner from PyPI:
+```shell
+pip install sharktuner
+```
+This will automatically install all required dependencies including IREE compiler and runtime.
+
+You can use nightly IREE's python bindings for the latest version:
+```shell
+pip install --upgrade --pre \
+    iree-base-compiler \
+    iree-base-runtime \
+    --find-links https://iree.dev/pip-release-links.html
+```
+
+Verify that the environment is set up correctly:
+```shell
+python -m model_tuner --help
+```
+or
+
+```shell
+python -m dispatch_tuner --help
+```
+
+**Model Tuner**
+Use the Model Tuner to tune a dispatch and a model:
+```bash
```

**Comment:**
why is this `bash` while the previous snippets used `shell`?

---

**File:** `sharktuner/PYPI_README.md:43`

```diff
@@ -0,0 +1,50 @@
+# Sharktuner
+Sharktuner automates the dispatch-level and model-level tuning for the IREE (Intermediate Representation Execution Environment) ML Compiler on AMD GPUs.
+
+## Installation
+Install Sharktuner from PyPI:
+```shell
+pip install sharktuner
+```
+This will automatically install all required dependencies including IREE compiler and runtime.
+
+You can use nightly IREE's python bindings for the latest version:
+```shell
+pip install --upgrade --pre \
+    iree-base-compiler \
+    iree-base-runtime \
+    --find-links https://iree.dev/pip-release-links.html
+```
+
+Verify that the environment is set up correctly:
+```shell
+python -m model_tuner --help
+```
+or
+
+```shell
+python -m dispatch_tuner --help
+```
+
+**Model Tuner**
+Use the Model Tuner to tune a dispatch and a model:
+```bash
+python -m model_tuner double_mmt.mlir mmt_benchmark.mlir \
+    --compile-flags-file=compile_flags.txt \
+    --model-benchmark-flags-file=model_benchmark_flags.txt \
+    --devices=hip://0 \
+    --num-candidates=30 \
+    --model-tuner-num-dispatch-candidates=5 \
+    --model-tuner-num-model-candidates=3`
+```
+Refer to [Mode Tuner README](https://github.com/nod-ai/shark-ai/tree/main/sharktuner/model_tuner) for detailed information on flags and MLIR files.
```

**Comment:**
````suggestion
```

Refer to [Mode Tuner README](https://github.com/nod-ai/shark-ai/tree/main/sharktuner/model_tuner) for detailed information on flags and MLIR input files.
````

---

**File:** `sharktuner/PYPI_README.md:42`

```diff
@@ -0,0 +1,50 @@
+# Sharktuner
+Sharktuner automates the dispatch-level and model-level tuning for the IREE (Intermediate Representation Execution Environment) ML Compiler on AMD GPUs.
+
+## Installation
+Install Sharktuner from PyPI:
+```shell
+pip install sharktuner
+```
+This will automatically install all required dependencies including IREE compiler and runtime.
+
+You can use nightly IREE's python bindings for the latest version:
+```shell
+pip install --upgrade --pre \
+    iree-base-compiler \
+    iree-base-runtime \
+    --find-links https://iree.dev/pip-release-links.html
+```
+
+Verify that the environment is set up correctly:
+```shell
+python -m model_tuner --help
+```
+or
+
+```shell
+python -m dispatch_tuner --help
+```
+
+**Model Tuner**
+Use the Model Tuner to tune a dispatch and a model:
+```bash
+python -m model_tuner double_mmt.mlir mmt_benchmark.mlir \
+    --compile-flags-file=compile_flags.txt \
+    --model-benchmark-flags-file=model_benchmark_flags.txt \
+    --devices=hip://0 \
+    --num-candidates=30 \
+    --model-tuner-num-dispatch-candidates=5 \
+    --model-tuner-num-model-candidates=3`
+```
+Refer to [Mode Tuner README](https://github.com/nod-ai/shark-ai/tree/main/sharktuner/model_tuner) for detailed information on flags and MLIR files.
+
+**Dispatch Tuner**
```

**Comment:**
also here

---

**File:** `sharktuner/PYPI_README.md:44`

```diff
@@ -0,0 +1,50 @@
+# Sharktuner
+Sharktuner automates the dispatch-level and model-level tuning for the IREE (Intermediate Representation Execution Environment) ML Compiler on AMD GPUs.
+
+## Installation
+Install Sharktuner from PyPI:
+```shell
+pip install sharktuner
+```
+This will automatically install all required dependencies including IREE compiler and runtime.
+
+You can use nightly IREE's python bindings for the latest version:
+```shell
+pip install --upgrade --pre \
+    iree-base-compiler \
+    iree-base-runtime \
+    --find-links https://iree.dev/pip-release-links.html
+```
+
+Verify that the environment is set up correctly:
+```shell
+python -m model_tuner --help
+```
+or
+
+```shell
+python -m dispatch_tuner --help
+```
+
+**Model Tuner**
+Use the Model Tuner to tune a dispatch and a model:
+```bash
+python -m model_tuner double_mmt.mlir mmt_benchmark.mlir \
+    --compile-flags-file=compile_flags.txt \
+    --model-benchmark-flags-file=model_benchmark_flags.txt \
+    --devices=hip://0 \
+    --num-candidates=30 \
+    --model-tuner-num-dispatch-candidates=5 \
+    --model-tuner-num-model-candidates=3`
+```
+Refer to [Mode Tuner README](https://github.com/nod-ai/shark-ai/tree/main/sharktuner/model_tuner) for detailed information on flags and MLIR files.
+
+**Dispatch Tuner**
+Use the Dispatch Tuner to tune a dispatch:
+```bash
```

**Comment:**
also here

---

**File:** `sharktuner/PYPI_README.md:54`

```diff
@@ -0,0 +1,50 @@
+# Sharktuner
+Sharktuner automates the dispatch-level and model-level tuning for the IREE (Intermediate Representation Execution Environment) ML Compiler on AMD GPUs.
+
+## Installation
+Install Sharktuner from PyPI:
+```shell
+pip install sharktuner
+```
+This will automatically install all required dependencies including IREE compiler and runtime.
+
+You can use nightly IREE's python bindings for the latest version:
+```shell
+pip install --upgrade --pre \
+    iree-base-compiler \
+    iree-base-runtime \
+    --find-links https://iree.dev/pip-release-links.html
+```
+
+Verify that the environment is set up correctly:
+```shell
+python -m model_tuner --help
+```
+or
+
+```shell
+python -m dispatch_tuner --help
+```
+
+**Model Tuner**
+Use the Model Tuner to tune a dispatch and a model:
+```bash
+python -m model_tuner double_mmt.mlir mmt_benchmark.mlir \
+    --compile-flags-file=compile_flags.txt \
+    --model-benchmark-flags-file=model_benchmark_flags.txt \
+    --devices=hip://0 \
+    --num-candidates=30 \
+    --model-tuner-num-dispatch-candidates=5 \
+    --model-tuner-num-model-candidates=3`
+```
+Refer to [Mode Tuner README](https://github.com/nod-ai/shark-ai/tree/main/sharktuner/model_tuner) for detailed information on flags and MLIR files.
+
+**Dispatch Tuner**
+Use the Dispatch Tuner to tune a dispatch:
+```bash
+python -m dispatch_tuner dispatch_sample.mlir dispatch_sample_benchmark.mlir \
+    --compile-flags-file=compile_flags.txt
+    --devices=hip://0 \
+    --num-candidates=30
+```
+Refer to [Dispatch Tuner README](https://github.com/nod-ai/shark-ai/tree/main/sharktuner/dispatch_tuner)for detailed information on flags and MLIR files.
```

**Comment:**
also here

---


---


## [PR #2478](https://github.com/nod-ai/amd-shark-ai/pull/2478): [tuner] add dispatch tuner for release

### Review Summary

**COMMENTED** (2025-10-10)

How did you test this?

**APPROVED** (2025-10-10)

Awesome, thanks for updating this


---


## [PR #2469](https://github.com/nod-ai/amd-shark-ai/pull/2469): [tuner][nfc] update the supported arch list

### Review Summary

**APPROVED** (2025-10-10)


---


## [PR #2420](https://github.com/nod-ai/amd-shark-ai/pull/2420): [Tuner] sync assembly format for match contraction op

### Review Summary

**APPROVED** (2025-10-03)


---


## [PR #2362](https://github.com/nod-ai/amd-shark-ai/pull/2362): [Tuner] use custom transform op for matching contraction op

### Review Summary

**COMMENTED** (2025-09-28)

Could we add some test that makes sure we look up the dimension values properly?

**COMMENTED** (2025-09-29)

**COMMENTED** (2025-09-29)

**APPROVED** (2025-09-30)

LGTM. Just checking, have you run the tuner and confirmed it works?


### Code Comments

**File:** `sharktuner/sharktuner/spec_builder.py:61`

```diff
@@ -57,25 +57,84 @@ def build_td_spec(
     if has_root_attr:
         op.opview.attributes[ROOT_OP_ATTR_NAME] = ir.UnitAttr.get(op.context)
 
-    # Get the names ssa names of operands to make sure they match in the
-    # template after string formatting.
-    captured_values: set[ir.Value] = set()
-    for operand in op.operands:
-        if operand in captured_values:
-            # TODO(Max191): Remove this warning when the transform for the
-            # `cast_compatible_dag_from_root` op fixes a bug in the matching
-            # logic that causes failure to match when the same operand is
-            # repeated. For now, still avoid adding duplicate SSA values to
-            # prevent parsing failure.
-            logging.warning(
-                f"Root op has repeated operand. This can cause failure to match in the resulting TD spec at compile time."
+    is_contraction = linalg.isa_contraction_op(op)
+    if is_contraction:
```

**Comment:**
```suggestion
    if linalg.isa_contraction_op(op):
```

---

**File:** `sharktuner/sharktuner/spec_builder.py:95`

```diff
@@ -57,25 +57,84 @@ def build_td_spec(
     if has_root_attr:
         op.opview.attributes[ROOT_OP_ATTR_NAME] = ir.UnitAttr.get(op.context)
 
-    # Get the names ssa names of operands to make sure they match in the
-    # template after string formatting.
-    captured_values: set[ir.Value] = set()
-    for operand in op.operands:
-        if operand in captured_values:
-            # TODO(Max191): Remove this warning when the transform for the
-            # `cast_compatible_dag_from_root` op fixes a bug in the matching
-            # logic that causes failure to match when the same operand is
-            # repeated. For now, still avoid adding duplicate SSA values to
-            # prevent parsing failure.
-            logging.warning(
-                f"Root op has repeated operand. This can cause failure to match in the resulting TD spec at compile time."
+    if linalg.isa_contraction_op(op):
+        # Temporary solution using custom contraction transform ops for contraction operations.
+        inputs = op.opview.operands
+        outputs = op.opview.results
+        lhs_type = str(ir.RankedTensorType(inputs[0].type).element_type)
+        rhs_type = str(ir.RankedTensorType(inputs[1].type).element_type)
+        output_type = str(ir.RankedTensorType(outputs[0].type).element_type)
+
+        contraction_dims = linalg.infer_contraction_dimensions(op)
+        indexing_maps = linalg.get_indexing_maps(op)
+        maps = [map_attr.value for map_attr in indexing_maps]
+        lhs_dims = common.get_map_result_dim_positions(maps[0])
+        rhs_dims = common.get_map_result_dim_positions(maps[1])
+        assert lhs_dims, "no lhs dimensions"
+        assert rhs_dims, "no rhs dimensions"
+        lhs_type_shape = ir.RankedTensorType(op.operands[0].type)
+        rhs_type_shape = ir.RankedTensorType(op.operands[1].type)
+
+        m_dims = [
+            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.m
+        ]
+        n_dims = [
+            rhs_type_shape.shape[rhs_dims.index(dim)] for dim in contraction_dims.n
+        ]
+        k_dims = [
+            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.k
+        ]
+        batch_dims = [
+            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.batch
+        ]
+
+        dims_equal_checks = []
+        if batch_dims:
+            dims_equal_checks.append(
+                f"transform.iree.match.dims_equal %batch, {batch_dims} : !transform.param<i64>"
             )
```

**Comment:**
Can we get rid of this branch and match with an empty array? Just to keep things uniform.

---

**File:** `sharktuner/tests/spec_builder_test.py:181`

```diff
@@ -112,6 +113,13 @@ def test_spec_builder(tuner_ctx: common.TunerContext) -> None:
     spec_str = str(spec_module)
     assert "@match_matmul -> @apply_op_config" in spec_str
     assert 'transform.annotate %arg0 "compilation_info" = %arg1' in spec_str
+    assert "transform.iree.match.contraction" in spec_str
+    assert "lhs_type =" in spec_str
+    assert "rhs_type =" in spec_str
+    assert "output_type =" in spec_str
+    assert "transform.iree.match.dims_equal %m_dims, [1024]" in spec_str
```

**Comment:**
Can we add a testcase that also exercises batch dim?

---

**File:** `sharktuner/sharktuner/spec_builder.py:89`

```diff
@@ -57,25 +57,82 @@ def build_td_spec(
     if has_root_attr:
         op.opview.attributes[ROOT_OP_ATTR_NAME] = ir.UnitAttr.get(op.context)
 
-    # Get the names ssa names of operands to make sure they match in the
-    # template after string formatting.
-    captured_values: set[ir.Value] = set()
-    for operand in op.operands:
-        if operand in captured_values:
-            # TODO(Max191): Remove this warning when the transform for the
-            # `cast_compatible_dag_from_root` op fixes a bug in the matching
-            # logic that causes failure to match when the same operand is
-            # repeated. For now, still avoid adding duplicate SSA values to
-            # prevent parsing failure.
-            logging.warning(
-                f"Root op has repeated operand. This can cause failure to match in the resulting TD spec at compile time."
-            )
-            continue
-        ssa_name = operand.get_name()
-        operand_type = operand.type
-        bbargs.append(f"{ssa_name}: {operand_type}")
-        captured_values.add(operand)
-    bbargs_str = ", ".join(bbargs)
+    if linalg.isa_contraction_op(op):
+        # Temporary solution using custom contraction transform ops for contraction operations.
+        inputs = op.opview.operands
+        outputs = op.opview.results
+        lhs_type = str(ir.RankedTensorType(inputs[0].type).element_type)
+        rhs_type = str(ir.RankedTensorType(inputs[1].type).element_type)
+        output_type = str(ir.RankedTensorType(outputs[0].type).element_type)
+
+        contraction_dims = linalg.infer_contraction_dimensions(op)
+        indexing_maps = linalg.get_indexing_maps(op)
+        maps = [map_attr.value for map_attr in indexing_maps]
+        lhs_dims = common.get_map_result_dim_positions(maps[0])
+        rhs_dims = common.get_map_result_dim_positions(maps[1])
+        assert lhs_dims, "no lhs dimensions"
+        assert rhs_dims, "no rhs dimensions"
+        lhs_type_shape = ir.RankedTensorType(op.operands[0].type)
+        rhs_type_shape = ir.RankedTensorType(op.operands[1].type)
+
+        m_dims = [
+            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.m
+        ]
+        n_dims = [
+            rhs_type_shape.shape[rhs_dims.index(dim)] for dim in contraction_dims.n
+        ]
+        k_dims = [
+            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.k
+        ]
+        batch_dims = [
+            lhs_type_shape.shape[lhs_dims.index(dim)] for dim in contraction_dims.batch
+        ]
```

**Comment:**
Are you sure the lhs always knows about the batch dim? In existing tuning specs for Unet, we sometimes had the RHS being broadcast: https://github.com/nod-ai/sdxl-scripts/blob/4fa7ccbc3de4873c0751a48ef9b4b86a7f24428e/int8-model/specs/attention_and_matmul_spec.mlir#L632-L648

I think it could also be that LHS gets broadcast.

---


---


## [PR #2330](https://github.com/nod-ai/amd-shark-ai/pull/2330): [tuner] NFC Fixes

### Review Summary

**APPROVED** (2025-09-24)


---


## [PR #2272](https://github.com/nod-ai/amd-shark-ai/pull/2272): [tuner] use subgroup basis for lowering config

### Review Summary

**COMMENTED** (2025-09-18)

**COMMENTED** (2025-09-18)

**COMMENTED** (2025-09-23)

**COMMENTED** (2025-09-23)

**COMMENTED** (2025-09-23)

**COMMENTED** (2025-09-23)

**COMMENTED** (2025-09-23)

**APPROVED** (2025-09-23)

LGTM % a TODO and mypy errors

**COMMENTED** (2025-09-23)


### Code Comments

**File:** `sharktuner/sharktuner/common.py:272`

```diff
@@ -256,10 +256,22 @@ def get_lowering_config(
                     assert (
                         False
                     ), f"Unsupported type for key '{key}': {type(value).__name__}"
-            case "subgroup_m_count" | "subgroup_n_count":
-                if isinstance(value, int):
-                    promoted_value = tuner_ctx.type.getI64(value)
-                elif not isinstance(value, tuner_ctx.type.i64):
+            case "subgroup_basis":
+                if isinstance(value, list) and len(value) == 2:
+                    counts, mapping = value
+                    if isinstance(counts, list) and isinstance(mapping, list):
+                        counts_attr = ir.ArrayAttr.get(
+                            [tuner_ctx.type.getI64(x) for x in counts]
+                        )
+                        mapping_attr = ir.ArrayAttr.get(
+                            [tuner_ctx.type.getI64(x) for x in mapping]
+                        )
+                        promoted_value = ir.ArrayAttr.get([counts_attr, mapping_attr])
+                    else:
+                        assert (
+                            False
```

**Comment:**
We don't need this control flow -- we can assert the negation of the `if` condition above

---

**File:** `sharktuner/sharktuner/common.py:268`

```diff
@@ -256,10 +256,22 @@ def get_lowering_config(
                     assert (
                         False
                     ), f"Unsupported type for key '{key}': {type(value).__name__}"
-            case "subgroup_m_count" | "subgroup_n_count":
-                if isinstance(value, int):
-                    promoted_value = tuner_ctx.type.getI64(value)
-                elif not isinstance(value, tuner_ctx.type.i64):
+            case "subgroup_basis":
+                if isinstance(value, list) and len(value) == 2:
+                    counts, mapping = value
+                    if isinstance(counts, list) and isinstance(mapping, list):
+                        counts_attr = ir.ArrayAttr.get(
+                            [tuner_ctx.type.getI64(x) for x in counts]
+                        )
+                        mapping_attr = ir.ArrayAttr.get(
+                            [tuner_ctx.type.getI64(x) for x in mapping]
+                        )
```

**Comment:**
Maybe add a helper to create an `I64ArrayAttr`? 

---

**File:** `sharktuner/sharktuner/constraint_generator.py:235`

```diff
@@ -219,6 +226,13 @@ def set_cdim_tile_sizes(tile_sizes, contraction_dims, csizes):
                     for d in contraction_dims.k
                 ),
             ]
+        # Setting subgroup basis.
+        subgroup_basis_counts = [1] * num_loops
+        m_dim = contraction_dims.m[-1]
+        subgroup_basis_counts[m_dim] = lookup(sg_m_cnt)
+        n_dim = contraction_dims.n[-1]
+        subgroup_basis_counts[n_dim] = lookup(sg_n_cnt)
```

**Comment:**
I think there's an upcoming PR to allow distributing subgroups on multiple m dims -- I wonder if the `sg_m_cnt` logic will get out of date very soon: https://github.com/iree-org/iree/pull/22000

cc: @Groverkss 

---

**File:** `sharktuner/sharktuner/common.py:263`

```diff
@@ -256,10 +259,17 @@ def get_lowering_config(
                     assert (
                         False
                     ), f"Unsupported type for key '{key}': {type(value).__name__}"
-            case "subgroup_m_count" | "subgroup_n_count":
-                if isinstance(value, int):
-                    promoted_value = tuner_ctx.type.getI64(value)
-                elif not isinstance(value, tuner_ctx.type.i64):
+            case "subgroup_basis":
+                if isinstance(value, list) and len(value) == 2:
```

**Comment:**
Wasn't this a `std::tuple` on the iree side? I'm surprised value became a list.

---

**File:** `sharktuner/sharktuner/constraint_generator.py:235`

```diff
@@ -219,6 +226,13 @@ def set_cdim_tile_sizes(tile_sizes, contraction_dims, csizes):
                     for d in contraction_dims.k
                 ),
             ]
+        # Setting subgroup basis.
+        subgroup_basis_counts = [1] * num_loops
+        m_dim = contraction_dims.m[-1]
+        subgroup_basis_counts[m_dim] = lookup(sg_m_cnt)
+        n_dim = contraction_dims.n[-1]
+        subgroup_basis_counts[n_dim] = lookup(sg_n_cnt)
```

**Comment:**
Has this been done @bangtianliu ?

---

**File:** `sharktuner/sharktuner/constraint_generator.py:383`

```diff
@@ -364,20 +378,33 @@ def generate_attention_solutions(
         workgroup_tile_sizes[opinfo.n_dims[-1]] = lookup(n_var)
         reduction_tile_sizes[opinfo.k2_dims[-1]] = lookup(k_var)
 
+        subgroup_basis_counts = [1] * opinfo.domain_rank
+        subgroup_basis_mapping = list(range(opinfo.domain_rank))
```

**Comment:**
How do we know it's always in this order?

---

**File:** `sharktuner/sharktuner/constraint_generator.py:235`

```diff
@@ -219,6 +226,13 @@ def set_cdim_tile_sizes(tile_sizes, contraction_dims, csizes):
                     for d in contraction_dims.k
                 ),
             ]
+        # Setting subgroup basis.
+        subgroup_basis_counts = [1] * num_loops
+        m_dim = contraction_dims.m[-1]
+        subgroup_basis_counts[m_dim] = lookup(sg_m_cnt)
+        n_dim = contraction_dims.n[-1]
+        subgroup_basis_counts[n_dim] = lookup(sg_n_cnt)
```

**Comment:**
Can you add a TODO comment then?

---

**File:** `sharktuner/sharktuner/constraint_generator.py:383`

```diff
@@ -364,20 +378,33 @@ def generate_attention_solutions(
         workgroup_tile_sizes[opinfo.n_dims[-1]] = lookup(n_var)
         reduction_tile_sizes[opinfo.k2_dims[-1]] = lookup(k_var)
 
+        subgroup_basis_counts = [1] * opinfo.domain_rank
+        subgroup_basis_mapping = list(range(opinfo.domain_rank))
```

**Comment:**
The code for attention calls the `projectBasis` helper though: https://github.com/iree-org/iree/pull/21912/files#diff-e491707653524e61bd22ba47ca5bd8414065b70bd1c6b85f45947929d7a49aafR1579-R1586

---

**File:** `sharktuner/sharktuner/constraint_generator.py:383`

```diff
@@ -364,20 +378,33 @@ def generate_attention_solutions(
         workgroup_tile_sizes[opinfo.n_dims[-1]] = lookup(n_var)
         reduction_tile_sizes[opinfo.k2_dims[-1]] = lookup(k_var)
 
+        subgroup_basis_counts = [1] * opinfo.domain_rank
+        subgroup_basis_mapping = list(range(opinfo.domain_rank))
```

**Comment:**
Oh, I think I see, you do it later on this code. Makes sense then.

---

**File:** `sharktuner/sharktuner/constraint_generator.py:230`

```diff
@@ -219,6 +226,14 @@ def set_cdim_tile_sizes(tile_sizes, contraction_dims, csizes):
                     for d in contraction_dims.k
                 ),
             ]
+        # Setting subgroup basis.
+        # TODO (Bangtian) : sync changes from IREE PR: https://github.com/iree-org/iree/pull/22000.
```

**Comment:**
```suggestion
        # TODO(Bangtian): Sync changes from IREE PR: https://github.com/iree-org/iree/pull/22000.
```

---


---


## [PR #2122](https://github.com/nod-ai/amd-shark-ai/pull/2122): [tuner]: add target info to tuner

### Review Summary

**COMMENTED** (2025-08-29)

**APPROVED** (2025-09-04)

Looks good overall, but let's wait for @Max191 before merging


### Code Comments

**File:** `sharktuner/sharktuner/common.py:31`

```diff
@@ -25,6 +25,31 @@
 from iree.compiler._mlir_libs._mlir import ir  # type: ignore
 
 
+@dataclass
+class GPUTargetInfo:
+    """
+    GPU target information extracted from an executable target attribute.
```

**Comment:**
Why do we need a separate definition on the python side in addition to the class defined by python bindings? Both represent contain the same information.

---

**File:** `sharktuner/sharktuner/dispatch_constraints.py:204`

```diff
@@ -200,7 +200,13 @@ def generate_vector_distribute_constraints(
     wg_x, wg_y, wg_z = workgroup_size
     wg_threads = z3.Int("wg_threads")
     constraints = [wg_threads == wg_x * wg_y * wg_z]
-    constraints += [subgroup_size == 64, wg_threads <= 1024]
+    # Use minimum subgroup size for consistency with IREE side.
+    # https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.td#L623-L632
```

**Comment:**
Use a permanent link, otherwise this will get out of date eventually and won't point to the same code

---

**File:** `sharktuner/sharktuner/candidate_gen.py:201`

```diff
@@ -190,28 +190,25 @@ def generate_configs_and_td_specs(
         spec_builder.get_placeholder_spec(input_module.context)
     ]
 
-    # Get the MMA intrinisic intructions supported by the target.
+    # Get GPU target information from the executable variant operation.
     variant_op_list = iree_codegen.get_executable_variant_ops(input_module)
     assert len(variant_op_list) == 1, "Expect one executable variant op"
     variant_op = variant_op_list[0]
-    mma_intrinsics = iree_codegen.query_mma_intrinsics(variant_op)
+    executable_variant_op = variant_op.opview
+    target = executable_variant_op.target
+    target_info = iree_gpu.TargetInfo.get_gpu_target_info(target)
 
-    # Collect both mma and derived virtual intrinsics.
-    all_intrinsics = []
-    for intrinsic in mma_intrinsics:
-        all_intrinsics.append(intrinsic)
-        mma_attr = iree_gpu.MMAAttr.get(intrinsic)
-        virtual_mma_intrinsics = mma_attr.get_virtual_intrinsics()
-        all_intrinsics.extend(virtual_mma_intrinsics)
+    if target_info.arch not in ["gfx942", "gfx1100"]:
```

**Comment:**
Can you open an issue to also test this on gfx950 and gfx1201?

---

**File:** `sharktuner/sharktuner/dispatch_constraints.py:205`

```diff
@@ -200,7 +200,13 @@ def generate_vector_distribute_constraints(
     wg_x, wg_y, wg_z = workgroup_size
     wg_threads = z3.Int("wg_threads")
     constraints = [wg_threads == wg_x * wg_y * wg_z]
-    constraints += [subgroup_size == 64, wg_threads <= 1024]
+    # Use minimum subgroup size for consistency with IREE side.
+    # https://github.com/iree-org/iree/blob/c37c680ae6e71f715bd7c540909155061bc44491/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.td#L623-L632
+    target_subgroup_size = min(gpu_target_info.subgroup_size_choices)
```

**Comment:**
Fine for now, but we should allow both in the future

---

**File:** `sharktuner/sharktuner/dispatch_constraints.py:293`

```diff
@@ -282,7 +288,11 @@ def generate_tile_and_fuse_constraints(
     wg_x, wg_y, wg_z = workgroup_size
     wg_threads = wg_x
     constraints = [wg_y == 1, wg_z == 1]
-    constraints += [subgroup_size == 64, wg_threads <= 1024]
+    target_subgroup_size = min(gpu_target_info.subgroup_size_choices)
+    constraints += [
+        subgroup_size == target_subgroup_size,
```

**Comment:**
Also here

---

**File:** `sharktuner/tests/constraint_generator_test.py:31`

```diff
@@ -28,6 +28,25 @@
 from sharktuner.test_utils import tuner_ctx
 
 
+@pytest.fixture(scope="function")
```

**Comment:**
Why do we have to specify scope? I haven't seen `scope` specified elsewhere. I think `function` is the default, no?

---

**File:** `sharktuner/tests/dispatch_constraints_test.py:25`

```diff
@@ -21,6 +22,25 @@
 from sharktuner.test_utils import tuner_ctx
 
 
+@pytest.fixture(scope="function")
```

**Comment:**
also here

---


---


## [PR #2104](https://github.com/nod-ai/amd-shark-ai/pull/2104): [tuner] add acc layout match to constraint generation for attention

### Review Summary

**APPROVED** (2025-08-27)

**COMMENTED** (2025-08-27)


### Code Comments

**File:** `sharktuner/sharktuner/dispatch_constraints.py:478`

```diff
@@ -468,10 +469,14 @@ def generate_attention_vector_distribute_constraints(
             mma_intrinsics=mma_intrinsics,
             lhs_layout=pv_mma_lhs_layout,
             rhs_layout=pv_mma_rhs_layout,
-            acc_layout=None,
+            acc_layout=pv_mma_acc_layout,
         )
     ]
 
+    constraints += [
+        match_layout(qk_mma_acc_layout, pv_mma_acc_layout),
+    ]
```

**Comment:**
Won't this fit on a single line?
```suggestion
    constraints += [match_layout(qk_mma_acc_layout, pv_mma_acc_layout)]
```
?

---


---


## [PR #1909](https://github.com/nod-ai/amd-shark-ai/pull/1909): [tuner] improve support for attention op

### Review Summary

**APPROVED** (2025-07-28)


---


## [PR #1772](https://github.com/nod-ai/amd-shark-ai/pull/1772): [tuner] add support for attention op

### Review Summary

**COMMENTED** (2025-07-21)

Looks good overall, just one code organization issue

**APPROVED** (2025-07-21)


### Code Comments

**File:** `sharktuner/sharktuner/common.py:184`

```diff
@@ -148,6 +150,40 @@ class ContractionDimensions:
     batch: list[int] = field(default_factory=list)
 
 
+@dataclass
+class MatmulShapeType:
+    m: int
+    n: int
+    k: int
+    lhs_type: ir.IntegerType | ir.FloatType
+    rhs_type: ir.IntegerType | ir.FloatType
+    acc_type: ir.IntegerType | ir.FloatType
+
+
+@dataclass
+class AttentionOpInfo:
+    domain_rank: int
+    batch_dims: list[int]
+    m_dims: list[int]
+    n_dims: list[int]
+    k1_dims: list[int]
+    k2_dims: list[int]
+
+
+@dataclass
+class GPUMMASchedule:
+    m_size: z3.ArithRef
+    n_size: z3.ArithRef
+    k_size: z3.ArithRef
+
+    m_subgroup_counts: z3.ArithRef
+    n_subgroup_counts: z3.ArithRef
+
+    m_tile_size: z3.ArithRef
+    n_tile_size: z3.ArithRef
+    k_tile_size: z3.ArithRef
```

**Comment:**
Can we take this class out of `common.py`? The unintuitive part to me is that all of these are z3 types, even though mma schedule is more general than that. I think this belongs to some file related strictly to constraint generation

---


---


## [PR #1733](https://github.com/nod-ai/amd-shark-ai/pull/1733): [tuner] add parser for attention op

### Review Summary

**APPROVED** (2025-06-30)


### Code Comments

**File:** `sharktuner/sharktuner/dispatch_parser.py:92`

```diff
@@ -81,3 +81,13 @@ def has_valid_root_op(self) -> bool:
         ):
             return False
         return True
+
+
+class AttentionOpInterfaceParser(DispatchParser):
+    def __init__(self, root_op: ir.Operation):
+        super().__init__(root_op)
+
+    def has_valid_root_op(self) -> bool:
+        root_op = self.get_root_op()
+        # TODO(Bangtian): will see whether to use `iree_codegen.isa_attention_op`
```

**Comment:**
```suggestion
        # TODO(Bangtian): Switch to `iree_codegen.isa_attention_op` once available.
```

---


---


## [PR #1641](https://github.com/nod-ai/amd-shark-ai/pull/1641): [tuner] use config list for constraint generators

### Review Summary

**COMMENTED** (2025-06-13)

**COMMENTED** (2025-06-17)

**APPROVED** (2025-06-17)

LGTM % small issue


### Code Comments

**File:** `sharktuner/sharktuner/candidate_gen.py:48`

```diff
@@ -42,9 +42,13 @@ class DispatchTuner(dispatch_parser.DispatchParser):
     @abstractmethod
     def get_td_spec(
         self,
-        compilation_info: iree_codegen.CompilationInfoAttr,
+        config: list[tuple[str, ir.Attribute]],
     ) -> ir.Module:
-        """Generate a transform dialect spec that applies the compilation info attr."""
+        """
+        Generate a transform dialect spec from a config list.
```

**Comment:**
```suggestion
        Generates a transform dialect spec from a config list.
```
```suggestion
        Generate a transform dialect spec from a config list.
```

---

**File:** `sharktuner/tests/constraint_generator_test.py:199`

```diff
@@ -181,17 +181,21 @@ def test_generate_solutions_tile_and_fuse_contraction_padding(
 
         assert len(solutions) > 0, "No solutions generated with TileAndFuse pipeline."
         assert all(
-            isinstance(sol, iree_codegen.CompilationInfoAttr) for sol in solutions
-        )
+            len(pair) == 1 and pair[0][0] == "compilation_info" for pair in solutions
+        ), "Each solution must be a single-item list with key 'compilation_info'"
 
+        compilation_infos = [pair[0][1] for pair in solutions]
+        assert all(
+            isinstance(sol, iree_codegen.CompilationInfoAttr)
+            for sol in compilation_infos
+        )
         assert all(
-            "padding =" in str(sol.lowering_config) for sol in solutions
+            "padding =" in str(sol.lowering_config) for sol in compilation_infos
         ), "Not all lowering configs have padding option."
-
         assert all(
             [int(x) for x in sol.lowering_config.attributes["promote_operands"]]
             == [0, 1, 2]
-            for sol in solutions
+            for sol in compilation_infos
         ), "Not all lowering configs have promote_operands = [0, 1, 2]."
```

**Comment:**
I think we could simplify this code ba bunch by writing a for loop and then checking each pair separately (instead of relying on `all`). Once you name the pair elements the code will become more readable.

---

**File:** `sharktuner/tests/constraint_generator_test.py:271`

```diff
@@ -249,15 +253,21 @@ def test_generate_solutions_tile_and_fuse_conv_padding(
 
         assert len(solutions) > 0, "No solutions generated with TileAndFuse pipeline."
         assert all(
-            isinstance(sol, iree_codegen.CompilationInfoAttr) for sol in solutions
+            len(pair) == 1 and pair[0][0] == "compilation_info" for pair in solutions
+        ), "Each solution must be a single-item list with key 'compilation_info'"
+
+        compilation_infos = [pair[0][1] for pair in solutions]
+        assert all(
+            isinstance(sol, iree_codegen.CompilationInfoAttr)
+            for sol in compilation_infos
         )
         assert all(
-            "padding =" in str(sol.lowering_config) for sol in solutions
+            "padding =" in str(sol.lowering_config) for sol in compilation_infos
         ), "Not all lowering configs have padding option"
         assert all(
             [int(x) for x in sol.lowering_config.attributes["promote_operands"]]
             == [0, 1, 2]
-            for sol in solutions
+            for sol in compilation_infos
         ), "Not all lowering configs have promote_operands = [0, 1, 2]"
```

**Comment:**
Similar here

---

**File:** `sharktuner/tests/constraint_generator_test.py:206`

```diff
@@ -180,19 +180,30 @@ def test_generate_solutions_tile_and_fuse_contraction_padding(
         )
 
         assert len(solutions) > 0, "No solutions generated with TileAndFuse pipeline."
-        assert all(
-            isinstance(sol, iree_codegen.CompilationInfoAttr) for sol in solutions
-        )
-
-        assert all(
-            "padding =" in str(sol.lowering_config) for sol in solutions
-        ), "Not all lowering configs have padding option."
-
-        assert all(
-            [int(x) for x in sol.lowering_config.attributes["promote_operands"]]
-            == [0, 1, 2]
-            for sol in solutions
-        ), "Not all lowering configs have promote_operands = [0, 1, 2]."
+        for solution in solutions:
+            assert len(solution) == 1, f"Expected a single-item list, got: {solution}"
+            config = solution[0]
+            assert isinstance(
+                config, common.TuningConfiguration
+            ), f"Expected TuningConfiguration, got: {type(config)}"
+
+            assert (
+                config.name == "compilation_info"
+            ), f"Expected key 'compilation_info', got: {config.name}"
+            assert isinstance(
+                config.configuration, iree_codegen.CompilationInfoAttr
+            ), f"Expected CompilationInfoAttr, got: {type(config.configuration)}"
+
+            lowering_config = config.configuration.lowering_config
+            assert "padding =" in str(
+                lowering_config
+            ), f"Missing padding in lowering config: {lowering_config}"
+            promote = [int(x) for x in lowering_config.attributes["promote_operands"]]
+            assert promote == [
+                0,
+                1,
+                2,
+            ], f"Expected promote_operands = [0, 1, 2], got: {promote}"
```

**Comment:**
IIRC assert already prints the expected and the actual values when you have `==`, could you check if we need this string?

---

**File:** `sharktuner/tests/constraint_generator_test.py:285`

```diff
@@ -248,17 +259,30 @@ def test_generate_solutions_tile_and_fuse_conv_padding(
         )
 
         assert len(solutions) > 0, "No solutions generated with TileAndFuse pipeline."
-        assert all(
-            isinstance(sol, iree_codegen.CompilationInfoAttr) for sol in solutions
-        )
-        assert all(
-            "padding =" in str(sol.lowering_config) for sol in solutions
-        ), "Not all lowering configs have padding option"
-        assert all(
-            [int(x) for x in sol.lowering_config.attributes["promote_operands"]]
-            == [0, 1, 2]
-            for sol in solutions
-        ), "Not all lowering configs have promote_operands = [0, 1, 2]"
+        for solution in solutions:
+            assert len(solution) == 1, f"Expected a single-item list, got: {solution}"
+            config = solution[0]
+            assert isinstance(
+                config, common.TuningConfiguration
+            ), f"Expected TuningConfiguration, got: {type(config)}"
+
+            assert (
+                config.name == "compilation_info"
+            ), f"Expected key 'compilation_info', got: {config.name}"
+            assert isinstance(
+                config.configuration, iree_codegen.CompilationInfoAttr
+            ), f"Expected CompilationInfoAttr, got: {type(config.configuration)}"
+
+            lowering_config = config.configuration.lowering_config
+            assert "padding =" in str(
+                lowering_config
+            ), f"Missing padding in lowering config: {lowering_config}"
+            promote = [int(x) for x in lowering_config.attributes["promote_operands"]]
+            assert promote == [
+                0,
+                1,
+                2,
+            ], f"Expected promote_operands = [0, 1, 2], got: {promote}"
```

**Comment:**
Also here

---

**File:** `sharktuner/sharktuner/candidate_gen.py:55`

```diff
@@ -42,9 +42,18 @@ class DispatchTuner(dispatch_parser.DispatchParser):
     @abstractmethod
     def get_td_spec(
         self,
-        compilation_info: iree_codegen.CompilationInfoAttr,
+        config_list: list[common.TuningConfiguration],
     ) -> ir.Module:
-        """Generate a transform dialect spec that applies the compilation info attr."""
+        """
+        Generates a transform dialect spec from a list of TuningConfiguration objects.
+
+        Each TuningConfiguration specifies a name (e.g., "compilation_info") and
+        its corresponding MLIR attribute (e.g., CompilationInfoAttr) to be applied
+        to the dispatch root operation.
+
+        Example:
+            TuningConfiguration(name="compilation_info", configuration=CompilationInfoAttr(...))
```

**Comment:**
I'm not sure what this example is trying to show here -- I think this belongs to the documentation for `common.TuningConfirguration`.

---


---


## [PR #1602](https://github.com/nod-ai/amd-shark-ai/pull/1602): [tuner] build td spec using config list

### Review Summary

**COMMENTED** (2025-06-09)

What's the overall plan?

**APPROVED** (2025-06-10)

LGTM, thanks for adding tests


### Code Comments

**File:** `sharktuner/sharktuner/spec_builder.py:131`

```diff
@@ -76,35 +76,57 @@ def build_td_spec(
         bbargs.append(f"{ssa_name}: {operand_type}")
         captured_values.add(operand)
     bbargs_str = ", ".join(bbargs)
-    spec_text = f"""
+
+    config_lines = []
+    yield_vars = []
+    for i, (key, attr) in enumerate(config_list):
+        config_var = f"%{key}_{i}"
+        config_lines.append(
+            f"{config_var} = transform.param.constant {attr} -> !transform.any_param"
+        )
+        yield_vars.append(config_var)
+    config_block = "\n                ".join(config_lines)
+    yield_list = ", ".join(["%cont"] + yield_vars)
+    yield_types = ", ".join(
+        ["!transform.any_op"] + ["!transform.any_param"] * len(yield_vars)
+    )
+
+    annotation_args = ", ".join(
+        f"%cfg_{i}: !transform.any_param {{transform.readonly}}"
+        for i in range(len(config_list))
+    )
+    annotation_lines = "\n".join(
+        f'                transform.annotate %op "{key}" = %cfg_{i} : !transform.any_op, !transform.any_param'
+        for i, (key, _) in enumerate(config_list)
+    )
+
+    spec_text = f"""\
         module attributes {{ transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint }} {{
-            // Annotation Transform
-            transform.named_sequence @apply_op_config(%op: !transform.any_op {{transform.readonly}},
-                                                        %config: !transform.any_param {{transform.readonly}}) {{
-                transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
-                transform.yield
-            }}
+        // Annotation Transform
+        transform.named_sequence @apply_op_config(%op: !transform.any_op {{transform.readonly}}, {annotation_args}) {{
+        {annotation_lines}
+            transform.yield
+        }}
 
-            // Custom Op Matcher
-            transform.named_sequence @{func_name}(%cont: !transform.any_op {{transform.readonly}})
-                -> (!transform.any_op, !transform.any_param) {{
-                %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %cont {{
-                ^bb0({bbargs_str}):
-                {root_operation}
-                }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
-                %config = transform.param.constant {compilation_info} -> !transform.any_param
-                transform.yield %cont, %config : !transform.any_op, !transform.any_param
-            }}
+        // Custom Op Matcher
+        transform.named_sequence @{func_name}(%cont: !transform.any_op {{transform.readonly}})
+            -> ({yield_types}) {{
+            %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %cont {{
+              ^bb0({bbargs_str}):
+              {root_operation}
+            }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
+            {config_block}
+            transform.yield {yield_list} : {yield_types}
+        }}
 
-            // Entry Point
-            transform.named_sequence
-            @__kernel_config(%variant_op: !transform.any_op {{transform.consumed}}) -> !transform.any_op
-                attributes {{ iree_codegen.tuning_spec_entrypoint }} {{
-                %res = transform.foreach_match in %variant_op
-                    @{func_name} -> @apply_op_config
-                : (!transform.any_op) -> !transform.any_op
-                transform.yield %res : !transform.any_op
-            }}
+        // Entry Point
+        transform.named_sequence
+        @__kernel_config(%variant_op: !transform.any_op {{transform.consumed}}) -> !transform.any_op
+            attributes {{ iree_codegen.tuning_spec_entrypoint }} {{
+            %res = transform.foreach_match in %variant_op
+                @{func_name} -> @apply_op_config
+            : (!transform.any_op) -> !transform.any_op
+            transform.yield %res : !transform.any_op
         }}
-        """
+    }}"""
```

**Comment:**
How could we test this? I think it should be possible to generate some TD IR with more than one config and then make sure that is parses.

---


---


## [PR #1307](https://github.com/nod-ai/amd-shark-ai/pull/1307): [tuner][NFC] remove walk function over input module

### Review Summary

**APPROVED** (2025-04-25)


---


## [PR #1304](https://github.com/nod-ai/amd-shark-ai/pull/1304): [tuner] get the function name from the func.func in the input module

### Review Summary

**APPROVED** (2025-04-24)

Nice

**COMMENTED** (2025-04-24)

**COMMENTED** (2025-04-24)

**COMMENTED** (2025-04-24)

**COMMENTED** (2025-04-24)


### Code Comments

**File:** `tuner/tuner/dispatch_parser.py:34`

```diff
@@ -29,10 +29,17 @@ def parse_mlir(mlir_text: str, ctx: TunerContext) -> ir.Module:
 class DispatchParser(metaclass=ABCMeta):
     def __init__(self, root_op: ir.Operation):
         self._root_op = root_op
+        func_op = self._root_op.parent
+        assert func_op.name == "func.func", f"Expected func.func, got {func_op.name}"
+        func_name_attr = func_op.attributes["sym_name"]
```

**Comment:**
This would be nice to have but I wouldn't wait for it here. If we are going to use it in one or two places, the ROI is kind of low IMO. That could be a good starter task if you want to open an issue for that.

---

**File:** `tuner/tuner/dispatch_parser.py:34`

```diff
@@ -29,10 +29,17 @@ def parse_mlir(mlir_text: str, ctx: TunerContext) -> ir.Module:
 class DispatchParser(metaclass=ABCMeta):
     def __init__(self, root_op: ir.Operation):
         self._root_op = root_op
+        func_op = self._root_op.parent
+        assert func_op.name == "func.func", f"Expected func.func, got {func_op.name}"
+        func_name_attr = func_op.attributes["sym_name"]
```

**Comment:**
You can cast to FuncOp though, right?

---

**File:** `tuner/tuner/dispatch_parser.py:34`

```diff
@@ -29,10 +29,17 @@ def parse_mlir(mlir_text: str, ctx: TunerContext) -> ir.Module:
 class DispatchParser(metaclass=ABCMeta):
     def __init__(self, root_op: ir.Operation):
         self._root_op = root_op
+        func_op = self._root_op.parent
+        assert func_op.name == "func.func", f"Expected func.func, got {func_op.name}"
+        func_name_attr = func_op.attributes["sym_name"]
```

**Comment:**
It's not a constructor though that we want to use, IIUC. cc: @makslevental 

---

**File:** `tuner/tuner/dispatch_parser.py:34`

```diff
@@ -29,10 +29,17 @@ def parse_mlir(mlir_text: str, ctx: TunerContext) -> ir.Module:
 class DispatchParser(metaclass=ABCMeta):
     def __init__(self, root_op: ir.Operation):
         self._root_op = root_op
+        func_op = self._root_op.parent
+        assert func_op.name == "func.func", f"Expected func.func, got {func_op.name}"
+        func_name_attr = func_op.attributes["sym_name"]
```

**Comment:**
@bangtianliu Maks suggested `func_op.opview` should give us the function op object -- can you check that?

---


---


## [PR #1290](https://github.com/nod-ai/amd-shark-ai/pull/1290): [tuner] use python binding to get indexing maps for root op

### Review Summary

**APPROVED** (2025-04-21)

Nice, looks much cleaner now


---


## [PR #1289](https://github.com/nod-ai/amd-shark-ai/pull/1289): [tuner] add tests for named ops

### Review Summary

**COMMENTED** (2025-04-21)

**APPROVED** (2025-04-21)


### Code Comments

**File:** `tuner/tuner/dispatch_parser_test.py:159`

```diff
@@ -128,6 +128,114 @@ def test_get_contraction_operation(tuner_ctx: common.TunerContext) -> None:
     assert shapes.matmul_size.K == [15, 256]
 
 
+def test_get_matmul_named_op(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    with ir.Location.unknown(context):
+        module = ir.Module.create()
+        f16 = ir.F16Type.get()
+        f32 = ir.F32Type.get()
+
+        with ir.InsertionPoint(module.body):
+            a_type = ir.RankedTensorType.get((16, 64), f16)
+            b_type = ir.RankedTensorType.get((64, 32), f16)
+            c_type = ir.RankedTensorType.get((16, 32), f32)
+
+            dim_m = ir.AffineDimExpr.get(0)
+            dim_n = ir.AffineDimExpr.get(1)
+            dim_k = ir.AffineDimExpr.get(2)
+            a_map = ir.AffineMap.get(3, 0, [dim_m, dim_k])
+            b_map = ir.AffineMap.get(3, 0, [dim_k, dim_n])
+            c_map = ir.AffineMap.get(3, 0, [dim_m, dim_n])
+
+            @func.FuncOp.from_py_func(a_type, b_type)
+            def named_matmul(a, b):
+                zero = arith.ConstantOp(f32, 0.0).result
+                init = tensor.EmptyOp(c_type.shape, c_type.element_type).result
+
+                filled = linalg.FillOp(
+                    result_tensors=[c_type],
+                    inputs=[zero],
+                    outputs=[init],
+                ).results[0]
```

**Comment:**
We can provide this as the third function argument

---

**File:** `tuner/tuner/dispatch_parser_test.py:205`

```diff
@@ -128,6 +128,114 @@ def test_get_contraction_operation(tuner_ctx: common.TunerContext) -> None:
     assert shapes.matmul_size.K == [15, 256]
 
 
+def test_get_matmul_named_op(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    with ir.Location.unknown(context):
+        module = ir.Module.create()
+        f16 = ir.F16Type.get()
+        f32 = ir.F32Type.get()
+
+        with ir.InsertionPoint(module.body):
+            a_type = ir.RankedTensorType.get((16, 64), f16)
+            b_type = ir.RankedTensorType.get((64, 32), f16)
+            c_type = ir.RankedTensorType.get((16, 32), f32)
+
+            dim_m = ir.AffineDimExpr.get(0)
+            dim_n = ir.AffineDimExpr.get(1)
+            dim_k = ir.AffineDimExpr.get(2)
+            a_map = ir.AffineMap.get(3, 0, [dim_m, dim_k])
+            b_map = ir.AffineMap.get(3, 0, [dim_k, dim_n])
+            c_map = ir.AffineMap.get(3, 0, [dim_m, dim_n])
+
+            @func.FuncOp.from_py_func(a_type, b_type)
+            def named_matmul(a, b):
+                zero = arith.ConstantOp(f32, 0.0).result
+                init = tensor.EmptyOp(c_type.shape, c_type.element_type).result
+
+                filled = linalg.FillOp(
+                    result_tensors=[c_type],
+                    inputs=[zero],
+                    outputs=[init],
+                ).results[0]
+
+                matmul_op = linalg.MatmulOp(
+                    result_tensors=[c_type],
+                    inputs=[a, b],
+                    outputs=[filled],
+                    indexing_maps=[a_map, b_map, c_map],
+                )
+                matmul_op.operation.attributes["root_op"] = ir.UnitAttr.get()
+
+        root_op_list = iree_codegen.get_tuner_root_ops(module)
+        assert len(root_op_list) == 1, "Expected one root op"
+        root_op = root_op_list[0]
+
+        parser = dispatch_parser.ContractionOpInterfaceParser(root_op)
+        shapes = parser.get_problem_size()
+
+        assert shapes.matmul_size.B == []
+        assert shapes.matmul_size.M == [16]
+        assert shapes.matmul_size.N == [32]
+        assert shapes.matmul_size.K == [64]
+        assert shapes.lhs_type.shape == [16, 64]
+        assert isinstance(shapes.lhs_type.element_type, ir.F16Type)
+        assert shapes.rhs_type.shape == [64, 32]
+        assert isinstance(shapes.rhs_type.element_type, ir.F16Type)
+        assert shapes.res_type.shape == [16, 32]
+        assert isinstance(shapes.res_type.element_type, ir.F32Type)
+
+
+def test_get_named_contraction_op():
+    with ir.Context(), ir.Location.unknown():
+        module = ir.Module.create()
+        f32 = ir.F32Type.get()
+
+        with ir.InsertionPoint(module.body):
+            lhs_type = ir.RankedTensorType.get((5, 3), f32)
+            rhs_type = ir.RankedTensorType.get((7, 3), f32)
+            res_type = ir.RankedTensorType.get((5, 7), f32)
+
+            @func.FuncOp.from_py_func(lhs_type, rhs_type, res_type)
+            def named_contraction(lhs, rhs, res):
+                zero = arith.ConstantOp(f32, 0.0).result
+                init = tensor.EmptyOp(res_type.shape, res_type.element_type).result
+
+                filled = linalg.FillOp(
+                    result_tensors=[res_type], inputs=[zero], outputs=[init]
+                ).results[0]
```

**Comment:**
Also here

---


---


## [PR #1264](https://github.com/nod-ai/amd-shark-ai/pull/1264): [tuner] retire op_matchers.py through IREE python bindings

### Review Summary

**APPROVED** (2025-04-18)

LGTM. I love seeing all the deleted code in red.


---


## [PR #1225](https://github.com/nod-ai/amd-shark-ai/pull/1225): [tuner] deferring the link phase to multi-threaded compilation phase

### Review Summary

**CHANGES_REQUESTED** (2025-04-08)

**COMMENTED** (2025-04-08)

**COMMENTED** (2025-04-08)

**COMMENTED** (2025-04-08)

**APPROVED** (2025-04-09)

LGTM but please wait for an approval from @Max191 before merging


### Code Comments

**File:** `tuner/tuner/libtuner.py:483`

```diff
@@ -458,9 +459,30 @@ def create_worker_context_queue(device_ids: list[str]) -> queue.Queue[tuple[int,
 
 def run_iree_compile_command(compile_pack: CompilePack) -> Optional[int]:
     candidate_tracker = compile_pack.candidate_tracker
+    assert candidate_tracker.spec_path, "expected candidate spec path"
+
+    if candidate_tracker.starter_td_spec_str is not None:
+        iree_opt = ireec.binaries.find_tool("iree-opt")
+        with tempfile.TemporaryDirectory() as tmpdir:
+            input_path = os.path.join(tmpdir, "tmp_input.mlir")
+            with open(input_path, "w") as f:
+                f.write(candidate_tracker.starter_td_spec_str)
+            link_result = subprocess.run(
+                [
+                    iree_opt,
+                    "--iree-codegen-link-tuning-specs",
+                    input_path,
+                    "-o",
+                    candidate_tracker.spec_path,
+                ],
+                capture_output=True,
+                text=True,
+            )
+
+            if link_result.returncode != 0:
+                raise RuntimeError(f"iree-opt failed: {link_result.stderr}")
```

**Comment:**
Could we move this to a helper function?

---

**File:** `tuner/tuner/libtuner.py:749`

```diff
@@ -720,6 +741,26 @@ def generate_candidate_specs(
             spec_path = path_config.specs_dir / path_config.get_candidate_spec_filename(
                 candidate_num
             )
+
+            starter_td_spec_str: Optional[str] = None
+            if starter_td_spec is not None:
+                td_specs: list[ir.Module] = []
+                td_specs.append(spec)
+                td_specs.append(starter_td_spec)
```

**Comment:**
```suggestion
                td_specs: list[ir.Module] = [spec, starter_td_spec]
```

---

**File:** `tuner/tuner/libtuner.py:754`

```diff
@@ -720,6 +741,26 @@ def generate_candidate_specs(
             spec_path = path_config.specs_dir / path_config.get_candidate_spec_filename(
                 candidate_num
             )
+
+            starter_td_spec_str: Optional[str] = None
+            if starter_td_spec is not None:
+                td_specs: list[ir.Module] = []
+                td_specs.append(spec)
+                td_specs.append(starter_td_spec)
+                # Only log duplicate matchers during the first iteration.
+                log_duplicates = candidate_num == 0
+                td_specs_to_link = determine_td_specs_to_link(
+                    td_specs,
+                    log_duplicates=log_duplicates,
```

**Comment:**
```suggestion
                td_specs_to_link = determine_td_specs_to_link(
                    td_specs,
                    log_duplicates=(candidate_num == 0),
```

---

**File:** `tuner/tuner/libtuner.py:465`

```diff
@@ -456,11 +457,40 @@ def create_worker_context_queue(device_ids: list[str]) -> queue.Queue[tuple[int,
     return worker_contexts_queue
 
 
+def link_td_specs_with_outpath(starter_td_spec_str: str, output_path: Path) -> None:
+    iree_opt = ireec.binaries.find_tool("iree-opt")
+    with tempfile.TemporaryDirectory() as tmpdir:
+        input_path = os.path.join(tmpdir, "tmp_input.mlir")
+        with open(input_path, "w") as f:
+            f.write(starter_td_spec_str)
```

**Comment:**
Why are we writing the td spec to a file instead of using the path provided through CLI arguments?

---

**File:** `tuner/tuner/libtuner.py:471`

```diff
@@ -456,11 +457,40 @@ def create_worker_context_queue(device_ids: list[str]) -> queue.Queue[tuple[int,
     return worker_contexts_queue
 
 
+def link_td_specs_with_outpath(starter_td_spec_str: str, output_path: Path) -> None:
+    iree_opt = ireec.binaries.find_tool("iree-opt")
+    with tempfile.TemporaryDirectory() as tmpdir:
+        input_path = os.path.join(tmpdir, "tmp_input.mlir")
+        with open(input_path, "w") as f:
+            f.write(starter_td_spec_str)
+
+        link_result = subprocess.run(
+            [
+                iree_opt,
+                "--iree-codegen-link-tuning-specs",
+                input_path,
```

**Comment:**
Are we linking a single input only? I'm confused by this code.

---

**File:** `tuner/tuner/libtuner.py:465`

```diff
@@ -456,11 +457,40 @@ def create_worker_context_queue(device_ids: list[str]) -> queue.Queue[tuple[int,
     return worker_contexts_queue
 
 
+def link_td_specs_with_outpath(starter_td_spec_str: str, output_path: Path) -> None:
+    iree_opt = ireec.binaries.find_tool("iree-opt")
+    with tempfile.TemporaryDirectory() as tmpdir:
+        input_path = os.path.join(tmpdir, "tmp_input.mlir")
+        with open(input_path, "w") as f:
+            f.write(starter_td_spec_str)
```

**Comment:**
Oh, I see what you mean. In this case, I think we'd need a different argument name. This is no longer just a `starter_td_spec_str`, it's a combined/nested spec that needs linking.

---

**File:** `tuner/tuner/libtuner.py:465`

```diff
@@ -456,11 +457,40 @@ def create_worker_context_queue(device_ids: list[str]) -> queue.Queue[tuple[int,
     return worker_contexts_queue
 
 
+def link_td_specs_with_outpath(starter_td_spec_str: str, output_path: Path) -> None:
+    iree_opt = ireec.binaries.find_tool("iree-opt")
+    with tempfile.TemporaryDirectory() as tmpdir:
+        input_path = os.path.join(tmpdir, "tmp_input.mlir")
+        with open(input_path, "w") as f:
+            f.write(starter_td_spec_str)
```

**Comment:**
Maybe something like:
```py
def flatten_nested_td_spec(td_spec_str: str, output_path: Path) -> None:
```

---


---


## [PR #1184](https://github.com/nod-ai/amd-shark-ai/pull/1184): [tuner] expose the function merge_td_specs as utility executable

### Review Summary

**COMMENTED** (2025-03-28)

Nice, this looks useful

**COMMENTED** (2025-03-28)

**APPROVED** (2025-03-28)

LGTM, thanks for cleaning up the tests.

**COMMENTED** (2025-04-02)

**COMMENTED** (2025-04-02)

**APPROVED** (2025-04-02)


### Code Comments

**File:** `tuner/tuner/merge_td_specs.py:1`

```diff
@@ -0,0 +1,145 @@
+# Copyright 2024 Advanced Micro Devices, Inc.
```

**Comment:**
```suggestion
# Copyright 2025 Advanced Micro Devices, Inc.
```

---

**File:** `tuner/tuner/merge_td_specs.py:10`

```diff
@@ -0,0 +1,145 @@
+# Copyright 2024 Advanced Micro Devices, Inc.
+#
+# Licensed under the Apache License v2.0 with LLVM Exceptions.
+# See https://llvm.org/LICENSE.txt for license information.
+# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+"""
+Merge multiple tuner-generated specs into a single one.
+
+This script wraps the `iree-opt --iree-codegen-link-tuning-specs` pass .
```

**Comment:**
```suggestion
This script wraps the `iree-opt --iree-codegen-link-tuning-specs` pass.
```

---

**File:** `tuner/tuner/merge_td_specs.py:102`

```diff
@@ -0,0 +1,145 @@
+# Copyright 2024 Advanced Micro Devices, Inc.
+#
+# Licensed under the Apache License v2.0 with LLVM Exceptions.
+# See https://llvm.org/LICENSE.txt for license information.
+# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+"""
+Merge multiple tuner-generated specs into a single one.
+
+This script wraps the `iree-opt --iree-codegen-link-tuning-specs` pass .
+It can be invoked in two ways:
+    1. From another python script by importing and calling `merge_tuning_specs()`
+    2. Directly from the command line to merge tuning spec files
+
+Usage:
+    python -m tuner.merge_td_specs input1.mlir input2.mlir -o merged.mlir
+"""
+
+import argparse
+import logging
+import subprocess
+import tempfile
+import os
+
+from iree.compiler import ir  # type: ignore
+
+from .common import *
+
+tune_logger = logging.getLogger("tune")
+
+
+def combine_tuning_specs(
+    tuner_ctx: TunerContext, td_specs: list[ir.Module]
+) -> ir.Module:
+    """
+    Puts multiple input modules `td_specs` into a single top-level container module.
+    This function does *not* attempt to merge or link `td_specs` across modules.
+    """
+    with tuner_ctx.mlir_ctx as ctx, ir.Location.unknown():
+        top_module = ir.Module.create()
+        top_module.operation.attributes[
+            "transform.with_named_sequence"
+        ] = ir.UnitAttr.get()
+
+        for td_spec in td_specs:
+            top_module.body.append(td_spec.operation.clone())
+        return top_module
+
+
+def merge_tuning_specs(tuner_ctx: TunerContext, td_specs: list[ir.Module]) -> ir.Module:
+    """
+    Merges multiple input modules (`td_specs`) into a single tuning specification module.
+    First, the input modules are combined into a container module. Then, the external
+    `iree-opt` tool is invoked with the `--iree-codegen-link-tuning-specs` pass to
+    link or merge the individual tuning specs. When all input specs are marked with the
+    default attribute `iree_codegen.tuning_spec_with_default_entrypoint`, they are merged
+    into one tuning spec.
+    """
+    module = combine_tuning_specs(tuner_ctx, td_specs)
+    iree_opt = ireec.binaries.find_tool("iree-opt")
+
+    with tempfile.TemporaryDirectory() as tmpdir:
+        input_path = os.path.join(tmpdir, "tmp_input.mlir")
+        output_path = os.path.join(tmpdir, "tmp_output.mlir")
+
+        with open(input_path, "w") as f:
+            f.write(str(module))
+
+        result = subprocess.run(
+            [
+                iree_opt,
+                "--iree-codegen-link-tuning-specs",
+                input_path,
+                "-o",
+                output_path,
+            ],
+            capture_output=True,
+            text=True,
+        )
+
+        if result.returncode != 0:
+            raise RuntimeError(f"iree-opt failed: {result.stderr}")
+
+        with open(output_path, "r") as f:
+            output_mlir = f.read()
+            return ir.Module.parse(output_mlir, tuner_ctx.mlir_ctx)
+
+
+def main() -> None:
+    parser = argparse.ArgumentParser(
+        prog="merge_td_specs",
+        description="""
+        Merge multiple tuning specification modules into a single one.
+
+        This script wraps the `--iree-codegen-link-tuning-specs` pass and is useful for merging
+        multiple tuning specs into one for further compilation or tuning.
+
+        Examples:
+            python -m tuner.merge_td_specs td_spec_1.mlir td_spec_2.mlir -o merged.mlir
+        """,
```

**Comment:**
You can use the top-level docstring here to avoid duplication

---

**File:** `tuner/tuner/merge_td_specs.py:130`

```diff
@@ -0,0 +1,145 @@
+# Copyright 2024 Advanced Micro Devices, Inc.
+#
+# Licensed under the Apache License v2.0 with LLVM Exceptions.
+# See https://llvm.org/LICENSE.txt for license information.
+# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+"""
+Merge multiple tuner-generated specs into a single one.
+
+This script wraps the `iree-opt --iree-codegen-link-tuning-specs` pass .
+It can be invoked in two ways:
+    1. From another python script by importing and calling `merge_tuning_specs()`
+    2. Directly from the command line to merge tuning spec files
+
+Usage:
+    python -m tuner.merge_td_specs input1.mlir input2.mlir -o merged.mlir
+"""
+
+import argparse
+import logging
+import subprocess
+import tempfile
+import os
+
+from iree.compiler import ir  # type: ignore
+
+from .common import *
+
+tune_logger = logging.getLogger("tune")
+
+
+def combine_tuning_specs(
+    tuner_ctx: TunerContext, td_specs: list[ir.Module]
+) -> ir.Module:
+    """
+    Puts multiple input modules `td_specs` into a single top-level container module.
+    This function does *not* attempt to merge or link `td_specs` across modules.
+    """
+    with tuner_ctx.mlir_ctx as ctx, ir.Location.unknown():
+        top_module = ir.Module.create()
+        top_module.operation.attributes[
+            "transform.with_named_sequence"
+        ] = ir.UnitAttr.get()
+
+        for td_spec in td_specs:
+            top_module.body.append(td_spec.operation.clone())
+        return top_module
+
+
+def merge_tuning_specs(tuner_ctx: TunerContext, td_specs: list[ir.Module]) -> ir.Module:
+    """
+    Merges multiple input modules (`td_specs`) into a single tuning specification module.
+    First, the input modules are combined into a container module. Then, the external
+    `iree-opt` tool is invoked with the `--iree-codegen-link-tuning-specs` pass to
+    link or merge the individual tuning specs. When all input specs are marked with the
+    default attribute `iree_codegen.tuning_spec_with_default_entrypoint`, they are merged
+    into one tuning spec.
+    """
+    module = combine_tuning_specs(tuner_ctx, td_specs)
+    iree_opt = ireec.binaries.find_tool("iree-opt")
+
+    with tempfile.TemporaryDirectory() as tmpdir:
+        input_path = os.path.join(tmpdir, "tmp_input.mlir")
+        output_path = os.path.join(tmpdir, "tmp_output.mlir")
+
+        with open(input_path, "w") as f:
+            f.write(str(module))
+
+        result = subprocess.run(
+            [
+                iree_opt,
+                "--iree-codegen-link-tuning-specs",
+                input_path,
+                "-o",
+                output_path,
+            ],
+            capture_output=True,
+            text=True,
+        )
+
+        if result.returncode != 0:
+            raise RuntimeError(f"iree-opt failed: {result.stderr}")
+
+        with open(output_path, "r") as f:
+            output_mlir = f.read()
+            return ir.Module.parse(output_mlir, tuner_ctx.mlir_ctx)
+
+
+def main() -> None:
+    parser = argparse.ArgumentParser(
+        prog="merge_td_specs",
+        description="""
+        Merge multiple tuning specification modules into a single one.
+
+        This script wraps the `--iree-codegen-link-tuning-specs` pass and is useful for merging
+        multiple tuning specs into one for further compilation or tuning.
+
+        Examples:
+            python -m tuner.merge_td_specs td_spec_1.mlir td_spec_2.mlir -o merged.mlir
+        """,
+        formatter_class=argparse.RawDescriptionHelpFormatter,
+    )
+
+    parser.add_argument(
+        "inputs", nargs="+", help="Input MLIR tuning spec files to merge"
+    )
+
+    parser.add_argument(
+        "-o",
+        "--output",
+        help="Output path for merged MLIR file (if omitted, prints to stdout)",
+    )
+
+    parser.add_argument(
+        "--verbose", "-v", action="store_true", help="Enable verbose output to stdout"
+    )
+
+    args = parser.parse_args()
+    tune_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
+
+    formatter = logging.Formatter("%(message)s")
+
+    console_handler = logging.StreamHandler()
+    console_handler.setFormatter(formatter)
+    tune_logger.addHandler(console_handler)
+
+    with TunerContext() as tuner_ctx:
+        td_specs = []
+        for input_path in args.inputs:
+            tune_logger.info(f"Reading td spec: {input_path}")
```

**Comment:**
Is this printed by default? If one of your supported output options is to print to stdout, you we can't interleave that with random logs

---

**File:** `tuner/tuner/merge_td_specs.py:139`

```diff
@@ -0,0 +1,145 @@
+# Copyright 2024 Advanced Micro Devices, Inc.
+#
+# Licensed under the Apache License v2.0 with LLVM Exceptions.
+# See https://llvm.org/LICENSE.txt for license information.
+# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+"""
+Merge multiple tuner-generated specs into a single one.
+
+This script wraps the `iree-opt --iree-codegen-link-tuning-specs` pass .
+It can be invoked in two ways:
+    1. From another python script by importing and calling `merge_tuning_specs()`
+    2. Directly from the command line to merge tuning spec files
+
+Usage:
+    python -m tuner.merge_td_specs input1.mlir input2.mlir -o merged.mlir
+"""
+
+import argparse
+import logging
+import subprocess
+import tempfile
+import os
+
+from iree.compiler import ir  # type: ignore
+
+from .common import *
+
+tune_logger = logging.getLogger("tune")
+
+
+def combine_tuning_specs(
+    tuner_ctx: TunerContext, td_specs: list[ir.Module]
+) -> ir.Module:
+    """
+    Puts multiple input modules `td_specs` into a single top-level container module.
+    This function does *not* attempt to merge or link `td_specs` across modules.
+    """
+    with tuner_ctx.mlir_ctx as ctx, ir.Location.unknown():
+        top_module = ir.Module.create()
+        top_module.operation.attributes[
+            "transform.with_named_sequence"
+        ] = ir.UnitAttr.get()
+
+        for td_spec in td_specs:
+            top_module.body.append(td_spec.operation.clone())
+        return top_module
+
+
+def merge_tuning_specs(tuner_ctx: TunerContext, td_specs: list[ir.Module]) -> ir.Module:
+    """
+    Merges multiple input modules (`td_specs`) into a single tuning specification module.
+    First, the input modules are combined into a container module. Then, the external
+    `iree-opt` tool is invoked with the `--iree-codegen-link-tuning-specs` pass to
+    link or merge the individual tuning specs. When all input specs are marked with the
+    default attribute `iree_codegen.tuning_spec_with_default_entrypoint`, they are merged
+    into one tuning spec.
+    """
+    module = combine_tuning_specs(tuner_ctx, td_specs)
+    iree_opt = ireec.binaries.find_tool("iree-opt")
+
+    with tempfile.TemporaryDirectory() as tmpdir:
+        input_path = os.path.join(tmpdir, "tmp_input.mlir")
+        output_path = os.path.join(tmpdir, "tmp_output.mlir")
+
+        with open(input_path, "w") as f:
+            f.write(str(module))
+
+        result = subprocess.run(
+            [
+                iree_opt,
+                "--iree-codegen-link-tuning-specs",
+                input_path,
+                "-o",
+                output_path,
+            ],
+            capture_output=True,
+            text=True,
+        )
+
+        if result.returncode != 0:
+            raise RuntimeError(f"iree-opt failed: {result.stderr}")
+
+        with open(output_path, "r") as f:
+            output_mlir = f.read()
+            return ir.Module.parse(output_mlir, tuner_ctx.mlir_ctx)
+
+
+def main() -> None:
+    parser = argparse.ArgumentParser(
+        prog="merge_td_specs",
+        description="""
+        Merge multiple tuning specification modules into a single one.
+
+        This script wraps the `--iree-codegen-link-tuning-specs` pass and is useful for merging
+        multiple tuning specs into one for further compilation or tuning.
+
+        Examples:
+            python -m tuner.merge_td_specs td_spec_1.mlir td_spec_2.mlir -o merged.mlir
+        """,
+        formatter_class=argparse.RawDescriptionHelpFormatter,
+    )
+
+    parser.add_argument(
+        "inputs", nargs="+", help="Input MLIR tuning spec files to merge"
+    )
+
+    parser.add_argument(
+        "-o",
+        "--output",
+        help="Output path for merged MLIR file (if omitted, prints to stdout)",
+    )
+
+    parser.add_argument(
+        "--verbose", "-v", action="store_true", help="Enable verbose output to stdout"
+    )
+
+    args = parser.parse_args()
+    tune_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
+
+    formatter = logging.Formatter("%(message)s")
+
+    console_handler = logging.StreamHandler()
+    console_handler.setFormatter(formatter)
+    tune_logger.addHandler(console_handler)
+
+    with TunerContext() as tuner_ctx:
+        td_specs = []
+        for input_path in args.inputs:
+            tune_logger.info(f"Reading td spec: {input_path}")
+            with open(input_path, "r") as f:
+                td_spec_str = f.read()
+                td_specs.append(ir.Module.parse(td_spec_str, tuner_ctx.mlir_ctx))
+
+        merged_td_spec = merge_tuning_specs(tuner_ctx, td_specs)
+        if args.output:
+            with open(args.output, "w") as f:
+                f.write(str(merged_td_spec))
+            tune_logger.info(f"Merged spec written to: {args.output}")
```

**Comment:**
Also here

---

**File:** `tuner/tuner/merge_td_specs_test.py:1`

```diff
@@ -0,0 +1,140 @@
+# Copyright 2024 Advanced Micro Devices, Inc.
```

**Comment:**
```suggestion
# Copyright 2025 Advanced Micro Devices, Inc.
```

---

**File:** `tuner/tuner/merge_td_specs_test.py:9`

```diff
@@ -0,0 +1,140 @@
+# Copyright 2024 Advanced Micro Devices, Inc.
+#
+# Licensed under the Apache License v2.0 with LLVM Exceptions.
+# See https://llvm.org/LICENSE.txt for license information.
+# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+"""
+Usage: python -m pytest merge_td_specs_test.py
+"""
```

**Comment:**
We don't need to explain how to run unit tests in each file

---

**File:** `tuner/tuner/merge_td_specs_test.py:28`

```diff
@@ -0,0 +1,140 @@
+# Copyright 2024 Advanced Micro Devices, Inc.
+#
+# Licensed under the Apache License v2.0 with LLVM Exceptions.
+# See https://llvm.org/LICENSE.txt for license information.
+# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+"""
+Usage: python -m pytest merge_td_specs_test.py
+"""
+
+import pytest
+
+from typing import Generator
+
+from iree.compiler import ir  # type: ignore
+
+from . import merge_td_specs
+from . import common
+
+
+@pytest.fixture
+def tuner_ctx() -> Generator[common.TunerContext, None, None]:
+    from logging import Logger
+    from unittest.mock import MagicMock
+
+    mock_logger = MagicMock(spec=Logger)
+    with common.TunerContext(logger=mock_logger) as ctx:
+        yield ctx
```

**Comment:**
Can we put this and similar functions in a new file with common testing utilities? Say, `test_utils.py`.

---

**File:** `tuner/tuner/merge_td_specs.py:101`

```diff
@@ -0,0 +1,147 @@
+# Copyright 2025 Advanced Micro Devices, Inc.
+#
+# Licensed under the Apache License v2.0 with LLVM Exceptions.
+# See https://llvm.org/LICENSE.txt for license information.
+# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+"""
+Merge multiple tuner-generated specs into a single one.
+
+This script wraps the `iree-opt --iree-codegen-link-tuning-specs` pass.
+It can be invoked in two ways:
+    1. From another python script by importing and calling `merge_tuning_specs()`
+    2. Directly from the command line to merge tuning spec files
+
+Usage:
+    python -m tuner.merge_td_specs input1.mlir input2.mlir -o merged.mlir
+"""
+
+import argparse
+import logging
+import subprocess
+import tempfile
+import os
+
+from iree.compiler import ir  # type: ignore
+
+from .common import *
+
+tune_logger = logging.getLogger("tune")
+
+
+def combine_tuning_specs(
+    tuner_ctx: TunerContext, td_specs: list[ir.Module]
+) -> ir.Module:
+    """
+    Puts multiple input modules `td_specs` into a single top-level container module.
+    This function does *not* attempt to merge or link `td_specs` across modules.
+    """
+    with tuner_ctx.mlir_ctx as ctx, ir.Location.unknown():
+        top_module = ir.Module.create()
+        top_module.operation.attributes[
+            "transform.with_named_sequence"
+        ] = ir.UnitAttr.get()
+
+        for td_spec in td_specs:
+            top_module.body.append(td_spec.operation.clone())
+        return top_module
+
+
+def merge_tuning_specs(tuner_ctx: TunerContext, td_specs: list[ir.Module]) -> ir.Module:
+    """
+    Merges multiple input modules (`td_specs`) into a single tuning specification module.
+    First, the input modules are combined into a container module. Then, the external
+    `iree-opt` tool is invoked with the `--iree-codegen-link-tuning-specs` pass to
+    link or merge the individual tuning specs. When all input specs are marked with the
+    default attribute `iree_codegen.tuning_spec_with_default_entrypoint`, they are merged
+    into one tuning spec.
+    """
+    module = combine_tuning_specs(tuner_ctx, td_specs)
+    iree_opt = ireec.binaries.find_tool("iree-opt")
+
+    with tempfile.TemporaryDirectory() as tmpdir:
+        input_path = os.path.join(tmpdir, "tmp_input.mlir")
+        output_path = os.path.join(tmpdir, "tmp_output.mlir")
+
+        with open(input_path, "w") as f:
+            f.write(str(module))
+
+        result = subprocess.run(
+            [
+                iree_opt,
+                "--iree-codegen-link-tuning-specs",
+                input_path,
+                "-o",
+                output_path,
+            ],
+            capture_output=True,
+            text=True,
+        )
+
+        if result.returncode != 0:
+            raise RuntimeError(f"iree-opt failed: {result.stderr}")
+
+        with open(output_path, "r") as f:
+            output_mlir = f.read()
+            return ir.Module.parse(output_mlir, tuner_ctx.mlir_ctx)
+
+
+def main() -> None:
+    parser = argparse.ArgumentParser(
+        prog="merge_td_specs",
+        description="""
+            Merge multiple tuner-generated specs into a single one.
+
+            This script wraps the `iree-opt --iree-codegen-link-tuning-specs` pass.
+            It can be invoked in two ways:
+                1. From another python script by importing and calling `merge_tuning_specs()`
+                2. Directly from the command line to merge tuning spec files
+
+            Usage:
+                python -m tuner.merge_td_specs input1.mlir input2.mlir -o merged.mlir
```

**Comment:**
We should use the top-level docstring here to avoid duplications (`__doc__`).

---

**File:** `tuner/tuner/merge_td_specs.py:70`

```diff
@@ -0,0 +1,80 @@
+# Copyright 2025 Advanced Micro Devices, Inc.
+#
+# Licensed under the Apache License v2.0 with LLVM Exceptions.
+# See https://llvm.org/LICENSE.txt for license information.
+# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+"""
+Merge multiple tuner-generated specs into a single one.
+
+This script wraps the `iree-opt --iree-codegen-link-tuning-specs` pass.
+It can be invoked in two ways:
+    1. From another python script by importing and calling `merge_tuning_specs()`
+    2. Directly from the command line to merge tuning spec files
+
+Usage:
+    python -m tuner.merge_td_specs input1.mlir input2.mlir -o merged.mlir
+"""
+
+import argparse
+import logging
+import subprocess
+import tempfile
+import os
+
+from iree.compiler import ir  # type: ignore
+
+from .common import *
+
+tune_logger = logging.getLogger("tune")
+
+
+def main() -> None:
+    parser = argparse.ArgumentParser(
+        prog="merge_td_specs",
+        description=__doc__,
+        formatter_class=argparse.RawDescriptionHelpFormatter,
+    )
+
+    parser.add_argument(
+        "inputs", nargs="+", help="Input MLIR tuning spec files to merge"
+    )
+
+    parser.add_argument(
+        "-o",
+        "--output",
+        help="Output path for merged MLIR file (if omitted, prints to stdout)",
+    )
+
+    parser.add_argument(
+        "--verbose", "-v", action="store_true", help="Enable verbose output to stdout"
+    )
+
+    args = parser.parse_args()
+    tune_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
+
+    formatter = logging.Formatter("%(message)s")
+
+    console_handler = logging.StreamHandler()
+    console_handler.setFormatter(formatter)
+    tune_logger.addHandler(console_handler)
+
+    with TunerContext() as tuner_ctx:
+        td_specs = []
+        for input_path in args.inputs:
+            tune_logger.debug(f"Reading td spec: {input_path}")
+            with open(input_path, "r") as f:
+                td_spec_str = f.read()
+                td_specs.append(ir.Module.parse(td_spec_str, tuner_ctx.mlir_ctx))
+
+        merged_td_spec = link_tuning_specs(tuner_ctx, td_specs)
```

**Comment:**
why doesn't this call the merge function?

---

**File:** `tuner/tuner/test_utils.py:22`

```diff
@@ -0,0 +1,30 @@
+# Copyright 2025 Advanced Micro Devices, Inc.
+#
+# Licensed under the Apache License v2.0 with LLVM Exceptions.
+# See https://llvm.org/LICENSE.txt for license information.
+# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+
+import pytest
+
+from typing import Generator
+from logging import Logger
+from unittest.mock import MagicMock
+
+from iree.compiler import ir  # type: ignore
+
+from . import common
+
+
+@pytest.fixture
+def tuner_ctx() -> Generator[common.TunerContext, None, None]:
+
+    mock_logger = MagicMock(spec=Logger)
```

**Comment:**
```suggestion
def tuner_ctx() -> Generator[common.TunerContext, None, None]:
    mock_logger = MagicMock(spec=Logger)
```

---

**File:** `tuner/tuner/merge_td_specs.py:70`

```diff
@@ -0,0 +1,80 @@
+# Copyright 2025 Advanced Micro Devices, Inc.
+#
+# Licensed under the Apache License v2.0 with LLVM Exceptions.
+# See https://llvm.org/LICENSE.txt for license information.
+# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+"""
+Merge multiple tuner-generated specs into a single one.
+
+This script wraps the `iree-opt --iree-codegen-link-tuning-specs` pass.
+It can be invoked in two ways:
+    1. From another python script by importing and calling `merge_tuning_specs()`
+    2. Directly from the command line to merge tuning spec files
+
+Usage:
+    python -m tuner.merge_td_specs input1.mlir input2.mlir -o merged.mlir
+"""
+
+import argparse
+import logging
+import subprocess
+import tempfile
+import os
+
+from iree.compiler import ir  # type: ignore
+
+from .common import *
+
+tune_logger = logging.getLogger("tune")
+
+
+def main() -> None:
+    parser = argparse.ArgumentParser(
+        prog="merge_td_specs",
+        description=__doc__,
+        formatter_class=argparse.RawDescriptionHelpFormatter,
+    )
+
+    parser.add_argument(
+        "inputs", nargs="+", help="Input MLIR tuning spec files to merge"
+    )
+
+    parser.add_argument(
+        "-o",
+        "--output",
+        help="Output path for merged MLIR file (if omitted, prints to stdout)",
+    )
+
+    parser.add_argument(
+        "--verbose", "-v", action="store_true", help="Enable verbose output to stdout"
+    )
+
+    args = parser.parse_args()
+    tune_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
+
+    formatter = logging.Formatter("%(message)s")
+
+    console_handler = logging.StreamHandler()
+    console_handler.setFormatter(formatter)
+    tune_logger.addHandler(console_handler)
+
+    with TunerContext() as tuner_ctx:
+        td_specs = []
+        for input_path in args.inputs:
+            tune_logger.debug(f"Reading td spec: {input_path}")
+            with open(input_path, "r") as f:
+                td_spec_str = f.read()
+                td_specs.append(ir.Module.parse(td_spec_str, tuner_ctx.mlir_ctx))
+
+        merged_td_spec = link_tuning_specs(tuner_ctx, td_specs)
```

**Comment:**
I meant the python function to determine the tuning specs to merge and print warnings in case of duplicate matchers -- it would be nice to emit the same warnings here

---


---


## [PR #1171](https://github.com/nod-ai/amd-shark-ai/pull/1171): [tuner] add the option of providing starter td spec

### Review Summary

**COMMENTED** (2025-03-27)

**CHANGES_REQUESTED** (2025-03-28)

**COMMENTED** (2025-03-28)

**COMMENTED** (2025-03-31)

**COMMENTED** (2025-03-31)

**COMMENTED** (2025-03-31)

**COMMENTED** (2025-03-31)

**APPROVED** (2025-04-01)

LGTM % nit

**APPROVED** (2025-04-01)


### Code Comments

**File:** `tuner/tuner/candidate_gen.py:218`

```diff
@@ -196,6 +198,24 @@ def generate_configs_and_td_specs(
             break
         tune_logger.debug(f"Solution #{i+1}: {config}")
         td_spec_module = dispatch_tuner.get_td_spec(input_module, config)
+        if starter_td_spec != None:
+            starter_matchers = get_matcher_names_from_td_spec(starter_td_spec)
+            current_matchers = get_matcher_names_from_td_spec(td_spec_module)
+            overlap_matchers = starter_matchers & current_matchers
+            unique_stater_matchers = starter_matchers - current_matchers
+            new_warnings = overlap_matchers - warned_overlap_matchers
+
+            # Log warnings only for newly detected overlapping target operations.
+            if new_warnings:
+                logging.warning(
+                    f"Operations have been tuned in the starter tuning spec: {sorted(new_warnings)}"
+                )
+                warned_overlap_matchers.update(new_warnings)
+            # Only link td spec and starter spec if it adds unique target operations.
+            if unique_stater_matchers:
+                td_spec_module = link_tuning_specs(
+                    tuner_context.mlir_ctx, [starter_td_spec, td_spec_module]
+                )
```

**Comment:**
This should be a function that comes with its own unit tests

---

**File:** `tuner/tuner/common_test.py:358`

```diff
@@ -312,7 +312,40 @@ def test_link_tuning_specs_raises_error(tuner_ctx: common.TunerContext) -> None:
         "iree_codegen.tuning_spec_with_default_entrypoint"
     ] = ir.UnitAttr.get()
     with pytest.raises(RuntimeError) as exc_info:
-        common.link_tuning_specs(tuner_ctx, [module])
+        common.link_tuning_specs(context, [module])
         # iree-opt should fail due to missing named sequence @__kernel_config entrypoint required
         # by the `iree_codegen.tuning_spec_with_default_entrypoint` attribute.
         assert "iree-opt failed" in str(exc_info.value)
+
+
+def test_get_matcher_names_from_td_spec(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    module_str = """
+        module attributes { transform.with_named_sequence } {
+        transform.named_sequence @apply_op_config(%arg0: !transform.any_op {transform.readonly}) {
+            transform.yield
+        }
+
+        transform.named_sequence @match_foo(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+            transform.yield %arg0 : !transform.any_op
+        }
+
+        transform.named_sequence @match_bar(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+            transform.yield %arg0 : !transform.any_op
+        }
+
+        transform.named_sequence @__kernel_config(%arg0: !transform.any_op ) -> !transform.any_op
+            attributes { iree_codegen.tuning_spec_entrypoint } {
+            %0 = transform.foreach_match in %arg0
+            @match_foo -> @apply_op_config
+            , @match_bar -> @apply_op_config
+            : (!transform.any_op) -> !transform.any_op
+            transform.yield %0 : !transform.any_op
+        }
+        }
+    """
+
+    module = ir.Module.parse(module_str, context)
+    matcher_names = common.get_matcher_names_from_td_spec(module)
+
+    assert matcher_names == {"match_foo", "match_bar"}
```

**Comment:**
We should also have a test for IR with no matchers

---

**File:** `tuner/tuner/candidate_gen_test.py:246`

```diff
@@ -207,3 +207,41 @@ def test_get_td_spec_convolution(tuner_ctx: common.TunerContext) -> None:
         "gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false>"
         in matcher_sequence_str
     )
+
+
+def test_check_td_spec_matchers_overlap(tuner_ctx: common.TunerContext) -> None:
+    starter = {
+        "match_contraction_2048x2048x2048_f16xf16xf32",
+        "match_contraction_4x640x4096_i8xi8xi32",
+    }
+    current = {
+        "match_attention_2x10x4096x64x64x64_f16",
+        "match_contraction_4x4096x1920_i8xi8xi32",
+    }
+    warned: set[str] = set()
+
+    should_link, new_warned = candidate_gen.check_td_spec_matchers_overlap(
+        starter, current, warned
+    )
+
+    assert should_link is True
+    assert new_warned == set()
+
+    current = {"match_contraction_2048x2048x2048_f16xf16xf32"}
+    should_link, new_warned = candidate_gen.check_td_spec_matchers_overlap(
+        starter, current, warned
+    )
+    assert should_link is True
+    assert new_warned == {"match_contraction_2048x2048x2048_f16xf16xf32"}
+    warned.update(new_warned)
+
+    current = {
+        "match_contraction_2048x2048x2048_f16xf16xf32",
+        "match_contraction_4x640x4096_i8xi8xi32",
+    }
+    should_link, new_warned = candidate_gen.check_td_spec_matchers_overlap(
+        starter, current, warned
+    )
+    assert should_link is False
+    print(new_warned)
```

**Comment:**
Debug print?

---

**File:** `tuner/tuner/candidate_gen_test.py:227`

```diff
@@ -207,3 +207,41 @@ def test_get_td_spec_convolution(tuner_ctx: common.TunerContext) -> None:
         "gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false>"
         in matcher_sequence_str
     )
+
+
+def test_check_td_spec_matchers_overlap(tuner_ctx: common.TunerContext) -> None:
+    starter = {
+        "match_contraction_2048x2048x2048_f16xf16xf32",
+        "match_contraction_4x640x4096_i8xi8xi32",
+    }
+    current = {
+        "match_attention_2x10x4096x64x64x64_f16",
+        "match_contraction_4x4096x1920_i8xi8xi32",
+    }
+    warned: set[str] = set()
+
+    should_link, new_warned = candidate_gen.check_td_spec_matchers_overlap(
+        starter, current, warned
+    )
+
+    assert should_link is True
```

**Comment:**
```suggestion
    assert should_link
```

---

**File:** `tuner/tuner/candidate_gen_test.py:234`

```diff
@@ -207,3 +207,41 @@ def test_get_td_spec_convolution(tuner_ctx: common.TunerContext) -> None:
         "gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false>"
         in matcher_sequence_str
     )
+
+
+def test_check_td_spec_matchers_overlap(tuner_ctx: common.TunerContext) -> None:
+    starter = {
+        "match_contraction_2048x2048x2048_f16xf16xf32",
+        "match_contraction_4x640x4096_i8xi8xi32",
+    }
+    current = {
+        "match_attention_2x10x4096x64x64x64_f16",
+        "match_contraction_4x4096x1920_i8xi8xi32",
+    }
+    warned: set[str] = set()
+
+    should_link, new_warned = candidate_gen.check_td_spec_matchers_overlap(
+        starter, current, warned
+    )
+
+    assert should_link is True
+    assert new_warned == set()
+
+    current = {"match_contraction_2048x2048x2048_f16xf16xf32"}
+    should_link, new_warned = candidate_gen.check_td_spec_matchers_overlap(
+        starter, current, warned
+    )
+    assert should_link is True
```

**Comment:**
```suggestion
    assert should_link
```

---

**File:** `tuner/tuner/candidate_gen_test.py:245`

```diff
@@ -207,3 +207,41 @@ def test_get_td_spec_convolution(tuner_ctx: common.TunerContext) -> None:
         "gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false>"
         in matcher_sequence_str
     )
+
+
+def test_check_td_spec_matchers_overlap(tuner_ctx: common.TunerContext) -> None:
+    starter = {
+        "match_contraction_2048x2048x2048_f16xf16xf32",
+        "match_contraction_4x640x4096_i8xi8xi32",
+    }
+    current = {
+        "match_attention_2x10x4096x64x64x64_f16",
+        "match_contraction_4x4096x1920_i8xi8xi32",
+    }
+    warned: set[str] = set()
+
+    should_link, new_warned = candidate_gen.check_td_spec_matchers_overlap(
+        starter, current, warned
+    )
+
+    assert should_link is True
+    assert new_warned == set()
+
+    current = {"match_contraction_2048x2048x2048_f16xf16xf32"}
+    should_link, new_warned = candidate_gen.check_td_spec_matchers_overlap(
+        starter, current, warned
+    )
+    assert should_link is True
+    assert new_warned == {"match_contraction_2048x2048x2048_f16xf16xf32"}
+    warned.update(new_warned)
+
+    current = {
+        "match_contraction_2048x2048x2048_f16xf16xf32",
+        "match_contraction_4x640x4096_i8xi8xi32",
+    }
+    should_link, new_warned = candidate_gen.check_td_spec_matchers_overlap(
+        starter, current, warned
+    )
+    assert should_link is False
```

**Comment:**
```suggestion
    assert not should_link
```

---

**File:** `tuner/tuner/candidate_gen.py:211`

```diff
@@ -181,6 +208,29 @@ def generate_configs_and_td_specs(
     assert len(variant_op_list) == 1, "Expect one executable variant op"
     variant_op = variant_op_list[0]
     mma_list = iree_codegen.query_mma_intrinsics(variant_op)
+    if starter_td_spec == None:
```

**Comment:**
```suggestion
    if starter_td_spec is None:
```

---

**File:** `tuner/tuner/candidate_gen.py:230`

```diff
@@ -181,6 +208,29 @@ def generate_configs_and_td_specs(
     assert len(variant_op_list) == 1, "Expect one executable variant op"
     variant_op = variant_op_list[0]
     mma_list = iree_codegen.query_mma_intrinsics(variant_op)
+    if starter_td_spec == None:
+        for i, config in enumerate(
+            generate_solutions(
+                tuner_context,
+                problem_size,
+                num_subgroups,
+                mma_list,
+                allowed_waves_per_eu,
+                pipeline_options_search_space,
+                codegen_pipeline,
+            )
+        ):
+            if i >= limit:
+                break
+            tune_logger.debug(f"Solution #{i+1}: {config}")
+            td_spec_module = dispatch_tuner.get_td_spec(input_module, config)
+            assert td_spec_module, "Failed to generate transform dialect spec"
+            config_specs.append(td_spec_module)
+            tune_logger.debug(f"Generated {len(config_specs)} tuning specs")
+            return config_specs
```

**Comment:**
I think we should unify these two code paths to reduce the overall complexity of the control flow

---

**File:** `tuner/tuner/candidate_gen.py:265`

```diff
@@ -196,6 +246,25 @@ def generate_configs_and_td_specs(
             break
         tune_logger.debug(f"Solution #{i+1}: {config}")
         td_spec_module = dispatch_tuner.get_td_spec(input_module, config)
+        current_matchers = get_matcher_names_from_td_spec(td_spec_module)
+
+        should_link, new_warned_matchers = check_td_spec_matchers_overlap(
+            starter_matchers, current_matchers, warned_overlap_matchers
+        )
```

**Comment:**
If you make `check_td_spec_matchers_overlap` accept td specs as arguments and call `get_matcher_names_from_td_spec` internally, you can make it take a list of tuning specs. For the case with no starter spec, the list will always have a single element.

---

**File:** `tuner/tuner/candidate_gen.py:266`

```diff
@@ -196,6 +246,25 @@ def generate_configs_and_td_specs(
             break
         tune_logger.debug(f"Solution #{i+1}: {config}")
         td_spec_module = dispatch_tuner.get_td_spec(input_module, config)
+        current_matchers = get_matcher_names_from_td_spec(td_spec_module)
+
+        should_link, new_warned_matchers = check_td_spec_matchers_overlap(
+            starter_matchers, current_matchers, warned_overlap_matchers
+        )
+
+        # Log warnings only for newly detected overlapping target operations.
+        if new_warned_matchers:
+            logging.warning(
+                f"Operations have been tuned in the starter tuning spec: {sorted(new_warned_matchers)}"
+            )
+            warned_overlap_matchers.update(new_warned_matchers)
+
+        # Only link td spec and starter spec if it adds unique target operations.
+        if should_link:
+            td_spec_module = link_tuning_specs(
+                tuner_context.mlir_ctx, [starter_td_spec, td_spec_module]
+            )
```

**Comment:**
Similarly, linking should be able to handle a single tuning spec passed to it

---

**File:** `tuner/examples/simple/simple_tuner.py:75`

```diff
@@ -70,9 +70,14 @@ def main():
         default="",
         help="Path to the flags file for iree-benchmark-module for model benchmarking.",
     )
+    client_args.add_argument(
+        "--simple-starter-td-spec",
+        type=str,
+        default="",
+        help="Path to a starter td spec file to merge with tuning spec files.",
+    )
     # Remaining arguments come from libtuner
     args = libtuner.parse_arguments(parser)
-
```

**Comment:**
Can you undo this formatting change?

---

**File:** `tuner/tuner/candidate_gen.py:251`

```diff
@@ -197,6 +244,22 @@ def generate_configs_and_td_specs(
         tune_logger.debug(f"Solution #{i+1}: {config}")
         td_spec_module = dispatch_tuner.get_td_spec(input_module, config)
         assert td_spec_module, "Failed to generate transform dialect spec"
+
+        td_specs_to_link, new_warned_matchers = determine_td_specs_to_link(
+            [starter_td_spec, td_spec_module]
+            if starter_td_spec is not None
+            else [td_spec_module],
```

**Comment:**
Make this list a local variable and append to it instead of having this branching logic

---

**File:** `tuner/tuner/candidate_gen_test.py:232`

```diff
@@ -207,3 +207,69 @@ def test_get_td_spec_convolution(tuner_ctx: common.TunerContext) -> None:
         "gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = false>"
         in matcher_sequence_str
     )
+
+
+def test_determine_td_specs_to_link(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    module_str = """
+        module attributes { transform.with_named_sequence } {
+            transform.named_sequence @apply_op_config(%arg0: !transform.any_op {transform.readonly}) {
+                transform.yield
+            }
+
+            transform.named_sequence @match_foo(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+                transform.yield %arg0 : !transform.any_op
+            }
+
+            transform.named_sequence @match_bar(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+                transform.yield %arg0 : !transform.any_op
+            }
+
+            transform.named_sequence @__kernel_config(%arg0: !transform.any_op ) -> !transform.any_op
+                attributes { iree_codegen.tuning_spec_entrypoint } {
+                %0 = transform.foreach_match in %arg0
+                @match_foo -> @apply_op_config
+                , @match_bar -> @apply_op_config
```

**Comment:**
The indentation is weird here

---

**File:** `tuner/tuner/candidate_gen.py:167`

```diff
@@ -147,6 +147,50 @@ def get_default_output_dir() -> str:
     return "tuning_" + datetime.now().strftime("%Y_%m_%d_%H_%M")
 
 
+def determine_td_specs_to_link(
+    td_specs: list[ir.Module], warned_overlap_matchers: set[str]
+) -> tuple[list[ir.Module], set[str]]:
+    """
+    Determines which tuning specs should be linked based on matcher overlap.
+
+    Additionally, the function identifies any overlapping matchers between the
+    starter and current tuning specs that haven't yet triggered a warning.
+
+    Args:
+        td_specs: A list of 1 or 2 tuning spec modules. If two are provided, the first is
+                the starter spec.
+        warned_overlap_matchers: Set of matchers already warned about for overlaps.
+
+    Returns:
+         A tuple containing:
+        - A list of td specs to link (possibly excluding the starter spec).
+        - A set of overlapping matchers that haven't been warned about yet.
```

**Comment:**
The indentation is weird here

---

**File:** `tuner/tuner/candidate_gen.py:184`

```diff
@@ -147,6 +147,50 @@ def get_default_output_dir() -> str:
     return "tuning_" + datetime.now().strftime("%Y_%m_%d_%H_%M")
 
 
+def determine_td_specs_to_link(
+    td_specs: list[ir.Module], warned_overlap_matchers: set[str]
+) -> tuple[list[ir.Module], set[str]]:
+    """
+    Determines which tuning specs should be linked based on matcher overlap.
+
+    Additionally, the function identifies any overlapping matchers between the
+    starter and current tuning specs that haven't yet triggered a warning.
+
+    Args:
+        td_specs: A list of 1 or 2 tuning spec modules. If two are provided, the first is
+                the starter spec.
+        warned_overlap_matchers: Set of matchers already warned about for overlaps.
+
+    Returns:
+         A tuple containing:
+        - A list of td specs to link (possibly excluding the starter spec).
+        - A set of overlapping matchers that haven't been warned about yet.
+    """
+
+    if len(td_specs) == 1:
+        # No starter td spec provided, nothing to merge.
+        return td_specs, set()
+
+    assert len(td_specs) == 2, "Expected exactly two TD specs (starter and current)"
+    starter_td_spec = td_specs[0]
+    current_td_spec = td_specs[1]
+
+    starter_matchers = get_matcher_names_from_td_spec(starter_td_spec)
+    current_matchers = get_matcher_names_from_td_spec(current_td_spec)
+
+    overlap_matchers = starter_matchers & current_matchers
+    unique_starter_matchers = starter_matchers - current_matchers
+
+    # Determine overlapping matchers which haven't been warned about yet.
```

**Comment:**
I can't parse this sentence, the grammar seems off.

---

**File:** `tuner/tuner/candidate_gen.py:174`

```diff
@@ -147,6 +147,50 @@ def get_default_output_dir() -> str:
     return "tuning_" + datetime.now().strftime("%Y_%m_%d_%H_%M")
 
 
+def determine_td_specs_to_link(
+    td_specs: list[ir.Module], warned_overlap_matchers: set[str]
+) -> tuple[list[ir.Module], set[str]]:
+    """
+    Determines which tuning specs should be linked based on matcher overlap.
+
+    Additionally, the function identifies any overlapping matchers between the
+    starter and current tuning specs that haven't yet triggered a warning.
+
+    Args:
+        td_specs: A list of 1 or 2 tuning spec modules. If two are provided, the first is
+                the starter spec.
+        warned_overlap_matchers: Set of matchers already warned about for overlaps.
+
+    Returns:
+         A tuple containing:
+        - A list of td specs to link (possibly excluding the starter spec).
+        - A set of overlapping matchers that haven't been warned about yet.
+    """
+
+    if len(td_specs) == 1:
+        # No starter td spec provided, nothing to merge.
+        return td_specs, set()
+
+    assert len(td_specs) == 2, "Expected exactly two TD specs (starter and current)"
```

**Comment:**
nit: I'd put this assert on top and check for `len(td_specs) <= 2` and also allow for empty inputs. This way the preconditions are clear.

---

**File:** `tuner/tuner/candidate_gen.py:176`

```diff
@@ -147,6 +147,50 @@ def get_default_output_dir() -> str:
     return "tuning_" + datetime.now().strftime("%Y_%m_%d_%H_%M")
 
 
+def determine_td_specs_to_link(
+    td_specs: list[ir.Module], warned_overlap_matchers: set[str]
+) -> tuple[list[ir.Module], set[str]]:
+    """
+    Determines which tuning specs should be linked based on matcher overlap.
+
+    Additionally, the function identifies any overlapping matchers between the
+    starter and current tuning specs that haven't yet triggered a warning.
+
+    Args:
+        td_specs: A list of 1 or 2 tuning spec modules. If two are provided, the first is
+                the starter spec.
+        warned_overlap_matchers: Set of matchers already warned about for overlaps.
+
+    Returns:
+         A tuple containing:
+        - A list of td specs to link (possibly excluding the starter spec).
+        - A set of overlapping matchers that haven't been warned about yet.
+    """
+
+    if len(td_specs) == 1:
+        # No starter td spec provided, nothing to merge.
+        return td_specs, set()
+
+    assert len(td_specs) == 2, "Expected exactly two TD specs (starter and current)"
+    starter_td_spec = td_specs[0]
+    current_td_spec = td_specs[1]
```

**Comment:**
nit: you can unpack it like this
```suggestion
    starter_td_spec, current_td_spec = td_specs
```

---

**File:** `tuner/tuner/candidate_gen.py:181`

```diff
@@ -147,6 +147,50 @@ def get_default_output_dir() -> str:
     return "tuning_" + datetime.now().strftime("%Y_%m_%d_%H_%M")
 
 
+def determine_td_specs_to_link(
+    td_specs: list[ir.Module], warned_overlap_matchers: set[str]
+) -> tuple[list[ir.Module], set[str]]:
+    """
+    Determines which tuning specs should be linked based on matcher overlap.
+
+    Additionally, the function identifies any overlapping matchers between the
+    starter and current tuning specs that haven't yet triggered a warning.
+
+    Args:
+        td_specs: A list of 1 or 2 tuning spec modules. If two are provided, the first is
+                the starter spec.
+        warned_overlap_matchers: Set of matchers already warned about for overlaps.
+
+    Returns:
+         A tuple containing:
+        - A list of td specs to link (possibly excluding the starter spec).
+        - A set of overlapping matchers that haven't been warned about yet.
+    """
+
+    if len(td_specs) == 1:
+        # No starter td spec provided, nothing to merge.
+        return td_specs, set()
+
+    assert len(td_specs) == 2, "Expected exactly two TD specs (starter and current)"
+    starter_td_spec = td_specs[0]
+    current_td_spec = td_specs[1]
+
+    starter_matchers = get_matcher_names_from_td_spec(starter_td_spec)
+    current_matchers = get_matcher_names_from_td_spec(current_td_spec)
+
+    overlap_matchers = starter_matchers & current_matchers
```

**Comment:**
```suggestion
    overlapping_matchers = starter_matchers & current_matchers
```

---

**File:** `tuner/tuner/candidate_gen.py:185`

```diff
@@ -147,6 +147,50 @@ def get_default_output_dir() -> str:
     return "tuning_" + datetime.now().strftime("%Y_%m_%d_%H_%M")
 
 
+def determine_td_specs_to_link(
+    td_specs: list[ir.Module], warned_overlap_matchers: set[str]
+) -> tuple[list[ir.Module], set[str]]:
+    """
+    Determines which tuning specs should be linked based on matcher overlap.
+
+    Additionally, the function identifies any overlapping matchers between the
+    starter and current tuning specs that haven't yet triggered a warning.
+
+    Args:
+        td_specs: A list of 1 or 2 tuning spec modules. If two are provided, the first is
+                the starter spec.
+        warned_overlap_matchers: Set of matchers already warned about for overlaps.
+
+    Returns:
+         A tuple containing:
+        - A list of td specs to link (possibly excluding the starter spec).
+        - A set of overlapping matchers that haven't been warned about yet.
+    """
+
+    if len(td_specs) == 1:
+        # No starter td spec provided, nothing to merge.
+        return td_specs, set()
+
+    assert len(td_specs) == 2, "Expected exactly two TD specs (starter and current)"
+    starter_td_spec = td_specs[0]
+    current_td_spec = td_specs[1]
+
+    starter_matchers = get_matcher_names_from_td_spec(starter_td_spec)
+    current_matchers = get_matcher_names_from_td_spec(current_td_spec)
+
+    overlap_matchers = starter_matchers & current_matchers
+    unique_starter_matchers = starter_matchers - current_matchers
+
+    # Determine overlapping matchers which haven't been warned about yet.
+    new_warned_matchers = overlap_matchers - warned_overlap_matchers
```

**Comment:**
`new_diplicate_matchers`?

---

**File:** `tuner/tuner/candidate_gen.py:152`

```diff
@@ -147,6 +147,50 @@ def get_default_output_dir() -> str:
     return "tuning_" + datetime.now().strftime("%Y_%m_%d_%H_%M")
 
 
+def determine_td_specs_to_link(
+    td_specs: list[ir.Module], warned_overlap_matchers: set[str]
+) -> tuple[list[ir.Module], set[str]]:
```

**Comment:**
Instead of having the caller provide the known duplicate matchers and returning new duplicated, I think it would be easier to add a boolean flag that decides whether to print a warning about duplicates. This simplifies the interface. If you want to test the duplicate finding logic, you can make it a helper function and test in isolation.

---

**File:** `tuner/tuner/candidate_gen.py:245`

```diff
@@ -197,6 +242,21 @@ def generate_configs_and_td_specs(
         tune_logger.debug(f"Solution #{i+1}: {config}")
         td_spec_module = dispatch_tuner.get_td_spec(input_module, config)
         assert td_spec_module, "Failed to generate transform dialect spec"
+
```

**Comment:**
It would be good to measure the candidate generation time before and after and decide based on this.

---

**File:** `tuner/tuner/candidate_gen.py:245`

```diff
@@ -197,6 +242,21 @@ def generate_configs_and_td_specs(
         tune_logger.debug(f"Solution #{i+1}: {config}")
         td_spec_module = dispatch_tuner.get_td_spec(input_module, config)
         assert td_spec_module, "Failed to generate transform dialect spec"
+
```

**Comment:**
On the other hand having two different td script inherently increases the complexity. It may be annoying to have to deal with potential bugs where we get the expected attributes when compiling dispatches but not model caused by issues with linking etc. I'd rather learn whether the specs are working or not when dealing with candidates.

---


---


## [PR #1136](https://github.com/nod-ai/amd-shark-ai/pull/1136): [tuner] merge default td specs

### Review Summary

**COMMENTED** (2025-03-24)

**COMMENTED** (2025-03-24)

I think it would be useful to expose this as a small utility executable in the tuner that can be invoked on demand to merge tuning specs. But this can be done in a separate PR

**COMMENTED** (2025-03-24)

**COMMENTED** (2025-03-25)

**APPROVED** (2025-03-25)


### Code Comments

**File:** `tuner/tuner/common.py:275`

```diff
@@ -266,3 +268,41 @@ class MLIRTransformation:
     template: list[str]
     modified: str
     embeddable: str
+
+
+def combine_tuning_specs(
```

**Comment:**
Why `combine` instead of `merge`? I think we should stick with consistent naming if we can

---

**File:** `tuner/tuner/common.py:313`

```diff
@@ -266,3 +268,41 @@ class MLIRTransformation:
     template: list[str]
     modified: str
     embeddable: str
+
+
+def combine_tuning_specs(
+    tuner_ctx: TunerContext, td_specs: list[ir.Module]
+) -> ir.Module:
+    with tuner_ctx.mlir_ctx as ctx, ir.Location.unknown():
+        top_module = ir.Module.create()
+        top_module.operation.attributes[
+            "transform.with_named_sequence"
+        ] = ir.UnitAttr.get()
+
+        for td_spec in td_specs:
+            top_module.body.append(td_spec.operation.clone())
+        return top_module
+
+
+def link_tuning_specs(tuner_ctx: TunerContext, td_specs: list[ir.Module]) -> ir.Module:
+    module = combine_tuning_specs(tuner_ctx, td_specs)
+    iree_opt = ireec.binaries.find_tool("iree-opt")
+
+    result = subprocess.run(
+        [
+            iree_opt,
+            "--split-input-file",
+            "--verify-diagnostics",
+            "--iree-codegen-link-tuning-specs",
+            "-",
+        ],
+        input=str(module),  # Provide MLIR text directly via stdin.
+        capture_output=True,
+        text=True,
+    )
+
+    if result.returncode != 0:
+        raise RuntimeError(f"iree-opt failed: {result.stderr}")
```

**Comment:**
I don't think this is covered by any tests?

---

**File:** `tuner/tuner/common.py:307`

```diff
@@ -266,3 +268,41 @@ class MLIRTransformation:
     template: list[str]
     modified: str
     embeddable: str
+
+
+def combine_tuning_specs(
+    tuner_ctx: TunerContext, td_specs: list[ir.Module]
+) -> ir.Module:
+    with tuner_ctx.mlir_ctx as ctx, ir.Location.unknown():
+        top_module = ir.Module.create()
+        top_module.operation.attributes[
+            "transform.with_named_sequence"
+        ] = ir.UnitAttr.get()
+
+        for td_spec in td_specs:
+            top_module.body.append(td_spec.operation.clone())
+        return top_module
+
+
+def link_tuning_specs(tuner_ctx: TunerContext, td_specs: list[ir.Module]) -> ir.Module:
+    module = combine_tuning_specs(tuner_ctx, td_specs)
+    iree_opt = ireec.binaries.find_tool("iree-opt")
+
+    result = subprocess.run(
+        [
+            iree_opt,
+            "--split-input-file",
+            "--verify-diagnostics",
+            "--iree-codegen-link-tuning-specs",
+            "-",
+        ],
+        input=str(module),  # Provide MLIR text directly via stdin.
```

**Comment:**
I think this may be very annoying to reproduce if it ever fails. Why not serialize it to disk and put under a temp directory? 

---

**File:** `tuner/tuner/common.py:308`

```diff
@@ -266,3 +268,41 @@ class MLIRTransformation:
     template: list[str]
     modified: str
     embeddable: str
+
+
+def combine_tuning_specs(
+    tuner_ctx: TunerContext, td_specs: list[ir.Module]
+) -> ir.Module:
+    with tuner_ctx.mlir_ctx as ctx, ir.Location.unknown():
+        top_module = ir.Module.create()
+        top_module.operation.attributes[
+            "transform.with_named_sequence"
+        ] = ir.UnitAttr.get()
+
+        for td_spec in td_specs:
+            top_module.body.append(td_spec.operation.clone())
+        return top_module
+
+
+def link_tuning_specs(tuner_ctx: TunerContext, td_specs: list[ir.Module]) -> ir.Module:
+    module = combine_tuning_specs(tuner_ctx, td_specs)
+    iree_opt = ireec.binaries.find_tool("iree-opt")
+
+    result = subprocess.run(
+        [
+            iree_opt,
+            "--split-input-file",
+            "--verify-diagnostics",
+            "--iree-codegen-link-tuning-specs",
+            "-",
+        ],
+        input=str(module),  # Provide MLIR text directly via stdin.
+        capture_output=True,
+        text=True,
+    )
+
+    if result.returncode != 0:
+        raise RuntimeError(f"iree-opt failed: {result.stderr}")
+
+    linked_module = ir.Module.parse(result.stdout, tuner_ctx.mlir_ctx)
+    return linked_module
```

**Comment:**
```suggestion
    return ir.Module.parse(result.stdout, tuner_ctx.mlir_ctx)
```

---

**File:** `tuner/tuner/common.py:275`

```diff
@@ -266,3 +268,41 @@ class MLIRTransformation:
     template: list[str]
     modified: str
     embeddable: str
+
+
+def combine_tuning_specs(
```

**Comment:**
Could you explain what this function does as a comment so that it's easy to understand the difference between combine/link/merge?

---

**File:** `tuner/tuner/common_test.py:347`

```diff
@@ -206,3 +206,142 @@ def test_get_lowering_config(tuner_ctx: common.TunerContext) -> None:
 
     assert compilation_info.lowering_config.mma_kind is None
     assert compilation_info.lowering_config.subgroup_count_mn == (1, 1)
+
+
+def test_combine_tuning_specs(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    first_module_str = """
+        module @inner_module_a
+            attributes { transform.with_named_sequence } {
+        }
+    """
+
+    second_module_str = """
+        module @inner_module_b
+            attributes { transform.with_named_sequence } {
+        }
+    """
+
+    first_ir_module = ir.Module.parse(first_module_str, context)
+    second_ir_module = ir.Module.parse(second_module_str, context)
+
+    module = common.combine_tuning_specs(tuner_ctx, [first_ir_module, second_ir_module])
+    assert module
+    assert "transform.with_named_sequence" in module.operation.attributes
+
+    inner_ops = list(module.body.operations)
+    assert all(
+        op.name == "builtin.module" for op in inner_ops
+    ), "Not all ops are builtin.module"
+    assert len(inner_ops) == 2, f"Expected 2 inner modules, got {len(inner_ops)}"
+    assert (
+        inner_ops[0].sym_name.value == "inner_module_a"
+    ), f"Expected 'inner_module_a', got '{inner_ops[0].sym_name.value}'"
+    assert (
+        inner_ops[1].sym_name.value == "inner_module_b"
+    ), f"Expected 'inner_module_b', got '{inner_ops[1].sym_name.value}'"
+
+
+def test_link_tuning_specs(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    first_module_str = """
+        module @inner_module_a
+            attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+            transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+                transform.yield %arg : !transform.any_op
+            }
+
+            transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+                transform.yield
+            }
+
+            transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+            -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+                %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+                    : (!transform.any_op) -> (!transform.any_op)
+                transform.yield %res : !transform.any_op
+            }
+        }
+    """
+
+    second_module_str = """
+        module @inner_module_b
+            attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+            transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+                transform.yield %arg : !transform.any_op
+            }
+
+            transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+                transform.yield
+            }
+
+            transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+            -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+                %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+                    : (!transform.any_op) -> (!transform.any_op)
+                transform.yield %res : !transform.any_op
+            }
+        }
+    """
+
+    first_ir_module = ir.Module.parse(first_module_str, context)
+    second_ir_module = ir.Module.parse(second_module_str, context)
+    linked_module = common.link_tuning_specs(
+        tuner_ctx, [first_ir_module, second_ir_module]
+    )
+    assert linked_module
+
+    assert "transform.with_named_sequence" in linked_module.operation.attributes
+    assert (
+        "iree_codegen.tuning_spec_with_default_entrypoint"
+        in linked_module.operation.attributes
+    )
+
+    inner_ops = list(linked_module.body.operations)
+    # Check that inner modules have been merged into the top-level module and no inner modules remain.
+    assert all(
+        op.name != "builtin.module" for op in inner_ops
+    ), "Unexpected inner builtin.module ops found"
+
+    named_sequences = []
+    kernel_config_op = None
+    for op in linked_module.body.operations:
+        if op.name == "transform.named_sequence":
+            sym_name_attr = op.sym_name
+            assert sym_name_attr is not None
+            named_sequences.append(sym_name_attr.value)
+            if sym_name_attr.value == "__kernel_config":
+                kernel_config_op = op
+
+    assert kernel_config_op is not None, "Missing @__kernel_config"
+    expected_names = {
+        "match",
+        "apply_op_config",
+        "inner_module_b_match",
+        "inner_module_b_apply_op_config",
+        "__kernel_config",
+    }
+    assert (
+        set(named_sequences) == expected_names
+    ), f"Unexpected named sequence names: {named_sequences}"
+
+    foreach_match_op = kernel_config_op.body.operations[0]
+    assert (
+        foreach_match_op.name == "transform.foreach_match"
+    ), f"Expected op name 'transform.foreach_match', got '{foreach_match_op.name}'"
+    assert (
+        len(foreach_match_op.matchers) == len(foreach_match_op.actions)
+        and len(foreach_match_op.matchers) == 2
+    )
+    assert (
+        foreach_match_op.matchers[0].value == "match"
+    ), f"Expected first matcher to be 'match', got '{foreach_match_op.matchers[0].value}'"
+    assert (
+        foreach_match_op.matchers[1].value == "inner_module_b_match"
+    ), f"Expected second matcher to be 'inner_module_b_match', got '{foreach_match_op.matchers[1].value}'"
+    assert (
+        foreach_match_op.actions[0].value == "apply_op_config"
+    ), f"Expected first action to be 'apply_op_config', got '{foreach_match_op.actions[0].value}'"
+    assert (
+        foreach_match_op.actions[1].value == "inner_module_b_apply_op_config"
+    ), f"Expected second action to be 'inner_module_b_apply_op_config', got '{foreach_match_op.actions[1].value}'"
```

**Comment:**
We shouldn't need to repeat the same tests for the logic already tested in IREE. Here, I'd only check that that there are no nested modules and that there is a `__kernel_config` op.

---

**File:** `tuner/tuner/common.py:274`

```diff
@@ -266,3 +268,48 @@ class MLIRTransformation:
     template: list[str]
     modified: str
     embeddable: str
+
+
+# Puts multiple input modules `td_specs` into a single top-level container module.
+# This function does *not* attempt to merge or link `td_specs` across modules.
```

**Comment:**
This should be a docstring

---

**File:** `tuner/tuner/common.py:294`

```diff
@@ -266,3 +268,48 @@ class MLIRTransformation:
     template: list[str]
     modified: str
     embeddable: str
+
+
+# Puts multiple input modules `td_specs` into a single top-level container module.
+# This function does *not* attempt to merge or link `td_specs` across modules.
+def combine_tuning_specs(
+    tuner_ctx: TunerContext, td_specs: list[ir.Module]
+) -> ir.Module:
+    with tuner_ctx.mlir_ctx as ctx, ir.Location.unknown():
+        top_module = ir.Module.create()
+        top_module.operation.attributes[
+            "transform.with_named_sequence"
+        ] = ir.UnitAttr.get()
+
+        for td_spec in td_specs:
+            top_module.body.append(td_spec.operation.clone())
+        return top_module
+
+
+# Links multiple input modules (`td_specs`) into a single tuning specification module.
+# First, the input modules are combined into a container module. Then, the external
+# `iree-opt` tool is invoked with the `--iree-codegen-link-tuning-specs` pass to
+# link or merge the individual tuning specs. When all input specs are marked with the
+# default attribute `iree_codegen.tuning_spec_with_default_entrypoint`, they are merged
+# into one tuning spec.
```

**Comment:**
This should be a docstring

---

**File:** `tuner/tuner/common_test.py:267`

```diff
@@ -206,3 +206,136 @@ def test_get_lowering_config(tuner_ctx: common.TunerContext) -> None:
 
     assert compilation_info.lowering_config.mma_kind is None
     assert compilation_info.lowering_config.subgroup_count_mn == (1, 1)
+
+
+def test_combine_tuning_specs(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    first_module_str = """
+        module @inner_module_a
+            attributes { transform.with_named_sequence } {
+        }
+    """
+
+    second_module_str = """
+        module @inner_module_b
+            attributes { transform.with_named_sequence } {
+        }
+    """
+
+    first_ir_module = ir.Module.parse(first_module_str, context)
+    second_ir_module = ir.Module.parse(second_module_str, context)
+
+    module = common.combine_tuning_specs(tuner_ctx, [first_ir_module, second_ir_module])
+    assert module
+    assert "transform.with_named_sequence" in module.operation.attributes
+
+    inner_ops = list(module.body.operations)
+    assert all(
+        op.name == "builtin.module" for op in inner_ops
+    ), "Not all ops are builtin.module"
+    assert len(inner_ops) == 2, f"Expected 2 inner modules, got {len(inner_ops)}"
+    assert (
+        inner_ops[0].sym_name.value == "inner_module_a"
+    ), f"Expected 'inner_module_a', got '{inner_ops[0].sym_name.value}'"
+    assert (
+        inner_ops[1].sym_name.value == "inner_module_b"
+    ), f"Expected 'inner_module_b', got '{inner_ops[1].sym_name.value}'"
+
+
+def test_link_tuning_specs(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    first_module_str = """
+        module @inner_module_a
+            attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+            transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+                transform.yield %arg : !transform.any_op
+            }
+
+            transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+                transform.yield
+            }
+
+            transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+            -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+                %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+                    : (!transform.any_op) -> (!transform.any_op)
+                transform.yield %res : !transform.any_op
+            }
+        }
+    """
+
+    second_module_str = """
```

**Comment:**
Could we reduce duplication by providing the same module twice?

---

**File:** `tuner/tuner/common_test.py:341`

```diff
@@ -206,3 +206,136 @@ def test_get_lowering_config(tuner_ctx: common.TunerContext) -> None:
 
     assert compilation_info.lowering_config.mma_kind is None
     assert compilation_info.lowering_config.subgroup_count_mn == (1, 1)
+
+
+def test_combine_tuning_specs(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    first_module_str = """
+        module @inner_module_a
+            attributes { transform.with_named_sequence } {
+        }
+    """
+
+    second_module_str = """
+        module @inner_module_b
+            attributes { transform.with_named_sequence } {
+        }
+    """
+
+    first_ir_module = ir.Module.parse(first_module_str, context)
+    second_ir_module = ir.Module.parse(second_module_str, context)
+
+    module = common.combine_tuning_specs(tuner_ctx, [first_ir_module, second_ir_module])
+    assert module
+    assert "transform.with_named_sequence" in module.operation.attributes
+
+    inner_ops = list(module.body.operations)
+    assert all(
+        op.name == "builtin.module" for op in inner_ops
+    ), "Not all ops are builtin.module"
+    assert len(inner_ops) == 2, f"Expected 2 inner modules, got {len(inner_ops)}"
+    assert (
+        inner_ops[0].sym_name.value == "inner_module_a"
+    ), f"Expected 'inner_module_a', got '{inner_ops[0].sym_name.value}'"
+    assert (
+        inner_ops[1].sym_name.value == "inner_module_b"
+    ), f"Expected 'inner_module_b', got '{inner_ops[1].sym_name.value}'"
+
+
+def test_link_tuning_specs(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    first_module_str = """
+        module @inner_module_a
+            attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+            transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+                transform.yield %arg : !transform.any_op
+            }
+
+            transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+                transform.yield
+            }
+
+            transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+            -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+                %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+                    : (!transform.any_op) -> (!transform.any_op)
+                transform.yield %res : !transform.any_op
+            }
+        }
+    """
+
+    second_module_str = """
+        module @inner_module_b
+            attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+            transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+                transform.yield %arg : !transform.any_op
+            }
+
+            transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+                transform.yield
+            }
+
+            transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+            -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+                %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+                    : (!transform.any_op) -> (!transform.any_op)
+                transform.yield %res : !transform.any_op
+            }
+        }
+    """
+
+    first_ir_module = ir.Module.parse(first_module_str, context)
+    second_ir_module = ir.Module.parse(second_module_str, context)
+    linked_module = common.link_tuning_specs(
+        tuner_ctx, [first_ir_module, second_ir_module]
+    )
+    assert linked_module
+
+    assert "transform.with_named_sequence" in linked_module.operation.attributes
+    assert (
+        "iree_codegen.tuning_spec_with_default_entrypoint"
+        in linked_module.operation.attributes
+    )
+
+    inner_ops = list(linked_module.body.operations)
+    # Check that inner modules have been merged into the top-level module and no inner modules remain.
+    assert all(
+        op.name != "builtin.module" for op in inner_ops
+    ), "Unexpected inner builtin.module ops found"
+
+    named_sequences = []
+    kernel_config_op = None
+    for op in linked_module.body.operations:
+        if op.name == "transform.named_sequence":
+            sym_name_attr = op.sym_name
+            assert sym_name_attr is not None
+            named_sequences.append(sym_name_attr.value)
+            if sym_name_attr.value == "__kernel_config":
+                kernel_config_op = op
+
+    assert kernel_config_op is not None, "Missing @__kernel_config"
+
+
+def test_link_tuning_specs_raises_error(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    module_str = """
+        module @inner_module_a
+            attributes { transform.with_named_sequence } {
+            transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+                transform.yield %arg : !transform.any_op
+            }
+
+            transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+                transform.yield
+            }
+        }
+    """
+
+    module = ir.Module.parse(module_str, context)
+    module.operation.attributes[
+        "iree_codegen.tuning_spec_with_default_entrypoint"
+    ] = ir.UnitAttr.get()
+    with pytest.raises(RuntimeError) as exc_info:
+        common.link_tuning_specs(tuner_ctx, [module])
+
+    assert "iree-opt failed" in str(exc_info.value)
```

**Comment:**
This assert should be inside the `with` statement

---

**File:** `tuner/tuner/common_test.py:315`

```diff
@@ -206,3 +206,136 @@ def test_get_lowering_config(tuner_ctx: common.TunerContext) -> None:
 
     assert compilation_info.lowering_config.mma_kind is None
     assert compilation_info.lowering_config.subgroup_count_mn == (1, 1)
+
+
+def test_combine_tuning_specs(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    first_module_str = """
+        module @inner_module_a
+            attributes { transform.with_named_sequence } {
+        }
+    """
+
+    second_module_str = """
+        module @inner_module_b
+            attributes { transform.with_named_sequence } {
+        }
+    """
+
+    first_ir_module = ir.Module.parse(first_module_str, context)
+    second_ir_module = ir.Module.parse(second_module_str, context)
+
+    module = common.combine_tuning_specs(tuner_ctx, [first_ir_module, second_ir_module])
+    assert module
+    assert "transform.with_named_sequence" in module.operation.attributes
+
+    inner_ops = list(module.body.operations)
+    assert all(
+        op.name == "builtin.module" for op in inner_ops
+    ), "Not all ops are builtin.module"
+    assert len(inner_ops) == 2, f"Expected 2 inner modules, got {len(inner_ops)}"
+    assert (
+        inner_ops[0].sym_name.value == "inner_module_a"
+    ), f"Expected 'inner_module_a', got '{inner_ops[0].sym_name.value}'"
+    assert (
+        inner_ops[1].sym_name.value == "inner_module_b"
+    ), f"Expected 'inner_module_b', got '{inner_ops[1].sym_name.value}'"
+
+
+def test_link_tuning_specs(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    first_module_str = """
+        module @inner_module_a
+            attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+            transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+                transform.yield %arg : !transform.any_op
+            }
+
+            transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+                transform.yield
+            }
+
+            transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+            -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+                %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+                    : (!transform.any_op) -> (!transform.any_op)
+                transform.yield %res : !transform.any_op
+            }
+        }
+    """
+
+    second_module_str = """
+        module @inner_module_b
+            attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+            transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+                transform.yield %arg : !transform.any_op
+            }
+
+            transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+                transform.yield
+            }
+
+            transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+            -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+                %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+                    : (!transform.any_op) -> (!transform.any_op)
+                transform.yield %res : !transform.any_op
+            }
+        }
+    """
+
+    first_ir_module = ir.Module.parse(first_module_str, context)
+    second_ir_module = ir.Module.parse(second_module_str, context)
+    linked_module = common.link_tuning_specs(
+        tuner_ctx, [first_ir_module, second_ir_module]
+    )
+    assert linked_module
+
+    assert "transform.with_named_sequence" in linked_module.operation.attributes
+    assert (
+        "iree_codegen.tuning_spec_with_default_entrypoint"
+        in linked_module.operation.attributes
+    )
+
+    inner_ops = list(linked_module.body.operations)
+    # Check that inner modules have been merged into the top-level module and no inner modules remain.
+    assert all(
+        op.name != "builtin.module" for op in inner_ops
+    ), "Unexpected inner builtin.module ops found"
+
+    named_sequences = []
+    kernel_config_op = None
+    for op in linked_module.body.operations:
+        if op.name == "transform.named_sequence":
+            sym_name_attr = op.sym_name
+            assert sym_name_attr is not None
+            named_sequences.append(sym_name_attr.value)
+            if sym_name_attr.value == "__kernel_config":
+                kernel_config_op = op
+
+    assert kernel_config_op is not None, "Missing @__kernel_config"
+
+
+def test_link_tuning_specs_raises_error(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    module_str = """
+        module @inner_module_a
+            attributes { transform.with_named_sequence } {
+            transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+                transform.yield %arg : !transform.any_op
+            }
+
+            transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+                transform.yield
+            }
+        }
+    """
+
+    module = ir.Module.parse(module_str, context)
+    module.operation.attributes[
+        "iree_codegen.tuning_spec_with_default_entrypoint"
+    ] = ir.UnitAttr.get()
+    with pytest.raises(RuntimeError) as exc_info:
+        common.link_tuning_specs(tuner_ctx, [module])
```

**Comment:**
Can you explain why you expect this to fail?

---

**File:** `tuner/tuner/common_test.py:324`

```diff
@@ -206,3 +206,120 @@ def test_get_lowering_config(tuner_ctx: common.TunerContext) -> None:
 
     assert compilation_info.lowering_config.mma_kind is None
     assert compilation_info.lowering_config.subgroup_count_mn == (1, 1)
+
+
+def test_combine_tuning_specs(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    first_module_str = """
+        module @inner_module_a
+            attributes { transform.with_named_sequence } {
+        }
+    """
+
+    second_module_str = """
+        module @inner_module_b
+            attributes { transform.with_named_sequence } {
+        }
+    """
+
+    first_ir_module = ir.Module.parse(first_module_str, context)
+    second_ir_module = ir.Module.parse(second_module_str, context)
+
+    module = common.combine_tuning_specs(tuner_ctx, [first_ir_module, second_ir_module])
+    assert module
+    assert "transform.with_named_sequence" in module.operation.attributes
+
+    inner_ops = list(module.body.operations)
+    assert all(
+        op.name == "builtin.module" for op in inner_ops
+    ), "Not all ops are builtin.module"
+    assert len(inner_ops) == 2, f"Expected 2 inner modules, got {len(inner_ops)}"
+    assert (
+        inner_ops[0].sym_name.value == "inner_module_a"
+    ), f"Expected 'inner_module_a', got '{inner_ops[0].sym_name.value}'"
+    assert (
+        inner_ops[1].sym_name.value == "inner_module_b"
+    ), f"Expected 'inner_module_b', got '{inner_ops[1].sym_name.value}'"
+
+
+def test_link_tuning_specs(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    module_str = """
+        module @inner_module_a
+            attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+            transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+                transform.yield %arg : !transform.any_op
+            }
+
+            transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+                transform.yield
+            }
+
+            transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+            -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+                %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+                    : (!transform.any_op) -> (!transform.any_op)
+                transform.yield %res : !transform.any_op
+            }
+        }
+    """
+
+    first_ir_module = ir.Module.parse(module_str, context)
+    second_ir_module = ir.Module.parse(module_str, context)
+    second_ir_module.operation.attributes["sym_name"] = ir.StringAttr.get(
+        "inner_module_b"
+    )
+    linked_module = common.link_tuning_specs(
+        tuner_ctx, [first_ir_module, second_ir_module]
+    )
+    assert linked_module
+
+    assert "transform.with_named_sequence" in linked_module.operation.attributes
+    assert (
+        "iree_codegen.tuning_spec_with_default_entrypoint"
+        in linked_module.operation.attributes
+    )
+
+    inner_ops = list(linked_module.body.operations)
+    # Check that inner modules have been merged into the top-level module and no inner modules remain.
+    assert all(
+        op.name != "builtin.module" for op in inner_ops
+    ), "Unexpected inner builtin.module ops found"
+
+    named_sequences = []
+    kernel_config_op = None
+    for op in linked_module.body.operations:
+        if op.name == "transform.named_sequence":
+            sym_name_attr = op.sym_name
+            assert sym_name_attr is not None
+            named_sequences.append(sym_name_attr.value)
+            if sym_name_attr.value == "__kernel_config":
+                kernel_config_op = op
+
+    assert kernel_config_op is not None, "Missing @__kernel_config"
+
+
+def test_link_tuning_specs_raises_error(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    module_str = """
+        module @inner_module_a
+            attributes { transform.with_named_sequence } {
+            transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+                transform.yield %arg : !transform.any_op
+            }
+
+            transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+                transform.yield
+            }
+        }
+    """
+
+    module = ir.Module.parse(module_str, context)
+    module.operation.attributes[
+        "iree_codegen.tuning_spec_with_default_entrypoint"
+    ] = ir.UnitAttr.get()
+    with pytest.raises(RuntimeError) as exc_info:
+        common.link_tuning_specs(tuner_ctx, [module])
+        # iree-opt should fail due to missing named sequence @__kernel_config entrypoint required
+        # by `iree_codegen.tuning_spec_with_default_entrypoint` attribute.
```

**Comment:**
```suggestion
        # by the `iree_codegen.tuning_spec_with_default_entrypoint` attribute.
```

---

**File:** `tuner/tuner/common_test.py:313`

```diff
@@ -206,3 +206,120 @@ def test_get_lowering_config(tuner_ctx: common.TunerContext) -> None:
 
     assert compilation_info.lowering_config.mma_kind is None
     assert compilation_info.lowering_config.subgroup_count_mn == (1, 1)
+
+
+def test_combine_tuning_specs(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    first_module_str = """
+        module @inner_module_a
+            attributes { transform.with_named_sequence } {
+        }
+    """
+
+    second_module_str = """
+        module @inner_module_b
+            attributes { transform.with_named_sequence } {
+        }
+    """
+
+    first_ir_module = ir.Module.parse(first_module_str, context)
+    second_ir_module = ir.Module.parse(second_module_str, context)
+
+    module = common.combine_tuning_specs(tuner_ctx, [first_ir_module, second_ir_module])
+    assert module
+    assert "transform.with_named_sequence" in module.operation.attributes
+
+    inner_ops = list(module.body.operations)
+    assert all(
+        op.name == "builtin.module" for op in inner_ops
+    ), "Not all ops are builtin.module"
+    assert len(inner_ops) == 2, f"Expected 2 inner modules, got {len(inner_ops)}"
+    assert (
+        inner_ops[0].sym_name.value == "inner_module_a"
+    ), f"Expected 'inner_module_a', got '{inner_ops[0].sym_name.value}'"
+    assert (
+        inner_ops[1].sym_name.value == "inner_module_b"
+    ), f"Expected 'inner_module_b', got '{inner_ops[1].sym_name.value}'"
+
+
+def test_link_tuning_specs(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    module_str = """
+        module @inner_module_a
+            attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+            transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+                transform.yield %arg : !transform.any_op
+            }
+
+            transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+                transform.yield
+            }
+
+            transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+            -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+                %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+                    : (!transform.any_op) -> (!transform.any_op)
+                transform.yield %res : !transform.any_op
+            }
+        }
+    """
+
+    first_ir_module = ir.Module.parse(module_str, context)
+    second_ir_module = ir.Module.parse(module_str, context)
+    second_ir_module.operation.attributes["sym_name"] = ir.StringAttr.get(
+        "inner_module_b"
+    )
+    linked_module = common.link_tuning_specs(
+        tuner_ctx, [first_ir_module, second_ir_module]
+    )
+    assert linked_module
+
+    assert "transform.with_named_sequence" in linked_module.operation.attributes
+    assert (
+        "iree_codegen.tuning_spec_with_default_entrypoint"
+        in linked_module.operation.attributes
+    )
+
+    inner_ops = list(linked_module.body.operations)
+    # Check that inner modules have been merged into the top-level module and no inner modules remain.
+    assert all(
+        op.name != "builtin.module" for op in inner_ops
+    ), "Unexpected inner builtin.module ops found"
+
+    named_sequences = []
+    kernel_config_op = None
+    for op in linked_module.body.operations:
+        if op.name == "transform.named_sequence":
+            sym_name_attr = op.sym_name
+            assert sym_name_attr is not None
+            named_sequences.append(sym_name_attr.value)
+            if sym_name_attr.value == "__kernel_config":
+                kernel_config_op = op
+
+    assert kernel_config_op is not None, "Missing @__kernel_config"
+
+
+def test_link_tuning_specs_raises_error(tuner_ctx: common.TunerContext) -> None:
+    context = tuner_ctx.mlir_ctx
+    module_str = """
+        module @inner_module_a
+            attributes { transform.with_named_sequence } {
+            transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+                transform.yield %arg : !transform.any_op
+            }
+
+            transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+                transform.yield
+            }
```

**Comment:**
Do we need these two named sequences to make it fail because of missing __kernel_config?

---


---


## [PR #1132](https://github.com/nod-ai/amd-shark-ai/pull/1132): [tuner] remove iree_input from the import list of dialects

### Review Summary

**COMMENTED** (2025-03-21)

Can you explain why we want to change this and link to the relevant IREE PR? We should include context like this in the PR description.

**APPROVED** (2025-03-21)


---


## [PR #1124](https://github.com/nod-ai/amd-shark-ai/pull/1124): [Tuner] remove default attribute for the place holder tuning spec

### Review Summary

**COMMENTED** (2025-03-19)

How did you test this?

**APPROVED** (2025-03-19)

**CHANGES_REQUESTED** (2025-03-20)

**APPROVED** (2025-03-21)

LGTM. Please also add the relevant IREE PR link to the description.


### Code Comments

**File:** `tuner/tuner/libtuner_test.py:191`

```diff
@@ -188,7 +188,7 @@ def test_get_compilation_success_rate():
 
 
 def test_enum_collision():
-    from iree.compiler.dialects import linalg, vector, iree_gpu, iree_codegen, iree_input  # type: ignore
+    from iree.compiler.dialects import linalg, vector, iree_gpu, iree_codegen  # type: ignore
```

**Comment:**
Can you put this in a separate PR? Let's not mix these two things

---


---


## [PR #1016](https://github.com/nod-ai/amd-shark-ai/pull/1016): [tuner] add default attribute to tuning spec

### Review Summary

**APPROVED** (2025-03-07)


---


## [PR #841](https://github.com/nod-ai/amd-shark-ai/pull/841): [tuner] clean up the code. NFC.

### Review Summary

**APPROVED** (2025-01-17)

Fantastic, thank you!


---


## [PR #815](https://github.com/nod-ai/amd-shark-ai/pull/815): [tuner] add the output file

### Review Summary

**CHANGES_REQUESTED** (2025-01-13)

Could you also add the time information (absolute time and % of the baseline)?

**COMMENTED** (2025-01-13)

We should open the file once and keep appending to it

**COMMENTED** (2025-01-14)

**COMMENTED** (2025-01-14)

**COMMENTED** (2025-01-14)

**COMMENTED** (2025-01-14)

Can you upload a sample summary log file so that we know what to expect after merging this?

**COMMENTED** (2025-01-14)

Overall looks good, but it seems like there's some stuff to clean up still

**COMMENTED** (2025-01-14)

**APPROVED** (2025-01-14)

LGTM


### Code Comments

**File:** `tuner/examples/simple/simple_tuner.py:78`

```diff
@@ -75,6 +75,7 @@ def main():
 
     path_config = libtuner.PathConfig()
     path_config.base_dir.mkdir(parents=True, exist_ok=True)
+    path_config._set_tune_output(Path("autotune_output.txt"))
```

**Comment:**
This appears to be a private function (it starts with an underscore), so we shouldn't call it directly in the current form.

---

**File:** `tuner/examples/simple/simple_tuner.py:168`

```diff
@@ -156,3 +157,13 @@ def main():
 
         print("Check the detailed execution logs in:")
         print(path_config.run_log.resolve())
+
+        print("Check the tuning results in:")
+        print(path_config.tune_output.resolve())
+        with open(path_config.tune_output, "w") as file:
+            file.write(f"Top dispatch candidates: {top_candidates}\n")
+            for id in top_candidates:
+                file.write(f"{candidate_trackers[id].spec_path.resolve()}\n")
+            file.write(f"Top model candidates: {top_model_candidates}\n")
+            for id in top_model_candidates:
```

**Comment:**
We should write these as they become available, not at the very end. This is in case something fails / we interrupt the tuner -- in these situations partial results are still useful.

---

**File:** `tuner/examples/simple/simple_tuner.py:78`

```diff
@@ -75,6 +75,7 @@ def main():
 
     path_config = libtuner.PathConfig()
     path_config.base_dir.mkdir(parents=True, exist_ok=True)
+    path_config._set_tune_output(Path("autotune_output.txt"))
```

**Comment:**
Also, can we give it a more descriptive name than `output`? It would not be clear to what to expect based on this name. Maybe `summary.log`?

---

**File:** `tuner/tuner/libtuner.py:808`

```diff
@@ -801,7 +805,7 @@ def compile(
         num_worker=num_worker, task_list=task_list, function=run_iree_compile_command
     )
     compiled_candidates = [c for c in compiled_candidates if c is not None]
-    success_rate = float(len(compiled_candidates)) / float(len(candidates))
+    success_rate = float(len(compiled_candidates)) / float(len(task_list))
```

**Comment:**
Nice catch! I wish we had a test for this too...

---

**File:** `tuner/tuner/libtuner.py:90`

```diff
@@ -86,6 +87,7 @@ def __post_init__(self):
         object.__setattr__(self, "candidates_dir", self.base_dir / "candidates")
         object.__setattr__(self, "compiled_dir", self.candidates_dir / "compiled")
         object.__setattr__(self, "specs_dir", self.candidates_dir / "specs")
+        object.__setattr__(self, "output_dir", self.base_dir / "summary.log")
```

**Comment:**
This is not a directory -- it's a path to a file

---

**File:** `tuner/tuner/libtuner.py:730`

```diff
@@ -717,10 +723,16 @@ def generate_candidate_specs(
         tune_logger.exception("Error in candidate_gen.py:")
         raise
 
-    logging.info(f"Generated [{len(candidates) - 1}] candidates")
+    logging.debug(f"Generated [{len(candidates) - 1}] candidates")
     return candidates
 
 
+def get_compilation_success_rate(compiled_candiates: list[Any]) -> float:
+    successful_candidates = [c for c in compiled_candiates if c is not None]
+    success_rate = float(len(successful_candidates)) / float(len(compiled_candiates))
```

**Comment:**
This can cause division by zero, no? Can we add a unit test for this?

---

**File:** `tuner/tuner/libtuner.py:730`

```diff
@@ -717,10 +723,16 @@ def generate_candidate_specs(
         tune_logger.exception("Error in candidate_gen.py:")
         raise
 
-    logging.info(f"Generated [{len(candidates) - 1}] candidates")
+    logging.debug(f"Generated [{len(candidates) - 1}] candidates")
     return candidates
 
 
+def get_compilation_success_rate(compiled_candiates: list[Any]) -> float:
```

**Comment:**
Why are we using `Any` for the element type?

---

**File:** `tuner/tuner/libtuner.py:884`

```diff
@@ -868,18 +881,20 @@ def get_speedup(result: BenchmarkResult) -> float:
     best_results = sorted(filtered_candidate_results, key=sorting_key)[
         :num_top_candidates
     ]
-    logging.info(f"Selected top[{len(best_results)}]:")
+    tuner_context.logger.info(f"Selected top[{len(best_results)}]:")
```

**Comment:**
What happens when we use the root logger here? Why do we have to explicitly log via the logger in the context?

---

**File:** `tuner/tuner/libtuner_test.py:185`

```diff
@@ -176,7 +188,18 @@ def test_validate_devices_with_invalid_device() -> None:
                 assert expected_call in mock_handle_error.call_args_list
 
 
-def test_select_best_benchmark_results() -> None:
+def test_get_compilation_success_rate():
+    compiled_candidates = [0, None, 2, None, 4]
+    assert libtuner.get_compilation_success_rate(compiled_candidates) == 3.0 / 5.0
+
+    compiled_candidates = [0, 1, 2, 3, 4]
+    assert libtuner.get_compilation_success_rate(compiled_candidates) == 1.0
+
+    compiled_candidates = [None, None, None]
+    assert libtuner.get_compilation_success_rate(compiled_candidates) == 0.0
```

**Comment:**
Missing testcase: empty input list

---

**File:** `tuner/tuner/libtuner_test.py:29`

```diff
@@ -10,13 +10,25 @@
 import json
 from subprocess import CompletedProcess
 from unittest.mock import call, patch, MagicMock
+from typing import Generator
 from . import libtuner
+from . import common
 
 """
 Usage: python -m pytest libtuner_test.py
 """
 
 
+@pytest.fixture
+def tuner_ctx() -> Generator[common.TunerContext, None, None]:
+    from logging import Logger
+    from unittest.mock import MagicMock
+
+    mock_logger = MagicMock(spec=Logger)
+    with common.TunerContext(logger=mock_logger) as ctx:
+        yield ctx
```

**Comment:**
This appears unused

---

**File:** `tuner/tuner/libtuner.py:884`

```diff
@@ -868,18 +881,20 @@ def get_speedup(result: BenchmarkResult) -> float:
     best_results = sorted(filtered_candidate_results, key=sorting_key)[
         :num_top_candidates
     ]
-    logging.info(f"Selected top[{len(best_results)}]:")
+    tuner_context.logger.info(f"Selected top[{len(best_results)}]:")
```

**Comment:**
Why is that? I thought that if we register the logging handler for summary, it will be attached to the root logger.

---

**File:** `tuner/tuner/libtuner.py:730`

```diff
@@ -717,10 +723,16 @@ def generate_candidate_specs(
         tune_logger.exception("Error in candidate_gen.py:")
         raise
 
-    logging.info(f"Generated [{len(candidates) - 1}] candidates")
+    logging.debug(f"Generated [{len(candidates) - 1}] candidates")
     return candidates
 
 
+def get_compilation_success_rate(compiled_candiates: list[Any]) -> float:
+    successful_candidates = [c for c in compiled_candiates if c is not None]
+    success_rate = float(len(successful_candidates)) / float(len(compiled_candiates))
```

**Comment:**
I can see this happening if we for whatever reason select the max number of candidates to generate to 0

---

**File:** `tuner/examples/simple/simple_tuner.py:125`

```diff
@@ -114,15 +122,20 @@ def main():
             return
 
         print("Benchmarking compiled dispatch candidates...")
+        logging.info(f"Summarization about top dispatch candidates:")
```

**Comment:**
No need for fstring, we don't print any variables here. Also, the grammar is weird. The summary is not ready at this point, so how about we just print the same thing as the print above?

---

**File:** `tuner/examples/simple/simple_tuner.py:157`

```diff
@@ -141,18 +154,25 @@ def main():
             return
 
         print("Benchmarking compiled model candidates...")
+        logging.info(f"Summarization about top model candidates:")
```

**Comment:**
Same here

---

**File:** `tuner/examples/simple/simple_tuner.py:177`

```diff
@@ -141,18 +154,25 @@ def main():
             return
 
         print("Benchmarking compiled model candidates...")
+        logging.info(f"Summarization about top model candidates:")
         simple_tuner.benchmark_flags = model_benchmark_flags
         simple_tuner.benchmark_timeout = 60
         top_model_candidates = libtuner.benchmark(
+            tuner_context,
             args,
             path_config,
             compiled_model_candidates,
             candidate_trackers,
             simple_tuner,
             args.simple_num_model_candidates,
         )
-
+        logging.info(f"Top model candidates: {top_model_candidates}")
+        for id in top_model_candidates:
+            logging.info(f"{candidate_trackers[id].spec_path.resolve()}")
         print(f"Top model candidates: {top_model_candidates}")
 
         print("Check the detailed execution logs in:")
         print(path_config.run_log.resolve())
+
+        print("Check the tuning results in:")
```

**Comment:**
```suggestion
        print("Check the summary in:")
```

---

**File:** `tuner/tuner/libtuner.py:898`

```diff
@@ -875,11 +889,13 @@ def get_speedup(result: BenchmarkResult) -> float:
             speedup = f"{round(get_speedup(r) * 100, 2)}% of baseline"
         else:
             speedup = "baseline unavailable"
-        logging.info(f"Candidate {r.candidate_id} time: {r.time:.2f} ({speedup})")
+        result = f"Candidate {r.candidate_id} time: {r.time:.2f} ms ({speedup})"
+        logging.info(result)
     return best_results
 
 
 def benchmark(
+    tuner_context: TunerContext,
```

**Comment:**
Do we still need this?

---

**File:** `tuner/tuner/common.py:50`

```diff
@@ -46,6 +46,9 @@ def __enter__(self) -> "TunerContext":
         self.mlir_ctx.__enter__()
         return self
 
+    def add_logging_handler(self, handler: logging.Handler) -> None:
+        self.logger.addHandler(handler)
```

**Comment:**
I don't think we need this helper, users can access the logger directly.

---

**File:** `tuner/tuner/libtuner.py:100`

```diff
@@ -92,9 +93,12 @@ def _name_base_dir(self) -> Path:
         base_dir = Path(f"./tuning_{timestamp}")
         return base_dir
 
-    def _set_run_log(self, run_log: Path):
+    def set_run_log(self, run_log: Path):
         object.__setattr__(self, "run_log", run_log)
 
+    def set_summary_log(self, summary_log: Path):
+        object.__setattr__(self, "summary_log", summary_log)
```

**Comment:**
Do we need to keep track of the summary log here? I'd think that the simple tuner has the path and that `libtuner` doesn't have to know about it

---


---


## [PR #809](https://github.com/nod-ai/amd-shark-ai/pull/809): [tuner] reduce log file size

### Review Summary

**APPROVED** (2025-01-10)

LGTM but this needs to be rebased

**APPROVED** (2025-01-10)


---


## [PR #797](https://github.com/nod-ai/amd-shark-ai/pull/797): [tuner] clean up candidate gen

### Review Summary

**APPROVED** (2025-01-09)

Nice, thanks!


---


## [PR #789](https://github.com/nod-ai/amd-shark-ai/pull/789): [tuner] Add BaselineResultHandler class

### Review Summary

**CHANGES_REQUESTED** (2025-01-09)

**COMMENTED** (2025-01-09)

**COMMENTED** (2025-01-10)

Can you edit the PR description and explain what error conditions you considered and how they are handled?

**COMMENTED** (2025-01-10)

**COMMENTED** (2025-01-13)

**COMMENTED** (2025-01-17)

**COMMENTED** (2025-01-17)

**COMMENTED** (2025-01-17)

**COMMENTED** (2025-01-17)

**COMMENTED** (2025-01-17)

**COMMENTED** (2025-01-17)

**COMMENTED** (2025-01-18)

**APPROVED** (2025-01-19)

LGTM % one suggestion.

**CHANGES_REQUESTED** (2025-01-20)

**APPROVED** (2025-01-20)


### Code Comments

**File:** `tuner/tuner/libtuner.py:734`

```diff
@@ -727,6 +727,35 @@ def collision_handler(index_hash_list: list[tuple[int, str]]) -> tuple[bool, lis
     return collision_detected, unique_indexes
 
 
+def benchmark_candidates(candidate_indices, devices, tuning_client, candidate_trackers):
+    """
+    Runs the benchmarking for a given list of candidate indices.
+    """
+    # Create worker context queue
```

**Comment:**
When writing comments, focus on **why** we are doing certain things, not **what** the code is doing. The **what** is usually redundant because it carries the same information content as the code.
```suggestion
```

---

**File:** `tuner/tuner/libtuner.py:764`

```diff
@@ -727,6 +727,35 @@ def collision_handler(index_hash_list: list[tuple[int, str]]) -> tuple[bool, lis
     return collision_detected, unique_indexes
 
 
+def benchmark_candidates(candidate_indices, devices, tuning_client, candidate_trackers):
+    """
+    Runs the benchmarking for a given list of candidate indices.
+    """
+    # Create worker context queue
+    worker_context_queue = create_worker_context_queue(devices)
+
+    # Prepare task list
+    task_list = [
```

**Comment:**
```suggestion
    task_list = [
```

---

**File:** `tuner/tuner/libtuner.py:756`

```diff
@@ -727,6 +727,35 @@ def collision_handler(index_hash_list: list[tuple[int, str]]) -> tuple[bool, lis
     return collision_detected, unique_indexes
 
 
+def benchmark_candidates(candidate_indices, devices, tuning_client, candidate_trackers):
+    """
+    Runs the benchmarking for a given list of candidate indices.
+    """
+    # Create worker context queue
+    worker_context_queue = create_worker_context_queue(devices)
+
+    # Prepare task list
+    task_list = [
+        BenchmarkPack(
+            iree_benchmark_module_flags=tuning_client.get_iree_benchmark_module_flags(),
+            benchmark_timeout=tuning_client.get_benchmark_timeout_s(),
+            candidate_tracker=candidate_trackers[idx],
+        )
+        for idx in candidate_indices
+    ]
+
+    # Perform benchmarking
+    benchmark_results = multiprocess_progress_wrapper(
+        num_worker=len(devices),
+        task_list=task_list,
+        function=run_iree_benchmark_module_command,
+        initializer=init_worker_context,
+        initializer_inputs=(worker_context_queue,),
+    )
+
+    return benchmark_results
```

**Comment:**
```suggestion
    # Perform benchmarking.
    return multiprocess_progress_wrapper(
        num_worker=len(devices),
        task_list=task_list,
        function=run_iree_benchmark_module_command,
        initializer=init_worker_context,
        initializer_inputs=(worker_context_queue,),
    )
```

---

**File:** `tuner/tuner/libtuner.py:889`

```diff
@@ -873,4 +885,38 @@ def get_speedup(result: BenchmarkResult) -> float:
         )
 
     top_candidates = [result.candidate_id for result in best_results]
+
+    # Benchmarking baselines on each involved device again to check performance regression on devices.
```

**Comment:**
```suggestion
    # Benchmarking baselines again to check for performance regressions. These may indicate machine instability, overheating, etc.
```

---

**File:** `tuner/tuner/libtuner.py:904`

```diff
@@ -873,4 +885,38 @@ def get_speedup(result: BenchmarkResult) -> float:
         )
 
     top_candidates = [result.candidate_id for result in best_results]
+
+    # Benchmarking baselines on each involved device again to check performance regression on devices.
+    post_baseline_indices = [0] * len(args.devices)
+    post_baseline_results = benchmark_candidates(
+        candidate_indices=post_baseline_indices,
+        devices=args.devices,
+        tuning_client=tuning_client,
+        candidate_trackers=candidate_trackers,
+    )
+
+    post_baseline_times_by_device = {}
+    for r in post_baseline_results:
+        post_baseline_times_by_device[r.device_id] = r.time
+
+    assert (
+        baseline_times_by_device.keys() == post_baseline_times_by_device.keys()
+    ), "Error: The device IDs in baseline and post-baseline results do not match."
```

**Comment:**
```suggestion
    ), "Device ID mismatch between baseline runs."
```

---

**File:** `tuner/tuner/libtuner.py:908`

```diff
@@ -873,4 +885,38 @@ def get_speedup(result: BenchmarkResult) -> float:
         )
 
     top_candidates = [result.candidate_id for result in best_results]
+
+    # Benchmarking baselines on each involved device again to check performance regression on devices.
+    post_baseline_indices = [0] * len(args.devices)
+    post_baseline_results = benchmark_candidates(
+        candidate_indices=post_baseline_indices,
+        devices=args.devices,
+        tuning_client=tuning_client,
+        candidate_trackers=candidate_trackers,
+    )
+
+    post_baseline_times_by_device = {}
+    for r in post_baseline_results:
+        post_baseline_times_by_device[r.device_id] = r.time
+
+    assert (
+        baseline_times_by_device.keys() == post_baseline_times_by_device.keys()
+    ), "Error: The device IDs in baseline and post-baseline results do not match."
+
+    regression_detected = False
+    for device_id in baseline_times_by_device:
+        baseline_time = baseline_times_by_device[device_id]
+        post_time = post_baseline_times_by_device[device_id]
```

**Comment:**
What happens when benchmarking fails altogether and there's no time?

---

**File:** `tuner/tuner/libtuner.py:921`

```diff
@@ -873,4 +885,38 @@ def get_speedup(result: BenchmarkResult) -> float:
         )
 
     top_candidates = [result.candidate_id for result in best_results]
+
+    # Benchmarking baselines on each involved device again to check performance regression on devices.
+    post_baseline_indices = [0] * len(args.devices)
+    post_baseline_results = benchmark_candidates(
+        candidate_indices=post_baseline_indices,
+        devices=args.devices,
+        tuning_client=tuning_client,
+        candidate_trackers=candidate_trackers,
+    )
+
+    post_baseline_times_by_device = {}
+    for r in post_baseline_results:
+        post_baseline_times_by_device[r.device_id] = r.time
+
+    assert (
+        baseline_times_by_device.keys() == post_baseline_times_by_device.keys()
+    ), "Error: The device IDs in baseline and post-baseline results do not match."
+
+    regression_detected = False
+    for device_id in baseline_times_by_device:
+        baseline_time = baseline_times_by_device[device_id]
+        post_time = post_baseline_times_by_device[device_id]
+
+        if post_time > baseline_time * 1.03:
+            regression_detected = True
+            percentage_slower = ((post_time - baseline_time) / baseline_time) * 100
+            logging.warning(
+                f"Performance regression detected on device {device_id}: "
+                f"Baseline time = {baseline_time}, Post-baseline time = {post_time}, "
+                f"Slower by {percentage_slower:.3f}%"
+            )
+
+    if not regression_detected:
+        logging.debug("No performance regressions detected.")
```

**Comment:**
```suggestion
```

---

**File:** `tuner/tuner/libtuner.py:745`

```diff
@@ -727,6 +727,31 @@ def collision_handler(index_hash_list: list[tuple[int, str]]) -> tuple[bool, lis
     return collision_detected, unique_indexes
 
 
+def benchmark_candidates(candidate_indices, devices, tuning_client, candidate_trackers):
+    """
+    Runs the benchmarking for a given list of candidate indices.
+    """
+    worker_context_queue = create_worker_context_queue(devices)
+
+    task_list = [
+        BenchmarkPack(
+            iree_benchmark_module_flags=tuning_client.get_iree_benchmark_module_flags(),
+            benchmark_timeout=tuning_client.get_benchmark_timeout_s(),
+            candidate_tracker=candidate_trackers[idx],
+        )
+        for idx in candidate_indices
+    ]
+
+    # Perform benchmarking
```

**Comment:**
Use proper punctuation.

```suggestion
    # Perform benchmarking.
```

---

**File:** `tuner/tuner/libtuner.py:885`

```diff
@@ -873,4 +881,38 @@ def get_speedup(result: BenchmarkResult) -> float:
         )
 
     top_candidates = [result.candidate_id for result in best_results]
+
+    # Benchmarking baselines again to check for performance regressions. These may indicate machine instability, overheating, etc.
```

**Comment:**
This seems to exceed the column limit, doesn't the formatter complain about it?

---

**File:** `tuner/tuner/libtuner.py:903`

```diff
@@ -873,4 +881,38 @@ def get_speedup(result: BenchmarkResult) -> float:
         )
 
     top_candidates = [result.candidate_id for result in best_results]
+
+    # Benchmarking baselines again to check for performance regressions. These may indicate machine instability, overheating, etc.
+    post_baseline_indices = [0] * len(args.devices)
+    post_baseline_results = benchmark_candidates(
+        candidate_indices=post_baseline_indices,
+        devices=args.devices,
+        tuning_client=tuning_client,
+        candidate_trackers=candidate_trackers,
+    )
+
+    post_baseline_times_by_device = {}
+    for r in post_baseline_results:
+        post_baseline_times_by_device[r.device_id] = r.time
+
+    assert (
+        baseline_times_by_device.keys() == post_baseline_times_by_device.keys()
+    ), "Device ID mismatch between baseline runs."
+
+    for device_id in baseline_times_by_device:
+        if device_id not in baseline_times_by_device:
```

**Comment:**
Have you tested this code? Make sure you exercise the failure conditions

---

**File:** `tuner/tuner/libtuner.py:905`

```diff
@@ -873,4 +881,38 @@ def get_speedup(result: BenchmarkResult) -> float:
         )
 
     top_candidates = [result.candidate_id for result in best_results]
+
+    # Benchmarking baselines again to check for performance regressions. These may indicate machine instability, overheating, etc.
+    post_baseline_indices = [0] * len(args.devices)
+    post_baseline_results = benchmark_candidates(
+        candidate_indices=post_baseline_indices,
+        devices=args.devices,
+        tuning_client=tuning_client,
+        candidate_trackers=candidate_trackers,
+    )
+
+    post_baseline_times_by_device = {}
+    for r in post_baseline_results:
+        post_baseline_times_by_device[r.device_id] = r.time
+
+    assert (
+        baseline_times_by_device.keys() == post_baseline_times_by_device.keys()
+    ), "Device ID mismatch between baseline runs."
+
+    for device_id in baseline_times_by_device:
+        if device_id not in baseline_times_by_device:
+            logging.warning(f"Baseline time missing for device {device_id}")
+            continue
```

**Comment:**
I think the correct way to handle this to issue a warning when any of the baseline benchmarks fail, but more importantly try to recover from that: try to use the time from the other baseline run, if that succeeded.

---

**File:** `tuner/tuner/candidate_gen.py:357`

```diff
@@ -355,7 +354,7 @@ def main():
             prefetch_shared_memory=args.prefetch_shared_memory_options,
             no_reduce_shared_memory_bank_conflicts=args.no_reduce_shared_memory_bank_conflicts_options,
         )
-        specs: list[ir.Module] = generate_configs_and_td_specs(
+        specs = generate_configs_and_td_specs(
```

**Comment:**
Why do we want to remove the type hints?

---

**File:** `tuner/tuner/candidate_gen.py:371`

```diff
@@ -369,7 +368,7 @@ def main():
             spec_path = spec_dir / f"{candidate_num}_spec.mlir"
             spec_dir.mkdir(parents=True, exist_ok=True)
             with open(spec_path, "w") as f:
-                local_scope_spec_str: str = spec.operation.get_asm(use_local_scope=True)
+                local_scope_spec_str = spec.operation.get_asm(use_local_scope=True)
```

**Comment:**
also here

---

**File:** `tuner/tuner/libtuner.py:227`

```diff
@@ -221,6 +221,53 @@ def validate_devices(user_devices: list[str]) -> None:
         )
 
 
+def validate_benchmark_results(
+    benchmark_results: list[BenchmarkResult],
+) -> list[BenchmarkResult]:
```

**Comment:**
Based on the function name alone, it's impossible to tell what this code does... Could we call it something like: `get_valid_benchmark_results` and add a one-line docstring explaining what is considered valid?

---

**File:** `tuner/tuner/libtuner.py:246`

```diff
@@ -221,6 +221,53 @@ def validate_devices(user_devices: list[str]) -> None:
         )
 
 
+def validate_benchmark_results(
+    benchmark_results: list[BenchmarkResult],
+) -> list[BenchmarkResult]:
+    filtered_benchmark_results = [r for r in benchmark_results if math.isfinite(r.time)]
+    if len(filtered_benchmark_results) == 0:
+        logging.error("No successful candidate benchmarks.")
+
+    return filtered_benchmark_results
+
+
+def map_baseline_by_device(baseline_results: list[BenchmarkResult]) -> dict[str, float]:
+    return {r.device_id: r.time for r in baseline_results}
```

**Comment:**
Should we check that the device IDs are unique? Otherwise this will result in data loss.

---

**File:** `tuner/tuner/libtuner.py:253`

```diff
@@ -221,6 +221,53 @@ def validate_devices(user_devices: list[str]) -> None:
         )
 
 
+def validate_benchmark_results(
+    benchmark_results: list[BenchmarkResult],
+) -> list[BenchmarkResult]:
+    filtered_benchmark_results = [r for r in benchmark_results if math.isfinite(r.time)]
+    if len(filtered_benchmark_results) == 0:
+        logging.error("No successful candidate benchmarks.")
+
+    return filtered_benchmark_results
+
+
+def map_baseline_by_device(baseline_results: list[BenchmarkResult]) -> dict[str, float]:
+    return {r.device_id: r.time for r in baseline_results}
+
+
+def validate_baselines_device_ids_match(
+    first_baseline_by_device: dict[str, float],
+    second_baseline_by_device: dict[str, float],
+) -> bool:
```

**Comment:**
I think it's better to expand this check inline, the helper function doesn't seem to help that much -- it's just a comparison

---

**File:** `tuner/tuner/libtuner.py:269`

```diff
@@ -221,6 +221,53 @@ def validate_devices(user_devices: list[str]) -> None:
         )
 
 
+def validate_benchmark_results(
+    benchmark_results: list[BenchmarkResult],
+) -> list[BenchmarkResult]:
+    filtered_benchmark_results = [r for r in benchmark_results if math.isfinite(r.time)]
+    if len(filtered_benchmark_results) == 0:
+        logging.error("No successful candidate benchmarks.")
+
+    return filtered_benchmark_results
+
+
+def map_baseline_by_device(baseline_results: list[BenchmarkResult]) -> dict[str, float]:
+    return {r.device_id: r.time for r in baseline_results}
+
+
+def validate_baselines_device_ids_match(
+    first_baseline_by_device: dict[str, float],
+    second_baseline_by_device: dict[str, float],
+) -> bool:
+    return first_baseline_by_device.keys() == second_baseline_by_device.keys()
+
+
+def validate_baseline_regression(
+    first_baseline_by_device: dict[str, float],
+    second_baseline_by_device: dict[str, float],
+) -> list[str]:
```

**Comment:**
Similar here, it's impossible to tell what this function does based on the name alone. We need a better name and some comment with an explanation.

---

**File:** `tuner/tuner/libtuner.py:1028`

```diff
@@ -892,41 +964,49 @@ def benchmark(
         logging.warning("No candidates to benchmark.")
         return []
 
-    task_list = [
-        BenchmarkPack(
-            iree_benchmark_module_flags=tuning_client.get_iree_benchmark_module_flags(),
-            benchmark_timeout=tuning_client.get_benchmark_timeout_s(),
-            candidate_tracker=candidate_trackers[i],
-        )
-        for i in compiled_candidates
-        if i != 0
-    ]
-    worker_context_queue = create_worker_context_queue(args.devices)
-    candidate_results: list[BenchmarkResult] = multiprocess_progress_wrapper(
-        num_worker=len(args.devices),
-        task_list=task_list,
-        function=run_iree_benchmark_module_command,
-        initializer=init_worker_context,
-        initializer_inputs=(worker_context_queue,),
+    # Benchmarking baselines on each involved device.
+    baseline_indices = [0] * len(args.devices)
+    baseline_results = benchmark_candidates(
+        candidate_indices=baseline_indices,
+        devices=args.devices,
+        tuning_client=tuning_client,
+        candidate_trackers=candidate_trackers,
     )
 
-    # Benchmarking baselines on each involved device.
-    worker_context_queue = create_worker_context_queue(args.devices)
-    baseline_task_list = [
-        BenchmarkPack(
-            iree_benchmark_module_flags=tuning_client.get_iree_benchmark_module_flags(),
-            benchmark_timeout=tuning_client.get_benchmark_timeout_s(),
-            candidate_tracker=candidate_trackers[0],
-        )
-    ] * len(args.devices)
-    baseline_results: list[BenchmarkResult] = multiprocess_progress_wrapper(
-        num_worker=len(args.devices),
-        task_list=baseline_task_list,
-        function=run_iree_benchmark_module_command,
-        initializer=init_worker_context,
-        initializer_inputs=(worker_context_queue,),
+    baseline_times_by_device = map_baseline_by_device(baseline_results)
+
+    candidate_indices = [i for i in compiled_candidates if i != 0]
+    candidate_results = benchmark_candidates(
+        candidate_indices=candidate_indices,
+        devices=args.devices,
+        tuning_client=tuning_client,
+        candidate_trackers=candidate_trackers,
```

**Comment:**
Should we put baseline benchmarking into it's own function that return a map from devices to results?

---

**File:** `tuner/tuner/libtuner_test.py:262`

```diff
@@ -233,3 +231,64 @@ def test_select_best_benchmark_results() -> None:
 
 def test_enum_collision():
     from iree.compiler.dialects import linalg, vector, iree_gpu, iree_codegen, iree_input  # type: ignore
+
+
+def test_validate_benchmark_results():
+    benchmark_results = [
+        libtuner.BenchmarkResult(0, math.inf, "hip://0"),
+    ]
+
+    result = libtuner.validate_benchmark_results(benchmark_results)
+    assert result == []
+
+    benchmark_results = [
+        libtuner.BenchmarkResult(0, math.inf, "hip://0"),
+        libtuner.BenchmarkResult(0, 0.1, "hip://1"),
+    ]
+    result = libtuner.validate_benchmark_results(benchmark_results)
+    assert len(result) == 1
+    assert result[0].candidate_id == 0
+    assert result[0].time == 0.1
+    assert result[0].device_id == "hip://1"
+
+
+def test_validate_baselines_device_id_match():
+    first_baseline = {"hip://0": 1000.0, "hip://1": 2000.0}
+    second_baseline = {"hip://1": 1500.0, "hip://2": 2500.0}
+
+    result = libtuner.validate_baselines_device_ids_match(
+        first_baseline, second_baseline
+    )
+    assert result is False
```

**Comment:**
```suggestion
    assert not result
```

---

**File:** `tuner/tuner/libtuner_test.py:270`

```diff
@@ -233,3 +231,64 @@ def test_select_best_benchmark_results() -> None:
 
 def test_enum_collision():
     from iree.compiler.dialects import linalg, vector, iree_gpu, iree_codegen, iree_input  # type: ignore
+
+
+def test_validate_benchmark_results():
+    benchmark_results = [
+        libtuner.BenchmarkResult(0, math.inf, "hip://0"),
+    ]
+
+    result = libtuner.validate_benchmark_results(benchmark_results)
+    assert result == []
+
+    benchmark_results = [
+        libtuner.BenchmarkResult(0, math.inf, "hip://0"),
+        libtuner.BenchmarkResult(0, 0.1, "hip://1"),
+    ]
+    result = libtuner.validate_benchmark_results(benchmark_results)
+    assert len(result) == 1
+    assert result[0].candidate_id == 0
+    assert result[0].time == 0.1
+    assert result[0].device_id == "hip://1"
+
+
+def test_validate_baselines_device_id_match():
+    first_baseline = {"hip://0": 1000.0, "hip://1": 2000.0}
+    second_baseline = {"hip://1": 1500.0, "hip://2": 2500.0}
+
+    result = libtuner.validate_baselines_device_ids_match(
+        first_baseline, second_baseline
+    )
+    assert result is False
+
+    first_baseline = {"hip://0": 1000.0, "hip://1": 2000.0}
+    second_baseline = {"hip://0": 1500.0, "hip://1": 2500.0}
+
+    result = libtuner.validate_baselines_device_ids_match(
+        first_baseline, second_baseline
+    )
+    assert result is True
```

**Comment:**
```suggestion
    assert result
```

---

**File:** `tuner/tuner/candidate_gen.py:357`

```diff
@@ -355,7 +354,7 @@ def main():
             prefetch_shared_memory=args.prefetch_shared_memory_options,
             no_reduce_shared_memory_bank_conflicts=args.no_reduce_shared_memory_bank_conflicts_options,
         )
-        specs: list[ir.Module] = generate_configs_and_td_specs(
+        specs = generate_configs_and_td_specs(
```

**Comment:**
make this function typed then?

---


---


## [PR #770](https://github.com/nod-ai/amd-shark-ai/pull/770): [Tuner] Fix context management

### Review Summary

**CHANGES_REQUESTED** (2025-01-07)

**COMMENTED** (2025-01-07)

Can you rebase this on top of https://github.com/nod-ai/shark-ai/pull/771 ?

**APPROVED** (2025-01-08)

LGTM % one minor issue


### Code Comments

**File:** `tuner/tuner/libtuner.py:184`

```diff
@@ -170,10 +170,18 @@ def get_compiled_model_index(self, file_path: Path) -> int:
 
 
 class TuningClient(ABC):
-    def __init__(self):
-        mlir_ctx = ir.Context()
-        logger = logging.getLogger("tune")
-        self.tuner_context = TunerContext(mlir_ctx, logger)
+    def __init__(self, tuner_context: TunerContext):
+        self.tuner_context = tuner_context
+
+    def __enter__(self):
+        # Enter the context of TunerContext
+        self.tuner_context.__enter__()
+        return self
+
+    def __exit__(self, exc_type, exc_value, traceback):
+        # Exit the context of TunerContext
+        self.tuner_context.__exit__(exc_type, exc_value, traceback)
+        return False
```

**Comment:**
TuningClient shouldn't define these

---

**File:** `tuner/examples/dispatch/dispatch_tuner.py:114`

```diff
@@ -103,7 +109,10 @@ def main():
     path_config.base_dir.mkdir(parents=True, exist_ok=True)
     path_config.output_unilog.touch()
     candidate_trackers: list[libtuner.CandidateTracker] = []
-    dispatch_tuner = DispatchTuner()
+    mlir_ctx = ir.Context()
+    logger = logging.getLogger("tune")
+    tuner_context = TunerContext(mlir_ctx, logger)
```

**Comment:**
I don't understand why we define these outside of the tuner class?

---

**File:** `tuner/tuner/common.py:50`

```diff
@@ -43,6 +43,14 @@ def __init__(self, mlir_ctx: ir.Context, logger: logging.Logger):
         self.logger: logging.Logger = logger
         self.type: CommonTypes = CommonTypes(mlir_ctx)
 
+    def __enter__(self):
+        self.mlir_ctx.__enter__()
+        return self
+
+    def __exit__(self, exc_type, exc_value, traceback):
```

**Comment:**
Can we add type hints for these?

---

**File:** `tuner/examples/test/tuner_test.py:8`

```diff
@@ -5,13 +5,16 @@
 # SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 
 import argparse
+import logging
```

**Comment:**
Do we need this import? I don't see it used.

---

**File:** `tuner/tuner/candidate_gen_test.py:33`

```diff
@@ -27,9 +27,12 @@ def tuner_ctx() -> Generator[common.TunerContext, None, None]:
     from logging import Logger
     from unittest.mock import MagicMock
 
-    with ir.Context() as ctx:
-        logger: Logger = MagicMock(spec=Logger)
-        yield common.TunerContext(ctx, logger)
+    # Mock the logger
+    mock_logger = MagicMock(spec=Logger)
+
+    # Use TunerContext with the mocked logger
```

**Comment:**
I don't think these comments help here

```suggestion
    mock_logger = MagicMock(spec=Logger)
```

---

**File:** `tuner/tuner/common.py:7`

```diff
@@ -4,12 +4,13 @@
 # See https://llvm.org/LICENSE.txt for license information.
 # SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 
-import re
+from __future__ import annotations
```

**Comment:**
Why do we need this import? 

---

**File:** `tuner/tuner/common.py:13`

```diff
@@ -4,12 +4,13 @@
 # See https://llvm.org/LICENSE.txt for license information.
 # SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 
-import re
+from __future__ import annotations
 import logging
 from dataclasses import astuple, dataclass
 from enum import Enum
 from typing import Optional
 from typing import Any
+from typing_extensions import Literal
```

**Comment:**
Why do we need this?

---

**File:** `tuner/tuner/common.py:53`

```diff
@@ -38,10 +39,20 @@ def getI64(self, value: int) -> ir.IntegerAttr:
 
 
 class TunerContext:
-    def __init__(self, mlir_ctx: ir.Context, logger: logging.Logger):
-        self.mlir_ctx: ir.Context = mlir_ctx
-        self.logger: logging.Logger = logger
-        self.type: CommonTypes = CommonTypes(mlir_ctx)
+    def __init__(self, logger: Optional[logging.Logger] = None):
+        self.mlir_ctx: ir.Context = ir.Context()
+        self.logger: logging.Logger = logger or logging.getLogger(
+            "tune"
+        )  # Default to "tune" logger
+        self.type: CommonTypes = CommonTypes(self.mlir_ctx)
+
+    def __enter__(self) -> TunerContext:
+        self.mlir_ctx.__enter__()
+        return self
+
+    def __exit__(self, exc_type, exc_value, traceback) -> Literal[False]:
```

**Comment:**
Can we add type hints for `exc_type`, `exc_value`, and `traceback`? Also, can we make this return whatever the nested exit returns?

---

**File:** `tuner/tuner/common.py:55`

```diff
@@ -38,10 +39,20 @@ def getI64(self, value: int) -> ir.IntegerAttr:
 
 
 class TunerContext:
-    def __init__(self, mlir_ctx: ir.Context, logger: logging.Logger):
-        self.mlir_ctx: ir.Context = mlir_ctx
-        self.logger: logging.Logger = logger
-        self.type: CommonTypes = CommonTypes(mlir_ctx)
+    def __init__(self, logger: Optional[logging.Logger] = None):
+        self.mlir_ctx: ir.Context = ir.Context()
+        self.logger: logging.Logger = logger or logging.getLogger(
+            "tune"
+        )  # Default to "tune" logger
+        self.type: CommonTypes = CommonTypes(self.mlir_ctx)
+
+    def __enter__(self) -> TunerContext:
+        self.mlir_ctx.__enter__()
+        return self
+
+    def __exit__(self, exc_type, exc_value, traceback) -> Literal[False]:
+        self.mlir_ctx.__exit__(exc_type, exc_value, traceback)
+        return False
```

**Comment:**
```suggestion
    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return self.mlir_ctx.__exit__(exc_type, exc_value, traceback)
```

---

**File:** `tuner/tuner/common_test.py:28`

```diff
@@ -23,9 +23,12 @@ def tuner_ctx() -> Generator[common.TunerContext, None, None]:
     from logging import Logger
     from unittest.mock import MagicMock
 
-    with ir.Context() as ctx:
-        logger: Logger = MagicMock(spec=Logger)
-        yield common.TunerContext(ctx, logger)
+    # Mock the logger
+    mock_logger = MagicMock(spec=Logger)
+
+    # Use TunerContext with the mocked logger
+    with common.TunerContext(logger=mock_logger) as ctx:
+        yield ctx
```

**Comment:**
Also here

---

**File:** `tuner/tuner/dispatch_constraints_test.py:30`

```diff
@@ -25,9 +25,12 @@ def tuner_ctx() -> Generator[common.TunerContext, None, None]:
     from logging import Logger
     from unittest.mock import MagicMock
 
-    with ir.Context() as ctx:
-        logger: Logger = MagicMock(spec=Logger)
-        yield common.TunerContext(ctx, logger)
+    # Mock the logger
+    mock_logger = MagicMock(spec=Logger)
+
+    # Use TunerContext with the mocked logger
+    with common.TunerContext(logger=mock_logger) as ctx:
+        yield ctx
```

**Comment:**
Also here

---

**File:** `tuner/tuner/dispatch_parser_test.py:32`

```diff
@@ -27,9 +27,12 @@ def tuner_ctx() -> Generator[common.TunerContext, None, None]:
     from logging import Logger
     from unittest.mock import MagicMock
 
-    with ir.Context() as ctx:
-        logger: Logger = MagicMock(spec=Logger)
-        yield common.TunerContext(ctx, logger)
+    # Mock the logger
+    mock_logger = MagicMock(spec=Logger)
+
+    # Use TunerContext with the mocked logger
+    with common.TunerContext(logger=mock_logger) as ctx:
+        yield ctx
```

**Comment:**
Also here

---

**File:** `tuner/tuner/libtuner.py:1061`

```diff
@@ -1058,11 +1058,11 @@ def generate_candidate_specs(
         # source mlir.
         mlir_text = strip_compilation_info(path_config.template_mlir)
         mlir_module = dispatch_parser.parse_mlir(mlir_text, tuning_client.tuner_context)
-        with tuning_client.tuner_context.mlir_ctx:
+        with tuning_client.tuner_context as tuner_context:
```

**Comment:**
Do we need this after your other changes?

---

**File:** `tuner/tuner/common.py:44`

```diff
@@ -38,10 +37,24 @@ def getI64(self, value: int) -> ir.IntegerAttr:
 
 
 class TunerContext:
-    def __init__(self, mlir_ctx: ir.Context, logger: logging.Logger):
-        self.mlir_ctx: ir.Context = mlir_ctx
-        self.logger: logging.Logger = logger
-        self.type: CommonTypes = CommonTypes(mlir_ctx)
+    def __init__(self, logger: Optional[logging.Logger] = None):
+        self.mlir_ctx: ir.Context = ir.Context()
+        self.logger: logging.Logger = logger or logging.getLogger(
+            "tune"
+        )  # Default to "tune" logger
```

**Comment:**
```suggestion
        self.logger: logging.Logger = logger or logging.getLogger("tune")
```

I don't see how this comments helps

---


---


## [PR #678](https://github.com/nod-ai/amd-shark-ai/pull/678): [tuner]: use compilation_info binding

### Review Summary

**APPROVED** (2024-12-12)

Nice 👍


---


## [PR #669](https://github.com/nod-ai/amd-shark-ai/pull/669): [tuner]: use translation_info binding

### Review Summary

**CHANGES_REQUESTED** (2024-12-10)

**COMMENTED** (2024-12-10)

**CHANGES_REQUESTED** (2024-12-10)

**COMMENTED** (2024-12-11)

**COMMENTED** (2024-12-11)

**APPROVED** (2024-12-11)

LGTM % comments

**COMMENTED** (2024-12-11)

**COMMENTED** (2024-12-11)

**COMMENTED** (2024-12-11)

**COMMENTED** (2024-12-11)


### Code Comments

**File:** `tuner/tuner/candidate_gen_test.py:61`

```diff
@@ -56,13 +57,17 @@ def test_apply_params_mmt(tuner_ctx: common.TunerContext) -> None:
         subgroup_m_count=16,
         subgroup_n_count=16,
     )
+    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+        iree_codegen.DispatchLoweringPassPipeline.None_
```

**Comment:**
The pipeline should be llvmgpuvectordistribution. Also everywhere else

---

**File:** `tuner/tuner/candidate_gen_test.py:64`

```diff
@@ -56,13 +57,17 @@ def test_apply_params_mmt(tuner_ctx: common.TunerContext) -> None:
         subgroup_m_count=16,
         subgroup_n_count=16,
     )
+    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+        iree_codegen.DispatchLoweringPassPipeline.None_
+    )
+    pipeline_option = iree_gpu.PipelineOptionsAttr.get(prefetch_shared_memory=True)
+    pipeline_option_dict = ir.DictAttr.get({"gpu_pipeline_options": pipeline_option})
```

**Comment:**
Can you put the gpu_pipeline_options key name as a constant somewhere in `common.py`?

---

**File:** `tuner/tuner/candidate_gen.py:156`

```diff
@@ -147,7 +150,7 @@ def get_transform_function_mmt(
     %config = transform.param.constant #iree_codegen.compilation_info<
         lowering_config = {configuration.lowering_config}>,
         translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
-        workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {configuration.subgroup_size},
+        workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {configuration.translation_info.subgroup_size},
         {{mma_schedule = #iree_gpu.mma_schedule<
             intrinsic = {intrinsic},
             subgroup_m_count = {subgroup_m_count}, subgroup_n_count = {subgroup_n_count}>
```

**Comment:**
Can we print the whole translation info attr here?

---

**File:** `tuner/tuner/candidate_gen.py:225`

```diff
@@ -219,7 +222,7 @@ def get_transform_function_conv(
         %config = transform.param.constant #iree_codegen.compilation_info<
         lowering_config = {configuration.lowering_config}>,
         translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
-        workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {configuration.subgroup_size},
+        workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {configuration.translation_info.subgroup_size},
```

**Comment:**
Also here

---

**File:** `tuner/tuner/candidate_gen.py:292`

```diff
@@ -286,7 +289,7 @@ def get_transform_function_broadcast_rhs_mmt(
 %config = transform.param.constant #iree_codegen.compilation_info<
     lowering_config = {configuration.lowering_config}>,
     translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
-    workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {configuration.subgroup_size},
+    workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {configuration.translation_info.subgroup_size},
```

**Comment:**
Also here

---

**File:** `tuner/tuner/candidate_gen.py:377`

```diff
@@ -371,7 +374,7 @@ def get_transform_function_batch_mmt(
 %config = transform.param.constant #iree_codegen.compilation_info<
     lowering_config = {configuration.lowering_config}>,
     translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
-    workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {configuration.subgroup_size},
+    workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {configuration.translation_info.subgroup_size},
```

**Comment:**
Also here

---

**File:** `tuner/tuner/candidate_gen.py:449`

```diff
@@ -443,7 +446,7 @@ def get_transform_function_batch_matmul(
         %config = transform.param.constant #iree_codegen.compilation_info<
         lowering_config = {configuration.lowering_config}>,
         translation_info = #iree_codegen.translation_info<LLVMGPUPadAndVectorDistribute
-        workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {configuration.subgroup_size},
+        workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {configuration.translation_info.subgroup_size},
```

**Comment:**
Also here

---

**File:** `tuner/tuner/candidate_gen_test.py:61`

```diff
@@ -56,13 +57,19 @@ def test_apply_params_mmt(tuner_ctx: common.TunerContext) -> None:
         subgroup_m_count=16,
         subgroup_n_count=16,
     )
+    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorize
```

**Comment:**
this is not the right pipeline

---

**File:** `tuner/tuner/candidate_gen_test.py:129`

```diff
@@ -118,15 +125,23 @@ def test_apply_params_conv(tuner_ctx: common.TunerContext) -> None:
         subgroup_m_count=1,
         subgroup_n_count=4,
     )
+    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorize
```

**Comment:**
also here

---

**File:** `tuner/tuner/candidate_gen_test.py:210`

```diff
@@ -191,11 +206,19 @@ def test_apply_params_contract(tuner_ctx: common.TunerContext) -> None:
         subgroup_m_count=1,
         subgroup_n_count=4,
     )
+    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorize
```

**Comment:**
also here

---

**File:** `tuner/tuner/candidate_gen_test.py:273`

```diff
@@ -246,11 +269,19 @@ def test_apply_params_batch_matmul(tuner_ctx: common.TunerContext) -> None:
         subgroup_m_count=2,
         subgroup_n_count=2,
     )
+    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorize
```

**Comment:**
also here

---

**File:** `tuner/tuner/candidate_gen_test.py:339`

```diff
@@ -304,11 +335,19 @@ def test_apply_params_batch_mmt_float(tuner_ctx: common.TunerContext) -> None:
         subgroup_m_count=2,
         subgroup_n_count=2,
     )
+    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorize
```

**Comment:**
also here

---

**File:** `tuner/tuner/candidate_gen_test.py:403`

```diff
@@ -360,11 +399,19 @@ def test_apply_params_batch_mmt_int(tuner_ctx: common.TunerContext) -> None:
         subgroup_m_count=2,
         subgroup_n_count=2,
     )
+    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorize
```

**Comment:**
also here

---

**File:** `tuner/tuner/candidate_gen_test.py:491`

```diff
@@ -440,11 +487,19 @@ def test_apply_params_broadcast_rhs_mmt(tuner_ctx: common.TunerContext) -> None:
         subgroup_m_count=2,
         subgroup_n_count=2,
     )
+    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorize
```

**Comment:**
also here

---

**File:** `tuner/tuner/common.py:122`

```diff
@@ -112,13 +113,15 @@ def is_comptible(mma_intrinsic: iree_gpu.MMAIntrinsic) -> bool:
 
 @dataclass
 class Configuration:
-    subgroup_size: int
-    workgroup_size: list[int]
+    translation_info: iree_codegen.TranslationInfoAttr
     lowering_config: iree_gpu.LoweringConfigAttr
-    gpu_pipeline_options: iree_gpu.PipelineOptionsAttr
     waves_per_eu: int
 
 
+# The key name for GPUPipelineOptionsAttr in the translation info config dictionary.
+GPU_PIPELINE_OPTIONS = "gpu_pipeline_options"
```

**Comment:**
```suggestion
GPU_PIPELINE_OPTIONS_KEY = "gpu_pipeline_options"
```

---

**File:** `tuner/tuner/common_test.py:89`

```diff
@@ -84,11 +85,19 @@ def test_get_pipeline_config(tuner_ctx: common.TunerContext) -> None:
         subgroup_m_count=1,
         subgroup_n_count=1,
     )
+    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorize
```

**Comment:**
wrong pipeline

---

**File:** `tuner/tuner/common_test.py:229`

```diff
@@ -207,11 +225,19 @@ def test_get_lowering_config(tuner_ctx: common.TunerContext) -> None:
         == "#iree_gpu.lowering_config<{reduction = [0, 0, 16], subgroup_m_count = 1 : i64, subgroup_n_count = 1 : i64, workgroup = [4, 8, 0]}>"
     )
 
+    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorize
```

**Comment:**
wrong pipeline

---

**File:** `tuner/tuner/dispatch_constraints.py:248`

```diff
@@ -244,13 +244,20 @@ def generate_solutions(
             subgroup_m_count=lookup(sg_m_cnt),
             subgroup_n_count=lookup(sg_n_cnt),
         )
-        config = Configuration(
-            lookup(subgroup_size),
+        pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+            iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
```

**Comment:**
this one is correct

---

**File:** `tuner/tuner/dispatch_parser_test.py:55`

```diff
@@ -50,11 +51,19 @@ def test_get_mmt_tile_sizes(tuner_ctx: common.TunerContext) -> None:
         subgroup_m_count=1,
         subgroup_n_count=4,
     )
-    config = dispatch_parser.Configuration(
-        subgroup_size=0,
-        workgroup_size=[],
+    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorize
```

**Comment:**
wrong pipeline

---

**File:** `tuner/tuner/dispatch_parser_test.py:86`

```diff
@@ -73,11 +82,19 @@ def test_get_conv_tile_sizes(tuner_ctx: common.TunerContext) -> None:
         subgroup_m_count=1,
         subgroup_n_count=4,
     )
-    config = dispatch_parser.Configuration(
-        subgroup_size=64,
-        workgroup_size=[256, 1, 1],
+    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorize
```

**Comment:**
wrong pipeline

---

**File:** `tuner/tuner/dispatch_parser_test.py:116`

```diff
@@ -95,11 +112,19 @@ def test_get_contract_tile_sizes(tuner_ctx: common.TunerContext) -> None:
         subgroup_m_count=1,
         subgroup_n_count=1,
     )
-    config = dispatch_parser.Configuration(
-        subgroup_size=32,
-        workgroup_size=[16, 16, 1],
+    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorize
```

**Comment:**
wrong pipeline

---

**File:** `tuner/tuner/candidate_gen_test.py:63`

```diff
@@ -56,13 +57,19 @@ def test_apply_params_mmt(tuner_ctx: common.TunerContext) -> None:
         subgroup_m_count=16,
         subgroup_n_count=16,
     )
+    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorize
+    )
+    pipeline_option = iree_gpu.PipelineOptionsAttr.get(prefetch_shared_memory=True)
```

**Comment:**
```suggestion
    pipeline_options = iree_gpu.PipelineOptionsAttr.get(prefetch_shared_memory=True)
```

---

**File:** `tuner/tuner/candidate_gen_test.py:64`

```diff
@@ -56,13 +57,19 @@ def test_apply_params_mmt(tuner_ctx: common.TunerContext) -> None:
         subgroup_m_count=16,
         subgroup_n_count=16,
     )
+    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorize
+    )
+    pipeline_option = iree_gpu.PipelineOptionsAttr.get(prefetch_shared_memory=True)
+    pipeline_option_dict = ir.DictAttr.get(
```

**Comment:**
```suggestion
    pipeline_options_dict = ir.DictAttr.get(
```

---

**File:** `tuner/tuner/candidate_gen_test.py:63`

```diff
@@ -56,13 +57,19 @@ def test_apply_params_mmt(tuner_ctx: common.TunerContext) -> None:
         subgroup_m_count=16,
         subgroup_n_count=16,
     )
+    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorize
+    )
+    pipeline_option = iree_gpu.PipelineOptionsAttr.get(prefetch_shared_memory=True)
```

**Comment:**
also everywhere else

---

**File:** `tuner/tuner/candidate_gen_test.py:64`

```diff
@@ -56,13 +57,19 @@ def test_apply_params_mmt(tuner_ctx: common.TunerContext) -> None:
         subgroup_m_count=16,
         subgroup_n_count=16,
     )
+    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorize
+    )
+    pipeline_option = iree_gpu.PipelineOptionsAttr.get(prefetch_shared_memory=True)
+    pipeline_option_dict = ir.DictAttr.get(
```

**Comment:**
also everywhere else

---

**File:** `tuner/tuner/candidate_gen.py:325`

```diff
@@ -369,13 +321,8 @@ def get_transform_function_batch_mmt(
 transform.iree.match.cast_compatible_type %lhs = tensor<{problem_size.lhs_type}> : !transform.any_value
 transform.iree.match.cast_compatible_type %rhs = tensor<{problem_size.rhs_type}> : !transform.any_value
 %config = transform.param.constant #iree_codegen.compilation_info<
-    lowering_config = {configuration.lowering_config}>,
-    translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
-    workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {configuration.subgroup_size},
-    {{mma_schedule = #iree_gpu.mma_schedule<
-        intrinsic = {intrinsic},
-        subgroup_m_count = {subgroup_m_count}, subgroup_n_count = {subgroup_n_count}>
-    {extra_config}}}>
+    lowering_config = {configuration.lowering_config},
+    translation_info ={configuration.translation_info}
```

**Comment:**
```suggestion
    translation_info = {configuration.translation_info}
```

---

**File:** `tuner/tuner/candidate_gen_test.py:70`

```diff
@@ -56,14 +57,23 @@ def test_apply_params_mmt(tuner_ctx: common.TunerContext) -> None:
         subgroup_m_count=16,
         subgroup_n_count=16,
     )
+    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
+        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute
+    )
+    pipeline_options = iree_gpu.PipelineOptionsAttr.get(prefetch_shared_memory=True)
+    waves_per_eu_dict = ir.DictAttr.get({"amdgpu-waves-per-eu": ir.StringAttr.get("8")})
+    config_dict = ir.DictAttr.get(
+        {
+            common.GPU_PIPELINE_OPTIONS_KEY: pipeline_options,
+            common.LLVM_FUNC_ATTRS_KEY: waves_per_eu_dict,
+        }
+    )
```

**Comment:**
You can create a helper function to build config_dict for you, similar to the helper function `get_lowering_config`

---

**File:** `tuner/tuner/common.py:167`

```diff
@@ -157,15 +163,29 @@ def get_lowering_config(
     return iree_gpu.LoweringConfigAttr.get(lowering_config_attrs)
 
 
-def get_pipeline_config(configuration: Configuration) -> str:
-    extra_config = ""
-    pipeline_options = configuration.gpu_pipeline_options
-    if pipeline_options != iree_gpu.PipelineOptionsAttr.get():
-        extra_config += f", gpu_pipeline_options = {pipeline_options}"
-
-    if configuration.waves_per_eu != 2:
-        extra_config += f', llvm_func_attrs = {{"amdgpu-waves-per-eu" = "{configuration.waves_per_eu}"}}'
-    return extra_config
+def get_translation_info_config(
+    pipeline_options: iree_gpu.PipelineOptionsAttr, waves_per_eu: int | str
```

**Comment:**
I think it would be easier to always require `waves_per_eu` to be an int

---

**File:** `tuner/tuner/common.py:167`

```diff
@@ -157,15 +163,29 @@ def get_lowering_config(
     return iree_gpu.LoweringConfigAttr.get(lowering_config_attrs)
 
 
-def get_pipeline_config(configuration: Configuration) -> str:
-    extra_config = ""
-    pipeline_options = configuration.gpu_pipeline_options
-    if pipeline_options != iree_gpu.PipelineOptionsAttr.get():
-        extra_config += f", gpu_pipeline_options = {pipeline_options}"
-
-    if configuration.waves_per_eu != 2:
-        extra_config += f', llvm_func_attrs = {{"amdgpu-waves-per-eu" = "{configuration.waves_per_eu}"}}'
-    return extra_config
+def get_translation_info_config(
```

**Comment:**
Could you add a comment explaining what the purpose of this function is? Maybe show a piece of IR and where this attribute fits?

---


---


## [PR #662](https://github.com/nod-ai/amd-shark-ai/pull/662): [tuner]: use property function from iree lowering config python binding

### Review Summary

**APPROVED** (2024-12-09)

LGTM % formatting


### Code Comments

**File:** `tuner/tuner/dispatch_parser_test.py:83`

```diff
@@ -79,7 +80,7 @@ def test_get_conv_tile_sizes(tuner_ctx: common.TunerContext) -> None:
         gpu_pipeline_options=iree_gpu.PipelineOptionsAttr.get(),
         waves_per_eu=1,
     )
-    assert dispatch_parser.ConvParser().get_conv_workgroup_sizes(config) == [
+    assert config.lowering_config.workgroup_tile_sizes == [
```

**Comment:**
nit: Will these arrays fit on the same line now?

---


---


## [PR #629](https://github.com/nod-ai/amd-shark-ai/pull/629): [tuner]: use lowering config binding

### Review Summary

**CHANGES_REQUESTED** (2024-11-29)

**COMMENTED** (2024-11-29)

**COMMENTED** (2024-11-29)

**COMMENTED** (2024-12-01)

**COMMENTED** (2024-12-02)

**COMMENTED** (2024-12-02)

**COMMENTED** (2024-12-02)

**COMMENTED** (2024-12-03)

**COMMENTED** (2024-12-03)

**APPROVED** (2024-12-03)

LGTM. Thanks for all the fixes.


### Code Comments

**File:** `tuner/tuner/candidate_gen_test.py:57`

```diff
@@ -48,13 +48,25 @@ def test_apply_params_mmt(tuner_ctx: common.TunerContext) -> None:
 
     mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
     mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
+    lowering_config_dict = {
+        "mma_kind": mma_attr,
+        "workgroup": ir.ArrayAttr.get(
+            [
+                ir.IntegerAttr.get(tuner_ctx.type.i32, 8),
+                ir.IntegerAttr.get(tuner_ctx.type.i32, 8),
+                ir.IntegerAttr.get(tuner_ctx.type.i32, 8),
```

**Comment:**
We can add a helper functions to `tuner_ctx.type` like `tuner_ctx.type.getI32(8)` etc. This should save a lot of typing in code like this.

---

**File:** `tuner/tuner/candidate_gen_test.py:65`

```diff
@@ -48,13 +48,25 @@ def test_apply_params_mmt(tuner_ctx: common.TunerContext) -> None:
 
     mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
     mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
+    lowering_config_dict = {
+        "mma_kind": mma_attr,
+        "workgroup": ir.ArrayAttr.get(
+            [
+                ir.IntegerAttr.get(tuner_ctx.type.i32, 8),
+                ir.IntegerAttr.get(tuner_ctx.type.i32, 8),
+                ir.IntegerAttr.get(tuner_ctx.type.i32, 8),
+            ]
+        ),
+        "reduction": ir.ArrayAttr.get([]),
+        "subgroup_m_count": ir.IntegerAttr.get(tuner_ctx.type.i32, 16),
+        "subgroup_n_count": ir.IntegerAttr.get(tuner_ctx.type.i32, 16),
+    }
+
+    lowering_config_attrs = ir.DictAttr.get(lowering_config_dict)
```

**Comment:**
We can create a helper function that accepts python types and returns a lowering config attribute to avoid having to create these dictionaries by hand.

For example, this could be something like: `common.getLoweringConfig(mma_kind=mma_attr, workgroup=[8,8,0], reduction=[0, 0, 8], subgroup_m_count=16, subgroup_n_count=16)`. 

---

**File:** `tuner/tuner/candidate_gen_test.py:60`

```diff
@@ -48,13 +48,25 @@ def test_apply_params_mmt(tuner_ctx: common.TunerContext) -> None:
 
     mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
     mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
+    lowering_config_dict = {
+        "mma_kind": mma_attr,
+        "workgroup": ir.ArrayAttr.get(
+            [
+                ir.IntegerAttr.get(tuner_ctx.type.i32, 8),
+                ir.IntegerAttr.get(tuner_ctx.type.i32, 8),
+                ir.IntegerAttr.get(tuner_ctx.type.i32, 8),
+            ]
+        ),
+        "reduction": ir.ArrayAttr.get([]),
```

**Comment:**
The tile sizes are incorrect here: `[8, 8, 8]` in the single-array format should be `workgroup=[8, 8, 0], reduction = [0, 0, 8]`

---

**File:** `tuner/tuner/common.py:118`

```diff
@@ -105,26 +105,34 @@ def is_comptible(mma_intrinsic: iree_gpu.MMAIntrinsic) -> bool:
     return list(filter(is_comptible, mma_intrinsics))
 
 
-class ReorderWorkgroupsStrategy(Enum):
-    NONE = 0
-    SWIZZLE = 1
-    TRANSPOSE = 2
-
-    def __str__(self) -> str:
-        return self.name.title()
-
-
 @dataclass
 class Configuration:
     subgroup_size: int
     workgroup_size: list[int]
-    intrinsic: iree_gpu.MMAAttr
-    tile_sizes: list[int]
-    subgroup_m_count: int
-    subgroup_n_count: int
+    lowering_config: iree_gpu.LoweringConfigAttr
     gpu_pipeline_options: iree_gpu.PipelineOptionsAttr
     waves_per_eu: int
 
+    @property
+    def intrinsic(self) -> iree_gpu.MMAAttr:
+        return self.lowering_config.attributes["mma_kind"]
```

**Comment:**
I wouldn't make these properties because this can fail when the lowering config doesn't contain that attribute

---

**File:** `tuner/tuner/common.py:134`

```diff
@@ -105,26 +105,34 @@ def is_comptible(mma_intrinsic: iree_gpu.MMAIntrinsic) -> bool:
     return list(filter(is_comptible, mma_intrinsics))
 
 
-class ReorderWorkgroupsStrategy(Enum):
-    NONE = 0
-    SWIZZLE = 1
-    TRANSPOSE = 2
-
-    def __str__(self) -> str:
-        return self.name.title()
-
-
 @dataclass
 class Configuration:
     subgroup_size: int
     workgroup_size: list[int]
-    intrinsic: iree_gpu.MMAAttr
-    tile_sizes: list[int]
-    subgroup_m_count: int
-    subgroup_n_count: int
+    lowering_config: iree_gpu.LoweringConfigAttr
     gpu_pipeline_options: iree_gpu.PipelineOptionsAttr
     waves_per_eu: int
 
+    @property
+    def intrinsic(self) -> iree_gpu.MMAAttr:
+        return self.lowering_config.attributes["mma_kind"]
+
+    @property
+    def tilesize_workgroup(self) -> list[int]:
+        return [attr.value for attr in self.lowering_config.attributes["workgroup"]]
+
+    @property
+    def tilesize_reduction(self) -> list[int]:
+        return [attr.value for attr in self.lowering_config.attributes["reduction"]]
+
+    @property
+    def subgroup_m_count(self) -> int:
+        return self.lowering_config.attributes["subgroup_m_count"].value
+
+    @property
+    def subgroup_n_count(self) -> int:
+        return self.lowering_config.attributes["subgroup_n_count"].value
```

**Comment:**
Adding these is a good idea, but we should make sure we have tests that exercise this code

---

**File:** `tuner/tuner/candidate_gen_test.py:65`

```diff
@@ -48,13 +48,25 @@ def test_apply_params_mmt(tuner_ctx: common.TunerContext) -> None:
 
     mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
     mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
+    lowering_config_dict = {
+        "mma_kind": mma_attr,
+        "workgroup": ir.ArrayAttr.get(
+            [
+                ir.IntegerAttr.get(tuner_ctx.type.i32, 8),
+                ir.IntegerAttr.get(tuner_ctx.type.i32, 8),
+                ir.IntegerAttr.get(tuner_ctx.type.i32, 8),
+            ]
+        ),
+        "reduction": ir.ArrayAttr.get([]),
+        "subgroup_m_count": ir.IntegerAttr.get(tuner_ctx.type.i32, 16),
+        "subgroup_n_count": ir.IntegerAttr.get(tuner_ctx.type.i32, 16),
+    }
+
+    lowering_config_attrs = ir.DictAttr.get(lowering_config_dict)
```

**Comment:**
This could also be new method defined in python bindings, but I think it will be easier to define this in python as a new helper.

---

**File:** `tuner/tuner/candidate_gen.py:144`

```diff
@@ -127,12 +141,12 @@ def get_transform_function_mmt(
     transform.iree.match.cast_compatible_type %lhs = tensor<{problem_size.lhs_type}> : !transform.any_value
     transform.iree.match.cast_compatible_type %rhs = tensor<{problem_size.rhs_type}> : !transform.any_value
     %config = transform.param.constant #iree_codegen.compilation_info<
-        lowering_config = #iree_codegen.lowering_config<tile_sizes = [[{tile_sizes}]]>,
+        lowering_config = #iree_codegen.lowering_config<workgroup = [[{workgroup_sizes}]], reduction = [[{reduction_sizes}]]>,
```

**Comment:**
We should print the lowering config attribute here instead of extracting stuff out of it.

---

**File:** `tuner/tuner/candidate_gen_test.py:93`

```diff
@@ -84,7 +89,8 @@ def test_apply_params_mmt(tuner_ctx: common.TunerContext) -> None:
         "LLVMGPUVectorDistribute workgroup_size = [16, 16, 1] subgroup_size = 16"
         in modified
     )
-    assert "tile_sizes = [[8, 8, 8]]" in modified
+    assert "workgroup = [[8, 8, 0]]" in modified
+    assert "reduction = [[0, 0, 8]]" in modified
```

**Comment:**
I don't these attributes have two levels of nesting -- can you check with the attributes that come out of iree?

---

**File:** `tuner/tuner/common.py:155`

```diff
@@ -105,26 +108,80 @@ def is_comptible(mma_intrinsic: iree_gpu.MMAIntrinsic) -> bool:
     return list(filter(is_comptible, mma_intrinsics))
 
 
-class ReorderWorkgroupsStrategy(Enum):
-    NONE = 0
-    SWIZZLE = 1
-    TRANSPOSE = 2
-
-    def __str__(self) -> str:
-        return self.name.title()
-
-
 @dataclass
 class Configuration:
     subgroup_size: int
     workgroup_size: list[int]
-    intrinsic: iree_gpu.MMAAttr
-    tile_sizes: list[int]
-    subgroup_m_count: int
-    subgroup_n_count: int
+    lowering_config: iree_gpu.LoweringConfigAttr
     gpu_pipeline_options: iree_gpu.PipelineOptionsAttr
     waves_per_eu: int
 
+    def intrinsic(self) -> Optional[iree_gpu.MMAAttr]:
+        if self.lowering_config.attributes.__contains__("mma_kind"):
+            return self.lowering_config.attributes.__getitem__("mma_kind")
+        return None
+
+    def tilesize_workgroup(self) -> list[int]:
+        if self.lowering_config.attributes.__contains__("workgroup"):
+            workgroup_attrs = self.lowering_config.attributes.__getitem__("workgroup")
+            return [attr.value for attr in workgroup_attrs]
+        return []
+
+    def tilesize_reduction(self) -> list[int]:
+        if self.lowering_config.attributes.__contains__("reduction"):
+            reduction_attrs = self.lowering_config.attributes.__getitem__("reduction")
+            return [attr.value for attr in reduction_attrs]
+        return []
+
+    def subgroup_m_count(self) -> Optional[int]:
+        if self.lowering_config.attributes.__contains__("subgroup_m_count"):
+            attr = self.lowering_config.attributes.__getitem__("subgroup_m_count")
+            return attr.value
+        return None
+
+    def subgroup_n_count(self) -> Optional[int]:
+        if self.lowering_config.attributes.__contains__("subgroup_n_count"):
+            attr = self.lowering_config.attributes.__getitem__("subgroup_n_count")
+            return attr.value
+        return None
+
+
+def get_lowering_config(
+    tuner_ctx: TunerContext,
+    mma_attr: Optional[iree_gpu.MMAAttr] = None,
+    workgroup: Optional[list[int]] = None,
+    reduction: Optional[list[int]] = None,
+    subgroup_m_count: Optional[int] = None,
+    subgroup_n_count: Optional[int] = None,
```

**Comment:**
This list of attributes is not 'static' in the sense that some lowering configs can have a subset or a superset of these. Instead, can we handle this via kwargs (essentially a dictionary with a list of known attrs)?

---

**File:** `tuner/tuner/common.py:123`

```diff
@@ -105,26 +110,63 @@ def is_comptible(mma_intrinsic: iree_gpu.MMAIntrinsic) -> bool:
     return list(filter(is_comptible, mma_intrinsics))
 
 
-class ReorderWorkgroupsStrategy(Enum):
-    NONE = 0
-    SWIZZLE = 1
-    TRANSPOSE = 2
-
-    def __str__(self) -> str:
-        return self.name.title()
-
-
 @dataclass
 class Configuration:
     subgroup_size: int
     workgroup_size: list[int]
-    intrinsic: iree_gpu.MMAAttr
-    tile_sizes: list[int]
-    subgroup_m_count: int
-    subgroup_n_count: int
+    lowering_config: iree_gpu.LoweringConfigAttr
     gpu_pipeline_options: iree_gpu.PipelineOptionsAttr
     waves_per_eu: int
 
+    def intrinsic(self) -> Optional[iree_gpu.MMAAttr]:
+        if self.lowering_config.attributes.__contains__("mma_kind"):
+            return self.lowering_config.attributes.__getitem__("mma_kind")
```

**Comment:**
Don't use private methods (that start with an underscore). For Dictattr, I think you can use the `in` and `[...]` operators

---

**File:** `tuner/tuner/common.py:123`

```diff
@@ -105,26 +110,63 @@ def is_comptible(mma_intrinsic: iree_gpu.MMAIntrinsic) -> bool:
     return list(filter(is_comptible, mma_intrinsics))
 
 
-class ReorderWorkgroupsStrategy(Enum):
-    NONE = 0
-    SWIZZLE = 1
-    TRANSPOSE = 2
-
-    def __str__(self) -> str:
-        return self.name.title()
-
-
 @dataclass
 class Configuration:
     subgroup_size: int
     workgroup_size: list[int]
-    intrinsic: iree_gpu.MMAAttr
-    tile_sizes: list[int]
-    subgroup_m_count: int
-    subgroup_n_count: int
+    lowering_config: iree_gpu.LoweringConfigAttr
     gpu_pipeline_options: iree_gpu.PipelineOptionsAttr
     waves_per_eu: int
 
+    def intrinsic(self) -> Optional[iree_gpu.MMAAttr]:
+        if self.lowering_config.attributes.__contains__("mma_kind"):
+            return self.lowering_config.attributes.__getitem__("mma_kind")
```

**Comment:**
Also, I think this pattern would be supported as `my_dict.get(key, default)`, but I'm not sure if the python bindings support that

---

**File:** `tuner/tuner/common.py:165`

```diff
@@ -105,26 +110,63 @@ def is_comptible(mma_intrinsic: iree_gpu.MMAIntrinsic) -> bool:
     return list(filter(is_comptible, mma_intrinsics))
 
 
-class ReorderWorkgroupsStrategy(Enum):
-    NONE = 0
-    SWIZZLE = 1
-    TRANSPOSE = 2
-
-    def __str__(self) -> str:
-        return self.name.title()
-
-
 @dataclass
 class Configuration:
     subgroup_size: int
     workgroup_size: list[int]
-    intrinsic: iree_gpu.MMAAttr
-    tile_sizes: list[int]
-    subgroup_m_count: int
-    subgroup_n_count: int
+    lowering_config: iree_gpu.LoweringConfigAttr
     gpu_pipeline_options: iree_gpu.PipelineOptionsAttr
     waves_per_eu: int
 
+    def intrinsic(self) -> Optional[iree_gpu.MMAAttr]:
+        if self.lowering_config.attributes.__contains__("mma_kind"):
+            return self.lowering_config.attributes.__getitem__("mma_kind")
+        return None
+
+    def tilesize_workgroup(self) -> list[int]:
+        if self.lowering_config.attributes.__contains__("workgroup"):
+            workgroup_attrs = self.lowering_config.attributes.__getitem__("workgroup")
+            return [attr.value for attr in workgroup_attrs]
+        return []
+
+    def tilesize_reduction(self) -> list[int]:
+        if self.lowering_config.attributes.__contains__("reduction"):
+            reduction_attrs = self.lowering_config.attributes.__getitem__("reduction")
+            return [attr.value for attr in reduction_attrs]
+        return []
+
+    def subgroup_m_count(self) -> Optional[int]:
+        if self.lowering_config.attributes.__contains__("subgroup_m_count"):
+            attr = self.lowering_config.attributes.__getitem__("subgroup_m_count")
+            return attr.value
+        return None
+
+    def subgroup_n_count(self) -> Optional[int]:
+        if self.lowering_config.attributes.__contains__("subgroup_n_count"):
+            attr = self.lowering_config.attributes.__getitem__("subgroup_n_count")
+            return attr.value
+        return None
+
+
+def get_lowering_config(
+    tuner_ctx: TunerContext,
+    **kwargs: Any,
+) -> iree_gpu.LoweringConfigAttr:
+    lowering_config_dict = {}
+    for key, value in kwargs.items():
+        if isinstance(value, list):
+            lowering_config_dict[key] = ir.ArrayAttr.get(
+                [tuner_ctx.type.getI64(x) for x in value]
+            )
+        elif isinstance(value, int):
+            lowering_config_dict[key] = tuner_ctx.type.getI64(value)
+        elif isinstance(value, iree_gpu.MMAAttr):
+            lowering_config_dict[key] = value
+        else:
```

**Comment:**
I think it would be better to do the conversion based on known keys instead of just values. We can check if the passes value is an attribute (nothing to do then), or dispatch based on the type otherwise (I think you can use a `match` statement for that ([some examples](https://benhoyt.com/writings/python-pattern-matching/)).

---

**File:** `tuner/tuner/common.py:156`

```diff
@@ -105,26 +110,63 @@ def is_comptible(mma_intrinsic: iree_gpu.MMAIntrinsic) -> bool:
     return list(filter(is_comptible, mma_intrinsics))
 
 
-class ReorderWorkgroupsStrategy(Enum):
-    NONE = 0
-    SWIZZLE = 1
-    TRANSPOSE = 2
-
-    def __str__(self) -> str:
-        return self.name.title()
-
-
 @dataclass
 class Configuration:
     subgroup_size: int
     workgroup_size: list[int]
-    intrinsic: iree_gpu.MMAAttr
-    tile_sizes: list[int]
-    subgroup_m_count: int
-    subgroup_n_count: int
+    lowering_config: iree_gpu.LoweringConfigAttr
     gpu_pipeline_options: iree_gpu.PipelineOptionsAttr
     waves_per_eu: int
 
+    def intrinsic(self) -> Optional[iree_gpu.MMAAttr]:
+        if self.lowering_config.attributes.__contains__("mma_kind"):
+            return self.lowering_config.attributes.__getitem__("mma_kind")
+        return None
+
+    def tilesize_workgroup(self) -> list[int]:
+        if self.lowering_config.attributes.__contains__("workgroup"):
+            workgroup_attrs = self.lowering_config.attributes.__getitem__("workgroup")
+            return [attr.value for attr in workgroup_attrs]
+        return []
+
+    def tilesize_reduction(self) -> list[int]:
+        if self.lowering_config.attributes.__contains__("reduction"):
+            reduction_attrs = self.lowering_config.attributes.__getitem__("reduction")
+            return [attr.value for attr in reduction_attrs]
+        return []
+
+    def subgroup_m_count(self) -> Optional[int]:
+        if self.lowering_config.attributes.__contains__("subgroup_m_count"):
+            attr = self.lowering_config.attributes.__getitem__("subgroup_m_count")
+            return attr.value
+        return None
+
+    def subgroup_n_count(self) -> Optional[int]:
+        if self.lowering_config.attributes.__contains__("subgroup_n_count"):
+            attr = self.lowering_config.attributes.__getitem__("subgroup_n_count")
+            return attr.value
+        return None
+
+
+def get_lowering_config(
```

**Comment:**
Please add tests for this and make sure the returned attributes roundtrip with the helpers from `Configuration`

---

**File:** `tuner/tuner/dispatch_parser.py:174`

```diff
@@ -140,18 +164,21 @@ class ConvParser(DispatchParser):
     def supports(self, op_name: str) -> bool:
         return "conv_2d_nhwc_hwcf" in op_name
 
-    def get_conv_tile_sizes(self, configuration: Configuration) -> list[int]:
-        m, n, k = configuration.tile_sizes
+    def get_conv_workgroup_sizes(self, configuration: Configuration) -> list[int]:
         batch = 1
         fh = 1
         fw = 1
 
         oh = 1
 
-        oc = n
-        ow = m
-        ic = k
-        return [batch, oh, ow, oc, fh, fw, ic]
+        ow, oc, _ = configuration.tilesize_workgroup()
```

**Comment:**
```suggestion
        ow, oc, _ic = configuration.tilesize_workgroup()
```

---

**File:** `tuner/tuner/dispatch_constraints.py:251`

```diff
@@ -217,25 +219,46 @@ def generate_solutions(
     )
     solver.add(z3.simplify(z3.And(constraints)))
     logger.debug(f"Initial constraints: {solver}")
+
+    int_type = ir.IntegerType.get_signless(64)
+
     i = 0
     while solver.check() == z3.sat:
         model = solver.model()
         lookup = lambda var: model[var].as_long()
-
-        config = Configuration(
-            lookup(subgroup_size),
-            [lookup(wg_x), lookup(wg_y), lookup(wg_z)],
-            getMMAAttr(
+        lowering_config_dict = {
+            "mma_kind": getMMAAttr(
                 problem_size.res_type.element_type,
                 lookup(intrinsic_mn),
                 lookup(intrinsic_mn),
                 lookup(intrinsic_k),
                 problem_size.lhs_type.element_type,
                 problem_size.rhs_type.element_type,
             ),
-            [lookup(m), lookup(n), lookup(k)],
-            lookup(sg_m_cnt),
-            lookup(sg_n_cnt),
+            "workgroup": ir.ArrayAttr.get(
+                [
+                    ir.IntegerAttr.get(int_type, lookup(m)),
+                    ir.IntegerAttr.get(int_type, lookup(n)),
+                    ir.IntegerAttr.get(int_type, 0),
+                ]
+            ),
+            "reduction": ir.ArrayAttr.get(
+                [
+                    ir.IntegerAttr.get(int_type, 0),
+                    ir.IntegerAttr.get(int_type, 0),
+                    ir.IntegerAttr.get(int_type, lookup(k)),
+                ]
+            ),  # placeholder now to be consistent with iree
```

**Comment:**
Use proper casing and punctuation.

---

**File:** `tuner/tuner/dispatch_parser_test.py:130`

```diff
@@ -42,60 +42,93 @@ def test_parse_tensor_type(tuner_ctx: common.TunerContext) -> None:
 def test_get_mmt_tile_sizes(tuner_ctx: common.TunerContext) -> None:
     mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
     mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
+    lowering_config = common.get_lowering_config(
+        tuner_ctx=tuner_ctx,
+        mma_attr=mma_attr,
+        workgroup=[128, 320, 0],
+        reduction=[0, 0, 32],
+        subgroup_m_count=1,
+        subgroup_n_count=4,
+    )
     config = dispatch_parser.Configuration(
         subgroup_size=0,
         workgroup_size=[],
-        intrinsic=mma_attr,
-        tile_sizes=[128, 320, 32],
-        subgroup_m_count=0,
-        subgroup_n_count=0,
+        lowering_config=lowering_config,
         gpu_pipeline_options=iree_gpu.PipelineOptionsAttr.get(),
         waves_per_eu=0,
     )
-    assert dispatch_parser.get_mmt_tile_sizes(config) == [128, 320, 32]
+    assert dispatch_parser.get_mmt_workgroup_sizes(config) == [128, 320, 0]
+    assert dispatch_parser.get_mmt_reduction_sizes(config) == [0, 0, 32]
 
 
 def test_get_conv_tile_sizes(tuner_ctx: common.TunerContext) -> None:
     mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
     mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
+    lowering_config = common.get_lowering_config(
+        tuner_ctx=tuner_ctx,
+        mma_attr=mma_attr,
+        workgroup=[464, 320, 0],
+        reduction=[0, 0, 16],
+        subgroup_m_count=1,
+        subgroup_n_count=4,
+    )
     config = dispatch_parser.Configuration(
         subgroup_size=64,
         workgroup_size=[256, 1, 1],
-        intrinsic=mma_attr,
-        tile_sizes=[464, 320, 16],
-        subgroup_m_count=1,
-        subgroup_n_count=4,
+        lowering_config=lowering_config,
         gpu_pipeline_options=iree_gpu.PipelineOptionsAttr.get(),
         waves_per_eu=1,
     )
-    assert dispatch_parser.ConvParser().get_conv_tile_sizes(config) == [
+    assert dispatch_parser.ConvParser().get_conv_workgroup_sizes(config) == [
         1,
         1,
         464,
         320,
         1,
         1,
+        0,
+    ]
+    assert dispatch_parser.ConvParser().get_conv_reduction_sizes(config) == [
+        0,
+        0,
+        0,
+        0,
+        0,
+        0,
         16,
     ]
 
 
 def test_get_contract_tile_sizes(tuner_ctx: common.TunerContext) -> None:
     mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
     mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
+    lowering_config = common.get_lowering_config(
+        tuner_ctx=tuner_ctx,
+        mma_attr=mma_attr,
+        workgroup=[4, 8, 0],
+        reduction=[0, 0, 16],
+        subgroup_m_count=1,
+        subgroup_n_count=1,
+    )
     config = dispatch_parser.Configuration(
         subgroup_size=32,
         workgroup_size=[16, 16, 1],
-        intrinsic=mma_attr,
-        tile_sizes=[4, 8, 16],
-        subgroup_m_count=1,
-        subgroup_n_count=1,
+        lowering_config=lowering_config,
         gpu_pipeline_options=iree_gpu.PipelineOptionsAttr.get(),
         waves_per_eu=2,
     )
-    assert dispatch_parser.get_contract_tile_sizes(config, "mnk") == [4, 8, 16]
-    assert dispatch_parser.get_contract_tile_sizes(config, "nmk") == [8, 4, 16]
-    assert dispatch_parser.get_contract_tile_sizes(config, "knm") == [16, 8, 4]
-    assert dispatch_parser.get_contract_tile_sizes(config, "kkk") == [
+    assert dispatch_parser.get_contract_workgroup_sizes(config, "mnk") == [4, 8, 0]
+    assert dispatch_parser.get_contract_reduction_sizes(config, "mnk") == [0, 0, 16]
+    assert dispatch_parser.get_contract_workgroup_sizes(config, "nmk") == [8, 4, 0]
+    assert dispatch_parser.get_contract_reduction_sizes(config, "nmk") == [0, 0, 16]
+    assert dispatch_parser.get_contract_workgroup_sizes(config, "knm") == [0, 8, 4]
+    assert dispatch_parser.get_contract_reduction_sizes(config, "knm") == [16, 0, 0]
+    assert dispatch_parser.get_contract_workgroup_sizes(config, "kkk") == [
+        0,
+        0,
+        0,
+    ]
```

**Comment:**
This formatting looks weird -- I'd expect the `0, 0, 0` to fit on the same line since `16, 0, 0` is longer and fits

---

**File:** `tuner/tuner/common.py:121`

```diff
@@ -105,26 +110,63 @@ def is_comptible(mma_intrinsic: iree_gpu.MMAIntrinsic) -> bool:
     return list(filter(is_comptible, mma_intrinsics))
 
 
-class ReorderWorkgroupsStrategy(Enum):
-    NONE = 0
-    SWIZZLE = 1
-    TRANSPOSE = 2
-
-    def __str__(self) -> str:
-        return self.name.title()
-
-
 @dataclass
 class Configuration:
     subgroup_size: int
     workgroup_size: list[int]
-    intrinsic: iree_gpu.MMAAttr
-    tile_sizes: list[int]
-    subgroup_m_count: int
-    subgroup_n_count: int
+    lowering_config: iree_gpu.LoweringConfigAttr
     gpu_pipeline_options: iree_gpu.PipelineOptionsAttr
     waves_per_eu: int
 
+    def intrinsic(self) -> Optional[iree_gpu.MMAAttr]:
```

**Comment:**
If these helpers are useful enough, we could also add them to the iree python bindings. I think these helpers are also fine, but I'd rather see them as free functions instead of methods in `Configuration`. They don't need to know about `subgroup_size` and workgroup_size` after all -- the only input should the the lowering config attr.

---

**File:** `tuner/tuner/common.py:167`

```diff
@@ -154,16 +153,36 @@ def get_lowering_config(
 ) -> iree_gpu.LoweringConfigAttr:
     lowering_config_dict = {}
     for key, value in kwargs.items():
-        if isinstance(value, list):
-            lowering_config_dict[key] = ir.ArrayAttr.get(
-                [tuner_ctx.type.getI64(x) for x in value]
-            )
-        elif isinstance(value, int):
-            lowering_config_dict[key] = tuner_ctx.type.getI64(value)
-        elif isinstance(value, iree_gpu.MMAAttr):
-            lowering_config_dict[key] = value
-        else:
-            raise TypeError(f"Unsupported type for key '{key}': {type(value).__name__}")
+        match key:
+            case "workgroup" | "reduction":
+                if isinstance(value, list):
+                    lowering_config_dict[key] = ir.ArrayAttr.get(
+                        [tuner_ctx.type.getI64(x) for x in value]
+                    )
+                elif isinstance(value, ir.ArrayAttr):
+                    lowering_config_dict[key] = value
+                else:
+                    raise TypeError(
+                        f"Unsupported type for key '{key}': {type(value).__name__}"
+                    )
```

**Comment:**
IMO we can assert instead -- this is internal logic

---

**File:** `tuner/tuner/common.py:165`

```diff
@@ -154,16 +153,36 @@ def get_lowering_config(
 ) -> iree_gpu.LoweringConfigAttr:
     lowering_config_dict = {}
     for key, value in kwargs.items():
-        if isinstance(value, list):
-            lowering_config_dict[key] = ir.ArrayAttr.get(
-                [tuner_ctx.type.getI64(x) for x in value]
-            )
-        elif isinstance(value, int):
-            lowering_config_dict[key] = tuner_ctx.type.getI64(value)
-        elif isinstance(value, iree_gpu.MMAAttr):
-            lowering_config_dict[key] = value
-        else:
-            raise TypeError(f"Unsupported type for key '{key}': {type(value).__name__}")
+        match key:
+            case "workgroup" | "reduction":
```

**Comment:**
I'd make a single assignment to `lowering_config[key]` just after the `match` and introduce a new local variable `promoted_value = value` that gets modified inside the match

---

**File:** `tuner/tuner/common.py:126`

```diff
@@ -105,26 +110,83 @@ def is_comptible(mma_intrinsic: iree_gpu.MMAIntrinsic) -> bool:
     return list(filter(is_comptible, mma_intrinsics))
 
 
-class ReorderWorkgroupsStrategy(Enum):
-    NONE = 0
-    SWIZZLE = 1
-    TRANSPOSE = 2
-
-    def __str__(self) -> str:
-        return self.name.title()
-
-
 @dataclass
 class Configuration:
     subgroup_size: int
     workgroup_size: list[int]
-    intrinsic: iree_gpu.MMAAttr
-    tile_sizes: list[int]
-    subgroup_m_count: int
-    subgroup_n_count: int
+    lowering_config: iree_gpu.LoweringConfigAttr
     gpu_pipeline_options: iree_gpu.PipelineOptionsAttr
     waves_per_eu: int
 
+    def intrinsic(self) -> Optional[iree_gpu.MMAAttr]:
+        if "mma_kind" in self.lowering_config.attributes:
+            return self.lowering_config.attributes["mma_kind"]
+        return None
+
+    def tilesize_workgroup(self) -> list[int]:
```

**Comment:**
There shouldn't be methods of `Configuration`. We can have them as free functions instead.

---

**File:** `tuner/tuner/candidate_gen.py:65`

```diff
@@ -40,36 +40,46 @@
 
 
 def apply_configuration(
-    template: list[str], configuration: Configuration, tile_sizes: list[int]
+    template: list[str],
+    configuration: Configuration,
+    workgroup_sizes: list[int],
+    reduction_sizes: list[int],
 ) -> str:
+    intrinsic = get_intrinsic(configuration)
+    subgroup_m_count = get_subgroup_m_count(configuration)
+    subgroup_n_count = get_subgroup_n_count(configuration)
     tune_logger.info(f"Applying: {configuration}")
     expr0 = re.compile(
         r"<intrinsic = #iree_gpu\.mma_layout<(.+)>, subgroup_m_count = ([0-9]+), subgroup_n_count = ([0-9]+)>"
     )
     expr1 = re.compile(
         r"LLVMGPUVectorDistribute workgroup_size = \[.+\] subgroup_size = ([0-9]+),"
     )
-    expr2 = re.compile(r"tile_sizes = \[\[([0-9]+)(, ([0-9]+))+\]\]")
-    expr3 = re.compile(r"gpu_pipeline_options = #iree_gpu\.pipeline_options<([^>]*)>")
-    expr4 = re.compile(r"\"amdgpu-waves-per-eu\" = \"([0-9])\"")
-    repl0 = f"<intrinsic = {configuration.intrinsic}, subgroup_m_count = {configuration.subgroup_m_count}, subgroup_n_count = {configuration.subgroup_n_count}>"
+    expr2 = re.compile(r"workgroup = \[([0-9]+)(, ([0-9]+))+\]")
+    expr3 = re.compile(r"reduction = \[([0-9]+)(, ([0-9]+))+\]")
+    expr4 = re.compile(r"gpu_pipeline_options = #iree_gpu\.pipeline_options<([^>]*)>")
+    expr5 = re.compile(r"\"amdgpu-waves-per-eu\" = \"([0-9])\"")
+    repl0 = f"<intrinsic = {intrinsic}, subgroup_m_count = {subgroup_m_count}, subgroup_n_count = {subgroup_n_count}>"
     repl1 = f'LLVMGPUVectorDistribute workgroup_size = [{", ".join(map(str, configuration.workgroup_size))}] subgroup_size = {configuration.subgroup_size},'
-    repl2 = f'tile_sizes = [[{", ".join(map(str, tile_sizes))}]]'
-    repl3 = f"gpu_pipeline_options = {configuration.gpu_pipeline_options}"
-    repl4 = f'"amdgpu-waves-per-eu" = "{configuration.waves_per_eu}"'
+    repl2 = f'workgroup = [{", ".join(map(str, workgroup_sizes))}]'
+    repl3 = f'reduction = [{", ".join(map(str, reduction_sizes))}]'
```

**Comment:**
Can we print the attributes directly? `config.lowering_config.attributes["workgroup"]`

---

**File:** `tuner/tuner/common.py:128`

```diff
@@ -105,27 +110,84 @@ def is_comptible(mma_intrinsic: iree_gpu.MMAIntrinsic) -> bool:
     return list(filter(is_comptible, mma_intrinsics))
 
 
-class ReorderWorkgroupsStrategy(Enum):
-    NONE = 0
-    SWIZZLE = 1
-    TRANSPOSE = 2
-
-    def __str__(self) -> str:
-        return self.name.title()
-
-
 @dataclass
 class Configuration:
     subgroup_size: int
     workgroup_size: list[int]
-    intrinsic: iree_gpu.MMAAttr
-    tile_sizes: list[int]
-    subgroup_m_count: int
-    subgroup_n_count: int
+    lowering_config: iree_gpu.LoweringConfigAttr
     gpu_pipeline_options: iree_gpu.PipelineOptionsAttr
     waves_per_eu: int
 
 
+def get_intrinsic(config: Configuration) -> Optional[iree_gpu.MMAAttr]:
+    if "mma_kind" in config.lowering_config.attributes:
+        return config.lowering_config.attributes["mma_kind"]
+    return None
+
+
+def get_tilesize_workgroup(config: Configuration) -> list[int]:
```

**Comment:**
```suggestion
def get_workgroup_tile_sizes(config: Configuration) -> list[int]:
```

---

**File:** `tuner/tuner/common.py:135`

```diff
@@ -105,27 +110,84 @@ def is_comptible(mma_intrinsic: iree_gpu.MMAIntrinsic) -> bool:
     return list(filter(is_comptible, mma_intrinsics))
 
 
-class ReorderWorkgroupsStrategy(Enum):
-    NONE = 0
-    SWIZZLE = 1
-    TRANSPOSE = 2
-
-    def __str__(self) -> str:
-        return self.name.title()
-
-
 @dataclass
 class Configuration:
     subgroup_size: int
     workgroup_size: list[int]
-    intrinsic: iree_gpu.MMAAttr
-    tile_sizes: list[int]
-    subgroup_m_count: int
-    subgroup_n_count: int
+    lowering_config: iree_gpu.LoweringConfigAttr
     gpu_pipeline_options: iree_gpu.PipelineOptionsAttr
     waves_per_eu: int
 
 
+def get_intrinsic(config: Configuration) -> Optional[iree_gpu.MMAAttr]:
+    if "mma_kind" in config.lowering_config.attributes:
+        return config.lowering_config.attributes["mma_kind"]
+    return None
+
+
+def get_tilesize_workgroup(config: Configuration) -> list[int]:
+    if "workgroup" in config.lowering_config.attributes:
+        workgroup_attrs = config.lowering_config.attributes["workgroup"]
+        return [attr.value for attr in workgroup_attrs]
+    return []
+
+
+def get_tilesize_reduction(config: Configuration) -> list[int]:
```

**Comment:**
```suggestion
def get_reduction_tile_sizes(config: Configuration) -> list[int]:
```

---

**File:** `tuner/tuner/common.py:185`

```diff
@@ -105,27 +110,84 @@ def is_comptible(mma_intrinsic: iree_gpu.MMAIntrinsic) -> bool:
     return list(filter(is_comptible, mma_intrinsics))
 
 
-class ReorderWorkgroupsStrategy(Enum):
-    NONE = 0
-    SWIZZLE = 1
-    TRANSPOSE = 2
-
-    def __str__(self) -> str:
-        return self.name.title()
-
-
 @dataclass
 class Configuration:
     subgroup_size: int
     workgroup_size: list[int]
-    intrinsic: iree_gpu.MMAAttr
-    tile_sizes: list[int]
-    subgroup_m_count: int
-    subgroup_n_count: int
+    lowering_config: iree_gpu.LoweringConfigAttr
     gpu_pipeline_options: iree_gpu.PipelineOptionsAttr
     waves_per_eu: int
 
 
+def get_intrinsic(config: Configuration) -> Optional[iree_gpu.MMAAttr]:
+    if "mma_kind" in config.lowering_config.attributes:
+        return config.lowering_config.attributes["mma_kind"]
+    return None
+
+
+def get_tilesize_workgroup(config: Configuration) -> list[int]:
+    if "workgroup" in config.lowering_config.attributes:
+        workgroup_attrs = config.lowering_config.attributes["workgroup"]
+        return [attr.value for attr in workgroup_attrs]
+    return []
+
+
+def get_tilesize_reduction(config: Configuration) -> list[int]:
+    if "reduction" in config.lowering_config.attributes:
+        reduction_attrs = config.lowering_config.attributes["reduction"]
+        return [attr.value for attr in reduction_attrs]
+    return []
+
+
+def get_subgroup_m_count(config: Configuration) -> Optional[int]:
+    if "subgroup_m_count" in config.lowering_config.attributes:
+        attr = config.lowering_config.attributes["subgroup_m_count"]
+        return attr.value
+    return None
+
+
+def get_subgroup_n_count(config: Configuration) -> Optional[int]:
+    if "subgroup_n_count" in config.lowering_config.attributes:
+        attr = config.lowering_config.attributes["subgroup_n_count"]
+        return attr.value
+    return None
+
+
+def get_lowering_config(
+    tuner_ctx: TunerContext,
+    **kwargs: Any,
+) -> iree_gpu.LoweringConfigAttr:
+    lowering_config_dict: dict[str, Any] = {}
+    for key, value in kwargs.items():
+        # A local variable to hold the transformed value.
+        promoted_value = value
+        match key:
+            case "workgroup" | "reduction":
+                assert isinstance(
+                    value, (list, ir.ArrayAttr)
+                ), f"Unsupported type for key '{key}': {type(value).__name__}"
+                if isinstance(value, list):
+                    promoted_value = ir.ArrayAttr.get(
+                        [tuner_ctx.type.getI64(x) for x in value]
+                    )
+            case "subgroup_m_count" | "subgroup_n_count":
+                assert isinstance(
+                    value, (int, tuner_ctx.type.i64)
+                ), f"Unsupported type for key '{key}': {type(value).__name__}"
+                if isinstance(value, int):
+                    promoted_value = tuner_ctx.type.getI64(value)
+            case "mma_kind":
+                assert isinstance(
+                    value, iree_gpu.MMAAttr
+                ), f"Unsupported type for key '{key}': {type(value).__name__}"
+            case _:
+                raise KeyError(f"Unhandled key in lowering configuration: {key}")
+        # Single assignment after the match.
```

**Comment:**
This is pretty obvious, I don't think we need this comment but could use an empty line

---

**File:** `tuner/tuner/common.py:168`

```diff
@@ -105,27 +110,84 @@ def is_comptible(mma_intrinsic: iree_gpu.MMAIntrinsic) -> bool:
     return list(filter(is_comptible, mma_intrinsics))
 
 
-class ReorderWorkgroupsStrategy(Enum):
-    NONE = 0
-    SWIZZLE = 1
-    TRANSPOSE = 2
-
-    def __str__(self) -> str:
-        return self.name.title()
-
-
 @dataclass
 class Configuration:
     subgroup_size: int
     workgroup_size: list[int]
-    intrinsic: iree_gpu.MMAAttr
-    tile_sizes: list[int]
-    subgroup_m_count: int
-    subgroup_n_count: int
+    lowering_config: iree_gpu.LoweringConfigAttr
     gpu_pipeline_options: iree_gpu.PipelineOptionsAttr
     waves_per_eu: int
 
 
+def get_intrinsic(config: Configuration) -> Optional[iree_gpu.MMAAttr]:
+    if "mma_kind" in config.lowering_config.attributes:
+        return config.lowering_config.attributes["mma_kind"]
+    return None
+
+
+def get_tilesize_workgroup(config: Configuration) -> list[int]:
+    if "workgroup" in config.lowering_config.attributes:
+        workgroup_attrs = config.lowering_config.attributes["workgroup"]
+        return [attr.value for attr in workgroup_attrs]
+    return []
+
+
+def get_tilesize_reduction(config: Configuration) -> list[int]:
+    if "reduction" in config.lowering_config.attributes:
+        reduction_attrs = config.lowering_config.attributes["reduction"]
+        return [attr.value for attr in reduction_attrs]
+    return []
+
+
+def get_subgroup_m_count(config: Configuration) -> Optional[int]:
+    if "subgroup_m_count" in config.lowering_config.attributes:
+        attr = config.lowering_config.attributes["subgroup_m_count"]
+        return attr.value
+    return None
+
+
+def get_subgroup_n_count(config: Configuration) -> Optional[int]:
+    if "subgroup_n_count" in config.lowering_config.attributes:
+        attr = config.lowering_config.attributes["subgroup_n_count"]
+        return attr.value
+    return None
+
+
+def get_lowering_config(
+    tuner_ctx: TunerContext,
+    **kwargs: Any,
+) -> iree_gpu.LoweringConfigAttr:
+    lowering_config_dict: dict[str, Any] = {}
+    for key, value in kwargs.items():
+        # A local variable to hold the transformed value.
+        promoted_value = value
+        match key:
+            case "workgroup" | "reduction":
+                assert isinstance(
+                    value, (list, ir.ArrayAttr)
+                ), f"Unsupported type for key '{key}': {type(value).__name__}"
```

**Comment:**
Don't check the type in the assert -- if the previous if doesn't match, `assert False`

---

**File:** `tuner/tuner/candidate_gen.py:46`

```diff
@@ -40,36 +40,46 @@
 
 
 def apply_configuration(
-    template: list[str], configuration: Configuration, tile_sizes: list[int]
+    template: list[str],
+    configuration: Configuration,
+    workgroup_sizes: list[int],
+    reduction_sizes: list[int],
```

**Comment:**
Could we extract workgroup and tile sizes from the `configuration` directly and remove these two function arguments?

---


---


## [PR #626](https://github.com/nod-ai/amd-shark-ai/pull/626): [tuner]: retire data class GPUPipelineOptions, use iree_gpu.PipelineOptionsAttr.

### Review Summary

**COMMENTED** (2024-11-27)

Looks good, just one issue

**APPROVED** (2024-11-28)

LGTM


### Code Comments

**File:** `tuner/tuner/common.py:137`

```diff
@@ -151,14 +122,21 @@ class Configuration:
     tile_sizes: list[int]
     subgroup_m_count: int
     subgroup_n_count: int
-    gpu_pipeline_options: GpuPipelineOptions
+    gpu_pipeline_options: iree_gpu.PipelineOptionsAttr
     waves_per_eu: int
 
 
 def get_pipeline_config(configuration: Configuration) -> str:
     extra_config = ""
-    if not configuration.gpu_pipeline_options.all_default():
-        extra_config += f", gpu_pipeline_options = {configuration.gpu_pipeline_options}"
+    pipeline_options = configuration.gpu_pipeline_options
+    if (
+        pipeline_options.prefetch_shared_memory is not None
+        or pipeline_options.no_reduce_shared_memory_bank_conflicts is not None
+        or pipeline_options.use_igemm_convolution is not None
+        or pipeline_options.reorder_workgroups_strategy is not None
+    ):
```

**Comment:**
You can compare `pipeline_options` with a freshly constructed attribute with all default values

---


---


## [PR #605](https://github.com/nod-ai/amd-shark-ai/pull/605): [tuner]: use iree_gpu.MMAIntrinsic and iree_gpu.MMAAttr

### Review Summary

**CHANGES_REQUESTED** (2024-11-25)

In the PR description, could you include the motivation behind this change?

**COMMENTED** (2024-11-26)

Looks good, just a couple of remaining issues

**COMMENTED** (2024-11-26)

**APPROVED** (2024-11-26)

LGTM % one suggestion. Thanks for all the fixes.


### Code Comments

**File:** `tuner/tuner/candidate_gen_test.py:49`

```diff
@@ -45,10 +46,12 @@ def test_apply_params_mmt(tuner_ctx: common.TunerContext) -> None:
 
     M, N, K = 2048, 1280, 1280
 
+    mma_intrinsic = getattr(iree_gpu.MMAIntrinsic, f"MFMA_F32_16x16x16_F16")
```

**Comment:**
Use the enum instead of passing a string. Also everywhere else

---

**File:** `tuner/tuner/common.py:93`

```diff
@@ -85,74 +85,24 @@ def MNK(self) -> tuple[int, int, int]:
         return (self.matmul_size.M, self.matmul_size.N, self.matmul_size.K)
 
 
-@dataclass
-class MfmaIntrinsic:
-    output_type: ir.IntegerType | ir.FloatType
-    m: int
-    n: int
-    k: int
-    input_type: ir.IntegerType | ir.FloatType
-
-    def __str__(self) -> str:
-        input = str(self.input_type).upper()
-        output = str(self.output_type).upper()
-        return f"MFMA_{output}_{self.m}x{self.n}x{self.k}_{input}"
-
-    @staticmethod
-    def mfma_f32_16x16x16_f16():
-        f16 = ir.F16Type.get()
-        f32 = ir.F32Type.get()
-        return MfmaIntrinsic(f32, 16, 16, 16, f16)
-
-    @staticmethod
-    def mfma_f32_32x32x8_f16():
-        f16 = ir.F16Type.get()
-        f32 = ir.F32Type.get()
-        return MfmaIntrinsic(f32, 32, 32, 8, f16)
-
-    @staticmethod
-    def mfma_i32_16x16x32_i8():
-        i32 = ir.IntegerType.get_signless(32)
-        i8 = ir.IntegerType.get_signless(8)
-        return MfmaIntrinsic(i32, 16, 16, 32, i8)
-
-    @staticmethod
-    def mfma_i32_32x32x16_i8():
-        i32 = ir.IntegerType.get_signless(32)
-        i8 = ir.IntegerType.get_signless(8)
-        return MfmaIntrinsic(i32, 32, 32, 16, i8)
-
-    @staticmethod
-    def all():
-        return [
-            MfmaIntrinsic.mfma_f32_16x16x16_f16(),
-            MfmaIntrinsic.mfma_f32_32x32x8_f16(),
-            MfmaIntrinsic.mfma_i32_16x16x32_i8(),
-            MfmaIntrinsic.mfma_i32_32x32x16_i8(),
-        ]
-
-
 def get_compatible_mfma_intrinsics(
     problem_size: ProblemSize,
     mma_intrinsics: list[iree_gpu.MMAIntrinsic],
-) -> list[MfmaIntrinsic]:
-    available_mma_intrinsics = [str(mma) for mma in mma_intrinsics]
-
-    def is_compatible(intrinsic: MfmaIntrinsic) -> bool:
-        if problem_size.res_type.element_type != intrinsic.output_type:
+) -> list[iree_gpu.MMAIntrinsic]:
+    def is_comptible(mma_intrinsic: iree_gpu.MMAIntrinsic) -> bool:
+        mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
```

**Comment:**
you can do `mma_intrinsic.mma`

---

**File:** `tuner/tuner/common_test.py:8`

```diff
@@ -5,7 +5,7 @@
 # SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 
 """
-Usage: python -m pytest candidate_gen_test.py
+Usage: python -m pytest common_test.py
```

**Comment:**
Good catch!

---

**File:** `tuner/tuner/dispatch_constraints.py:33`

```diff
@@ -27,8 +27,14 @@ def get_mfma_intrinsic_constraints(
     assert len(compatible_intrinsics) > 0, "No compatible intrinsics found"
     return z3.Or(
         *(
-            z3.And(intrinsic_m == mfma.m, intrinsic_n == mfma.n, intrinsic_k == mfma.k)
-            for mfma in compatible_intrinsics
+            z3.And(
+                intrinsic_m == mma_attr.mnk_shape[0],
+                intrinsic_n == mma_attr.mnk_shape[1],
+                intrinsic_k == mma_attr.mnk_shape[2],
```

**Comment:**
Let's try to avoid repeated CAPI calls here -- we can hoist mnk into a local variable so that we only call this once

---

**File:** `tuner/tuner/dispatch_constraints.py:162`

```diff
@@ -134,6 +140,30 @@ def generate_constraints(
     return constraints
 
 
+def getMMAAttr(
+    output_type: ir.IntegerType | ir.FloatType,
+    m: int,
+    n: int,
+    k: int,
+    lhs_type: ir.IntegerType | ir.FloatType,
+    rhs_type: ir.IntegerType | ir.FloatType,
+) -> iree_gpu.MMAAttr:
+    mma_str = ""
+    if lhs_type == rhs_type:
+        input = str(lhs_type).upper()
+        output = str(output_type).upper()
+        mma_str = f"MFMA_{output}_{m}x{n}x{k}_{input}"
+    else:
+        lhs = str(lhs_type).upper()
+        rhs = str(rhs_type).upper()
+        output = str(output_type).upper()
+        mma_str = f"MFMA_{output}_{m}x{n}x{k}_{lhs}_{rhs}"
+
+    mma_intrinsic = getattr(iree_gpu.MMAIntrinsic, mma_str)
```

**Comment:**
Let's try to avoid text-based processing here and instead enumerate all available intrinsics and select the enum that matches these parameters

---

**File:** `tuner/tuner/dispatch_constraints.py:38`

```diff
@@ -27,8 +27,15 @@ def get_mfma_intrinsic_constraints(
     assert len(compatible_intrinsics) > 0, "No compatible intrinsics found"
     return z3.Or(
         *(
-            z3.And(intrinsic_m == mfma.m, intrinsic_n == mfma.n, intrinsic_k == mfma.k)
-            for mfma in compatible_intrinsics
+            z3.And(
+                intrinsic_m == mnk[0],
+                intrinsic_n == mnk[1],
+                intrinsic_k == mnk[2],
+            )
+            for mma_attr in (
+                iree_gpu.MMAAttr.get(mfma) for mfma in compatible_intrinsics
+            )
+            for mnk in [mma_attr.mnk_shape]
```

**Comment:**
Instead of a for loop, can we make it a local variable? Double for loop with just one iteration is confusing to me -- it took me a good minute to figure out what we are trying to do here.

---

**File:** `tuner/tuner/dispatch_constraints.py:153`

```diff
@@ -134,6 +141,35 @@ def generate_constraints(
     return constraints
 
 
+def getMMAAttr(
+    output_type: ir.IntegerType | ir.FloatType,
+    m: int,
+    n: int,
+    k: int,
+    lhs_type: ir.IntegerType | ir.FloatType,
+    rhs_type: ir.IntegerType | ir.FloatType,
+) -> iree_gpu.MMAAttr:
+    for mma_intrinsic in iree_gpu.MMAIntrinsic:
+        mma_attr = iree_gpu.MMAIntrinsicAttr.get(mma_intrinsic).mma
```

**Comment:**
Here we can construct mma directly from the intrinsic without creating an mmaintrinsicattr first

---

**File:** `tuner/tuner/dispatch_constraints.py:42`

```diff
@@ -25,10 +25,21 @@ def get_mfma_intrinsic_constraints(
 ) -> z3.BoolRef:
     compatible_intrinsics = get_compatible_mfma_intrinsics(problem_size, mma_intrinsics)
     assert len(compatible_intrinsics) > 0, "No compatible intrinsics found"
+
+    mma_attrs = [iree_gpu.MMAAttr.get(mfma) for mfma in compatible_intrinsics]
+    mnk_shapes = [mma_attr.mnk_shape for mma_attr in mma_attrs]
+
     return z3.Or(
         *(
-            z3.And(intrinsic_m == mfma.m, intrinsic_n == mfma.n, intrinsic_k == mfma.k)
-            for mfma in compatible_intrinsics
+            z3.And(
+                intrinsic_m == mnk[0],
+                intrinsic_n == mnk[1],
+                intrinsic_k == mnk[2],
+            )
+            for mma_attr in (
+                iree_gpu.MMAAttr.get(mfma) for mfma in compatible_intrinsics
+            )
+            for mnk in mnk_shapes
```

**Comment:**
I'm still confused by this loop structure. Why do we need two `for`s?

---

**File:** `tuner/tuner/dispatch_constraints.py:39`

```diff
@@ -25,10 +25,18 @@ def get_mfma_intrinsic_constraints(
 ) -> z3.BoolRef:
     compatible_intrinsics = get_compatible_mfma_intrinsics(problem_size, mma_intrinsics)
     assert len(compatible_intrinsics) > 0, "No compatible intrinsics found"
+
+    mma_attrs = [iree_gpu.MMAAttr.get(mfma) for mfma in compatible_intrinsics]
+    mnk_shapes = [mma_attr.mnk_shape for mma_attr in mma_attrs]
+
     return z3.Or(
         *(
-            z3.And(intrinsic_m == mfma.m, intrinsic_n == mfma.n, intrinsic_k == mfma.k)
-            for mfma in compatible_intrinsics
+            z3.And(
+                intrinsic_m == mnk[0],
+                intrinsic_n == mnk[1],
+                intrinsic_k == mnk[2],
+            )
+            for mnk in mnk_shapes
```

**Comment:**
```suggestion
                intrinsic_m == m,
                intrinsic_n == n,
                intrinsic_k == k,
            )
            for m, n, k in mnk_shapes
```

---


---


## [PR #586](https://github.com/nod-ai/amd-shark-ai/pull/586): [tuner]: use python binding to select mma intrinsics

### Review Summary

**CHANGES_REQUESTED** (2024-11-22)

The CI doesn't pass -- seems like the code doesn't type check

**COMMENTED** (2024-11-22)

**COMMENTED** (2024-11-22)

We should add a test that shows that filtering based on available intrinsics works in `get_compatible_mfma_intrinsics`

**APPROVED** (2024-11-22)


### Code Comments

**File:** `tuner/tuner/candidate_gen.py:541`

```diff
@@ -535,13 +537,19 @@ def tune(
 
         walk_result: OpWalkResult = walk_mlir_op(mlir_module, dispatch_tuner_registry)
 
+        variant_op_list = iree_codegen.get_executable_variant_ops(mlir_module)
+        assert len(variant_op_list) == 1, "Support only one op in one disptach"
```

**Comment:**
This assertion message doesn't match the code. I'd say something like `Expected one executable variant op`. Saying `one op` only is misleading.

---

**File:** `tuner/tuner/common.py:137`

```diff
@@ -130,7 +132,12 @@ def all():
         ]
 
 
-def get_compatible_mfma_intrinsics(problem_size: ProblemSize) -> list[MfmaIntrinsic]:
+def get_compatible_mfma_intrinsics(
+    problem_size: ProblemSize,
+    mma_intrinsics: Optional[list[iree_gpu.MMAIntrinsic]] = None,
```

**Comment:**
Why is this optional? I think we can make it a required argument.

---

**File:** `tuner/tuner/dispatch_constraints.py:25`

```diff
@@ -18,8 +22,9 @@ def get_mfma_intrinsic_constraints(
     intrinsic_m: z3.ArithRef,
     intrinsic_n: z3.ArithRef,
     intrinsic_k: z3.ArithRef,
+    mma_intrinsics: Optional[list[iree_gpu.MMAIntrinsic]] = None,
```

**Comment:**
Also here

---

**File:** `tuner/tuner/dispatch_constraints.py:142`

```diff
@@ -130,7 +136,10 @@ def generate_constraints(
 
 
 def generate_solutions(
-    logger: logging.Logger, problem_size: ProblemSize, num_subgrups: int
+    logger: logging.Logger,
+    problem_size: ProblemSize,
+    num_subgrups: int,
+    mma_intrinsics: Optional[list[iree_gpu.MMAIntrinsic]] = None,
```

**Comment:**
also here

---

**File:** `tuner/tuner/common.py:139`

```diff
@@ -130,7 +132,12 @@ def all():
         ]
 
 
-def get_compatible_mfma_intrinsics(problem_size: ProblemSize) -> list[MfmaIntrinsic]:
+def get_compatible_mfma_intrinsics(
+    problem_size: ProblemSize,
+    mma_intrinsics: Optional[list[iree_gpu.MMAIntrinsic]] = None,
+) -> list[MfmaIntrinsic]:
+    mma_list_target = {str(mma) for mma in mma_intrinsics} if mma_intrinsics else None
```

**Comment:**
This is not the best variable name. Consider changing it to something like `available_mma_intrinsics`

---

**File:** `tuner/tuner/common.py:150`

```diff
@@ -139,6 +146,10 @@ def is_compatible(intrinsic: MfmaIntrinsic) -> bool:
                 return False
             if problem_size.rhs_type.element_type != intrinsic.input_type:
                 return False
+
+        if available_mma_intrinsics and str(intrinsic) not in available_mma_intrinsics:
```

**Comment:**
I don't think we need to check that `available_mma_intrinsics` is non-empty?

---

**File:** `tuner/tuner/common_test.py:113`

```diff
@@ -109,7 +109,8 @@ def test_get_compatible_mfma_intrinsics(tuner_ctx: common.TunerContext) -> None:
             common.ShapedType([1280, 1280], tuner_ctx.type.f16),
             common.ShapedType([2048, 1280], tuner_ctx.type.f32),
             common.DispatchKind.mmt,
-        )
+        ),
+        [],
```

**Comment:**
We should make sure the new logic is covered by tests. Please add/modify existing tests to pass in the list of available intrinsics.

---

**File:** `tuner/tuner/dispatch_constraints_test.py:42`

```diff
@@ -37,7 +38,9 @@ def test_generate_solutions(tuner_ctx: common.TunerContext) -> None:
     problem_size = common.ProblemSize(
         matmul_size, lhs_type, rhs_type, res_type, common.DispatchKind.mmt
     )
-    configs = dispatch_constraints.generate_solutions(tuner_ctx.logger, problem_size, 4)
+    configs = dispatch_constraints.generate_solutions(
+        tuner_ctx.logger, problem_size, 4, []
```

**Comment:**
Doesn't this change the reason for no solutions? Now it can't produce any because there are no MFMAs to choose from.

---


---
