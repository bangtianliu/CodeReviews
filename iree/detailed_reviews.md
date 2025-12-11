# Detailed Code Reviews

**Total PRs: 74**

**Generated:** 2025-12-10 18:49:41

---


## [PR #22864](https://github.com/iree-org/iree/pull/22864): Integrates/llvm 20251209

### Review Summary

**APPROVED** (2025-12-09)



---


## [PR #22727](https://github.com/iree-org/iree/pull/22727): [DispatchCreation][NFC] Refactor split reduction helper methods to static functions

### Review Summary

**APPROVED** (2025-11-21)

Thanks. Can you mark your PR title with 'NFC' (non-functional change)?



---


## [PR #22692](https://github.com/iree-org/iree/pull/22692): [tuner][docs] update the example td spec in sharktuner readme

### Review Summary

**APPROVED** (2025-11-18)



---


## [PR #22683](https://github.com/iree-org/iree/pull/22683): [tuner][docs] update sharktuner readme

### Review Summary

**APPROVED** (2025-11-18)


### Code Comments

**File:** `docs/website/docs/reference/tuning.md:308`

```diff
@@ -306,6 +305,12 @@ that conform to the following format:
   the tuning spec includes a named sequence op with name `__kernel_config`,
   which must contain exactly one `foreach_match` op. That `foreach_match` op
   must have exactly one argument and one result of type any_op.
+* IREE provides transform matcher operations (e.g.,
```

**Comment:**
```suggestion
* IREE provides transform match operations (e.g.,
```

---


---


## [PR #22598](https://github.com/iree-org/iree/pull/22598): [Codegen][Tuner] expose python binding for getIGEMMGenericConvDetails

### Review Summary

**CHANGES_REQUESTED** (2025-11-08)


**COMMENTED** (2025-11-08)


**CHANGES_REQUESTED** (2025-11-08)


**COMMENTED** (2025-11-08)


**COMMENTED** (2025-11-10)


**COMMENTED** (2025-11-10)


**COMMENTED** (2025-11-10)


**COMMENTED** (2025-11-13)


**APPROVED** (2025-11-13)


### Code Comments

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_codegen.h:103`

```diff
@@ -91,6 +91,21 @@ ireeCodegenGetAttentionOpDetail(MlirAffineMap qMap, MlirAffineMap kMap,
 MLIR_CAPI_EXPORTED bool
 ireeCodegenMlirOperationIsACodegenAttentionOp(MlirOperation op);
 
+struct ireeCodegenIGEMMGenericConvDetails {
+  MlirAttribute igemmLoopBounds;
+  MlirAttribute convDimsBatch;
+  MlirAttribute convDimsOutputImage;
+  MlirAttribute convDimsOutputChannel;
+  MlirAttribute convDimsFilterLoop;
+  MlirAttribute convDimsInputChannel;
+  MlirAttribute convDimsDepth;
+  bool isOutputChannelFirst;
+  bool isValid;
```

**Comment:**
What does this mean? Would it make sense to have a separate helper function for checking if IGEMM details can be queried for a given op in the first place?

---

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp:256`

```diff
@@ -244,3 +246,71 @@ bool ireeCodegenMlirOperationIsACodegenAttentionOp(MlirOperation op) {
   return llvm::isa<mlir::iree_compiler::IREE::LinalgExt::AttentionOp>(
       unwrap(op));
 }
+
+ireeCodegenIGEMMGenericConvDetails
+ireeCodegenGetIGEMMGenericConvDetails(MlirOperation op) {
+  mlir::Operation *operation = unwrap(op);
+  auto linalgOp = llvm::dyn_cast<mlir::linalg::LinalgOp>(operation);
+
+  if (!linalgOp) {
+    return ireeCodegenIGEMMGenericConvDetails{
```

**Comment:**
Yeah, I think this would be better handled inside python bindings, and to assert here instead.

---

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_codegen.h:103`

```diff
@@ -91,6 +91,21 @@ ireeCodegenGetAttentionOpDetail(MlirAffineMap qMap, MlirAffineMap kMap,
 MLIR_CAPI_EXPORTED bool
 ireeCodegenMlirOperationIsACodegenAttentionOp(MlirOperation op);
 
+struct ireeCodegenIGEMMGenericConvDetails {
+  MlirAttribute igemmLoopBounds;
+  MlirAttribute convDimsBatch;
+  MlirAttribute convDimsOutputImage;
+  MlirAttribute convDimsOutputChannel;
+  MlirAttribute convDimsFilterLoop;
+  MlirAttribute convDimsInputChannel;
+  MlirAttribute convDimsDepth;
+  bool isOutputChannelFirst;
+  bool isValid;
```

**Comment:**
The rationale is that we generally prefer to invalid states not to be representable: https://geeklaunch.io/blog/make-invalid-states-unrepresentable/

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:33`

```diff
@@ -29,6 +29,9 @@ namespace py = nanobind;
 using namespace nanobind::literals;
 using namespace mlir::python::nanobind_adaptors;
 
+using ireeCodegenIGEMMGenericConvDetails =
+    struct ireeCodegenIGEMMGenericConvDetails;
```

**Comment:**
What does this do?

---

**File:** `compiler/bindings/python/test/api/tuner_api_test.py:305`

```diff
@@ -149,3 +149,182 @@ def test_isa_attention_op():
     assert len(root_op_list) == 1
     assert root_op_list[0].name == "iree_linalg_ext.attention"
     assert iree_codegen.isa_attention_op(root_op_list[0])
+
+
+@run
+def test_igemm_conv_details():
+    # Test 1: conv_2d_nhwc_hwcf.
+    module_str = """
+        module {
+            func.func @conv_2d_nhwc_hwcf(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
+                %0 = linalg.conv_2d_nhwc_hwcf { root_op, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
+                    ins(%arg0, %arg1 : tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
+                    outs(%arg2 : tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
+                return %0 : tensor<1x14x14x16xf32>
+            }
+        }
+    """
+    input_module = ir.Module.parse(module_str)
+    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
+
+    details = iree_codegen.get_igemm_generic_conv_details(root_op_list[0])
+    assert details is not None, "IGEMM details should be valid for NHWC_HWCF conv"
+    assert details.igemm_loop_bounds == [
+        1,
+        14,
+        14,
+        16,
+        36,
+    ], f"Expected [1,14,14,16,36], got {details.igemm_loop_bounds}"
+    assert details.conv_dims_batch == [
+        0
+    ], f"Expected batch=[0], got {details.conv_dims_batch}"
+    assert details.conv_dims_output_image == [
+        1,
+        2,
+    ], f"Expected output_image=[1,2], got {details.conv_dims_output_image}"
+    assert details.conv_dims_output_channel == [
+        3
+    ], f"Expected output_channel=[3], got {details.conv_dims_output_channel}"
+    assert details.conv_dims_filter_loop == [
+        4,
+        5,
+    ], f"Expected filter_loop=[4,5], got {details.conv_dims_filter_loop}"
+    assert details.conv_dims_input_channel == [
+        6
+    ], f"Expected input_channel=[6], got {details.conv_dims_input_channel}"
+    assert (
+        details.is_output_channel_first == False
+    ), f"Expected is_output_channel_first=False, got {details.is_output_channel_first}"
+
+    # Test 2: conv_2d_nhwc_fhwc.
+    module_str = """
+        module {
+            func.func @conv_2d_nhwc_fhwc(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<16x3x3x4xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
+                %0 = linalg.conv_2d_nhwc_fhwc { root_op, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
+                    ins(%arg0, %arg1 : tensor<1x16x16x4xf32>, tensor<16x3x3x4xf32>)
+                    outs(%arg2 : tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
+                return %0 : tensor<1x14x14x16xf32>
+            }
+        }
+    """
+    input_module = ir.Module.parse(module_str)
+    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
+
+    details = iree_codegen.get_igemm_generic_conv_details(root_op_list[0])
+    assert details is not None, "IGEMM details should be valid for NHWC_FHWC conv"
+    assert details.igemm_loop_bounds == [
+        1,
+        14,
+        14,
+        16,
+        36,
+    ], f"Expected [1,14,14,16,36], got {details.igemm_loop_bounds}"
+    assert details.conv_dims_batch == [
+        0
+    ], f"Expected batch=[0], got {details.conv_dims_batch}"
+    assert details.conv_dims_output_image == [
+        1,
+        2,
+    ], f"Expected output_image=[1,2], got {details.conv_dims_output_image}"
+    assert details.conv_dims_output_channel == [
+        3
+    ], f"Expected output_channel=[3], got {details.conv_dims_output_channel}"
+    assert isinstance(
+        details.is_output_channel_first, bool
+    ), "Should have is_output_channel_first flag"
+
+    # Test 3: conv_2d_nchw_fchw.
+    module_str = """
+        module {
+            func.func @conv_2d_nchw_fchw(%arg0: tensor<1x4x16x16xf32>, %arg1: tensor<16x4x3x3xf32>, %arg2: tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32> {
+                %0 = linalg.conv_2d_nchw_fchw { root_op, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
+                    ins(%arg0, %arg1 : tensor<1x4x16x16xf32>, tensor<16x4x3x3xf32>)
+                    outs(%arg2 : tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32>
+                return %0 : tensor<1x16x14x14xf32>
+            }
+        }
+    """
+    input_module = ir.Module.parse(module_str)
+    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
+
+    details = iree_codegen.get_igemm_generic_conv_details(root_op_list[0])
+    assert details is not None, "IGEMM details should be valid for NCHW conv"
+    assert details.igemm_loop_bounds == [
+        1,
+        16,
+        14,
+        14,
+        36,
+    ], f"Expected [1,16,14,14,36], got {details.igemm_loop_bounds}"
+    assert details.conv_dims_batch == [
+        0
+    ], f"Expected batch=[0], got {details.conv_dims_batch}"
+    assert details.conv_dims_output_image == [
+        2,
+        3,
+    ], f"Expected output_image=[2,3], got {details.conv_dims_output_image}"
+    assert details.conv_dims_filter_loop == [
+        5,
+        6,
+    ], f"Expected filter_loop=[5,6], got {details.conv_dims_filter_loop}"
+
+    # Test 4: linalg.generic with convolution pattern (weight backward).
+    module_str = """
+        module {
+            func.func @conv_generic_weight_backward(%arg0: tensor<16x98x64x96xf32>, %arg1: tensor<16x96x64x96xf32>, %arg2: tensor<96x3x96xf32>) -> tensor<96x3x96xf32> {
+                %0 = linalg.generic {
+                    indexing_maps = [
+                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d1 + d4, d5, d2)>,
+                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d0)>,
+                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
+                    ],
+                    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
+                } ins(%arg0, %arg1 : tensor<16x98x64x96xf32>, tensor<16x96x64x96xf32>) outs(%arg2 : tensor<96x3x96xf32>) attrs = {root_op} {
+                ^bb0(%in: f32, %in_1: f32, %out: f32):
+                    %mul = arith.mulf %in, %in_1 : f32
+                    %add = arith.addf %out, %mul : f32
+                    linalg.yield %add : f32
+                } -> tensor<96x3x96xf32>
+                return %0 : tensor<96x3x96xf32>
+            }
+        }
+    """
+    input_module = ir.Module.parse(module_str)
+    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
+
+    details = iree_codegen.get_igemm_generic_conv_details(root_op_list[0])
+    assert (
+        details is not None
+    ), "IGEMM details should be valid for generic 1D conv weight backward"
+    assert details.igemm_loop_bounds == [
+        96,
+        3,
+        96,
+        98304,
+    ], f"Expected [96,3,96,98304], got {details.igemm_loop_bounds}"
```

**Comment:**
I think we should either switch these to pytest and get these prints for free or repeating the expected values in error messages

---

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp:264`

```diff
@@ -244,3 +246,49 @@ bool ireeCodegenMlirOperationIsACodegenAttentionOp(MlirOperation op) {
   return llvm::isa<mlir::iree_compiler::IREE::LinalgExt::AttentionOp>(
       unwrap(op));
 }
+
+ireeCodegenIGEMMGenericConvDetails
+ireeCodegenGetIGEMMGenericConvDetails(MlirOperation op) {
+  ireeCodegenIGEMMGenericConvDetails result{};
+  auto linalgOp = llvm::dyn_cast<mlir::linalg::LinalgOp>(unwrap(op));
+  if (!linalgOp) {
+    return result;
+  }
+
+  llvm::FailureOr<mlir::iree_compiler::IREE::LinalgExt::IGEMMGenericConvDetails>
+      maybeDetails =
+          mlir::iree_compiler::IREE::LinalgExt::getIGEMMGenericConvDetails(
+              linalgOp);
+  if (failed(maybeDetails)) {
+    return result;
+  }
```

**Comment:**
This doesn't remove the invalid state, it just changes the representation.

My suggestion is to have a new public C API function that checks if something has igemm details, and separately a get function that asserts that the igemm details can be queried.

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:33`

```diff
@@ -29,6 +29,9 @@ namespace py = nanobind;
 using namespace nanobind::literals;
 using namespace mlir::python::nanobind_adaptors;
 
+using ireeCodegenIGEMMGenericConvDetails =
+    struct ireeCodegenIGEMMGenericConvDetails;
```

**Comment:**
I don't understand what it does

---

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp:260`

```diff
@@ -244,3 +246,59 @@ bool ireeCodegenMlirOperationIsACodegenAttentionOp(MlirOperation op) {
   return llvm::isa<mlir::iree_compiler::IREE::LinalgExt::AttentionOp>(
       unwrap(op));
 }
+
+bool ireeCodegenHasIGEMMGenericConvDetails(MlirOperation op) {
+  auto linalgOp = llvm::dyn_cast<mlir::linalg::LinalgOp>(unwrap(op));
+  if (!linalgOp) {
+    return false;
+  }
+
+  llvm::FailureOr<mlir::iree_compiler::IREE::LinalgExt::IGEMMGenericConvDetails>
+      maybeDetails =
+          mlir::iree_compiler::IREE::LinalgExt::getIGEMMGenericConvDetails(
+              linalgOp);
+  return succeeded(maybeDetails);
```

**Comment:**
```suggestion
  return succeeded(mlir::iree_compiler::IREE::LinalgExt::getIGEMMGenericConvDetails(
              linalgOp));
```

---

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp:283`

```diff
@@ -244,3 +246,59 @@ bool ireeCodegenMlirOperationIsACodegenAttentionOp(MlirOperation op) {
   return llvm::isa<mlir::iree_compiler::IREE::LinalgExt::AttentionOp>(
       unwrap(op));
 }
+
+bool ireeCodegenHasIGEMMGenericConvDetails(MlirOperation op) {
+  auto linalgOp = llvm::dyn_cast<mlir::linalg::LinalgOp>(unwrap(op));
+  if (!linalgOp) {
+    return false;
+  }
+
+  llvm::FailureOr<mlir::iree_compiler::IREE::LinalgExt::IGEMMGenericConvDetails>
+      maybeDetails =
+          mlir::iree_compiler::IREE::LinalgExt::getIGEMMGenericConvDetails(
+              linalgOp);
+  return succeeded(maybeDetails);
+}
+
+ireeCodegenIGEMMGenericConvDetails
+ireeCodegenGetIGEMMGenericConvDetails(MlirOperation op) {
+  auto linalgOp = llvm::cast<mlir::linalg::LinalgOp>(unwrap(op));
+
+  llvm::FailureOr<mlir::iree_compiler::IREE::LinalgExt::IGEMMGenericConvDetails>
+      maybeDetails =
+          mlir::iree_compiler::IREE::LinalgExt::getIGEMMGenericConvDetails(
+              linalgOp);
+  assert(succeeded(maybeDetails) &&
+         "Failed to get IGEMM details; must check with "
+         "ireeCodegenCanGetIGEMMGenericConvDetails first");
+
+  const mlir::iree_compiler::IREE::LinalgExt::IGEMMGenericConvDetails &details =
+      *maybeDetails;
+
+  mlir::Builder builder(linalgOp.getContext());
+
+  // Helper to convert unsigned to int64_t.
+  auto toInt64 = [](const llvm::SmallVector<unsigned, 2> &vec) {
+    return llvm::map_to_vector(
+        vec, [](unsigned val) { return static_cast<int64_t>(val); });
```

**Comment:**
```suggestion
    return llvm::map_to_vector(
        vec, llvm::StaticCastTo<int64_t>);
```

---

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp:283`

```diff
@@ -244,3 +246,59 @@ bool ireeCodegenMlirOperationIsACodegenAttentionOp(MlirOperation op) {
   return llvm::isa<mlir::iree_compiler::IREE::LinalgExt::AttentionOp>(
       unwrap(op));
 }
+
+bool ireeCodegenHasIGEMMGenericConvDetails(MlirOperation op) {
+  auto linalgOp = llvm::dyn_cast<mlir::linalg::LinalgOp>(unwrap(op));
+  if (!linalgOp) {
+    return false;
+  }
+
+  llvm::FailureOr<mlir::iree_compiler::IREE::LinalgExt::IGEMMGenericConvDetails>
+      maybeDetails =
+          mlir::iree_compiler::IREE::LinalgExt::getIGEMMGenericConvDetails(
+              linalgOp);
+  return succeeded(maybeDetails);
+}
+
+ireeCodegenIGEMMGenericConvDetails
+ireeCodegenGetIGEMMGenericConvDetails(MlirOperation op) {
+  auto linalgOp = llvm::cast<mlir::linalg::LinalgOp>(unwrap(op));
+
+  llvm::FailureOr<mlir::iree_compiler::IREE::LinalgExt::IGEMMGenericConvDetails>
+      maybeDetails =
+          mlir::iree_compiler::IREE::LinalgExt::getIGEMMGenericConvDetails(
+              linalgOp);
+  assert(succeeded(maybeDetails) &&
+         "Failed to get IGEMM details; must check with "
+         "ireeCodegenCanGetIGEMMGenericConvDetails first");
+
+  const mlir::iree_compiler::IREE::LinalgExt::IGEMMGenericConvDetails &details =
+      *maybeDetails;
+
+  mlir::Builder builder(linalgOp.getContext());
+
+  // Helper to convert unsigned to int64_t.
+  auto toInt64 = [](const llvm::SmallVector<unsigned, 2> &vec) {
+    return llvm::map_to_vector(
+        vec, [](unsigned val) { return static_cast<int64_t>(val); });
```

**Comment:**
It landed in llvm a week ago

---

**File:** `compiler/bindings/python/test/api/tuner_api_test.py:195`

```diff
@@ -149,3 +149,174 @@ def test_isa_attention_op():
     assert len(root_op_list) == 1
     assert root_op_list[0].name == "iree_linalg_ext.attention"
     assert iree_codegen.isa_attention_op(root_op_list[0])
+
+
+@run
+def test_igemm_conv_details():
+    # Test 1: conv_2d_nhwc_hwcf.
+    module_str = """
+        module {
+            func.func @conv_2d_nhwc_hwcf(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
+                %0 = linalg.conv_2d_nhwc_hwcf { root_op, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
+                    ins(%arg0, %arg1 : tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
+                    outs(%arg2 : tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
+                return %0 : tensor<1x14x14x16xf32>
+            }
+        }
+    """
+    input_module = ir.Module.parse(module_str)
+    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
+
+    details = iree_codegen.get_igemm_generic_conv_details(root_op_list[0])
+    assert details is not None, "IGEMM details should be valid for NHWC_HWCF conv"
+    assert details.igemm_loop_bounds == [
+        1,
+        14,
+        14,
+        16,
+        36,
+    ], f"got {details.igemm_loop_bounds}"
+    assert details.conv_dims_batch == [0], f"got {details.conv_dims_batch}"
+    assert details.conv_dims_output_image == [
+        1,
+        2,
+    ], f"got {details.conv_dims_output_image}"
+    assert details.conv_dims_output_channel == [
+        3
+    ], f"got {details.conv_dims_output_channel}"
+    assert details.conv_dims_filter_loop == [
+        4,
+        5,
+    ], f"got {details.conv_dims_filter_loop}"
+    assert details.conv_dims_input_channel == [
+        6
+    ], f"got {details.conv_dims_input_channel}"
+    assert (
+        details.is_output_channel_first == False
```

**Comment:**
```suggestion
        not details.is_output_channel_first
```

---

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp:279`

```diff
@@ -244,3 +246,56 @@ bool ireeCodegenMlirOperationIsACodegenAttentionOp(MlirOperation op) {
   return llvm::isa<mlir::iree_compiler::IREE::LinalgExt::AttentionOp>(
       unwrap(op));
 }
+
+bool ireeCodegenHasIGEMMGenericConvDetails(MlirOperation op) {
+  auto linalgOp = llvm::dyn_cast<mlir::linalg::LinalgOp>(unwrap(op));
+  if (!linalgOp) {
+    return false;
+  }
+
+  return succeeded(
+      mlir::iree_compiler::IREE::LinalgExt::getIGEMMGenericConvDetails(
+          linalgOp));
+}
+
+ireeCodegenIGEMMGenericConvDetails
+ireeCodegenGetIGEMMGenericConvDetails(MlirOperation op) {
+  auto linalgOp = llvm::cast<mlir::linalg::LinalgOp>(unwrap(op));
+
+  llvm::FailureOr<mlir::iree_compiler::IREE::LinalgExt::IGEMMGenericConvDetails>
+      maybeDetails =
+          mlir::iree_compiler::IREE::LinalgExt::getIGEMMGenericConvDetails(
+              linalgOp);
+  assert(succeeded(maybeDetails) &&
+         "Failed to get IGEMM details; must check with "
+         "ireeCodegenHasIGEMMGenericConvDetails first");
+
+  const mlir::iree_compiler::IREE::LinalgExt::IGEMMGenericConvDetails &details =
+      *maybeDetails;
+
+  mlir::Builder builder(linalgOp.getContext());
+
+  // Helper to convert unsigned to int64_t.
+  auto toInt64 = [](const llvm::SmallVector<unsigned, 2> &vec) {
```

**Comment:**
```suggestion
  auto toInt64 = [](ArrayRef<unsigned> vec) {
```
to avoid over-relying on the exact small vector size

---

**File:** `compiler/bindings/python/test/api/tuner_api_test.py:311`

```diff
@@ -149,3 +149,174 @@ def test_isa_attention_op():
     assert len(root_op_list) == 1
     assert root_op_list[0].name == "iree_linalg_ext.attention"
     assert iree_codegen.isa_attention_op(root_op_list[0])
+
+
+@run
+def test_igemm_conv_details():
+    # Test 1: conv_2d_nhwc_hwcf.
+    module_str = """
+        module {
+            func.func @conv_2d_nhwc_hwcf(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
+                %0 = linalg.conv_2d_nhwc_hwcf { root_op, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
+                    ins(%arg0, %arg1 : tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
+                    outs(%arg2 : tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
+                return %0 : tensor<1x14x14x16xf32>
+            }
+        }
+    """
+    input_module = ir.Module.parse(module_str)
+    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
+
+    details = iree_codegen.get_igemm_generic_conv_details(root_op_list[0])
+    assert details is not None, "IGEMM details should be valid for NHWC_HWCF conv"
+    assert details.igemm_loop_bounds == [
+        1,
+        14,
+        14,
+        16,
+        36,
+    ], f"got {details.igemm_loop_bounds}"
+    assert details.conv_dims_batch == [0], f"got {details.conv_dims_batch}"
+    assert details.conv_dims_output_image == [
+        1,
+        2,
+    ], f"got {details.conv_dims_output_image}"
+    assert details.conv_dims_output_channel == [
+        3
+    ], f"got {details.conv_dims_output_channel}"
+    assert details.conv_dims_filter_loop == [
+        4,
+        5,
+    ], f"got {details.conv_dims_filter_loop}"
+    assert details.conv_dims_input_channel == [
+        6
+    ], f"got {details.conv_dims_input_channel}"
+    assert (
+        details.is_output_channel_first == False
+    ), f"got {details.is_output_channel_first}"
+
+    # Test 2: conv_2d_nhwc_fhwc.
+    module_str = """
+        module {
+            func.func @conv_2d_nhwc_fhwc(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<16x3x3x4xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
+                %0 = linalg.conv_2d_nhwc_fhwc { root_op, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
+                    ins(%arg0, %arg1 : tensor<1x16x16x4xf32>, tensor<16x3x3x4xf32>)
+                    outs(%arg2 : tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
+                return %0 : tensor<1x14x14x16xf32>
+            }
+        }
+    """
+    input_module = ir.Module.parse(module_str)
+    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
+
+    details = iree_codegen.get_igemm_generic_conv_details(root_op_list[0])
+    assert details is not None, "IGEMM details should be valid for NHWC_FHWC conv"
+    assert details.igemm_loop_bounds == [
+        1,
+        14,
+        14,
+        16,
+        36,
+    ], f"got {details.igemm_loop_bounds}"
+    assert details.conv_dims_batch == [0], f"got {details.conv_dims_batch}"
+    assert details.conv_dims_output_image == [
+        1,
+        2,
+    ], f"got {details.conv_dims_output_image}"
+    assert details.conv_dims_output_channel == [
+        3
+    ], f"got {details.conv_dims_output_channel}"
+    assert isinstance(
+        details.is_output_channel_first, bool
+    ), f"got {type(details.is_output_channel_first)}"
+
+    # Test 3: conv_2d_nchw_fchw.
+    module_str = """
+        module {
+            func.func @conv_2d_nchw_fchw(%arg0: tensor<1x4x16x16xf32>, %arg1: tensor<16x4x3x3xf32>, %arg2: tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32> {
+                %0 = linalg.conv_2d_nchw_fchw { root_op, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
+                    ins(%arg0, %arg1 : tensor<1x4x16x16xf32>, tensor<16x4x3x3xf32>)
+                    outs(%arg2 : tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32>
+                return %0 : tensor<1x16x14x14xf32>
+            }
+        }
+    """
+    input_module = ir.Module.parse(module_str)
+    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
+
+    details = iree_codegen.get_igemm_generic_conv_details(root_op_list[0])
+    assert details is not None, "IGEMM details should be valid for NCHW conv"
+    assert details.igemm_loop_bounds == [
+        1,
+        16,
+        14,
+        14,
+        36,
+    ], f"got {details.igemm_loop_bounds}"
+    assert details.conv_dims_batch == [0], f"got {details.conv_dims_batch}"
+    assert details.conv_dims_output_image == [
+        2,
+        3,
+    ], f"got {details.conv_dims_output_image}"
+    assert details.conv_dims_filter_loop == [
+        5,
+        6,
+    ], f"got {details.conv_dims_filter_loop}"
+
+    # Test 4: linalg.generic with convolution pattern (weight backward).
+    module_str = """
+        module {
+            func.func @conv_generic_weight_backward(%arg0: tensor<16x98x64x96xf32>, %arg1: tensor<16x96x64x96xf32>, %arg2: tensor<96x3x96xf32>) -> tensor<96x3x96xf32> {
+                %0 = linalg.generic {
+                    indexing_maps = [
+                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d1 + d4, d5, d2)>,
+                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d0)>,
+                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
+                    ],
+                    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
+                } ins(%arg0, %arg1 : tensor<16x98x64x96xf32>, tensor<16x96x64x96xf32>) outs(%arg2 : tensor<96x3x96xf32>) attrs = {root_op} {
+                ^bb0(%in: f32, %in_1: f32, %out: f32):
+                    %mul = arith.mulf %in, %in_1 : f32
+                    %add = arith.addf %out, %mul : f32
+                    linalg.yield %add : f32
+                } -> tensor<96x3x96xf32>
+                return %0 : tensor<96x3x96xf32>
+            }
+        }
+    """
+    input_module = ir.Module.parse(module_str)
+    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
+
+    details = iree_codegen.get_igemm_generic_conv_details(root_op_list[0])
+    assert (
+        details is not None
+    ), "IGEMM details should be valid for generic 1D conv weight backward"
+    assert details.igemm_loop_bounds == [
+        96,
+        3,
+        96,
+        98304,
+    ], f"got {details.igemm_loop_bounds}"
+    assert details.conv_dims_output_image == [
+        1
+    ], f"got {details.conv_dims_output_image}"
+    assert details.conv_dims_filter_loop == [4], f"got {details.conv_dims_filter_loop}"
+
+    # Test with a non-conv operation.
+    module_str = """
+        module {
+            func.func @matmul(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
+                %cst = arith.constant 0.000000e+00 : f32
+                %0 = tensor.empty() : tensor<4x4xf32>
+                %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
```

**Comment:**
You can make %1 a third function argument to make this test case more concise

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:728`

```diff
@@ -693,4 +694,80 @@ NB_MODULE(_ireeCompilerDialects, m) {
       "isa_attention_op", &ireeCodegenMlirOperationIsACodegenAttentionOp,
       "Checks if the given operation is an IREE LinalgExt attention op.",
       py::arg("op"));
+
+  //===-------------------------------------------------------------------===//
+  // Binding to utility function ireeCodegenGetIGEMMGenericConvDetails
+  //===-------------------------------------------------------------------===//
+  py::class_<ireeCodegenIGEMMGenericConvDetails>(iree_codegen_module,
+                                                 "IGEMMGenericConvDetails")
+      .def_prop_ro("igemm_contraction_maps",
+                   [](const ireeCodegenIGEMMGenericConvDetails &self) {
+                     return self.igemmContractionMaps;
+                   })
+      .def_prop_ro("igemm_loop_bounds",
+                   [](const ireeCodegenIGEMMGenericConvDetails &self) {
+                     return getIntArrayAttrValues(self.igemmLoopBounds);
+                   })
+      .def_prop_ro("igemm_loop_iterators",
+                   [](const ireeCodegenIGEMMGenericConvDetails &self) {
+                     return self.igemmLoopIterators;
+                   })
+      .def_prop_ro("im2col_output_perm",
+                   [](const ireeCodegenIGEMMGenericConvDetails &self) {
+                     return getIntArrayAttrValues(self.im2colOutputPerm);
+                   })
+      .def_prop_ro(
+          "filter_reassoc_indices",
+          [](const ireeCodegenIGEMMGenericConvDetails &self)
+              -> std::vector<std::vector<int64_t>> {
+            std::vector<std::vector<int64_t>> result;
+            MlirAttribute attr = self.filterReassocIndices;
+            assert(!mlirAttributeIsNull(attr) && mlirAttributeIsAArray(attr) &&
+                   "filterReassocIndices should be a valid ArrayAttr");
+            size_t n = mlirArrayAttrGetNumElements(attr);
+            result.reserve(n);
```

**Comment:**
```suggestion
            MlirAttribute attr = self.filterReassocIndices;
            assert(!mlirAttributeIsNull(attr) && mlirAttributeIsAArray(attr) &&
                   "filterReassocIndices should be a valid ArrayAttr");
            size_t n = mlirArrayAttrGetNumElements(attr);
            std::vector<std::vector<int64_t>> result;
            result.reserve(n);
```
try to keep the definition close to the first use

---

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp:298`

```diff
@@ -244,3 +245,76 @@ bool ireeCodegenMlirOperationIsACodegenAttentionOp(MlirOperation op) {
   return llvm::isa<mlir::iree_compiler::IREE::LinalgExt::AttentionOp>(
       unwrap(op));
 }
+
+bool ireeCodegenHasIGEMMGenericConvDetails(MlirOperation op) {
+  auto linalgOp = llvm::dyn_cast<mlir::linalg::LinalgOp>(unwrap(op));
+  if (!linalgOp) {
+    return false;
+  }
+
+  return succeeded(
+      mlir::iree_compiler::IREE::LinalgExt::getIGEMMGenericConvDetails(
+          linalgOp));
+}
+
+ireeCodegenIGEMMGenericConvDetails
+ireeCodegenGetIGEMMGenericConvDetails(MlirOperation op) {
+  auto linalgOp = llvm::cast<mlir::linalg::LinalgOp>(unwrap(op));
+
+  llvm::FailureOr<mlir::iree_compiler::IREE::LinalgExt::IGEMMGenericConvDetails>
+      maybeDetails =
+          mlir::iree_compiler::IREE::LinalgExt::getIGEMMGenericConvDetails(
+              linalgOp);
+  assert(succeeded(maybeDetails) &&
+         "Failed to get IGEMM details; must check with "
+         "ireeCodegenHasIGEMMGenericConvDetails first");
+
+  const mlir::iree_compiler::IREE::LinalgExt::IGEMMGenericConvDetails &details =
+      *maybeDetails;
+
+  mlir::Builder builder(linalgOp.getContext());
+
+  ireeCodegenIGEMMGenericConvDetails result;
+
+  result.igemmContractionMaps = wrap(builder.getArrayAttr(llvm::map_to_vector(
+      details.igemmContractionMaps, [](auto map) -> mlir::Attribute {
+        return mlir::AffineMapAttr::get(map);
+      })));
+
+  result.igemmLoopBounds =
+      wrap(builder.getI64ArrayAttr(details.igemmLoopBounds));
+
+  llvm::SmallVector<mlir::Attribute> iteratorAttrs;
+  for (auto iterType : details.igemmLoopIterators) {
+    iteratorAttrs.push_back(
+        builder.getStringAttr(mlir::utils::stringifyIteratorType(iterType)));
+  }
+  result.igemmLoopIterators = wrap(builder.getArrayAttr(iteratorAttrs));
+
+  result.im2colOutputPerm =
+      wrap(builder.getI64ArrayAttr(details.im2colOutputPerm));
+
+  llvm::SmallVector<mlir::Attribute> reassocAttrs;
+  for (const auto &indices : details.filterReassocIndices) {
```

**Comment:**
can you spell out this type? https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---


---


## [PR #22490](https://github.com/iree-org/iree/pull/22490): [Codegen][GPU] Generalize linalg.reduce operations

### Review Summary

**COMMENTED** (2025-10-31)


**APPROVED** (2025-11-20)


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Common/TypePropagationPass.cpp:434`

```diff
@@ -429,6 +429,91 @@ struct IREELinalgExtSortTypePropagation
   }
 };
 
+/// Pattern to legalize `linalg.reduce` operations.
+struct ReduceOpTypePropagation
+    : public TypePropagationPattern<linalg::ReduceOp> {
```

**Comment:**
```suggestion
struct ReduceOpTypePropagation final
    : TypePropagationPattern<linalg::ReduceOp> {
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/TypePropagationPass.cpp:435`

```diff
@@ -429,6 +429,91 @@ struct IREELinalgExtSortTypePropagation
   }
 };
 
+/// Pattern to legalize `linalg.reduce` operations.
+struct ReduceOpTypePropagation
+    : public TypePropagationPattern<linalg::ReduceOp> {
+  using TypePropagationPattern<linalg::ReduceOp>::TypePropagationPattern;
```

**Comment:**
```suggestion
  using TypePropagationPattern::TypePropagationPattern;
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/TypePropagationPass.cpp:441`

```diff
@@ -429,6 +429,91 @@ struct IREELinalgExtSortTypePropagation
   }
 };
 
+/// Pattern to legalize `linalg.reduce` operations.
+struct ReduceOpTypePropagation
+    : public TypePropagationPattern<linalg::ReduceOp> {
+  using TypePropagationPattern<linalg::ReduceOp>::TypePropagationPattern;
+
+  LogicalResult
+  matchAndRewrite(linalg::ReduceOp reduceOp, OpAdaptor adaptor,
+                  ConversionPatternRewriter &rewriter) const final {
+    bool needsConversion = false;
+    for (auto &operand : reduceOp->getOpOperands()) {
```

**Comment:**
spell out types that are not obvious based on the immediate context: https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

**File:** `compiler/src/iree/compiler/Codegen/Common/TypePropagationPass.cpp:443`

```diff
@@ -429,6 +429,91 @@ struct IREELinalgExtSortTypePropagation
   }
 };
 
+/// Pattern to legalize `linalg.reduce` operations.
+struct ReduceOpTypePropagation
+    : public TypePropagationPattern<linalg::ReduceOp> {
+  using TypePropagationPattern<linalg::ReduceOp>::TypePropagationPattern;
+
+  LogicalResult
+  matchAndRewrite(linalg::ReduceOp reduceOp, OpAdaptor adaptor,
+                  ConversionPatternRewriter &rewriter) const final {
+    bool needsConversion = false;
+    for (auto &operand : reduceOp->getOpOperands()) {
+      Type operandType = operand.get().getType();
+      Type legalizedType = this->getTypeConverter()->convertType(operandType);
```

**Comment:**
what if type conversion fails?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/TypePropagationPass.cpp:465`

```diff
@@ -429,6 +429,91 @@ struct IREELinalgExtSortTypePropagation
   }
 };
 
+/// Pattern to legalize `linalg.reduce` operations.
+struct ReduceOpTypePropagation
+    : public TypePropagationPattern<linalg::ReduceOp> {
+  using TypePropagationPattern<linalg::ReduceOp>::TypePropagationPattern;
+
+  LogicalResult
+  matchAndRewrite(linalg::ReduceOp reduceOp, OpAdaptor adaptor,
+                  ConversionPatternRewriter &rewriter) const final {
+    bool needsConversion = false;
+    for (auto &operand : reduceOp->getOpOperands()) {
+      Type operandType = operand.get().getType();
+      Type legalizedType = this->getTypeConverter()->convertType(operandType);
+      if (operandType != legalizedType) {
+        needsConversion = true;
+        break;
+      }
+    }
+
+    if (!needsConversion) {
+      return rewriter.notifyMatchFailure(
+          reduceOp, "all types already legal within conversion pattern");
+    }
+
+    SmallVector<Type> resultTypes;
+    for (auto init : reduceOp.getInits()) {
+      Type legalizedType =
+          this->getTypeConverter()->convertType(init.getType());
+      resultTypes.push_back(legalizedType);
```

**Comment:**
also here

---

**File:** `compiler/src/iree/compiler/Codegen/Common/TypePropagationPass.cpp:497`

```diff
@@ -429,6 +429,91 @@ struct IREELinalgExtSortTypePropagation
   }
 };
 
+/// Pattern to legalize `linalg.reduce` operations.
+struct ReduceOpTypePropagation
+    : public TypePropagationPattern<linalg::ReduceOp> {
+  using TypePropagationPattern<linalg::ReduceOp>::TypePropagationPattern;
+
+  LogicalResult
+  matchAndRewrite(linalg::ReduceOp reduceOp, OpAdaptor adaptor,
+                  ConversionPatternRewriter &rewriter) const final {
+    bool needsConversion = false;
+    for (auto &operand : reduceOp->getOpOperands()) {
+      Type operandType = operand.get().getType();
+      Type legalizedType = this->getTypeConverter()->convertType(operandType);
+      if (operandType != legalizedType) {
+        needsConversion = true;
+        break;
+      }
+    }
+
+    if (!needsConversion) {
+      return rewriter.notifyMatchFailure(
+          reduceOp, "all types already legal within conversion pattern");
+    }
+
+    SmallVector<Type> resultTypes;
+    for (auto init : reduceOp.getInits()) {
+      Type legalizedType =
+          this->getTypeConverter()->convertType(init.getType());
+      resultTypes.push_back(legalizedType);
+    }
+
+    auto modifiedOp = cast<linalg::ReduceOp>(mlir::cloneWithoutRegions(
+        rewriter, reduceOp.getOperation(), resultTypes, adaptor.getOperands()));
+
+    rewriter.inlineRegionBefore(reduceOp.getCombiner(),
+                                modifiedOp.getCombiner(),
+                                modifiedOp.getCombiner().begin());
+    Region &modifiedOpRegion = modifiedOp.getCombiner();
+
+    TypeConverter::SignatureConversion signatureConverter(
+        modifiedOpRegion.getNumArguments());
+    for (auto [index, arg] : llvm::enumerate(modifiedOpRegion.getArguments())) {
+      Type argType = arg.getType();
+      std::optional<Type> legalizedArgType =
+          legalizeStorageElementType(argType);
+      if (!legalizedArgType) {
+        return reduceOp.emitOpError("failed to get legalized type for arg ")
+               << index;
+      }
+      signatureConverter.addInputs(index, legalizedArgType.value());
+    }
+    rewriter.applySignatureConversion(&modifiedOpRegion.front(),
+                                      signatureConverter, getTypeConverter());
+
+    OpBuilder::InsertionGuard g(rewriter);
+    Block *entryBlock = &modifiedOpRegion.front();
+    Operation *yieldOp = entryBlock->getTerminator();
+    rewriter.setInsertionPoint(yieldOp);
+
+    SmallVector<Value> convertedYieldValues;
+    bool needsUpdate = false;
+    for (Value yieldValue : yieldOp->getOperands()) {
+      Type yieldType = yieldValue.getType();
+      std::optional<Type> legalizedYieldType =
+          legalizeStorageElementType(yieldType);
+
+      Value valueToYield = yieldValue; // Default to original value
```

**Comment:**
```suggestion
      Value valueToYield = yieldValue; // Default to original value.
```
https://llvm.org/docs/CodingStandards.html#commenting

---


---


## [PR #22466](https://github.com/iree-org/iree/pull/22466): [DispatchCreation] Set split reduction size for ArgCompare

### Review Summary

**COMMENTED** (2025-11-20)


**APPROVED** (2025-11-20)


### Code Comments

**File:** `compiler/src/iree/compiler/DispatchCreation/test/set_split_reduction_sizes_outer_reduction.mlir:188`

```diff
@@ -179,3 +179,24 @@ util.func public @arg_compare_negative_outer_dynamic_reduction(
 
   util.return %res_val, %res_idx : tensor<64xf32>, tensor<64xindex>
 }
+
+// -----
+
+// CHECK-LABEL: @arg_compare_large_inner_reduction
+util.func public @arg_compare_large_inner_reduction(%arg0: tensor<4x1x128256xf16>)
+    -> tensor<4x1xi32> {
+  // CHECK: iree_linalg_ext.split_reduction = [1336 : index]
```

**Comment:**
Maybe it would be also worth adding a test case for when split reduction is not applied because the reduction dimension is below the threshold?

---


---


## [PR #22409](https://github.com/iree-org/iree/pull/22409): [Codegen][Tuner] solve name conflicts for merging td specs

### Review Summary

**CHANGES_REQUESTED** (2025-10-25)

Typo in the PR title: s/conflictions/conflicts/


**COMMENTED** (2025-10-27)


**APPROVED** (2025-10-27)


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:189`

```diff
@@ -174,7 +177,17 @@ static void updateNamedSequenceOp(
   }
 
   std::string newSpecName = llvm::formatv("{}_{}", moduleName, specName).str();
-  op.setSymName(newSpecName);
+
+  // Ensure the new name is unique by appending a counter if needed.
+  std::string uniqueNewSpecName = newSpecName;
+  unsigned suffix = 0;
+  while (seenNames.contains(uniqueNewSpecName)) {
+    uniqueNewSpecName = llvm::formatv("{}_{}", newSpecName, suffix).str();
+    ++suffix;
+  }
+
+  seenNames.insert(builder.getStringAttr(uniqueNewSpecName).getValue());
```

**Comment:**
There's no need to build a string attribute here

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:153`

```diff
@@ -144,16 +144,19 @@ static TuningSpecsToMerge collectTuningSpecsToMerge(ModuleOp module) {
 
 // Renames a `NamedSequenceOp` to resolve name conflicts caused by merging
 // tuning specs.
-// The name conflict resolution strategy follows below two rules:
+// The name conflict resolution strategy follows below rules:
 //   1. If the `NamedSequenceOp` is inside a module with a valid symbol name,
 //      its new name is prefixed with its containing module's symbol name.
 //   2. If the module has no symbol name, an incrementing counter is used
 //      to generate a unique prefix (e.g., `m0_`, `m1_`, etc.).
+//   3. If the prefixed name still conflicts with an existing name, a numeric
+//      suffix is appended (e.g., `m0_apply_op_config_0`) until a unique name
```

**Comment:**
Could we make it simpler by appending unique suffix only? Not sure how much value there is in having these prefixes since usually the names we append come from tiny modules with a single matcher only.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:159`

```diff
@@ -143,38 +143,24 @@ static TuningSpecsToMerge collectTuningSpecsToMerge(ModuleOp module) {
 }
 
 // Renames a `NamedSequenceOp` to resolve name conflicts caused by merging
-// tuning specs.
-// The name conflict resolution strategy follows below two rules:
-//   1. If the `NamedSequenceOp` is inside a module with a valid symbol name,
-//      its new name is prefixed with its containing module's symbol name.
-//   2. If the module has no symbol name, an incrementing counter is used
-//      to generate a unique prefix (e.g., `m0_`, `m1_`, etc.).
+// tuning specs by appending a numeric suffix until a unique name is found.
 static void updateNamedSequenceOp(
     NamedSequenceOp op, OpBuilder &builder,
     llvm::DenseMap<NamedSequenceOp, ForeachMatchOp> &namedSequenceToUser,
-    llvm::DenseMap<ModuleOp, std::string> &unnamedModuleNames,
-    unsigned &unnamedModuleCounter) {
+    llvm::DenseSet<StringRef> &seenNames) {
   StringRef specName = op.getSymName();
-  ModuleOp parentModule = op->getParentOfType<ModuleOp>();
-  assert(parentModule);
-  StringAttr parentSymbol = parentModule.getSymNameAttr();
-  std::string moduleName;
-  if (parentSymbol) {
-    moduleName = parentSymbol.getValue().str();
-  } else {
-    if (unnamedModuleNames.contains(parentModule)) {
-      moduleName = unnamedModuleNames[parentModule];
-    } else {
-      std::string newModuleName =
-          llvm::formatv("m{}", unnamedModuleCounter).str();
-      ++unnamedModuleCounter;
-      unnamedModuleNames[parentModule] = newModuleName;
-      moduleName = newModuleName;
-    }
+
+  // Ensure the name is unique by appending a numeric suffix if needed.
+  std::string uniqueNewSpecName = specName.str();
+  unsigned suffix = 0;
+  while (seenNames.contains(uniqueNewSpecName)) {
+    uniqueNewSpecName = llvm::formatv("{}_{}", specName, suffix).str();
+    ++suffix;
   }
```

**Comment:**
nit: You can turn this into a for loop since `suffix` is not used anywhere else

```suggestion
  for (unsigned suffix = 0; seenNames.contains(uniqueNewSpecName); ++suffix) {
    uniqueNewSpecName = llvm::formatv("{}_{}", specName, suffix).str();
  }
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:226`

```diff
@@ -237,11 +223,8 @@ static LogicalResult resolveAndMoveNamedSequenceOps(
 
   // Update conflicted named sequence ops.
   if (!nameConflictOps.empty()) {
-    llvm::DenseMap<ModuleOp, std::string> unnamedModuleNames;
-    unsigned unnamedModuleCounter = 0;
     for (NamedSequenceOp op : nameConflictOps) {
```

**Comment:**
We don't need the `if` statement above

---


---


## [PR #22348](https://github.com/iree-org/iree/pull/22348): [Codegen][Tuner] Add root_op for matvec and reduction along VectorDistribute pipeline 

### Review Summary

**COMMENTED** (2025-10-18)

Nice catch. Can we simply the tests though (e.g., use function arguments like the other existing rest)?


**COMMENTED** (2025-10-20)


**APPROVED** (2025-10-20)


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/config_root_op_attribute.mlir:22`

```diff
@@ -10,3 +10,52 @@ func.func @matmul(%lhs: tensor<4x4xf32>, %rhs: tensor<4x4xf32>) -> tensor<4x4xf3
 }
 
 // CHECK: %2 = linalg.matmul {lowering_config = #{{.*}}, root_op} ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%1 : tensor<4x4xf32>) -> tensor<4x4xf32>
+
+// -----
+
+#map = affine_map<(d0, d1, d2) -> (d0, d2)>
+#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
+#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
+
+func.func @matvec_like(%lhs: tensor<1x4096xf16>, %rhs: tensor<32000x4096xf16>, %init: tensor<1x32000xf16>) {
+  %output = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer>]>) binding(0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x32000xf16>>
+  %result = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
```

**Comment:**
Can we use linalg.matmul like in the test cases above? This should make the IR more concise.

---


---


## [PR #22311](https://github.com/iree-org/iree/pull/22311): [python] Set up python binding for matcher convolution and attention op

### Review Summary

**APPROVED** (2025-10-15)



---


## [PR #22270](https://github.com/iree-org/iree/pull/22270): [Codegen] Update the assembly formats and corresponding tests for matcher ops

### Review Summary

**APPROVED** (2025-10-10)



---


## [PR #22266](https://github.com/iree-org/iree/pull/22266): [Codegen] Update the td spec using the attention matcher op

### Review Summary

**COMMENTED** (2025-10-10)

Can you un-stack this PR? I don't think it depends on any of the base PRs.


**COMMENTED** (2025-10-10)

Can we layer these upgrades to that we handle contractions / convs / attention separately, in case one of them causes issues? Splitting them by file doesn't really change the blast radius IMO.


**COMMENTED** (2025-10-10)

LGTM but this doesn't need the `(2/2)` in the PR title since there's only one PR for attention


**APPROVED** (2025-10-10)



---


## [PR #22249](https://github.com/iree-org/iree/pull/22249): [Codegen] Update the td spec using the contraction matcher op

### Review Summary

**COMMENTED** (2025-10-10)

Can you rebase it and remove unrelated base commits?

> It seems that using
> // transform.print %attention {name = "Applied attention config"} : !transform.any_op
> causes a segment fault when compiling sdxl mlir (stable_diffusion_xl_base_1_0_punet_bs1_64_1024x1024_i8.mlir) .

You need to run with threading disabled for prints to work. `--mlir-disable-threading`


**COMMENTED** (2025-10-10)

Looks good, just two high level comments:
1. Can you specify it's for contraction ops in the PR description?
2. Have you checked that the new transform applies to the same ops and results in the same compilation info applied across all dispatches?


**COMMENTED** (2025-10-10)

Looks good overall, but I'd like to make sure we can drop the flux spec.

Also, we don't need the '(1/2)' in the PR title anymore, since there's no follow up PR for contraction matrchers


**COMMENTED** (2025-10-10)


**APPROVED** (2025-10-10)


### Code Comments

**File:** `tests/external/iree-test-suites/test_suite_files/attention_and_matmul_spec_punet_mi300.mlir:9`

```diff
@@ -6,7 +6,6 @@ module attributes { transform.with_named_sequence } {
   transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly},
                                             %config: !transform.any_param {transform.readonly}) {
     transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
-    // transform.print %op {name = "Applied"} : !transform.any_op
```

**Comment:**
Can you undo this? These prints are useful for local debugging and it's nice to able to just uncomment them.

---

**File:** `tests/external/iree-test-suites/test_suite_files/attention_and_matmul_spec_punet_mi300.mlir:18`

```diff
@@ -15,32 +14,9 @@ module attributes { transform.with_named_sequence } {
                                                  %decomposition_config: !transform.any_param {transform.readonly}) {
     transform.annotate %attention "compilation_info" = %config : !transform.any_op, !transform.any_param
     transform.annotate %attention "decomposition_config" = %decomposition_config : !transform.any_op, !transform.any_param
-    // transform.print %attention {name = "Applied attention config"} : !transform.any_op
```

**Comment:**
also here

---

**File:** `tests/external/iree-test-suites/test_suite_files/attention_and_matmul_spec_flux_mi300.mlir:1`

```diff
@@ -1,186 +0,0 @@
-module attributes {transform.with_named_sequence} {
```

**Comment:**
How do you know this is unused?

---

**File:** `tests/external/iree-test-suites/test_suite_files/attention_and_matmul_spec_flux_mi300.mlir:1`

```diff
@@ -1,186 +0,0 @@
-module attributes {transform.with_named_sequence} {
```

**Comment:**
I also checked iree-test-suites and I don't see it being used there either: https://github.com/search?q=repo%3Airee-org%2Firee-test-suites%20attention_and_matmul&type=code

---


---


## [PR #22227](https://github.com/iree-org/iree/pull/22227): [python] Set up binding for preprocessing transform ops

### Review Summary

**COMMENTED** (2025-10-07)

Looks good overall. Maybe reference the mlir PR you used as a reference in the PR description, so that we can track the overall design/mechanics to that original PR?


**COMMENTED** (2025-10-08)


**APPROVED** (2025-10-10)


### Code Comments

**File:** `compiler/bindings/python/test/ir/dialects_test.py:650`

```diff
@@ -602,3 +611,83 @@ def gpu_target_info_constructor_error_cases():
         assert False, "Expected TypeError for wrong MMA intrinsic object type"
     except TypeError:
         pass
+
+
+# ======================================================================
+# Preprocessing Transform Extensions
+# ======================================================================
+
+
+@run
+def preprocessing_transform_contraction_example():
+    ctx = ir.Context.current
+    module = ir.Module.parse(
+        """
+        #map_lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
+        #map_rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
+        #map_output = affine_map<(d0, d1, d2) -> (d0, d1)>
+        module attributes {transform.with_named_sequence} {
+            transform.named_sequence @match_matmul(
+                %module: !transform.any_op {transform.readonly})
+                -> !transform.any_op {
+                %matmuls = transform.structured.match ops{["linalg.matmul"]}
+                    in %module : (!transform.any_op) -> !transform.any_op
+                %batch, %m, %n, %k =
+                    transform.iree.match.contraction %matmuls,
+                    lhs_type = f32, rhs_type = f32, output_type = f32
+                    {indexing_maps = [#map_lhs, #map_rhs, #map_output]}
+                    : !transform.any_op -> !transform.param<i64>
+                transform.iree.match.dims_equal %batch, [] : !transform.param<i64>
+                transform.iree.match.dims_equal %m, [4096] : !transform.param<i64>
+                transform.iree.match.dims_equal %n, [2048] : !transform.param<i64>
+                transform.iree.match.dims_equal %k, [8192] : !transform.param<i64>
+                transform.yield %matmuls : !transform.any_op
+            }
+        }
+    """,
+        ctx,
+    )
+    assert module is not None
```

**Comment:**
I don't understand why we have this test case -- I think it will parse correctly no matter how we set up python bindings, won't it?

---

**File:** `compiler/bindings/python/test/ir/dialects_test.py:693`

```diff
@@ -602,3 +611,83 @@ def gpu_target_info_constructor_error_cases():
         assert False, "Expected TypeError for wrong MMA intrinsic object type"
     except TypeError:
         pass
+
+
+# ======================================================================
+# Preprocessing Transform Extensions
+# ======================================================================
+
+
+@run
+def preprocessing_transform_contraction_example():
+    ctx = ir.Context.current
+    module = ir.Module.parse(
+        """
+        #map_lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
+        #map_rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
+        #map_output = affine_map<(d0, d1, d2) -> (d0, d1)>
+        module attributes {transform.with_named_sequence} {
+            transform.named_sequence @match_matmul(
+                %module: !transform.any_op {transform.readonly})
+                -> !transform.any_op {
+                %matmuls = transform.structured.match ops{["linalg.matmul"]}
+                    in %module : (!transform.any_op) -> !transform.any_op
+                %batch, %m, %n, %k =
+                    transform.iree.match.contraction %matmuls,
+                    lhs_type = f32, rhs_type = f32, output_type = f32
+                    {indexing_maps = [#map_lhs, #map_rhs, #map_output]}
+                    : !transform.any_op -> !transform.param<i64>
+                transform.iree.match.dims_equal %batch, [] : !transform.param<i64>
+                transform.iree.match.dims_equal %m, [4096] : !transform.param<i64>
+                transform.iree.match.dims_equal %n, [2048] : !transform.param<i64>
+                transform.iree.match.dims_equal %k, [8192] : !transform.param<i64>
+                transform.yield %matmuls : !transform.any_op
+            }
+        }
+    """,
+        ctx,
+    )
+    assert module is not None
+
+
+@run
+def preprocessing_transform_match_contraction_in_named_sequence():
+    module_op = ir.Module.create()
+    module_op.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
+    map_lhs = ir.AffineMap.get(
+        dim_count=3,
+        symbol_count=0,
+        exprs=[ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(2)],
+    )
+    map_rhs = ir.AffineMap.get(
+        dim_count=3,
+        symbol_count=0,
+        exprs=[ir.AffineExpr.get_dim(2), ir.AffineExpr.get_dim(1)],
+    )
+    map_output = ir.AffineMap.get(
+        dim_count=3,
+        symbol_count=0,
+        exprs=[ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)],
+    )
+
+    with ir.InsertionPoint(module_op.body):
+        named_seq = transform.NamedSequenceOp(
+            "match_matmul", [transform.AnyOpType.get()], [transform.AnyOpType.get()]
+        )
+        with ir.InsertionPoint(named_seq.body):
+            batch, m, n, k = preprocessing_transform.MatchContractionOp(
+                operand_handle=named_seq.bodyTarget,
+                lhs_type=ir.F32Type.get(),
+                rhs_type=ir.F32Type.get(),
+                output_type=ir.F32Type.get(),
+                indexing_maps=[map_lhs, map_rhs, map_output],
+            )
+            transform.YieldOp([named_seq.bodyTarget])
+
+    module_str = str(module_op)
+    assert "affine_map<(d0, d1, d2) -> (d0, d2)>" in module_str
+    assert "affine_map<(d0, d1, d2) -> (d2, d1)>" in module_str
+    assert "affine_map<(d0, d1, d2) -> (d0, d1)>" in module_str
+    assert "transform.with_named_sequence" in module_str
+    assert "transform.named_sequence @match_matmul" in module_str
+    assert "transform.iree.match.contraction" in module_str
```

**Comment:**
Can we check that this contraction matcher has the expected indexing maps?

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td:239`

```diff
@@ -219,6 +219,7 @@ def MatchContractionOp : Op<Transform_Dialect, "iree.match.contraction",
     `,` `lhs_type` `=` $lhs_type
     `,` `rhs_type` `=` $rhs_type
     `,` `output_type` `=` $output_type
+    (`,` `indexing_maps` `=` $indexing_maps^)?
```

**Comment:**
Can we also use this syntax in all tests and add an op example in `let description = ` above?

---


---


## [PR #22201](https://github.com/iree-org/iree/pull/22201): [Codegen] Follow-up Fix for MatchContractionOp

### Review Summary

**APPROVED** (2025-10-03)



---


## [PR #22199](https://github.com/iree-org/iree/pull/22199): [Codegen] add transform op for matching attention op

### Review Summary

**COMMENTED** (2025-10-03)


**APPROVED** (2025-10-03)


### Code Comments

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td:377`

```diff
@@ -225,6 +225,66 @@ def MatchContractionOp : Op<Transform_Dialect, "iree.match.contraction",
   let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
 }
 
+def MatchAttentionOp : Op<Transform_Dialect, "iree.match.attention",
+    [MatchOpInterface,
+     SingleOpMatcher,
+     MemoryEffectsOpInterface,
+     AllTypesMatch<["batch_dims", "m_dims", "n_dims",
+                    "k1_dims", "k2_dims"]>
+     ]> {
+  let summary = [{Check whether the op is an attention operation.}];
+  let description = [{
+    Matches operations from the IREELinalgExt dialect that implement
+    attention: iree_linalg_ext.attention.
+
+    #### Return modes
+
+    Succeeds if the operation is an attention operation, and
+    produces a silenceable failure otherwise.
+
+    #### Results
+
+    Returns arrays of dimension sizes extracted from the iteration domain:
+    - batch_dims: Array of batch dimension sizes.
+    - m_dims: Array of query sequence length dimension sizes.
+    - n_dims: Array of number of heads dimension sizes.
+    - k1_dims: Array of key/value sequence length dimension sizes.
+    - k2_dims: Array of key embedding dimension sizes.
+
+    The exact interpretation depends on the indexing maps of the attention op.
```

**Comment:**
Can you also add an example?

---


---


## [PR #22194](https://github.com/iree-org/iree/pull/22194): [Codegen] Add transform op for matching convolution ops

### Review Summary

**COMMENTED** (2025-10-02)


**COMMENTED** (2025-10-03)

Could you move the contraction matcher syntax and error printing changes to a separate PR?


**COMMENTED** (2025-10-03)


**COMMENTED** (2025-10-03)


**COMMENTED** (2025-10-03)


**APPROVED** (2025-10-03)


**COMMENTED** (2025-10-03)


### Code Comments

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:318`

```diff
@@ -303,6 +303,108 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
   return DiagnosedSilenceableFailure::success();
 }
 
+//===----------------------------------------------------------------------===//
+// MatchConvolutionOp
+//===----------------------------------------------------------------------===//
+
+DiagnosedSilenceableFailure
+IREE::transform_dialect::MatchConvolutionOp::matchOperation(
+    Operation *current, transform::TransformResults &results,
+    transform::TransformState &state) {
+  Location loc = current->getLoc();
+  auto linalgOp = dyn_cast<linalg::LinalgOp>(current);
+  if (!linalgOp) {
+    return emitSilenceableFailure(loc)
+           << "Operation " << *current << " is not a LinalgOp.";
```

**Comment:**
Don't print the whole op, this can be very verbose. We should also remove that from the matcher for contractions.

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:323`

```diff
@@ -303,6 +303,108 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
   return DiagnosedSilenceableFailure::success();
 }
 
+//===----------------------------------------------------------------------===//
+// MatchConvolutionOp
+//===----------------------------------------------------------------------===//
+
+DiagnosedSilenceableFailure
+IREE::transform_dialect::MatchConvolutionOp::matchOperation(
+    Operation *current, transform::TransformResults &results,
+    transform::TransformState &state) {
+  Location loc = current->getLoc();
+  auto linalgOp = dyn_cast<linalg::LinalgOp>(current);
+  if (!linalgOp) {
+    return emitSilenceableFailure(loc)
+           << "Operation " << *current << " is not a LinalgOp.";
+  }
+
+  if (!linalg::isaConvolutionOpInterface(linalgOp)) {
+    return emitSilenceableFailure(loc)
+           << "Operation " << *current << " is not a convolution operation.";
```

**Comment:**
also here

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td:283`

```diff
@@ -225,6 +225,68 @@ def MatchContractionOp : Op<Transform_Dialect, "iree.match.contraction",
   let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
 }
 
+def MatchConvolutionOp : Op<Transform_Dialect, "iree.match.convolution",
+    [MatchOpInterface,
+     SingleOpMatcher,
+     MemoryEffectsOpInterface]> {
+  let summary = [{Check whether the op is a convolution operation.}];
+  let description = [{
+    Matches operations that implement the ConvolutionOpInterface.
+    This includes operations like linalg.conv_2d_nhwc_hwcf,
+    linalg.conv_2d_nchw_fchw, linalg.depthwise_conv_2d_nhwc_hwc, etc.
+
+    Optionally matches specific indexing maps patterns.
+
+    #### Return modes
+
+    Succeeds if the operation is a convolution operation, and
+    produces a silenceable failure otherwise.
+
+    #### Results
+
+    Returns arrays of dimension sizes for each convolution dimension:
+    - batch_dims: Array of batch dimension sizes.
+    - output_image_dims: Array of output spatial dimension sizes.
+    - output_channel_dims: Array of output channel dimension sizes.
+    - filter_dims: Array of filter spatial dimension sizes.
+    - input_channel_dims: Array of input channel dimension sizes.
+    - depth_dims: Array of depth dimension sizes (for depthwise convolutions).
+    - strides: Array of stride values.
+    - dilations: Array of dilation values.
+  }];
+
+  let arguments = (ins
+    TransformHandleTypeInterface:$operand_handle,
+    TypeAttr:$lhs_type,
+    TypeAttr:$rhs_type,
+    TypeAttr:$output_type,
+    OptionalAttr<AffineMapArrayAttr>:$indexing_maps
+  );
+
+  let results = (outs
+    TransformParamTypeInterface:$batch_dims,
+    TransformParamTypeInterface:$output_image_dims,
+    TransformParamTypeInterface:$output_channel_dims,
+    TransformParamTypeInterface:$filter_dims,
+    TransformParamTypeInterface:$input_channel_dims,
+    TransformParamTypeInterface:$depth_dims,
+    TransformParamTypeInterface:$strides,
+    TransformParamTypeInterface:$dilations
+  );
+
+  let assemblyFormat = [{
+    $operand_handle
+    `,` `lhs_type` `=` $lhs_type
+    `,` `rhs_type` `=` $rhs_type
+    `,` `output_type` `=` $output_type
+    (`indexing_maps` $indexing_maps^)?
+    attr-dict `:` functional-type(operands, results)
```

**Comment:**
Can we pick an assembly format that doesn't print `!transform.param<i64>` 8 times? This was still OK with contractions but with convs it got very verbose... Maybe we can do:
`!transform.any_op -> !transform.param<i64>`
?

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:245`

```diff
@@ -233,15 +233,16 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
     Operation *current, transform::TransformResults &results,
     transform::TransformState &state) {
   Location loc = current->getLoc();
+  StringRef opName = current->getName().getStringRef();
   auto linalgOp = dyn_cast<linalg::LinalgOp>(current);
   if (!linalgOp) {
     return emitSilenceableFailure(loc)
-           << "Operation " << *current << " is not a LinalgOp.";
+           << "Operation " << opName << " is not a LinalgOp.";
   }
 
   if (!linalg::isaContractionOpInterface(linalgOp)) {
     return emitSilenceableFailure(loc)
-           << "Operation " << *current << " is not a contraction operation.";
+           << "Operation " << opName << " is not a contraction operation.";
```

**Comment:**
I don't think we need to print the op name either -- we already pass in the location

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:326`

```diff
@@ -303,6 +304,109 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
   return DiagnosedSilenceableFailure::success();
 }
 
+//===----------------------------------------------------------------------===//
+// MatchConvolutionOp
+//===----------------------------------------------------------------------===//
+
+DiagnosedSilenceableFailure
+IREE::transform_dialect::MatchConvolutionOp::matchOperation(
+    Operation *current, transform::TransformResults &results,
+    transform::TransformState &state) {
+  Location loc = current->getLoc();
+  StringRef opName = current->getName().getStringRef();
+  auto linalgOp = dyn_cast<linalg::LinalgOp>(current);
+  if (!linalgOp) {
+    return emitSilenceableFailure(loc)
+           << "Operation " << opName << " is not a LinalgOp.";
+  }
+
+  if (!linalg::isaConvolutionOpInterface(linalgOp)) {
+    return emitSilenceableFailure(loc)
+           << "Operation " << opName << " is not a convolution operation.";
+  }
```

**Comment:**
Also here, we don't need the op name

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:371`

```diff
@@ -303,6 +304,109 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
   return DiagnosedSilenceableFailure::success();
 }
 
+//===----------------------------------------------------------------------===//
+// MatchConvolutionOp
+//===----------------------------------------------------------------------===//
+
+DiagnosedSilenceableFailure
+IREE::transform_dialect::MatchConvolutionOp::matchOperation(
+    Operation *current, transform::TransformResults &results,
+    transform::TransformState &state) {
+  Location loc = current->getLoc();
+  StringRef opName = current->getName().getStringRef();
+  auto linalgOp = dyn_cast<linalg::LinalgOp>(current);
+  if (!linalgOp) {
+    return emitSilenceableFailure(loc)
+           << "Operation " << opName << " is not a LinalgOp.";
+  }
+
+  if (!linalg::isaConvolutionOpInterface(linalgOp)) {
+    return emitSilenceableFailure(loc)
+           << "Operation " << opName << " is not a convolution operation.";
+  }
+
+  Type targetLhsType = getLhsType();
+  Type currentLhsType = getElementTypeOrSelf(linalgOp.getDpsInputs()[0]);
+  if (currentLhsType != targetLhsType) {
+    return emitSilenceableFailure(loc)
+           << "LHS type doesn't match: expected " << targetLhsType << ", got "
+           << currentLhsType;
+  }
+
+  Type targetRhsType = getRhsType();
+  Type currentRhsType = getElementTypeOrSelf(linalgOp.getDpsInputs()[1]);
+  if (currentRhsType != targetRhsType) {
+    return emitSilenceableFailure(loc)
+           << "RHS type doesn't match: expected " << targetRhsType << ", got "
+           << currentRhsType;
+  }
+
+  Type targetOutputType = getOutputType();
+  Type currentOutputType =
+      getElementTypeOrSelf(linalgOp.getDpsInits()[0].getType());
+  if (currentOutputType != targetOutputType) {
+    return emitSilenceableFailure(loc)
+           << "output type doesn't match: expected " << targetOutputType
+           << ", got " << currentOutputType;
+  }
+
+  ArrayAttr currentIndexingMaps = linalgOp.getIndexingMaps();
+  if (std::optional<ArrayAttr> targetIndexingMaps = getIndexingMaps()) {
+    if (currentIndexingMaps != *targetIndexingMaps) {
+      return emitSilenceableFailure(loc) << "indexing maps don't match";
+    }
+  }
+
+  FailureOr<linalg::ConvolutionDimensions> maybeConvDims =
+      linalg::inferConvolutionDims(linalgOp);
+  if (failed(maybeConvDims)) {
+    return emitSilenceableFailure(loc)
+           << "Failed to infer convolution dimensions.";
+  }
+  linalg::ConvolutionDimensions convDims = *maybeConvDims;
+  SmallVector<int64_t> iterationDomain = linalgOp.getStaticLoopRanges();
+  MLIRContext *ctx = getContext();
+  Builder builder(ctx);
+
+  auto buildI64Attrs = [&](auto values, auto transform) {
```

**Comment:**
We should avoid copying vectors inside convDims: https://llvm.org/docs/CodingStandards.html#beware-unnecessary-copies-with-auto
```suggestion
  auto buildI64Attrs = [&builder](const auto& values, const auto& transform) {
```

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:369`

```diff
@@ -303,6 +304,109 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
   return DiagnosedSilenceableFailure::success();
 }
 
+//===----------------------------------------------------------------------===//
+// MatchConvolutionOp
+//===----------------------------------------------------------------------===//
+
+DiagnosedSilenceableFailure
+IREE::transform_dialect::MatchConvolutionOp::matchOperation(
+    Operation *current, transform::TransformResults &results,
+    transform::TransformState &state) {
+  Location loc = current->getLoc();
+  StringRef opName = current->getName().getStringRef();
+  auto linalgOp = dyn_cast<linalg::LinalgOp>(current);
+  if (!linalgOp) {
+    return emitSilenceableFailure(loc)
+           << "Operation " << opName << " is not a LinalgOp.";
+  }
+
+  if (!linalg::isaConvolutionOpInterface(linalgOp)) {
+    return emitSilenceableFailure(loc)
+           << "Operation " << opName << " is not a convolution operation.";
+  }
+
+  Type targetLhsType = getLhsType();
+  Type currentLhsType = getElementTypeOrSelf(linalgOp.getDpsInputs()[0]);
+  if (currentLhsType != targetLhsType) {
+    return emitSilenceableFailure(loc)
+           << "LHS type doesn't match: expected " << targetLhsType << ", got "
+           << currentLhsType;
+  }
+
+  Type targetRhsType = getRhsType();
+  Type currentRhsType = getElementTypeOrSelf(linalgOp.getDpsInputs()[1]);
+  if (currentRhsType != targetRhsType) {
+    return emitSilenceableFailure(loc)
+           << "RHS type doesn't match: expected " << targetRhsType << ", got "
+           << currentRhsType;
+  }
+
+  Type targetOutputType = getOutputType();
+  Type currentOutputType =
+      getElementTypeOrSelf(linalgOp.getDpsInits()[0].getType());
+  if (currentOutputType != targetOutputType) {
+    return emitSilenceableFailure(loc)
+           << "output type doesn't match: expected " << targetOutputType
+           << ", got " << currentOutputType;
+  }
+
+  ArrayAttr currentIndexingMaps = linalgOp.getIndexingMaps();
+  if (std::optional<ArrayAttr> targetIndexingMaps = getIndexingMaps()) {
+    if (currentIndexingMaps != *targetIndexingMaps) {
+      return emitSilenceableFailure(loc) << "indexing maps don't match";
+    }
+  }
+
+  FailureOr<linalg::ConvolutionDimensions> maybeConvDims =
+      linalg::inferConvolutionDims(linalgOp);
+  if (failed(maybeConvDims)) {
+    return emitSilenceableFailure(loc)
+           << "Failed to infer convolution dimensions.";
+  }
+  linalg::ConvolutionDimensions convDims = *maybeConvDims;
+  SmallVector<int64_t> iterationDomain = linalgOp.getStaticLoopRanges();
+  MLIRContext *ctx = getContext();
+  Builder builder(ctx);
+
+  auto buildI64Attrs = [&](auto values, auto transform) {
+    return llvm::map_to_vector(values, [&](auto val) -> Attribute {
```

**Comment:**
```suggestion
    return llvm::map_to_vector(values, [&](unsigned val) -> Attribute {
```

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:369`

```diff
@@ -303,6 +304,109 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
   return DiagnosedSilenceableFailure::success();
 }
 
+//===----------------------------------------------------------------------===//
+// MatchConvolutionOp
+//===----------------------------------------------------------------------===//
+
+DiagnosedSilenceableFailure
+IREE::transform_dialect::MatchConvolutionOp::matchOperation(
+    Operation *current, transform::TransformResults &results,
+    transform::TransformState &state) {
+  Location loc = current->getLoc();
+  StringRef opName = current->getName().getStringRef();
+  auto linalgOp = dyn_cast<linalg::LinalgOp>(current);
+  if (!linalgOp) {
+    return emitSilenceableFailure(loc)
+           << "Operation " << opName << " is not a LinalgOp.";
+  }
+
+  if (!linalg::isaConvolutionOpInterface(linalgOp)) {
+    return emitSilenceableFailure(loc)
+           << "Operation " << opName << " is not a convolution operation.";
+  }
+
+  Type targetLhsType = getLhsType();
+  Type currentLhsType = getElementTypeOrSelf(linalgOp.getDpsInputs()[0]);
+  if (currentLhsType != targetLhsType) {
+    return emitSilenceableFailure(loc)
+           << "LHS type doesn't match: expected " << targetLhsType << ", got "
+           << currentLhsType;
+  }
+
+  Type targetRhsType = getRhsType();
+  Type currentRhsType = getElementTypeOrSelf(linalgOp.getDpsInputs()[1]);
+  if (currentRhsType != targetRhsType) {
+    return emitSilenceableFailure(loc)
+           << "RHS type doesn't match: expected " << targetRhsType << ", got "
+           << currentRhsType;
+  }
+
+  Type targetOutputType = getOutputType();
+  Type currentOutputType =
+      getElementTypeOrSelf(linalgOp.getDpsInits()[0].getType());
+  if (currentOutputType != targetOutputType) {
+    return emitSilenceableFailure(loc)
+           << "output type doesn't match: expected " << targetOutputType
+           << ", got " << currentOutputType;
+  }
+
+  ArrayAttr currentIndexingMaps = linalgOp.getIndexingMaps();
+  if (std::optional<ArrayAttr> targetIndexingMaps = getIndexingMaps()) {
+    if (currentIndexingMaps != *targetIndexingMaps) {
+      return emitSilenceableFailure(loc) << "indexing maps don't match";
+    }
+  }
+
+  FailureOr<linalg::ConvolutionDimensions> maybeConvDims =
+      linalg::inferConvolutionDims(linalgOp);
+  if (failed(maybeConvDims)) {
+    return emitSilenceableFailure(loc)
+           << "Failed to infer convolution dimensions.";
+  }
+  linalg::ConvolutionDimensions convDims = *maybeConvDims;
+  SmallVector<int64_t> iterationDomain = linalgOp.getStaticLoopRanges();
+  MLIRContext *ctx = getContext();
+  Builder builder(ctx);
+
+  auto buildI64Attrs = [&](auto values, auto transform) {
+    return llvm::map_to_vector(values, [&](auto val) -> Attribute {
```

**Comment:**
ah no, this can be int64_t for dilations, disregard

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:377`

```diff
@@ -303,6 +303,107 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
   return DiagnosedSilenceableFailure::success();
 }
 
+//===----------------------------------------------------------------------===//
+// MatchConvolutionOp
+//===----------------------------------------------------------------------===//
+
+DiagnosedSilenceableFailure
+IREE::transform_dialect::MatchConvolutionOp::matchOperation(
+    Operation *current, transform::TransformResults &results,
+    transform::TransformState &state) {
+  Location loc = current->getLoc();
+  auto linalgOp = dyn_cast<linalg::LinalgOp>(current);
+  if (!linalgOp) {
+    return emitSilenceableFailure(loc) << "Operation is not a LinalgOp.";
+  }
+
+  if (!linalg::isaConvolutionOpInterface(linalgOp)) {
+    return emitSilenceableFailure(loc)
+           << "Operation is not a convolution operation.";
+  }
+
+  Type targetLhsType = getLhsType();
+  Type currentLhsType = getElementTypeOrSelf(linalgOp.getDpsInputs()[0]);
+  if (currentLhsType != targetLhsType) {
+    return emitSilenceableFailure(loc)
+           << "LHS type doesn't match: expected " << targetLhsType << ", got "
+           << currentLhsType;
+  }
+
+  Type targetRhsType = getRhsType();
+  Type currentRhsType = getElementTypeOrSelf(linalgOp.getDpsInputs()[1]);
+  if (currentRhsType != targetRhsType) {
+    return emitSilenceableFailure(loc)
+           << "RHS type doesn't match: expected " << targetRhsType << ", got "
+           << currentRhsType;
+  }
+
+  Type targetOutputType = getOutputType();
+  Type currentOutputType =
+      getElementTypeOrSelf(linalgOp.getDpsInits()[0].getType());
+  if (currentOutputType != targetOutputType) {
+    return emitSilenceableFailure(loc)
+           << "output type doesn't match: expected " << targetOutputType
+           << ", got " << currentOutputType;
+  }
+
+  ArrayAttr currentIndexingMaps = linalgOp.getIndexingMaps();
+  if (std::optional<ArrayAttr> targetIndexingMaps = getIndexingMaps()) {
+    if (currentIndexingMaps != *targetIndexingMaps) {
+      return emitSilenceableFailure(loc) << "indexing maps don't match";
+    }
+  }
+
+  FailureOr<linalg::ConvolutionDimensions> maybeConvDims =
+      linalg::inferConvolutionDims(linalgOp);
+  if (failed(maybeConvDims)) {
+    return emitSilenceableFailure(loc)
+           << "Failed to infer convolution dimensions.";
+  }
+  linalg::ConvolutionDimensions convDims = *maybeConvDims;
+  SmallVector<int64_t> iterationDomain = linalgOp.getStaticLoopRanges();
+  MLIRContext *ctx = getContext();
+  Builder builder(ctx);
+
+  auto buildI64Attrs = [&builder](const auto &values, const auto &transform) {
+    return llvm::map_to_vector(values, [&](auto val) -> Attribute {
+      return builder.getI64IntegerAttr(transform(val));
+    });
+  };
+
+  results.setParams(cast<OpResult>(getBatchDims()),
+                    buildI64Attrs(convDims.batch, [&](unsigned idx) {
+                      return iterationDomain[idx];
+                    }));
```

**Comment:**
Can you hoist this lambda to a local variable? It's used 6 time total

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:366`

```diff
@@ -303,6 +303,107 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
   return DiagnosedSilenceableFailure::success();
 }
 
+//===----------------------------------------------------------------------===//
+// MatchConvolutionOp
+//===----------------------------------------------------------------------===//
+
+DiagnosedSilenceableFailure
+IREE::transform_dialect::MatchConvolutionOp::matchOperation(
+    Operation *current, transform::TransformResults &results,
+    transform::TransformState &state) {
+  Location loc = current->getLoc();
+  auto linalgOp = dyn_cast<linalg::LinalgOp>(current);
+  if (!linalgOp) {
+    return emitSilenceableFailure(loc) << "Operation is not a LinalgOp.";
+  }
+
+  if (!linalg::isaConvolutionOpInterface(linalgOp)) {
+    return emitSilenceableFailure(loc)
+           << "Operation is not a convolution operation.";
+  }
+
+  Type targetLhsType = getLhsType();
+  Type currentLhsType = getElementTypeOrSelf(linalgOp.getDpsInputs()[0]);
+  if (currentLhsType != targetLhsType) {
+    return emitSilenceableFailure(loc)
+           << "LHS type doesn't match: expected " << targetLhsType << ", got "
+           << currentLhsType;
+  }
+
+  Type targetRhsType = getRhsType();
+  Type currentRhsType = getElementTypeOrSelf(linalgOp.getDpsInputs()[1]);
+  if (currentRhsType != targetRhsType) {
+    return emitSilenceableFailure(loc)
+           << "RHS type doesn't match: expected " << targetRhsType << ", got "
+           << currentRhsType;
+  }
+
+  Type targetOutputType = getOutputType();
+  Type currentOutputType =
+      getElementTypeOrSelf(linalgOp.getDpsInits()[0].getType());
+  if (currentOutputType != targetOutputType) {
+    return emitSilenceableFailure(loc)
+           << "output type doesn't match: expected " << targetOutputType
+           << ", got " << currentOutputType;
+  }
+
+  ArrayAttr currentIndexingMaps = linalgOp.getIndexingMaps();
+  if (std::optional<ArrayAttr> targetIndexingMaps = getIndexingMaps()) {
+    if (currentIndexingMaps != *targetIndexingMaps) {
+      return emitSilenceableFailure(loc) << "indexing maps don't match";
+    }
+  }
+
+  FailureOr<linalg::ConvolutionDimensions> maybeConvDims =
+      linalg::inferConvolutionDims(linalgOp);
+  if (failed(maybeConvDims)) {
+    return emitSilenceableFailure(loc)
+           << "Failed to infer convolution dimensions.";
+  }
+  linalg::ConvolutionDimensions convDims = *maybeConvDims;
+  SmallVector<int64_t> iterationDomain = linalgOp.getStaticLoopRanges();
+  MLIRContext *ctx = getContext();
+  Builder builder(ctx);
```

**Comment:**
```suggestion
  Builder builder(getContext());
```
`ctx` is not used anywhere else AFAICT

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td:259`

```diff
@@ -225,6 +225,72 @@ def MatchContractionOp : Op<Transform_Dialect, "iree.match.contraction",
   let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
 }
 
+def MatchConvolutionOp : Op<Transform_Dialect, "iree.match.convolution",
+    [MatchOpInterface,
+     SingleOpMatcher,
+     MemoryEffectsOpInterface,
+     AllTypesMatch<["batch_dims", "output_image_dims", "output_channel_dims",
+                    "filter_dims", "input_channel_dims", "depth_dims",
+                    "strides", "dilations"]>
+     ]> {
+  let summary = [{Check whether the op is a convolution operation.}];
+  let description = [{
+    Matches operations that implement the ConvolutionOpInterface.
+    This includes operations like linalg.conv_2d_nhwc_hwcf,
+    linalg.conv_2d_nchw_fchw, linalg.depthwise_conv_2d_nhwc_hwc, etc.
+
+    Optionally matches specific indexing maps patterns.
+
+    #### Return modes
+
+    Succeeds if the operation is a convolution operation, and
+    produces a silenceable failure otherwise.
+
+    #### Results
+
+    Returns arrays of dimension sizes for each convolution dimension:
+    - batch_dims: Array of batch dimension sizes.
+    - output_image_dims: Array of output spatial dimension sizes.
+    - output_channel_dims: Array of output channel dimension sizes.
+    - filter_dims: Array of filter spatial dimension sizes.
+    - input_channel_dims: Array of input channel dimension sizes.
+    - depth_dims: Array of depth dimension sizes (for depthwise convolutions).
+    - strides: Array of stride values.
+    - dilations: Array of dilation values.
```

**Comment:**
Can you also add an example?

---


---


## [PR #22178](https://github.com/iree-org/iree/pull/22178): [Codegen][GPU] Disable MMA Intrinsics Sorting

### Review Summary

**COMMENTED** (2025-10-02)


**COMMENTED** (2025-10-02)


**COMMENTED** (2025-10-02)


**APPROVED** (2025-10-02)

LGTM but please add a brief description of why we need to delete this to the PR description, not just the issue number


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.cpp:641`

```diff
@@ -636,11 +636,9 @@ FailureOr<GPUMMASchedule> deduceMMASchedule(
     int64_t subgroupSize, std::optional<int64_t> wgpCount, bool transposedLhs,
     bool transposedRhs, bool canUpcastAcc, bool mustBeAligned,
     bool doCPromotion) {
-
-  SmallVector<GPUIntrinsicType> sortedIntrinsics =
-      sortMMAIntrinsics(problem, intrinsics);
-
-  for (const GPUIntrinsicType &intrinsic : sortedIntrinsics) {
+  // TODO(#22160): sortMMAIntrinsics call is disabled for now since it causes
+  // performance regression. Re-enable once the issue is addressed.
+  for (const GPUIntrinsicType &intrinsic : intrinsics) {
```

**Comment:**
We should also remove the sorting functions. Just commenting out the code will lead to unused private function warnings.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.cpp:580`

```diff
@@ -571,17 +571,6 @@ static bool compareIntrinsics(const GPUMatmulShapeType &problem,
          ShapedType::getNumElements(rhs.kSizes);
 }
 
-static SmallVector<GPUIntrinsicType>
-sortMMAIntrinsics(GPUMatmulShapeType problem,
-                  ArrayRef<GPUIntrinsicType> intrinsics) {
-  SmallVector<GPUIntrinsicType> sortedIntrinsics(intrinsics);
-  llvm::stable_sort(sortedIntrinsics, [&](const GPUMatmulShapeType &lhs,
-                                          const GPUMatmulShapeType &rhs) {
-    return compareIntrinsics(problem, lhs, rhs);
```

**Comment:**
The comparison function also needs to be removed

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.cpp:580`

```diff
@@ -571,17 +571,6 @@ static bool compareIntrinsics(const GPUMatmulShapeType &problem,
          ShapedType::getNumElements(rhs.kSizes);
 }
 
-static SmallVector<GPUIntrinsicType>
-sortMMAIntrinsics(GPUMatmulShapeType problem,
-                  ArrayRef<GPUIntrinsicType> intrinsics) {
-  SmallVector<GPUIntrinsicType> sortedIntrinsics(intrinsics);
-  llvm::stable_sort(sortedIntrinsics, [&](const GPUMatmulShapeType &lhs,
-                                          const GPUMatmulShapeType &rhs) {
-    return compareIntrinsics(problem, lhs, rhs);
```

**Comment:**
Gotcha. BTW, the code where this is used performs another unstable sorting -- I wonder if that's also asking for trouble

---


---


## [PR #22154](https://github.com/iree-org/iree/pull/22154): [DispatchCreation] infer split-reduction sizes for ArgCompare

### Review Summary

**COMMENTED** (2025-10-01)


**COMMENTED** (2025-10-01)


**COMMENTED** (2025-10-01)


**COMMENTED** (2025-10-01)


**APPROVED** (2025-10-01)

LGTM % one minor issue


### Code Comments

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp:1248`

```diff
@@ -1220,6 +1220,34 @@ LogicalResult ArgCompareOp::reifyResultShapes(
       .reifyResultShapes(b, reifiedReturnShapes);
 }
 
+SmallVector<AffineMap>
+IREE::LinalgExt::ArgCompareOp::getIndexingMapsForOperands() {
+  Builder b(getContext());
+  const int64_t rank = getInputRank();
+  return SmallVector<AffineMap>{b.getMultiDimIdentityMap(rank)};
+}
+
+SmallVector<AffineMap>
+IREE::LinalgExt::ArgCompareOp::getIndexingMapsForResults() {
+  MLIRContext *ctx = getContext();
+  const int64_t rank = getInputRank();
+  const int64_t redDim = static_cast<int64_t>(getDimension());
+
+  SmallVector<AffineExpr> proj;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == redDim)
+      continue;
+    proj.push_back(getAffineDimExpr(i, ctx));
+  }
+
+  AffineMap resultMap = AffineMap::get(rank, 0, proj, ctx);
+  return SmallVector<AffineMap>{resultMap, resultMap};
+}
+
+SmallVector<int64_t> IREE::LinalgExt::ArgCompareOp::getStaticLoopRanges() {
+  return SmallVector<int64_t>(getInputType().getShape());
```

**Comment:**
nit: you can use brace initialization
```suggestion
  return {getInputType().getShape()};
```

---

**File:** `compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp:45`

```diff
@@ -29,6 +29,28 @@ static SmallVector<int64_t> getStaticReductionDimSizes(linalg::LinalgOp op) {
   return dimSizes;
 }
 
+static std::optional<SmallVector<int64_t>> getReductionDimSizes(Operation *Op) {
+  SmallVector<int64_t> loopRanges;
+  if (auto fusionOp = dyn_cast<IREE::LinalgExt::LinalgFusionOpInterface>(Op)) {
+    loopRanges = fusionOp.getStaticLoopRanges();
+  }
+
+  auto tilingInterfaceOp = dyn_cast<TilingInterface>(Op);
+  if (!tilingInterfaceOp) {
+    LDBG() << "skipping op; not a TilingInterface op";
+    return std::nullopt;
+  }
+
+  SmallVector<utils::IteratorType> iters;
+  iters = tilingInterfaceOp.getLoopIteratorTypes();
```

**Comment:**
```suggestion
  SmallVector<utils::IteratorType> iters = tilingInterfaceOp.getLoopIteratorTypes();
```

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp:1227`

```diff
@@ -1220,6 +1220,34 @@ LogicalResult ArgCompareOp::reifyResultShapes(
       .reifyResultShapes(b, reifiedReturnShapes);
 }
 
+SmallVector<AffineMap>
+IREE::LinalgExt::ArgCompareOp::getIndexingMapsForOperands() {
+  Builder b(getContext());
+  const int64_t rank = getInputRank();
+  return SmallVector<AffineMap>{b.getMultiDimIdentityMap(rank)};
```

**Comment:**
nit: this is only used in one place, I'd inline it and use brace initialization
```suggestion
  return {b.getMultiDimIdentityMap(getInputRank())};
```

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp:1244`

```diff
@@ -1220,6 +1220,34 @@ LogicalResult ArgCompareOp::reifyResultShapes(
       .reifyResultShapes(b, reifiedReturnShapes);
 }
 
+SmallVector<AffineMap>
+IREE::LinalgExt::ArgCompareOp::getIndexingMapsForOperands() {
+  Builder b(getContext());
+  const int64_t rank = getInputRank();
+  return SmallVector<AffineMap>{b.getMultiDimIdentityMap(rank)};
+}
+
+SmallVector<AffineMap>
+IREE::LinalgExt::ArgCompareOp::getIndexingMapsForResults() {
+  MLIRContext *ctx = getContext();
+  const int64_t rank = getInputRank();
+  const int64_t redDim = static_cast<int64_t>(getDimension());
+
+  SmallVector<AffineExpr> proj;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == redDim)
+      continue;
+    proj.push_back(getAffineDimExpr(i, ctx));
+  }
+
+  AffineMap resultMap = AffineMap::get(rank, 0, proj, ctx);
+  return SmallVector<AffineMap>{resultMap, resultMap};
```

**Comment:**
nit: you can return an initializer list
```suggestion
  AffineMap resultMap = AffineMap::get(rank, 0, proj, ctx);
  return {resultMap, resultMap};
```

---

**File:** `compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp:36`

```diff
@@ -29,6 +29,29 @@ static SmallVector<int64_t> getStaticReductionDimSizes(linalg::LinalgOp op) {
   return dimSizes;
 }
 
+static std::optional<SmallVector<int64_t>> getReductionDimSizes(Operation *Op) {
+  SmallVector<int64_t> loopRanges;
+  if (auto fusionOp = dyn_cast<IREE::LinalgExt::LinalgFusionOpInterface>(Op)) {
+    loopRanges = fusionOp.getStaticLoopRanges();
+  }
```

**Comment:**
Would it be possible to have a test that covers this?

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp:1248`

```diff
@@ -1220,6 +1220,34 @@ LogicalResult ArgCompareOp::reifyResultShapes(
       .reifyResultShapes(b, reifiedReturnShapes);
 }
 
+SmallVector<AffineMap>
+IREE::LinalgExt::ArgCompareOp::getIndexingMapsForOperands() {
+  Builder b(getContext());
+  const int64_t rank = getInputRank();
+  return SmallVector<AffineMap>{b.getMultiDimIdentityMap(rank)};
+}
+
+SmallVector<AffineMap>
+IREE::LinalgExt::ArgCompareOp::getIndexingMapsForResults() {
+  MLIRContext *ctx = getContext();
+  const int64_t rank = getInputRank();
+  const int64_t redDim = static_cast<int64_t>(getDimension());
+
+  SmallVector<AffineExpr> proj;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == redDim)
+      continue;
+    proj.push_back(getAffineDimExpr(i, ctx));
+  }
+
+  AffineMap resultMap = AffineMap::get(rank, 0, proj, ctx);
+  return SmallVector<AffineMap>{resultMap, resultMap};
+}
+
+SmallVector<int64_t> IREE::LinalgExt::ArgCompareOp::getStaticLoopRanges() {
+  return SmallVector<int64_t>(getInputType().getShape());
```

**Comment:**
the alternative is to use `llvm::to_vector(someArrayRef)` -- you don't need the exact type. But fine to keep as-is.

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp:1238`

```diff
@@ -1220,6 +1220,33 @@ LogicalResult ArgCompareOp::reifyResultShapes(
       .reifyResultShapes(b, reifiedReturnShapes);
 }
 
+SmallVector<AffineMap>
+IREE::LinalgExt::ArgCompareOp::getIndexingMapsForOperands() {
+  Builder b(getContext());
+  return {b.getMultiDimIdentityMap(getInputRank())};
+}
+
+SmallVector<AffineMap>
+IREE::LinalgExt::ArgCompareOp::getIndexingMapsForResults() {
+  MLIRContext *ctx = getContext();
+  const int64_t rank = getInputRank();
+  const int64_t redDim = static_cast<int64_t>(getDimension());
+
+  SmallVector<AffineExpr> proj;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == redDim)
+      continue;
```

**Comment:**
missing braces, see https://iree.dev/developers/general/contributing/#compiler

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp:1225`

```diff
@@ -1220,6 +1220,43 @@ LogicalResult ArgCompareOp::reifyResultShapes(
       .reifyResultShapes(b, reifiedReturnShapes);
 }
 
+SmallVector<AffineMap> IREE::LinalgExt::ArgCompareOp::getIndexingMapsArray() {
+  Builder b(getContext());
+  MLIRContext *ctx = b.getContext();
```

**Comment:**
Move `b` closer to the first use and use the `ctx` variable

---


---


## [PR #22149](https://github.com/iree-org/iree/pull/22149): [Codegen] support matching any values for dims_equal transform op

### Review Summary

**CHANGES_REQUESTED** (2025-09-30)

If you want to replicate what the old tuning spec was doing in https://github.com/nod-ai/sdxl-scripts/blob/4fa7ccbc3de4873c0751a48ef9b4b86a7f24428e/int8-model/specs/attention_and_matmul_spec.mlir#L636-L637 , your testcase should use static tensors only. And in your dims_equal implementation, you can teach it to treat -1 as 'match any value'.

We can still allow for dynamic dims on the input IR side, but the priority is to support any values as the match target. 

You can see how the `match_cast_compatible_types` matcher handled this here: https://github.com/iree-org/iree/blob/e7bd805ccea762c462069d47f4ea5be364636168/compiler/src/iree/compiler/Utils/ShapeUtils.cpp#L41-L65 and
https://github.com/iree-org/iree/blob/e7bd805ccea762c462069d47f4ea5be364636168/compiler/src/iree/compiler/Utils/ShapeUtils.cpp#L54-L57


**COMMENTED** (2025-09-30)


**COMMENTED** (2025-09-30)


**COMMENTED** (2025-10-01)


**COMMENTED** (2025-10-01)


**COMMENTED** (2025-10-01)


**COMMENTED** (2025-10-01)


**COMMENTED** (2025-10-01)


**COMMENTED** (2025-10-01)


**COMMENTED** (2025-10-01)


**COMMENTED** (2025-10-01)


**APPROVED** (2025-10-01)

LGTM % nit


### Code Comments

**File:** `compiler/src/iree/compiler/Preprocessing/Common/test/preprocessing_match_ops.mlir:716`

```diff
@@ -644,3 +644,136 @@ module attributes {transform.with_named_sequence} {
     transform.yield
   }
 }
+
+// -----
+
+// CHECK-LABEL: func.func @op_broadcast_rhs_mmt_d0
+func.func @op_broadcast_rhs_mmt_d0(
+  %lhs: tensor<4x8x512xi8>,
+  %rhs: tensor<1024x512xi8>,
+  %out: tensor<4x8x1024xi32>)
+  -> tensor<4x8x1024xi32> {
+  // CHECK-NEXT: linalg.generic
+  // CHECK-SAME:   match_status = "matched"
+  %res = linalg.generic
+    { indexing_maps = [
+        affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
+        affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
+        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
+      ],
+      iterator_types = ["parallel", "parallel", "parallel", "reduction"], match_status = "unmatched"}
+    ins(%lhs, %rhs : tensor<4x8x512xi8>, tensor<1024x512xi8>)
+    outs(%out : tensor<4x8x1024xi32>) {
+  ^bb0(%in_l: i8, %in_r: i8, %acc: i32):
+    %l = arith.extsi %in_l : i8 to i32
+    %r = arith.extsi %in_r : i8 to i32
+    %m = arith.muli %l, %r : i32
+    %a = arith.addi %acc, %m : i32
+    linalg.yield %a : i32
+  } -> tensor<4x8x1024xi32>
+  return %res : tensor<4x8x1024xi32>
+}
+
+module attributes {transform.with_named_sequence} {
+  transform.named_sequence @match_broadcast_rhs_mmt_i8_i8_i32_d0(
+    %op: !transform.any_op {transform.readonly}) -> !transform.any_op {
+
+    %batch, %m, %n, %k = transform.iree.match.contraction %op,
+      lhs_type = i8, rhs_type = i8, output_type = i32
+      { indexing_maps = [
+          affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
+          affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
+          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
+        ] } :
+      (!transform.any_op)
+      -> (!transform.param<i64>, !transform.param<i64>,
+          !transform.param<i64>, !transform.param<i64>)
+
+    transform.iree.match.dims_equal %batch, []      : !transform.param<i64>
+    transform.iree.match.dims_equal %m, [-1, 8]     : !transform.param<i64>
+    transform.iree.match.dims_equal %n, [1024]      : !transform.param<i64>
+    transform.iree.match.dims_equal %k, [512]       : !transform.param<i64>
+
+    transform.yield %op : !transform.any_op
+  }
+
+  transform.named_sequence @annotate(%op: !transform.any_op {transform.readonly}) {
+    %0 = transform.param.constant "matched" -> !transform.any_param
+    transform.annotate %op "match_status" = %0 : !transform.any_op, !transform.any_param
+    transform.yield
+  }
+
+  transform.named_sequence @__transform_main(%module: !transform.any_op) {
+    transform.foreach_match in %module
+        @match_broadcast_rhs_mmt_i8_i8_i32_d0 -> @annotate
+      : (!transform.any_op) -> (!transform.any_op)
+    transform.yield
+  }
+}
+
+// -----
+
+// CHECK-LABEL: func.func @op_broadcast_lhs_mmt_d0
```

**Comment:**
Instead of having each test cases in a different split, could we have multiple ops to match in the same function? These tests are currently quite verbose.

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:339`

```diff
@@ -314,9 +314,29 @@ DiagnosedSilenceableFailure IREE::transform_dialect::MatchDimsEqualOp::apply(
       state.getParams(getDimensionSizes());
   ArrayAttr targetDimensionSizes = getExpectedValues();
 
-  if (!llvm::equal(currentDimSizes, targetDimensionSizes)) {
+  if (currentDimSizes.size() != targetDimensionSizes.size()) {
     return emitSilenceableError()
-           << "Dimension sizes do not match expected values";
+           << "dimension sizes and expected values have different lengths";
+  }
+
+  for (auto [currentDimSizeAttr, targetDimSizeAttr] :
+       llvm::zip_equal(currentDimSizes, targetDimensionSizes)) {
+    auto currentDimSizeIntegerAttr = dyn_cast<IntegerAttr>(currentDimSizeAttr);
+    auto targetDimSizeIntegerAttr = dyn_cast<IntegerAttr>(targetDimSizeAttr);
+    if (!currentDimSizeIntegerAttr || !targetDimSizeIntegerAttr) {
+      return emitSilenceableError() << "expected integer attributes";
+    }
+
+    int64_t currentDimSize = currentDimSizeIntegerAttr.getInt();
+    int64_t targetDimSize = targetDimSizeIntegerAttr.getInt();
+
+    if (targetDimSize == -1)
+      continue;
+
+    if (currentDimSize != targetDimSize) {
+      return emitSilenceableError() << "dimension value " << currentDimSize
+                                    << " does not match " << targetDimSize;
+    }
```

**Comment:**
It would be simpler to have a helper lambda/function that takes a range of attrs and returns something like `FailureOr<SmallVector<int64_t>>`. Then you don't have to repeat the same extraction logic for lhs and rhs and can do the comparison with a simple `!=`.

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:339`

```diff
@@ -314,9 +314,29 @@ DiagnosedSilenceableFailure IREE::transform_dialect::MatchDimsEqualOp::apply(
       state.getParams(getDimensionSizes());
   ArrayAttr targetDimensionSizes = getExpectedValues();
 
-  if (!llvm::equal(currentDimSizes, targetDimensionSizes)) {
+  if (currentDimSizes.size() != targetDimensionSizes.size()) {
     return emitSilenceableError()
-           << "Dimension sizes do not match expected values";
+           << "dimension sizes and expected values have different lengths";
+  }
+
+  for (auto [currentDimSizeAttr, targetDimSizeAttr] :
+       llvm::zip_equal(currentDimSizes, targetDimensionSizes)) {
+    auto currentDimSizeIntegerAttr = dyn_cast<IntegerAttr>(currentDimSizeAttr);
+    auto targetDimSizeIntegerAttr = dyn_cast<IntegerAttr>(targetDimSizeAttr);
+    if (!currentDimSizeIntegerAttr || !targetDimSizeIntegerAttr) {
+      return emitSilenceableError() << "expected integer attributes";
+    }
+
+    int64_t currentDimSize = currentDimSizeIntegerAttr.getInt();
+    int64_t targetDimSize = targetDimSizeIntegerAttr.getInt();
+
+    if (targetDimSize == -1)
+      continue;
+
+    if (currentDimSize != targetDimSize) {
+      return emitSilenceableError() << "dimension value " << currentDimSize
+                                    << " does not match " << targetDimSize;
+    }
```

**Comment:**
Or alternatively map to vector of `std::optional<int64_t>`, I think that's even simpler

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:357`

```diff
@@ -307,16 +307,54 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
 // MatchDimsEqualOp
 //===----------------------------------------------------------------------===//
 
+template <typename Range, typename Mapper>
+static FailureOr<SmallVector<std::optional<int64_t>>>
+extractDimSizes(Range &&attrs, Mapper mapValueToOptional) {
+  SmallVector<std::optional<int64_t>> dimSizes;
+  dimSizes.reserve(std::distance(std::begin(attrs), std::end(attrs)));
+  for (Attribute attr : attrs) {
+    auto valAttr = dyn_cast<IntegerAttr>(attr);
+    if (!valAttr) {
+      return failure();
+    }
+    dimSizes.push_back(mapValueToOptional(valAttr.getInt()));
+  }
+  return dimSizes;
+}
+
 DiagnosedSilenceableFailure IREE::transform_dialect::MatchDimsEqualOp::apply(
     transform::TransformRewriter &rewriter,
     transform::TransformResults &results, transform::TransformState &state) {
   ArrayRef<transform::Param> currentDimSizes =
       state.getParams(getDimensionSizes());
-  ArrayAttr targetDimensionSizes = getExpectedValues();
+  ArrayAttr targetDimSizes = getExpectedValues();
+
+  if (currentDimSizes.size() != targetDimSizes.size()) {
+    return emitSilenceableError()
+           << "dimension sizes and expected values have different lengths";
+  }
 
-  if (!llvm::equal(currentDimSizes, targetDimensionSizes)) {
+  FailureOr<SmallVector<std::optional<int64_t>>> currentDims = extractDimSizes(
+      currentDimSizes, [](int64_t v) { return std::optional<int64_t>(v); });
+  if (failed(currentDims)) {
     return emitSilenceableError()
-           << "Dimension sizes do not match expected values";
+           << "expected integer attributes for current sizes";
+  }
+
+  FailureOr<SmallVector<std::optional<int64_t>>> targetDims =
+      extractDimSizes(targetDimSizes.getValue(), [](int64_t v) {
+        return v == -1 ? std::nullopt : std::optional<int64_t>(v);
+      });
+  if (failed(targetDims)) {
+    return emitSilenceableError()
+           << "expected integer attributes for expected sizes";
+  }
+
+  for (auto [current, target] : llvm::zip_equal(*currentDims, *targetDims)) {
+    if (target && *current != *target) {
+      return emitSilenceableError()
+             << "dimension size " << *current << " does not match " << *target;
+    }
```

**Comment:**
This whole thing seems needlessly complicated -- I don't think we need two layers of possible failures and emit precise silenceable failures for every failure case. I'd do something like:
```c++
auto extractDimSizes = [](const auto &range) {
  return llvm::map_to_vector(range, [](Attribute attr) -> std::optional<int64_t> {
    if (intAttr = dyn_cast<IntegerAttr>(attr)) {
      return intAttr.getValue();
    }
    return std::nullopt;
  });
};
```

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:357`

```diff
@@ -307,16 +307,54 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
 // MatchDimsEqualOp
 //===----------------------------------------------------------------------===//
 
+template <typename Range, typename Mapper>
+static FailureOr<SmallVector<std::optional<int64_t>>>
+extractDimSizes(Range &&attrs, Mapper mapValueToOptional) {
+  SmallVector<std::optional<int64_t>> dimSizes;
+  dimSizes.reserve(std::distance(std::begin(attrs), std::end(attrs)));
+  for (Attribute attr : attrs) {
+    auto valAttr = dyn_cast<IntegerAttr>(attr);
+    if (!valAttr) {
+      return failure();
+    }
+    dimSizes.push_back(mapValueToOptional(valAttr.getInt()));
+  }
+  return dimSizes;
+}
+
 DiagnosedSilenceableFailure IREE::transform_dialect::MatchDimsEqualOp::apply(
     transform::TransformRewriter &rewriter,
     transform::TransformResults &results, transform::TransformState &state) {
   ArrayRef<transform::Param> currentDimSizes =
       state.getParams(getDimensionSizes());
-  ArrayAttr targetDimensionSizes = getExpectedValues();
+  ArrayAttr targetDimSizes = getExpectedValues();
+
+  if (currentDimSizes.size() != targetDimSizes.size()) {
+    return emitSilenceableError()
+           << "dimension sizes and expected values have different lengths";
+  }
 
-  if (!llvm::equal(currentDimSizes, targetDimensionSizes)) {
+  FailureOr<SmallVector<std::optional<int64_t>>> currentDims = extractDimSizes(
+      currentDimSizes, [](int64_t v) { return std::optional<int64_t>(v); });
+  if (failed(currentDims)) {
     return emitSilenceableError()
-           << "Dimension sizes do not match expected values";
+           << "expected integer attributes for current sizes";
+  }
+
+  FailureOr<SmallVector<std::optional<int64_t>>> targetDims =
+      extractDimSizes(targetDimSizes.getValue(), [](int64_t v) {
+        return v == -1 ? std::nullopt : std::optional<int64_t>(v);
+      });
+  if (failed(targetDims)) {
+    return emitSilenceableError()
+           << "expected integer attributes for expected sizes";
+  }
+
+  for (auto [current, target] : llvm::zip_equal(*currentDims, *targetDims)) {
+    if (target && *current != *target) {
+      return emitSilenceableError()
+             << "dimension size " << *current << " does not match " << *target;
+    }
```

**Comment:**
And then you can check for equality with https://github.com/llvm/llvm-project/blob/9ce0dae54e7d34ef4e0266069c0d3f1ae5968612/llvm/include/llvm/ADT/STLExtras.h#L2073-L2077:

```c++
if (!llvm::equal(actual, target, [](std::optional<int64_t> lhs, std::optional<int64_t> rhs) {
    return rhs == -1 || lhs == rhs;
  }) {
  return emitSilenceableError() << ...;
}

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:320`

```diff
@@ -310,13 +310,36 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
 DiagnosedSilenceableFailure IREE::transform_dialect::MatchDimsEqualOp::apply(
     transform::TransformRewriter &rewriter,
     transform::TransformResults &results, transform::TransformState &state) {
-  ArrayRef<transform::Param> currentDimSizes =
+  ArrayRef<transform::Param> currentDimAttrs =
       state.getParams(getDimensionSizes());
-  ArrayAttr targetDimensionSizes = getExpectedValues();
+  ArrayAttr targetDimAttrs = getExpectedValues();
 
-  if (!llvm::equal(currentDimSizes, targetDimensionSizes)) {
+  if (currentDimAttrs.size() != targetDimAttrs.size()) {
     return emitSilenceableError()
-           << "Dimension sizes do not match expected values";
+           << "dimension sizes and expected values have different lengths";
+  }
```

**Comment:**
nit: We could skip the size check as `llvm::equal` is going to do it anyway. I don't anticipate more than a few dims, so this should fit well within the small vector static size and not allocate any memory. But you can also keep as-is if you want.

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:340`

```diff
@@ -310,13 +310,36 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
 DiagnosedSilenceableFailure IREE::transform_dialect::MatchDimsEqualOp::apply(
     transform::TransformRewriter &rewriter,
     transform::TransformResults &results, transform::TransformState &state) {
-  ArrayRef<transform::Param> currentDimSizes =
+  ArrayRef<transform::Param> currentDimAttrs =
       state.getParams(getDimensionSizes());
-  ArrayAttr targetDimensionSizes = getExpectedValues();
+  ArrayAttr targetDimAttrs = getExpectedValues();
 
-  if (!llvm::equal(currentDimSizes, targetDimensionSizes)) {
+  if (currentDimAttrs.size() != targetDimAttrs.size()) {
     return emitSilenceableError()
-           << "Dimension sizes do not match expected values";
+           << "dimension sizes and expected values have different lengths";
+  }
+
+  auto extractDims = [](const auto &range) {
+    return llvm::map_to_vector(
+        range, [](Attribute attr) -> std::optional<int64_t> {
+          if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
+            return intAttr.getInt();
+          }
+          return std::nullopt;
+        });
+  };
+
+  SmallVector<std::optional<int64_t>> currentDims =
+      extractDims(currentDimAttrs);
+  SmallVector<std::optional<int64_t>> targetDims =
+      extractDims(targetDimAttrs.getValue());
+
+  if (!llvm::equal(currentDims, targetDims,
+                   [](const std::optional<int64_t> &lhs,
+                      const std::optional<int64_t> &rhs) {
+                     return (rhs && *rhs == -1) || (lhs && rhs && *lhs == *rhs);
```

**Comment:**
Do we need to check for the `nullopt` state explicitly? I thought that `std::optional`'s comparison operators handle it like you would expect: https://en.cppreference.com/w/cpp/utility/optional/operator_cmp.html

I wrote a quick test and it seems to be the case: https://godbolt.org/z/hbfvbjK98

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:326`

```diff
@@ -310,13 +310,36 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
 DiagnosedSilenceableFailure IREE::transform_dialect::MatchDimsEqualOp::apply(
     transform::TransformRewriter &rewriter,
     transform::TransformResults &results, transform::TransformState &state) {
-  ArrayRef<transform::Param> currentDimSizes =
+  ArrayRef<transform::Param> currentDimAttrs =
       state.getParams(getDimensionSizes());
-  ArrayAttr targetDimensionSizes = getExpectedValues();
+  ArrayAttr targetDimAttrs = getExpectedValues();
 
-  if (!llvm::equal(currentDimSizes, targetDimensionSizes)) {
+  if (currentDimAttrs.size() != targetDimAttrs.size()) {
     return emitSilenceableError()
-           << "Dimension sizes do not match expected values";
+           << "dimension sizes and expected values have different lengths";
+  }
+
+  auto extractDims = [](const auto &range) {
+    return llvm::map_to_vector(
+        range, [](Attribute attr) -> std::optional<int64_t> {
+          if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
+            return intAttr.getInt();
```

**Comment:**
If you don't want to deal with arbitrary values, you can also tighten the op definition in tablegen / move this to the verifier. That's probably a better way to go about this error mode. (This is a side comment, we don't have to fix it in this PR.)

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:326`

```diff
@@ -310,13 +310,36 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
 DiagnosedSilenceableFailure IREE::transform_dialect::MatchDimsEqualOp::apply(
     transform::TransformRewriter &rewriter,
     transform::TransformResults &results, transform::TransformState &state) {
-  ArrayRef<transform::Param> currentDimSizes =
+  ArrayRef<transform::Param> currentDimAttrs =
       state.getParams(getDimensionSizes());
-  ArrayAttr targetDimensionSizes = getExpectedValues();
+  ArrayAttr targetDimAttrs = getExpectedValues();
 
-  if (!llvm::equal(currentDimSizes, targetDimensionSizes)) {
+  if (currentDimAttrs.size() != targetDimAttrs.size()) {
     return emitSilenceableError()
-           << "Dimension sizes do not match expected values";
+           << "dimension sizes and expected values have different lengths";
+  }
+
+  auto extractDims = [](const auto &range) {
+    return llvm::map_to_vector(
+        range, [](Attribute attr) -> std::optional<int64_t> {
+          if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
+            return intAttr.getInt();
```

**Comment:**
For example, you can make the target dims be `DenseI64ArrayAttr` instead of `ArrayAttr`. I don't think we can do anything about `currentDimAttrs` though.

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:340`

```diff
@@ -310,13 +310,36 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
 DiagnosedSilenceableFailure IREE::transform_dialect::MatchDimsEqualOp::apply(
     transform::TransformRewriter &rewriter,
     transform::TransformResults &results, transform::TransformState &state) {
-  ArrayRef<transform::Param> currentDimSizes =
+  ArrayRef<transform::Param> currentDimAttrs =
       state.getParams(getDimensionSizes());
-  ArrayAttr targetDimensionSizes = getExpectedValues();
+  ArrayAttr targetDimAttrs = getExpectedValues();
 
-  if (!llvm::equal(currentDimSizes, targetDimensionSizes)) {
+  if (currentDimAttrs.size() != targetDimAttrs.size()) {
     return emitSilenceableError()
-           << "Dimension sizes do not match expected values";
+           << "dimension sizes and expected values have different lengths";
+  }
+
+  auto extractDims = [](const auto &range) {
+    return llvm::map_to_vector(
+        range, [](Attribute attr) -> std::optional<int64_t> {
+          if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
+            return intAttr.getInt();
+          }
+          return std::nullopt;
+        });
+  };
+
+  SmallVector<std::optional<int64_t>> currentDims =
+      extractDims(currentDimAttrs);
+  SmallVector<std::optional<int64_t>> targetDims =
+      extractDims(targetDimAttrs.getValue());
+
+  if (!llvm::equal(currentDims, targetDims,
+                   [](const std::optional<int64_t> &lhs,
+                      const std::optional<int64_t> &rhs) {
+                     return (rhs && *rhs == -1) || (lhs && rhs && *lhs == *rhs);
```

**Comment:**
This doesn't seem to have been addressed

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:315`

```diff
@@ -310,13 +310,23 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
 DiagnosedSilenceableFailure IREE::transform_dialect::MatchDimsEqualOp::apply(
     transform::TransformRewriter &rewriter,
     transform::TransformResults &results, transform::TransformState &state) {
-  ArrayRef<transform::Param> currentDimSizes =
+  ArrayRef<transform::Param> currentDimAttrs =
       state.getParams(getDimensionSizes());
-  ArrayAttr targetDimensionSizes = getExpectedValues();
+  ArrayRef<int64_t> targetDims = getExpectedValues();
```

**Comment:**
nit: since you this is called in `expected_values` in tablegen, I'd keep similar variables names in the code here. Then you can have `actualDimAttrs` and `expectedDims`, which is a common pairing in any test code that performs pairwise comparisons.

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:340`

```diff
@@ -310,13 +310,36 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
 DiagnosedSilenceableFailure IREE::transform_dialect::MatchDimsEqualOp::apply(
     transform::TransformRewriter &rewriter,
     transform::TransformResults &results, transform::TransformState &state) {
-  ArrayRef<transform::Param> currentDimSizes =
+  ArrayRef<transform::Param> currentDimAttrs =
       state.getParams(getDimensionSizes());
-  ArrayAttr targetDimensionSizes = getExpectedValues();
+  ArrayAttr targetDimAttrs = getExpectedValues();
 
-  if (!llvm::equal(currentDimSizes, targetDimensionSizes)) {
+  if (currentDimAttrs.size() != targetDimAttrs.size()) {
     return emitSilenceableError()
-           << "Dimension sizes do not match expected values";
+           << "dimension sizes and expected values have different lengths";
+  }
+
+  auto extractDims = [](const auto &range) {
+    return llvm::map_to_vector(
+        range, [](Attribute attr) -> std::optional<int64_t> {
+          if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
+            return intAttr.getInt();
+          }
+          return std::nullopt;
+        });
+  };
+
+  SmallVector<std::optional<int64_t>> currentDims =
+      extractDims(currentDimAttrs);
+  SmallVector<std::optional<int64_t>> targetDims =
+      extractDims(targetDimAttrs.getValue());
+
+  if (!llvm::equal(currentDims, targetDims,
+                   [](const std::optional<int64_t> &lhs,
+                      const std::optional<int64_t> &rhs) {
+                     return (rhs && *rhs == -1) || (lhs && rhs && *lhs == *rhs);
```

**Comment:**
It's supported in c++17 https://godbolt.org/z/WzMvh6eYP
<img width="820" height="145" alt="image" src="https://github.com/user-attachments/assets/f012f6b2-614e-48db-87b1-dd99b4f3c24c" />

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:327`

```diff
@@ -310,21 +310,21 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
 DiagnosedSilenceableFailure IREE::transform_dialect::MatchDimsEqualOp::apply(
     transform::TransformRewriter &rewriter,
     transform::TransformResults &results, transform::TransformState &state) {
-  ArrayRef<transform::Param> currentDimAttrs =
+  ArrayRef<transform::Param> actualDimAttrs =
       state.getParams(getDimensionSizes());
-  ArrayRef<int64_t> targetDims = getExpectedValues();
+  ArrayRef<int64_t> expectedDims = getExpectedValues();
 
-  SmallVector<std::optional<int64_t>> currentDims = llvm::map_to_vector(
-      currentDimAttrs, [](Attribute attr) -> std::optional<int64_t> {
+  SmallVector<std::optional<int64_t>> actualDims = llvm::map_to_vector(
+      actualDimAttrs, [](Attribute attr) -> std::optional<int64_t> {
         if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
           return intAttr.getInt();
         }
         return std::nullopt;
       });
 
-  if (!llvm::equal(currentDims, targetDims,
+  if (!llvm::equal(actualDims, expectedDims,
                    [](const std::optional<int64_t> &lhs, int64_t rhs) {
-                     return (rhs == -1) || (lhs && *lhs == rhs);
+                     return (rhs == -1) || (lhs == rhs);
```

**Comment:**
```suggestion
                     return rhs == -1 || lhs == rhs;
```

---


---


## [PR #22137](https://github.com/iree-org/iree/pull/22137): [Codegen][GPU] Perfer MMA over VirtualMMA in Sorting. 

### Review Summary

**CHANGES_REQUESTED** (2025-09-28)

Does it matter that `sortMMAInstrinsics` uses `llvm::sort` instead of `llvm::stable_sort`?


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.cpp:579`

```diff
@@ -566,9 +566,30 @@ bool compareIntrinsics(const GPUMatmulShapeType &problem,
     return lhsArea > rhsArea;
   }
 
-  // Finally if everything else is the same, prefer large K size.
-  return ShapedType::getNumElements(lhs.kSizes) >
-         ShapedType::getNumElements(rhs.kSizes);
+  // Prefer large K size here.
+  int64_t lhsKSize = ShapedType::getNumElements(lhs.kSizes);
+  int64_t rhsKSize = ShapedType::getNumElements(rhs.kSizes);
+  if (lhsKSize != rhsKSize) {
+    return lhsKSize > rhsKSize;
+  }
+
+  const GPUIntrinsicType *lhsIntrinsic =
+      static_cast<const GPUIntrinsicType *>(&lhs);
+  const GPUIntrinsicType *rhsIntrinsic =
+      static_cast<const GPUIntrinsicType *>(&rhs);
```

**Comment:**
See https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable
```suggestion
  const auto *lhsIntrinsic =
      static_cast<const GPUIntrinsicType *>(&lhs);
  const auto *rhsIntrinsic =
      static_cast<const GPUIntrinsicType *>(&rhs);
```

But this cast seems inherently unsafe -- if we need `GPUIntrinsicType`, we should change the function signature of `compareIntrinsics` to take that instead.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.cpp:574`

```diff
@@ -566,9 +566,30 @@ bool compareIntrinsics(const GPUMatmulShapeType &problem,
     return lhsArea > rhsArea;
   }
 
-  // Finally if everything else is the same, prefer large K size.
-  return ShapedType::getNumElements(lhs.kSizes) >
-         ShapedType::getNumElements(rhs.kSizes);
+  // Prefer large K size here.
+  int64_t lhsKSize = ShapedType::getNumElements(lhs.kSizes);
+  int64_t rhsKSize = ShapedType::getNumElements(rhs.kSizes);
+  if (lhsKSize != rhsKSize) {
+    return lhsKSize > rhsKSize;
+  }
```

**Comment:**
An easier way to write this comparison is to come up with a feature tuple for `GPUInstrinsicType` and then delegate the exact comparison logic to the `std::tuple` comparison operator.

---


---


## [PR #22134](https://github.com/iree-org/iree/pull/22134): [Codegen][GPU] update the tuning spec using new transform ops

### Review Summary

**CHANGES_REQUESTED** (2025-10-07)

Nice, the new matcher ops make this much more manageable. Two things:
1. Can you split this up into multiple PRs, one per each new matcher? This way we can tell which one is at fault if anything breaks.
2. Can you confirm that the new specs work? I'd expect that the should apply to the same number of ops and that the resulting .vmfb binaries are identical.



---


## [PR #22122](https://github.com/iree-org/iree/pull/22122): [Codegen][GPU][NFC] Fix mma sort follow up

### Review Summary

**APPROVED** (2025-09-26)

Thanks. Can you also add `NFC` to the PR subject line?



---


## [PR #22090](https://github.com/iree-org/iree/pull/22090): [Codegen][GPU] Fix MMA Intrinsics Sorting

### Review Summary

**APPROVED** (2025-09-24)

Nice catch


**COMMENTED** (2025-09-26)


**COMMENTED** (2025-09-26)


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.h:47`

```diff
@@ -30,6 +36,18 @@ struct GPUMatmulShapeType {
                      Type b, Type c)
       : mSizes(m), nSizes(n), kSizes(k), batchSizes(batch), aType(a), bType(b),
         cType(c) {}
+
+  // Constructor with the num_rhs parameter.
+  GPUMatmulShapeType(int64_t m, int64_t n, int64_t k, Type a, Type b, Type c,
+                     int64_t numHorizontallyFusedOps)
+      : mSizes({m}), nSizes({n}), kSizes({k}), batchSizes({}), aType(a),
+        bType(b), cType(c), numHorizontallyFusedOps(numHorizontallyFusedOps) {}
+
+  GPUMatmulShapeType(ArrayRef<int64_t> m, ArrayRef<int64_t> n,
+                     ArrayRef<int64_t> k, ArrayRef<int64_t> batch, Type a,
```

**Comment:**
Why not make `numHorizontallyFusedOps` a default constructor parameter to the existing consturctors?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.h:40`

```diff
@@ -30,6 +36,18 @@ struct GPUMatmulShapeType {
                      Type b, Type c)
       : mSizes(m), nSizes(n), kSizes(k), batchSizes(batch), aType(a), bType(b),
         cType(c) {}
+
+  // Constructor with the num_rhs parameter.
```

**Comment:**
This is not super helpful IMO and the variable name in the comment doesn't match the code.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.h:47`

```diff
@@ -30,6 +36,18 @@ struct GPUMatmulShapeType {
                      Type b, Type c)
       : mSizes(m), nSizes(n), kSizes(k), batchSizes(batch), aType(a), bType(b),
         cType(c) {}
+
+  // Constructor with the num_rhs parameter.
+  GPUMatmulShapeType(int64_t m, int64_t n, int64_t k, Type a, Type b, Type c,
+                     int64_t numHorizontallyFusedOps)
+      : mSizes({m}), nSizes({n}), kSizes({k}), batchSizes({}), aType(a),
+        bType(b), cType(c), numHorizontallyFusedOps(numHorizontallyFusedOps) {}
+
+  GPUMatmulShapeType(ArrayRef<int64_t> m, ArrayRef<int64_t> n,
+                     ArrayRef<int64_t> k, ArrayRef<int64_t> batch, Type a,
```

**Comment:**
I'm not super familiar with this code, does a default argument make it incompatible with the existing usage?

---


---


## [PR #22040](https://github.com/iree-org/iree/pull/22040): [Codegen] Add transform op for matching dimension sizes.

### Review Summary

**COMMENTED** (2025-09-22)


**COMMENTED** (2025-09-22)


**COMMENTED** (2025-09-22)


**COMMENTED** (2025-09-22)


**COMMENTED** (2025-09-22)


**COMMENTED** (2025-09-24)


**COMMENTED** (2025-09-24)


**COMMENTED** (2025-09-25)


**APPROVED** (2025-09-25)

LGTM but let's wait for @qedawkins or @Max191 to review too


### Code Comments

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:331`

```diff
@@ -303,6 +303,41 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
   return DiagnosedSilenceableFailure::success();
 }
 
+//===----------------------------------------------------------------------===//
+// MatchSizeEqualsOp
+//===----------------------------------------------------------------------===//
+
+DiagnosedSilenceableFailure
+IREE::transform_dialect::MatchSizeEqualsOp::matchOperation(
+    Operation *current, transform::TransformResults &results,
+    transform::TransformState &state) {
+  Location loc = current->getLoc();
+  ArrayRef<transform::Param> actualDimSizes =
+      state.getParams(getDimensionSizes());
+  ArrayAttr targetDimensionSizes = getExpectedValues();
+
+  if (actualDimSizes.size() != targetDimensionSizes.size()) {
+    return emitSilenceableFailure(loc)
+           << "Dimension sizes array and target sizes array have different "
+              "lengths";
+  }
+
+  for (auto [dimParam, targetValuesAttr] :
+       llvm::zip_equal(actualDimSizes, targetDimensionSizes)) {
+    int64_t currentDimSize = cast<IntegerAttr>(dimParam).getInt();
+    ArrayAttr allowedSizes = cast<ArrayAttr>(targetValuesAttr);
+    bool foundMatch = llvm::any_of(allowedSizes, [&](Attribute targetAttr) {
+      return cast<IntegerAttr>(targetAttr).getInt() == currentDimSize;
+    });
```

**Comment:**
How do you know you know each dim gets matched? For example, I think this would match:
expected: `[64, 64, 64]`
actual: `[64, 128, 256]`

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td:243`

```diff
@@ -225,4 +225,42 @@ def MatchContractionOp : Op<Transform_Dialect, "iree.match.contraction",
   let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
 }
 
+def MatchSizeEqualsOp : Op<Transform_Dialect, "iree.match.size_equals",
+    [MatchOpInterface,
+     SingleOpMatcher,
+     MemoryEffectsOpInterface]> {
+  let summary = [{Check whether a single transform parameter matches expected size values.}];
+  let description = [{
+    Matches dimension sizes against expected values.
+    The dimension sizes is an array of transform parameters, and the expected values
+    are structured as nested arrays where each sub-array corresponds to a position
+    in the size array.
+
+    Example: %batch_dims = [[2, 4], [8, 16], [32, 64]]
+    This means:
+    - %batch_dims[0] should be either 2 or 4.
+    - %batch_dims[1] should be either 8 or 16.
+    - %batch_dims[2] should be either 32 or 64.
```

**Comment:**
I don't understand why we need to support alternatives. What's the usecase?

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:331`

```diff
@@ -303,6 +303,41 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
   return DiagnosedSilenceableFailure::success();
 }
 
+//===----------------------------------------------------------------------===//
+// MatchSizeEqualsOp
+//===----------------------------------------------------------------------===//
+
+DiagnosedSilenceableFailure
+IREE::transform_dialect::MatchSizeEqualsOp::matchOperation(
+    Operation *current, transform::TransformResults &results,
+    transform::TransformState &state) {
+  Location loc = current->getLoc();
+  ArrayRef<transform::Param> actualDimSizes =
+      state.getParams(getDimensionSizes());
+  ArrayAttr targetDimensionSizes = getExpectedValues();
+
+  if (actualDimSizes.size() != targetDimensionSizes.size()) {
+    return emitSilenceableFailure(loc)
+           << "Dimension sizes array and target sizes array have different "
+              "lengths";
+  }
+
+  for (auto [dimParam, targetValuesAttr] :
+       llvm::zip_equal(actualDimSizes, targetDimensionSizes)) {
+    int64_t currentDimSize = cast<IntegerAttr>(dimParam).getInt();
+    ArrayAttr allowedSizes = cast<ArrayAttr>(targetValuesAttr);
+    bool foundMatch = llvm::any_of(allowedSizes, [&](Attribute targetAttr) {
+      return cast<IntegerAttr>(targetAttr).getInt() == currentDimSize;
+    });
```

**Comment:**
OK, I see what's going on: https://github.com/iree-org/iree/pull/22040/files#r2370334264

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td:243`

```diff
@@ -225,4 +225,42 @@ def MatchContractionOp : Op<Transform_Dialect, "iree.match.contraction",
   let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
 }
 
+def MatchSizeEqualsOp : Op<Transform_Dialect, "iree.match.size_equals",
+    [MatchOpInterface,
+     SingleOpMatcher,
+     MemoryEffectsOpInterface]> {
+  let summary = [{Check whether a single transform parameter matches expected size values.}];
+  let description = [{
+    Matches dimension sizes against expected values.
+    The dimension sizes is an array of transform parameters, and the expected values
+    are structured as nested arrays where each sub-array corresponds to a position
+    in the size array.
+
+    Example: %batch_dims = [[2, 4], [8, 16], [32, 64]]
+    This means:
+    - %batch_dims[0] should be either 2 or 4.
+    - %batch_dims[1] should be either 8 or 16.
+    - %batch_dims[2] should be either 32 or 64.
```

**Comment:**
but why do we need this?

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td:243`

```diff
@@ -225,4 +225,42 @@ def MatchContractionOp : Op<Transform_Dialect, "iree.match.contraction",
   let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
 }
 
+def MatchSizeEqualsOp : Op<Transform_Dialect, "iree.match.size_equals",
+    [MatchOpInterface,
+     SingleOpMatcher,
+     MemoryEffectsOpInterface]> {
+  let summary = [{Check whether a single transform parameter matches expected size values.}];
+  let description = [{
+    Matches dimension sizes against expected values.
+    The dimension sizes is an array of transform parameters, and the expected values
+    are structured as nested arrays where each sub-array corresponds to a position
+    in the size array.
+
+    Example: %batch_dims = [[2, 4], [8, 16], [32, 64]]
+    This means:
+    - %batch_dims[0] should be either 2 or 4.
+    - %batch_dims[1] should be either 8 or 16.
+    - %batch_dims[2] should be either 32 or 64.
```

**Comment:**
Yeah I don't think we will have to emit this from the tuner. Later on we could potentially allow for merging of matchers, but I don't think we will need it any time soon. For the v0, I'd like to be able to replace DAG matching only.

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:332`

```diff
@@ -303,6 +303,42 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
   return DiagnosedSilenceableFailure::success();
 }
 
+//===----------------------------------------------------------------------===//
+// MatchSizeEqualsOp
+//===----------------------------------------------------------------------===//
+
+DiagnosedSilenceableFailure IREE::transform_dialect::MatchSizeEqualsOp::apply(
+    transform::TransformRewriter &rewriter,
+    transform::TransformResults &results, transform::TransformState &state) {
+  ArrayRef<transform::Param> currentDimSizes =
+      state.getParams(getDimensionSizes());
+  ArrayAttr targetDimensionSizes = getExpectedValues();
+
+  if (currentDimSizes.size() != targetDimensionSizes.size()) {
+    return emitSilenceableError() << "dimension sizes have different lengths ("
+                                  << currentDimSizes.size() << " vs "
+                                  << targetDimensionSizes.size() << ")";
+  }
+
+  for (auto [currentDimSizeAttr, targetDimSizeAttr] :
+       llvm::zip_equal(currentDimSizes, targetDimensionSizes)) {
+    int64_t currentDimSize = cast<IntegerAttr>(currentDimSizeAttr).getInt();
+    int64_t targetDimSize = cast<IntegerAttr>(targetDimSizeAttr).getInt();
+    if (currentDimSize != targetDimSize) {
+      return emitSilenceableError()
+             << "Dimension size " << currentDimSize
+             << " does not match expected size " << targetDimSize;
+    }
+  }
```

**Comment:**
Could we replace this with `if(!llvm::equal(currentDimSizes, targetDimSizes))`? I don't think we need a precise diagnostics beyond that the two arrays don't match

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td:241`

```diff
@@ -225,4 +225,40 @@ def MatchContractionOp : Op<Transform_Dialect, "iree.match.contraction",
   let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
 }
 
+def MatchSizeEqualsOp : Op<Transform_Dialect, "iree.match.size_equals",
+    [DeclareOpInterfaceMethods<TransformOpInterface>,
+     MatchOpInterface,
+     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
+  let summary = [{Check whether transform parameters match expected size values exactly.}];
+  let description = [{
+    Matches dimension sizes against expected values.
+    Each position in the dimension sizes array must match the corresponding
+    expected value exactly.
+
+    Example: %batch_dims contains [2, 4] and expected_values is [2, 4].
+    This means:
+    - %batch_dims[0] must equal 2.
+    - %batch_dims[1] must equal 4.
```

**Comment:**
Can we use an actual op to show the syntax?

Something like:
```
 `transform.iree.match.size_equals %m, [512, 256]` matches when `%m` has two dims, 512 and 256.
```

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:322`

```diff
@@ -303,6 +304,36 @@ IREE::transform_dialect::MatchContractionOp::matchOperation(
   return DiagnosedSilenceableFailure::success();
 }
 
+//===----------------------------------------------------------------------===//
+// MatchSizeEqualsOp
+//===----------------------------------------------------------------------===//
+
+DiagnosedSilenceableFailure IREE::transform_dialect::MatchSizeEqualsOp::apply(
+    transform::TransformRewriter &rewriter,
+    transform::TransformResults &results, transform::TransformState &state) {
+  ArrayRef<transform::Param> currentDimSizes =
+      state.getParams(getDimensionSizes());
+  ArrayAttr targetDimensionSizes = getExpectedValues();
+
+  if (currentDimSizes.size() != targetDimensionSizes.size()) {
+    return emitSilenceableError() << "dimension sizes have different lengths ("
+                                  << currentDimSizes.size() << " vs "
+                                  << targetDimensionSizes.size() << ")";
+  }
```

**Comment:**
`llvm::equal` below already checks the sizes for you: https://en.cppreference.com/w/cpp/algorithm/equal.html#:~:text=of%20BinaryPredicate.-,Return%20value,-1%2D4)

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td:241`

```diff
@@ -225,4 +225,40 @@ def MatchContractionOp : Op<Transform_Dialect, "iree.match.contraction",
   let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
 }
 
+def MatchSizeEqualsOp : Op<Transform_Dialect, "iree.match.size_equals",
+    [DeclareOpInterfaceMethods<TransformOpInterface>,
+     MatchOpInterface,
+     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
+  let summary = [{Check whether transform parameters match expected size values exactly.}];
+  let description = [{
+    Matches dimension sizes against expected values.
+    Each position in the dimension sizes array must match the corresponding
+    expected value exactly.
+
+    ### Example
+
+    `transform.iree.match.size_equals %m, [512, 256] : !transform.param<i64>`
+    succeeds when `%m` has exactly two dimensions, 512 and 256.
```

**Comment:**
Code blocks render nicely on the website
```suggestion
    ```milr
    transform.iree.match.size_equals %m, [512, 256] : !transform.param<i64>
    ```

    This succeeds when `%m` has exactly two dimensions, 512 and 256.
```

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td:228`

```diff
@@ -225,4 +225,40 @@ def MatchContractionOp : Op<Transform_Dialect, "iree.match.contraction",
   let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
 }
 
+def MatchSizeEqualsOp : Op<Transform_Dialect, "iree.match.size_equals",
```

**Comment:**
I wonder if we should call it `iree.match.dims_equal` or something like that -- we are no longer taking a handle to the op as an argument, we are only looking at some of its dimension. I think sizes would imply we are looking at the overall size.

We already have two custom dim matching ops: https://github.com/iree-org/iree/blob/2c5b07daa588cd81c41e1ff19aa4130c7e41e6ce/compiler/src/iree/compiler/Preprocessing/Common/test/preprocessing_match_ops.mlir#L180-L181

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td:228`

```diff
@@ -225,4 +225,40 @@ def MatchContractionOp : Op<Transform_Dialect, "iree.match.contraction",
   let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
 }
 
+def MatchSizeEqualsOp : Op<Transform_Dialect, "iree.match.size_equals",
```

**Comment:**
Another example: https://mlir.llvm.org/docs/Dialects/Transform/#transformmatchparamcmpi-transformmatchparamcmpiop

https://github.com/search?q=repo%3Allvm%2Fllvm-project%20transform.match.param.cmpi&type=code

---


---


## [PR #22027](https://github.com/iree-org/iree/pull/22027): [Codegen][Tuner] update lowering config binding for subgroup basis

### Review Summary

**APPROVED** (2025-09-18)


**COMMENTED** (2025-09-18)


### Code Comments

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:378`

```diff
@@ -356,6 +356,33 @@ ireeGPUGetSingleSubgroupLayout(MlirAttribute attr, uint32_t fragment) {
   return result;
 }
 
+ireeGPUSubgroupBasisInfo
+ireeGPULoweringConfigAttrGetSubgroupBasis(MlirAttribute attr) {
+  auto loweringConfigAttr =
+      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
+          unwrap(attr));
+
+  mlir::FailureOr<mlir::iree_compiler::IREE::GPU::Basis> basisResult =
+      mlir::iree_compiler::IREE::GPU::getBasis(
+          loweringConfigAttr,
+          mlir::iree_compiler::IREE::GPU::TilingLevel::Subgroup);
+
+  ireeGPUSubgroupBasisInfo info = {};
+  if (failed(basisResult)) {
+    return info;
+  }
+
+  mlir::iree_compiler::IREE::GPU::Basis basis = *basisResult;
+  mlir::Builder builder(loweringConfigAttr.getContext());
+  mlir::ArrayAttr countsAttr = builder.getI64ArrayAttr(basis.counts);
+  mlir::ArrayAttr mappingAttr = builder.getI64ArrayAttr(basis.mapping);
```

**Comment:**
Do we know what's the cost of creating the builder just to create some attributes? I know we've been doing that for a while in this file, but this stood out for me just now. No need to change this here, just something to check separately.

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:378`

```diff
@@ -356,6 +356,33 @@ ireeGPUGetSingleSubgroupLayout(MlirAttribute attr, uint32_t fragment) {
   return result;
 }
 
+ireeGPUSubgroupBasisInfo
+ireeGPULoweringConfigAttrGetSubgroupBasis(MlirAttribute attr) {
+  auto loweringConfigAttr =
+      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
+          unwrap(attr));
+
+  mlir::FailureOr<mlir::iree_compiler::IREE::GPU::Basis> basisResult =
+      mlir::iree_compiler::IREE::GPU::getBasis(
+          loweringConfigAttr,
+          mlir::iree_compiler::IREE::GPU::TilingLevel::Subgroup);
+
+  ireeGPUSubgroupBasisInfo info = {};
+  if (failed(basisResult)) {
+    return info;
+  }
+
+  mlir::iree_compiler::IREE::GPU::Basis basis = *basisResult;
+  mlir::Builder builder(loweringConfigAttr.getContext());
+  mlir::ArrayAttr countsAttr = builder.getI64ArrayAttr(basis.counts);
+  mlir::ArrayAttr mappingAttr = builder.getI64ArrayAttr(basis.mapping);
```

**Comment:**
I've just checked and it's cheap: https://github.com/llvm/llvm-project/blob/1a172b9924948f10f1bd3db07a83fe5e884f7b64/mlir/include/mlir/IR/Builders.h#L51-L55

---


---


## [PR #21981](https://github.com/iree-org/iree/pull/21981): [Codegen] Add transform ops for matching contraction ops

### Review Summary

**COMMENTED** (2025-09-15)

How do you expect these to be used within a tuning spec? Right now, these don't have a way of specifying things like indexing maps or operand/result types, so I think we would still have to perform dag-based matching, no?


**COMMENTED** (2025-09-21)


**COMMENTED** (2025-09-22)


**COMMENTED** (2025-09-22)


**COMMENTED** (2025-09-22)


**APPROVED** (2025-09-22)

I think calling it `transform.iree.match.contraction` would be more consistent with the other match extension ops. Otherwise looks good.


### Code Comments

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td:205`

```diff
@@ -172,4 +172,57 @@ def MatchRegionsOp : Op<Transform_Dialect, "iree.match.regions",
   let hasVerifier = 1;
 }
 
+def MatchContractionOp : Op<Transform_Dialect, "iree.match.is_contraction",
+    [MatchOpInterface,
+     SingleOpMatcher,
+     MemoryEffectsOpInterface]> {
+  let summary = [{Check whether the op is a contraction operation.}];
+  let description = [{
+    Matches operations that implement the ContractionOpInterface.
+    This includes operations like linalg.matmul, linalg.batch_matmul, etc.
+
+    Optionally matches specific indexing maps patterns.
+
+    #### Return modes
+
+    Succeeds if the operation is a contraction operation, and
+    produces a silenceable failure otherwise.
+
+    #### Results
+
+    Returns arrays of dimension sizes for each contraction dimension:
+    - batch_dims: Array of batch dimension sizes.
+    - m_dims: Array of M dimension sizes.
+    - n_dims: Array of N dimension sizes.
+    - k_dims: Array of K dimension sizes.
+  }];
+
+  let arguments = (ins
+    TransformHandleTypeInterface:$operand_handle,
+    OptionalAttr<AffineMapArrayAttr>:$indexing_maps,
+    OptionalAttr<TypeAttr>:$lhs_type,                 // LHS input type.
+    OptionalAttr<TypeAttr>:$rhs_type,                 // RHS input type.
+    OptionalAttr<TypeAttr>:$output_type               // Output type.
```

**Comment:**
Should we make these non-optional? I don't think we have any use for matching contraction ops with arbitrary types

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td:202`

```diff
@@ -172,4 +172,57 @@ def MatchRegionsOp : Op<Transform_Dialect, "iree.match.regions",
   let hasVerifier = 1;
 }
 
+def MatchContractionOp : Op<Transform_Dialect, "iree.match.is_contraction",
+    [MatchOpInterface,
+     SingleOpMatcher,
+     MemoryEffectsOpInterface]> {
+  let summary = [{Check whether the op is a contraction operation.}];
+  let description = [{
+    Matches operations that implement the ContractionOpInterface.
+    This includes operations like linalg.matmul, linalg.batch_matmul, etc.
+
+    Optionally matches specific indexing maps patterns.
+
+    #### Return modes
+
+    Succeeds if the operation is a contraction operation, and
+    produces a silenceable failure otherwise.
+
+    #### Results
+
+    Returns arrays of dimension sizes for each contraction dimension:
+    - batch_dims: Array of batch dimension sizes.
+    - m_dims: Array of M dimension sizes.
+    - n_dims: Array of N dimension sizes.
+    - k_dims: Array of K dimension sizes.
+  }];
+
+  let arguments = (ins
+    TransformHandleTypeInterface:$operand_handle,
+    OptionalAttr<AffineMapArrayAttr>:$indexing_maps,
```

**Comment:**
Why not define a few variants like `NN`, `NT`, etc., like we discussed before? I'm not saying indexing maps are a bad option, but these are quite verbose and I'd like to understand what motivates this extra complexity.

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:256`

```diff
@@ -223,6 +223,114 @@ IREE::transform_dialect::MatchCastCompatibleDagFromRootOp::verify() {
   return success();
 }
 
+//===----------------------------------------------------------------------===//
+// MatchContractionOp
+//===----------------------------------------------------------------------===//
+
+DiagnosedSilenceableFailure
+IREE::transform_dialect::MatchContractionOp::matchOperation(
+    Operation *current, transform::TransformResults &results,
+    transform::TransformState &state) {
+  auto linalgOp = dyn_cast<linalg::LinalgOp>(current);
+  if (!linalgOp) {
+    return emitSilenceableFailure(current->getLoc())
+           << "Operation " << *current << " is not a LinalgOp.";
+  }
+
+  if (!linalg::isaContractionOpInterface(linalgOp)) {
+    return emitSilenceableFailure(current->getLoc())
+           << "Operation " << *current << " is not a contraction operation.";
+  }
+
+  if (std::optional<ArrayAttr> indexingMaps = getIndexingMaps()) {
+    ArrayAttr currentIndexingMaps = linalgOp.getIndexingMaps();
+    ArrayAttr targetIndexingMaps = *indexingMaps;
+    if (currentIndexingMaps.size() != targetIndexingMaps.size()) {
+      return emitSilenceableFailure(current->getLoc())
+             << "indexing maps count mismatch: expected "
+             << targetIndexingMaps.size() << ", got "
+             << currentIndexingMaps.size();
+    }
+
+    for (auto [currentMapAttr, targetMapAttr] :
+         llvm::zip(currentIndexingMaps, targetIndexingMaps)) {
```

**Comment:**
Use `zip_equal` for ranges of equal lengths. See https://llvm.org/docs/ProgrammersManual.html#iterating-over-ranges

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:248`

```diff
@@ -223,6 +223,114 @@ IREE::transform_dialect::MatchCastCompatibleDagFromRootOp::verify() {
   return success();
 }
 
+//===----------------------------------------------------------------------===//
+// MatchContractionOp
+//===----------------------------------------------------------------------===//
+
+DiagnosedSilenceableFailure
+IREE::transform_dialect::MatchContractionOp::matchOperation(
+    Operation *current, transform::TransformResults &results,
+    transform::TransformState &state) {
+  auto linalgOp = dyn_cast<linalg::LinalgOp>(current);
+  if (!linalgOp) {
+    return emitSilenceableFailure(current->getLoc())
+           << "Operation " << *current << " is not a LinalgOp.";
+  }
+
+  if (!linalg::isaContractionOpInterface(linalgOp)) {
+    return emitSilenceableFailure(current->getLoc())
+           << "Operation " << *current << " is not a contraction operation.";
+  }
+
+  if (std::optional<ArrayAttr> indexingMaps = getIndexingMaps()) {
+    ArrayAttr currentIndexingMaps = linalgOp.getIndexingMaps();
+    ArrayAttr targetIndexingMaps = *indexingMaps;
+    if (currentIndexingMaps.size() != targetIndexingMaps.size()) {
```

**Comment:**
Why not compare two array attributes? I don't think there's value in producing verbose silenceable failures -- nobody looks at them

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:236`

```diff
@@ -223,6 +223,114 @@ IREE::transform_dialect::MatchCastCompatibleDagFromRootOp::verify() {
   return success();
 }
 
+//===----------------------------------------------------------------------===//
+// MatchContractionOp
+//===----------------------------------------------------------------------===//
+
+DiagnosedSilenceableFailure
+IREE::transform_dialect::MatchContractionOp::matchOperation(
+    Operation *current, transform::TransformResults &results,
+    transform::TransformState &state) {
+  auto linalgOp = dyn_cast<linalg::LinalgOp>(current);
+  if (!linalgOp) {
+    return emitSilenceableFailure(current->getLoc())
```

**Comment:**
Since the `current->getLoc()` expression appears often in this function, I'd hoist it to a local variable

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:317`

```diff
@@ -223,6 +223,114 @@ IREE::transform_dialect::MatchCastCompatibleDagFromRootOp::verify() {
   return success();
 }
 
+//===----------------------------------------------------------------------===//
+// MatchContractionOp
+//===----------------------------------------------------------------------===//
+
+DiagnosedSilenceableFailure
+IREE::transform_dialect::MatchContractionOp::matchOperation(
+    Operation *current, transform::TransformResults &results,
+    transform::TransformState &state) {
+  auto linalgOp = dyn_cast<linalg::LinalgOp>(current);
+  if (!linalgOp) {
+    return emitSilenceableFailure(current->getLoc())
+           << "Operation " << *current << " is not a LinalgOp.";
+  }
+
+  if (!linalg::isaContractionOpInterface(linalgOp)) {
+    return emitSilenceableFailure(current->getLoc())
+           << "Operation " << *current << " is not a contraction operation.";
+  }
+
+  if (std::optional<ArrayAttr> indexingMaps = getIndexingMaps()) {
+    ArrayAttr currentIndexingMaps = linalgOp.getIndexingMaps();
+    ArrayAttr targetIndexingMaps = *indexingMaps;
+    if (currentIndexingMaps.size() != targetIndexingMaps.size()) {
+      return emitSilenceableFailure(current->getLoc())
+             << "indexing maps count mismatch: expected "
+             << targetIndexingMaps.size() << ", got "
+             << currentIndexingMaps.size();
+    }
+
+    for (auto [currentMapAttr, targetMapAttr] :
+         llvm::zip(currentIndexingMaps, targetIndexingMaps)) {
+      AffineMapAttr currentMap = cast<AffineMapAttr>(currentMapAttr);
+      AffineMapAttr targetMap = cast<AffineMapAttr>(targetMapAttr);
+      if (currentMap.getValue() != targetMap.getValue()) {
+        return emitSilenceableFailure(current->getLoc())
+               << "indexing maps don't match: expected " << targetMap
+               << ", got " << currentMap;
+      }
+    }
+  }
+
+  if (std::optional<Type> lhsType = getLhsType()) {
+    Type targetLhsType = *lhsType;
+    Type currentLhsType = getElementTypeOrSelf(linalgOp.getDpsInputs()[0]);
+    if (currentLhsType != targetLhsType) {
+      return emitSilenceableFailure(current->getLoc())
+             << "LHS type doesn't match: expected " << targetLhsType << ", got "
+             << currentLhsType;
+    }
+  }
+
+  if (std::optional<Type> rhsType = getRhsType()) {
+    Type targetRhsType = *rhsType;
+    Type currentRhsType = getElementTypeOrSelf(linalgOp.getDpsInputs()[1]);
+    if (currentRhsType != targetRhsType) {
+      return emitSilenceableFailure(current->getLoc())
+             << "RHS type doesn't match: expected " << targetRhsType << ", got "
+             << currentRhsType;
+    }
+  }
+
+  if (std::optional<Type> outputType = getOutputType()) {
+    const Type targetOutputType = *outputType;
+    SmallVector<Type> currentOutputTypes;
+    for (Value output : linalgOp.getDpsInits()) {
+      currentOutputTypes.push_back(getElementTypeOrSelf(output.getType()));
+    }
+
+    if (currentOutputTypes.size() != 1) {
+      return emitSilenceableFailure(current->getLoc())
+             << "expected single output, got " << currentOutputTypes.size();
+    }
+
+    Type currentOutputType = currentOutputTypes[0];
+    if (currentOutputType != targetOutputType) {
+      return emitSilenceableFailure(current->getLoc())
+             << "output type doesn't match: expected " << targetOutputType
+             << ", got " << currentOutputType;
+    }
+  }
+
+  // Get the actual size values for batch/m/n/k dimensions after verifying it's
+  // a contraction operation.
+  linalg::ContractionDimensions contractionDims =
+      linalg::inferContractionDims(linalgOp).value();
+  SmallVector<int64_t> iterationDomain = linalgOp.getStaticLoopRanges();
+  MLIRContext *context = current->getContext();
+  Builder builder(context);
+
+  auto iterationSizes = [&](ArrayRef<unsigned> dimIndices) {
+    return llvm::to_vector(
+        llvm::map_range(dimIndices, [&](unsigned dimIdx) -> Attribute {
```

**Comment:**
Use `map_to_vector`

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td:202`

```diff
@@ -172,4 +172,57 @@ def MatchRegionsOp : Op<Transform_Dialect, "iree.match.regions",
   let hasVerifier = 1;
 }
 
+def MatchContractionOp : Op<Transform_Dialect, "iree.match.is_contraction",
+    [MatchOpInterface,
+     SingleOpMatcher,
+     MemoryEffectsOpInterface]> {
+  let summary = [{Check whether the op is a contraction operation.}];
+  let description = [{
+    Matches operations that implement the ContractionOpInterface.
+    This includes operations like linalg.matmul, linalg.batch_matmul, etc.
+
+    Optionally matches specific indexing maps patterns.
+
+    #### Return modes
+
+    Succeeds if the operation is a contraction operation, and
+    produces a silenceable failure otherwise.
+
+    #### Results
+
+    Returns arrays of dimension sizes for each contraction dimension:
+    - batch_dims: Array of batch dimension sizes.
+    - m_dims: Array of M dimension sizes.
+    - n_dims: Array of N dimension sizes.
+    - k_dims: Array of K dimension sizes.
+  }];
+
+  let arguments = (ins
+    TransformHandleTypeInterface:$operand_handle,
+    OptionalAttr<AffineMapArrayAttr>:$indexing_maps,
```

**Comment:**
You can also see how we defined these in named matmul ops in linalg: https://github.com/llvm/llvm-project/blob/105fc90b6b96e0edb7529062fcba513a3a347820/mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp#L4053-L4076

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td:202`

```diff
@@ -172,4 +172,57 @@ def MatchRegionsOp : Op<Transform_Dialect, "iree.match.regions",
   let hasVerifier = 1;
 }
 
+def MatchContractionOp : Op<Transform_Dialect, "iree.match.is_contraction",
+    [MatchOpInterface,
+     SingleOpMatcher,
+     MemoryEffectsOpInterface]> {
+  let summary = [{Check whether the op is a contraction operation.}];
+  let description = [{
+    Matches operations that implement the ContractionOpInterface.
+    This includes operations like linalg.matmul, linalg.batch_matmul, etc.
+
+    Optionally matches specific indexing maps patterns.
+
+    #### Return modes
+
+    Succeeds if the operation is a contraction operation, and
+    produces a silenceable failure otherwise.
+
+    #### Results
+
+    Returns arrays of dimension sizes for each contraction dimension:
+    - batch_dims: Array of batch dimension sizes.
+    - m_dims: Array of M dimension sizes.
+    - n_dims: Array of N dimension sizes.
+    - k_dims: Array of K dimension sizes.
+  }];
+
+  let arguments = (ins
+    TransformHandleTypeInterface:$operand_handle,
+    OptionalAttr<AffineMapArrayAttr>:$indexing_maps,
```

**Comment:**
I think the main criteria has to be that we should be able to easily emit these matchers inside the tuner, given a linalg.generic. In this setting, getting indexing maps is easier than deciding the matmul variant. Let's leave this as-is then -- I think we don't need these `NN`/`NT`/etc. variants after all.

---

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp:232`

```diff
@@ -223,6 +224,112 @@ IREE::transform_dialect::MatchCastCompatibleDagFromRootOp::verify() {
   return success();
 }
 
+//===----------------------------------------------------------------------===//
+// MatchContractionOp
+//===----------------------------------------------------------------------===//
+
+// Helper function to get indexing maps for matmul variants.
+static ArrayAttr getMatmulIndexingMaps(StringAttr variant,
```

**Comment:**
This function is unused now

---

**File:** `compiler/src/iree/compiler/Preprocessing/Common/test/preprocessing_match_ops.mlir:447`

```diff
@@ -271,3 +271,177 @@ module attributes {transform.with_named_sequence} {
     transform.yield
   }
 }
+
+// -----
+
+// Verify that the basic contraction matcher works and can extract dimension sizes.
+
+#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
+#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
+#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
+#map_batch0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
+#map_batch1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
+#map_batch2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
+
+// CHECK-LABEL: func.func @op_matmul
+func.func @op_matmul(%input0: tensor<32x64xi8>, %input1: tensor<32x64xi8>, %dest: tensor<32x32xi32>) -> tensor<32x32xi32> {
+  // CHECK-NEXT: linalg.matmul
+  // CHECK-SAME:   match_status = "matched"
+  %res = linalg.matmul
+        indexing_maps = [#map0, #map1, #map2]
+        ins(%input0, %input1 : tensor<32x64xi8>, tensor<32x64xi8>)
+        outs(%dest : tensor<32x32xi32>) {match_status = "unmatched"} -> tensor<32x32xi32>
+  return %res : tensor<32x32xi32>
+}
+
+// CHECK-LABEL: func.func @op_batch_matmul
+func.func @op_batch_matmul(%input0: tensor<2x32x64xi8>, %input1: tensor<2x32x64xi8>, %dest: tensor<2x32x32xi32>) -> tensor<2x32x32xi32> {
+  // CHECK-NEXT: linalg.batch_matmul
+  // CHECK-SAME:   match_status = "matched"
+  %res = linalg.batch_matmul
+        indexing_maps = [#map_batch0, #map_batch1, #map_batch2]
+        ins(%input0, %input1 : tensor<2x32x64xi8>, tensor<2x32x64xi8>)
+        outs(%dest : tensor<2x32x32xi32>) {match_status = "unmatched"} -> tensor<2x32x32xi32>
+  return %res : tensor<2x32x32xi32>
+}
+
+// CHECK-LABEL: func.func @op_fill
+func.func @op_fill(%dest: tensor<32x64xf32>, %value: f32) -> tensor<32x64xf32> {
+  // CHECK-NEXT: linalg.fill
+  // CHECK-SAME:   match_status = "unmatched"
+  %res = linalg.fill ins(%value : f32) outs(%dest : tensor<32x64xf32>) {match_status = "unmatched"} -> tensor<32x64xf32>
+  return %res : tensor<32x64xf32>
+}
+
+module attributes {transform.with_named_sequence} {
+  transform.named_sequence @match_matmul(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
+    %batch_dims, %m_dims, %n_dims, %k_dims = transform.iree.match.is_contraction %op,
+      lhs_type = i8, rhs_type = i8, output_type = i32 :
+      (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
+    %c32 = transform.param.constant 32 : i64 -> !transform.param<i64>
+    transform.match.param.cmpi eq %m_dims, %c32 : !transform.param<i64>
+    transform.match.param.cmpi eq %n_dims, %c32 : !transform.param<i64>
+    transform.yield %op : !transform.any_op
+  }
+
+  transform.named_sequence @match_batch_matmul(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
+    %batch_dims, %m_dims, %n_dims, %k_dims = transform.iree.match.is_contraction %op,
+      lhs_type = i8, rhs_type = i8, output_type = i32 :
+      (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
+    %c2 = transform.param.constant 2 : i64 -> !transform.param<i64>
+    %c32 = transform.param.constant 32 : i64 -> !transform.param<i64>
+    transform.match.param.cmpi eq %batch_dims, %c2 : !transform.param<i64>
+    transform.match.param.cmpi eq %m_dims, %c32 : !transform.param<i64>
+    transform.match.param.cmpi eq %n_dims, %c32 : !transform.param<i64>
+    transform.yield %op : !transform.any_op
+  }
+
+  transform.named_sequence @annotate(%op: !transform.any_op {transform.readonly}) {
+    %0 = transform.param.constant "matched" -> !transform.any_param
+    transform.annotate %op "match_status" = %0 : !transform.any_op, !transform.any_param
+    transform.yield
+  }
+
+  transform.named_sequence @__transform_main(%module: !transform.any_op) {
+    transform.foreach_match in %module
+        @match_matmul -> @annotate,
+        @match_batch_matmul -> @annotate
+      : (!transform.any_op) -> (!transform.any_op)
+    transform.yield
+  }
+}
+
+// -----
+
+// Verify that operations with exact same matching indexing maps are matched correctly,
+// and operations with different indexing map patterns are not matched.
+
+#map_matmul0 = affine_map<(d0, d1, d2) -> (d0, d2)>
+#map_matmul1 = affine_map<(d0, d1, d2) -> (d1, d2)>
+#map_matmul2 = affine_map<(d0, d1, d2) -> (d0, d1)>
+#map_transpose_b = affine_map<(d0, d1, d2) -> (d2, d1)>  // Transpose_b RHS map.
+
+// CHECK-LABEL: func.func @op_matmul
+func.func @op_matmul(%input0: tensor<32x64xi8>, %input1: tensor<32x64xi8>, %input1_transposed: tensor<64x32xi8>, %dest: tensor<32x32xi32>) -> tensor<32x32xi32> {
+  // CHECK-NEXT: linalg.matmul
+  // CHECK-SAME:   indexing_maps_match = "matched"
+  %res1 = linalg.matmul
+        indexing_maps = [#map_matmul0, #map_matmul1, #map_matmul2]
+        ins(%input0, %input1 : tensor<32x64xi8>, tensor<32x64xi8>)
+        outs(%dest : tensor<32x32xi32>) {indexing_maps_match = "unmatched"} -> tensor<32x32xi32>
+
+  // Transpose_b matmul - should NOT match (different RHS indexing map).
+  // CHECK-NEXT: linalg.matmul
+  // CHECK-SAME:   indexing_maps_match = "unmatched"
+  %res2 = linalg.matmul
+        indexing_maps = [#map_matmul0, #map_transpose_b, #map_matmul2]
+        ins(%input0, %input1_transposed : tensor<32x64xi8>, tensor<64x32xi8>)
+        outs(%dest : tensor<32x32xi32>) {indexing_maps_match = "unmatched"} -> tensor<32x32xi32>
+
+  return %res1 : tensor<32x32xi32>
+}
+
+module attributes {transform.with_named_sequence} {
+  transform.named_sequence @match_correct_maps(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
+   %batch, %m, %n, %k = transform.iree.match.is_contraction %op,
+    lhs_type = i8, rhs_type = i8, output_type = i32 {indexing_maps = [#map_matmul0, #map_matmul1, #map_matmul2]} :
+    (!transform.any_op) ->
+    (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
+    transform.yield %op : !transform.any_op
+  }
+
+  transform.named_sequence @annotate_matched(%op: !transform.any_op {transform.readonly}) {
+    %0 = transform.param.constant "matched" -> !transform.any_param
+    transform.annotate %op "indexing_maps_match" = %0 : !transform.any_op, !transform.any_param
+    transform.yield
+  }
+
+  transform.named_sequence @__transform_main(%module: !transform.any_op) {
+    transform.foreach_match in %module
+        @match_correct_maps -> @annotate_matched
+      : (!transform.any_op) -> (!transform.any_op)
+    transform.yield
+  }
+}
+
+// -----
+
+// Verify that operations with different number of indexing maps are correctly not matched.
+
+#map_matmul0 = affine_map<(d0, d1, d2) -> (d0, d2)>
+#map_matmul1 = affine_map<(d0, d1, d2) -> (d1, d2)>
+#map_matmul2 = affine_map<(d0, d1, d2) -> (d0, d1)>
+
+// CHECK-LABEL: func.func @op_matmul
+func.func @op_matmul(%input0: tensor<32x64xi8>, %input1: tensor<32x64xi8>, %dest: tensor<32x32xi32>) -> tensor<32x32xi32> {
+  // CHECK-NEXT: linalg.matmul
+  // CHECK-SAME:   indexing_maps_match = "unmatched"
+  %res = linalg.matmul
+        indexing_maps = [#map_matmul0, #map_matmul1, #map_matmul2]
+        ins(%input0, %input1 : tensor<32x64xi8>, tensor<32x64xi8>)
+        outs(%dest : tensor<32x32xi32>) {indexing_maps_match = "unmatched"} -> tensor<32x32xi32>
+  return %res : tensor<32x32xi32>
+}
+
+module attributes {transform.with_named_sequence} {
+  transform.named_sequence @match_different_count(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
+    %batch, %m, %n, %k = transform.iree.match.is_contraction %op,
+      lhs_type = i8, rhs_type = i8, output_type = i32 {indexing_maps = [#map_matmul0, #map_matmul1, #map_matmul2, #map_matmul0]} :
+      (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
+    transform.yield %op : !transform.any_op
+  }
+
+  transform.named_sequence @annotate_matched(%op: !transform.any_op {transform.readonly}) {
+    %0 = transform.param.constant "matched" -> !transform.any_param
+    transform.annotate %op "indexing_maps_match" = %0 : !transform.any_op, !transform.any_param
+    transform.yield
+  }
+
+  transform.named_sequence @__transform_main(%module: !transform.any_op) {
+     // Should NOT match: operation has 3 indexing maps but matcher expects 4.
+    transform.foreach_match in %module
+        @match_different_count -> @annotate_matched
+      : (!transform.any_op) -> (!transform.any_op)
+    transform.yield
+  }
+}
```

**Comment:**
Can you add a testcase that shows if we match contractions that already have some attribute?

---


---


## [PR #21903](https://github.com/iree-org/iree/pull/21903): [DispatchCreation]: Add FormSplitReductionDispatchesPass support for ArgCompare op

### Review Summary

**COMMENTED** (2025-09-09)


**COMMENTED** (2025-09-09)


**COMMENTED** (2025-09-09)


**COMMENTED** (2025-09-09)


**COMMENTED** (2025-09-09)


**COMMENTED** (2025-09-10)

I don't have any other comments. Because I don't maintain this part of the codebase, I'd leave the final approval to Ian.


**COMMENTED** (2025-09-11)


### Code Comments

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1458`

```diff
@@ -1440,6 +1440,225 @@ LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
   return success();
 }
 
+FailureOr<SmallVector<Value>>
+ArgCompareOp::generateInitialTensorForPartialReduction(
+    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims) {
+  // Get the original init tensors.
+  Value valueInit = outputValue();
+  Value indexInit = outputIndex();
+
+  // Create tensors with the partial result shape.
+  Type valueElTy = getElementTypeOrSelf(valueInit.getType());
+  Type indexElTy = getElementTypeOrSelf(indexInit.getType());
+  SmallVector<OpFoldResult> partialResultShape(sizes.begin(), sizes.end());
+  Value emptyValueTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, valueElTy);
+  Value emptyIndexTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, indexElTy);
```

**Comment:**
These are deprecated, use free create functions instead: https://discourse.llvm.org/t/psa-opty-create-now-with-100-more-tab-complete/87339

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1468`

```diff
@@ -1440,6 +1440,225 @@ LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
   return success();
 }
 
+FailureOr<SmallVector<Value>>
+ArgCompareOp::generateInitialTensorForPartialReduction(
+    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims) {
+  // Get the original init tensors.
+  Value valueInit = outputValue();
+  Value indexInit = outputIndex();
+
+  // Create tensors with the partial result shape.
+  Type valueElTy = getElementTypeOrSelf(valueInit.getType());
+  Type indexElTy = getElementTypeOrSelf(indexInit.getType());
+  SmallVector<OpFoldResult> partialResultShape(sizes.begin(), sizes.end());
+  Value emptyValueTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, valueElTy);
+  Value emptyIndexTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, indexElTy);
+
+  // Broadcast init values to partial result shape for slicing inside
+  // scf.forall. Each tile in the parallel loop will extract slices from
+  // these broadcasted tensors to get initialized values for the ArgCompareOp.
+  // Example: tensor<64xf32> -> tensor<64x32xf32> for 32 reduction tiles along
+  // dim 1.
+  SmallVector<int64_t> broadcastDims;
+  for (unsigned dim : reductionDims) {
+    broadcastDims.push_back(static_cast<int64_t>(dim));
+  }
```

**Comment:**
use `llvm::to_vector_of<int64_t>`

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1532`

```diff
@@ -1440,6 +1440,225 @@ LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
   return success();
 }
 
+FailureOr<SmallVector<Value>>
+ArgCompareOp::generateInitialTensorForPartialReduction(
+    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims) {
+  // Get the original init tensors.
+  Value valueInit = outputValue();
+  Value indexInit = outputIndex();
+
+  // Create tensors with the partial result shape.
+  Type valueElTy = getElementTypeOrSelf(valueInit.getType());
+  Type indexElTy = getElementTypeOrSelf(indexInit.getType());
+  SmallVector<OpFoldResult> partialResultShape(sizes.begin(), sizes.end());
+  Value emptyValueTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, valueElTy);
+  Value emptyIndexTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, indexElTy);
+
+  // Broadcast init values to partial result shape for slicing inside
+  // scf.forall. Each tile in the parallel loop will extract slices from
+  // these broadcasted tensors to get initialized values for the ArgCompareOp.
+  // Example: tensor<64xf32> -> tensor<64x32xf32> for 32 reduction tiles along
+  // dim 1.
+  SmallVector<int64_t> broadcastDims;
+  for (unsigned dim : reductionDims) {
+    broadcastDims.push_back(static_cast<int64_t>(dim));
+  }
+  Operation *valueBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, valueInit, emptyValueTensor, broadcastDims);
+  Operation *indexBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, indexInit, emptyIndexTensor, broadcastDims);
+
+  return SmallVector<Value>{valueBroadcastOp->getResult(0),
+                            indexBroadcastOp->getResult(0)};
+}
+
+FailureOr<TilingResult> ArgCompareOp::tileToPartialReduction(
+    OpBuilder &b, Location loc, ReductionTilingStrategy strategy,
+    ValueRange init, ArrayRef<OpFoldResult> offsets,
+    ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims,
+    ArrayRef<OpFoldResult> splitReductionIvs) {
+  assert(strategy == ReductionTilingStrategy::PartialReductionOuterParallel &&
+         "Requires PartialReductionOuterParallel strategy");
+  OpBuilder::InsertionGuard guard(b);
+
+  int64_t rank = getInputRank();
+  int64_t reductionDim = getDimension();
+
+  assert(offsets.size() == static_cast<size_t>(rank) &&
+         "Unexpected offsets size");
+  assert(sizes.size() == static_cast<size_t>(rank) && "Unexpected sizes size");
+
+  SmallVector<Operation *> slices;
+  SmallVector<Value> tiledOperands;
+
+  // Extract a slice of the input operand.
+  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
+  Operation *inputSlice =
+      getSlice(b, loc, getInputValue(), offsets, sizes, strides);
+  tiledOperands.push_back(inputSlice->getResult(0));
+  slices.push_back(inputSlice);
+
+  // Extract slices of the init operands (partial results).
+  // For split-reduction, we need to slice the init values along
+  // the reduction dimension to get the appropriate chunk.
+  SmallVector<OpFoldResult> initOffsets, initSizes, initStrides;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim) {
+      initOffsets.push_back(splitReductionIvs[0]);
+      initSizes.push_back(b.getIndexAttr(1));
+    } else {
+      // For non-reduction dimensions, use the same offsets/sizes as input.
+      initOffsets.push_back(offsets[i]);
+      initSizes.push_back(sizes[i]);
+    }
+    initStrides.push_back(b.getIndexAttr(1));
+  }
+
+  const ShapedType &initValType = getOutputValueType();
+  const ShapedType &initIdxType = getOutputIndexType();
+  SmallVector<int64_t> resultValShape, resultIdxShape;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim)
+      continue;
```

**Comment:**
missing braces

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1536`

```diff
@@ -1440,6 +1440,225 @@ LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
   return success();
 }
 
+FailureOr<SmallVector<Value>>
+ArgCompareOp::generateInitialTensorForPartialReduction(
+    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims) {
+  // Get the original init tensors.
+  Value valueInit = outputValue();
+  Value indexInit = outputIndex();
+
+  // Create tensors with the partial result shape.
+  Type valueElTy = getElementTypeOrSelf(valueInit.getType());
+  Type indexElTy = getElementTypeOrSelf(indexInit.getType());
+  SmallVector<OpFoldResult> partialResultShape(sizes.begin(), sizes.end());
+  Value emptyValueTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, valueElTy);
+  Value emptyIndexTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, indexElTy);
+
+  // Broadcast init values to partial result shape for slicing inside
+  // scf.forall. Each tile in the parallel loop will extract slices from
+  // these broadcasted tensors to get initialized values for the ArgCompareOp.
+  // Example: tensor<64xf32> -> tensor<64x32xf32> for 32 reduction tiles along
+  // dim 1.
+  SmallVector<int64_t> broadcastDims;
+  for (unsigned dim : reductionDims) {
+    broadcastDims.push_back(static_cast<int64_t>(dim));
+  }
+  Operation *valueBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, valueInit, emptyValueTensor, broadcastDims);
+  Operation *indexBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, indexInit, emptyIndexTensor, broadcastDims);
+
+  return SmallVector<Value>{valueBroadcastOp->getResult(0),
+                            indexBroadcastOp->getResult(0)};
+}
+
+FailureOr<TilingResult> ArgCompareOp::tileToPartialReduction(
+    OpBuilder &b, Location loc, ReductionTilingStrategy strategy,
+    ValueRange init, ArrayRef<OpFoldResult> offsets,
+    ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims,
+    ArrayRef<OpFoldResult> splitReductionIvs) {
+  assert(strategy == ReductionTilingStrategy::PartialReductionOuterParallel &&
+         "Requires PartialReductionOuterParallel strategy");
+  OpBuilder::InsertionGuard guard(b);
+
+  int64_t rank = getInputRank();
+  int64_t reductionDim = getDimension();
+
+  assert(offsets.size() == static_cast<size_t>(rank) &&
+         "Unexpected offsets size");
+  assert(sizes.size() == static_cast<size_t>(rank) && "Unexpected sizes size");
+
+  SmallVector<Operation *> slices;
+  SmallVector<Value> tiledOperands;
+
+  // Extract a slice of the input operand.
+  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
+  Operation *inputSlice =
+      getSlice(b, loc, getInputValue(), offsets, sizes, strides);
+  tiledOperands.push_back(inputSlice->getResult(0));
+  slices.push_back(inputSlice);
+
+  // Extract slices of the init operands (partial results).
+  // For split-reduction, we need to slice the init values along
+  // the reduction dimension to get the appropriate chunk.
+  SmallVector<OpFoldResult> initOffsets, initSizes, initStrides;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim) {
+      initOffsets.push_back(splitReductionIvs[0]);
+      initSizes.push_back(b.getIndexAttr(1));
+    } else {
+      // For non-reduction dimensions, use the same offsets/sizes as input.
+      initOffsets.push_back(offsets[i]);
+      initSizes.push_back(sizes[i]);
+    }
+    initStrides.push_back(b.getIndexAttr(1));
+  }
+
+  const ShapedType &initValType = getOutputValueType();
+  const ShapedType &initIdxType = getOutputIndexType();
+  SmallVector<int64_t> resultValShape, resultIdxShape;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim)
+      continue;
+
+    if (auto sizeAttr = dyn_cast<Attribute>(initSizes[i])) {
+      int64_t size = (cast<IntegerAttr>(sizeAttr)).getInt();
```

**Comment:**
```suggestion
    if (auto sizeAttr = dyn_cast<IntegerAttr>(initSizes[i])) {
      int64_t size = sizeAttr.getInt();
```

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1543`

```diff
@@ -1440,6 +1440,225 @@ LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
   return success();
 }
 
+FailureOr<SmallVector<Value>>
+ArgCompareOp::generateInitialTensorForPartialReduction(
+    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims) {
+  // Get the original init tensors.
+  Value valueInit = outputValue();
+  Value indexInit = outputIndex();
+
+  // Create tensors with the partial result shape.
+  Type valueElTy = getElementTypeOrSelf(valueInit.getType());
+  Type indexElTy = getElementTypeOrSelf(indexInit.getType());
+  SmallVector<OpFoldResult> partialResultShape(sizes.begin(), sizes.end());
+  Value emptyValueTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, valueElTy);
+  Value emptyIndexTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, indexElTy);
+
+  // Broadcast init values to partial result shape for slicing inside
+  // scf.forall. Each tile in the parallel loop will extract slices from
+  // these broadcasted tensors to get initialized values for the ArgCompareOp.
+  // Example: tensor<64xf32> -> tensor<64x32xf32> for 32 reduction tiles along
+  // dim 1.
+  SmallVector<int64_t> broadcastDims;
+  for (unsigned dim : reductionDims) {
+    broadcastDims.push_back(static_cast<int64_t>(dim));
+  }
+  Operation *valueBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, valueInit, emptyValueTensor, broadcastDims);
+  Operation *indexBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, indexInit, emptyIndexTensor, broadcastDims);
+
+  return SmallVector<Value>{valueBroadcastOp->getResult(0),
+                            indexBroadcastOp->getResult(0)};
+}
+
+FailureOr<TilingResult> ArgCompareOp::tileToPartialReduction(
+    OpBuilder &b, Location loc, ReductionTilingStrategy strategy,
+    ValueRange init, ArrayRef<OpFoldResult> offsets,
+    ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims,
+    ArrayRef<OpFoldResult> splitReductionIvs) {
+  assert(strategy == ReductionTilingStrategy::PartialReductionOuterParallel &&
+         "Requires PartialReductionOuterParallel strategy");
+  OpBuilder::InsertionGuard guard(b);
+
+  int64_t rank = getInputRank();
+  int64_t reductionDim = getDimension();
+
+  assert(offsets.size() == static_cast<size_t>(rank) &&
+         "Unexpected offsets size");
+  assert(sizes.size() == static_cast<size_t>(rank) && "Unexpected sizes size");
+
+  SmallVector<Operation *> slices;
+  SmallVector<Value> tiledOperands;
+
+  // Extract a slice of the input operand.
+  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
+  Operation *inputSlice =
+      getSlice(b, loc, getInputValue(), offsets, sizes, strides);
+  tiledOperands.push_back(inputSlice->getResult(0));
+  slices.push_back(inputSlice);
+
+  // Extract slices of the init operands (partial results).
+  // For split-reduction, we need to slice the init values along
+  // the reduction dimension to get the appropriate chunk.
+  SmallVector<OpFoldResult> initOffsets, initSizes, initStrides;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim) {
+      initOffsets.push_back(splitReductionIvs[0]);
+      initSizes.push_back(b.getIndexAttr(1));
+    } else {
+      // For non-reduction dimensions, use the same offsets/sizes as input.
+      initOffsets.push_back(offsets[i]);
+      initSizes.push_back(sizes[i]);
+    }
+    initStrides.push_back(b.getIndexAttr(1));
+  }
+
+  const ShapedType &initValType = getOutputValueType();
+  const ShapedType &initIdxType = getOutputIndexType();
+  SmallVector<int64_t> resultValShape, resultIdxShape;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim)
+      continue;
+
+    if (auto sizeAttr = dyn_cast<Attribute>(initSizes[i])) {
+      int64_t size = (cast<IntegerAttr>(sizeAttr)).getInt();
+      resultValShape.push_back(size);
+      resultIdxShape.push_back(size);
+      continue;
+    }
+
+    resultValShape.push_back(ShapedType::kDynamic);
+    resultIdxShape.push_back(ShapedType::kDynamic);
+  }
+
+  RankedTensorType sliceValResultType =
+      RankedTensorType::get(resultValShape, initValType.getElementType(),
+                            cast<RankedTensorType>(initValType).getEncoding());
+  RankedTensorType sliceIdxResultType =
+      RankedTensorType::get(resultIdxShape, initIdxType.getElementType(),
```

**Comment:**
```suggestion
  auto sliceValResultType =
      RankedTensorType::get(resultValShape, initValType.getElementType(),
                            cast<RankedTensorType>(initValType).getEncoding());
  auto sliceIdxResultType =
      RankedTensorType::get(resultIdxShape, initIdxType.getElementType(),
```
See https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1566`

```diff
@@ -1440,6 +1440,225 @@ LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
   return success();
 }
 
+FailureOr<SmallVector<Value>>
+ArgCompareOp::generateInitialTensorForPartialReduction(
+    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims) {
+  // Get the original init tensors.
+  Value valueInit = outputValue();
+  Value indexInit = outputIndex();
+
+  // Create tensors with the partial result shape.
+  Type valueElTy = getElementTypeOrSelf(valueInit.getType());
+  Type indexElTy = getElementTypeOrSelf(indexInit.getType());
+  SmallVector<OpFoldResult> partialResultShape(sizes.begin(), sizes.end());
+  Value emptyValueTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, valueElTy);
+  Value emptyIndexTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, indexElTy);
+
+  // Broadcast init values to partial result shape for slicing inside
+  // scf.forall. Each tile in the parallel loop will extract slices from
+  // these broadcasted tensors to get initialized values for the ArgCompareOp.
+  // Example: tensor<64xf32> -> tensor<64x32xf32> for 32 reduction tiles along
+  // dim 1.
+  SmallVector<int64_t> broadcastDims;
+  for (unsigned dim : reductionDims) {
+    broadcastDims.push_back(static_cast<int64_t>(dim));
+  }
+  Operation *valueBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, valueInit, emptyValueTensor, broadcastDims);
+  Operation *indexBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, indexInit, emptyIndexTensor, broadcastDims);
+
+  return SmallVector<Value>{valueBroadcastOp->getResult(0),
+                            indexBroadcastOp->getResult(0)};
+}
+
+FailureOr<TilingResult> ArgCompareOp::tileToPartialReduction(
+    OpBuilder &b, Location loc, ReductionTilingStrategy strategy,
+    ValueRange init, ArrayRef<OpFoldResult> offsets,
+    ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims,
+    ArrayRef<OpFoldResult> splitReductionIvs) {
+  assert(strategy == ReductionTilingStrategy::PartialReductionOuterParallel &&
+         "Requires PartialReductionOuterParallel strategy");
+  OpBuilder::InsertionGuard guard(b);
+
+  int64_t rank = getInputRank();
+  int64_t reductionDim = getDimension();
+
+  assert(offsets.size() == static_cast<size_t>(rank) &&
+         "Unexpected offsets size");
+  assert(sizes.size() == static_cast<size_t>(rank) && "Unexpected sizes size");
+
+  SmallVector<Operation *> slices;
+  SmallVector<Value> tiledOperands;
+
+  // Extract a slice of the input operand.
+  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
+  Operation *inputSlice =
+      getSlice(b, loc, getInputValue(), offsets, sizes, strides);
+  tiledOperands.push_back(inputSlice->getResult(0));
+  slices.push_back(inputSlice);
+
+  // Extract slices of the init operands (partial results).
+  // For split-reduction, we need to slice the init values along
+  // the reduction dimension to get the appropriate chunk.
+  SmallVector<OpFoldResult> initOffsets, initSizes, initStrides;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim) {
+      initOffsets.push_back(splitReductionIvs[0]);
+      initSizes.push_back(b.getIndexAttr(1));
+    } else {
+      // For non-reduction dimensions, use the same offsets/sizes as input.
+      initOffsets.push_back(offsets[i]);
+      initSizes.push_back(sizes[i]);
+    }
+    initStrides.push_back(b.getIndexAttr(1));
+  }
+
+  const ShapedType &initValType = getOutputValueType();
+  const ShapedType &initIdxType = getOutputIndexType();
+  SmallVector<int64_t> resultValShape, resultIdxShape;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim)
+      continue;
+
+    if (auto sizeAttr = dyn_cast<Attribute>(initSizes[i])) {
+      int64_t size = (cast<IntegerAttr>(sizeAttr)).getInt();
+      resultValShape.push_back(size);
+      resultIdxShape.push_back(size);
+      continue;
+    }
+
+    resultValShape.push_back(ShapedType::kDynamic);
+    resultIdxShape.push_back(ShapedType::kDynamic);
+  }
+
+  RankedTensorType sliceValResultType =
+      RankedTensorType::get(resultValShape, initValType.getElementType(),
+                            cast<RankedTensorType>(initValType).getEncoding());
+  RankedTensorType sliceIdxResultType =
+      RankedTensorType::get(resultIdxShape, initIdxType.getElementType(),
+                            cast<RankedTensorType>(initIdxType).getEncoding());
+
+  auto initValSlice = tensor::ExtractSliceOp::create(
+      b, loc, sliceValResultType, init[0], initOffsets, initSizes, initStrides);
+  tiledOperands.push_back(initValSlice.getResult());
+  slices.push_back(initValSlice);
+
+  auto initIdxSlice = tensor::ExtractSliceOp::create(
+      b, loc, sliceIdxResultType, init[1], initOffsets, initSizes, initStrides);
+  tiledOperands.push_back(initIdxSlice.getResult());
+  slices.push_back(initIdxSlice);
+
+  // Create index_base for this chunk.
+  Value tileIndex =
+      getValueOrCreateConstantIndexOp(b, loc, splitReductionIvs[0]);
+  Value tileSize = getValueOrCreateConstantIndexOp(b, loc, sizes[reductionDim]);
+  Value tileStartIndex = b.create<arith::MulIOp>(loc, tileIndex, tileSize);
+
+  Value newIndexBase;
+  if (Value globalIndexBase = getIndexBase()) {
+    // Add chunk start to existing index_base.
+    newIndexBase =
+        b.create<arith::AddIOp>(loc, globalIndexBase, tileStartIndex);
```

**Comment:**
also here

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1572`

```diff
@@ -1440,6 +1440,225 @@ LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
   return success();
 }
 
+FailureOr<SmallVector<Value>>
+ArgCompareOp::generateInitialTensorForPartialReduction(
+    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims) {
+  // Get the original init tensors.
+  Value valueInit = outputValue();
+  Value indexInit = outputIndex();
+
+  // Create tensors with the partial result shape.
+  Type valueElTy = getElementTypeOrSelf(valueInit.getType());
+  Type indexElTy = getElementTypeOrSelf(indexInit.getType());
+  SmallVector<OpFoldResult> partialResultShape(sizes.begin(), sizes.end());
+  Value emptyValueTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, valueElTy);
+  Value emptyIndexTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, indexElTy);
+
+  // Broadcast init values to partial result shape for slicing inside
+  // scf.forall. Each tile in the parallel loop will extract slices from
+  // these broadcasted tensors to get initialized values for the ArgCompareOp.
+  // Example: tensor<64xf32> -> tensor<64x32xf32> for 32 reduction tiles along
+  // dim 1.
+  SmallVector<int64_t> broadcastDims;
+  for (unsigned dim : reductionDims) {
+    broadcastDims.push_back(static_cast<int64_t>(dim));
+  }
+  Operation *valueBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, valueInit, emptyValueTensor, broadcastDims);
+  Operation *indexBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, indexInit, emptyIndexTensor, broadcastDims);
+
+  return SmallVector<Value>{valueBroadcastOp->getResult(0),
+                            indexBroadcastOp->getResult(0)};
+}
+
+FailureOr<TilingResult> ArgCompareOp::tileToPartialReduction(
+    OpBuilder &b, Location loc, ReductionTilingStrategy strategy,
+    ValueRange init, ArrayRef<OpFoldResult> offsets,
+    ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims,
+    ArrayRef<OpFoldResult> splitReductionIvs) {
+  assert(strategy == ReductionTilingStrategy::PartialReductionOuterParallel &&
+         "Requires PartialReductionOuterParallel strategy");
+  OpBuilder::InsertionGuard guard(b);
+
+  int64_t rank = getInputRank();
+  int64_t reductionDim = getDimension();
+
+  assert(offsets.size() == static_cast<size_t>(rank) &&
+         "Unexpected offsets size");
+  assert(sizes.size() == static_cast<size_t>(rank) && "Unexpected sizes size");
+
+  SmallVector<Operation *> slices;
+  SmallVector<Value> tiledOperands;
+
+  // Extract a slice of the input operand.
+  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
+  Operation *inputSlice =
+      getSlice(b, loc, getInputValue(), offsets, sizes, strides);
+  tiledOperands.push_back(inputSlice->getResult(0));
+  slices.push_back(inputSlice);
+
+  // Extract slices of the init operands (partial results).
+  // For split-reduction, we need to slice the init values along
+  // the reduction dimension to get the appropriate chunk.
+  SmallVector<OpFoldResult> initOffsets, initSizes, initStrides;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim) {
+      initOffsets.push_back(splitReductionIvs[0]);
+      initSizes.push_back(b.getIndexAttr(1));
+    } else {
+      // For non-reduction dimensions, use the same offsets/sizes as input.
+      initOffsets.push_back(offsets[i]);
+      initSizes.push_back(sizes[i]);
+    }
+    initStrides.push_back(b.getIndexAttr(1));
+  }
+
+  const ShapedType &initValType = getOutputValueType();
+  const ShapedType &initIdxType = getOutputIndexType();
+  SmallVector<int64_t> resultValShape, resultIdxShape;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim)
+      continue;
+
+    if (auto sizeAttr = dyn_cast<Attribute>(initSizes[i])) {
+      int64_t size = (cast<IntegerAttr>(sizeAttr)).getInt();
+      resultValShape.push_back(size);
+      resultIdxShape.push_back(size);
+      continue;
+    }
+
+    resultValShape.push_back(ShapedType::kDynamic);
+    resultIdxShape.push_back(ShapedType::kDynamic);
+  }
+
+  RankedTensorType sliceValResultType =
+      RankedTensorType::get(resultValShape, initValType.getElementType(),
+                            cast<RankedTensorType>(initValType).getEncoding());
+  RankedTensorType sliceIdxResultType =
+      RankedTensorType::get(resultIdxShape, initIdxType.getElementType(),
+                            cast<RankedTensorType>(initIdxType).getEncoding());
+
+  auto initValSlice = tensor::ExtractSliceOp::create(
+      b, loc, sliceValResultType, init[0], initOffsets, initSizes, initStrides);
+  tiledOperands.push_back(initValSlice.getResult());
+  slices.push_back(initValSlice);
+
+  auto initIdxSlice = tensor::ExtractSliceOp::create(
+      b, loc, sliceIdxResultType, init[1], initOffsets, initSizes, initStrides);
+  tiledOperands.push_back(initIdxSlice.getResult());
+  slices.push_back(initIdxSlice);
+
+  // Create index_base for this chunk.
+  Value tileIndex =
+      getValueOrCreateConstantIndexOp(b, loc, splitReductionIvs[0]);
+  Value tileSize = getValueOrCreateConstantIndexOp(b, loc, sizes[reductionDim]);
+  Value tileStartIndex = b.create<arith::MulIOp>(loc, tileIndex, tileSize);
+
+  Value newIndexBase;
+  if (Value globalIndexBase = getIndexBase()) {
+    // Add chunk start to existing index_base.
+    newIndexBase =
+        b.create<arith::AddIOp>(loc, globalIndexBase, tileStartIndex);
+  } else {
+    // Use chunk start as index_base.
+    newIndexBase = tileStartIndex;
+  }
+
+  SmallVector<Type> resultTypes;
```

**Comment:**
This can be an array of exactly two elements

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1581`

```diff
@@ -1440,6 +1440,225 @@ LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
   return success();
 }
 
+FailureOr<SmallVector<Value>>
+ArgCompareOp::generateInitialTensorForPartialReduction(
+    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims) {
+  // Get the original init tensors.
+  Value valueInit = outputValue();
+  Value indexInit = outputIndex();
+
+  // Create tensors with the partial result shape.
+  Type valueElTy = getElementTypeOrSelf(valueInit.getType());
+  Type indexElTy = getElementTypeOrSelf(indexInit.getType());
+  SmallVector<OpFoldResult> partialResultShape(sizes.begin(), sizes.end());
+  Value emptyValueTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, valueElTy);
+  Value emptyIndexTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, indexElTy);
+
+  // Broadcast init values to partial result shape for slicing inside
+  // scf.forall. Each tile in the parallel loop will extract slices from
+  // these broadcasted tensors to get initialized values for the ArgCompareOp.
+  // Example: tensor<64xf32> -> tensor<64x32xf32> for 32 reduction tiles along
+  // dim 1.
+  SmallVector<int64_t> broadcastDims;
+  for (unsigned dim : reductionDims) {
+    broadcastDims.push_back(static_cast<int64_t>(dim));
+  }
+  Operation *valueBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, valueInit, emptyValueTensor, broadcastDims);
+  Operation *indexBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, indexInit, emptyIndexTensor, broadcastDims);
+
+  return SmallVector<Value>{valueBroadcastOp->getResult(0),
+                            indexBroadcastOp->getResult(0)};
+}
+
+FailureOr<TilingResult> ArgCompareOp::tileToPartialReduction(
+    OpBuilder &b, Location loc, ReductionTilingStrategy strategy,
+    ValueRange init, ArrayRef<OpFoldResult> offsets,
+    ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims,
+    ArrayRef<OpFoldResult> splitReductionIvs) {
+  assert(strategy == ReductionTilingStrategy::PartialReductionOuterParallel &&
+         "Requires PartialReductionOuterParallel strategy");
+  OpBuilder::InsertionGuard guard(b);
+
+  int64_t rank = getInputRank();
+  int64_t reductionDim = getDimension();
+
+  assert(offsets.size() == static_cast<size_t>(rank) &&
+         "Unexpected offsets size");
+  assert(sizes.size() == static_cast<size_t>(rank) && "Unexpected sizes size");
+
+  SmallVector<Operation *> slices;
+  SmallVector<Value> tiledOperands;
+
+  // Extract a slice of the input operand.
+  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
+  Operation *inputSlice =
+      getSlice(b, loc, getInputValue(), offsets, sizes, strides);
+  tiledOperands.push_back(inputSlice->getResult(0));
+  slices.push_back(inputSlice);
+
+  // Extract slices of the init operands (partial results).
+  // For split-reduction, we need to slice the init values along
+  // the reduction dimension to get the appropriate chunk.
+  SmallVector<OpFoldResult> initOffsets, initSizes, initStrides;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim) {
+      initOffsets.push_back(splitReductionIvs[0]);
+      initSizes.push_back(b.getIndexAttr(1));
+    } else {
+      // For non-reduction dimensions, use the same offsets/sizes as input.
+      initOffsets.push_back(offsets[i]);
+      initSizes.push_back(sizes[i]);
+    }
+    initStrides.push_back(b.getIndexAttr(1));
+  }
+
+  const ShapedType &initValType = getOutputValueType();
+  const ShapedType &initIdxType = getOutputIndexType();
+  SmallVector<int64_t> resultValShape, resultIdxShape;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim)
+      continue;
+
+    if (auto sizeAttr = dyn_cast<Attribute>(initSizes[i])) {
+      int64_t size = (cast<IntegerAttr>(sizeAttr)).getInt();
+      resultValShape.push_back(size);
+      resultIdxShape.push_back(size);
+      continue;
+    }
+
+    resultValShape.push_back(ShapedType::kDynamic);
+    resultIdxShape.push_back(ShapedType::kDynamic);
+  }
+
+  RankedTensorType sliceValResultType =
+      RankedTensorType::get(resultValShape, initValType.getElementType(),
+                            cast<RankedTensorType>(initValType).getEncoding());
+  RankedTensorType sliceIdxResultType =
+      RankedTensorType::get(resultIdxShape, initIdxType.getElementType(),
+                            cast<RankedTensorType>(initIdxType).getEncoding());
+
+  auto initValSlice = tensor::ExtractSliceOp::create(
+      b, loc, sliceValResultType, init[0], initOffsets, initSizes, initStrides);
+  tiledOperands.push_back(initValSlice.getResult());
+  slices.push_back(initValSlice);
+
+  auto initIdxSlice = tensor::ExtractSliceOp::create(
+      b, loc, sliceIdxResultType, init[1], initOffsets, initSizes, initStrides);
+  tiledOperands.push_back(initIdxSlice.getResult());
+  slices.push_back(initIdxSlice);
+
+  // Create index_base for this chunk.
+  Value tileIndex =
+      getValueOrCreateConstantIndexOp(b, loc, splitReductionIvs[0]);
+  Value tileSize = getValueOrCreateConstantIndexOp(b, loc, sizes[reductionDim]);
+  Value tileStartIndex = b.create<arith::MulIOp>(loc, tileIndex, tileSize);
+
+  Value newIndexBase;
+  if (Value globalIndexBase = getIndexBase()) {
+    // Add chunk start to existing index_base.
+    newIndexBase =
+        b.create<arith::AddIOp>(loc, globalIndexBase, tileStartIndex);
+  } else {
+    // Use chunk start as index_base.
+    newIndexBase = tileStartIndex;
+  }
+
+  SmallVector<Type> resultTypes;
+  if (hasPureTensorSemantics()) {
+    resultTypes.push_back(initValSlice->getResult(0).getType());
+    resultTypes.push_back(initIdxSlice->getResult(0).getType());
+  }
+
+  // Create the tiled operation with adjusted index_base.
+  SmallVector<Value> operands = std::move(tiledOperands);
+  operands.push_back(newIndexBase);
+  Operation *tiledArgmaxOp = b.create<ArgCompareOp>(
```

**Comment:**
also here

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1603`

```diff
@@ -1440,6 +1440,225 @@ LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
   return success();
 }
 
+FailureOr<SmallVector<Value>>
+ArgCompareOp::generateInitialTensorForPartialReduction(
+    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims) {
+  // Get the original init tensors.
+  Value valueInit = outputValue();
+  Value indexInit = outputIndex();
+
+  // Create tensors with the partial result shape.
+  Type valueElTy = getElementTypeOrSelf(valueInit.getType());
+  Type indexElTy = getElementTypeOrSelf(indexInit.getType());
+  SmallVector<OpFoldResult> partialResultShape(sizes.begin(), sizes.end());
+  Value emptyValueTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, valueElTy);
+  Value emptyIndexTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, indexElTy);
+
+  // Broadcast init values to partial result shape for slicing inside
+  // scf.forall. Each tile in the parallel loop will extract slices from
+  // these broadcasted tensors to get initialized values for the ArgCompareOp.
+  // Example: tensor<64xf32> -> tensor<64x32xf32> for 32 reduction tiles along
+  // dim 1.
+  SmallVector<int64_t> broadcastDims;
+  for (unsigned dim : reductionDims) {
+    broadcastDims.push_back(static_cast<int64_t>(dim));
+  }
+  Operation *valueBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, valueInit, emptyValueTensor, broadcastDims);
+  Operation *indexBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, indexInit, emptyIndexTensor, broadcastDims);
+
+  return SmallVector<Value>{valueBroadcastOp->getResult(0),
+                            indexBroadcastOp->getResult(0)};
+}
+
+FailureOr<TilingResult> ArgCompareOp::tileToPartialReduction(
+    OpBuilder &b, Location loc, ReductionTilingStrategy strategy,
+    ValueRange init, ArrayRef<OpFoldResult> offsets,
+    ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims,
+    ArrayRef<OpFoldResult> splitReductionIvs) {
+  assert(strategy == ReductionTilingStrategy::PartialReductionOuterParallel &&
+         "Requires PartialReductionOuterParallel strategy");
+  OpBuilder::InsertionGuard guard(b);
+
+  int64_t rank = getInputRank();
+  int64_t reductionDim = getDimension();
+
+  assert(offsets.size() == static_cast<size_t>(rank) &&
+         "Unexpected offsets size");
+  assert(sizes.size() == static_cast<size_t>(rank) && "Unexpected sizes size");
+
+  SmallVector<Operation *> slices;
+  SmallVector<Value> tiledOperands;
+
+  // Extract a slice of the input operand.
+  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
+  Operation *inputSlice =
+      getSlice(b, loc, getInputValue(), offsets, sizes, strides);
+  tiledOperands.push_back(inputSlice->getResult(0));
+  slices.push_back(inputSlice);
+
+  // Extract slices of the init operands (partial results).
+  // For split-reduction, we need to slice the init values along
+  // the reduction dimension to get the appropriate chunk.
+  SmallVector<OpFoldResult> initOffsets, initSizes, initStrides;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim) {
+      initOffsets.push_back(splitReductionIvs[0]);
+      initSizes.push_back(b.getIndexAttr(1));
+    } else {
+      // For non-reduction dimensions, use the same offsets/sizes as input.
+      initOffsets.push_back(offsets[i]);
+      initSizes.push_back(sizes[i]);
+    }
+    initStrides.push_back(b.getIndexAttr(1));
+  }
+
+  const ShapedType &initValType = getOutputValueType();
+  const ShapedType &initIdxType = getOutputIndexType();
+  SmallVector<int64_t> resultValShape, resultIdxShape;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim)
+      continue;
+
+    if (auto sizeAttr = dyn_cast<Attribute>(initSizes[i])) {
+      int64_t size = (cast<IntegerAttr>(sizeAttr)).getInt();
+      resultValShape.push_back(size);
+      resultIdxShape.push_back(size);
+      continue;
+    }
+
+    resultValShape.push_back(ShapedType::kDynamic);
+    resultIdxShape.push_back(ShapedType::kDynamic);
+  }
+
+  RankedTensorType sliceValResultType =
+      RankedTensorType::get(resultValShape, initValType.getElementType(),
+                            cast<RankedTensorType>(initValType).getEncoding());
+  RankedTensorType sliceIdxResultType =
+      RankedTensorType::get(resultIdxShape, initIdxType.getElementType(),
+                            cast<RankedTensorType>(initIdxType).getEncoding());
+
+  auto initValSlice = tensor::ExtractSliceOp::create(
+      b, loc, sliceValResultType, init[0], initOffsets, initSizes, initStrides);
+  tiledOperands.push_back(initValSlice.getResult());
+  slices.push_back(initValSlice);
+
+  auto initIdxSlice = tensor::ExtractSliceOp::create(
+      b, loc, sliceIdxResultType, init[1], initOffsets, initSizes, initStrides);
+  tiledOperands.push_back(initIdxSlice.getResult());
+  slices.push_back(initIdxSlice);
+
+  // Create index_base for this chunk.
+  Value tileIndex =
+      getValueOrCreateConstantIndexOp(b, loc, splitReductionIvs[0]);
+  Value tileSize = getValueOrCreateConstantIndexOp(b, loc, sizes[reductionDim]);
+  Value tileStartIndex = b.create<arith::MulIOp>(loc, tileIndex, tileSize);
+
+  Value newIndexBase;
+  if (Value globalIndexBase = getIndexBase()) {
+    // Add chunk start to existing index_base.
+    newIndexBase =
+        b.create<arith::AddIOp>(loc, globalIndexBase, tileStartIndex);
+  } else {
+    // Use chunk start as index_base.
+    newIndexBase = tileStartIndex;
+  }
+
+  SmallVector<Type> resultTypes;
+  if (hasPureTensorSemantics()) {
+    resultTypes.push_back(initValSlice->getResult(0).getType());
+    resultTypes.push_back(initIdxSlice->getResult(0).getType());
+  }
+
+  // Create the tiled operation with adjusted index_base.
+  SmallVector<Value> operands = std::move(tiledOperands);
+  operands.push_back(newIndexBase);
+  Operation *tiledArgmaxOp = b.create<ArgCompareOp>(
+      loc, resultTypes,
+      operands[0],                          // input slice.
+      ValueRange{operands[1], operands[2]}, // init slices.
+      operands[3],                          // index_base.
+      reductionDim                          // dimension.
+  );
+
+  // Copy the region.
+  Region &targetRegion = tiledArgmaxOp->getRegion(0);
+  Region &sourceRegion = getRegion();
+  targetRegion.takeBody(sourceRegion);
+
+  return TilingResult{
+      {tiledArgmaxOp}, SmallVector<Value>(tiledArgmaxOp->getResults()), slices};
+}
+
+FailureOr<MergeResult>
+ArgCompareOp::mergeReductions(OpBuilder &b, Location loc,
+                              ValueRange partialReduce,
+                              const llvm::SetVector<unsigned> &reductionDims) {
+  SmallVector<int64_t> mergeReductionDims(reductionDims.begin(),
+                                          reductionDims.end());
```

**Comment:**
use `llvm::to_vector`

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1560`

```diff
@@ -1440,6 +1440,225 @@ LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
   return success();
 }
 
+FailureOr<SmallVector<Value>>
+ArgCompareOp::generateInitialTensorForPartialReduction(
+    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims) {
+  // Get the original init tensors.
+  Value valueInit = outputValue();
+  Value indexInit = outputIndex();
+
+  // Create tensors with the partial result shape.
+  Type valueElTy = getElementTypeOrSelf(valueInit.getType());
+  Type indexElTy = getElementTypeOrSelf(indexInit.getType());
+  SmallVector<OpFoldResult> partialResultShape(sizes.begin(), sizes.end());
+  Value emptyValueTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, valueElTy);
+  Value emptyIndexTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, indexElTy);
+
+  // Broadcast init values to partial result shape for slicing inside
+  // scf.forall. Each tile in the parallel loop will extract slices from
+  // these broadcasted tensors to get initialized values for the ArgCompareOp.
+  // Example: tensor<64xf32> -> tensor<64x32xf32> for 32 reduction tiles along
+  // dim 1.
+  SmallVector<int64_t> broadcastDims;
+  for (unsigned dim : reductionDims) {
+    broadcastDims.push_back(static_cast<int64_t>(dim));
+  }
+  Operation *valueBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, valueInit, emptyValueTensor, broadcastDims);
+  Operation *indexBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, indexInit, emptyIndexTensor, broadcastDims);
+
+  return SmallVector<Value>{valueBroadcastOp->getResult(0),
+                            indexBroadcastOp->getResult(0)};
+}
+
+FailureOr<TilingResult> ArgCompareOp::tileToPartialReduction(
+    OpBuilder &b, Location loc, ReductionTilingStrategy strategy,
+    ValueRange init, ArrayRef<OpFoldResult> offsets,
+    ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims,
+    ArrayRef<OpFoldResult> splitReductionIvs) {
+  assert(strategy == ReductionTilingStrategy::PartialReductionOuterParallel &&
+         "Requires PartialReductionOuterParallel strategy");
+  OpBuilder::InsertionGuard guard(b);
+
+  int64_t rank = getInputRank();
+  int64_t reductionDim = getDimension();
+
+  assert(offsets.size() == static_cast<size_t>(rank) &&
+         "Unexpected offsets size");
+  assert(sizes.size() == static_cast<size_t>(rank) && "Unexpected sizes size");
+
+  SmallVector<Operation *> slices;
+  SmallVector<Value> tiledOperands;
+
+  // Extract a slice of the input operand.
+  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
+  Operation *inputSlice =
+      getSlice(b, loc, getInputValue(), offsets, sizes, strides);
+  tiledOperands.push_back(inputSlice->getResult(0));
+  slices.push_back(inputSlice);
+
+  // Extract slices of the init operands (partial results).
+  // For split-reduction, we need to slice the init values along
+  // the reduction dimension to get the appropriate chunk.
+  SmallVector<OpFoldResult> initOffsets, initSizes, initStrides;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim) {
+      initOffsets.push_back(splitReductionIvs[0]);
+      initSizes.push_back(b.getIndexAttr(1));
+    } else {
+      // For non-reduction dimensions, use the same offsets/sizes as input.
+      initOffsets.push_back(offsets[i]);
+      initSizes.push_back(sizes[i]);
+    }
+    initStrides.push_back(b.getIndexAttr(1));
+  }
+
+  const ShapedType &initValType = getOutputValueType();
+  const ShapedType &initIdxType = getOutputIndexType();
+  SmallVector<int64_t> resultValShape, resultIdxShape;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim)
+      continue;
+
+    if (auto sizeAttr = dyn_cast<Attribute>(initSizes[i])) {
+      int64_t size = (cast<IntegerAttr>(sizeAttr)).getInt();
+      resultValShape.push_back(size);
+      resultIdxShape.push_back(size);
+      continue;
+    }
+
+    resultValShape.push_back(ShapedType::kDynamic);
+    resultIdxShape.push_back(ShapedType::kDynamic);
+  }
+
+  RankedTensorType sliceValResultType =
+      RankedTensorType::get(resultValShape, initValType.getElementType(),
+                            cast<RankedTensorType>(initValType).getEncoding());
+  RankedTensorType sliceIdxResultType =
+      RankedTensorType::get(resultIdxShape, initIdxType.getElementType(),
+                            cast<RankedTensorType>(initIdxType).getEncoding());
+
+  auto initValSlice = tensor::ExtractSliceOp::create(
+      b, loc, sliceValResultType, init[0], initOffsets, initSizes, initStrides);
+  tiledOperands.push_back(initValSlice.getResult());
+  slices.push_back(initValSlice);
+
+  auto initIdxSlice = tensor::ExtractSliceOp::create(
+      b, loc, sliceIdxResultType, init[1], initOffsets, initSizes, initStrides);
+  tiledOperands.push_back(initIdxSlice.getResult());
+  slices.push_back(initIdxSlice);
+
+  // Create index_base for this chunk.
+  Value tileIndex =
+      getValueOrCreateConstantIndexOp(b, loc, splitReductionIvs[0]);
+  Value tileSize = getValueOrCreateConstantIndexOp(b, loc, sizes[reductionDim]);
+  Value tileStartIndex = b.create<arith::MulIOp>(loc, tileIndex, tileSize);
```

**Comment:**
also here

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1586`

```diff
@@ -1440,6 +1440,225 @@ LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
   return success();
 }
 
+FailureOr<SmallVector<Value>>
+ArgCompareOp::generateInitialTensorForPartialReduction(
+    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims) {
+  // Get the original init tensors.
+  Value valueInit = outputValue();
+  Value indexInit = outputIndex();
+
+  // Create tensors with the partial result shape.
+  Type valueElTy = getElementTypeOrSelf(valueInit.getType());
+  Type indexElTy = getElementTypeOrSelf(indexInit.getType());
+  SmallVector<OpFoldResult> partialResultShape(sizes.begin(), sizes.end());
+  Value emptyValueTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, valueElTy);
+  Value emptyIndexTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, indexElTy);
+
+  // Broadcast init values to partial result shape for slicing inside
+  // scf.forall. Each tile in the parallel loop will extract slices from
+  // these broadcasted tensors to get initialized values for the ArgCompareOp.
+  // Example: tensor<64xf32> -> tensor<64x32xf32> for 32 reduction tiles along
+  // dim 1.
+  SmallVector<int64_t> broadcastDims;
+  for (unsigned dim : reductionDims) {
+    broadcastDims.push_back(static_cast<int64_t>(dim));
+  }
+  Operation *valueBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, valueInit, emptyValueTensor, broadcastDims);
+  Operation *indexBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, indexInit, emptyIndexTensor, broadcastDims);
+
+  return SmallVector<Value>{valueBroadcastOp->getResult(0),
+                            indexBroadcastOp->getResult(0)};
+}
+
+FailureOr<TilingResult> ArgCompareOp::tileToPartialReduction(
+    OpBuilder &b, Location loc, ReductionTilingStrategy strategy,
+    ValueRange init, ArrayRef<OpFoldResult> offsets,
+    ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims,
+    ArrayRef<OpFoldResult> splitReductionIvs) {
+  assert(strategy == ReductionTilingStrategy::PartialReductionOuterParallel &&
+         "Requires PartialReductionOuterParallel strategy");
+  OpBuilder::InsertionGuard guard(b);
+
+  int64_t rank = getInputRank();
+  int64_t reductionDim = getDimension();
+
+  assert(offsets.size() == static_cast<size_t>(rank) &&
+         "Unexpected offsets size");
+  assert(sizes.size() == static_cast<size_t>(rank) && "Unexpected sizes size");
+
+  SmallVector<Operation *> slices;
+  SmallVector<Value> tiledOperands;
+
+  // Extract a slice of the input operand.
+  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
+  Operation *inputSlice =
+      getSlice(b, loc, getInputValue(), offsets, sizes, strides);
+  tiledOperands.push_back(inputSlice->getResult(0));
+  slices.push_back(inputSlice);
+
+  // Extract slices of the init operands (partial results).
+  // For split-reduction, we need to slice the init values along
+  // the reduction dimension to get the appropriate chunk.
+  SmallVector<OpFoldResult> initOffsets, initSizes, initStrides;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim) {
+      initOffsets.push_back(splitReductionIvs[0]);
+      initSizes.push_back(b.getIndexAttr(1));
+    } else {
+      // For non-reduction dimensions, use the same offsets/sizes as input.
+      initOffsets.push_back(offsets[i]);
+      initSizes.push_back(sizes[i]);
+    }
+    initStrides.push_back(b.getIndexAttr(1));
+  }
+
+  const ShapedType &initValType = getOutputValueType();
+  const ShapedType &initIdxType = getOutputIndexType();
+  SmallVector<int64_t> resultValShape, resultIdxShape;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim)
+      continue;
+
+    if (auto sizeAttr = dyn_cast<Attribute>(initSizes[i])) {
+      int64_t size = (cast<IntegerAttr>(sizeAttr)).getInt();
+      resultValShape.push_back(size);
+      resultIdxShape.push_back(size);
+      continue;
+    }
+
+    resultValShape.push_back(ShapedType::kDynamic);
+    resultIdxShape.push_back(ShapedType::kDynamic);
+  }
+
+  RankedTensorType sliceValResultType =
+      RankedTensorType::get(resultValShape, initValType.getElementType(),
+                            cast<RankedTensorType>(initValType).getEncoding());
+  RankedTensorType sliceIdxResultType =
+      RankedTensorType::get(resultIdxShape, initIdxType.getElementType(),
+                            cast<RankedTensorType>(initIdxType).getEncoding());
+
+  auto initValSlice = tensor::ExtractSliceOp::create(
+      b, loc, sliceValResultType, init[0], initOffsets, initSizes, initStrides);
+  tiledOperands.push_back(initValSlice.getResult());
+  slices.push_back(initValSlice);
+
+  auto initIdxSlice = tensor::ExtractSliceOp::create(
+      b, loc, sliceIdxResultType, init[1], initOffsets, initSizes, initStrides);
+  tiledOperands.push_back(initIdxSlice.getResult());
+  slices.push_back(initIdxSlice);
+
+  // Create index_base for this chunk.
+  Value tileIndex =
+      getValueOrCreateConstantIndexOp(b, loc, splitReductionIvs[0]);
+  Value tileSize = getValueOrCreateConstantIndexOp(b, loc, sizes[reductionDim]);
+  Value tileStartIndex = b.create<arith::MulIOp>(loc, tileIndex, tileSize);
+
+  Value newIndexBase;
+  if (Value globalIndexBase = getIndexBase()) {
+    // Add chunk start to existing index_base.
+    newIndexBase =
+        b.create<arith::AddIOp>(loc, globalIndexBase, tileStartIndex);
+  } else {
+    // Use chunk start as index_base.
+    newIndexBase = tileStartIndex;
+  }
+
+  SmallVector<Type> resultTypes;
+  if (hasPureTensorSemantics()) {
+    resultTypes.push_back(initValSlice->getResult(0).getType());
+    resultTypes.push_back(initIdxSlice->getResult(0).getType());
+  }
+
+  // Create the tiled operation with adjusted index_base.
+  SmallVector<Value> operands = std::move(tiledOperands);
+  operands.push_back(newIndexBase);
+  Operation *tiledArgmaxOp = b.create<ArgCompareOp>(
+      loc, resultTypes,
+      operands[0],                          // input slice.
+      ValueRange{operands[1], operands[2]}, // init slices.
+      operands[3],                          // index_base.
+      reductionDim                          // dimension.
```

**Comment:**
Use the `/*argName=*/ paramName` format. Clang tidy and other linters can check that the names are up to date

---

**File:** `compiler/src/iree/compiler/DispatchCreation/test/form_split_reduction_dispatches.mlir:161`

```diff
@@ -137,3 +137,116 @@ util.func public @split_reduction_multiple_dims(%arg0: tensor<?x?x?xf32>) -> ten
 //  CHECK-SAME:        ins(%[[DISPATCH]] :
 //  CHECK-SAME:        dimensions = [1, 2]
 //       CHECK:    util.return %[[REDUCE]]
+
+// -----
+
+util.func public @arg_compare_split_reduction_dynamic(%arg0: tensor<?x?xf32>) -> (tensor<?xf32>, tensor<?xi32>) {
+  %c0 = arith.constant 0 : index
+  %cst = arith.constant 0.000000e+00 : f32
+  %c0_i32 = arith.constant 0 : i32
+
+  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
+  %0 = tensor.empty(%dim) : tensor<?xf32>
+  %1 = tensor.empty(%dim) : tensor<?xi32>
+
+  %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
+  %3 = linalg.fill ins(%c0_i32 : i32) outs(%1 : tensor<?xi32>) -> tensor<?xi32>
+
+  %4:2 = iree_linalg_ext.arg_compare {
+      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>],
+      iterator_types = ["parallel", "reduction"],
+      iree_linalg_ext.split_reduction = [128]}
+      dimension(1)
+      ins(%arg0 : tensor<?x?xf32>) outs(%2, %3 : tensor<?xf32>, tensor<?xi32>) {
+    ^bb0(%in: f32, %out_val: f32):  // Only 2 arguments: input and output value
```

**Comment:**
```suggestion
    ^bb0(%in: f32, %out_val: f32):  // Only 2 arguments: input and output value.
```

See https://llvm.org/docs/CodingStandards.html#commenting

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1507`

```diff
@@ -1440,6 +1440,225 @@ LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
   return success();
 }
 
+FailureOr<SmallVector<Value>>
+ArgCompareOp::generateInitialTensorForPartialReduction(
+    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims) {
+  // Get the original init tensors.
+  Value valueInit = outputValue();
+  Value indexInit = outputIndex();
+
+  // Create tensors with the partial result shape.
+  Type valueElTy = getElementTypeOrSelf(valueInit.getType());
+  Type indexElTy = getElementTypeOrSelf(indexInit.getType());
+  SmallVector<OpFoldResult> partialResultShape(sizes.begin(), sizes.end());
+  Value emptyValueTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, valueElTy);
+  Value emptyIndexTensor =
+      b.create<tensor::EmptyOp>(loc, partialResultShape, indexElTy);
+
+  // Broadcast init values to partial result shape for slicing inside
+  // scf.forall. Each tile in the parallel loop will extract slices from
+  // these broadcasted tensors to get initialized values for the ArgCompareOp.
+  // Example: tensor<64xf32> -> tensor<64x32xf32> for 32 reduction tiles along
+  // dim 1.
+  SmallVector<int64_t> broadcastDims;
+  for (unsigned dim : reductionDims) {
+    broadcastDims.push_back(static_cast<int64_t>(dim));
+  }
+  Operation *valueBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, valueInit, emptyValueTensor, broadcastDims);
+  Operation *indexBroadcastOp = b.create<linalg::BroadcastOp>(
+      loc, indexInit, emptyIndexTensor, broadcastDims);
+
+  return SmallVector<Value>{valueBroadcastOp->getResult(0),
+                            indexBroadcastOp->getResult(0)};
+}
+
+FailureOr<TilingResult> ArgCompareOp::tileToPartialReduction(
+    OpBuilder &b, Location loc, ReductionTilingStrategy strategy,
+    ValueRange init, ArrayRef<OpFoldResult> offsets,
+    ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims,
+    ArrayRef<OpFoldResult> splitReductionIvs) {
+  assert(strategy == ReductionTilingStrategy::PartialReductionOuterParallel &&
+         "Requires PartialReductionOuterParallel strategy");
+  OpBuilder::InsertionGuard guard(b);
+
+  int64_t rank = getInputRank();
+  int64_t reductionDim = getDimension();
+
+  assert(offsets.size() == static_cast<size_t>(rank) &&
+         "Unexpected offsets size");
+  assert(sizes.size() == static_cast<size_t>(rank) && "Unexpected sizes size");
+
+  SmallVector<Operation *> slices;
+  SmallVector<Value> tiledOperands;
+
+  // Extract a slice of the input operand.
+  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
+  Operation *inputSlice =
+      getSlice(b, loc, getInputValue(), offsets, sizes, strides);
+  tiledOperands.push_back(inputSlice->getResult(0));
+  slices.push_back(inputSlice);
+
+  // Extract slices of the init operands (partial results).
+  // For split-reduction, we need to slice the init values along
+  // the reduction dimension to get the appropriate chunk.
```

**Comment:**
nit reflow this comment

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1454`

```diff
@@ -1440,6 +1440,225 @@ LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
   return success();
 }
 
+FailureOr<SmallVector<Value>>
+ArgCompareOp::generateInitialTensorForPartialReduction(
+    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims) {
+  // Get the original init tensors.
+  Value valueInit = outputValue();
+  Value indexInit = outputIndex();
+
+  // Create tensors with the partial result shape.
+  Type valueElTy = getElementTypeOrSelf(valueInit.getType());
+  Type indexElTy = getElementTypeOrSelf(indexInit.getType());
+  SmallVector<OpFoldResult> partialResultShape(sizes.begin(), sizes.end());
```

**Comment:**
```suggestion
  SmallVector<OpFoldResult> partialResultShape(sizes);
```

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1466`

```diff
@@ -1440,6 +1440,222 @@ LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
   return success();
 }
 
+FailureOr<SmallVector<Value>>
+ArgCompareOp::generateInitialTensorForPartialReduction(
+    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims) {
+  // Get the original init tensors.
+  Value valueInit = outputValue();
+  Value indexInit = outputIndex();
+
+  // Create tensors with the partial result shape.
+  Type valueElTy = getElementTypeOrSelf(valueInit.getType());
+  Type indexElTy = getElementTypeOrSelf(indexInit.getType());
+  SmallVector<OpFoldResult> partialResultShape(sizes);
+  Value emptyValueTensor =
+      tensor::EmptyOp::create(b, loc, partialResultShape, valueElTy);
+  Value emptyIndexTensor =
+      tensor::EmptyOp::create(b, loc, partialResultShape, indexElTy);
+
+  // Broadcast init values to partial result shape for slicing inside
+  // scf.forall. Each tile in the parallel loop will extract slices from
+  // these broadcasted tensors to get initialized values for the ArgCompareOp.
+  // Example: tensor<64xf32> -> tensor<64x32xf32> for 32 reduction tiles along
+  // dim 1.
+  SmallVector<int64_t> broadcastDims =
+      llvm::to_vector_of<int64_t>(reductionDims);
```

**Comment:**
```suggestion
  auto broadcastDims =
      llvm::to_vector_of<int64_t>(reductionDims);
```
See https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1519`

```diff
@@ -1440,6 +1440,222 @@ LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
   return success();
 }
 
+FailureOr<SmallVector<Value>>
+ArgCompareOp::generateInitialTensorForPartialReduction(
+    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims) {
+  // Get the original init tensors.
+  Value valueInit = outputValue();
+  Value indexInit = outputIndex();
+
+  // Create tensors with the partial result shape.
+  Type valueElTy = getElementTypeOrSelf(valueInit.getType());
+  Type indexElTy = getElementTypeOrSelf(indexInit.getType());
+  SmallVector<OpFoldResult> partialResultShape(sizes);
+  Value emptyValueTensor =
+      tensor::EmptyOp::create(b, loc, partialResultShape, valueElTy);
+  Value emptyIndexTensor =
+      tensor::EmptyOp::create(b, loc, partialResultShape, indexElTy);
+
+  // Broadcast init values to partial result shape for slicing inside
+  // scf.forall. Each tile in the parallel loop will extract slices from
+  // these broadcasted tensors to get initialized values for the ArgCompareOp.
+  // Example: tensor<64xf32> -> tensor<64x32xf32> for 32 reduction tiles along
+  // dim 1.
+  SmallVector<int64_t> broadcastDims =
+      llvm::to_vector_of<int64_t>(reductionDims);
+  Operation *valueBroadcastOp = linalg::BroadcastOp::create(
+      b, loc, valueInit, emptyValueTensor, broadcastDims);
+  Operation *indexBroadcastOp = linalg::BroadcastOp::create(
+      b, loc, indexInit, emptyIndexTensor, broadcastDims);
+
+  return SmallVector<Value>{valueBroadcastOp->getResult(0),
+                            indexBroadcastOp->getResult(0)};
+}
+
+FailureOr<TilingResult> ArgCompareOp::tileToPartialReduction(
+    OpBuilder &b, Location loc, ReductionTilingStrategy strategy,
+    ValueRange init, ArrayRef<OpFoldResult> offsets,
+    ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims,
+    ArrayRef<OpFoldResult> splitReductionIvs) {
+  assert(strategy == ReductionTilingStrategy::PartialReductionOuterParallel &&
+         "Requires PartialReductionOuterParallel strategy");
+  OpBuilder::InsertionGuard guard(b);
+
+  int64_t rank = getInputRank();
+  int64_t reductionDim = getDimension();
+
+  assert(offsets.size() == static_cast<size_t>(rank) &&
+         "Unexpected offsets size");
+  assert(sizes.size() == static_cast<size_t>(rank) && "Unexpected sizes size");
+
+  SmallVector<Operation *> slices;
+  SmallVector<Value> tiledOperands;
+
+  // Extract a slice of the input operand.
+  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
+  Operation *inputSlice =
+      getSlice(b, loc, getInputValue(), offsets, sizes, strides);
+  tiledOperands.push_back(inputSlice->getResult(0));
+  slices.push_back(inputSlice);
+
+  // Extract slices of the init operands (partial results). For split-reduction,
+  // slice along the reduction dimension to get extra parallelism.
+  SmallVector<OpFoldResult> initOffsets, initSizes, initStrides;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim) {
+      initOffsets.push_back(splitReductionIvs[0]);
+      initSizes.push_back(b.getIndexAttr(1));
+    } else {
+      // For non-reduction dimensions, use the same offsets/sizes as input.
+      initOffsets.push_back(offsets[i]);
+      initSizes.push_back(sizes[i]);
+    }
+    initStrides.push_back(b.getIndexAttr(1));
+  }
+
+  const ShapedType &initValType = getOutputValueType();
+  const ShapedType &initIdxType = getOutputIndexType();
```

**Comment:**
Do these have to be const references?

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1600`

```diff
@@ -1440,6 +1440,222 @@ LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
   return success();
 }
 
+FailureOr<SmallVector<Value>>
+ArgCompareOp::generateInitialTensorForPartialReduction(
+    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims) {
+  // Get the original init tensors.
+  Value valueInit = outputValue();
+  Value indexInit = outputIndex();
+
+  // Create tensors with the partial result shape.
+  Type valueElTy = getElementTypeOrSelf(valueInit.getType());
+  Type indexElTy = getElementTypeOrSelf(indexInit.getType());
+  SmallVector<OpFoldResult> partialResultShape(sizes);
+  Value emptyValueTensor =
+      tensor::EmptyOp::create(b, loc, partialResultShape, valueElTy);
+  Value emptyIndexTensor =
+      tensor::EmptyOp::create(b, loc, partialResultShape, indexElTy);
+
+  // Broadcast init values to partial result shape for slicing inside
+  // scf.forall. Each tile in the parallel loop will extract slices from
+  // these broadcasted tensors to get initialized values for the ArgCompareOp.
+  // Example: tensor<64xf32> -> tensor<64x32xf32> for 32 reduction tiles along
+  // dim 1.
+  SmallVector<int64_t> broadcastDims =
+      llvm::to_vector_of<int64_t>(reductionDims);
+  Operation *valueBroadcastOp = linalg::BroadcastOp::create(
+      b, loc, valueInit, emptyValueTensor, broadcastDims);
+  Operation *indexBroadcastOp = linalg::BroadcastOp::create(
+      b, loc, indexInit, emptyIndexTensor, broadcastDims);
+
+  return SmallVector<Value>{valueBroadcastOp->getResult(0),
+                            indexBroadcastOp->getResult(0)};
+}
+
+FailureOr<TilingResult> ArgCompareOp::tileToPartialReduction(
+    OpBuilder &b, Location loc, ReductionTilingStrategy strategy,
+    ValueRange init, ArrayRef<OpFoldResult> offsets,
+    ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims,
+    ArrayRef<OpFoldResult> splitReductionIvs) {
+  assert(strategy == ReductionTilingStrategy::PartialReductionOuterParallel &&
+         "Requires PartialReductionOuterParallel strategy");
+  OpBuilder::InsertionGuard guard(b);
+
+  int64_t rank = getInputRank();
+  int64_t reductionDim = getDimension();
+
+  assert(offsets.size() == static_cast<size_t>(rank) &&
+         "Unexpected offsets size");
+  assert(sizes.size() == static_cast<size_t>(rank) && "Unexpected sizes size");
+
+  SmallVector<Operation *> slices;
+  SmallVector<Value> tiledOperands;
+
+  // Extract a slice of the input operand.
+  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
+  Operation *inputSlice =
+      getSlice(b, loc, getInputValue(), offsets, sizes, strides);
+  tiledOperands.push_back(inputSlice->getResult(0));
+  slices.push_back(inputSlice);
+
+  // Extract slices of the init operands (partial results). For split-reduction,
+  // slice along the reduction dimension to get extra parallelism.
+  SmallVector<OpFoldResult> initOffsets, initSizes, initStrides;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim) {
+      initOffsets.push_back(splitReductionIvs[0]);
+      initSizes.push_back(b.getIndexAttr(1));
+    } else {
+      // For non-reduction dimensions, use the same offsets/sizes as input.
+      initOffsets.push_back(offsets[i]);
+      initSizes.push_back(sizes[i]);
+    }
+    initStrides.push_back(b.getIndexAttr(1));
+  }
+
+  const ShapedType &initValType = getOutputValueType();
+  const ShapedType &initIdxType = getOutputIndexType();
+  SmallVector<int64_t> resultValShape, resultIdxShape;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim) {
+      continue;
+    }
+
+    if (auto sizeAttr = dyn_cast<Attribute>(initSizes[i])) {
+      int64_t size = (cast<IntegerAttr>(sizeAttr)).getInt();
+      resultValShape.push_back(size);
+      resultIdxShape.push_back(size);
+      continue;
+    }
+
+    resultValShape.push_back(ShapedType::kDynamic);
+    resultIdxShape.push_back(ShapedType::kDynamic);
+  }
+
+  auto sliceValResultType =
+      RankedTensorType::get(resultValShape, initValType.getElementType(),
+                            cast<RankedTensorType>(initValType).getEncoding());
+  auto sliceIdxResultType =
+      RankedTensorType::get(resultIdxShape, initIdxType.getElementType(),
+                            cast<RankedTensorType>(initIdxType).getEncoding());
+
+  auto initValSlice = tensor::ExtractSliceOp::create(
+      b, loc, sliceValResultType, init[0], initOffsets, initSizes, initStrides);
+  tiledOperands.push_back(initValSlice.getResult());
+  slices.push_back(initValSlice);
+
+  auto initIdxSlice = tensor::ExtractSliceOp::create(
+      b, loc, sliceIdxResultType, init[1], initOffsets, initSizes, initStrides);
+  tiledOperands.push_back(initIdxSlice.getResult());
+  slices.push_back(initIdxSlice);
+
+  // Create index_base for this chunk.
+  Value tileIndex =
+      getValueOrCreateConstantIndexOp(b, loc, splitReductionIvs[0]);
+  Value tileSize = getValueOrCreateConstantIndexOp(b, loc, sizes[reductionDim]);
+  Value tileStartIndex = arith::MulIOp::create(b, loc, tileIndex, tileSize);
+
+  Value newIndexBase;
+  if (Value globalIndexBase = getIndexBase()) {
+    // Add chunk start to existing index_base.
+    newIndexBase =
+        arith::AddIOp::create(b, loc, globalIndexBase, tileStartIndex);
+  } else {
+    // Use chunk start as index_base.
+    newIndexBase = tileStartIndex;
+  }
+
+  SmallVector<Type, 2> resultTypes;
+  if (hasPureTensorSemantics()) {
+    resultTypes = {initValSlice->getResult(0).getType(),
+                   initIdxSlice->getResult(0).getType()};
+  }
+
+  // Create the tiled operation with adjusted index_base.
+  SmallVector<Value> operands = std::move(tiledOperands);
+  operands.push_back(newIndexBase);
+  Operation *tiledArgmaxOp =
+      b.create<ArgCompareOp>(loc, /*results=*/resultTypes,
+                             /*inputs=*/operands[0],
+                             /*outputs=*/ValueRange{operands[1], operands[2]},
+                             /*indexBase=*/operands[3],
+                             /*dimension=*/reductionDim);
+
+  // Copy the region.
+  Region &targetRegion = tiledArgmaxOp->getRegion(0);
+  Region &sourceRegion = getRegion();
+  targetRegion.takeBody(sourceRegion);
+
+  return TilingResult{
+      {tiledArgmaxOp}, SmallVector<Value>(tiledArgmaxOp->getResults()), slices};
+}
+
+FailureOr<MergeResult>
+ArgCompareOp::mergeReductions(OpBuilder &b, Location loc,
+                              ValueRange partialReduce,
+                              const llvm::SetVector<unsigned> &reductionDims) {
+  SmallVector<int64_t> mergeReductionDims =
+      llvm::to_vector_of<int64_t>(reductionDims);
```

**Comment:**
```suggestion
  auto mergeReductionDims =
      llvm::to_vector_of<int64_t>(reductionDims);
```
See https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1580`

```diff
@@ -1440,6 +1440,222 @@ LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
   return success();
 }
 
+FailureOr<SmallVector<Value>>
+ArgCompareOp::generateInitialTensorForPartialReduction(
+    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims) {
+  // Get the original init tensors.
+  Value valueInit = outputValue();
+  Value indexInit = outputIndex();
+
+  // Create tensors with the partial result shape.
+  Type valueElTy = getElementTypeOrSelf(valueInit.getType());
+  Type indexElTy = getElementTypeOrSelf(indexInit.getType());
+  SmallVector<OpFoldResult> partialResultShape(sizes);
+  Value emptyValueTensor =
+      tensor::EmptyOp::create(b, loc, partialResultShape, valueElTy);
+  Value emptyIndexTensor =
+      tensor::EmptyOp::create(b, loc, partialResultShape, indexElTy);
+
+  // Broadcast init values to partial result shape for slicing inside
+  // scf.forall. Each tile in the parallel loop will extract slices from
+  // these broadcasted tensors to get initialized values for the ArgCompareOp.
+  // Example: tensor<64xf32> -> tensor<64x32xf32> for 32 reduction tiles along
+  // dim 1.
+  SmallVector<int64_t> broadcastDims =
+      llvm::to_vector_of<int64_t>(reductionDims);
+  Operation *valueBroadcastOp = linalg::BroadcastOp::create(
+      b, loc, valueInit, emptyValueTensor, broadcastDims);
+  Operation *indexBroadcastOp = linalg::BroadcastOp::create(
+      b, loc, indexInit, emptyIndexTensor, broadcastDims);
+
+  return SmallVector<Value>{valueBroadcastOp->getResult(0),
+                            indexBroadcastOp->getResult(0)};
+}
+
+FailureOr<TilingResult> ArgCompareOp::tileToPartialReduction(
+    OpBuilder &b, Location loc, ReductionTilingStrategy strategy,
+    ValueRange init, ArrayRef<OpFoldResult> offsets,
+    ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims,
+    ArrayRef<OpFoldResult> splitReductionIvs) {
+  assert(strategy == ReductionTilingStrategy::PartialReductionOuterParallel &&
+         "Requires PartialReductionOuterParallel strategy");
+  OpBuilder::InsertionGuard guard(b);
+
+  int64_t rank = getInputRank();
+  int64_t reductionDim = getDimension();
+
+  assert(offsets.size() == static_cast<size_t>(rank) &&
+         "Unexpected offsets size");
+  assert(sizes.size() == static_cast<size_t>(rank) && "Unexpected sizes size");
+
+  SmallVector<Operation *> slices;
+  SmallVector<Value> tiledOperands;
+
+  // Extract a slice of the input operand.
+  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
+  Operation *inputSlice =
+      getSlice(b, loc, getInputValue(), offsets, sizes, strides);
+  tiledOperands.push_back(inputSlice->getResult(0));
+  slices.push_back(inputSlice);
+
+  // Extract slices of the init operands (partial results). For split-reduction,
+  // slice along the reduction dimension to get extra parallelism.
+  SmallVector<OpFoldResult> initOffsets, initSizes, initStrides;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim) {
+      initOffsets.push_back(splitReductionIvs[0]);
+      initSizes.push_back(b.getIndexAttr(1));
+    } else {
+      // For non-reduction dimensions, use the same offsets/sizes as input.
+      initOffsets.push_back(offsets[i]);
+      initSizes.push_back(sizes[i]);
+    }
+    initStrides.push_back(b.getIndexAttr(1));
+  }
+
+  const ShapedType &initValType = getOutputValueType();
+  const ShapedType &initIdxType = getOutputIndexType();
+  SmallVector<int64_t> resultValShape, resultIdxShape;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim) {
+      continue;
+    }
+
+    if (auto sizeAttr = dyn_cast<Attribute>(initSizes[i])) {
+      int64_t size = (cast<IntegerAttr>(sizeAttr)).getInt();
+      resultValShape.push_back(size);
+      resultIdxShape.push_back(size);
+      continue;
+    }
+
+    resultValShape.push_back(ShapedType::kDynamic);
+    resultIdxShape.push_back(ShapedType::kDynamic);
+  }
+
+  auto sliceValResultType =
+      RankedTensorType::get(resultValShape, initValType.getElementType(),
+                            cast<RankedTensorType>(initValType).getEncoding());
+  auto sliceIdxResultType =
+      RankedTensorType::get(resultIdxShape, initIdxType.getElementType(),
+                            cast<RankedTensorType>(initIdxType).getEncoding());
+
+  auto initValSlice = tensor::ExtractSliceOp::create(
+      b, loc, sliceValResultType, init[0], initOffsets, initSizes, initStrides);
+  tiledOperands.push_back(initValSlice.getResult());
+  slices.push_back(initValSlice);
+
+  auto initIdxSlice = tensor::ExtractSliceOp::create(
+      b, loc, sliceIdxResultType, init[1], initOffsets, initSizes, initStrides);
+  tiledOperands.push_back(initIdxSlice.getResult());
+  slices.push_back(initIdxSlice);
+
+  // Create index_base for this chunk.
+  Value tileIndex =
+      getValueOrCreateConstantIndexOp(b, loc, splitReductionIvs[0]);
+  Value tileSize = getValueOrCreateConstantIndexOp(b, loc, sizes[reductionDim]);
+  Value tileStartIndex = arith::MulIOp::create(b, loc, tileIndex, tileSize);
+
+  Value newIndexBase;
+  if (Value globalIndexBase = getIndexBase()) {
+    // Add chunk start to existing index_base.
+    newIndexBase =
+        arith::AddIOp::create(b, loc, globalIndexBase, tileStartIndex);
+  } else {
+    // Use chunk start as index_base.
+    newIndexBase = tileStartIndex;
+  }
+
+  SmallVector<Type, 2> resultTypes;
+  if (hasPureTensorSemantics()) {
+    resultTypes = {initValSlice->getResult(0).getType(),
+                   initIdxSlice->getResult(0).getType()};
+  }
+
+  // Create the tiled operation with adjusted index_base.
+  SmallVector<Value> operands = std::move(tiledOperands);
+  operands.push_back(newIndexBase);
+  Operation *tiledArgmaxOp =
+      b.create<ArgCompareOp>(loc, /*results=*/resultTypes,
```

**Comment:**
This is deprecated

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1540`

```diff
@@ -1440,6 +1441,221 @@ LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
   return success();
 }
 
+FailureOr<SmallVector<Value>>
+ArgCompareOp::generateInitialTensorForPartialReduction(
+    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims) {
+  // Get the original init tensors.
+  Value valueInit = outputValue();
+  Value indexInit = outputIndex();
+
+  // Create tensors with the partial result shape.
+  Type valueElTy = getElementTypeOrSelf(valueInit.getType());
+  Type indexElTy = getElementTypeOrSelf(indexInit.getType());
+  SmallVector<OpFoldResult> partialResultShape(sizes);
+  Value emptyValueTensor =
+      tensor::EmptyOp::create(b, loc, partialResultShape, valueElTy);
+  Value emptyIndexTensor =
+      tensor::EmptyOp::create(b, loc, partialResultShape, indexElTy);
+
+  // Broadcast init values to partial result shape for slicing inside
+  // scf.forall. Each tile in the parallel loop will extract slices from
+  // these broadcasted tensors to get initialized values for the ArgCompareOp.
+  // Example: tensor<64xf32> -> tensor<64x32xf32> for 32 reduction tiles along
+  // dim 1.
+  auto broadcastDims = llvm::to_vector_of<int64_t>(reductionDims);
+  Operation *valueBroadcastOp = linalg::BroadcastOp::create(
+      b, loc, valueInit, emptyValueTensor, broadcastDims);
+  Operation *indexBroadcastOp = linalg::BroadcastOp::create(
+      b, loc, indexInit, emptyIndexTensor, broadcastDims);
+
+  return SmallVector<Value>{valueBroadcastOp->getResult(0),
+                            indexBroadcastOp->getResult(0)};
+}
+
+FailureOr<TilingResult> ArgCompareOp::tileToPartialReduction(
+    OpBuilder &b, Location loc, ReductionTilingStrategy strategy,
+    ValueRange init, ArrayRef<OpFoldResult> offsets,
+    ArrayRef<OpFoldResult> sizes,
+    const llvm::SetVector<unsigned> &reductionDims,
+    ArrayRef<OpFoldResult> splitReductionIvs) {
+  if (strategy != ReductionTilingStrategy::PartialReductionOuterParallel) {
+    return failure();
+  }
+  OpBuilder::InsertionGuard guard(b);
+
+  int64_t rank = getInputRank();
+  int64_t reductionDim = getDimension();
+
+  assert(offsets.size() == static_cast<size_t>(rank) &&
+         "Unexpected offsets size");
+  assert(sizes.size() == static_cast<size_t>(rank) && "Unexpected sizes size");
+
+  SmallVector<Operation *> slices;
+  SmallVector<Value> tiledOperands;
+
+  // Extract a slice of the input operand.
+  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
+  Operation *inputSlice =
+      getSlice(b, loc, getInputValue(), offsets, sizes, strides);
+  tiledOperands.push_back(inputSlice->getResult(0));
+  slices.push_back(inputSlice);
+
+  // Extract slices of the init operands (partial results). For split-reduction,
+  // slice along the reduction dimension to get extra parallelism.
+  SmallVector<OpFoldResult> initOffsets, initSizes, initStrides;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim) {
+      initOffsets.push_back(splitReductionIvs[0]);
+      initSizes.push_back(b.getIndexAttr(1));
+    } else {
+      // For non-reduction dimensions, use the same offsets/sizes as input.
+      initOffsets.push_back(offsets[i]);
+      initSizes.push_back(sizes[i]);
+    }
+    initStrides.push_back(b.getIndexAttr(1));
+  }
+
+  ShapedType initValType = getOutputValueType();
+  ShapedType initIdxType = getOutputIndexType();
+  SmallVector<int64_t> resultValShape, resultIdxShape;
+  for (int64_t i = 0; i < rank; ++i) {
+    if (i == reductionDim) {
+      continue;
+    }
+
+    if (auto sizeAttr = dyn_cast<Attribute>(initSizes[i])) {
+      int64_t size = (cast<IntegerAttr>(sizeAttr)).getInt();
+      resultValShape.push_back(size);
+      resultIdxShape.push_back(size);
+      continue;
+    }
+
+    resultValShape.push_back(ShapedType::kDynamic);
+    resultIdxShape.push_back(ShapedType::kDynamic);
+  }
+
+  auto sliceValResultType =
+      RankedTensorType::get(resultValShape, initValType.getElementType(),
+                            cast<RankedTensorType>(initValType).getEncoding());
```

**Comment:**
Since this is the only use of these two types, you can query them here where they are needed

---


---


## [PR #21816](https://github.com/iree-org/iree/pull/21816): [Codegen][Tuner] retire the C/Python binding for querying mma intrinsic. NFC. 

### Review Summary

**APPROVED** (2025-09-04)



---


## [PR #21812](https://github.com/iree-org/iree/pull/21812): [Codegen][Tuner]: improve python binding to query target info

### Review Summary

**CHANGES_REQUESTED** (2025-09-02)

I think this is a good idea in general but I'm confused by the implementation and so much work is being done on the python bindings code instead of the C API.

Also, make sure you follow the LLVM coding standards throughout the PR: https://llvm.org/docs/CodingStandards.html


**COMMENTED** (2025-09-02)


**COMMENTED** (2025-09-02)


**COMMENTED** (2025-09-03)


**CHANGES_REQUESTED** (2025-09-03)


**COMMENTED** (2025-09-03)


**COMMENTED** (2025-09-03)


**CHANGES_REQUESTED** (2025-09-03)


**COMMENTED** (2025-09-03)


**COMMENTED** (2025-09-04)


**COMMENTED** (2025-09-04)


**COMMENTED** (2025-09-04)


**COMMENTED** (2025-09-04)


**COMMENTED** (2025-09-04)


**COMMENTED** (2025-09-04)


**COMMENTED** (2025-09-04)


**COMMENTED** (2025-09-04)

The static assert is still missing


**COMMENTED** (2025-09-04)


### Code Comments

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:86`

```diff
@@ -76,6 +77,56 @@ static std::vector<int64_t> getIntArrayAttrValues(MlirAttribute attr) {
   return result;
 }
 
+static ireeGPUTargetInfo
+createGPUTargetInfo(MlirContext context, const std::string &arch,
+                    const std::vector<int64_t> &subgroupChoices,
+                    const std::vector<int64_t> &workgroupSizes,
+                    int64_t threadCount, int64_t memoryBytes,
+                    const py::list &mmaIntrinsicObjects) {
+  ireeGPUTargetInfo gpuTargetInfo;
```

**Comment:**
Let's initialize this since it's a C struct and we don't want to accidentally end up with uninitialized fields.

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:81`

```diff
@@ -76,6 +77,56 @@ static std::vector<int64_t> getIntArrayAttrValues(MlirAttribute attr) {
   return result;
 }
 
+static ireeGPUTargetInfo
+createGPUTargetInfo(MlirContext context, const std::string &arch,
```

**Comment:**
Can you explain why we need this function? I'd expect this to be queried by C bindings which return a complete `ireeGPUTargetInfo` struct

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:600`

```diff
@@ -529,7 +587,37 @@ NB_MODULE(_ireeCompilerDialects, m) {
       .def_prop_ro("max_workgroup_memory_bytes",
                    [](const ireeGPUTargetInfo &self) -> int64_t {
                      return self.maxWorkgroupMemoryBytes;
-                   });
+                   })
+      .def_prop_ro(
+          "mma_intrinsics", [](const ireeGPUTargetInfo &self) -> py::list {
+            std::vector<int64_t> rawValues =
+                getIntArrayAttrValues(self.mmaIntrinsics);
+
+            py::list result;
+            py::module_ gpuModule = py::module_::import_(kGpuModuleImportPath);
+
+            for (uint32_t rawValue : rawValues) {
+              try {
```

**Comment:**
Could we do this without throwing any exceptions?

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:81`

```diff
@@ -76,6 +77,56 @@ static std::vector<int64_t> getIntArrayAttrValues(MlirAttribute attr) {
   return result;
 }
 
+static ireeGPUTargetInfo
+createGPUTargetInfo(MlirContext context, const std::string &arch,
```

**Comment:**
This is understand, I just don't think we should have so much work to do on the python binding side. We should only do minimal work to take data from C structs and put them in a format friendly to python.

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:81`

```diff
@@ -76,6 +77,56 @@ static std::vector<int64_t> getIntArrayAttrValues(MlirAttribute attr) {
   return result;
 }
 
+static ireeGPUTargetInfo
+createGPUTargetInfo(MlirContext context, const std::string &arch,
```

**Comment:**
you can have two constructors (or one constructor and one static method): one to allow construction in python, and one to query target details and return you `TargetInfo`: https://nanobind.readthedocs.io/en/latest/classes.html . Not every python class has to be backed by a struct in the C api

> But I can look into how to make it work that way.

+1

---

**File:** `compiler/bindings/python/test/ir/dialects_test.py:469`

```diff
@@ -456,3 +456,53 @@ def gpu_target_info_attribute_parsing():
         512,
         1024,
     ], f"Expected max_workgroup_sizes [256, 512, 1024], got {max_workgroup_sizes}"
+
+    mma_intrinsics = gpu_target_info.mma_intrinsics
+    assert mma_intrinsics == [
+        iree_gpu.MMAIntrinsic.MFMA_F32_16x16x4_F32,
+        iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
+        iree_gpu.VirtualMMAIntrinsic.VMFMA_F32_16x16x32_F16,
+    ], f"Expected mma_intrinsics [MFMA_F32_16x16x4_F32, MFMA_F32_16x16x16_F16, VMFMA_F32_16x16x32_F16], got {mma_intrinsics}"
+
+
+@run
+def gpu_target_info_constructor():
```

**Comment:**
We also need a test for the cases when the constructor is given values of wrong type

---

**File:** `compiler/bindings/python/test/ir/dialects_test.py:7`

```diff
@@ -4,6 +4,7 @@
 # See https://llvm.org/LICENSE.txt for license information.
 # SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 
+import pytest
```

**Comment:**
I don't think we use pytest anywhere else in IREE -- would be worth checking with other folks if we want to add it and then clean up other pieces of the infra.

If you grep python files, you will notice that some use `untitest` instead -- can we switch to that?

---

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h:164`

```diff
@@ -150,12 +150,19 @@ struct ireeGPUTargetInfo {
   MlirAttribute maxWorkgroupSizes;    // Max threads per X/Y/Z dimension.
   int64_t maxThreadCountPerWorkgroup; // Max threads per workgroup.
   int64_t maxWorkgroupMemoryBytes;    // Max workgroup memory.
+  MlirAttribute mmaIntrinsics;        // MMA Intrinsics.
 };
 
 // Queries GPU target info from the given `ExecutableTargetAttr` attribute.
 MLIR_CAPI_EXPORTED ireeGPUTargetInfo
 ireeHALExecutableTargetAttrGetGPUTargetInfo(MlirAttribute attr);
 
+MLIR_CAPI_EXPORTED ireeGPUTargetInfo ireeGPUTargetInfoGet(
+    MlirContext mlirCtx, const char *arch, const int32_t *subgroupChoices,
+    size_t numSubgroupChoices, const int32_t *workgroupSizes,
+    size_t numWorkgroupSizes, int64_t threadCount, int64_t memoryBytes,
+    const int32_t *mmaIntrinsics, size_t numMmaIntrinsics);
```

**Comment:**
How do you know `mmaIntrinsics` map to `int32_t`?

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:522`

```diff
@@ -509,6 +510,37 @@ NB_MODULE(_ireeCompilerDialects, m) {
   //===-------------------------------------------------------------------===//
 
   py::class_<ireeGPUTargetInfo>(iree_gpu_module, "TargetInfo")
+      .def(
+          "__init__",
+          [](ireeGPUTargetInfo *self, MlirContext context,
+             const std::string &arch,
+             const std::vector<int32_t> &subgroupChoices,
+             const std::vector<int32_t> &workgroupSizes, int64_t threadCount,
+             int64_t memoryBytes, const py::list &mmaIntrinsicObjs) {
+            std::vector<int32_t> mmaIntrinsicVals;
+            for (auto item : mmaIntrinsicObjs) {
+              int32_t enumValue = py::cast<int32_t>(item.attr("value"));
```

**Comment:**
1. How do we know this is the underlying enum type?
2. How do we know that this is an mma intrinsic type and not some other enum?

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:541`

```diff
@@ -509,6 +510,37 @@ NB_MODULE(_ireeCompilerDialects, m) {
   //===-------------------------------------------------------------------===//
 
   py::class_<ireeGPUTargetInfo>(iree_gpu_module, "TargetInfo")
+      .def(
+          "__init__",
+          [](ireeGPUTargetInfo *self, MlirContext context,
+             const std::string &arch,
+             const std::vector<int32_t> &subgroupChoices,
+             const std::vector<int32_t> &workgroupSizes, int64_t threadCount,
+             int64_t memoryBytes, const py::list &mmaIntrinsicObjs) {
+            std::vector<int32_t> mmaIntrinsicVals;
+            for (auto item : mmaIntrinsicObjs) {
+              int32_t enumValue = py::cast<int32_t>(item.attr("value"));
+              mmaIntrinsicVals.push_back(enumValue);
+            }
+
+            *self = ireeGPUTargetInfoGet(
+                context, arch.c_str(), subgroupChoices.data(),
+                subgroupChoices.size(), workgroupSizes.data(),
+                workgroupSizes.size(), threadCount, memoryBytes,
+                mmaIntrinsicVals.data(), mmaIntrinsicVals.size());
+          },
+          "context"_a, "arch"_a, "subgroup_size_choices"_a,
+          "max_workgroup_sizes"_a, "max_thread_count_per_workgroup"_a,
+          "max_workgroup_memory_bytes"_a, "mma_intrinsics"_a = py::list{},
+          "Create a GPUTargetInfo with the given parameters")
+      .def_static(
+          "get_gpu_target_info",
+          [](MlirAttribute executable_target_attr) -> ireeGPUTargetInfo {
+            return ireeHALExecutableTargetAttrGetGPUTargetInfo(
+                executable_target_attr);
+          },
```

**Comment:**
Do we need the lambda?

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:566`

```diff
@@ -529,12 +561,48 @@ NB_MODULE(_ireeCompilerDialects, m) {
       .def_prop_ro("max_workgroup_memory_bytes",
                    [](const ireeGPUTargetInfo &self) -> int64_t {
                      return self.maxWorkgroupMemoryBytes;
-                   });
+                   })
+      .def_prop_ro(
+          "mma_intrinsics", [](const ireeGPUTargetInfo &self) -> py::list {
```

**Comment:**
Can we move this to the C API instead?

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:478`

```diff
@@ -420,5 +420,95 @@ ireeHALExecutableTargetAttrGetGPUTargetInfo(MlirAttribute attr) {
       wgpAttr.getMaxThreadCountPerWorkgroup();
   targetInfo.maxWorkgroupMemoryBytes = wgpAttr.getMaxWorkgroupMemoryBytes();
 
+  targetInfo.mmaIntrinsics = wrap(builder.getArrayAttr({}));
+  mlir::iree_compiler::IREE::GPU::MMAOpsArrayAttr mmaOpsArray =
+      wgpAttr.getMma();
+  if (mmaOpsArray) {
+    std::vector<mlir::Attribute> mmaIntrinsicAttrs;
+    for (mlir::iree_compiler::IREE::GPU::MMAAttr mmaAttr : mmaOpsArray) {
+      mlir::iree_compiler::IREE::GPU::MMAIntrinsic intrinsic =
+          mmaAttr.getIntrinsic();
+      auto mmaIntrinsicAttr =
+          mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr::get(context,
+                                                                intrinsic);
+      mmaIntrinsicAttrs.push_back(mmaIntrinsicAttr);
+
+      for (mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsic
+               virtualIntrinsic : mmaAttr.getVirtualIntrinsics()) {
+        auto virtualMmaIntrinsicAttr =
+            mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsicAttr::get(
+                context, virtualIntrinsic);
+        mmaIntrinsicAttrs.push_back(virtualMmaIntrinsicAttr);
+      }
+    }
+    targetInfo.mmaIntrinsics = wrap(builder.getArrayAttr(mmaIntrinsicAttrs));
+  }
+  return targetInfo;
+}
+
+ireeGPUTargetInfo
+ireeGPUTargetInfoGet(MlirContext mlirCtx, const char *arch,
+                     const int32_t *subgroupChoices, size_t numSubgroupChoices,
+                     const int32_t *workgroupSizes, size_t numWorkgroupSizes,
+                     int64_t threadCount, int64_t memoryBytes,
+                     const int32_t *mmaIntrinsics, size_t numMmaIntrinsics) {
+  assert(!mlirContextIsNull(mlirCtx) && "mlirCtx cannot be null");
+  assert(arch && "arch cannot be null");
+
+  mlir::MLIRContext *context = unwrap(mlirCtx);
+  mlir::Builder builder(context);
+
+  ireeGPUTargetInfo targetInfo = {};
+
+  targetInfo.arch = wrap(mlir::StringAttr::get(context, arch));
+  std::vector<int32_t> subgroupChoicesVec(subgroupChoices,
+                                          subgroupChoices + numSubgroupChoices);
+  targetInfo.subgroupSizeChoices =
+      wrap(builder.getI32ArrayAttr(subgroupChoicesVec));
+  std::vector<int32_t> workgroupSizesVec(workgroupSizes,
+                                         workgroupSizes + numWorkgroupSizes);
+  targetInfo.maxWorkgroupSizes =
+      wrap(builder.getI32ArrayAttr(workgroupSizesVec));
+
+  targetInfo.maxThreadCountPerWorkgroup = threadCount;
+  targetInfo.maxWorkgroupMemoryBytes = memoryBytes;
+
+  if (numMmaIntrinsics == 0) {
+    targetInfo.mmaIntrinsics = wrap(builder.getArrayAttr({}));
+  } else {
```

**Comment:**
Do we need to handle the empty case in a separate code path?

---

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h:164`

```diff
@@ -150,12 +150,19 @@ struct ireeGPUTargetInfo {
   MlirAttribute maxWorkgroupSizes;    // Max threads per X/Y/Z dimension.
   int64_t maxThreadCountPerWorkgroup; // Max threads per workgroup.
   int64_t maxWorkgroupMemoryBytes;    // Max workgroup memory.
+  MlirAttribute mmaIntrinsics;        // MMA Intrinsics.
 };
 
 // Queries GPU target info from the given `ExecutableTargetAttr` attribute.
 MLIR_CAPI_EXPORTED ireeGPUTargetInfo
 ireeHALExecutableTargetAttrGetGPUTargetInfo(MlirAttribute attr);
 
+MLIR_CAPI_EXPORTED ireeGPUTargetInfo ireeGPUTargetInfoGet(
+    MlirContext mlirCtx, const char *arch, const int32_t *subgroupChoices,
+    size_t numSubgroupChoices, const int32_t *workgroupSizes,
+    size_t numWorkgroupSizes, int64_t threadCount, int64_t memoryBytes,
+    const int32_t *mmaIntrinsics, size_t numMmaIntrinsics);
```

**Comment:**
How can I tell it's i32 instead of u32? Also, what happens if this changes in the future? I think we need to have a typedef on the C api side, static_assert in the C implementation, and then use this typedef on the python side.

---

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h:20`

```diff
@@ -14,6 +14,11 @@
 extern "C" {
 #endif
 
+// This typedef ensures consistency between the C API, C++ implementation, and
+// Python bindings. Update both this typedef and the static assertions if the
+// enum underlying types change.
+typedef uint32_t mma_intrinsic_t;
```

**Comment:**
```suggestion
typedef uint32_t mma_intrinsic_enum_t;
```

---

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h:175`

```diff
@@ -160,8 +167,20 @@ ireeHALExecutableTargetAttrGetGPUTargetInfo(MlirAttribute attr);
 MLIR_CAPI_EXPORTED ireeGPUTargetInfo ireeGPUTargetInfoGet(
     MlirContext mlirCtx, const char *arch, const int32_t *subgroupChoices,
     size_t numSubgroupChoices, const int32_t *workgroupSizes,
-    size_t numWorkgroupSizes, int64_t threadCount, int64_t memoryBytes,
-    const int32_t *mmaIntrinsics, size_t numMmaIntrinsics);
+    size_t numWorkgroupSizes, int32_t threadCount, int32_t memoryBytes,
+    const mma_intrinsic_t *mmaIntrinsics, size_t numMmaIntrinsics);
+
+struct ireeGPUMMAIntrinsicResult {
+  mma_intrinsic_t *mmaIntrinsicVals;
+  bool *isVirtual; // true if VirtualMMAIntrinsic, false if MMAIntrinsic.
```

**Comment:**
Could you make this a plural so that it's clear there is more than one value behind this pointer?

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:528`

```diff
@@ -515,11 +515,22 @@ NB_MODULE(_ireeCompilerDialects, m) {
           [](ireeGPUTargetInfo *self, MlirContext context,
              const std::string &arch,
              const std::vector<int32_t> &subgroupChoices,
-             const std::vector<int32_t> &workgroupSizes, int64_t threadCount,
-             int64_t memoryBytes, const py::list &mmaIntrinsicObjs) {
-            std::vector<int32_t> mmaIntrinsicVals;
+             const std::vector<int32_t> &workgroupSizes, int32_t threadCount,
+             int32_t memoryBytes, const py::list &mmaIntrinsicObjs) {
+            std::vector<mma_intrinsic_t> mmaIntrinsicVals;
+            py::module_ gpuModule = py::module_::import_(kGpuModuleImportPath);
+            py::object mmaIntrinsicClass = gpuModule.attr("MMAIntrinsic");
+            py::object virtualMmaIntrinsicClass =
+                gpuModule.attr("VirtualMMAIntrinsic");
+
             for (auto item : mmaIntrinsicObjs) {
-              int32_t enumValue = py::cast<int32_t>(item.attr("value"));
+              if (!py::isinstance(item, mmaIntrinsicClass) &&
+                  !py::isinstance(item, virtualMmaIntrinsicClass)) {
```

**Comment:**
Could we make these share the same base class?

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:528`

```diff
@@ -473,42 +474,85 @@ ireeGPUTargetInfoGet(MlirContext mlirCtx, const char *arch,
   targetInfo.maxThreadCountPerWorkgroup = threadCount;
   targetInfo.maxWorkgroupMemoryBytes = memoryBytes;
 
-  if (numMmaIntrinsics == 0) {
-    targetInfo.mmaIntrinsics = wrap(builder.getArrayAttr({}));
-  } else {
-    std::vector<mlir::Attribute> mmaIntrinsicAttrs;
-    mmaIntrinsicAttrs.reserve(numMmaIntrinsics);
-    for (size_t i = 0; i < numMmaIntrinsics; i++) {
-      int32_t enumValue = mmaIntrinsics[i];
-
-      std::optional<mlir::iree_compiler::IREE::GPU::MMAIntrinsic> mmaIntrinsic =
-          mlir::iree_compiler::IREE::GPU::symbolizeMMAIntrinsic(enumValue);
-      if (mmaIntrinsic) {
-        auto mmaIntrinsicAttr =
-            mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr::get(
-                context, *mmaIntrinsic);
-        mmaIntrinsicAttrs.push_back(mmaIntrinsicAttr);
-        continue;
-      }
+  std::vector<mlir::Attribute> mmaIntrinsicAttrs;
+  mmaIntrinsicAttrs.reserve(numMmaIntrinsics);
+  for (size_t i = 0; i < numMmaIntrinsics; i++) {
+    mma_intrinsic_t enumValue = mmaIntrinsics[i];
 
-      std::optional<mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsic>
-          virtualMmaIntrinsic =
-              mlir::iree_compiler::IREE::GPU::symbolizeVirtualMMAIntrinsic(
-                  enumValue);
-      if (virtualMmaIntrinsic) {
-        auto virtualMmaIntrinsicAttr =
-            mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsicAttr::get(
-                context, *virtualMmaIntrinsic);
-        mmaIntrinsicAttrs.push_back(virtualMmaIntrinsicAttr);
-        continue;
-      }
+    std::optional<mlir::iree_compiler::IREE::GPU::MMAIntrinsic> mmaIntrinsic =
+        mlir::iree_compiler::IREE::GPU::symbolizeMMAIntrinsic(enumValue);
+    if (mmaIntrinsic) {
+      auto mmaIntrinsicAttr =
+          mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr::get(context,
+                                                                *mmaIntrinsic);
+      mmaIntrinsicAttrs.push_back(mmaIntrinsicAttr);
+      continue;
+    }
 
-      assert(false &&
-             ("Invalid MMA intrinsic value: " + std::to_string(enumValue))
-                 .c_str());
+    std::optional<mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsic>
+        virtualMmaIntrinsic =
+            mlir::iree_compiler::IREE::GPU::symbolizeVirtualMMAIntrinsic(
+                enumValue);
+    if (virtualMmaIntrinsic) {
+      auto virtualMmaIntrinsicAttr =
+          mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsicAttr::get(
+              context, *virtualMmaIntrinsic);
+      mmaIntrinsicAttrs.push_back(virtualMmaIntrinsicAttr);
+      continue;
     }
-    targetInfo.mmaIntrinsics = wrap(builder.getArrayAttr(mmaIntrinsicAttrs));
+
+    assert(
+        false &&
+        ("Invalid MMA intrinsic value: " + std::to_string(enumValue)).c_str());
   }
+  targetInfo.mmaIntrinsics = wrap(builder.getArrayAttr(mmaIntrinsicAttrs));
 
   return targetInfo;
 }
+
+ireeGPUMMAIntrinsicResult
+ireeGPUTargetInfoGetMMAIntrinsics(MlirAttribute mmaIntrinsics) {
+  ireeGPUMMAIntrinsicResult result = {NULL, NULL, 0};
+  if (mlirAttributeIsNull(mmaIntrinsics) ||
+      !mlirAttributeIsAArray(mmaIntrinsics)) {
+    return result;
+  }
+
+  size_t numElements = mlirArrayAttrGetNumElements(mmaIntrinsics);
+  if (numElements == 0) {
+    return result;
+  }
+
+  result.mmaIntrinsicVals =
+      (mma_intrinsic_t *)malloc(numElements * sizeof(mma_intrinsic_t));
+  result.isVirtual = (bool *)malloc(numElements * sizeof(bool));
```

**Comment:**
llvm prefers C++ casts: https://llvm.org/docs/CodingStandards.html#prefer-c-style-casts

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:537`

```diff
@@ -473,42 +474,85 @@ ireeGPUTargetInfoGet(MlirContext mlirCtx, const char *arch,
   targetInfo.maxThreadCountPerWorkgroup = threadCount;
   targetInfo.maxWorkgroupMemoryBytes = memoryBytes;
 
-  if (numMmaIntrinsics == 0) {
-    targetInfo.mmaIntrinsics = wrap(builder.getArrayAttr({}));
-  } else {
-    std::vector<mlir::Attribute> mmaIntrinsicAttrs;
-    mmaIntrinsicAttrs.reserve(numMmaIntrinsics);
-    for (size_t i = 0; i < numMmaIntrinsics; i++) {
-      int32_t enumValue = mmaIntrinsics[i];
-
-      std::optional<mlir::iree_compiler::IREE::GPU::MMAIntrinsic> mmaIntrinsic =
-          mlir::iree_compiler::IREE::GPU::symbolizeMMAIntrinsic(enumValue);
-      if (mmaIntrinsic) {
-        auto mmaIntrinsicAttr =
-            mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr::get(
-                context, *mmaIntrinsic);
-        mmaIntrinsicAttrs.push_back(mmaIntrinsicAttr);
-        continue;
-      }
+  std::vector<mlir::Attribute> mmaIntrinsicAttrs;
+  mmaIntrinsicAttrs.reserve(numMmaIntrinsics);
+  for (size_t i = 0; i < numMmaIntrinsics; i++) {
+    mma_intrinsic_t enumValue = mmaIntrinsics[i];
 
-      std::optional<mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsic>
-          virtualMmaIntrinsic =
-              mlir::iree_compiler::IREE::GPU::symbolizeVirtualMMAIntrinsic(
-                  enumValue);
-      if (virtualMmaIntrinsic) {
-        auto virtualMmaIntrinsicAttr =
-            mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsicAttr::get(
-                context, *virtualMmaIntrinsic);
-        mmaIntrinsicAttrs.push_back(virtualMmaIntrinsicAttr);
-        continue;
-      }
+    std::optional<mlir::iree_compiler::IREE::GPU::MMAIntrinsic> mmaIntrinsic =
+        mlir::iree_compiler::IREE::GPU::symbolizeMMAIntrinsic(enumValue);
+    if (mmaIntrinsic) {
+      auto mmaIntrinsicAttr =
+          mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr::get(context,
+                                                                *mmaIntrinsic);
+      mmaIntrinsicAttrs.push_back(mmaIntrinsicAttr);
+      continue;
+    }
 
-      assert(false &&
-             ("Invalid MMA intrinsic value: " + std::to_string(enumValue))
-                 .c_str());
+    std::optional<mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsic>
+        virtualMmaIntrinsic =
+            mlir::iree_compiler::IREE::GPU::symbolizeVirtualMMAIntrinsic(
+                enumValue);
+    if (virtualMmaIntrinsic) {
+      auto virtualMmaIntrinsicAttr =
+          mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsicAttr::get(
+              context, *virtualMmaIntrinsic);
+      mmaIntrinsicAttrs.push_back(virtualMmaIntrinsicAttr);
+      continue;
     }
-    targetInfo.mmaIntrinsics = wrap(builder.getArrayAttr(mmaIntrinsicAttrs));
+
+    assert(
+        false &&
+        ("Invalid MMA intrinsic value: " + std::to_string(enumValue)).c_str());
   }
+  targetInfo.mmaIntrinsics = wrap(builder.getArrayAttr(mmaIntrinsicAttrs));
 
   return targetInfo;
 }
+
+ireeGPUMMAIntrinsicResult
+ireeGPUTargetInfoGetMMAIntrinsics(MlirAttribute mmaIntrinsics) {
+  ireeGPUMMAIntrinsicResult result = {NULL, NULL, 0};
+  if (mlirAttributeIsNull(mmaIntrinsics) ||
+      !mlirAttributeIsAArray(mmaIntrinsics)) {
+    return result;
+  }
+
+  size_t numElements = mlirArrayAttrGetNumElements(mmaIntrinsics);
+  if (numElements == 0) {
+    return result;
+  }
+
+  result.mmaIntrinsicVals =
+      (mma_intrinsic_t *)malloc(numElements * sizeof(mma_intrinsic_t));
+  result.isVirtual = (bool *)malloc(numElements * sizeof(bool));
+  result.numMmaIntrinsics = numElements;
+
+  for (size_t i = 0; i < numElements; i++) {
+    MlirAttribute element = mlirArrayAttrGetElement(mmaIntrinsics, i);
+    if (ireeAttributeIsAGPUMMAIntrinsicAttr(element)) {
+      result.mmaIntrinsicVals[i] = ireeGPUMMAIntrinsicAttrGetValue(element);
+      result.isVirtual[i] = false;
+    } else if (ireeAttributeIsAGPUVirtualMMAIntrinsicAttr(element)) {
+      result.mmaIntrinsicVals[i] =
```

**Comment:**
Use early exits: https://llvm.org/docs/CodingStandards.html#use-early-exits-and-continue-to-simplify-code

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:527`

```diff
@@ -473,42 +474,85 @@ ireeGPUTargetInfoGet(MlirContext mlirCtx, const char *arch,
   targetInfo.maxThreadCountPerWorkgroup = threadCount;
   targetInfo.maxWorkgroupMemoryBytes = memoryBytes;
 
-  if (numMmaIntrinsics == 0) {
-    targetInfo.mmaIntrinsics = wrap(builder.getArrayAttr({}));
-  } else {
-    std::vector<mlir::Attribute> mmaIntrinsicAttrs;
-    mmaIntrinsicAttrs.reserve(numMmaIntrinsics);
-    for (size_t i = 0; i < numMmaIntrinsics; i++) {
-      int32_t enumValue = mmaIntrinsics[i];
-
-      std::optional<mlir::iree_compiler::IREE::GPU::MMAIntrinsic> mmaIntrinsic =
-          mlir::iree_compiler::IREE::GPU::symbolizeMMAIntrinsic(enumValue);
-      if (mmaIntrinsic) {
-        auto mmaIntrinsicAttr =
-            mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr::get(
-                context, *mmaIntrinsic);
-        mmaIntrinsicAttrs.push_back(mmaIntrinsicAttr);
-        continue;
-      }
+  std::vector<mlir::Attribute> mmaIntrinsicAttrs;
+  mmaIntrinsicAttrs.reserve(numMmaIntrinsics);
+  for (size_t i = 0; i < numMmaIntrinsics; i++) {
+    mma_intrinsic_t enumValue = mmaIntrinsics[i];
 
-      std::optional<mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsic>
-          virtualMmaIntrinsic =
-              mlir::iree_compiler::IREE::GPU::symbolizeVirtualMMAIntrinsic(
-                  enumValue);
-      if (virtualMmaIntrinsic) {
-        auto virtualMmaIntrinsicAttr =
-            mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsicAttr::get(
-                context, *virtualMmaIntrinsic);
-        mmaIntrinsicAttrs.push_back(virtualMmaIntrinsicAttr);
-        continue;
-      }
+    std::optional<mlir::iree_compiler::IREE::GPU::MMAIntrinsic> mmaIntrinsic =
+        mlir::iree_compiler::IREE::GPU::symbolizeMMAIntrinsic(enumValue);
+    if (mmaIntrinsic) {
+      auto mmaIntrinsicAttr =
+          mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr::get(context,
+                                                                *mmaIntrinsic);
+      mmaIntrinsicAttrs.push_back(mmaIntrinsicAttr);
+      continue;
+    }
 
-      assert(false &&
-             ("Invalid MMA intrinsic value: " + std::to_string(enumValue))
-                 .c_str());
+    std::optional<mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsic>
+        virtualMmaIntrinsic =
+            mlir::iree_compiler::IREE::GPU::symbolizeVirtualMMAIntrinsic(
+                enumValue);
+    if (virtualMmaIntrinsic) {
+      auto virtualMmaIntrinsicAttr =
+          mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsicAttr::get(
+              context, *virtualMmaIntrinsic);
+      mmaIntrinsicAttrs.push_back(virtualMmaIntrinsicAttr);
+      continue;
     }
-    targetInfo.mmaIntrinsics = wrap(builder.getArrayAttr(mmaIntrinsicAttrs));
+
+    assert(
+        false &&
+        ("Invalid MMA intrinsic value: " + std::to_string(enumValue)).c_str());
   }
+  targetInfo.mmaIntrinsics = wrap(builder.getArrayAttr(mmaIntrinsicAttrs));
 
   return targetInfo;
 }
+
+ireeGPUMMAIntrinsicResult
+ireeGPUTargetInfoGetMMAIntrinsics(MlirAttribute mmaIntrinsics) {
+  ireeGPUMMAIntrinsicResult result = {NULL, NULL, 0};
+  if (mlirAttributeIsNull(mmaIntrinsics) ||
+      !mlirAttributeIsAArray(mmaIntrinsics)) {
+    return result;
+  }
+
+  size_t numElements = mlirArrayAttrGetNumElements(mmaIntrinsics);
+  if (numElements == 0) {
+    return result;
+  }
+
+  result.mmaIntrinsicVals =
+      (mma_intrinsic_t *)malloc(numElements * sizeof(mma_intrinsic_t));
```

**Comment:**
The memory should be allocated and freed on the python side. We have a few examples of this already.

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:528`

```diff
@@ -515,11 +515,22 @@ NB_MODULE(_ireeCompilerDialects, m) {
           [](ireeGPUTargetInfo *self, MlirContext context,
              const std::string &arch,
              const std::vector<int32_t> &subgroupChoices,
-             const std::vector<int32_t> &workgroupSizes, int64_t threadCount,
-             int64_t memoryBytes, const py::list &mmaIntrinsicObjs) {
-            std::vector<int32_t> mmaIntrinsicVals;
+             const std::vector<int32_t> &workgroupSizes, int32_t threadCount,
+             int32_t memoryBytes, const py::list &mmaIntrinsicObjs) {
+            std::vector<mma_intrinsic_t> mmaIntrinsicVals;
+            py::module_ gpuModule = py::module_::import_(kGpuModuleImportPath);
+            py::object mmaIntrinsicClass = gpuModule.attr("MMAIntrinsic");
+            py::object virtualMmaIntrinsicClass =
+                gpuModule.attr("VirtualMMAIntrinsic");
+
             for (auto item : mmaIntrinsicObjs) {
-              int32_t enumValue = py::cast<int32_t>(item.attr("value"));
+              if (!py::isinstance(item, mmaIntrinsicClass) &&
+                  !py::isinstance(item, virtualMmaIntrinsicClass)) {
```

**Comment:**
There are defined on the Python side though: https://github.com/iree-org/iree/blob/933f798046a817dcff48d84df8fd987c5cb9e72b/compiler/bindings/python/IREECompilerDialectsModule.cpp#L312 so I'd think we can add a python base clase that doesn't exist in tablegen/c++

Fine to leave as is though for now

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:526`

```diff
@@ -509,6 +510,44 @@ NB_MODULE(_ireeCompilerDialects, m) {
   //===-------------------------------------------------------------------===//
 
   py::class_<ireeGPUTargetInfo>(iree_gpu_module, "TargetInfo")
+      .def(
+          "__init__",
+          [](ireeGPUTargetInfo *self, MlirContext context,
+             const std::string &arch,
+             const std::vector<int32_t> &subgroupChoices,
+             const std::vector<int32_t> &workgroupSizes, int32_t threadCount,
+             int32_t memoryBytes, const py::list &mmaIntrinsicObjs) {
+            std::vector<mma_intrinsic_enum_t> mmaIntrinsicVals;
+            py::module_ gpuModule = py::module_::import_(kGpuModuleImportPath);
+            py::object mmaIntrinsicClass = gpuModule.attr("MMAIntrinsic");
+            py::object virtualMmaIntrinsicClass =
+                gpuModule.attr("VirtualMMAIntrinsic");
+
+            for (auto item : mmaIntrinsicObjs) {
```

**Comment:**
https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h:176`

```diff
@@ -148,14 +155,26 @@ struct ireeGPUTargetInfo {
   MlirIdentifier arch;                // E.g., "gfx942".
   MlirAttribute subgroupSizeChoices;  // Subgroup size choices.
   MlirAttribute maxWorkgroupSizes;    // Max threads per X/Y/Z dimension.
-  int64_t maxThreadCountPerWorkgroup; // Max threads per workgroup.
-  int64_t maxWorkgroupMemoryBytes;    // Max workgroup memory.
+  int32_t maxThreadCountPerWorkgroup; // Max threads per workgroup.
+  int32_t maxWorkgroupMemoryBytes;    // Max workgroup memory.
+  MlirAttribute mmaIntrinsics;        // MMA Intrinsics.
 };
 
 // Queries GPU target info from the given `ExecutableTargetAttr` attribute.
 MLIR_CAPI_EXPORTED ireeGPUTargetInfo
 ireeHALExecutableTargetAttrGetGPUTargetInfo(MlirAttribute attr);
 
+MLIR_CAPI_EXPORTED ireeGPUTargetInfo ireeGPUTargetInfoGet(
+    MlirContext mlirCtx, const char *arch, const int32_t *subgroupChoices,
+    size_t numSubgroupChoices, const int32_t *workgroupSizes,
+    size_t numWorkgroupSizes, int32_t threadCount, int32_t memoryBytes,
+    const mma_intrinsic_enum_t *mmaIntrinsics, size_t numMmaIntrinsics);
+
+MLIR_CAPI_EXPORTED void
+ireeGPUTargetInfoGetMMAIntrinsics(MlirAttribute mmaIntrinsics,
+                                  mma_intrinsic_enum_t *mmaIntrinsicVals,
+                                  uint8_t *isVirtuals, size_t numElements);
```

**Comment:**
This name doesn't make sense to me, I'd call it something like `virtualMmaIntrinsicTags` and add a comment explaining that it's to distinguish virtual from non-virtual intrinsics

---

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h:20`

```diff
@@ -14,6 +14,11 @@
 extern "C" {
 #endif
 
+// This typedef ensures consistency between the C API, C++ implementation, and
+// Python bindings. Update both this typedef and the static assertions if the
+// enum underlying types change.
+typedef uint32_t mma_intrinsic_enum_t;
```

**Comment:**
This needs a static assert on the C bindings implementation side

---


---


## [PR #21782](https://github.com/iree-org/iree/pull/21782): [Codegen][Tuner] expose python binding to query target info

### Review Summary

**COMMENTED** (2025-08-27)


**COMMENTED** (2025-08-27)


**COMMENTED** (2025-08-27)


**APPROVED** (2025-08-28)

LGTM % one remaining issue


### Code Comments

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h:148`

```diff
@@ -144,6 +144,18 @@ struct ireeGPUMMASingleSubgroupLayout {
 MLIR_CAPI_EXPORTED ireeGPUMMASingleSubgroupLayout
 ireeGPUGetSingleSubgroupLayout(MlirAttribute attr, uint32_t fragment);
 
+struct ireeGPUTargetInfo {
+  MlirIdentifier arch;                 // "gfx942".
```

**Comment:**
```suggestion
  MlirIdentifier arch;                 // E.g., "gfx942".
```

---

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h:152`

```diff
@@ -144,6 +144,18 @@ struct ireeGPUMMASingleSubgroupLayout {
 MLIR_CAPI_EXPORTED ireeGPUMMASingleSubgroupLayout
 ireeGPUGetSingleSubgroupLayout(MlirAttribute attr, uint32_t fragment);
 
+struct ireeGPUTargetInfo {
+  MlirIdentifier arch;                 // "gfx942".
+  MlirAttribute subgroup_size_choices; // Subgroup size choices.
+  MlirAttribute max_workgroup_sizes;   // Max threads per X/Y/Z dimension.
+  MlirAttribute max_thread_count_per_workgroup; // Max threads per workgroup.
+  MlirAttribute max_workgroup_memory_bytes;     // Max workgroup memory.
```

**Comment:**
Can this be int64_t?

---

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h:151`

```diff
@@ -144,6 +144,18 @@ struct ireeGPUMMASingleSubgroupLayout {
 MLIR_CAPI_EXPORTED ireeGPUMMASingleSubgroupLayout
 ireeGPUGetSingleSubgroupLayout(MlirAttribute attr, uint32_t fragment);
 
+struct ireeGPUTargetInfo {
+  MlirIdentifier arch;                 // "gfx942".
+  MlirAttribute subgroup_size_choices; // Subgroup size choices.
+  MlirAttribute max_workgroup_sizes;   // Max threads per X/Y/Z dimension.
+  MlirAttribute max_thread_count_per_workgroup; // Max threads per workgroup.
```

**Comment:**
Can we make this int64_t?

---

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h:155`

```diff
@@ -144,6 +144,18 @@ struct ireeGPUMMASingleSubgroupLayout {
 MLIR_CAPI_EXPORTED ireeGPUMMASingleSubgroupLayout
 ireeGPUGetSingleSubgroupLayout(MlirAttribute attr, uint32_t fragment);
 
+struct ireeGPUTargetInfo {
+  MlirIdentifier arch;                 // "gfx942".
+  MlirAttribute subgroup_size_choices; // Subgroup size choices.
+  MlirAttribute max_workgroup_sizes;   // Max threads per X/Y/Z dimension.
+  MlirAttribute max_thread_count_per_workgroup; // Max threads per workgroup.
+  MlirAttribute max_workgroup_memory_bytes;     // Max workgroup memory.
+};
+
+// Add function to query GPU target info from ExecutableTargetAttr.
```

**Comment:**
This doesn't add any functions, this is a function to query something
```suggestion
// Queries GPU target info from the given `ExecutableTargetAttr` attribute |attr|.
```

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:542`

```diff
@@ -504,6 +504,45 @@ NB_MODULE(_ireeCompilerDialects, m) {
             return std::nullopt;
           });
 
+  //===-------------------------------------------------------------------===//
+  // Binding to query target info
+  //===-------------------------------------------------------------------===//
+
+  py::class_<ireeGPUTargetInfo>(iree_gpu_module, "TargetInfo")
+      .def_prop_ro("arch",
+                   [](const ireeGPUTargetInfo &self) -> std::string {
+                     MlirStringRef strRef = mlirIdentifierStr(self.arch);
+                     return std::string(strRef.data, strRef.length);
+                   })
+      .def_prop_ro("subgroup_size_choices",
+                   [](const ireeGPUTargetInfo &self) -> std::vector<int64_t> {
+                     return getIntArrayAttrValues(self.subgroup_size_choices);
+                   })
+      .def_prop_ro("max_thread_count_per_workgroup",
+                   [](const ireeGPUTargetInfo &self) -> int64_t {
+                     return mlirIntegerAttrGetValueInt(
+                         self.max_thread_count_per_workgroup);
+                   })
+      .def_prop_ro("max_workgroup_sizes",
+                   [](const ireeGPUTargetInfo &self) -> std::vector<int64_t> {
+                     return getIntArrayAttrValues(self.max_workgroup_sizes);
+                   })
+      .def_prop_ro("max_workgroup_memory_bytes",
+                   [](const ireeGPUTargetInfo &self) -> int64_t {
+                     return mlirIntegerAttrGetValueInt(
+                         self.max_workgroup_memory_bytes);
+                   });
+
+  iree_gpu_module.def(
+      "get_gpu_target_info",
+      [](MlirAttribute executableTargetAttr) {
+        ireeGPUTargetInfo result =
+            ireeHALExecutableTargetAttrGetGPUTargetInfo(executableTargetAttr);
+        return result;
+      },
```

**Comment:**
Can we use `ireeHALExecutableTargetAttrGetGPUTargetInfo` directly instead of wrapping it in a lambda?

---

**File:** `compiler/bindings/python/test/ir/dialects_test.py:410`

```diff
@@ -391,3 +391,70 @@ def compilation_info():
     assert compilation_info is not None
     assert compilation_info.lowering_config == lowering_config
     assert compilation_info.translation_info == translation_info
+
+
+@run
+def gpu_target_info_attribute_parsing():
+    mlir_string = """
+    hal.executable private @main_dispatch_0 {
+        hal.executable.variant public @rocm_hsaco_fb
+            target(<"rocm", "rocm-hsaco-fb",
+                {
+                abi = "hip",
+                iree_codegen.target_info = #iree_gpu.target<
+                    arch = "gfx942",
+                    features = "",
+                    wgp = <
+                    compute = fp64|fp32|fp16,
+                    storage = b64|b32|b16|b8,
+                    subgroup = shuffle|arithmetic,
```

**Comment:**
Could we set the attributes we don't care about to the empty value, e.g., `none`, to keep this minimal?

---

**File:** `compiler/bindings/python/test/ir/dialects_test.py:413`

```diff
@@ -391,3 +391,70 @@ def compilation_info():
     assert compilation_info is not None
     assert compilation_info.lowering_config == lowering_config
     assert compilation_info.translation_info == translation_info
+
+
+@run
+def gpu_target_info_attribute_parsing():
+    mlir_string = """
+    hal.executable private @main_dispatch_0 {
+        hal.executable.variant public @rocm_hsaco_fb
+            target(<"rocm", "rocm-hsaco-fb",
+                {
+                abi = "hip",
+                iree_codegen.target_info = #iree_gpu.target<
+                    arch = "gfx942",
+                    features = "",
+                    wgp = <
+                    compute = fp64|fp32|fp16,
+                    storage = b64|b32|b16|b8,
+                    subgroup = shuffle|arithmetic,
+                    dot = dp4xi8toi32,
+                    mma = [<MFMA_F32_16x16x4_F32>],
+                    subgroup_size_choices = [64],
```

**Comment:**
Can you add a testcase that has more than one value for subgroup size coices?

---

**File:** `compiler/bindings/python/test/ir/dialects_test.py:417`

```diff
@@ -391,3 +391,70 @@ def compilation_info():
     assert compilation_info is not None
     assert compilation_info.lowering_config == lowering_config
     assert compilation_info.translation_info == translation_info
+
+
+@run
+def gpu_target_info_attribute_parsing():
+    mlir_string = """
+    hal.executable private @main_dispatch_0 {
+        hal.executable.variant public @rocm_hsaco_fb
+            target(<"rocm", "rocm-hsaco-fb",
+                {
+                abi = "hip",
+                iree_codegen.target_info = #iree_gpu.target<
+                    arch = "gfx942",
+                    features = "",
+                    wgp = <
+                    compute = fp64|fp32|fp16,
+                    storage = b64|b32|b16|b8,
+                    subgroup = shuffle|arithmetic,
+                    dot = dp4xi8toi32,
+                    mma = [<MFMA_F32_16x16x4_F32>],
+                    subgroup_size_choices = [64],
+                    max_workgroup_sizes = [1024, 1024, 1024],
+                    max_thread_count_per_workgroup = 1024,
+                    max_workgroup_memory_bytes = 65536,
+                    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
```

**Comment:**
For max wokrgroup sizes and counts, I would pick 3 distinct values so that we can test they appear in the correct order on the python side

---

**File:** `compiler/bindings/python/test/ir/dialects_test.py:438`

```diff
@@ -391,3 +391,70 @@ def compilation_info():
     assert compilation_info is not None
     assert compilation_info.lowering_config == lowering_config
     assert compilation_info.translation_info == translation_info
+
+
+@run
+def gpu_target_info_attribute_parsing():
+    mlir_string = """
+    hal.executable private @main_dispatch_0 {
+        hal.executable.variant public @rocm_hsaco_fb
+            target(<"rocm", "rocm-hsaco-fb",
+                {
+                abi = "hip",
+                iree_codegen.target_info = #iree_gpu.target<
+                    arch = "gfx942",
+                    features = "",
+                    wgp = <
+                    compute = fp64|fp32|fp16,
+                    storage = b64|b32|b16|b8,
+                    subgroup = shuffle|arithmetic,
+                    dot = dp4xi8toi32,
+                    mma = [<MFMA_F32_16x16x4_F32>],
+                    subgroup_size_choices = [64],
+                    max_workgroup_sizes = [1024, 1024, 1024],
+                    max_thread_count_per_workgroup = 1024,
+                    max_workgroup_memory_bytes = 65536,
+                    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
+                    max_load_instruction_bits = 128,
+                    simds_per_wgp = 4,
+                    vgpr_space_bits = 16384
+                    >
+                >
+                }>
+            ) {
+        }
+    }
+    """
+
+    module = ir.Module.parse(mlir_string)
+    variant_op_list = iree_codegen.get_executable_variant_ops(module)
+    assert len(variant_op_list) == 1, "Expect one executable variant op"
+    variant_op = variant_op_list[0]
+    executable_variant_op = variant_op.opview
+    target = executable_variant_op.target
+    gpu_target_info = iree_gpu.get_gpu_target_info(target)
+
+    arch = gpu_target_info.arch
+    assert arch == "gfx942", f"Expected arch 'gfx942', got '{arch}'"
```

**Comment:**
I think failed assertions are going to print the actual and expected values -- could you check if this is the case, and remove those messages if it works like I described? Also below.

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:403`

```diff
@@ -387,3 +388,41 @@ ireeGPUGetSingleSubgroupLayout(MlirAttribute attr, uint32_t fragment) {
   result.element = wrap(builder.getI64ArrayAttr(layout.element));
   return result;
 }
+
+ireeGPUTargetInfo
+ireeHALExecutableTargetAttrGetGPUTargetInfo(MlirAttribute attr) {
+  assert(!mlirAttributeIsNull(attr) && "attr cannot be null");
+  auto executableTargetAttr =
+      llvm::cast<mlir::iree_compiler::IREE::HAL::ExecutableTargetAttr>(
+          unwrap(attr));
+
+  assert(executableTargetAttr && "attr is not a HAL::ExecutableTargetAttr");
+
+  ireeGPUTargetInfo targetInfo = {};
+  auto context = executableTargetAttr.getContext();
+  auto gpuTargetAttr =
```

**Comment:**
Please spell out the types here and below, where the type is not obvious based on the RHS only. See https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h:149`

```diff
@@ -144,6 +144,18 @@ struct ireeGPUMMASingleSubgroupLayout {
 MLIR_CAPI_EXPORTED ireeGPUMMASingleSubgroupLayout
 ireeGPUGetSingleSubgroupLayout(MlirAttribute attr, uint32_t fragment);
 
+struct ireeGPUTargetInfo {
+  MlirIdentifier arch;                 // "gfx942".
+  MlirAttribute subgroup_size_choices; // Subgroup size choices.
```

**Comment:**
Also use the mlir naming standard for all fields in this class

---

**File:** `compiler/bindings/python/test/ir/dialects_test.py:413`

```diff
@@ -391,3 +391,109 @@ def compilation_info():
     assert compilation_info is not None
     assert compilation_info.lowering_config == lowering_config
     assert compilation_info.translation_info == translation_info
+
+
+@run
+def gpu_target_info_attribute_parsing():
+    mlir_string = """
+    hal.executable private @main_dispatch_0 {
+        hal.executable.variant public @rocm_hsaco_fb
+            target(<"rocm", "rocm-hsaco-fb",
+                {
+                abi = "hip",
+                iree_codegen.target_info = #iree_gpu.target<
+                    arch = "gfx942",
+                    features = "",
+                    wgp = <
+                    compute = fp64,
+                    storage = b64,
+                    subgroup = none,
+                    dot = none,
+                    mma = [<MFMA_F32_16x16x4_F32>],
+                    subgroup_size_choices = [64],
```

**Comment:**
Can we populate this with at least two values and make sure they are in the correct order?

---

**File:** `compiler/bindings/python/test/ir/dialects_test.py:417`

```diff
@@ -391,3 +391,109 @@ def compilation_info():
     assert compilation_info is not None
     assert compilation_info.lowering_config == lowering_config
     assert compilation_info.translation_info == translation_info
+
+
+@run
+def gpu_target_info_attribute_parsing():
+    mlir_string = """
+    hal.executable private @main_dispatch_0 {
+        hal.executable.variant public @rocm_hsaco_fb
+            target(<"rocm", "rocm-hsaco-fb",
+                {
+                abi = "hip",
+                iree_codegen.target_info = #iree_gpu.target<
+                    arch = "gfx942",
+                    features = "",
+                    wgp = <
+                    compute = fp64,
+                    storage = b64,
+                    subgroup = none,
+                    dot = none,
+                    mma = [<MFMA_F32_16x16x4_F32>],
+                    subgroup_size_choices = [64],
+                    max_workgroup_sizes = [256, 512, 1024],
+                    max_thread_count_per_workgroup = 1024,
+                    max_workgroup_memory_bytes = 65536,
+                    max_workgroup_counts = [2147483647, 2147483647, 2147483647]
```

**Comment:**
Can we make these distinct?

---

**File:** `compiler/bindings/python/test/ir/dialects_test.py:478`

```diff
@@ -391,3 +391,109 @@ def compilation_info():
     assert compilation_info is not None
     assert compilation_info.lowering_config == lowering_config
     assert compilation_info.translation_info == translation_info
+
+
+@run
+def gpu_target_info_attribute_parsing():
+    mlir_string = """
+    hal.executable private @main_dispatch_0 {
+        hal.executable.variant public @rocm_hsaco_fb
+            target(<"rocm", "rocm-hsaco-fb",
+                {
+                abi = "hip",
+                iree_codegen.target_info = #iree_gpu.target<
+                    arch = "gfx942",
+                    features = "",
+                    wgp = <
+                    compute = fp64,
+                    storage = b64,
+                    subgroup = none,
+                    dot = none,
+                    mma = [<MFMA_F32_16x16x4_F32>],
+                    subgroup_size_choices = [64],
+                    max_workgroup_sizes = [256, 512, 1024],
+                    max_thread_count_per_workgroup = 1024,
+                    max_workgroup_memory_bytes = 65536,
+                    max_workgroup_counts = [2147483647, 2147483647, 2147483647]
+                    >
+                >
+                }>
+            ) {
+        }
+    }
+    """
+
+    module = ir.Module.parse(mlir_string)
+    variant_op_list = iree_codegen.get_executable_variant_ops(module)
+    assert len(variant_op_list) == 1, "Expect one executable variant op"
+    variant_op = variant_op_list[0]
+    executable_variant_op = variant_op.opview
+    target = executable_variant_op.target
+    gpu_target_info = iree_gpu.get_gpu_target_info(target)
+
+    arch = gpu_target_info.arch
+    assert arch == "gfx942", f"Expected arch 'gfx942', got '{arch}'"
+
+    subgroup_size_choices = gpu_target_info.subgroup_size_choices
+    assert subgroup_size_choices == [
+        64
+    ], f"Expected subgroup_size_choice [64], got {subgroup_size_choices}"
+
+    max_thread_count = gpu_target_info.max_thread_count_per_workgroup
+    assert (
+        max_thread_count == 1024
+    ), f"Expected max_thread_count_per_workgroup 1024, got {max_thread_count}"
+
+    max_memory_bytes = gpu_target_info.max_workgroup_memory_bytes
+    assert (
+        max_memory_bytes == 65536
+    ), f"Expected max_workgroup_memory_bytes 65536, got {max_memory_bytes}"
+
+    max_workgroup_sizes = gpu_target_info.max_workgroup_sizes
+    assert max_workgroup_sizes == [
+        256,
+        512,
+        1024,
+    ], f"Expected max_workgroup_sizes [256, 512, 1024], got {max_workgroup_sizes}"
+
+    mlir_string = """
+    hal.executable private @main_dispatch_1 {
+        hal.executable.variant public @rocm_hsaco_fb
+            target(<"rocm", "rocm-hsaco-fb",
+                {
+                abi = "hip",
+                iree_codegen.target_info = #iree_gpu.target<
+                    arch = "gfx942",
+                    features = "",
+                    wgp = <
+                    compute = fp16,
+                    storage = b16,
+                    subgroup = none,
+                    dot = none,
+                    mma = [],
+                    subgroup_size_choices = [32, 64],
+                    max_workgroup_sizes = [1024, 1024, 1024],
+                    max_thread_count_per_workgroup = 1024,
+                    max_workgroup_memory_bytes = 65536,
+                    max_workgroup_counts = [1024]
```

**Comment:**
IIRC, this should always match the x / y / z dimensions, so this config seems invalid

---

**File:** `compiler/bindings/python/test/ir/dialects_test.py:460`

```diff
@@ -391,3 +391,109 @@ def compilation_info():
     assert compilation_info is not None
     assert compilation_info.lowering_config == lowering_config
     assert compilation_info.translation_info == translation_info
+
+
+@run
+def gpu_target_info_attribute_parsing():
+    mlir_string = """
+    hal.executable private @main_dispatch_0 {
+        hal.executable.variant public @rocm_hsaco_fb
+            target(<"rocm", "rocm-hsaco-fb",
+                {
+                abi = "hip",
+                iree_codegen.target_info = #iree_gpu.target<
+                    arch = "gfx942",
+                    features = "",
+                    wgp = <
+                    compute = fp64,
+                    storage = b64,
+                    subgroup = none,
+                    dot = none,
+                    mma = [<MFMA_F32_16x16x4_F32>],
+                    subgroup_size_choices = [64],
+                    max_workgroup_sizes = [256, 512, 1024],
+                    max_thread_count_per_workgroup = 1024,
+                    max_workgroup_memory_bytes = 65536,
+                    max_workgroup_counts = [2147483647, 2147483647, 2147483647]
+                    >
+                >
+                }>
+            ) {
+        }
+    }
+    """
+
+    module = ir.Module.parse(mlir_string)
+    variant_op_list = iree_codegen.get_executable_variant_ops(module)
+    assert len(variant_op_list) == 1, "Expect one executable variant op"
+    variant_op = variant_op_list[0]
+    executable_variant_op = variant_op.opview
+    target = executable_variant_op.target
+    gpu_target_info = iree_gpu.get_gpu_target_info(target)
+
+    arch = gpu_target_info.arch
+    assert arch == "gfx942", f"Expected arch 'gfx942', got '{arch}'"
+
+    subgroup_size_choices = gpu_target_info.subgroup_size_choices
+    assert subgroup_size_choices == [
+        64
+    ], f"Expected subgroup_size_choice [64], got {subgroup_size_choices}"
+
+    max_thread_count = gpu_target_info.max_thread_count_per_workgroup
+    assert (
+        max_thread_count == 1024
+    ), f"Expected max_thread_count_per_workgroup 1024, got {max_thread_count}"
+
+    max_memory_bytes = gpu_target_info.max_workgroup_memory_bytes
+    assert (
+        max_memory_bytes == 65536
+    ), f"Expected max_workgroup_memory_bytes 65536, got {max_memory_bytes}"
+
+    max_workgroup_sizes = gpu_target_info.max_workgroup_sizes
+    assert max_workgroup_sizes == [
+        256,
+        512,
+        1024,
+    ], f"Expected max_workgroup_sizes [256, 512, 1024], got {max_workgroup_sizes}"
+
+    mlir_string = """
+    hal.executable private @main_dispatch_1 {
```

**Comment:**
Why do we need a second test input? I think one should be enough to exercise everything we need?

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:406`

```diff
@@ -387,3 +388,37 @@ ireeGPUGetSingleSubgroupLayout(MlirAttribute attr, uint32_t fragment) {
   result.element = wrap(builder.getI64ArrayAttr(layout.element));
   return result;
 }
+
+ireeGPUTargetInfo
+ireeHALExecutableTargetAttrGetGPUTargetInfo(MlirAttribute attr) {
+  assert(!mlirAttributeIsNull(attr) && "attr cannot be null");
+  auto executableTargetAttr =
+      llvm::cast<mlir::iree_compiler::IREE::HAL::ExecutableTargetAttr>(
+          unwrap(attr));
+
+  assert(executableTargetAttr && "attr is not a HAL::ExecutableTargetAttr");
+
+  ireeGPUTargetInfo targetInfo = {};
+  mlir::MLIRContext *context = executableTargetAttr.getContext();
+  mlir::iree_compiler::IREE::GPU::TargetAttr gpuTargetAttr =
+      mlir::iree_compiler::getGPUTargetAttr(context, executableTargetAttr);
+
+  if (gpuTargetAttr) {
```

**Comment:**
Use an early return instead: https://llvm.org/docs/CodingStandards.html#use-early-exits-and-continue-to-simplify-code

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:399`

```diff
@@ -387,3 +388,39 @@ ireeGPUGetSingleSubgroupLayout(MlirAttribute attr, uint32_t fragment) {
   result.element = wrap(builder.getI64ArrayAttr(layout.element));
   return result;
 }
+
+ireeGPUTargetInfo
+ireeHALExecutableTargetAttrGetGPUTargetInfo(MlirAttribute attr) {
+  assert(!mlirAttributeIsNull(attr) && "attr cannot be null");
+  auto executableTargetAttr =
+      llvm::cast<mlir::iree_compiler::IREE::HAL::ExecutableTargetAttr>(
+          unwrap(attr));
+
+  assert(executableTargetAttr && "attr is not a HAL::ExecutableTargetAttr");
```

**Comment:**
This assert will never trigger because `llvm::cast` checks that the types match and asserts internally (unlike `llvm::dyn_cast`). See https://llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:399`

```diff
@@ -387,3 +388,39 @@ ireeGPUGetSingleSubgroupLayout(MlirAttribute attr, uint32_t fragment) {
   result.element = wrap(builder.getI64ArrayAttr(layout.element));
   return result;
 }
+
+ireeGPUTargetInfo
+ireeHALExecutableTargetAttrGetGPUTargetInfo(MlirAttribute attr) {
+  assert(!mlirAttributeIsNull(attr) && "attr cannot be null");
+  auto executableTargetAttr =
+      llvm::cast<mlir::iree_compiler::IREE::HAL::ExecutableTargetAttr>(
+          unwrap(attr));
+
+  assert(executableTargetAttr && "attr is not a HAL::ExecutableTargetAttr");
```

**Comment:**
We can delete this assert

---


---


## [PR #21537](https://github.com/iree-org/iree/pull/21537): [GPU] Add col_major optional attribution to VirtualMMAAttr

### Review Summary

**APPROVED** (2025-07-31)



---


## [PR #21454](https://github.com/iree-org/iree/pull/21454): [Codegen][Tuner]: expose python binding for mma single subgroup layout

### Review Summary

**COMMENTED** (2025-07-22)


**COMMENTED** (2025-07-22)


**APPROVED** (2025-07-22)

LGTM and thanks for the cleanup! The old code looks much cleaner now


### Code Comments

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:548`

```diff
@@ -518,6 +518,36 @@ NB_MODULE(_ireeCompilerDialects, m) {
             return std::nullopt;
           });
 
+  //===-------------------------------------------------------------------===//
+  // Binding to utility function getSingleSubgroupLayout
+  //===-------------------------------------------------------------------===//
+  py::class_<ireeGPUMMASingleSubgroupLayout>(iree_gpu_module,
+                                             "GPUMMASingleSubgroupLayout")
+      .def_prop_ro(
+          "outer",
+          [](const ireeGPUMMASingleSubgroupLayout &self) { return self.outer; })
+      .def_prop_ro("thread",
+                   [](const ireeGPUMMASingleSubgroupLayout &self) {
+                     return self.thread;
+                   })
+      .def_prop_ro("tstrides",
+                   [](const ireeGPUMMASingleSubgroupLayout &self) {
+                     return self.tstrides;
+                   })
+      .def_prop_ro("element", [](const ireeGPUMMASingleSubgroupLayout &self) {
+        return self.element;
+      });
+
+  iree_gpu_module.def(
+      "get_single_subgroup_layout",
+      [](MlirAttribute attr, int fragment) {
+        return ireeGPUGetSingleSubgroupLayout(attr, fragment);
+      },
+      "Returns the single subgroup layout (element, thread, outer, "
+      "tstrides) "
+      "for a given MMA or VirtualMMA intrinsic and fragment.",
```

**Comment:**
nit: this string is broken in a weird way -- could you reflow this? I'd expect something like
```suggestion
      "tstrides) for a given MMA or VirtualMMA intrinsic and "
      "fragment.",
```

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:390`

```diff
@@ -350,3 +350,42 @@ MlirAttribute ireeGPULoweringConfigAttrGetMmaKind(MlirAttribute attr) {
 
   return wrap(mma_attr);
 }
+
+ireeGPUMMASingleSubgroupLayout
+ireeGPUGetSingleSubgroupLayout(MlirAttribute attr, uint32_t fragment) {
+  assert(ireeAttributeIsAGPUMMAIntrinsicAttr(attr) ||
+         ireeAttributeIsAGPUVirtualMMAIntrinsicAttr(attr) &&
+             "Expected MMA or VirtualMMA Intrinsic");
+
+  mlir::Attribute baseAttr = unwrap(attr);
+  mlir::iree_compiler::IREE::GPU::MMASingleSubgroupLayout layout;
+  mlir::iree_compiler::IREE::GPU::MMAFragment frag =
+      static_cast<mlir::iree_compiler::IREE::GPU::MMAFragment>(fragment);
+
+  if (auto intrinsicAttr =
+          llvm::dyn_cast<mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr>(
+              baseAttr)) {
+    layout = mlir::iree_compiler::IREE::GPU::getSingleSubgroupLayout(
+        intrinsicAttr.getValue(), frag);
+  } else if (auto virtualIntrinsicAttr = llvm::dyn_cast<
+                 mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsicAttr>(
+                 baseAttr)) {
+    layout = mlir::iree_compiler::IREE::GPU::getSingleSubgroupLayout(
+        virtualIntrinsicAttr.getValue(), frag);
+  } else {
+    assert(false &&
+           "Unreachable: attribute must be MMA or VirtualMMA intrinsic");
+  }
+
+  mlir::MLIRContext *context = baseAttr.getContext();
+  mlir::Builder builder(context);
+
+  ireeGPUMMASingleSubgroupLayout result;
+
+  result.outer = wrap(builder.getI64ArrayAttr(layout.outer));
+  result.thread = wrap(builder.getI64ArrayAttr(layout.thread));
+  result.tstrides = wrap(builder.getI64ArrayAttr(layout.tstrides));
+  result.element = wrap(builder.getI64ArrayAttr(layout.element));
+
+  return result;
```

**Comment:**
```suggestion
  ireeGPUMMASingleSubgroupLayout result;
  result.outer = wrap(builder.getI64ArrayAttr(layout.outer));
  result.thread = wrap(builder.getI64ArrayAttr(layout.thread));
  result.tstrides = wrap(builder.getI64ArrayAttr(layout.tstrides));
  result.element = wrap(builder.getI64ArrayAttr(layout.element));
  return result;
```

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:383`

```diff
@@ -350,3 +350,42 @@ MlirAttribute ireeGPULoweringConfigAttrGetMmaKind(MlirAttribute attr) {
 
   return wrap(mma_attr);
 }
+
+ireeGPUMMASingleSubgroupLayout
+ireeGPUGetSingleSubgroupLayout(MlirAttribute attr, uint32_t fragment) {
+  assert(ireeAttributeIsAGPUMMAIntrinsicAttr(attr) ||
+         ireeAttributeIsAGPUVirtualMMAIntrinsicAttr(attr) &&
+             "Expected MMA or VirtualMMA Intrinsic");
+
+  mlir::Attribute baseAttr = unwrap(attr);
+  mlir::iree_compiler::IREE::GPU::MMASingleSubgroupLayout layout;
+  mlir::iree_compiler::IREE::GPU::MMAFragment frag =
+      static_cast<mlir::iree_compiler::IREE::GPU::MMAFragment>(fragment);
+
+  if (auto intrinsicAttr =
+          llvm::dyn_cast<mlir::iree_compiler::IREE::GPU::MMAIntrinsicAttr>(
+              baseAttr)) {
+    layout = mlir::iree_compiler::IREE::GPU::getSingleSubgroupLayout(
+        intrinsicAttr.getValue(), frag);
+  } else if (auto virtualIntrinsicAttr = llvm::dyn_cast<
+                 mlir::iree_compiler::IREE::GPU::VirtualMMAIntrinsicAttr>(
+                 baseAttr)) {
+    layout = mlir::iree_compiler::IREE::GPU::getSingleSubgroupLayout(
+        virtualIntrinsicAttr.getValue(), frag);
+  } else {
+    assert(false &&
+           "Unreachable: attribute must be MMA or VirtualMMA intrinsic");
+  }
+
+  mlir::MLIRContext *context = baseAttr.getContext();
+  mlir::Builder builder(context);
+
+  ireeGPUMMASingleSubgroupLayout result;
```

**Comment:**
We can start by zero-initializing the whole struct in case we forgot to initialize some field later on. Zero values are usually easier to track down than uninitialized garbage.
```suggestion
  ireeGPUMMASingleSubgroupLayout result = {};
```

---

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h:139`

```diff
@@ -132,6 +132,16 @@ ireeGPULoweringConfigAttrGetSubgroupCount(MlirAttribute attr);
 MLIR_CAPI_EXPORTED MlirAttribute
 ireeGPULoweringConfigAttrGetMmaKind(MlirAttribute attr);
 
+struct ireeGPUMMASingleSubgroupLayout {
+  MlirAttribute outer;
+  MlirAttribute thread;
+  MlirAttribute tstrides;
+  MlirAttribute element;
```

**Comment:**
Can you comment on the underlying data type? I think this will be an arrayattr of integers/index?

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:528`

```diff
@@ -518,6 +518,36 @@ NB_MODULE(_ireeCompilerDialects, m) {
             return std::nullopt;
           });
 
+  //===-------------------------------------------------------------------===//
+  // Binding to utility function getSingleSubgroupLayout
+  //===-------------------------------------------------------------------===//
+  py::class_<ireeGPUMMASingleSubgroupLayout>(iree_gpu_module,
+                                             "GPUMMASingleSubgroupLayout")
+      .def_prop_ro(
+          "outer",
+          [](const ireeGPUMMASingleSubgroupLayout &self) { return self.outer; })
```

**Comment:**
Should we return an arrayattr or a python list of integers? I think the latter will be easier to work with, especially since all of these are read only

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:66`

```diff
@@ -61,6 +61,23 @@ ireeCodegenGetTunerRootOpsBinding(MlirModule module) {
   return ops;
 }
 
+static std::vector<int64_t> getIntArrayAttrValues(MlirAttribute attr) {
+  mlir::Attribute Attr = unwrap(attr);
+  auto arrayAttr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(Attr);
```

**Comment:**
Use the C API for the type check instead -- notice how the whole file uses C APIs only.

---


---


## [PR #21408](https://github.com/iree-org/iree/pull/21408): Integrate LLVM to llvm/llvm-project@5f53182

### Review Summary

**APPROVED** (2025-07-19)



---


## [PR #21403](https://github.com/iree-org/iree/pull/21403): [Codegen][Tuner] add python binding for VirtualMMAIntrinsic

### Review Summary

**COMMENTED** (2025-07-18)


**COMMENTED** (2025-07-18)


**COMMENTED** (2025-07-19)


**APPROVED** (2025-07-19)

LGTM % nit


### Code Comments

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h:106`

```diff
@@ -82,6 +99,12 @@ struct ireeGPUMMAInfo {
 
 MLIR_CAPI_EXPORTED ireeGPUMMAInfo ireeGPUMMAAttrGetInfo(MlirAttribute attr);
 
+MLIR_CAPI_EXPORTED ireeGPUMMAInfo
+ireeGPUVirtualMMAAttrGetInfo(MlirAttribute attr);
+
+MLIR_CAPI_EXPORTED MlirAttribute
+ireeGPUMMAAttrGetVirtualMMAIntrinsic(MlirAttribute attr);
```

**Comment:**
Could we use the same functions to handle virtual and non-virtual mma intrinsics? I mean one function that would subsume both `ireeGPUVirtualMMAAttrGetInfo` and  `ireeGPUMMAAttrGetInfo`.

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:249`

```diff
@@ -220,21 +220,34 @@ MlirAttribute ireeGPUVirtualMMAAttrGet(MlirContext mlirCtx, uint32_t value) {
 }
 
 ireeGPUMMAInfo ireeGPUMMAAttrGetInfo(MlirAttribute attr) {
-  assert(ireeAttributeIsAGPUMMAAttr(attr) && "attr is not a MMAAttr");
-  auto mma = llvm::cast<mlir::iree_compiler::IREE::GPU::MMAAttr>(unwrap(attr));
-
+  assert(ireeAttributeIsAGPUMMAAttr(attr) ||
+         ireeAttributeIsAGPUVirtualMMAAttr(attr) &&
+             "Expected MMAAttr or VirtualMMAAttr");
   ireeGPUMMAInfo info = {};
-  auto [aType, bType, cType] = mma.getABCElementTypes();
-  info.aElementType = wrap(aType);
-  info.bElementType = wrap(bType);
-  info.cElementType = wrap(cType);
 
-  auto [aVecType, bVecType, cVecType] = mma.getABCVectorTypes();
-  info.aVectorType = wrap(aVecType);
-  info.bVectorType = wrap(bVecType);
-  info.cVectorType = wrap(cVecType);
+  auto setMMAInfo = [&](auto mma) {
+    auto [aType, bType, cType] = mma.getABCElementTypes();
+    info.aElementType = wrap(aType);
+    info.bElementType = wrap(bType);
+    info.cElementType = wrap(cType);
+
+    auto [aVecType, bVecType, cVecType] = mma.getABCVectorTypes();
+    info.aVectorType = wrap(aVecType);
+    info.bVectorType = wrap(bVecType);
+    info.cVectorType = wrap(cVecType);
+
+    std::tie(info.mElements, info.nElements, info.kElements) =
+        mma.getMNKShape();
+  };
+
+  if (ireeAttributeIsAGPUMMAAttr(attr)) {
+    setMMAInfo(
+        llvm::cast<mlir::iree_compiler::IREE::GPU::MMAAttr>(unwrap(attr)));
+    return info;
+  }
 
-  std::tie(info.mElements, info.nElements, info.kElements) = mma.getMNKShape();
+  setMMAInfo(
```

**Comment:**
you can use `llvm::TypeSwitch`

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:249`

```diff
@@ -220,21 +220,34 @@ MlirAttribute ireeGPUVirtualMMAAttrGet(MlirContext mlirCtx, uint32_t value) {
 }
 
 ireeGPUMMAInfo ireeGPUMMAAttrGetInfo(MlirAttribute attr) {
-  assert(ireeAttributeIsAGPUMMAAttr(attr) && "attr is not a MMAAttr");
-  auto mma = llvm::cast<mlir::iree_compiler::IREE::GPU::MMAAttr>(unwrap(attr));
-
+  assert(ireeAttributeIsAGPUMMAAttr(attr) ||
+         ireeAttributeIsAGPUVirtualMMAAttr(attr) &&
+             "Expected MMAAttr or VirtualMMAAttr");
   ireeGPUMMAInfo info = {};
-  auto [aType, bType, cType] = mma.getABCElementTypes();
-  info.aElementType = wrap(aType);
-  info.bElementType = wrap(bType);
-  info.cElementType = wrap(cType);
 
-  auto [aVecType, bVecType, cVecType] = mma.getABCVectorTypes();
-  info.aVectorType = wrap(aVecType);
-  info.bVectorType = wrap(bVecType);
-  info.cVectorType = wrap(cVecType);
+  auto setMMAInfo = [&](auto mma) {
+    auto [aType, bType, cType] = mma.getABCElementTypes();
+    info.aElementType = wrap(aType);
+    info.bElementType = wrap(bType);
+    info.cElementType = wrap(cType);
+
+    auto [aVecType, bVecType, cVecType] = mma.getABCVectorTypes();
+    info.aVectorType = wrap(aVecType);
+    info.bVectorType = wrap(bVecType);
+    info.cVectorType = wrap(cVecType);
+
+    std::tie(info.mElements, info.nElements, info.kElements) =
+        mma.getMNKShape();
+  };
+
+  if (ireeAttributeIsAGPUMMAAttr(attr)) {
+    setMMAInfo(
+        llvm::cast<mlir::iree_compiler::IREE::GPU::MMAAttr>(unwrap(attr)));
+    return info;
+  }
 
-  std::tie(info.mElements, info.nElements, info.kElements) = mma.getMNKShape();
+  setMMAInfo(
```

**Comment:**
My apologies, I should have been more explicit here. Type switch is a standard pattern to deal with the case you ran into here: passing type-erased type to same generic code that needs to know the derived type. You can drop the whole `setMMAInfo` lambda and do something like this:
```c++
  return TypeSwitch<Attribute, ireeGPUMMAInfo>(unwrap(attr))
    .Case<AttrTypeA, AttrTypeB>([](auto mmaAttr){
      // The logic from setMMAInfo...
    }).Default([](Attribute) { assert(false && "..."); return ireeGPUMMAInfo{}; });
```

This way you don't have type switch do all the `dyn_cast`s for you and handle the wrong type case. You can also return the mma info directly without having to declare it outside of the lambda and capture it for modification.

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:223`

```diff
@@ -220,40 +220,34 @@ MlirAttribute ireeGPUVirtualMMAAttrGet(MlirContext mlirCtx, uint32_t value) {
 }
 
 ireeGPUMMAInfo ireeGPUMMAAttrGetInfo(MlirAttribute attr) {
-  assert(ireeAttributeIsAGPUMMAAttr(attr) ||
-         ireeAttributeIsAGPUVirtualMMAAttr(attr) &&
-             "Expected MMAAttr or VirtualMMAAttr");
-  ireeGPUMMAInfo info = {};
-
-  auto setMMAInfo = [&](auto mma) {
-    auto [aType, bType, cType] = mma.getABCElementTypes();
-    info.aElementType = wrap(aType);
-    info.bElementType = wrap(bType);
-    info.cElementType = wrap(cType);
-
-    auto [aVecType, bVecType, cVecType] = mma.getABCVectorTypes();
-    info.aVectorType = wrap(aVecType);
-    info.bVectorType = wrap(bVecType);
-    info.cVectorType = wrap(cVecType);
-
-    std::tie(info.mElements, info.nElements, info.kElements) =
-        mma.getMNKShape();
-  };
-
-  if (ireeAttributeIsAGPUMMAAttr(attr)) {
-    setMMAInfo(
-        llvm::cast<mlir::iree_compiler::IREE::GPU::MMAAttr>(unwrap(attr)));
-    return info;
-  }
+  assert((ireeAttributeIsAGPUMMAAttr(attr) ||
```

**Comment:**
This assertion os redundant to the one in the default case

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:230`

```diff
@@ -220,40 +220,34 @@ MlirAttribute ireeGPUVirtualMMAAttrGet(MlirContext mlirCtx, uint32_t value) {
 }
 
 ireeGPUMMAInfo ireeGPUMMAAttrGetInfo(MlirAttribute attr) {
-  assert(ireeAttributeIsAGPUMMAAttr(attr) ||
-         ireeAttributeIsAGPUVirtualMMAAttr(attr) &&
-             "Expected MMAAttr or VirtualMMAAttr");
-  ireeGPUMMAInfo info = {};
-
-  auto setMMAInfo = [&](auto mma) {
-    auto [aType, bType, cType] = mma.getABCElementTypes();
-    info.aElementType = wrap(aType);
-    info.bElementType = wrap(bType);
-    info.cElementType = wrap(cType);
-
-    auto [aVecType, bVecType, cVecType] = mma.getABCVectorTypes();
-    info.aVectorType = wrap(aVecType);
-    info.bVectorType = wrap(bVecType);
-    info.cVectorType = wrap(cVecType);
-
-    std::tie(info.mElements, info.nElements, info.kElements) =
-        mma.getMNKShape();
-  };
-
-  if (ireeAttributeIsAGPUMMAAttr(attr)) {
-    setMMAInfo(
-        llvm::cast<mlir::iree_compiler::IREE::GPU::MMAAttr>(unwrap(attr)));
-    return info;
-  }
+  assert((ireeAttributeIsAGPUMMAAttr(attr) ||
+          ireeAttributeIsAGPUVirtualMMAAttr(attr)) &&
+         "Expected MMAAttr or VirtualMMAAttr");
 
-  llvm::TypeSwitch<mlir::Attribute, void>(unwrap(attr))
+  return llvm::TypeSwitch<mlir::Attribute, ireeGPUMMAInfo>(unwrap(attr))
       .Case<mlir::iree_compiler::IREE::GPU::MMAAttr,
             mlir::iree_compiler::IREE::GPU::VirtualMMAAttr>(
-          [&](auto mma) { setMMAInfo(mma); })
-      .Default([&](mlir::Attribute) {
-        assert(false && "Expected MMAAttr or VirtualMMAAttr");
+          [](auto mma) -> ireeGPUMMAInfo {
```

**Comment:**
```suggestion
          [](auto mma) {
```

The return type appears in the typeswitch definition and in side the lamdba body already

---


---


## [PR #21398](https://github.com/iree-org/iree/pull/21398): Integrate LLVM to llvm/llvm-project@e0cce5c

### Review Summary

**APPROVED** (2025-07-17)



---


## [PR #21377](https://github.com/iree-org/iree/pull/21377): Integrate LLVM to llvm/llvm-project@bda5602

### Review Summary

**APPROVED** (2025-07-15)



---


## [PR #21364](https://github.com/iree-org/iree/pull/21364): Integrate LLVM to llvm/llvm-project@3ed3a33

### Review Summary

**COMMENTED** (2025-07-14)

Can you also mention how stablehlo and torch-mlir were updated (SHAs)?


### Code Comments

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td:1662`

```diff
@@ -1666,11 +1655,11 @@ def IREELinalgExt_WinogradFilterTransformOp : IREELinalgExt_Op<"winograd.filter_
                        DenseI64ArrayAttr:$kernel_dimensions
   );
 
-  let builders = [
-    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputs,
-      CArg<"int64_t", "8">:$output_tile_size, CArg<"int64_t", "3">:$kernel_size,
-      CArg<"ArrayRef<int64_t>", "{0, 1}">:$kernel_dimensions)>
-  ];
+  // let builders = [
+  //   OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputs,
+  //     CArg<"int64_t", "8">:$output_tile_size, CArg<"int64_t", "3">:$kernel_size,
+  //     CArg<"ArrayRef<int64_t>", "{0, 1}">:$kernel_dimensions)>
+  // ];
```

**Comment:**
Delete this instead of commenting out?

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td:1782`

```diff
@@ -1786,11 +1775,11 @@ def IREELinalgExt_WinogradOutputTransformOp : IREELinalgExt_Op<"winograd.output_
                        DenseI64ArrayAttr:$image_dimensions
   );
 
-  let builders = [
-    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputs,
-      CArg<"int64_t", "8">:$output_tile_size, CArg<"int64_t", "3">:$kernel_size,
-      CArg<"ArrayRef<int64_t>", "{1, 2}">:$image_dimensions)>
-  ];
+  // let builders = [
+  //   OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputs,
+  //     CArg<"int64_t", "8">:$output_tile_size, CArg<"int64_t", "3">:$kernel_size,
+  //     CArg<"ArrayRef<int64_t>", "{1, 2}">:$image_dimensions)>
+  // ];
```

**Comment:**
Also here

---


---


## [PR #21345](https://github.com/iree-org/iree/pull/21345): [Codegen][Tuner] remove decomposition attr for attention op

### Review Summary

**APPROVED** (2025-07-11)


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:86`

```diff
@@ -79,7 +79,11 @@ struct StripAttentionOpCompilationInfo final
                    attr.getName() !=
                        IREE::LinalgExt::AttentionOp::getPVAttrStr();
           }));
-      attentionOp.setDecompositionConfigAttr(newConfig);
+      if (!newConfig.empty()) {
+        attentionOp.setDecompositionConfigAttr(newConfig);
+      } else {
+        attentionOp.removeDecompositionConfigAttr();
+      }
```

**Comment:**
nit: flip this condition to avoid negation

---


---


## [PR #21216](https://github.com/iree-org/iree/pull/21216): [Codegen][Tuner] expose python binding isa_attention_op

### Review Summary

**COMMENTED** (2025-06-30)


**APPROVED** (2025-06-30)


### Code Comments

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:509`

```diff
@@ -501,4 +501,12 @@ NB_MODULE(_ireeCompilerDialects, m) {
       "Infers the structure of an attention operation from affine indexing "
       "maps.",
       py::arg("q"), py::arg("k"), py::arg("v"), py::arg("o"));
+
+  iree_codegen_module.def(
+      "isa_attention_op",
+      [](MlirOperation op) -> bool {
+        return ireeCodegenMlirOperationIsACodegenAttentionOp(op);
+      },
```

**Comment:**
Do we need the lambda here? If the types match, you shouldn't need a wrapper around `ireeCodegenMlirOperationIsACodegenAttentionOp`

---


---


## [PR #21170](https://github.com/iree-org/iree/pull/21170): [Codegen][Tuner] expose python binding for attention op details

### Review Summary

**COMMENTED** (2025-06-24)


**APPROVED** (2025-06-25)


### Code Comments

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_codegen.h:94`

```diff
@@ -78,6 +79,19 @@ MLIR_CAPI_EXPORTED void ireeCodegenQueryMMAIntrinsics(MlirOperation op,
                                                       size_t *numIntrinsics,
                                                       uint32_t *mmaIntrinsics);
 
+struct ireeCodegenAttentionOpDetail {
+  MlirAttribute batch;
+  MlirAttribute m;
+  MlirAttribute k1;
+  MlirAttribute k2;
+  MlirAttribute n;
+  int64_t rank;
+};
+
+MLIR_CAPI_EXPORTED ireeCodegenAttentionOpDetail
+ireeCodegenGetAttentionOpDetail(MlirAffineMap qMap, MlirAffineMap kMap,
+                                MlirAffineMap vMap, MlirAffineMap oMap);
+
```

**Comment:**
What about the rank?

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:25`

```diff
@@ -13,13 +13,16 @@
 #include "mlir-c/IR.h"
 #include "mlir/Bindings/Python/Nanobind.h"
 #include "mlir/Bindings/Python/NanobindAdaptors.h"
+#include "mlir/CAPI/IR.h"
+#include "mlir/IR/BuiltinAttributes.h"
 
 static const char *kCodegenModuleImportPath =
     MAKE_MLIR_PYTHON_QUALNAME("dialects.iree_codegen");
 static const char *kGpuModuleImportPath =
     MAKE_MLIR_PYTHON_QUALNAME("dialects.iree_gpu");
 
 namespace py = nanobind;
+using namespace mlir;
```

**Comment:**
Let's keep the use of mlir C++ apis explicit here

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:16`

```diff
@@ -13,6 +13,8 @@
 #include "mlir-c/IR.h"
 #include "mlir/Bindings/Python/Nanobind.h"
 #include "mlir/Bindings/Python/NanobindAdaptors.h"
+#include "mlir/CAPI/IR.h"
```

**Comment:**
What do we use this for?

---

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp:250`

```diff
@@ -220,3 +224,40 @@ void ireeCodegenGetTunerRootOps(MlirModule module, size_t *numOps,
     rootOps[i] = wrap(tunerRootOps[i]);
   }
 }
+
+ireeCodegenAttentionOpDetail
+ireeCodegenGetAttentionOpDetail(MlirAffineMap qMap, MlirAffineMap kMap,
+                                MlirAffineMap vMap, MlirAffineMap oMap) {
+  mlir::AffineMap QMap = unwrap(qMap);
+  mlir::AffineMap KMap = unwrap(kMap);
+  mlir::AffineMap VMap = unwrap(vMap);
+  mlir::AffineMap OMap = unwrap(oMap);
+
+  llvm::FailureOr<mlir::iree_compiler::IREE::LinalgExt::AttentionOpDetail>
+      maybeDetail =
+          mlir::iree_compiler::IREE::LinalgExt::AttentionOpDetail::get(
+              QMap, KMap, VMap, OMap);
+
+  if (failed(maybeDetail)) {
+    return ireeCodegenAttentionOpDetail{/*batch=*/wrap(mlir::Attribute()),
+                                        /*m=*/wrap(mlir::Attribute()),
+                                        /*k1=*/wrap(mlir::Attribute()),
+                                        /*k2=*/wrap(mlir::Attribute()),
+                                        /*n=*/wrap(mlir::Attribute()),
+                                        /*domainRank=*/-1};
+  }
+
+  const auto &opInfo = *maybeDetail;
```

**Comment:**
Don't use auto here, the type is not obvious without IDE

---


---


## [PR #21138](https://github.com/iree-org/iree/pull/21138): [LinalgExt] support converting argcompare to loops.

### Review Summary

**COMMENTED** (2025-06-20)


**COMMENTED** (2025-06-20)


**COMMENTED** (2025-06-20)


### Code Comments

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1365`

```diff
@@ -1340,6 +1340,88 @@ LogicalResult TopkOp::getResultTilePosition(
   return success();
 }
 
+//===----------------------------------------------------------------------===//
+// ArgCompareOp
+//===----------------------------------------------------------------------===//
+
+SmallVector<Range> ArgCompareOp::getIterationDomain(OpBuilder &builder) {
+  Location loc = getLoc();
+  OpFoldResult zero = builder.getIndexAttr(0);
+  OpFoldResult one = builder.getIndexAttr(1);
+  int64_t rank = getInputRank();
+  SmallVector<Range> ranges;
+  for (int64_t dim = 0; dim < rank; ++dim) {
+    OpFoldResult ub = getDim(builder, loc, getInputValue(), dim);
+    ranges.push_back(Range{zero, ub, one});
+  }
+  return ranges;
+}
+
+LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
+                                                         Location loc,
+                                                         ValueRange ivs) {
+  uint64_t reductionDim = getDimension();
+  SmallVector<Value> parallelIndices;
+  for (size_t i = 0; i < ivs.size(); ++i) {
```

**Comment:**
nit: do not recalculate the end index at each loop iteration

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1377`

```diff
@@ -1340,6 +1340,88 @@ LogicalResult TopkOp::getResultTilePosition(
   return success();
 }
 
+//===----------------------------------------------------------------------===//
+// ArgCompareOp
+//===----------------------------------------------------------------------===//
+
+SmallVector<Range> ArgCompareOp::getIterationDomain(OpBuilder &builder) {
+  Location loc = getLoc();
+  OpFoldResult zero = builder.getIndexAttr(0);
+  OpFoldResult one = builder.getIndexAttr(1);
+  int64_t rank = getInputRank();
+  SmallVector<Range> ranges;
+  for (int64_t dim = 0; dim < rank; ++dim) {
+    OpFoldResult ub = getDim(builder, loc, getInputValue(), dim);
+    ranges.push_back(Range{zero, ub, one});
+  }
+  return ranges;
+}
+
+LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
+                                                         Location loc,
+                                                         ValueRange ivs) {
+  uint64_t reductionDim = getDimension();
+  SmallVector<Value> parallelIndices;
+  for (size_t i = 0; i < ivs.size(); ++i) {
+    if (i == reductionDim)
+      continue;
+    parallelIndices.push_back(ivs[i]);
+  }
+
+  Value candidateValue = b.create<memref::LoadOp>(loc, getInputValue(), ivs);
+  Value indexValue = ivs[reductionDim];
+  if (getIndexBase()) {
+    indexValue = b.create<arith::AddIOp>(loc, getIndexBase(), indexValue);
+  }
+  Value castedIndex = indexValue;
+  auto indexType = getOutputIndexType().getElementType();
```

**Comment:**
```suggestion
  Type indexType = getOutputIndexType().getElementType();
```

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1406`

```diff
@@ -1340,6 +1340,88 @@ LogicalResult TopkOp::getResultTilePosition(
   return success();
 }
 
+//===----------------------------------------------------------------------===//
+// ArgCompareOp
+//===----------------------------------------------------------------------===//
+
+SmallVector<Range> ArgCompareOp::getIterationDomain(OpBuilder &builder) {
+  Location loc = getLoc();
+  OpFoldResult zero = builder.getIndexAttr(0);
+  OpFoldResult one = builder.getIndexAttr(1);
+  int64_t rank = getInputRank();
+  SmallVector<Range> ranges;
+  for (int64_t dim = 0; dim < rank; ++dim) {
+    OpFoldResult ub = getDim(builder, loc, getInputValue(), dim);
+    ranges.push_back(Range{zero, ub, one});
+  }
+  return ranges;
+}
+
+LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
+                                                         Location loc,
+                                                         ValueRange ivs) {
+  uint64_t reductionDim = getDimension();
+  SmallVector<Value> parallelIndices;
+  for (size_t i = 0; i < ivs.size(); ++i) {
+    if (i == reductionDim)
+      continue;
+    parallelIndices.push_back(ivs[i]);
+  }
+
+  Value candidateValue = b.create<memref::LoadOp>(loc, getInputValue(), ivs);
+  Value indexValue = ivs[reductionDim];
+  if (getIndexBase()) {
+    indexValue = b.create<arith::AddIOp>(loc, getIndexBase(), indexValue);
+  }
+  Value castedIndex = indexValue;
+  auto indexType = getOutputIndexType().getElementType();
+  if (castedIndex.getType() != indexType) {
+    castedIndex = b.create<arith::IndexCastOp>(loc, indexType, castedIndex);
+  }
+
+  Value isFirst =
+      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, ivs[reductionDim],
+                              b.create<arith::ConstantIndexOp>(loc, 0));
+  auto ifOp = b.create<scf::IfOp>(loc, isFirst, /*withElseRegion=*/true);
+  {
+    OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
+    thenBuilder.create<memref::StoreOp>(loc, candidateValue, outputValue(),
+                                        parallelIndices);
+    thenBuilder.create<memref::StoreOp>(loc, castedIndex, outputIndex(),
+                                        parallelIndices);
+  }
+
+  {
+    OpBuilder elseBuilder = ifOp.getElseBodyBuilder();
+
+    Value bestValueSoFar =
+        elseBuilder.create<memref::LoadOp>(loc, outputValue(), parallelIndices);
+    Value bestIndexSoFar =
+        elseBuilder.create<memref::LoadOp>(loc, outputIndex(), parallelIndices);
+
+    auto &srcBlock = getRegion().front();
+    IRMapping bvm;
+    bvm.map(srcBlock.getArgument(0), candidateValue);
+    bvm.map(srcBlock.getArgument(1), bestValueSoFar);
+    for (auto &op : srcBlock.without_terminator()) {
```

**Comment:**
nit: The types are not obvious in these two places without using an IDE, we shouldn't use `auto` here

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1403`

```diff
@@ -1340,6 +1340,88 @@ LogicalResult TopkOp::getResultTilePosition(
   return success();
 }
 
+//===----------------------------------------------------------------------===//
+// ArgCompareOp
+//===----------------------------------------------------------------------===//
+
+SmallVector<Range> ArgCompareOp::getIterationDomain(OpBuilder &builder) {
+  Location loc = getLoc();
+  OpFoldResult zero = builder.getIndexAttr(0);
+  OpFoldResult one = builder.getIndexAttr(1);
+  int64_t rank = getInputRank();
+  SmallVector<Range> ranges;
+  for (int64_t dim = 0; dim < rank; ++dim) {
+    OpFoldResult ub = getDim(builder, loc, getInputValue(), dim);
+    ranges.push_back(Range{zero, ub, one});
+  }
+  return ranges;
+}
+
+LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
+                                                         Location loc,
+                                                         ValueRange ivs) {
+  uint64_t reductionDim = getDimension();
+  SmallVector<Value> parallelIndices;
+  for (size_t i = 0; i < ivs.size(); ++i) {
+    if (i == reductionDim)
+      continue;
+    parallelIndices.push_back(ivs[i]);
+  }
+
+  Value candidateValue = b.create<memref::LoadOp>(loc, getInputValue(), ivs);
+  Value indexValue = ivs[reductionDim];
+  if (getIndexBase()) {
+    indexValue = b.create<arith::AddIOp>(loc, getIndexBase(), indexValue);
+  }
+  Value castedIndex = indexValue;
+  auto indexType = getOutputIndexType().getElementType();
+  if (castedIndex.getType() != indexType) {
+    castedIndex = b.create<arith::IndexCastOp>(loc, indexType, castedIndex);
+  }
+
+  Value isFirst =
+      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, ivs[reductionDim],
+                              b.create<arith::ConstantIndexOp>(loc, 0));
+  auto ifOp = b.create<scf::IfOp>(loc, isFirst, /*withElseRegion=*/true);
+  {
+    OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
+    thenBuilder.create<memref::StoreOp>(loc, candidateValue, outputValue(),
+                                        parallelIndices);
+    thenBuilder.create<memref::StoreOp>(loc, castedIndex, outputIndex(),
+                                        parallelIndices);
+  }
+
+  {
+    OpBuilder elseBuilder = ifOp.getElseBodyBuilder();
+
+    Value bestValueSoFar =
+        elseBuilder.create<memref::LoadOp>(loc, outputValue(), parallelIndices);
+    Value bestIndexSoFar =
+        elseBuilder.create<memref::LoadOp>(loc, outputIndex(), parallelIndices);
+
+    auto &srcBlock = getRegion().front();
+    IRMapping bvm;
```

**Comment:**
What is `bvm`? I don't know this TLA.

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1366`

```diff
@@ -1362,7 +1362,8 @@ LogicalResult ArgCompareOp::generateScalarImplementation(OpBuilder &b,
                                                          ValueRange ivs) {
   uint64_t reductionDim = getDimension();
   SmallVector<Value> parallelIndices;
-  for (size_t i = 0; i < ivs.size(); ++i) {
+  size_t rank = ivs.size();
+  for (size_t i = 0; i < rank; ++i) {
```

**Comment:**
FYI, you can also do:
```suggestion
  for (size_t i = 0, rank = ivs.size(); i < rank; ++i) {
```
if you don't use `rank` outside of the loop

---


---


## [PR #21106](https://github.com/iree-org/iree/pull/21106): [LinalgExt] fix arg_compare op with region and start index

### Review Summary

**COMMENTED** (2025-06-18)


**COMMENTED** (2025-06-19)


**APPROVED** (2025-06-19)


### Code Comments

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td:687`

```diff
@@ -668,30 +668,42 @@ def IREELinalgExt_TopkOp : IREELinalgExt_Op<"topk",[
   }];
 }
 
-def IREELinalgExt_ArgmaxOp : IREELinalgExt_Op<"argmax", [
+def IREELinalgExt_ArgCompareOp : IREELinalgExt_Op<"arg_compare", [
+  SingleBlockImplicitTerminator<"::mlir::iree_compiler::IREE::LinalgExt::YieldOp">,
   DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
 ]> {
-  let summary = "Argmax reduction op.";
+  let summary = "Performs an arg-reduction using a user-defined comparator.";
   let description = [{
-    An argmax op that reduces along a given dimension, returning the max value and its index.
+    The `arg_compare` op performs a reduction over a given dimension of a tensor,
+    returning both the selected value and its corresponding index. The selection
+    logic is defined by a user-specified comparator region.
+
+    The comparator region receives two candidate values and returns a single `i1`
+    result indicating whether the first argument should be preferred over the second.
+
+    This region defines the sorting rule, e.g., "greater than" for argmax or
+    "less than" for argmin. It allows for generalization beyond simple argmax-style
+    behavior.
   }];
```

**Comment:**
It would be nice to add an example of the op here, I rely on these heavily when writing IR by hand. I think argmax specifically would make a nice sample.

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td:700`

```diff
@@ -684,6 +684,39 @@ def IREELinalgExt_ArgCompareOp : IREELinalgExt_Op<"arg_compare", [
     This region defines the sorting rule, e.g., "greater than" for argmax or
     "less than" for argmin. It allows for generalization beyond simple argmax-style
     behavior.
+
+    Example (argmax over dim 1):
+
+    %input = memref<2x10xf32>
+    %out_val = memref<2xf32>
+    %out_idx = memref<2xi32>
+    iree_linalg_ext.arg_compare
+      dimension(1)
+      ins(%input : memref<2x10xf32>)
+      outs(%out_val, %out_idx : memref<2xf32>, memref<2xi32>) {
+    ^bb0(%a: f32, %b: f32):
+      %cmp = arith.cmpf ogt, %a, %b : f32
+      iree_linalg_ext.yield %cmp : i1
+    }
```

**Comment:**
Please put this in code blocks so that it renders nicely on the website

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td:716`

```diff
@@ -684,6 +684,39 @@ def IREELinalgExt_ArgCompareOp : IREELinalgExt_Op<"arg_compare", [
     This region defines the sorting rule, e.g., "greater than" for argmax or
     "less than" for argmin. It allows for generalization beyond simple argmax-style
     behavior.
+
+    Example (argmax over dim 1):
+
+    %input = memref<2x10xf32>
+    %out_val = memref<2xf32>
+    %out_idx = memref<2xi32>
+    iree_linalg_ext.arg_compare
+      dimension(1)
+      ins(%input : memref<2x10xf32>)
+      outs(%out_val, %out_idx : memref<2xf32>, memref<2xi32>) {
+    ^bb0(%a: f32, %b: f32):
+      %cmp = arith.cmpf ogt, %a, %b : f32
+      iree_linalg_ext.yield %cmp : i1
+    }
+
+  Example with index_base = 5 (i.e., indices start counting from 5):
+
+    %input = memref<2x10xf32>
+    %out_val = memref<2xf32>
+    %out_idx = memref<2xi32>
+    %base = arith.constant 5 : index
+    iree_linalg_ext.arg_compare
+      dimension(1)
+      ins(%input : memref<2x10xf32>)
+      outs(%out_val, %out_idx : memref<2xf32>, memref<2xi32>)
+      index_base(%base : index) {
+    ^bb0(%a: f32, %b: f32):
+      %cmp = arith.cmpf ogt, %a, %b : f32
+      iree_linalg_ext.yield %cmp : i1
+    }
```

**Comment:**
Also here

---


---


## [PR #21077](https://github.com/iree-org/iree/pull/21077): [LinalgExt] add TilingInterface support for ArgCompareOp

### Review Summary

**COMMENTED** (2025-06-12)


**COMMENTED** (2025-06-13)


### Code Comments

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1352`

```diff
@@ -1340,6 +1340,116 @@ LogicalResult TopkOp::getResultTilePosition(
   return success();
 }
 
+//===----------------------------------------------------------------------===//
+// ArgmaxOp
+//===----------------------------------------------------------------------===//
+
+SmallVector<Range> ArgmaxOp::getIterationDomain(OpBuilder &builder) {
+  Location loc = getLoc();
+  OpFoldResult zero = builder.getIndexAttr(0);
+  OpFoldResult one = builder.getIndexAttr(1);
+  SmallVector<Range> ranges;
+  for (auto dim : llvm::seq<int64_t>(0, getInputRank())) {
```

**Comment:**
nit: I think a plain loop would be fine as well. If you keep as-is, I think there's an unary version of `seq` that would be more concise.

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1392`

```diff
@@ -1340,6 +1340,116 @@ LogicalResult TopkOp::getResultTilePosition(
   return success();
 }
 
+//===----------------------------------------------------------------------===//
+// ArgmaxOp
+//===----------------------------------------------------------------------===//
+
+SmallVector<Range> ArgmaxOp::getIterationDomain(OpBuilder &builder) {
+  Location loc = getLoc();
+  OpFoldResult zero = builder.getIndexAttr(0);
+  OpFoldResult one = builder.getIndexAttr(1);
+  SmallVector<Range> ranges;
+  for (auto dim : llvm::seq<int64_t>(0, getInputRank())) {
+    OpFoldResult ub = getDim(builder, loc, getInputValue(), dim);
+    ranges.push_back(Range{zero, ub, one});
+  }
+  return ranges;
+}
+
+SmallVector<utils::IteratorType> ArgmaxOp::getLoopIteratorTypes() {
+  SmallVector<utils::IteratorType> iteratorTypes(getInputRank(),
+                                                 utils::IteratorType::parallel);
+  iteratorTypes[getDimension()] = utils::IteratorType::reduction;
+  return iteratorTypes;
+}
+
+FailureOr<TilingResult>
+ArgmaxOp::getTiledImplementation(OpBuilder &builder,
+                                 ArrayRef<OpFoldResult> offsets,
+                                 ArrayRef<OpFoldResult> sizes) {
+  Location loc = getLoc();
+  int64_t rank = getInputRank();
+  assert(offsets.size() == static_cast<size_t>(rank) &&
+         sizes.size() == static_cast<size_t>(rank));
+
+  SmallVector<Operation *> slices;
+  SmallVector<Value> tiledOperands;
+
+  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
+  Operation *inputSlice =
+      getSlice(builder, loc, getInputValue(), offsets, sizes, strides);
+
+  if (!inputSlice)
+    return emitOpError("failed to slice input");
+  tiledOperands.push_back(inputSlice->getResult(0));
+  slices.push_back(inputSlice);
+
+  SmallVector<OpFoldResult> outputOffsets, outputSizes;
+  if (failed(getResultTilePosition(builder, 0, offsets, sizes, outputOffsets,
+                                   outputSizes))) {
+    return emitOpError("failed to compute output tile position");
```

**Comment:**
IREE requires braces around bodies of single-statements `if`s/loops

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1373`

```diff
@@ -1340,6 +1340,116 @@ LogicalResult TopkOp::getResultTilePosition(
   return success();
 }
 
+//===----------------------------------------------------------------------===//
+// ArgmaxOp
+//===----------------------------------------------------------------------===//
+
+SmallVector<Range> ArgmaxOp::getIterationDomain(OpBuilder &builder) {
+  Location loc = getLoc();
+  OpFoldResult zero = builder.getIndexAttr(0);
+  OpFoldResult one = builder.getIndexAttr(1);
+  SmallVector<Range> ranges;
+  for (auto dim : llvm::seq<int64_t>(0, getInputRank())) {
+    OpFoldResult ub = getDim(builder, loc, getInputValue(), dim);
+    ranges.push_back(Range{zero, ub, one});
+  }
+  return ranges;
+}
+
+SmallVector<utils::IteratorType> ArgmaxOp::getLoopIteratorTypes() {
+  SmallVector<utils::IteratorType> iteratorTypes(getInputRank(),
+                                                 utils::IteratorType::parallel);
+  iteratorTypes[getDimension()] = utils::IteratorType::reduction;
+  return iteratorTypes;
+}
+
+FailureOr<TilingResult>
+ArgmaxOp::getTiledImplementation(OpBuilder &builder,
+                                 ArrayRef<OpFoldResult> offsets,
+                                 ArrayRef<OpFoldResult> sizes) {
+  Location loc = getLoc();
+  int64_t rank = getInputRank();
+  assert(offsets.size() == static_cast<size_t>(rank) &&
+         sizes.size() == static_cast<size_t>(rank));
```

**Comment:**
Can you break this down into two separate assertions? This way we will know which one failed without having to start a debugger or recompile the source.

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1392`

```diff
@@ -1340,6 +1340,116 @@ LogicalResult TopkOp::getResultTilePosition(
   return success();
 }
 
+//===----------------------------------------------------------------------===//
+// ArgmaxOp
+//===----------------------------------------------------------------------===//
+
+SmallVector<Range> ArgmaxOp::getIterationDomain(OpBuilder &builder) {
+  Location loc = getLoc();
+  OpFoldResult zero = builder.getIndexAttr(0);
+  OpFoldResult one = builder.getIndexAttr(1);
+  SmallVector<Range> ranges;
+  for (auto dim : llvm::seq<int64_t>(0, getInputRank())) {
+    OpFoldResult ub = getDim(builder, loc, getInputValue(), dim);
+    ranges.push_back(Range{zero, ub, one});
+  }
+  return ranges;
+}
+
+SmallVector<utils::IteratorType> ArgmaxOp::getLoopIteratorTypes() {
+  SmallVector<utils::IteratorType> iteratorTypes(getInputRank(),
+                                                 utils::IteratorType::parallel);
+  iteratorTypes[getDimension()] = utils::IteratorType::reduction;
+  return iteratorTypes;
+}
+
+FailureOr<TilingResult>
+ArgmaxOp::getTiledImplementation(OpBuilder &builder,
+                                 ArrayRef<OpFoldResult> offsets,
+                                 ArrayRef<OpFoldResult> sizes) {
+  Location loc = getLoc();
+  int64_t rank = getInputRank();
+  assert(offsets.size() == static_cast<size_t>(rank) &&
+         sizes.size() == static_cast<size_t>(rank));
+
+  SmallVector<Operation *> slices;
+  SmallVector<Value> tiledOperands;
+
+  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
+  Operation *inputSlice =
+      getSlice(builder, loc, getInputValue(), offsets, sizes, strides);
+
+  if (!inputSlice)
+    return emitOpError("failed to slice input");
+  tiledOperands.push_back(inputSlice->getResult(0));
+  slices.push_back(inputSlice);
+
+  SmallVector<OpFoldResult> outputOffsets, outputSizes;
+  if (failed(getResultTilePosition(builder, 0, offsets, sizes, outputOffsets,
+                                   outputSizes))) {
+    return emitOpError("failed to compute output tile position");
```

**Comment:**
Also elsewhere below

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp:1352`

```diff
@@ -1340,6 +1340,118 @@ LogicalResult TopkOp::getResultTilePosition(
   return success();
 }
 
+//===----------------------------------------------------------------------===//
+// ArgmaxOp
+//===----------------------------------------------------------------------===//
+
+SmallVector<Range> ArgmaxOp::getIterationDomain(OpBuilder &builder) {
+  Location loc = getLoc();
+  OpFoldResult zero = builder.getIndexAttr(0);
+  OpFoldResult one = builder.getIndexAttr(1);
+  SmallVector<Range> ranges;
+  for (int64_t dim = 0; dim < getInputRank(); ++dim) {
```

**Comment:**
Do not recalculate the trip count: https://llvm.org/docs/CodingStandards.html#don-t-evaluate-end-every-time-through-a-loop

---


---


## [PR #21021](https://github.com/iree-org/iree/pull/21021): [LinalgExt] Add argmax op with rountrip and invalid mlir test

### Review Summary

**COMMENTED** (2025-06-06)


**COMMENTED** (2025-06-06)


**COMMENTED** (2025-06-06)


**APPROVED** (2025-06-06)

Looks OK % nits.

I guess the next step is to implement the interfaces mentioned by @hanhanW?


### Code Comments

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp:924`

```diff
@@ -908,6 +908,62 @@ TopkOp::reifyResultShapes(OpBuilder &b,
       .reifyResultShapes(b, reifiedReturnShapes);
 }
 
+//===----------------------------------------------------------------------===//
+// ArgmaxOp
+//===----------------------------------------------------------------------===//
+
+LogicalResult ArgmaxOp::verify() {
+  Operation *op = getOperation();
+
+  // Check number of inputs and outputs.
+  if (getNumDpsInputs() != 1) {
+    return op->emitOpError("expected exactly one input operand (values)");
+  }
+
+  if (getNumDpsInits() != 2) {
+    return op->emitOpError("expected two output operands (value and index)");
```

**Comment:**
It would be nice to also print the actual number (in addition to the expected one)

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp:929`

```diff
@@ -908,6 +908,62 @@ TopkOp::reifyResultShapes(OpBuilder &b,
       .reifyResultShapes(b, reifiedReturnShapes);
 }
 
+//===----------------------------------------------------------------------===//
+// ArgmaxOp
+//===----------------------------------------------------------------------===//
+
+LogicalResult ArgmaxOp::verify() {
+  Operation *op = getOperation();
+
+  // Check number of inputs and outputs.
+  if (getNumDpsInputs() != 1) {
+    return op->emitOpError("expected exactly one input operand (values)");
+  }
+
+  if (getNumDpsInits() != 2) {
+    return op->emitOpError("expected two output operands (value and index)");
+  }
+
+  uint64_t dim = getDimension();
+  if (dim >= getInputRank()) {
+    return op->emitOpError("reduction dimension exceeds input rank");
```

**Comment:**
Also here: print what the wrong value is

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp:934`

```diff
@@ -908,6 +908,62 @@ TopkOp::reifyResultShapes(OpBuilder &b,
       .reifyResultShapes(b, reifiedReturnShapes);
 }
 
+//===----------------------------------------------------------------------===//
+// ArgmaxOp
+//===----------------------------------------------------------------------===//
+
+LogicalResult ArgmaxOp::verify() {
+  Operation *op = getOperation();
+
+  // Check number of inputs and outputs.
+  if (getNumDpsInputs() != 1) {
+    return op->emitOpError("expected exactly one input operand (values)");
+  }
+
+  if (getNumDpsInits() != 2) {
+    return op->emitOpError("expected two output operands (value and index)");
+  }
+
+  uint64_t dim = getDimension();
+  if (dim >= getInputRank()) {
+    return op->emitOpError("reduction dimension exceeds input rank");
+  }
+
+  ShapedType inputType = getInputType();
+  ShapedType outputValueType = cast<ShapedType>(outputValue().getType());
+  ShapedType outputIndexType = cast<ShapedType>(outputIndex().getType());
```

**Comment:**
```suggestion
  auto outputValueType = cast<ShapedType>(outputValue().getType());
  auto outputIndexType = cast<ShapedType>(outputIndex().getType());
```
See https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp:936`

```diff
@@ -908,6 +908,62 @@ TopkOp::reifyResultShapes(OpBuilder &b,
       .reifyResultShapes(b, reifiedReturnShapes);
 }
 
+//===----------------------------------------------------------------------===//
+// ArgmaxOp
+//===----------------------------------------------------------------------===//
+
+LogicalResult ArgmaxOp::verify() {
+  Operation *op = getOperation();
+
+  // Check number of inputs and outputs.
+  if (getNumDpsInputs() != 1) {
+    return op->emitOpError("expected exactly one input operand (values)");
+  }
+
+  if (getNumDpsInits() != 2) {
+    return op->emitOpError("expected two output operands (value and index)");
+  }
+
+  uint64_t dim = getDimension();
+  if (dim >= getInputRank()) {
+    return op->emitOpError("reduction dimension exceeds input rank");
+  }
+
+  ShapedType inputType = getInputType();
+  ShapedType outputValueType = cast<ShapedType>(outputValue().getType());
+  ShapedType outputIndexType = cast<ShapedType>(outputIndex().getType());
+
+  // Element type compatibility.
```

**Comment:**
I don't think this comment adds much value -- it's pretty clear what is being check. In general, focus on the *why* when writing the comments, not on what the code does.

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp:941`

```diff
@@ -908,6 +908,62 @@ TopkOp::reifyResultShapes(OpBuilder &b,
       .reifyResultShapes(b, reifiedReturnShapes);
 }
 
+//===----------------------------------------------------------------------===//
+// ArgmaxOp
+//===----------------------------------------------------------------------===//
+
+LogicalResult ArgmaxOp::verify() {
+  Operation *op = getOperation();
+
+  // Check number of inputs and outputs.
+  if (getNumDpsInputs() != 1) {
+    return op->emitOpError("expected exactly one input operand (values)");
+  }
+
+  if (getNumDpsInits() != 2) {
+    return op->emitOpError("expected two output operands (value and index)");
+  }
+
+  uint64_t dim = getDimension();
+  if (dim >= getInputRank()) {
+    return op->emitOpError("reduction dimension exceeds input rank");
+  }
+
+  ShapedType inputType = getInputType();
+  ShapedType outputValueType = cast<ShapedType>(outputValue().getType());
+  ShapedType outputIndexType = cast<ShapedType>(outputIndex().getType());
+
+  // Element type compatibility.
+  if (inputType.getElementType() != outputValueType.getElementType()) {
+    return op->emitOpError("input and output value element types must match");
+  }
+
+  // Output indicies and values must have the same shape.
```

**Comment:**
Same here...

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp:952`

```diff
@@ -908,6 +908,62 @@ TopkOp::reifyResultShapes(OpBuilder &b,
       .reifyResultShapes(b, reifiedReturnShapes);
 }
 
+//===----------------------------------------------------------------------===//
+// ArgmaxOp
+//===----------------------------------------------------------------------===//
+
+LogicalResult ArgmaxOp::verify() {
+  Operation *op = getOperation();
+
+  // Check number of inputs and outputs.
+  if (getNumDpsInputs() != 1) {
+    return op->emitOpError("expected exactly one input operand (values)");
+  }
+
+  if (getNumDpsInits() != 2) {
+    return op->emitOpError("expected two output operands (value and index)");
+  }
+
+  uint64_t dim = getDimension();
+  if (dim >= getInputRank()) {
+    return op->emitOpError("reduction dimension exceeds input rank");
+  }
+
+  ShapedType inputType = getInputType();
+  ShapedType outputValueType = cast<ShapedType>(outputValue().getType());
+  ShapedType outputIndexType = cast<ShapedType>(outputIndex().getType());
+
+  // Element type compatibility.
+  if (inputType.getElementType() != outputValueType.getElementType()) {
+    return op->emitOpError("input and output value element types must match");
+  }
+
+  // Output indicies and values must have the same shape.
+  if (failed(verifyCompatibleShape(outputValueType, outputIndexType))) {
+    return op->emitOpError("output indices/values shape must match");
+  }
+
+  // Expected output shape = input shape with `dim` removed.
+  SmallVector<int64_t> expectedShape;
+  for (int64_t i = 0; i < getInputRank(); ++i) {
+    if (static_cast<uint64_t>(i) != dim)
+      expectedShape.push_back(inputType.getDimSize(i));
+  }
+  if (!llvm::equal(expectedShape, outputValueType.getShape())) {
```

**Comment:**
nit: I think `==` would work here?

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp:954`

```diff
@@ -908,6 +908,62 @@ TopkOp::reifyResultShapes(OpBuilder &b,
       .reifyResultShapes(b, reifiedReturnShapes);
 }
 
+//===----------------------------------------------------------------------===//
+// ArgmaxOp
+//===----------------------------------------------------------------------===//
+
+LogicalResult ArgmaxOp::verify() {
+  Operation *op = getOperation();
+
+  // Check number of inputs and outputs.
+  if (getNumDpsInputs() != 1) {
+    return op->emitOpError("expected exactly one input operand (values)");
+  }
+
+  if (getNumDpsInits() != 2) {
+    return op->emitOpError("expected two output operands (value and index)");
+  }
+
+  uint64_t dim = getDimension();
+  if (dim >= getInputRank()) {
+    return op->emitOpError("reduction dimension exceeds input rank");
+  }
+
+  ShapedType inputType = getInputType();
+  ShapedType outputValueType = cast<ShapedType>(outputValue().getType());
+  ShapedType outputIndexType = cast<ShapedType>(outputIndex().getType());
+
+  // Element type compatibility.
+  if (inputType.getElementType() != outputValueType.getElementType()) {
+    return op->emitOpError("input and output value element types must match");
+  }
+
+  // Output indicies and values must have the same shape.
+  if (failed(verifyCompatibleShape(outputValueType, outputIndexType))) {
+    return op->emitOpError("output indices/values shape must match");
+  }
+
+  // Expected output shape = input shape with `dim` removed.
+  SmallVector<int64_t> expectedShape;
+  for (int64_t i = 0; i < getInputRank(); ++i) {
+    if (static_cast<uint64_t>(i) != dim)
+      expectedShape.push_back(inputType.getDimSize(i));
+  }
+  if (!llvm::equal(expectedShape, outputValueType.getShape())) {
+    return op->emitOpError(
+        "output shape must match input shape with reduction dimension removed");
```

**Comment:**
I think it would help to print the expected and the actual shape

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td:710`

```diff
@@ -668,6 +668,60 @@ def IREELinalgExt_TopkOp : IREELinalgExt_Op<"topk",[
   }];
 }
 
+def IREELinalgExt_ArgmaxOp : IREELinalgExt_Op<"argmax", [
+  DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
+]> {
+  let summary = "Argmax reduction op.";
+  let description = [{
+    An argmax op that reduces along a given dimension, returning the max value and its index.
+  }];
+
+  let arguments = (ins
+    Variadic<AnyShaped>:$inputs,
+    Variadic<AnyShaped>:$outputs,
+    I64Attr:$dimension
+  );
+
+  let results = (outs
+    Variadic<AnyRankedTensor>:$results
+  );
+
+let assemblyFormat = [{
+    attr-dict
+    `dimension` `(` $dimension `)`
+    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
+    `outs` `(` $outputs `:` type($outputs) `)`
+    (`:` type($results)^)?
+  }];
+
+  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
+    Value getInputValue() {
+      return getDpsInputOperand(0)->get();
+    }
+
+    Value outputValue() {
+      return getDpsInitOperand(0)->get();
+    }
+
+    Value outputIndex() {
+      return getDpsInitOperand(1)->get();
+    }
+
+    ShapedType getInputType() {
```

**Comment:**
Do we also want `getOutput*Type` for index and value?

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp:951`

```diff
@@ -915,43 +915,65 @@ TopkOp::reifyResultShapes(OpBuilder &b,
 LogicalResult ArgmaxOp::verify() {
   Operation *op = getOperation();
 
-  // Check number of inputs and outputs.
   if (getNumDpsInputs() != 1) {
-    return op->emitOpError("expected exactly one input operand (values)");
+    return op->emitOpError(
+               "expected exactly one input operand (values), but got ")
+           << getNumDpsInputs();
   }
 
   if (getNumDpsInits() != 2) {
-    return op->emitOpError("expected two output operands (value and index)");
+    return op->emitOpError(
+               "expected two output operands (value and index), but got ")
+           << getNumDpsInits();
   }
 
   uint64_t dim = getDimension();
-  if (dim >= getInputRank()) {
-    return op->emitOpError("reduction dimension exceeds input rank");
+  int64_t rank = getInputRank();
+  if (dim >= rank) {
+    return op->emitOpError("reduction dimension exceeds or equals input rank. ")
+           << "got dimension: " << dim << ", but input rank is: " << rank;
   }
 
   ShapedType inputType = getInputType();
-  ShapedType outputValueType = cast<ShapedType>(outputValue().getType());
-  ShapedType outputIndexType = cast<ShapedType>(outputIndex().getType());
+  auto outputValueType = getOutputValueType();
+  auto outputIndexType = getOutputIndexType();
 
-  // Element type compatibility.
   if (inputType.getElementType() != outputValueType.getElementType()) {
-    return op->emitOpError("input and output value element types must match");
+    return op->emitOpError("input and output value element types must match. ")
+           << "Input type: " << inputType.getElementType()
+           << ", output value type: " << outputValueType.getElementType();
   }
 
-  // Output indicies and values must have the same shape.
   if (failed(verifyCompatibleShape(outputValueType, outputIndexType))) {
-    return op->emitOpError("output indices/values shape must match");
+    return op->emitOpError("output indices/values shape must match. ")
+           << "Output value shape: ["
+           << llvm::join(map_range(outputValueType.getShape(),
+                                   [](int64_t d) { return std::to_string(d); }),
```

**Comment:**
use `llvm::interleaved_array`

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp:955`

```diff
@@ -915,43 +915,65 @@ TopkOp::reifyResultShapes(OpBuilder &b,
 LogicalResult ArgmaxOp::verify() {
   Operation *op = getOperation();
 
-  // Check number of inputs and outputs.
   if (getNumDpsInputs() != 1) {
-    return op->emitOpError("expected exactly one input operand (values)");
+    return op->emitOpError(
+               "expected exactly one input operand (values), but got ")
+           << getNumDpsInputs();
   }
 
   if (getNumDpsInits() != 2) {
-    return op->emitOpError("expected two output operands (value and index)");
+    return op->emitOpError(
+               "expected two output operands (value and index), but got ")
+           << getNumDpsInits();
   }
 
   uint64_t dim = getDimension();
-  if (dim >= getInputRank()) {
-    return op->emitOpError("reduction dimension exceeds input rank");
+  int64_t rank = getInputRank();
+  if (dim >= rank) {
+    return op->emitOpError("reduction dimension exceeds or equals input rank. ")
+           << "got dimension: " << dim << ", but input rank is: " << rank;
   }
 
   ShapedType inputType = getInputType();
-  ShapedType outputValueType = cast<ShapedType>(outputValue().getType());
-  ShapedType outputIndexType = cast<ShapedType>(outputIndex().getType());
+  auto outputValueType = getOutputValueType();
+  auto outputIndexType = getOutputIndexType();
 
-  // Element type compatibility.
   if (inputType.getElementType() != outputValueType.getElementType()) {
-    return op->emitOpError("input and output value element types must match");
+    return op->emitOpError("input and output value element types must match. ")
+           << "Input type: " << inputType.getElementType()
+           << ", output value type: " << outputValueType.getElementType();
   }
 
-  // Output indicies and values must have the same shape.
   if (failed(verifyCompatibleShape(outputValueType, outputIndexType))) {
-    return op->emitOpError("output indices/values shape must match");
+    return op->emitOpError("output indices/values shape must match. ")
+           << "Output value shape: ["
+           << llvm::join(map_range(outputValueType.getShape(),
+                                   [](int64_t d) { return std::to_string(d); }),
+                         ", ")
+           << "]" << ", output index shape: ["
+           << llvm::join(map_range(outputIndexType.getShape(),
+                                   [](int64_t d) { return std::to_string(d); }),
```

**Comment:**
also here

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp:974`

```diff
@@ -915,43 +915,65 @@ TopkOp::reifyResultShapes(OpBuilder &b,
 LogicalResult ArgmaxOp::verify() {
   Operation *op = getOperation();
 
-  // Check number of inputs and outputs.
   if (getNumDpsInputs() != 1) {
-    return op->emitOpError("expected exactly one input operand (values)");
+    return op->emitOpError(
+               "expected exactly one input operand (values), but got ")
+           << getNumDpsInputs();
   }
 
   if (getNumDpsInits() != 2) {
-    return op->emitOpError("expected two output operands (value and index)");
+    return op->emitOpError(
+               "expected two output operands (value and index), but got ")
+           << getNumDpsInits();
   }
 
   uint64_t dim = getDimension();
-  if (dim >= getInputRank()) {
-    return op->emitOpError("reduction dimension exceeds input rank");
+  int64_t rank = getInputRank();
+  if (dim >= rank) {
+    return op->emitOpError("reduction dimension exceeds or equals input rank. ")
+           << "got dimension: " << dim << ", but input rank is: " << rank;
   }
 
   ShapedType inputType = getInputType();
-  ShapedType outputValueType = cast<ShapedType>(outputValue().getType());
-  ShapedType outputIndexType = cast<ShapedType>(outputIndex().getType());
+  auto outputValueType = getOutputValueType();
+  auto outputIndexType = getOutputIndexType();
 
-  // Element type compatibility.
   if (inputType.getElementType() != outputValueType.getElementType()) {
-    return op->emitOpError("input and output value element types must match");
+    return op->emitOpError("input and output value element types must match. ")
+           << "Input type: " << inputType.getElementType()
+           << ", output value type: " << outputValueType.getElementType();
   }
 
-  // Output indicies and values must have the same shape.
   if (failed(verifyCompatibleShape(outputValueType, outputIndexType))) {
-    return op->emitOpError("output indices/values shape must match");
+    return op->emitOpError("output indices/values shape must match. ")
+           << "Output value shape: ["
+           << llvm::join(map_range(outputValueType.getShape(),
+                                   [](int64_t d) { return std::to_string(d); }),
+                         ", ")
+           << "]" << ", output index shape: ["
+           << llvm::join(map_range(outputIndexType.getShape(),
+                                   [](int64_t d) { return std::to_string(d); }),
+                         ", ")
+           << "]";
   }
 
-  // Expected output shape = input shape with `dim` removed.
   SmallVector<int64_t> expectedShape;
   for (int64_t i = 0; i < getInputRank(); ++i) {
-    if (static_cast<uint64_t>(i) != dim)
+    if (i != dim)
       expectedShape.push_back(inputType.getDimSize(i));
   }
   if (!llvm::equal(expectedShape, outputValueType.getShape())) {
-    return op->emitOpError(
-        "output shape must match input shape with reduction dimension removed");
+    return op->emitOpError("output shape must match input shape with reduction "
+                           "dimension removed. ")
+           << "Expected: ["
+           << llvm::join(map_range(expectedShape,
+                                   [](int64_t d) { return std::to_string(d); }),
+                         ", ")
+           << "]" << ", but got: ["
+           << llvm::join(map_range(outputValueType.getShape(),
+                                   [](int64_t d) { return std::to_string(d); }),
```

**Comment:**
Also here

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp:953`

```diff
@@ -946,15 +947,10 @@ LogicalResult ArgmaxOp::verify() {
 
   if (failed(verifyCompatibleShape(outputValueType, outputIndexType))) {
     return op->emitOpError("output indices/values shape must match. ")
-           << "Output value shape: ["
-           << llvm::join(map_range(outputValueType.getShape(),
-                                   [](int64_t d) { return std::to_string(d); }),
-                         ", ")
-           << "]" << ", output index shape: ["
-           << llvm::join(map_range(outputIndexType.getShape(),
-                                   [](int64_t d) { return std::to_string(d); }),
-                         ", ")
-           << "]";
+           << "Output value shape: "
+           << llvm::interleaved_array(outputValueType.getShape(), ", ")
+           << ", output index shape: "
+           << llvm::interleaved_array(outputIndexType.getShape(), ", ");
```

**Comment:**
```suggestion
           << llvm::interleaved_array(outputValueType.getShape())
           << ", output index shape: "
           << llvm::interleaved_array(outputIndexType.getShape());
```

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp:966`

```diff
@@ -965,15 +961,9 @@ LogicalResult ArgmaxOp::verify() {
   if (!llvm::equal(expectedShape, outputValueType.getShape())) {
     return op->emitOpError("output shape must match input shape with reduction "
                            "dimension removed. ")
-           << "Expected: ["
-           << llvm::join(map_range(expectedShape,
-                                   [](int64_t d) { return std::to_string(d); }),
-                         ", ")
-           << "]" << ", but got: ["
-           << llvm::join(map_range(outputValueType.getShape(),
-                                   [](int64_t d) { return std::to_string(d); }),
-                         ", ")
-           << "]";
+           << "Expected: " << llvm::interleaved_array(expectedShape, ", ")
+           << ", but got: "
+           << llvm::interleaved_array(outputValueType.getShape(), ", ");
```

**Comment:**
```suggestion
           << "Expected: " << llvm::interleaved_array(expectedShape)
           << ", but got: "
           << llvm::interleaved_array(outputValueType.getShape());
```

---


---


## [PR #20906](https://github.com/iree-org/iree/pull/20906): [Codegen] split-k on argmax to ensure ukernel support

### Review Summary

**COMMENTED** (2025-06-04)


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/test/gpu_lower_to_ukernels.mlir:147`

```diff
@@ -119,6 +119,48 @@ func.func @argmax_f32i64_without_selected_ukernel(%arg0 : tensor<1x?xf32>) -> te
 
 // -----
 
+func.func @argmax_invalid_index_without_selected_ukernel(
+    %input: tensor<131072xf32>,
+    %init_val_arg: tensor<f32>,
+    %init_idx_arg: tensor<i64>
+) -> tensor<i64> {
+  %c0_i64 = arith.constant 0 : i64
+  %cst_min = arith.constant 0xFF800000 : f32  // -inf
+  %init_val = linalg.fill ins(%cst_min : f32)
+              outs(%init_val_arg : tensor<f32>) -> tensor<f32>
+  %init_idx = linalg.fill ins(%c0_i64 : i64)
+              outs(%init_idx_arg : tensor<i64>) -> tensor<i64>
+
+  // Argmax-style reduction with a matcher-breaking intermediate op (`addi`).
+  %result:2 = linalg.generic {
+      indexing_maps = [
+        affine_map<(d0) -> (d0)>,
+        affine_map<(d0) -> ()>,
+        affine_map<(d0) -> ()>
+      ],
+      iterator_types = ["reduction"]
+    } ins(%input : tensor<131072xf32>)
+      outs(%init_val, %init_idx : tensor<f32>, tensor<i64>) {
+    ^bb0(%in: f32, %val: f32, %idx: i64):
+      %i = linalg.index 0 : index
+      %cast = arith.index_cast %i : index to i64
+      // Breaks isArgmaxOp matching
```

**Comment:**
```suggestion
      // Breaks isArgmaxOp matching.
```

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp:644`

```diff
@@ -641,39 +641,30 @@ splitArgmaxReduction(RewriterBase &rewriter, linalg::GenericOp genericOp,
     }
   }
 
-  // Create partial linalg.generic op with global index computation.
-  Value tileSize = rewriter.create<arith::ConstantIndexOp>(loc, ratio);
-  auto partialOp = rewriter.create<linalg::GenericOp>(
+  // Step 1: Create a pure argmax to partially reduce the split dimension. The
```

**Comment:**
What do you mean by 'pure argmax'?

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp:652`

```diff
@@ -641,39 +641,30 @@ splitArgmaxReduction(RewriterBase &rewriter, linalg::GenericOp genericOp,
     }
   }
 
-  // Create partial linalg.generic op with global index computation.
-  Value tileSize = rewriter.create<arith::ConstantIndexOp>(loc, ratio);
-  auto partialOp = rewriter.create<linalg::GenericOp>(
+  // Step 1: Create a pure argmax to partially reduce the split dimension. The
+  // result will contain local indices within each reduction group, which need
+  // to be adjusted to the global index later.
+  auto partialArgmax = rewriter.create<linalg::GenericOp>(
       loc, TypeRange{identityValue.getType(), identityIndex.getType()},
       newInputs, ValueRange{identityValue, identityIndex}, newMaps,
-      newIteratorTypes);
-
-  rewriter.inlineRegionBefore(genericOp.getRegion(), partialOp.getRegion(),
-                              partialOp.getRegion().begin());
-
-  Block &body = partialOp.getRegion().front();
-  rewriter.setInsertionPointToStart(&body);
-
-  unsigned innerIdxDim = reductionDim + 1;
-  unsigned outerIdxDim = insertSplitDimension;
-
-  // Compute global index (gidx) for reduction when the original reduction
-  // dimension is split into [outerIdx, innerIdx] using `ratio`. This is used to
-  // correctly compute the global index for comparisons and index selection.
-  Value outerIdx = rewriter.create<linalg::IndexOp>(loc, outerIdxDim);
-  Value innerIdx = rewriter.create<linalg::IndexOp>(loc, innerIdxDim);
-  Value offset = rewriter.create<arith::MulIOp>(loc, outerIdx, tileSize);
-  Value gidx = rewriter.create<arith::AddIOp>(loc, offset, innerIdx);
-
-  auto selectOp = dyn_cast<arith::SelectOp>(combinerOps.selectOp);
-  Value oldIdx = selectOp.getTrueValue();
-  Value newIdx = gidx;
-  if (oldIdx.getType() != gidx.getType()) {
-    newIdx = rewriter.create<arith::IndexCastOp>(loc, oldIdx.getType(), gidx);
-  }
-  selectOp.setOperand(1, newIdx);
-  rewriter.setInsertionPointAfter(partialOp);
+      newIteratorTypes,
+      [reductionDim](OpBuilder &b, Location loc, ValueRange args) {
+        Value in = args[0], outVal = args[1], outIdx = args[2];
```

**Comment:**
nit: llvm discourages defining multiple variables on a single line.

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp:694`

```diff
@@ -693,19 +684,26 @@ splitArgmaxReduction(RewriterBase &rewriter, linalg::GenericOp genericOp,
       AffineMap::get(intermRank, 0, resultExprs, rewriter.getContext());
   SmallVector<AffineMap> finalReductionMaps = {valueMap, indexMap, outputMap,
                                                outputMap};
-
-  // Create block for final reduction region.
   auto finalReduction = rewriter.create<linalg::GenericOp>(
       loc, genericOp.getResultTypes(),
-      ValueRange{partialOp.getResult(0), partialOp.getResult(1)},
+      ValueRange{partialArgmax.getResult(0), partialArgmax.getResult(1)},
       genericOp.getDpsInits(), finalReductionMaps, reductionIteratorTypes,
-      [combinerOps](OpBuilder &b, Location loc, ValueRange inputs) {
+      [combinerOps, tileSize, insertSplitDimension](OpBuilder &b, Location loc,
+                                                    ValueRange inputs) {
+        Value val = inputs[0], local = inputs[1];
+        Value outVal = inputs[2], outIdx = inputs[3];
```

**Comment:**
also here

---

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp:894`

```diff
@@ -891,6 +891,15 @@ bool isArgmaxOp(linalg::GenericOp genericOp) {
     if (!matchPattern(producer, m_Op<arith::SelectOp>())) {
       return false;
     }
+    auto selectOp = dyn_cast<arith::SelectOp>(producerOutput.getDefiningOp());
```

**Comment:**
Use `cast` if you require the result to be a select. Otherwise check if the result is not null. See https://llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates

---


---


## [PR #20438](https://github.com/iree-org/iree/pull/20438): [tuner] expose python binding for getting the tuner root ops

### Review Summary

**CHANGES_REQUESTED** (2025-04-02)

Although the logic is straightforward, I think this needs tests to make sure everything is threaded through properly across iree / c / python.


**COMMENTED** (2025-04-02)


**COMMENTED** (2025-04-02)


**APPROVED** (2025-04-09)

Looks good % nit


### Code Comments

**File:** `compiler/bindings/python/test/ir/dialects_test.py:329`

```diff
@@ -309,3 +309,22 @@ def compilation_info():
     assert compilation_info is not None
     assert compilation_info.lowering_config == lowering_config
     assert compilation_info.translation_info == translation_info
+
+
+@run
+def root_op():
+    module_str = """
+        module {
+            func.func @matmul(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
+                %cst = arith.constant 0.000000e+00 : f32
+                %0 = tensor.empty() : tensor<4x4xf32>
+                %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
+                %2 = linalg.matmul { root_op } ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%1 : tensor<4x4xf32>) -> tensor<4x4xf32>
+                return %2 : tensor<4x4xf32>
+            }
+        }
+    """
+    input_module = ir.Module.parse(module_str)
+    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
+    assert len(root_op_list) == 1
```

**Comment:**
Could you add two more test cases:
1. no root ops
2. two or more root ops

This is so that we can exercise the vector population logic.

---

**File:** `compiler/bindings/python/test/ir/dialects_test.py:315`

```diff
@@ -309,3 +309,22 @@ def compilation_info():
     assert compilation_info is not None
     assert compilation_info.lowering_config == lowering_config
     assert compilation_info.translation_info == translation_info
+
+
+@run
+def root_op():
```

**Comment:**
This test is not a dialect test, we should move it to a different file

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp:1055`

```diff
@@ -1049,4 +1049,17 @@ queryMMAIntrinsics(IREE::HAL::ExecutableVariantOp executableOp) {
   return mmaIntrinsics;
 }
 
+SmallVector<Operation *> getTunerRootOps(mlir::ModuleOp moduleOp) {
+  SmallVector<Operation *> rootOps;
+
+  // Walk all operations in the module recursively
```

**Comment:**
https://llvm.org/docs/CodingStandards.html#commenting

---

**File:** `compiler/bindings/python/test/ir/tuner_test.py:39`

```diff
@@ -0,0 +1,46 @@
+# Copyright 2025 The IREE Authors
+#
+# Licensed under the Apache License v2.0 with LLVM Exceptions.
+# See https://llvm.org/LICENSE.txt for license information.
+# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+from iree.compiler import ir
+from iree.compiler.dialects import iree_codegen
+
+
+def run(fn):
+    with ir.Context(), ir.Location.unknown():
+        module = ir.Module.create()
+        with ir.InsertionPoint(module.body):
+            print("\nTEST:", fn.__name__)
+            fn()
+    return fn
+
+
+@run
+def root_op():
+    module_str = """
+        module {
+            func.func @matmul(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
+                %cst = arith.constant 0.000000e+00 : f32
+                %0 = tensor.empty() : tensor<4x4xf32>
+                %1 = linalg.fill { root_op } ins(%cst : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
+                %2 = linalg.matmul { root_op } ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%1 : tensor<4x4xf32>) -> tensor<4x4xf32>
+                return %2 : tensor<4x4xf32>
+            }
+        }
+    """
+    input_module = ir.Module.parse(module_str)
+    root_op_list = iree_codegen.get_tuner_root_ops(input_module)
+    assert len(root_op_list) == 2
+    assert root_op_list[0].name == "linalg.fill"
+    assert root_op_list[1].name == "linalg.matmul"
+
+    del root_op_list[0].attributes["root_op"]
```

**Comment:**
I don't think python is supposed to modify IR like this [1]: there may be some dangling references to IR objects. Instead, can you make it a few separate test cases with their own individual inputs?

[1] https://mlir.llvm.org/docs/Bindings/Python/#ownership-in-the-core-ir

---

**File:** `compiler/bindings/python/test/ir/tuner_test.py:21`

```diff
@@ -0,0 +1,46 @@
+# Copyright 2025 The IREE Authors
+#
+# Licensed under the Apache License v2.0 with LLVM Exceptions.
+# See https://llvm.org/LICENSE.txt for license information.
+# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+from iree.compiler import ir
+from iree.compiler.dialects import iree_codegen
+
+
+def run(fn):
+    with ir.Context(), ir.Location.unknown():
+        module = ir.Module.create()
+        with ir.InsertionPoint(module.body):
+            print("\nTEST:", fn.__name__)
+            fn()
+    return fn
+
+
+@run
+def root_op():
```

**Comment:**
I think this code should be in a directory outside of `ir`, like `compiler/bindings/python/test/api/tuner_api_test.py`

---

**File:** `compiler/bindings/python/test/api/tuner_api_test.py:23`

```diff
@@ -0,0 +1,68 @@
+# Copyright 2025 The IREE Authors
+#
+# Licensed under the Apache License v2.0 with LLVM Exceptions.
+# See https://llvm.org/LICENSE.txt for license information.
+# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+from iree.compiler import ir
+from iree.compiler.dialects import iree_codegen
+
+
+def run(fn):
+    with ir.Context(), ir.Location.unknown():
+        module = ir.Module.create()
+        with ir.InsertionPoint(module.body):
+            print("\nTEST:", fn.__name__)
+            fn()
+    return fn
+
+
+@run
+def root_op():
+    module_str = """
+        module {
```

**Comment:**
Can we use implicit modules here to make this a bit shorter?

---

**File:** `compiler/bindings/python/test/api/tuner_api_test.py:33`

```diff
@@ -0,0 +1,68 @@
+# Copyright 2025 The IREE Authors
+#
+# Licensed under the Apache License v2.0 with LLVM Exceptions.
+# See https://llvm.org/LICENSE.txt for license information.
+# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+from iree.compiler import ir
+from iree.compiler.dialects import iree_codegen
+
+
+def run(fn):
+    with ir.Context(), ir.Location.unknown():
+        module = ir.Module.create()
+        with ir.InsertionPoint(module.body):
+            print("\nTEST:", fn.__name__)
+            fn()
+    return fn
+
+
+@run
+def root_op():
+    module_str = """
+        module {
+            func.func @matmul(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
+                %cst = arith.constant 0.000000e+00 : f32
+                %0 = tensor.empty() : tensor<4x4xf32>
+                %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
+                %2 = linalg.matmul ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%1 : tensor<4x4xf32>) -> tensor<4x4xf32>
+                return %2 : tensor<4x4xf32>
+            }
+        }
+    """
+    input_module = ir.Module.parse(module_str)
```

**Comment:**
assert that the module parsed so that we can quickly identify issues with mlir syntax changes. Also in other testcases

---


---


## [PR #20246](https://github.com/iree-org/iree/pull/20246): Integrates/llvm 20250314 llvm/llvm-project@e45090e

### Review Summary

**APPROVED** (2025-03-15)



---


## [PR #20207](https://github.com/iree-org/iree/pull/20207): Integrates/llvm 20250310: Bump to llvm/llvm-project@967ab7e

### Review Summary

**COMMENTED** (2025-03-11)


**COMMENTED** (2025-03-11)


**COMMENTED** (2025-03-11)


**APPROVED** (2025-03-11)


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/LLVMCPU/ConvertToLLVM.cpp:986`

```diff
@@ -975,15 +975,15 @@ void ConvertToLLVMPass::runOnOperation() {
     // TODO: doubtful that the "default" does what one want here, it is likely
     // better to use outerproduct.
     vector::populateVectorContractLoweringPatterns(
-        patterns, vector::VectorTransformsOptions());
+        patterns, vector::VectorContractLowering::Dot);
     vector::populateVectorMaskMaterializationPatterns(
         patterns, /*force32BitVectorIndices=*/false);
     vector::populateVectorMaskOpLoweringPatterns(patterns);
     vector::populateVectorShapeCastLoweringPatterns(patterns);
     // TODO: doubtful that the "default" does what one want here, it is likely
     // better to use shuffle.
     vector::populateVectorTransposeLoweringPatterns(
-        patterns, vector::VectorTransformsOptions());
+        patterns, vector::VectorTransposeLowering::EltWise);
```

**Comment:**
How did you decide these options? Could we also pick the default values for these new enums, e.g., `vector::VectorTransposeLowering()`?

Could you add the mapping to the PR description?

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMCPU/ConvertToLLVM.cpp:986`

```diff
@@ -975,15 +975,15 @@ void ConvertToLLVMPass::runOnOperation() {
     // TODO: doubtful that the "default" does what one want here, it is likely
     // better to use outerproduct.
     vector::populateVectorContractLoweringPatterns(
-        patterns, vector::VectorTransformsOptions());
+        patterns, vector::VectorContractLowering::Dot);
     vector::populateVectorMaskMaterializationPatterns(
         patterns, /*force32BitVectorIndices=*/false);
     vector::populateVectorMaskOpLoweringPatterns(patterns);
     vector::populateVectorShapeCastLoweringPatterns(patterns);
     // TODO: doubtful that the "default" does what one want here, it is likely
     // better to use shuffle.
     vector::populateVectorTransposeLoweringPatterns(
-        patterns, vector::VectorTransformsOptions());
+        patterns, vector::VectorTransposeLowering::EltWise);
```

**Comment:**
Why not do this then to pick up the defaults:
```c++
    VectorTransformsOptions defaultOptions;
    ...
    vector::populateVectorTransposeLoweringPatterns(
        patterns, defaultOptions.vectorTransposeLowering);
 ```

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMCPU/ConvertToLLVM.cpp:986`

```diff
@@ -975,15 +975,15 @@ void ConvertToLLVMPass::runOnOperation() {
     // TODO: doubtful that the "default" does what one want here, it is likely
     // better to use outerproduct.
     vector::populateVectorContractLoweringPatterns(
-        patterns, vector::VectorTransformsOptions());
+        patterns, vector::VectorContractLowering::Dot);
     vector::populateVectorMaskMaterializationPatterns(
         patterns, /*force32BitVectorIndices=*/false);
     vector::populateVectorMaskOpLoweringPatterns(patterns);
     vector::populateVectorShapeCastLoweringPatterns(patterns);
     // TODO: doubtful that the "default" does what one want here, it is likely
     // better to use shuffle.
     vector::populateVectorTransposeLoweringPatterns(
-        patterns, vector::VectorTransformsOptions());
+        patterns, vector::VectorTransposeLowering::EltWise);
```

**Comment:**
This is trivial to be optimized out for any c++ compiler -- just SROA and DCE

---


---


## [PR #20173](https://github.com/iree-org/iree/pull/20173): [Codegen][Tuner] improve verifier for the default attribute

### Review Summary

**CHANGES_REQUESTED** (2025-03-06)

Could you also update the documentation in https://iree.dev/reference/tuning/ ?


**COMMENTED** (2025-03-06)


**COMMENTED** (2025-03-06)


**COMMENTED** (2025-03-07)


**COMMENTED** (2025-03-07)


**CHANGES_REQUESTED** (2025-03-10)


**COMMENTED** (2025-03-10)


**COMMENTED** (2025-03-10)


**COMMENTED** (2025-03-12)

The new tests look solid, just a few remaining issues


**COMMENTED** (2025-03-12)


**COMMENTED** (2025-03-12)


**APPROVED** (2025-03-12)

LGTM


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:261`

```diff
@@ -103,3 +103,30 @@ module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_defa
       transform.yield %res_b : !transform.any_op
   }
 }
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+      transform.yield
+  }
```

**Comment:**
Do we need these to check for this error? Let's make sure we minimize the amount of unrelated code and keep the tests minimal

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp:77`

```diff
@@ -58,17 +58,25 @@ IREECodegenDialect::verifyOperationAttribute(Operation *op,
   StringRef symbol = attribute.getName().strref();
   Attribute attr = attribute.getValue();
   // This function verifies the validity of a specific operation attribute.
-  // - If the attribute's name matches `kTuningDefaultSpecAttrName`, make
-  //   sure it contains a single named sequence op with name `__kernel_config`.
+  // - If the attribute's name matches kTuningSpecDefaultEntrypointAttrName
+  // (`iree_codegen.tuning_spec_with_default_entrypoint`):
+  //   1. Ensure that the module contains a single named sequence operation with
+  //   the name `__kernel_config`.
+  //   2. Verify that this `__kernel_config` named sequence operation has the
+  //   attribute `iree_codegen.tuning_spec_entrypoint`.
+  //   3. Ensure that the named sequence operation contains exactly **one**
+  //   `ForeachMatchOp`.
+  //   4. Ensure that only one named sequence operation with the
+  //   `iree_codegen.tuning_spec_entrypoint` attribute.
   // - If the attribute's name matches `kTuningSpecEntrypointAttrName`
-  // ("iree_codegen.tuning_spec_entrypoint"):
+  // (`iree_codegen.tuning_spec_entrypoint`):
   //   1. The attribute value must be a UnitAttr.
   //   2. If the operation is a transform::NamedSequenceOp:
   //      - The operation's function signature must satisfy the following:
-  //         a. It must have exactly one result type, and the result must be of
-  //         type `transform::AnyOpType`.
-  //         b. It must have exactly one argument type, and the argument must be
-  //         of type `transform::AnyOpType`.
+  //         a. It must have exactly one result type, and the result must be
+  //         of type `transform::AnyOpType`. b. It must have exactly one
```

**Comment:**
b. should be on a new line

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:99`

```diff
@@ -66,3 +66,67 @@ module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_defa
     return
   }
 }
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  // expected-error @+1{{The named sequence '__kernel_config' must have the attribute 'iree_codegen.tuning_spec_entrypoint'}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) {
+      transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match_a(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @match_b(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+      transform.yield
+  }
+
+  // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'ForeachMatchOp', but found 2}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+
+      %res_a = transform.foreach_match in %arg0 @match_a -> @apply_op_config
```

**Comment:**
What if there are other ops like `transform.include`?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:99`

```diff
@@ -66,3 +66,67 @@ module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_defa
     return
   }
 }
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  // expected-error @+1{{The named sequence '__kernel_config' must have the attribute 'iree_codegen.tuning_spec_entrypoint'}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) {
+      transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match_a(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @match_b(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+      transform.yield
+  }
+
+  // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'ForeachMatchOp', but found 2}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+
+      %res_a = transform.foreach_match in %arg0 @match_a -> @apply_op_config
```

**Comment:**
I'm thinking whether requiring a single for_each only would make merging simpler

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:99`

```diff
@@ -66,3 +66,67 @@ module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_defa
     return
   }
 }
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  // expected-error @+1{{The named sequence '__kernel_config' must have the attribute 'iree_codegen.tuning_spec_entrypoint'}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) {
+      transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match_a(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @match_b(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+      transform.yield
+  }
+
+  // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'ForeachMatchOp', but found 2}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+
+      %res_a = transform.foreach_match in %arg0 @match_a -> @apply_op_config
```

**Comment:**
Let's check for this then

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/materialize_tuning_specs.mlir:14`

```diff
@@ -11,7 +11,6 @@
 // Check that the final tuning spec is as expected when the user tuning spec is provided.
 
 // CHECK-LABEL: module @iree_linked_tuning_spec
-// CHECK-SAME:    iree_codegen.tuning_spec_with_default_entrypoint
```

**Comment:**
I think this effectively break this test. This is fine, but we shoulda add a TODO to fix it once the merging logic can handle foreach_match ops.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:118`

```diff
@@ -66,3 +66,56 @@ module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_defa
     return
   }
 }
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  // expected-error @+1{{The named sequence '__kernel_config' must have the attribute 'iree_codegen.tuning_spec_entrypoint'}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) {
+      transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match_a(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @match_b(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+      transform.yield
+  }
+
+  // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'ForeachMatchOp', but found 2}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+
+      %res_a = transform.foreach_match in %arg0 @match_a -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      %res_b = transform.foreach_match in %res_a @match_b -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      transform.yield %res_b : !transform.any_op
+  }
+}
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+
+  transform.named_sequence @main(%arg0: !transform.any_op {transform.readonly})
+      -> !transform.any_op attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+}
```

**Comment:**
We should add a test that has a foreach_match op in `__kernel_config` and some other op like print or include.

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp:112`

```diff
@@ -58,29 +58,75 @@ IREECodegenDialect::verifyOperationAttribute(Operation *op,
   StringRef symbol = attribute.getName().strref();
   Attribute attr = attribute.getValue();
   // This function verifies the validity of a specific operation attribute.
-  // - If the attribute's name matches `kTuningDefaultSpecAttrName`, make
-  //   sure it contains a single named sequence op with name `__kernel_config`.
+  // - If the attribute's name matches kTuningSpecDefaultEntrypointAttrName
+  // (`iree_codegen.tuning_spec_with_default_entrypoint`):
+  //   1. Ensure that the module contains a single named sequence operation with
+  //   the name `__kernel_config`.
+  //   2. Verify that this `__kernel_config` named sequence operation has the
+  //   attribute `iree_codegen.tuning_spec_entrypoint`.
+  //   3. Ensure that the named sequence operation contains exactly **one**
+  //   `ForeachMatchOp`.
+  //   4. Ensure that only one named sequence operation with the
+  //   `iree_codegen.tuning_spec_entrypoint` attribute.
   // - If the attribute's name matches `kTuningSpecEntrypointAttrName`
-  // ("iree_codegen.tuning_spec_entrypoint"):
+  // (`iree_codegen.tuning_spec_entrypoint`):
   //   1. The attribute value must be a UnitAttr.
   //   2. If the operation is a transform::NamedSequenceOp:
   //      - The operation's function signature must satisfy the following:
-  //         a. It must have exactly one result type, and the result must be of
-  //         type `transform::AnyOpType`.
+  //         a. It must have exactly one result type, and the result must be
+  //         of type `transform::AnyOpType`.
   //         b. It must have exactly one argument type, and the argument must be
   //         of type `transform::AnyOpType`.
 
   if (symbol == kTuningSpecDefaultEntrypointAttrName) {
     if (auto moduleOp = dyn_cast<ModuleOp>(op)) {
-      if (!llvm::any_of(moduleOp.getOps<transform::NamedSequenceOp>(),
+      auto kernelConfigOpIt =
+          llvm::find_if(moduleOp.getOps<transform::NamedSequenceOp>(),
                         [](transform::NamedSequenceOp op) {
                           return op.getName() == kKernelConfigSpecName;
-                        })) {
+                        });
+
+      if (kernelConfigOpIt ==
+          moduleOp.getOps<transform::NamedSequenceOp>().end()) {
         return moduleOp.emitError()
-               << "The tuning specification must include a named "
-                  "sequence with the symbol name '"
+               << "The tuning specification must include a named sequence with "
+                  "the symbol name '"
                << kKernelConfigSpecName << "'.";
       }
+
+      transform::NamedSequenceOp kernelConfigOp = *kernelConfigOpIt;
+
+      // Verify that the kernelConfigOp has the attribute
+      // `iree_codegen.tuning_spec_entrypoint`.
+      if (!kernelConfigOp->hasAttr(kTuningSpecEntrypointAttrName)) {
+        return kernelConfigOp.emitError()
+               << "The named sequence '" << kKernelConfigSpecName
+               << "' must have the attribute '" << kTuningSpecEntrypointAttrName
+               << "'.";
+      }
+
+      auto tuningSpecOps = llvm::filter_to_vector(
+          moduleOp.getOps<transform::NamedSequenceOp>(),
+          [](transform::NamedSequenceOp op) {
+            return op->hasAttr(kTuningSpecEntrypointAttrName);
+          });
```

**Comment:**
What if we find ops that are not named sequences?

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp:123`

```diff
@@ -58,29 +58,75 @@ IREECodegenDialect::verifyOperationAttribute(Operation *op,
   StringRef symbol = attribute.getName().strref();
   Attribute attr = attribute.getValue();
   // This function verifies the validity of a specific operation attribute.
-  // - If the attribute's name matches `kTuningDefaultSpecAttrName`, make
-  //   sure it contains a single named sequence op with name `__kernel_config`.
+  // - If the attribute's name matches kTuningSpecDefaultEntrypointAttrName
+  // (`iree_codegen.tuning_spec_with_default_entrypoint`):
+  //   1. Ensure that the module contains a single named sequence operation with
+  //   the name `__kernel_config`.
+  //   2. Verify that this `__kernel_config` named sequence operation has the
+  //   attribute `iree_codegen.tuning_spec_entrypoint`.
+  //   3. Ensure that the named sequence operation contains exactly **one**
+  //   `ForeachMatchOp`.
+  //   4. Ensure that only one named sequence operation with the
+  //   `iree_codegen.tuning_spec_entrypoint` attribute.
   // - If the attribute's name matches `kTuningSpecEntrypointAttrName`
-  // ("iree_codegen.tuning_spec_entrypoint"):
+  // (`iree_codegen.tuning_spec_entrypoint`):
   //   1. The attribute value must be a UnitAttr.
   //   2. If the operation is a transform::NamedSequenceOp:
   //      - The operation's function signature must satisfy the following:
-  //         a. It must have exactly one result type, and the result must be of
-  //         type `transform::AnyOpType`.
+  //         a. It must have exactly one result type, and the result must be
+  //         of type `transform::AnyOpType`.
   //         b. It must have exactly one argument type, and the argument must be
   //         of type `transform::AnyOpType`.
 
   if (symbol == kTuningSpecDefaultEntrypointAttrName) {
     if (auto moduleOp = dyn_cast<ModuleOp>(op)) {
-      if (!llvm::any_of(moduleOp.getOps<transform::NamedSequenceOp>(),
+      auto kernelConfigOpIt =
+          llvm::find_if(moduleOp.getOps<transform::NamedSequenceOp>(),
                         [](transform::NamedSequenceOp op) {
                           return op.getName() == kKernelConfigSpecName;
-                        })) {
+                        });
+
+      if (kernelConfigOpIt ==
+          moduleOp.getOps<transform::NamedSequenceOp>().end()) {
         return moduleOp.emitError()
-               << "The tuning specification must include a named "
-                  "sequence with the symbol name '"
+               << "The tuning specification must include a named sequence with "
+                  "the symbol name '"
                << kKernelConfigSpecName << "'.";
       }
+
+      transform::NamedSequenceOp kernelConfigOp = *kernelConfigOpIt;
+
+      // Verify that the kernelConfigOp has the attribute
+      // `iree_codegen.tuning_spec_entrypoint`.
+      if (!kernelConfigOp->hasAttr(kTuningSpecEntrypointAttrName)) {
+        return kernelConfigOp.emitError()
+               << "The named sequence '" << kKernelConfigSpecName
+               << "' must have the attribute '" << kTuningSpecEntrypointAttrName
+               << "'.";
+      }
+
+      auto tuningSpecOps = llvm::filter_to_vector(
+          moduleOp.getOps<transform::NamedSequenceOp>(),
+          [](transform::NamedSequenceOp op) {
+            return op->hasAttr(kTuningSpecEntrypointAttrName);
+          });
+
+      if (tuningSpecOps.size() != 1) {
+        return moduleOp.emitError()
+               << "Expected exactly one NamedSequenceOp with the attribute '"
+               << kTuningSpecEntrypointAttrName << "', but found "
+               << tuningSpecOps.size() << ".";
+      }
+
+      // Ensure there is exactly one ForeachMatchOp inside the kernelConfigOp.
+      auto foreachMatchOps =
+          llvm::to_vector(kernelConfigOp.getOps<transform::ForeachMatchOp>());
```

**Comment:**
What if there's a single foreach_match but also some other ops like `transform.include`?

---

**File:** `docs/website/docs/reference/tuning.md:129`

```diff
@@ -124,7 +124,9 @@ that conform to the following format:
 * All entry points in the final tuning specs must either read
   (`transform.readonly`) or consume (`transform.consumed`) the argument.
 * The `iree_codegen.tuning_spec_with_default_entrypoint` attribute ensures that
-  the tuning spec includes a named sequence op with name `__kernel_config`.
+  the tuning spec includes a named sequence op with name `__kernel_config`, which
+  must contain exactly one `foreach_match` op. Furthermore, only one tuning spec
+  entry point is allowed, and it must be `__kernel_config` op.
```

**Comment:**
```suggestion
  the tuning spec includes a named sequence op with name `__kernel_config`, which
  must contain exactly one `foreach_match` op.
```
this seems redundant to me

---

**File:** `compiler/plugins/target/ROCM/test/default_tuning_specs_amdgpu.mlir:39`

```diff
@@ -36,7 +36,6 @@
 // materialized. The user spec should have precedence over the default one.
 
 // BOTH-LABEL: module @iree_linked_tuning_spec
-// BOTH-SAME:    iree_codegen.tuning_spec_with_default_entrypoint
```

**Comment:**
Can we add a TODO to revisit this test and re-add this CHECK when the new linking lands?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:86`

```diff
@@ -83,7 +83,6 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
       0, hasConsumedSequences ? kArgConsumedAttrName : kArgReadOnlyAttrName,
       builder.getUnitAttr());
   newSpec->setAttr(kTuningSpecEntrypointAttrName, builder.getUnitAttr());
-  module->setAttr(kTuningSpecDefaultEntrypointAttrName, builder.getUnitAttr());
```

**Comment:**
Also here: let's keep track of re-enabling this

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/tuning_spec_default.mlir:14`

```diff
@@ -1,9 +1,18 @@
 // RUN: iree-opt %s
 
 module @user_spec attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
-  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
-    attributes { iree_codegen.tuning_spec_entrypoint } {
-    transform.print {name = "Hello Tuning Spec", skip_regions}
-    transform.yield %arg0 : !transform.any_op
+  transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+    transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+    transform.yield
+  }
+
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+      %res = transform.foreach_match in %arg0 @match -> @apply_op_config
```

**Comment:**
nit: put match on its own line -- I find it weird to have the whole loop with its body on a single line

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:95`

```diff
@@ -66,3 +66,267 @@ module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_defa
     return
   }
 }
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  // expected-error @+1{{The named sequence '__kernel_config' must have the attribute 'iree_codegen.tuning_spec_entrypoint'}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) {
+      transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match_a(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @match_b(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+      transform.yield
+  }
+
+  // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'ForeachMatchOp', but found 2}}
```

**Comment:**
Can we add some clue as to where this rule comes from? This is related to the attributes you set. Otherwise the `__kernel_config` name is not special on its own.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:85`

```diff
@@ -66,3 +66,267 @@ module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_defa
     return
   }
 }
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  // expected-error @+1{{The named sequence '__kernel_config' must have the attribute 'iree_codegen.tuning_spec_entrypoint'}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) {
+      transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match_a(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @match_b(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
```

**Comment:**
We don't need this named sequence in this test -- we can check the same thing by having two foreach_match ops that use the same match function

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:109`

```diff
@@ -66,3 +66,267 @@ module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_defa
     return
   }
 }
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  // expected-error @+1{{The named sequence '__kernel_config' must have the attribute 'iree_codegen.tuning_spec_entrypoint'}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) {
+      transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match_a(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @match_b(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+      transform.yield
+  }
+
+  // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'ForeachMatchOp', but found 2}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+
+      %res_a = transform.foreach_match in %arg0 @match_a -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      %res_b = transform.foreach_match in %res_a @match_b -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      transform.yield %res_b : !transform.any_op
+  }
+}
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
```

**Comment:**
Also here: we should print which attribute adds this verification rule. It's not `tuning_spec_entrypoint` that's at issue

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:140`

```diff
@@ -66,3 +66,267 @@ module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_defa
     return
   }
 }
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  // expected-error @+1{{The named sequence '__kernel_config' must have the attribute 'iree_codegen.tuning_spec_entrypoint'}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) {
+      transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match_a(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @match_b(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+      transform.yield
+  }
+
+  // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'ForeachMatchOp', but found 2}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+
+      %res_a = transform.foreach_match in %arg0 @match_a -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      %res_b = transform.foreach_match in %res_a @match_b -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      transform.yield %res_b : !transform.any_op
+  }
+}
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+
+  transform.named_sequence @main(%arg0: !transform.any_op {transform.readonly})
+      -> !transform.any_op attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'transform::YieldOp', but found 2}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+        %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+        transform.yield %res : !transform.any_op
```

**Comment:**
Two yields don't make sense -- I'd expect this to be a terminator. Instead of checking that there is exact one yield and one foreach_match, we can check that the only two ops are foreach_match and yield without counting how many there are.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:170`

```diff
@@ -66,3 +66,267 @@ module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_defa
     return
   }
 }
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  // expected-error @+1{{The named sequence '__kernel_config' must have the attribute 'iree_codegen.tuning_spec_entrypoint'}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) {
+      transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match_a(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @match_b(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+      transform.yield
+  }
+
+  // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'ForeachMatchOp', but found 2}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+
+      %res_a = transform.foreach_match in %arg0 @match_a -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      %res_b = transform.foreach_match in %res_a @match_b -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      transform.yield %res_b : !transform.any_op
+  }
+}
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+
+  transform.named_sequence @main(%arg0: !transform.any_op {transform.readonly})
+      -> !transform.any_op attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'transform::YieldOp', but found 2}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+        %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+
+  module @extra_module attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence private @dummy_func(!transform.any_op {transform.consumed}) -> !transform.any_op
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_configbut found an unsupported operation: transform.include}}
```

**Comment:**
The quote is not closed

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:193`

```diff
@@ -66,3 +66,267 @@ module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_defa
     return
   }
 }
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  // expected-error @+1{{The named sequence '__kernel_config' must have the attribute 'iree_codegen.tuning_spec_entrypoint'}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) {
+      transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match_a(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @match_b(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+      transform.yield
+  }
+
+  // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'ForeachMatchOp', but found 2}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+
+      %res_a = transform.foreach_match in %arg0 @match_a -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      %res_b = transform.foreach_match in %res_a @match_b -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      transform.yield %res_b : !transform.any_op
+  }
+}
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+
+  transform.named_sequence @main(%arg0: !transform.any_op {transform.readonly})
+      -> !transform.any_op attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'transform::YieldOp', but found 2}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+        %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+
+  module @extra_module attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence private @dummy_func(!transform.any_op {transform.consumed}) -> !transform.any_op
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_configbut found an unsupported operation: transform.include}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+        %tmp = transform.include @dummy_func failures(suppress) (%arg0) : (!transform.any_op) -> (!transform.any_op)
+        %res = transform.foreach_match in %tmp @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence private @dummy_func(!transform.any_op {transform.consumed}) -> !transform.any_op
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_configbut found an unsupported operation: transform.print}}
```

**Comment:**
Something is wrong with this error message -- the closing quote is missing and there's no space before 'but'.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:220`

```diff
@@ -66,3 +66,267 @@ module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_defa
     return
   }
 }
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  // expected-error @+1{{The named sequence '__kernel_config' must have the attribute 'iree_codegen.tuning_spec_entrypoint'}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) {
+      transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match_a(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @match_b(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+      transform.yield
+  }
+
+  // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'ForeachMatchOp', but found 2}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+
+      %res_a = transform.foreach_match in %arg0 @match_a -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      %res_b = transform.foreach_match in %res_a @match_b -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      transform.yield %res_b : !transform.any_op
+  }
+}
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+
+  transform.named_sequence @main(%arg0: !transform.any_op {transform.readonly})
+      -> !transform.any_op attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'transform::YieldOp', but found 2}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+        %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+
+  module @extra_module attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence private @dummy_func(!transform.any_op {transform.consumed}) -> !transform.any_op
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_configbut found an unsupported operation: transform.include}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+        %tmp = transform.include @dummy_func failures(suppress) (%arg0) : (!transform.any_op) -> (!transform.any_op)
+        %res = transform.foreach_match in %tmp @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence private @dummy_func(!transform.any_op {transform.consumed}) -> !transform.any_op
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_configbut found an unsupported operation: transform.print}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+         transform.print {name = "Hello"}
+        %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg: !transform.any_op {transform.readonly})
+    -> (!transform.any_op, !transform.any_op) {
+    transform.yield %arg, %arg : !transform.any_op, !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op1: !transform.any_op {transform.readonly}, %op2: !transform.any_op {transform.readonly})
+    -> (!transform.any_op) {
+    transform.yield %op1 : !transform.any_op
+  }
+
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+    -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    // expected-error @+1 {{ForeachMatchOp must return exactly one any_op result}}
+    %res1, %res2 = transform.foreach_match in %arg0 @match -> @apply_op_config
```

**Comment:**
Also here: let's print where this rule comes from

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:245`

```diff
@@ -66,3 +66,267 @@ module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_defa
     return
   }
 }
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  // expected-error @+1{{The named sequence '__kernel_config' must have the attribute 'iree_codegen.tuning_spec_entrypoint'}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) {
+      transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match_a(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @match_b(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+      transform.yield
+  }
+
+  // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'ForeachMatchOp', but found 2}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+
+      %res_a = transform.foreach_match in %arg0 @match_a -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      %res_b = transform.foreach_match in %res_a @match_b -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      transform.yield %res_b : !transform.any_op
+  }
+}
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+
+  transform.named_sequence @main(%arg0: !transform.any_op {transform.readonly})
+      -> !transform.any_op attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'transform::YieldOp', but found 2}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+        %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+
+  module @extra_module attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence private @dummy_func(!transform.any_op {transform.consumed}) -> !transform.any_op
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_configbut found an unsupported operation: transform.include}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+        %tmp = transform.include @dummy_func failures(suppress) (%arg0) : (!transform.any_op) -> (!transform.any_op)
+        %res = transform.foreach_match in %tmp @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence private @dummy_func(!transform.any_op {transform.consumed}) -> !transform.any_op
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_configbut found an unsupported operation: transform.print}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+         transform.print {name = "Hello"}
+        %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg: !transform.any_op {transform.readonly})
+    -> (!transform.any_op, !transform.any_op) {
+    transform.yield %arg, %arg : !transform.any_op, !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op1: !transform.any_op {transform.readonly}, %op2: !transform.any_op {transform.readonly})
+    -> (!transform.any_op) {
+    transform.yield %op1 : !transform.any_op
+  }
+
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+    -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    // expected-error @+1 {{ForeachMatchOp must return exactly one any_op result}}
+    %res1, %res2 = transform.foreach_match in %arg0 @match -> @apply_op_config
+      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
+
+    transform.yield %res1 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg: !transform.any_op) -> (!transform.any_op) {
+    transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op) {
+    transform.yield
+  }
+
+  // expected-error @+1 {{Tuning spec entry point expected to return any_op}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op)
+      -> (f32) attributes { iree_codegen.tuning_spec_entrypoint } {
+     // expected-error @+1 {{ForeachMatchOp must return exactly one any_op result}}
+    %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+      : (!transform.any_op) -> (f32)
```

**Comment:**
This looks like an ill-formed forech_match op. It should check that the return type makes sense. Should we fix the `transform_match` verifier instead?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:263`

```diff
@@ -66,3 +66,267 @@ module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_defa
     return
   }
 }
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  // expected-error @+1{{The named sequence '__kernel_config' must have the attribute 'iree_codegen.tuning_spec_entrypoint'}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) {
+      transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match_a(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @match_b(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+      transform.yield
+  }
+
+  // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'ForeachMatchOp', but found 2}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+
+      %res_a = transform.foreach_match in %arg0 @match_a -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      %res_b = transform.foreach_match in %res_a @match_b -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      transform.yield %res_b : !transform.any_op
+  }
+}
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+
+  transform.named_sequence @main(%arg0: !transform.any_op {transform.readonly})
+      -> !transform.any_op attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'transform::YieldOp', but found 2}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+        %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+
+  module @extra_module attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence private @dummy_func(!transform.any_op {transform.consumed}) -> !transform.any_op
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_configbut found an unsupported operation: transform.include}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+        %tmp = transform.include @dummy_func failures(suppress) (%arg0) : (!transform.any_op) -> (!transform.any_op)
+        %res = transform.foreach_match in %tmp @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence private @dummy_func(!transform.any_op {transform.consumed}) -> !transform.any_op
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_configbut found an unsupported operation: transform.print}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+         transform.print {name = "Hello"}
+        %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg: !transform.any_op {transform.readonly})
+    -> (!transform.any_op, !transform.any_op) {
+    transform.yield %arg, %arg : !transform.any_op, !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op1: !transform.any_op {transform.readonly}, %op2: !transform.any_op {transform.readonly})
+    -> (!transform.any_op) {
+    transform.yield %op1 : !transform.any_op
+  }
+
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+    -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    // expected-error @+1 {{ForeachMatchOp must return exactly one any_op result}}
+    %res1, %res2 = transform.foreach_match in %arg0 @match -> @apply_op_config
+      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
+
+    transform.yield %res1 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg: !transform.any_op) -> (!transform.any_op) {
+    transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op) {
+    transform.yield
+  }
+
+  // expected-error @+1 {{Tuning spec entry point expected to return any_op}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op)
+      -> (f32) attributes { iree_codegen.tuning_spec_entrypoint } {
+     // expected-error @+1 {{ForeachMatchOp must return exactly one any_op result}}
+    %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+      : (!transform.any_op) -> (f32)
+
+    transform.yield %res : f32
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg1: !transform.any_op {transform.readonly}, %arg2: !transform.any_op {transform.readonly})
+        -> (!transform.any_op) {
+        transform.yield %arg1 : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+  }
+
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+    -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    // expected-error @+1 {{ForeachMatchOp must take exactly one any_op argument}}
```

**Comment:**
Also here: we should either tighten the verifier on `transform.foreach_match` or say which attribute adds this verification rule

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:290`

```diff
@@ -66,3 +66,267 @@ module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_defa
     return
   }
 }
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  // expected-error @+1{{The named sequence '__kernel_config' must have the attribute 'iree_codegen.tuning_spec_entrypoint'}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) {
+      transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match_a(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @match_b(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+      transform.yield
+  }
+
+  // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'ForeachMatchOp', but found 2}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+
+      %res_a = transform.foreach_match in %arg0 @match_a -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      %res_b = transform.foreach_match in %res_a @match_b -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      transform.yield %res_b : !transform.any_op
+  }
+}
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+
+  transform.named_sequence @main(%arg0: !transform.any_op {transform.readonly})
+      -> !transform.any_op attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'transform::YieldOp', but found 2}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+        %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+
+  module @extra_module attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence private @dummy_func(!transform.any_op {transform.consumed}) -> !transform.any_op
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_configbut found an unsupported operation: transform.include}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+        %tmp = transform.include @dummy_func failures(suppress) (%arg0) : (!transform.any_op) -> (!transform.any_op)
+        %res = transform.foreach_match in %tmp @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence private @dummy_func(!transform.any_op {transform.consumed}) -> !transform.any_op
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_configbut found an unsupported operation: transform.print}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+         transform.print {name = "Hello"}
+        %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg: !transform.any_op {transform.readonly})
+    -> (!transform.any_op, !transform.any_op) {
+    transform.yield %arg, %arg : !transform.any_op, !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op1: !transform.any_op {transform.readonly}, %op2: !transform.any_op {transform.readonly})
+    -> (!transform.any_op) {
+    transform.yield %op1 : !transform.any_op
+  }
+
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+    -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    // expected-error @+1 {{ForeachMatchOp must return exactly one any_op result}}
+    %res1, %res2 = transform.foreach_match in %arg0 @match -> @apply_op_config
+      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
+
+    transform.yield %res1 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg: !transform.any_op) -> (!transform.any_op) {
+    transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op) {
+    transform.yield
+  }
+
+  // expected-error @+1 {{Tuning spec entry point expected to return any_op}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op)
+      -> (f32) attributes { iree_codegen.tuning_spec_entrypoint } {
+     // expected-error @+1 {{ForeachMatchOp must return exactly one any_op result}}
+    %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+      : (!transform.any_op) -> (f32)
+
+    transform.yield %res : f32
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg1: !transform.any_op {transform.readonly}, %arg2: !transform.any_op {transform.readonly})
+        -> (!transform.any_op) {
+        transform.yield %arg1 : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+  }
+
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+    -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    // expected-error @+1 {{ForeachMatchOp must take exactly one any_op argument}}
+    %res = transform.foreach_match in %arg0, %arg0 @match -> @apply_op_config
+    : (!transform.any_op, !transform.any_op) -> (!transform.any_op)
+
+    transform.yield %res : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg: index) -> (index) {
+    transform.yield %arg : index
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op) {
+    transform.yield
+  }
+
+  transform.named_sequence @__kernel_config(%arg0: index)
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    // expected-error @+1 {{ForeachMatchOp must take exactly one any_op argument}}
+    %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+      : (index) -> (!transform.any_op)
```

**Comment:**
Also here

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:305`

```diff
@@ -66,3 +66,267 @@ module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_defa
     return
   }
 }
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  // expected-error @+1{{The named sequence '__kernel_config' must have the attribute 'iree_codegen.tuning_spec_entrypoint'}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) {
+      transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match_a(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @match_b(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+      transform.yield
+  }
+
+  // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'ForeachMatchOp', but found 2}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+
+      %res_a = transform.foreach_match in %arg0 @match_a -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      %res_b = transform.foreach_match in %res_a @match_b -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      transform.yield %res_b : !transform.any_op
+  }
+}
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+
+  transform.named_sequence @main(%arg0: !transform.any_op {transform.readonly})
+      -> !transform.any_op attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'transform::YieldOp', but found 2}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+        %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+
+  module @extra_module attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence private @dummy_func(!transform.any_op {transform.consumed}) -> !transform.any_op
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_configbut found an unsupported operation: transform.include}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+        %tmp = transform.include @dummy_func failures(suppress) (%arg0) : (!transform.any_op) -> (!transform.any_op)
+        %res = transform.foreach_match in %tmp @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence private @dummy_func(!transform.any_op {transform.consumed}) -> !transform.any_op
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_configbut found an unsupported operation: transform.print}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+         transform.print {name = "Hello"}
+        %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg: !transform.any_op {transform.readonly})
+    -> (!transform.any_op, !transform.any_op) {
+    transform.yield %arg, %arg : !transform.any_op, !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op1: !transform.any_op {transform.readonly}, %op2: !transform.any_op {transform.readonly})
+    -> (!transform.any_op) {
+    transform.yield %op1 : !transform.any_op
+  }
+
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+    -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    // expected-error @+1 {{ForeachMatchOp must return exactly one any_op result}}
+    %res1, %res2 = transform.foreach_match in %arg0 @match -> @apply_op_config
+      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
+
+    transform.yield %res1 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg: !transform.any_op) -> (!transform.any_op) {
+    transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op) {
+    transform.yield
+  }
+
+  // expected-error @+1 {{Tuning spec entry point expected to return any_op}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op)
+      -> (f32) attributes { iree_codegen.tuning_spec_entrypoint } {
+     // expected-error @+1 {{ForeachMatchOp must return exactly one any_op result}}
+    %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+      : (!transform.any_op) -> (f32)
+
+    transform.yield %res : f32
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg1: !transform.any_op {transform.readonly}, %arg2: !transform.any_op {transform.readonly})
+        -> (!transform.any_op) {
+        transform.yield %arg1 : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+  }
+
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+    -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    // expected-error @+1 {{ForeachMatchOp must take exactly one any_op argument}}
+    %res = transform.foreach_match in %arg0, %arg0 @match -> @apply_op_config
+    : (!transform.any_op, !transform.any_op) -> (!transform.any_op)
+
+    transform.yield %res : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg: index) -> (index) {
+    transform.yield %arg : index
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op) {
+    transform.yield
+  }
+
+  transform.named_sequence @__kernel_config(%arg0: index)
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    // expected-error @+1 {{ForeachMatchOp must take exactly one any_op argument}}
+    %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+      : (index) -> (!transform.any_op)
+
+    transform.yield %res : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg: !transform.any_op) -> (!transform.any_op) {
+    transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op) {
+    transform.yield
+  }
+
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op)
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+     // expected-error @+1{{ForeachMatchOp must not have the 'restrict_root' attribute}}
```

**Comment:**
Print which attribute adds this rule

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:326`

```diff
@@ -66,3 +66,267 @@ module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_defa
     return
   }
 }
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  // expected-error @+1{{The named sequence '__kernel_config' must have the attribute 'iree_codegen.tuning_spec_entrypoint'}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) {
+      transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match_a(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @match_b(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+      transform.yield
+  }
+
+  // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'ForeachMatchOp', but found 2}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+
+      %res_a = transform.foreach_match in %arg0 @match_a -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      %res_b = transform.foreach_match in %res_a @match_b -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+      transform.yield %res_b : !transform.any_op
+  }
+}
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+
+  transform.named_sequence @main(%arg0: !transform.any_op {transform.readonly})
+      -> !transform.any_op attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_config' must contain exactly one 'transform::YieldOp', but found 2}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+        %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+// expected-error @+1{{Expected exactly one NamedSequenceOp with the attribute 'iree_codegen.tuning_spec_entrypoint', but found 2}}
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+
+  module @extra_module attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence private @dummy_func(!transform.any_op {transform.consumed}) -> !transform.any_op
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_configbut found an unsupported operation: transform.include}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+        %tmp = transform.include @dummy_func failures(suppress) (%arg0) : (!transform.any_op) -> (!transform.any_op)
+        %res = transform.foreach_match in %tmp @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence private @dummy_func(!transform.any_op {transform.consumed}) -> !transform.any_op
+    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+      transform.yield %arg : !transform.any_op
+    }
+
+    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+    }
+
+    // expected-error @+1{{The named sequence '__kernel_configbut found an unsupported operation: transform.print}}
+    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+        -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+         transform.print {name = "Hello"}
+        %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+        : (!transform.any_op) -> (!transform.any_op)
+
+        transform.yield %res : !transform.any_op
+    }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg: !transform.any_op {transform.readonly})
+    -> (!transform.any_op, !transform.any_op) {
+    transform.yield %arg, %arg : !transform.any_op, !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op1: !transform.any_op {transform.readonly}, %op2: !transform.any_op {transform.readonly})
+    -> (!transform.any_op) {
+    transform.yield %op1 : !transform.any_op
+  }
+
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+    -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    // expected-error @+1 {{ForeachMatchOp must return exactly one any_op result}}
+    %res1, %res2 = transform.foreach_match in %arg0 @match -> @apply_op_config
+      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
+
+    transform.yield %res1 : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg: !transform.any_op) -> (!transform.any_op) {
+    transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op) {
+    transform.yield
+  }
+
+  // expected-error @+1 {{Tuning spec entry point expected to return any_op}}
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op)
+      -> (f32) attributes { iree_codegen.tuning_spec_entrypoint } {
+     // expected-error @+1 {{ForeachMatchOp must return exactly one any_op result}}
+    %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+      : (!transform.any_op) -> (f32)
+
+    transform.yield %res : f32
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg1: !transform.any_op {transform.readonly}, %arg2: !transform.any_op {transform.readonly})
+        -> (!transform.any_op) {
+        transform.yield %arg1 : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
+        transform.yield
+  }
+
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
+    -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    // expected-error @+1 {{ForeachMatchOp must take exactly one any_op argument}}
+    %res = transform.foreach_match in %arg0, %arg0 @match -> @apply_op_config
+    : (!transform.any_op, !transform.any_op) -> (!transform.any_op)
+
+    transform.yield %res : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg: index) -> (index) {
+    transform.yield %arg : index
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op) {
+    transform.yield
+  }
+
+  transform.named_sequence @__kernel_config(%arg0: index)
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+    // expected-error @+1 {{ForeachMatchOp must take exactly one any_op argument}}
+    %res = transform.foreach_match in %arg0 @match -> @apply_op_config
+      : (index) -> (!transform.any_op)
+
+    transform.yield %res : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg: !transform.any_op) -> (!transform.any_op) {
+    transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op) {
+    transform.yield
+  }
+
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op)
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+     // expected-error @+1{{ForeachMatchOp must not have the 'restrict_root' attribute}}
+    %res = transform.foreach_match restrict_root in %arg0 @match -> @apply_op_config
+      : (!transform.any_op) -> (!transform.any_op)
+
+    transform.yield %res : !transform.any_op
+  }
+}
+
+// -----
+
+module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
+  transform.named_sequence @match(%arg: !transform.any_op) -> (!transform.any_op) {
+    transform.yield %arg : !transform.any_op
+  }
+
+  transform.named_sequence @apply_op_config(%op: !transform.any_op) {
+    transform.yield
+  }
+
+  transform.named_sequence @__kernel_config(%arg0: !transform.any_op)
+      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
+     // expected-error @+1{{ForeachMatchOp must not have the 'flatten_results' attribute}}
```

**Comment:**
Also here

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp:134`

```diff
@@ -58,28 +58,130 @@ IREECodegenDialect::verifyOperationAttribute(Operation *op,
   StringRef symbol = attribute.getName().strref();
   Attribute attr = attribute.getValue();
   // This function verifies the validity of a specific operation attribute.
-  // - If the attribute's name matches `kTuningDefaultSpecAttrName`, make
-  //   sure it contains a single named sequence op with name `__kernel_config`.
+  // - If the attribute's name matches kTuningSpecDefaultEntrypointAttrName
+  // (`iree_codegen.tuning_spec_with_default_entrypoint`):
+  //   1. Ensure that the module contains a single named sequence operation with
+  //   the name `__kernel_config`.
+  //   2. Verify that this `__kernel_config` named sequence operation has the
+  //   attribute `iree_codegen.tuning_spec_entrypoint`.
+  //   3. Ensure that the named sequence operation contains exactly **one**
+  //   `ForeachMatchOp`.
+  //      - ForeachMatchOp must not have `flatten_results` and `restrict_root`
+  //        attributes.
+  //      - ForeachMatchOp must have exactly one argument of type any_op.
+  //      - ForeachMatchOp must have exactly one result of type any_op.
+  //   4. Ensure that only one named sequence operation with the
+  //   `iree_codegen.tuning_spec_entrypoint` attribute.
   // - If the attribute's name matches `kTuningSpecEntrypointAttrName`
-  // ("iree_codegen.tuning_spec_entrypoint"):
+  // (`iree_codegen.tuning_spec_entrypoint`):
   //   1. The attribute value must be a UnitAttr.
   //   2. If the operation is a transform::NamedSequenceOp:
   //      - The operation's function signature must satisfy the following:
-  //         a. It must have exactly one result type, and the result must be of
-  //         type `transform::AnyOpType`.
+  //         a. It must have exactly one result type, and the result must be
+  //         of type `transform::AnyOpType`.
   //         b. It must have exactly one argument type, and the argument must be
   //         of type `transform::AnyOpType`.
 
   if (symbol == kTuningSpecDefaultEntrypointAttrName) {
     if (auto moduleOp = dyn_cast<ModuleOp>(op)) {
-      if (!llvm::any_of(moduleOp.getOps<transform::NamedSequenceOp>(),
-                        [](transform::NamedSequenceOp op) {
-                          return op.getName() == kKernelConfigSpecName;
-                        })) {
+      transform::NamedSequenceOp kernelConfigOp;
+      int numTuningEntryPoints = 0;
+      for (Region &region : moduleOp->getRegions()) {
+        for (Block &block : region) {
+          for (Operation &op : block) {
+            if (auto namedSeqOp = dyn_cast<transform::NamedSequenceOp>(&op)) {
+              if (namedSeqOp.getName() == kKernelConfigSpecName) {
+                kernelConfigOp = namedSeqOp;
+              }
+            }
+
+            if (op.hasAttr(kTuningSpecEntrypointAttrName)) {
+              ++numTuningEntryPoints;
+            }
+          }
+        }
+      }
+
+      if (!kernelConfigOp) {
+        return moduleOp->emitError()
+               << "The tuning specification must include a named sequence with "
+               << "the symbol name '" << kKernelConfigSpecName << "'.";
+      }
+
+      // Verify that the kernelConfigOp has the attribute
+      // `iree_codegen.tuning_spec_entrypoint`.
+      if (!kernelConfigOp->hasAttr(kTuningSpecEntrypointAttrName)) {
+        return kernelConfigOp.emitError()
+               << "The named sequence '" << kKernelConfigSpecName
+               << "' must have the attribute '" << kTuningSpecEntrypointAttrName
+               << "'.";
+      }
+
+      if (numTuningEntryPoints != 1) {
         return moduleOp.emitError()
-               << "The tuning specification must include a named "
-                  "sequence with the symbol name '"
-               << kKernelConfigSpecName << "'.";
+               << "Expected exactly one NamedSequenceOp with the attribute '"
+               << kTuningSpecEntrypointAttrName << "', but found "
+               << numTuningEntryPoints << ".";
+      }
+
+      transform::ForeachMatchOp foreachMatchOp;
+      int numForeachMatchOps = 0;
+      int numYieldOps = 0;
+
+      for (Block &block : kernelConfigOp.getBlocks()) {
+        for (Operation &op : block) {
+          if (auto foreachOp = dyn_cast<transform::ForeachMatchOp>(op)) {
+            numForeachMatchOps++;
```

**Comment:**
https://llvm.org/docs/CodingStandards.html#prefer-preincrement

---


---


## [PR #20127](https://github.com/iree-org/iree/pull/20127): [Codegen][Tuner] merge the default td specs

### Review Summary

**CHANGES_REQUESTED** (2025-03-03)

To thoroughly test this code, we can start by assuming that tuning specs with default entypoints should always link, and then emitting warning whenever we notice they can't be linked. This is something we can test with `verify-diagnostics`.

 The way we usually do this in compilers is that we try to have two phases:
1. Analysis that determines transformation legality
2. Transformation that cannot bail out


**CHANGES_REQUESTED** (2025-03-07)


**COMMENTED** (2025-03-07)


**CHANGES_REQUESTED** (2025-03-10)


**COMMENTED** (2025-03-17)


**COMMENTED** (2025-03-17)


**COMMENTED** (2025-03-17)

Also, this PR doesn't actually fix https://github.com/nod-ai/shark-ai/issues/810 -- we also need to emit merged tuning specs in the tuner to close this issue


**COMMENTED** (2025-03-18)


**COMMENTED** (2025-03-18)


**COMMENTED** (2025-03-18)


**COMMENTED** (2025-03-18)


**COMMENTED** (2025-03-18)


### Code Comments

**File:** `compiler/plugins/target/ROCM/test/default_tuning_specs_amdgpu.mlir:49`

```diff
@@ -32,21 +32,22 @@
 
 // ============================================================================
 
-// Check that both the user tuning spec and the default spec get linked and
-// materialized. The user spec should have precedence over the default one.
+// Check that both the user tuning spec and the default spec get merged and
+// materialized, in which nested structure should not present and merged foreach_match op
+// should exist. The user spec should have precedence over the default one.
 
 // BOTH-LABEL: module @iree_linked_tuning_spec
 // BOTH-SAME:    iree_codegen.tuning_spec_with_default_entrypoint
 // BOTH-SAME:    transform.with_named_sequence
-// BOTH-LABEL:   module @mmt_tile_and_fuse_spec_0 attributes {transform.with_named_sequence}
-// BOTH-LABEL:     transform.named_sequence @main
-// BOTH-SAME:        attributes {iree_codegen.tuning_spec_entrypoint}
-// BOTH-LABEL:   module @iree_default_tuning_spec_gfx942_1 attributes {transform.with_named_sequence}
-// BOTH:           transform.named_sequence @__kernel_config
-// BOTH-SAME:        attributes {iree_codegen.tuning_spec_entrypoint}
+// BOTH-NOT:     module @mmt_tile_and_fuse_spec
+// BOTH-NOT:     module @iree_default_tuning_spec_gfx942
 // BOTH:         transform.named_sequence @__kernel_config
-// BOTH:           @mmt_tile_and_fuse_spec_0::@main
-// BOTH:           @iree_default_tuning_spec_gfx942_1::@__kernel_config
+// BOTH-SAME:    attributes {iree_codegen.tuning_spec_entrypoint}
+// BOTH:         transform.foreach_match
+// BOTH:         @match_mmt -> @apply_mmt_op_config
+// BOTH-NEXT:    @match_attention_2x10x4096x64x64x64_f16 -> @apply_attn_op_config
+// BOTH-NEXT:    @match_mmt_2048x1280x5120_f16_f16_f32 -> @apply_op_config
```

**Comment:**
nit: indent these to match the print format
```suggestion
// BOTH:           @match_mmt -> @apply_mmt_op_config
// BOTH-NEXT:      @match_attention_2x10x4096x64x64x64_f16 -> @apply_attn_op_config
// BOTH-NEXT:      @match_mmt_2048x1280x5120_f16_f16_f32 -> @apply_op_config
```

---

**File:** `compiler/plugins/target/ROCM/test/default_tuning_specs_amdgpu.mlir:49`

```diff
@@ -32,21 +32,22 @@
 
 // ============================================================================
 
-// Check that both the user tuning spec and the default spec get linked and
-// materialized. The user spec should have precedence over the default one.
+// Check that both the user tuning spec and the default spec get merged and
+// materialized, in which nested structure should not present and merged foreach_match op
+// should exist. The user spec should have precedence over the default one.
 
 // BOTH-LABEL: module @iree_linked_tuning_spec
 // BOTH-SAME:    iree_codegen.tuning_spec_with_default_entrypoint
 // BOTH-SAME:    transform.with_named_sequence
-// BOTH-LABEL:   module @mmt_tile_and_fuse_spec_0 attributes {transform.with_named_sequence}
-// BOTH-LABEL:     transform.named_sequence @main
-// BOTH-SAME:        attributes {iree_codegen.tuning_spec_entrypoint}
-// BOTH-LABEL:   module @iree_default_tuning_spec_gfx942_1 attributes {transform.with_named_sequence}
-// BOTH:           transform.named_sequence @__kernel_config
-// BOTH-SAME:        attributes {iree_codegen.tuning_spec_entrypoint}
+// BOTH-NOT:     module @mmt_tile_and_fuse_spec
+// BOTH-NOT:     module @iree_default_tuning_spec_gfx942
 // BOTH:         transform.named_sequence @__kernel_config
-// BOTH:           @mmt_tile_and_fuse_spec_0::@main
-// BOTH:           @iree_default_tuning_spec_gfx942_1::@__kernel_config
+// BOTH-SAME:    attributes {iree_codegen.tuning_spec_entrypoint}
+// BOTH:         transform.foreach_match
+// BOTH:         @match_mmt -> @apply_op_config
+// BOTH-NEXT:    @match_attention_2x10x4096x64x64x64_f16 -> @apply_attn_op_config
+// BOTH-NEXT:    @match_mmt_2048x1280x5120_f16_f16_f32 -> @apply_op_config_1
```

**Comment:**
nit: indent this like the printer does
```suggestion
// BOTH:         @match_mmt -> @apply_op_config
// BOTH-NEXT:    @match_attention_2x10x4096x64x64x64_f16 -> @apply_attn_op_config
// BOTH-NEXT:    @match_mmt_2048x1280x5120_f16_f16_f32 -> @apply_op_config_1
```
```suggestion
// BOTH:           @match_mmt -> @apply_op_config
// BOTH-NEXT:      @match_attention_2x10x4096x64x64x64_f16 -> @apply_attn_op_config
// BOTH-NEXT:      @match_mmt_2048x1280x5120_f16_f16_f32 -> @apply_op_config_1
```

---

**File:** `compiler/plugins/target/ROCM/test/tuning_spec_mmt_tile_and_fuse.mlir:4`

```diff
@@ -1,6 +1,6 @@
 // RUN: iree-opt %s
 
-module @mmt_tile_and_fuse_spec attributes { transform.with_named_sequence } {
+module @mmt_tile_and_fuse_spec attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
 transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly},
```

**Comment:**
Why are we changing this? Do you think we should disallow tuning specs without default entrypoints?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:90`

```diff
@@ -85,6 +85,14 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   newSpec->setAttr(kTuningSpecEntrypointAttrName, builder.getUnitAttr());
   module->setAttr(kTuningSpecDefaultEntrypointAttrName, builder.getUnitAttr());
 
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
+    // Remove the default tuning spec attribute from inner modules,
+    // as the top-level module is attached with default attribute.
```

**Comment:**
Do we remove it because the entrypoint name can change? If that's the case, I think we'd want to iterate over the parents of `specsToLink` instead, since this functions doesn't assume a specific nesting structure.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:405`

```diff
@@ -154,6 +398,27 @@ struct LinkTuningSpecsPass final
 FailureOr<NamedSequenceOp> linkTuningSpecs(ModuleOp module) {
   SmallVector<NamedSequenceOp> tuningSpecs;
 
+  int matchingModules = 0;
+  int totalModules = 0;
+
+  for (auto module : module.getBody()->getOps<ModuleOp>()) {
+    totalModules++;
```

**Comment:**
Prefer pre-increment: https://llvm.org/docs/CodingStandards.html#prefer-preincrement

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:407`

```diff
@@ -154,6 +398,27 @@ struct LinkTuningSpecsPass final
 FailureOr<NamedSequenceOp> linkTuningSpecs(ModuleOp module) {
   SmallVector<NamedSequenceOp> tuningSpecs;
 
+  int matchingModules = 0;
+  int totalModules = 0;
+
+  for (auto module : module.getBody()->getOps<ModuleOp>()) {
+    totalModules++;
+    if (module->hasAttr(kTuningSpecDefaultEntrypointAttrName)) {
+      matchingModules++;
```

**Comment:**
also here

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:401`

```diff
@@ -154,6 +398,27 @@ struct LinkTuningSpecsPass final
 FailureOr<NamedSequenceOp> linkTuningSpecs(ModuleOp module) {
   SmallVector<NamedSequenceOp> tuningSpecs;
 
+  int matchingModules = 0;
```

**Comment:**
I don't understand this variable name. What do these modules match?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:402`

```diff
@@ -154,6 +398,27 @@ struct LinkTuningSpecsPass final
 FailureOr<NamedSequenceOp> linkTuningSpecs(ModuleOp module) {
   SmallVector<NamedSequenceOp> tuningSpecs;
 
+  int matchingModules = 0;
+  int totalModules = 0;
```

**Comment:**
I'd put this variable first since you update it unconditionally in the loop below.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:401`

```diff
@@ -154,6 +398,27 @@ struct LinkTuningSpecsPass final
 FailureOr<NamedSequenceOp> linkTuningSpecs(ModuleOp module) {
   SmallVector<NamedSequenceOp> tuningSpecs;
 
+  int matchingModules = 0;
```

**Comment:**
Maybe `numDefaultEntrypoint`?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:152`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
```

**Comment:**
Can we call the variable something like `namedSequenceToForeach` so that we don't need this comment?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:250`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
+  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
+  // ForeachMatchOp is also updated.
+  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
+      foreachMatchMap;
+
+  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
```

**Comment:**
This function is very long -- can we outline this loop to a helper function?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:163`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
+  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
+  // ForeachMatchOp is also updated.
+  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
+      foreachMatchMap;
+
+  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
+    for (auto namedSequenceOp :
+         innerModule.getBody()->getOps<transform::NamedSequenceOp>()) {
+      if (namedSequenceOp.getSymName() == kKernelConfigSpecName) {
+        transform::ForeachMatchOp foreachMatch = nullptr;
```

**Comment:**
You don't need to explicitly initialize IR types to nullptr.
```suggestion
        transform::ForeachMatchOp foreachMatch;
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:164`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
+  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
+  // ForeachMatchOp is also updated.
+  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
+      foreachMatchMap;
+
+  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
+    for (auto namedSequenceOp :
+         innerModule.getBody()->getOps<transform::NamedSequenceOp>()) {
+      if (namedSequenceOp.getSymName() == kKernelConfigSpecName) {
+        transform::ForeachMatchOp foreachMatch = nullptr;
+        int matchCount = 0;
```

**Comment:**
This needs a more descriptive name

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:170`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
+  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
+  // ForeachMatchOp is also updated.
+  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
+      foreachMatchMap;
+
+  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
+    for (auto namedSequenceOp :
+         innerModule.getBody()->getOps<transform::NamedSequenceOp>()) {
+      if (namedSequenceOp.getSymName() == kKernelConfigSpecName) {
+        transform::ForeachMatchOp foreachMatch = nullptr;
+        int matchCount = 0;
+        // Iterate directly over ForeachMatchOp within kernelConfig.
+        for (auto op : namedSequenceOp.getOps<transform::ForeachMatchOp>()) {
+          if (!foreachMatch) {
+            foreachMatch = op;
+          }
+          matchCount++;
```

**Comment:**
also here

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:179`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
+  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
+  // ForeachMatchOp is also updated.
+  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
+      foreachMatchMap;
+
+  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
+    for (auto namedSequenceOp :
+         innerModule.getBody()->getOps<transform::NamedSequenceOp>()) {
+      if (namedSequenceOp.getSymName() == kKernelConfigSpecName) {
+        transform::ForeachMatchOp foreachMatch = nullptr;
+        int matchCount = 0;
+        // Iterate directly over ForeachMatchOp within kernelConfig.
+        for (auto op : namedSequenceOp.getOps<transform::ForeachMatchOp>()) {
+          if (!foreachMatch) {
+            foreachMatch = op;
+          }
+          matchCount++;
+        }
+
+        // Return failure if multiple occurrences exist.
+        if (matchCount > 1) {
+          return failure();
+        }
+        // Return failure if not foreach match op found.
+        if (!foreachMatch)
+          return failure();
```

**Comment:**
```suggestion
        if (matchCount == 0 || matchCount > 1) {
          return failure();
        }
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:188`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
+  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
+  // ForeachMatchOp is also updated.
+  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
+      foreachMatchMap;
+
+  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
+    for (auto namedSequenceOp :
+         innerModule.getBody()->getOps<transform::NamedSequenceOp>()) {
+      if (namedSequenceOp.getSymName() == kKernelConfigSpecName) {
+        transform::ForeachMatchOp foreachMatch = nullptr;
+        int matchCount = 0;
+        // Iterate directly over ForeachMatchOp within kernelConfig.
+        for (auto op : namedSequenceOp.getOps<transform::ForeachMatchOp>()) {
+          if (!foreachMatch) {
+            foreachMatch = op;
+          }
+          matchCount++;
+        }
+
+        // Return failure if multiple occurrences exist.
+        if (matchCount > 1) {
+          return failure();
+        }
+        // Return failure if not foreach match op found.
+        if (!foreachMatch)
+          return failure();
+
+        foreachMatchOps.push_back(foreachMatch);
+
+        for (auto matcher : foreachMatch.getMatchers()) {
+          if (auto matcherSymRef = dyn_cast<SymbolRefAttr>(matcher)) {
+            if (auto matcherOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         matcherSymRef))) {
+              if (!foreachMatchMap.count(matcherOp)) {
```

**Comment:**
Use `.contains(...)` for map types

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:199`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
+  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
+  // ForeachMatchOp is also updated.
+  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
+      foreachMatchMap;
+
+  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
+    for (auto namedSequenceOp :
+         innerModule.getBody()->getOps<transform::NamedSequenceOp>()) {
+      if (namedSequenceOp.getSymName() == kKernelConfigSpecName) {
+        transform::ForeachMatchOp foreachMatch = nullptr;
+        int matchCount = 0;
+        // Iterate directly over ForeachMatchOp within kernelConfig.
+        for (auto op : namedSequenceOp.getOps<transform::ForeachMatchOp>()) {
+          if (!foreachMatch) {
+            foreachMatch = op;
+          }
+          matchCount++;
+        }
+
+        // Return failure if multiple occurrences exist.
+        if (matchCount > 1) {
+          return failure();
+        }
+        // Return failure if not foreach match op found.
+        if (!foreachMatch)
+          return failure();
+
+        foreachMatchOps.push_back(foreachMatch);
+
+        for (auto matcher : foreachMatch.getMatchers()) {
+          if (auto matcherSymRef = dyn_cast<SymbolRefAttr>(matcher)) {
+            if (auto matcherOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         matcherSymRef))) {
+              if (!foreachMatchMap.count(matcherOp)) {
+                foreachMatchMap[matcherOp] = foreachMatch;
+              }
+            }
+          }
+        }
+        for (auto action : foreachMatch.getActions()) {
+          if (auto actionSymRef = dyn_cast<SymbolRefAttr>(action)) {
+            if (auto actionOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         actionSymRef))) {
+              if (!foreachMatchMap.count(actionOp)) {
```

**Comment:**
also here

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:187`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
+  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
+  // ForeachMatchOp is also updated.
+  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
+      foreachMatchMap;
+
+  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
+    for (auto namedSequenceOp :
+         innerModule.getBody()->getOps<transform::NamedSequenceOp>()) {
+      if (namedSequenceOp.getSymName() == kKernelConfigSpecName) {
+        transform::ForeachMatchOp foreachMatch = nullptr;
+        int matchCount = 0;
+        // Iterate directly over ForeachMatchOp within kernelConfig.
+        for (auto op : namedSequenceOp.getOps<transform::ForeachMatchOp>()) {
+          if (!foreachMatch) {
+            foreachMatch = op;
+          }
+          matchCount++;
+        }
+
+        // Return failure if multiple occurrences exist.
+        if (matchCount > 1) {
+          return failure();
+        }
+        // Return failure if not foreach match op found.
+        if (!foreachMatch)
+          return failure();
+
+        foreachMatchOps.push_back(foreachMatch);
+
+        for (auto matcher : foreachMatch.getMatchers()) {
+          if (auto matcherSymRef = dyn_cast<SymbolRefAttr>(matcher)) {
+            if (auto matcherOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         matcherSymRef))) {
```

**Comment:**
Can this ever not be a named sequence op?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:214`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
+  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
+  // ForeachMatchOp is also updated.
+  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
+      foreachMatchMap;
+
+  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
+    for (auto namedSequenceOp :
+         innerModule.getBody()->getOps<transform::NamedSequenceOp>()) {
+      if (namedSequenceOp.getSymName() == kKernelConfigSpecName) {
+        transform::ForeachMatchOp foreachMatch = nullptr;
+        int matchCount = 0;
+        // Iterate directly over ForeachMatchOp within kernelConfig.
+        for (auto op : namedSequenceOp.getOps<transform::ForeachMatchOp>()) {
+          if (!foreachMatch) {
+            foreachMatch = op;
+          }
+          matchCount++;
+        }
+
+        // Return failure if multiple occurrences exist.
+        if (matchCount > 1) {
+          return failure();
+        }
+        // Return failure if not foreach match op found.
+        if (!foreachMatch)
+          return failure();
+
+        foreachMatchOps.push_back(foreachMatch);
+
+        for (auto matcher : foreachMatch.getMatchers()) {
+          if (auto matcherSymRef = dyn_cast<SymbolRefAttr>(matcher)) {
+            if (auto matcherOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         matcherSymRef))) {
+              if (!foreachMatchMap.count(matcherOp)) {
+                foreachMatchMap[matcherOp] = foreachMatch;
+              }
+            }
+          }
+        }
+        for (auto action : foreachMatch.getActions()) {
+          if (auto actionSymRef = dyn_cast<SymbolRefAttr>(action)) {
+            if (auto actionOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         actionSymRef))) {
+              if (!foreachMatchMap.count(actionOp)) {
+                foreachMatchMap[actionOp] = foreachMatch;
+              }
+            }
+          }
+        }
+      } else {
+        namedSequenceOpsToMove.push_back(namedSequenceOp);
+      }
+    }
+  }
+
+  // Step 2-a: Ensure all ForeachMatchOps have the same result types before
+  // merging.
+  SmallVector<Type, 4> expectedResultTypes =
+      llvm::to_vector<4>(foreachMatchOps.front()->getResultTypes());
```

**Comment:**
```suggestion
  auto expectedResultTypes =
      llvm::to_vector_of<Type, 4>(foreachMatchOps.front()->getResultTypes());
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:257`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
+  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
+  // ForeachMatchOp is also updated.
+  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
+      foreachMatchMap;
+
+  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
+    for (auto namedSequenceOp :
+         innerModule.getBody()->getOps<transform::NamedSequenceOp>()) {
+      if (namedSequenceOp.getSymName() == kKernelConfigSpecName) {
+        transform::ForeachMatchOp foreachMatch = nullptr;
+        int matchCount = 0;
+        // Iterate directly over ForeachMatchOp within kernelConfig.
+        for (auto op : namedSequenceOp.getOps<transform::ForeachMatchOp>()) {
+          if (!foreachMatch) {
+            foreachMatch = op;
+          }
+          matchCount++;
+        }
+
+        // Return failure if multiple occurrences exist.
+        if (matchCount > 1) {
+          return failure();
+        }
+        // Return failure if not foreach match op found.
+        if (!foreachMatch)
+          return failure();
+
+        foreachMatchOps.push_back(foreachMatch);
+
+        for (auto matcher : foreachMatch.getMatchers()) {
+          if (auto matcherSymRef = dyn_cast<SymbolRefAttr>(matcher)) {
+            if (auto matcherOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         matcherSymRef))) {
+              if (!foreachMatchMap.count(matcherOp)) {
+                foreachMatchMap[matcherOp] = foreachMatch;
+              }
+            }
+          }
+        }
+        for (auto action : foreachMatch.getActions()) {
+          if (auto actionSymRef = dyn_cast<SymbolRefAttr>(action)) {
+            if (auto actionOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         actionSymRef))) {
+              if (!foreachMatchMap.count(actionOp)) {
+                foreachMatchMap[actionOp] = foreachMatch;
+              }
+            }
+          }
+        }
+      } else {
+        namedSequenceOpsToMove.push_back(namedSequenceOp);
+      }
+    }
+  }
+
+  // Step 2-a: Ensure all ForeachMatchOps have the same result types before
+  // merging.
```

**Comment:**
This could also be a helper function

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:262`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
+  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
+  // ForeachMatchOp is also updated.
+  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
+      foreachMatchMap;
+
+  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
+    for (auto namedSequenceOp :
+         innerModule.getBody()->getOps<transform::NamedSequenceOp>()) {
+      if (namedSequenceOp.getSymName() == kKernelConfigSpecName) {
+        transform::ForeachMatchOp foreachMatch = nullptr;
+        int matchCount = 0;
+        // Iterate directly over ForeachMatchOp within kernelConfig.
+        for (auto op : namedSequenceOp.getOps<transform::ForeachMatchOp>()) {
+          if (!foreachMatch) {
+            foreachMatch = op;
+          }
+          matchCount++;
+        }
+
+        // Return failure if multiple occurrences exist.
+        if (matchCount > 1) {
+          return failure();
+        }
+        // Return failure if not foreach match op found.
+        if (!foreachMatch)
+          return failure();
+
+        foreachMatchOps.push_back(foreachMatch);
+
+        for (auto matcher : foreachMatch.getMatchers()) {
+          if (auto matcherSymRef = dyn_cast<SymbolRefAttr>(matcher)) {
+            if (auto matcherOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         matcherSymRef))) {
+              if (!foreachMatchMap.count(matcherOp)) {
+                foreachMatchMap[matcherOp] = foreachMatch;
+              }
+            }
+          }
+        }
+        for (auto action : foreachMatch.getActions()) {
+          if (auto actionSymRef = dyn_cast<SymbolRefAttr>(action)) {
+            if (auto actionOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         actionSymRef))) {
+              if (!foreachMatchMap.count(actionOp)) {
+                foreachMatchMap[actionOp] = foreachMatch;
+              }
+            }
+          }
+        }
+      } else {
+        namedSequenceOpsToMove.push_back(namedSequenceOp);
+      }
+    }
+  }
+
+  // Step 2-a: Ensure all ForeachMatchOps have the same result types before
+  // merging.
+  SmallVector<Type, 4> expectedResultTypes =
+      llvm::to_vector<4>(foreachMatchOps.front()->getResultTypes());
+
+  for (auto foreachMatchOp : foreachMatchOps) {
+    SmallVector<Type, 4> currentResultTypes =
+        llvm::to_vector<4>(foreachMatchOp.getResultTypes());
+
+    if (!llvm::equal(currentResultTypes, expectedResultTypes)) {
+      return failure();
+    }
```

**Comment:**
Instead, should we check that the result type is exactly what we expect? I think it must take any_op and return any_op.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:228`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
+  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
+  // ForeachMatchOp is also updated.
+  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
+      foreachMatchMap;
+
+  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
+    for (auto namedSequenceOp :
+         innerModule.getBody()->getOps<transform::NamedSequenceOp>()) {
+      if (namedSequenceOp.getSymName() == kKernelConfigSpecName) {
+        transform::ForeachMatchOp foreachMatch = nullptr;
+        int matchCount = 0;
+        // Iterate directly over ForeachMatchOp within kernelConfig.
+        for (auto op : namedSequenceOp.getOps<transform::ForeachMatchOp>()) {
+          if (!foreachMatch) {
+            foreachMatch = op;
+          }
+          matchCount++;
+        }
+
+        // Return failure if multiple occurrences exist.
+        if (matchCount > 1) {
+          return failure();
+        }
+        // Return failure if not foreach match op found.
+        if (!foreachMatch)
+          return failure();
+
+        foreachMatchOps.push_back(foreachMatch);
+
+        for (auto matcher : foreachMatch.getMatchers()) {
+          if (auto matcherSymRef = dyn_cast<SymbolRefAttr>(matcher)) {
+            if (auto matcherOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         matcherSymRef))) {
+              if (!foreachMatchMap.count(matcherOp)) {
+                foreachMatchMap[matcherOp] = foreachMatch;
+              }
+            }
+          }
+        }
+        for (auto action : foreachMatch.getActions()) {
+          if (auto actionSymRef = dyn_cast<SymbolRefAttr>(action)) {
+            if (auto actionOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         actionSymRef))) {
+              if (!foreachMatchMap.count(actionOp)) {
+                foreachMatchMap[actionOp] = foreachMatch;
+              }
+            }
+          }
+        }
+      } else {
+        namedSequenceOpsToMove.push_back(namedSequenceOp);
+      }
+    }
+  }
+
+  // Step 2-a: Ensure all ForeachMatchOps have the same result types before
+  // merging.
+  SmallVector<Type, 4> expectedResultTypes =
+      llvm::to_vector<4>(foreachMatchOps.front()->getResultTypes());
+
+  for (auto foreachMatchOp : foreachMatchOps) {
+    SmallVector<Type, 4> currentResultTypes =
+        llvm::to_vector<4>(foreachMatchOp.getResultTypes());
+
+    if (!llvm::equal(currentResultTypes, expectedResultTypes)) {
+      return failure();
+    }
+  }
+
+  // Step 2-b: Ensure all ForeachMatchOps have the same `restrictRoot` and
+  // `flattenResults` attributes.
+  UnitAttr restrictRoot = nullptr;
+  UnitAttr flattenResults = nullptr;
```

**Comment:**
also here

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:270`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
+  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
+  // ForeachMatchOp is also updated.
+  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
+      foreachMatchMap;
+
+  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
+    for (auto namedSequenceOp :
+         innerModule.getBody()->getOps<transform::NamedSequenceOp>()) {
+      if (namedSequenceOp.getSymName() == kKernelConfigSpecName) {
+        transform::ForeachMatchOp foreachMatch = nullptr;
+        int matchCount = 0;
+        // Iterate directly over ForeachMatchOp within kernelConfig.
+        for (auto op : namedSequenceOp.getOps<transform::ForeachMatchOp>()) {
+          if (!foreachMatch) {
+            foreachMatch = op;
+          }
+          matchCount++;
+        }
+
+        // Return failure if multiple occurrences exist.
+        if (matchCount > 1) {
+          return failure();
+        }
+        // Return failure if not foreach match op found.
+        if (!foreachMatch)
+          return failure();
+
+        foreachMatchOps.push_back(foreachMatch);
+
+        for (auto matcher : foreachMatch.getMatchers()) {
+          if (auto matcherSymRef = dyn_cast<SymbolRefAttr>(matcher)) {
+            if (auto matcherOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         matcherSymRef))) {
+              if (!foreachMatchMap.count(matcherOp)) {
+                foreachMatchMap[matcherOp] = foreachMatch;
+              }
+            }
+          }
+        }
+        for (auto action : foreachMatch.getActions()) {
+          if (auto actionSymRef = dyn_cast<SymbolRefAttr>(action)) {
+            if (auto actionOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         actionSymRef))) {
+              if (!foreachMatchMap.count(actionOp)) {
+                foreachMatchMap[actionOp] = foreachMatch;
+              }
+            }
+          }
+        }
+      } else {
+        namedSequenceOpsToMove.push_back(namedSequenceOp);
+      }
+    }
+  }
+
+  // Step 2-a: Ensure all ForeachMatchOps have the same result types before
+  // merging.
+  SmallVector<Type, 4> expectedResultTypes =
+      llvm::to_vector<4>(foreachMatchOps.front()->getResultTypes());
+
+  for (auto foreachMatchOp : foreachMatchOps) {
+    SmallVector<Type, 4> currentResultTypes =
+        llvm::to_vector<4>(foreachMatchOp.getResultTypes());
+
+    if (!llvm::equal(currentResultTypes, expectedResultTypes)) {
+      return failure();
+    }
+  }
+
+  // Step 2-b: Ensure all ForeachMatchOps have the same `restrictRoot` and
+  // `flattenResults` attributes.
+  UnitAttr restrictRoot = nullptr;
+  UnitAttr flattenResults = nullptr;
+  bool hasMismatchAttr = false;
+
+  for (auto foreachMatchOp : foreachMatchOps) {
+    UnitAttr currentRestrictRoot = foreachMatchOp.getRestrictRootAttr();
+    UnitAttr currentFlattenResults = foreachMatchOp.getFlattenResultsAttr();
+
+    if (!restrictRoot) {
+      restrictRoot = currentRestrictRoot; // First encountered value.
+    } else if (restrictRoot != currentRestrictRoot) {
+      hasMismatchAttr = true;
+      break; // Exit early when a mismatch is found.
+    }
+
+    if (!flattenResults) {
+      flattenResults = currentFlattenResults; // First encountered value.
+    } else if (flattenResults != currentFlattenResults) {
+      hasMismatchAttr = true;
+      break; // Exit early when a mismatch is found.
+    }
+  }
+
+  // If there's a mismatch in attributes, do not merge.
+  if (hasMismatchAttr) {
+    return failure();
+  }
+
+  llvm::StringMap<unsigned> specNameCounts;
+  // Step 3-a: Make sure the name sequence names are unique, and then move
+  // collected NamedSequenceOps to the top-level module.
+  for (transform::NamedSequenceOp op : namedSequenceOpsToMove) {
+    StringRef specName = op.getSymName();
+    unsigned specNameSeenCount = specNameCounts[specName]++;
+    std::string newSpecName = specName.str();
+    if (specNameSeenCount > 0) {
+      newSpecName = llvm::formatv("{}_{}", specName, specNameSeenCount).str();
+      op.setSymName(newSpecName);
+    }
```

**Comment:**
We have the same logic elsewhere: could we make it a helper function?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:269`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
+  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
+  // ForeachMatchOp is also updated.
+  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
+      foreachMatchMap;
+
+  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
+    for (auto namedSequenceOp :
+         innerModule.getBody()->getOps<transform::NamedSequenceOp>()) {
+      if (namedSequenceOp.getSymName() == kKernelConfigSpecName) {
+        transform::ForeachMatchOp foreachMatch = nullptr;
+        int matchCount = 0;
+        // Iterate directly over ForeachMatchOp within kernelConfig.
+        for (auto op : namedSequenceOp.getOps<transform::ForeachMatchOp>()) {
+          if (!foreachMatch) {
+            foreachMatch = op;
+          }
+          matchCount++;
+        }
+
+        // Return failure if multiple occurrences exist.
+        if (matchCount > 1) {
+          return failure();
+        }
+        // Return failure if not foreach match op found.
+        if (!foreachMatch)
+          return failure();
+
+        foreachMatchOps.push_back(foreachMatch);
+
+        for (auto matcher : foreachMatch.getMatchers()) {
+          if (auto matcherSymRef = dyn_cast<SymbolRefAttr>(matcher)) {
+            if (auto matcherOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         matcherSymRef))) {
+              if (!foreachMatchMap.count(matcherOp)) {
+                foreachMatchMap[matcherOp] = foreachMatch;
+              }
+            }
+          }
+        }
+        for (auto action : foreachMatch.getActions()) {
+          if (auto actionSymRef = dyn_cast<SymbolRefAttr>(action)) {
+            if (auto actionOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         actionSymRef))) {
+              if (!foreachMatchMap.count(actionOp)) {
+                foreachMatchMap[actionOp] = foreachMatch;
+              }
+            }
+          }
+        }
+      } else {
+        namedSequenceOpsToMove.push_back(namedSequenceOp);
+      }
+    }
+  }
+
+  // Step 2-a: Ensure all ForeachMatchOps have the same result types before
+  // merging.
+  SmallVector<Type, 4> expectedResultTypes =
+      llvm::to_vector<4>(foreachMatchOps.front()->getResultTypes());
+
+  for (auto foreachMatchOp : foreachMatchOps) {
+    SmallVector<Type, 4> currentResultTypes =
+        llvm::to_vector<4>(foreachMatchOp.getResultTypes());
+
+    if (!llvm::equal(currentResultTypes, expectedResultTypes)) {
+      return failure();
+    }
+  }
+
+  // Step 2-b: Ensure all ForeachMatchOps have the same `restrictRoot` and
+  // `flattenResults` attributes.
+  UnitAttr restrictRoot = nullptr;
+  UnitAttr flattenResults = nullptr;
+  bool hasMismatchAttr = false;
+
+  for (auto foreachMatchOp : foreachMatchOps) {
+    UnitAttr currentRestrictRoot = foreachMatchOp.getRestrictRootAttr();
+    UnitAttr currentFlattenResults = foreachMatchOp.getFlattenResultsAttr();
+
+    if (!restrictRoot) {
+      restrictRoot = currentRestrictRoot; // First encountered value.
+    } else if (restrictRoot != currentRestrictRoot) {
+      hasMismatchAttr = true;
+      break; // Exit early when a mismatch is found.
+    }
+
+    if (!flattenResults) {
+      flattenResults = currentFlattenResults; // First encountered value.
+    } else if (flattenResults != currentFlattenResults) {
+      hasMismatchAttr = true;
+      break; // Exit early when a mismatch is found.
+    }
+  }
+
+  // If there's a mismatch in attributes, do not merge.
+  if (hasMismatchAttr) {
+    return failure();
+  }
+
+  llvm::StringMap<unsigned> specNameCounts;
+  // Step 3-a: Make sure the name sequence names are unique, and then move
+  // collected NamedSequenceOps to the top-level module.
+  for (transform::NamedSequenceOp op : namedSequenceOpsToMove) {
+    StringRef specName = op.getSymName();
+    unsigned specNameSeenCount = specNameCounts[specName]++;
+    std::string newSpecName = specName.str();
+    if (specNameSeenCount > 0) {
+      newSpecName = llvm::formatv("{}_{}", specName, specNameSeenCount).str();
+      op.setSymName(newSpecName);
+    }
+
+    // Only update ForeachMatchOp if there's a reference and the name has
+    // changed.
+    if (foreachMatchMap.count(op) && newSpecName != specName) {
```

**Comment:**
also here

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:267`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
+  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
+  // ForeachMatchOp is also updated.
+  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
+      foreachMatchMap;
+
+  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
+    for (auto namedSequenceOp :
+         innerModule.getBody()->getOps<transform::NamedSequenceOp>()) {
+      if (namedSequenceOp.getSymName() == kKernelConfigSpecName) {
+        transform::ForeachMatchOp foreachMatch = nullptr;
+        int matchCount = 0;
+        // Iterate directly over ForeachMatchOp within kernelConfig.
+        for (auto op : namedSequenceOp.getOps<transform::ForeachMatchOp>()) {
+          if (!foreachMatch) {
+            foreachMatch = op;
+          }
+          matchCount++;
+        }
+
+        // Return failure if multiple occurrences exist.
+        if (matchCount > 1) {
+          return failure();
+        }
+        // Return failure if not foreach match op found.
+        if (!foreachMatch)
+          return failure();
+
+        foreachMatchOps.push_back(foreachMatch);
+
+        for (auto matcher : foreachMatch.getMatchers()) {
+          if (auto matcherSymRef = dyn_cast<SymbolRefAttr>(matcher)) {
+            if (auto matcherOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         matcherSymRef))) {
+              if (!foreachMatchMap.count(matcherOp)) {
+                foreachMatchMap[matcherOp] = foreachMatch;
+              }
+            }
+          }
+        }
+        for (auto action : foreachMatch.getActions()) {
+          if (auto actionSymRef = dyn_cast<SymbolRefAttr>(action)) {
+            if (auto actionOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         actionSymRef))) {
+              if (!foreachMatchMap.count(actionOp)) {
+                foreachMatchMap[actionOp] = foreachMatch;
+              }
+            }
+          }
+        }
+      } else {
+        namedSequenceOpsToMove.push_back(namedSequenceOp);
+      }
+    }
+  }
+
+  // Step 2-a: Ensure all ForeachMatchOps have the same result types before
+  // merging.
+  SmallVector<Type, 4> expectedResultTypes =
+      llvm::to_vector<4>(foreachMatchOps.front()->getResultTypes());
+
+  for (auto foreachMatchOp : foreachMatchOps) {
+    SmallVector<Type, 4> currentResultTypes =
+        llvm::to_vector<4>(foreachMatchOp.getResultTypes());
+
+    if (!llvm::equal(currentResultTypes, expectedResultTypes)) {
+      return failure();
+    }
+  }
+
+  // Step 2-b: Ensure all ForeachMatchOps have the same `restrictRoot` and
+  // `flattenResults` attributes.
+  UnitAttr restrictRoot = nullptr;
+  UnitAttr flattenResults = nullptr;
+  bool hasMismatchAttr = false;
+
+  for (auto foreachMatchOp : foreachMatchOps) {
+    UnitAttr currentRestrictRoot = foreachMatchOp.getRestrictRootAttr();
+    UnitAttr currentFlattenResults = foreachMatchOp.getFlattenResultsAttr();
+
+    if (!restrictRoot) {
+      restrictRoot = currentRestrictRoot; // First encountered value.
+    } else if (restrictRoot != currentRestrictRoot) {
+      hasMismatchAttr = true;
+      break; // Exit early when a mismatch is found.
+    }
+
+    if (!flattenResults) {
+      flattenResults = currentFlattenResults; // First encountered value.
+    } else if (flattenResults != currentFlattenResults) {
+      hasMismatchAttr = true;
+      break; // Exit early when a mismatch is found.
+    }
+  }
+
+  // If there's a mismatch in attributes, do not merge.
+  if (hasMismatchAttr) {
+    return failure();
+  }
+
+  llvm::StringMap<unsigned> specNameCounts;
+  // Step 3-a: Make sure the name sequence names are unique, and then move
+  // collected NamedSequenceOps to the top-level module.
+  for (transform::NamedSequenceOp op : namedSequenceOpsToMove) {
+    StringRef specName = op.getSymName();
+    unsigned specNameSeenCount = specNameCounts[specName]++;
+    std::string newSpecName = specName.str();
+    if (specNameSeenCount > 0) {
+      newSpecName = llvm::formatv("{}_{}", specName, specNameSeenCount).str();
+      op.setSymName(newSpecName);
+    }
+
+    // Only update ForeachMatchOp if there's a reference and the name has
```

**Comment:**
What do you mean by 'if there's a reference'?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:324`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
+  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
+  // ForeachMatchOp is also updated.
+  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
+      foreachMatchMap;
+
+  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
+    for (auto namedSequenceOp :
+         innerModule.getBody()->getOps<transform::NamedSequenceOp>()) {
+      if (namedSequenceOp.getSymName() == kKernelConfigSpecName) {
+        transform::ForeachMatchOp foreachMatch = nullptr;
+        int matchCount = 0;
+        // Iterate directly over ForeachMatchOp within kernelConfig.
+        for (auto op : namedSequenceOp.getOps<transform::ForeachMatchOp>()) {
+          if (!foreachMatch) {
+            foreachMatch = op;
+          }
+          matchCount++;
+        }
+
+        // Return failure if multiple occurrences exist.
+        if (matchCount > 1) {
+          return failure();
+        }
+        // Return failure if not foreach match op found.
+        if (!foreachMatch)
+          return failure();
+
+        foreachMatchOps.push_back(foreachMatch);
+
+        for (auto matcher : foreachMatch.getMatchers()) {
+          if (auto matcherSymRef = dyn_cast<SymbolRefAttr>(matcher)) {
+            if (auto matcherOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         matcherSymRef))) {
+              if (!foreachMatchMap.count(matcherOp)) {
+                foreachMatchMap[matcherOp] = foreachMatch;
+              }
+            }
+          }
+        }
+        for (auto action : foreachMatch.getActions()) {
+          if (auto actionSymRef = dyn_cast<SymbolRefAttr>(action)) {
+            if (auto actionOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         actionSymRef))) {
+              if (!foreachMatchMap.count(actionOp)) {
+                foreachMatchMap[actionOp] = foreachMatch;
+              }
+            }
+          }
+        }
+      } else {
+        namedSequenceOpsToMove.push_back(namedSequenceOp);
+      }
+    }
+  }
+
+  // Step 2-a: Ensure all ForeachMatchOps have the same result types before
+  // merging.
+  SmallVector<Type, 4> expectedResultTypes =
+      llvm::to_vector<4>(foreachMatchOps.front()->getResultTypes());
+
+  for (auto foreachMatchOp : foreachMatchOps) {
+    SmallVector<Type, 4> currentResultTypes =
+        llvm::to_vector<4>(foreachMatchOp.getResultTypes());
+
+    if (!llvm::equal(currentResultTypes, expectedResultTypes)) {
+      return failure();
+    }
+  }
+
+  // Step 2-b: Ensure all ForeachMatchOps have the same `restrictRoot` and
+  // `flattenResults` attributes.
+  UnitAttr restrictRoot = nullptr;
+  UnitAttr flattenResults = nullptr;
+  bool hasMismatchAttr = false;
+
+  for (auto foreachMatchOp : foreachMatchOps) {
+    UnitAttr currentRestrictRoot = foreachMatchOp.getRestrictRootAttr();
+    UnitAttr currentFlattenResults = foreachMatchOp.getFlattenResultsAttr();
+
+    if (!restrictRoot) {
+      restrictRoot = currentRestrictRoot; // First encountered value.
+    } else if (restrictRoot != currentRestrictRoot) {
+      hasMismatchAttr = true;
+      break; // Exit early when a mismatch is found.
+    }
+
+    if (!flattenResults) {
+      flattenResults = currentFlattenResults; // First encountered value.
+    } else if (flattenResults != currentFlattenResults) {
+      hasMismatchAttr = true;
+      break; // Exit early when a mismatch is found.
+    }
+  }
+
+  // If there's a mismatch in attributes, do not merge.
+  if (hasMismatchAttr) {
+    return failure();
+  }
+
+  llvm::StringMap<unsigned> specNameCounts;
+  // Step 3-a: Make sure the name sequence names are unique, and then move
+  // collected NamedSequenceOps to the top-level module.
+  for (transform::NamedSequenceOp op : namedSequenceOpsToMove) {
+    StringRef specName = op.getSymName();
+    unsigned specNameSeenCount = specNameCounts[specName]++;
+    std::string newSpecName = specName.str();
+    if (specNameSeenCount > 0) {
+      newSpecName = llvm::formatv("{}_{}", specName, specNameSeenCount).str();
+      op.setSymName(newSpecName);
+    }
+
+    // Only update ForeachMatchOp if there's a reference and the name has
+    // changed.
+    if (foreachMatchMap.count(op) && newSpecName != specName) {
+      transform::ForeachMatchOp foreachMatchOp = foreachMatchMap[op];
+
+      SmallVector<Attribute> updatedMatchers, updatedActions;
+      for (auto matcherAttr : foreachMatchOp.getMatchers()) {
+        StringRef matcherName =
+            cast<SymbolRefAttr>(matcherAttr).getRootReference();
+        updatedMatchers.push_back(
+            (matcherName == specName)
+                ? SymbolRefAttr::get(builder.getContext(), newSpecName)
+                : matcherAttr);
+      }
+
+      for (auto actionAttr : foreachMatchOp.getActions()) {
+        StringRef actionName =
+            cast<SymbolRefAttr>(actionAttr).getRootReference();
+        updatedActions.push_back(
+            (actionName == specName)
+                ? SymbolRefAttr::get(builder.getContext(), newSpecName)
+                : actionAttr);
+      }
+
+      // Apply the updated matchers and actions.
+      foreachMatchOp.setMatchersAttr(builder.getArrayAttr(updatedMatchers));
+      foreachMatchOp.setActionsAttr(builder.getArrayAttr(updatedActions));
+    }
+    op.getOperation()->moveBefore(module.getBody(), module.getBody()->end());
+  }
+
+  // Step 3-b: Create a new NamedSequenceOp `__kernel_config` in the top-level
+  // module.
+  builder.setInsertionPointToEnd(module.getBody());
+  Location loc = module.getLoc();
+  Type anyOpType = builder.getType<transform::AnyOpType>();
+  FunctionType seqType =
+      builder.getFunctionType(TypeRange{anyOpType}, TypeRange{anyOpType});
+
+  auto newNamedSequence = builder.create<transform::NamedSequenceOp>(
+      loc, kKernelConfigSpecName, TypeAttr::get(seqType),
+      /*sym_visibility=*/StringAttr{},
+      /*arg_attrs=*/ArrayAttr{},
+      /*res_attrs*/ ArrayAttr{});
+
+  bool hasConsumedArg =
+      llvm::any_of(foreachMatchOps, [](transform::ForeachMatchOp op) {
+        Value operand = op->getOperand(0);
+        if (auto blockArg = mlir::dyn_cast<BlockArgument>(operand)) {
+          Operation *parentOp = blockArg.getOwner()->getParentOp();
+          if (auto namedSequenceOp =
+                  mlir::dyn_cast<transform::NamedSequenceOp>(parentOp)) {
+            return namedSequenceOp.getArgAttr(blockArg.getArgNumber(),
+                                              kArgConsumedAttrName) != nullptr;
+          }
+        }
+        return false;
+      });
```

**Comment:**
Make this a helper function

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:345`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
+  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
+  // ForeachMatchOp is also updated.
+  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
+      foreachMatchMap;
+
+  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
+    for (auto namedSequenceOp :
+         innerModule.getBody()->getOps<transform::NamedSequenceOp>()) {
+      if (namedSequenceOp.getSymName() == kKernelConfigSpecName) {
+        transform::ForeachMatchOp foreachMatch = nullptr;
+        int matchCount = 0;
+        // Iterate directly over ForeachMatchOp within kernelConfig.
+        for (auto op : namedSequenceOp.getOps<transform::ForeachMatchOp>()) {
+          if (!foreachMatch) {
+            foreachMatch = op;
+          }
+          matchCount++;
+        }
+
+        // Return failure if multiple occurrences exist.
+        if (matchCount > 1) {
+          return failure();
+        }
+        // Return failure if not foreach match op found.
+        if (!foreachMatch)
+          return failure();
+
+        foreachMatchOps.push_back(foreachMatch);
+
+        for (auto matcher : foreachMatch.getMatchers()) {
+          if (auto matcherSymRef = dyn_cast<SymbolRefAttr>(matcher)) {
+            if (auto matcherOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         matcherSymRef))) {
+              if (!foreachMatchMap.count(matcherOp)) {
+                foreachMatchMap[matcherOp] = foreachMatch;
+              }
+            }
+          }
+        }
+        for (auto action : foreachMatch.getActions()) {
+          if (auto actionSymRef = dyn_cast<SymbolRefAttr>(action)) {
+            if (auto actionOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         actionSymRef))) {
+              if (!foreachMatchMap.count(actionOp)) {
+                foreachMatchMap[actionOp] = foreachMatch;
+              }
+            }
+          }
+        }
+      } else {
+        namedSequenceOpsToMove.push_back(namedSequenceOp);
+      }
+    }
+  }
+
+  // Step 2-a: Ensure all ForeachMatchOps have the same result types before
+  // merging.
+  SmallVector<Type, 4> expectedResultTypes =
+      llvm::to_vector<4>(foreachMatchOps.front()->getResultTypes());
+
+  for (auto foreachMatchOp : foreachMatchOps) {
+    SmallVector<Type, 4> currentResultTypes =
+        llvm::to_vector<4>(foreachMatchOp.getResultTypes());
+
+    if (!llvm::equal(currentResultTypes, expectedResultTypes)) {
+      return failure();
+    }
+  }
+
+  // Step 2-b: Ensure all ForeachMatchOps have the same `restrictRoot` and
+  // `flattenResults` attributes.
+  UnitAttr restrictRoot = nullptr;
+  UnitAttr flattenResults = nullptr;
+  bool hasMismatchAttr = false;
+
+  for (auto foreachMatchOp : foreachMatchOps) {
+    UnitAttr currentRestrictRoot = foreachMatchOp.getRestrictRootAttr();
+    UnitAttr currentFlattenResults = foreachMatchOp.getFlattenResultsAttr();
+
+    if (!restrictRoot) {
+      restrictRoot = currentRestrictRoot; // First encountered value.
+    } else if (restrictRoot != currentRestrictRoot) {
+      hasMismatchAttr = true;
+      break; // Exit early when a mismatch is found.
+    }
+
+    if (!flattenResults) {
+      flattenResults = currentFlattenResults; // First encountered value.
+    } else if (flattenResults != currentFlattenResults) {
+      hasMismatchAttr = true;
+      break; // Exit early when a mismatch is found.
+    }
+  }
+
+  // If there's a mismatch in attributes, do not merge.
+  if (hasMismatchAttr) {
+    return failure();
+  }
+
+  llvm::StringMap<unsigned> specNameCounts;
+  // Step 3-a: Make sure the name sequence names are unique, and then move
+  // collected NamedSequenceOps to the top-level module.
+  for (transform::NamedSequenceOp op : namedSequenceOpsToMove) {
+    StringRef specName = op.getSymName();
+    unsigned specNameSeenCount = specNameCounts[specName]++;
+    std::string newSpecName = specName.str();
+    if (specNameSeenCount > 0) {
+      newSpecName = llvm::formatv("{}_{}", specName, specNameSeenCount).str();
+      op.setSymName(newSpecName);
+    }
+
+    // Only update ForeachMatchOp if there's a reference and the name has
+    // changed.
+    if (foreachMatchMap.count(op) && newSpecName != specName) {
+      transform::ForeachMatchOp foreachMatchOp = foreachMatchMap[op];
+
+      SmallVector<Attribute> updatedMatchers, updatedActions;
+      for (auto matcherAttr : foreachMatchOp.getMatchers()) {
+        StringRef matcherName =
+            cast<SymbolRefAttr>(matcherAttr).getRootReference();
+        updatedMatchers.push_back(
+            (matcherName == specName)
+                ? SymbolRefAttr::get(builder.getContext(), newSpecName)
+                : matcherAttr);
+      }
+
+      for (auto actionAttr : foreachMatchOp.getActions()) {
+        StringRef actionName =
+            cast<SymbolRefAttr>(actionAttr).getRootReference();
+        updatedActions.push_back(
+            (actionName == specName)
+                ? SymbolRefAttr::get(builder.getContext(), newSpecName)
+                : actionAttr);
+      }
+
+      // Apply the updated matchers and actions.
+      foreachMatchOp.setMatchersAttr(builder.getArrayAttr(updatedMatchers));
+      foreachMatchOp.setActionsAttr(builder.getArrayAttr(updatedActions));
+    }
+    op.getOperation()->moveBefore(module.getBody(), module.getBody()->end());
+  }
+
+  // Step 3-b: Create a new NamedSequenceOp `__kernel_config` in the top-level
+  // module.
+  builder.setInsertionPointToEnd(module.getBody());
+  Location loc = module.getLoc();
+  Type anyOpType = builder.getType<transform::AnyOpType>();
+  FunctionType seqType =
+      builder.getFunctionType(TypeRange{anyOpType}, TypeRange{anyOpType});
+
+  auto newNamedSequence = builder.create<transform::NamedSequenceOp>(
+      loc, kKernelConfigSpecName, TypeAttr::get(seqType),
+      /*sym_visibility=*/StringAttr{},
+      /*arg_attrs=*/ArrayAttr{},
+      /*res_attrs*/ ArrayAttr{});
+
+  bool hasConsumedArg =
+      llvm::any_of(foreachMatchOps, [](transform::ForeachMatchOp op) {
+        Value operand = op->getOperand(0);
+        if (auto blockArg = mlir::dyn_cast<BlockArgument>(operand)) {
+          Operation *parentOp = blockArg.getOwner()->getParentOp();
+          if (auto namedSequenceOp =
+                  mlir::dyn_cast<transform::NamedSequenceOp>(parentOp)) {
+            return namedSequenceOp.getArgAttr(blockArg.getArgNumber(),
+                                              kArgConsumedAttrName) != nullptr;
+          }
+        }
+        return false;
+      });
+
+  StringRef attrName =
+      hasConsumedArg ? kArgConsumedAttrName : kArgReadOnlyAttrName;
+  newNamedSequence.setArgAttr(0, attrName, builder.getUnitAttr());
+  newNamedSequence->setAttr(kTuningSpecEntrypointAttrName,
+                            builder.getUnitAttr());
+  // Indicate the output module is a default tuning spec after merging.
+  module->setAttr(kTuningSpecDefaultEntrypointAttrName, builder.getUnitAttr());
+
+  // Step 3-C: Create a new block inside the NamedSequenceOp and merging
+  // ForeachMatchOp from each inner modules into one ForachMatchOp.
+  SmallVector<Type, 4> resultTypes;
+  llvm::append_range(resultTypes, expectedResultTypes);
+
+  SmallVector<std::pair<SymbolRefAttr, SymbolRefAttr>> matcherActionPairs;
+  SmallVector<Value, 4> forwardedInputs;
+  for (auto foreachMatchOp : foreachMatchOps) {
+    ArrayAttr matchers = foreachMatchOp.getMatchers();
+    ArrayAttr actions = foreachMatchOp.getActions();
+
+    for (size_t i = 0; i < matchers.size(); i++) {
```

**Comment:**
Do not re-evaluate the end iterator: https://llvm.org/docs/CodingStandards.html#don-t-evaluate-end-every-time-through-a-loop

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:351`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
+  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
+  // ForeachMatchOp is also updated.
+  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
+      foreachMatchMap;
+
+  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
+    for (auto namedSequenceOp :
+         innerModule.getBody()->getOps<transform::NamedSequenceOp>()) {
+      if (namedSequenceOp.getSymName() == kKernelConfigSpecName) {
+        transform::ForeachMatchOp foreachMatch = nullptr;
+        int matchCount = 0;
+        // Iterate directly over ForeachMatchOp within kernelConfig.
+        for (auto op : namedSequenceOp.getOps<transform::ForeachMatchOp>()) {
+          if (!foreachMatch) {
+            foreachMatch = op;
+          }
+          matchCount++;
+        }
+
+        // Return failure if multiple occurrences exist.
+        if (matchCount > 1) {
+          return failure();
+        }
+        // Return failure if not foreach match op found.
+        if (!foreachMatch)
+          return failure();
+
+        foreachMatchOps.push_back(foreachMatch);
+
+        for (auto matcher : foreachMatch.getMatchers()) {
+          if (auto matcherSymRef = dyn_cast<SymbolRefAttr>(matcher)) {
+            if (auto matcherOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         matcherSymRef))) {
+              if (!foreachMatchMap.count(matcherOp)) {
+                foreachMatchMap[matcherOp] = foreachMatch;
+              }
+            }
+          }
+        }
+        for (auto action : foreachMatch.getActions()) {
+          if (auto actionSymRef = dyn_cast<SymbolRefAttr>(action)) {
+            if (auto actionOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         actionSymRef))) {
+              if (!foreachMatchMap.count(actionOp)) {
+                foreachMatchMap[actionOp] = foreachMatch;
+              }
+            }
+          }
+        }
+      } else {
+        namedSequenceOpsToMove.push_back(namedSequenceOp);
+      }
+    }
+  }
+
+  // Step 2-a: Ensure all ForeachMatchOps have the same result types before
+  // merging.
+  SmallVector<Type, 4> expectedResultTypes =
+      llvm::to_vector<4>(foreachMatchOps.front()->getResultTypes());
+
+  for (auto foreachMatchOp : foreachMatchOps) {
+    SmallVector<Type, 4> currentResultTypes =
+        llvm::to_vector<4>(foreachMatchOp.getResultTypes());
+
+    if (!llvm::equal(currentResultTypes, expectedResultTypes)) {
+      return failure();
+    }
+  }
+
+  // Step 2-b: Ensure all ForeachMatchOps have the same `restrictRoot` and
+  // `flattenResults` attributes.
+  UnitAttr restrictRoot = nullptr;
+  UnitAttr flattenResults = nullptr;
+  bool hasMismatchAttr = false;
+
+  for (auto foreachMatchOp : foreachMatchOps) {
+    UnitAttr currentRestrictRoot = foreachMatchOp.getRestrictRootAttr();
+    UnitAttr currentFlattenResults = foreachMatchOp.getFlattenResultsAttr();
+
+    if (!restrictRoot) {
+      restrictRoot = currentRestrictRoot; // First encountered value.
+    } else if (restrictRoot != currentRestrictRoot) {
+      hasMismatchAttr = true;
+      break; // Exit early when a mismatch is found.
+    }
+
+    if (!flattenResults) {
+      flattenResults = currentFlattenResults; // First encountered value.
+    } else if (flattenResults != currentFlattenResults) {
+      hasMismatchAttr = true;
+      break; // Exit early when a mismatch is found.
+    }
+  }
+
+  // If there's a mismatch in attributes, do not merge.
+  if (hasMismatchAttr) {
+    return failure();
+  }
+
+  llvm::StringMap<unsigned> specNameCounts;
+  // Step 3-a: Make sure the name sequence names are unique, and then move
+  // collected NamedSequenceOps to the top-level module.
+  for (transform::NamedSequenceOp op : namedSequenceOpsToMove) {
+    StringRef specName = op.getSymName();
+    unsigned specNameSeenCount = specNameCounts[specName]++;
+    std::string newSpecName = specName.str();
+    if (specNameSeenCount > 0) {
+      newSpecName = llvm::formatv("{}_{}", specName, specNameSeenCount).str();
+      op.setSymName(newSpecName);
+    }
+
+    // Only update ForeachMatchOp if there's a reference and the name has
+    // changed.
+    if (foreachMatchMap.count(op) && newSpecName != specName) {
+      transform::ForeachMatchOp foreachMatchOp = foreachMatchMap[op];
+
+      SmallVector<Attribute> updatedMatchers, updatedActions;
+      for (auto matcherAttr : foreachMatchOp.getMatchers()) {
+        StringRef matcherName =
+            cast<SymbolRefAttr>(matcherAttr).getRootReference();
+        updatedMatchers.push_back(
+            (matcherName == specName)
+                ? SymbolRefAttr::get(builder.getContext(), newSpecName)
+                : matcherAttr);
+      }
+
+      for (auto actionAttr : foreachMatchOp.getActions()) {
+        StringRef actionName =
+            cast<SymbolRefAttr>(actionAttr).getRootReference();
+        updatedActions.push_back(
+            (actionName == specName)
+                ? SymbolRefAttr::get(builder.getContext(), newSpecName)
+                : actionAttr);
+      }
+
+      // Apply the updated matchers and actions.
+      foreachMatchOp.setMatchersAttr(builder.getArrayAttr(updatedMatchers));
+      foreachMatchOp.setActionsAttr(builder.getArrayAttr(updatedActions));
+    }
+    op.getOperation()->moveBefore(module.getBody(), module.getBody()->end());
+  }
+
+  // Step 3-b: Create a new NamedSequenceOp `__kernel_config` in the top-level
+  // module.
+  builder.setInsertionPointToEnd(module.getBody());
+  Location loc = module.getLoc();
+  Type anyOpType = builder.getType<transform::AnyOpType>();
+  FunctionType seqType =
+      builder.getFunctionType(TypeRange{anyOpType}, TypeRange{anyOpType});
+
+  auto newNamedSequence = builder.create<transform::NamedSequenceOp>(
+      loc, kKernelConfigSpecName, TypeAttr::get(seqType),
+      /*sym_visibility=*/StringAttr{},
+      /*arg_attrs=*/ArrayAttr{},
+      /*res_attrs*/ ArrayAttr{});
+
+  bool hasConsumedArg =
+      llvm::any_of(foreachMatchOps, [](transform::ForeachMatchOp op) {
+        Value operand = op->getOperand(0);
+        if (auto blockArg = mlir::dyn_cast<BlockArgument>(operand)) {
+          Operation *parentOp = blockArg.getOwner()->getParentOp();
+          if (auto namedSequenceOp =
+                  mlir::dyn_cast<transform::NamedSequenceOp>(parentOp)) {
+            return namedSequenceOp.getArgAttr(blockArg.getArgNumber(),
+                                              kArgConsumedAttrName) != nullptr;
+          }
+        }
+        return false;
+      });
+
+  StringRef attrName =
+      hasConsumedArg ? kArgConsumedAttrName : kArgReadOnlyAttrName;
+  newNamedSequence.setArgAttr(0, attrName, builder.getUnitAttr());
+  newNamedSequence->setAttr(kTuningSpecEntrypointAttrName,
+                            builder.getUnitAttr());
+  // Indicate the output module is a default tuning spec after merging.
+  module->setAttr(kTuningSpecDefaultEntrypointAttrName, builder.getUnitAttr());
+
+  // Step 3-C: Create a new block inside the NamedSequenceOp and merging
+  // ForeachMatchOp from each inner modules into one ForachMatchOp.
+  SmallVector<Type, 4> resultTypes;
+  llvm::append_range(resultTypes, expectedResultTypes);
+
+  SmallVector<std::pair<SymbolRefAttr, SymbolRefAttr>> matcherActionPairs;
+  SmallVector<Value, 4> forwardedInputs;
+  for (auto foreachMatchOp : foreachMatchOps) {
+    ArrayAttr matchers = foreachMatchOp.getMatchers();
+    ArrayAttr actions = foreachMatchOp.getActions();
+
+    for (size_t i = 0; i < matchers.size(); i++) {
+      matcherActionPairs.push_back({mlir::cast<SymbolRefAttr>(matchers[i]),
+                                    mlir::cast<SymbolRefAttr>(actions[i])});
+    }
+    // Collect forwarded inputs (if any).
+    for (Value input : foreachMatchOp.getForwardedInputs()) {
+      if (llvm::find(forwardedInputs, input) == forwardedInputs.end()) {
```

**Comment:**
Use `llvm::is_contained`

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:344`

```diff
@@ -136,6 +144,242 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   return newSpec;
 }
 
+static FailureOr<NamedSequenceOp> emitLinkedDefaultTuningSpec(ModuleOp module) {
+  OpBuilder builder(module.getContext());
+  SmallVector<transform::NamedSequenceOp> namedSequenceOpsToMove;
+  SmallVector<transform::ForeachMatchOp> foreachMatchOps;
+  // foreachMatchMap: NamedSequenceOp -> ForeachMatchOps that reference it
+  // (either as a matcher or an action). It ensures
+  // that when a NamedSequenceOp is renamed for uniqueness, the corresponding
+  // ForeachMatchOp is also updated.
+  llvm::DenseMap<transform::NamedSequenceOp, transform::ForeachMatchOp>
+      foreachMatchMap;
+
+  // Step 1: Collect NamedSequenceOps and ForeachMatchOps from inner modules.
+  for (auto innerModule : module.getBody()->getOps<ModuleOp>()) {
+    for (auto namedSequenceOp :
+         innerModule.getBody()->getOps<transform::NamedSequenceOp>()) {
+      if (namedSequenceOp.getSymName() == kKernelConfigSpecName) {
+        transform::ForeachMatchOp foreachMatch = nullptr;
+        int matchCount = 0;
+        // Iterate directly over ForeachMatchOp within kernelConfig.
+        for (auto op : namedSequenceOp.getOps<transform::ForeachMatchOp>()) {
+          if (!foreachMatch) {
+            foreachMatch = op;
+          }
+          matchCount++;
+        }
+
+        // Return failure if multiple occurrences exist.
+        if (matchCount > 1) {
+          return failure();
+        }
+        // Return failure if not foreach match op found.
+        if (!foreachMatch)
+          return failure();
+
+        foreachMatchOps.push_back(foreachMatch);
+
+        for (auto matcher : foreachMatch.getMatchers()) {
+          if (auto matcherSymRef = dyn_cast<SymbolRefAttr>(matcher)) {
+            if (auto matcherOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         matcherSymRef))) {
+              if (!foreachMatchMap.count(matcherOp)) {
+                foreachMatchMap[matcherOp] = foreachMatch;
+              }
+            }
+          }
+        }
+        for (auto action : foreachMatch.getActions()) {
+          if (auto actionSymRef = dyn_cast<SymbolRefAttr>(action)) {
+            if (auto actionOp = dyn_cast_or_null<transform::NamedSequenceOp>(
+                    SymbolTable::lookupNearestSymbolFrom(innerModule,
+                                                         actionSymRef))) {
+              if (!foreachMatchMap.count(actionOp)) {
+                foreachMatchMap[actionOp] = foreachMatch;
+              }
+            }
+          }
+        }
+      } else {
+        namedSequenceOpsToMove.push_back(namedSequenceOp);
+      }
+    }
+  }
+
+  // Step 2-a: Ensure all ForeachMatchOps have the same result types before
+  // merging.
+  SmallVector<Type, 4> expectedResultTypes =
+      llvm::to_vector<4>(foreachMatchOps.front()->getResultTypes());
+
+  for (auto foreachMatchOp : foreachMatchOps) {
+    SmallVector<Type, 4> currentResultTypes =
+        llvm::to_vector<4>(foreachMatchOp.getResultTypes());
+
+    if (!llvm::equal(currentResultTypes, expectedResultTypes)) {
+      return failure();
+    }
+  }
+
+  // Step 2-b: Ensure all ForeachMatchOps have the same `restrictRoot` and
+  // `flattenResults` attributes.
+  UnitAttr restrictRoot = nullptr;
+  UnitAttr flattenResults = nullptr;
+  bool hasMismatchAttr = false;
+
+  for (auto foreachMatchOp : foreachMatchOps) {
+    UnitAttr currentRestrictRoot = foreachMatchOp.getRestrictRootAttr();
+    UnitAttr currentFlattenResults = foreachMatchOp.getFlattenResultsAttr();
+
+    if (!restrictRoot) {
+      restrictRoot = currentRestrictRoot; // First encountered value.
+    } else if (restrictRoot != currentRestrictRoot) {
+      hasMismatchAttr = true;
+      break; // Exit early when a mismatch is found.
+    }
+
+    if (!flattenResults) {
+      flattenResults = currentFlattenResults; // First encountered value.
+    } else if (flattenResults != currentFlattenResults) {
+      hasMismatchAttr = true;
+      break; // Exit early when a mismatch is found.
+    }
+  }
+
+  // If there's a mismatch in attributes, do not merge.
+  if (hasMismatchAttr) {
+    return failure();
+  }
+
+  llvm::StringMap<unsigned> specNameCounts;
+  // Step 3-a: Make sure the name sequence names are unique, and then move
+  // collected NamedSequenceOps to the top-level module.
+  for (transform::NamedSequenceOp op : namedSequenceOpsToMove) {
+    StringRef specName = op.getSymName();
+    unsigned specNameSeenCount = specNameCounts[specName]++;
+    std::string newSpecName = specName.str();
+    if (specNameSeenCount > 0) {
+      newSpecName = llvm::formatv("{}_{}", specName, specNameSeenCount).str();
+      op.setSymName(newSpecName);
+    }
+
+    // Only update ForeachMatchOp if there's a reference and the name has
+    // changed.
+    if (foreachMatchMap.count(op) && newSpecName != specName) {
+      transform::ForeachMatchOp foreachMatchOp = foreachMatchMap[op];
+
+      SmallVector<Attribute> updatedMatchers, updatedActions;
+      for (auto matcherAttr : foreachMatchOp.getMatchers()) {
+        StringRef matcherName =
+            cast<SymbolRefAttr>(matcherAttr).getRootReference();
+        updatedMatchers.push_back(
+            (matcherName == specName)
+                ? SymbolRefAttr::get(builder.getContext(), newSpecName)
+                : matcherAttr);
+      }
+
+      for (auto actionAttr : foreachMatchOp.getActions()) {
+        StringRef actionName =
+            cast<SymbolRefAttr>(actionAttr).getRootReference();
+        updatedActions.push_back(
+            (actionName == specName)
+                ? SymbolRefAttr::get(builder.getContext(), newSpecName)
+                : actionAttr);
+      }
+
+      // Apply the updated matchers and actions.
+      foreachMatchOp.setMatchersAttr(builder.getArrayAttr(updatedMatchers));
+      foreachMatchOp.setActionsAttr(builder.getArrayAttr(updatedActions));
+    }
+    op.getOperation()->moveBefore(module.getBody(), module.getBody()->end());
+  }
+
+  // Step 3-b: Create a new NamedSequenceOp `__kernel_config` in the top-level
+  // module.
+  builder.setInsertionPointToEnd(module.getBody());
+  Location loc = module.getLoc();
+  Type anyOpType = builder.getType<transform::AnyOpType>();
+  FunctionType seqType =
+      builder.getFunctionType(TypeRange{anyOpType}, TypeRange{anyOpType});
+
+  auto newNamedSequence = builder.create<transform::NamedSequenceOp>(
+      loc, kKernelConfigSpecName, TypeAttr::get(seqType),
+      /*sym_visibility=*/StringAttr{},
+      /*arg_attrs=*/ArrayAttr{},
+      /*res_attrs*/ ArrayAttr{});
+
+  bool hasConsumedArg =
+      llvm::any_of(foreachMatchOps, [](transform::ForeachMatchOp op) {
+        Value operand = op->getOperand(0);
+        if (auto blockArg = mlir::dyn_cast<BlockArgument>(operand)) {
+          Operation *parentOp = blockArg.getOwner()->getParentOp();
+          if (auto namedSequenceOp =
+                  mlir::dyn_cast<transform::NamedSequenceOp>(parentOp)) {
+            return namedSequenceOp.getArgAttr(blockArg.getArgNumber(),
+                                              kArgConsumedAttrName) != nullptr;
+          }
+        }
+        return false;
+      });
+
+  StringRef attrName =
+      hasConsumedArg ? kArgConsumedAttrName : kArgReadOnlyAttrName;
+  newNamedSequence.setArgAttr(0, attrName, builder.getUnitAttr());
+  newNamedSequence->setAttr(kTuningSpecEntrypointAttrName,
+                            builder.getUnitAttr());
+  // Indicate the output module is a default tuning spec after merging.
+  module->setAttr(kTuningSpecDefaultEntrypointAttrName, builder.getUnitAttr());
+
+  // Step 3-C: Create a new block inside the NamedSequenceOp and merging
+  // ForeachMatchOp from each inner modules into one ForachMatchOp.
+  SmallVector<Type, 4> resultTypes;
+  llvm::append_range(resultTypes, expectedResultTypes);
+
+  SmallVector<std::pair<SymbolRefAttr, SymbolRefAttr>> matcherActionPairs;
+  SmallVector<Value, 4> forwardedInputs;
+  for (auto foreachMatchOp : foreachMatchOps) {
+    ArrayAttr matchers = foreachMatchOp.getMatchers();
+    ArrayAttr actions = foreachMatchOp.getActions();
+
+    for (size_t i = 0; i < matchers.size(); i++) {
+      matcherActionPairs.push_back({mlir::cast<SymbolRefAttr>(matchers[i]),
+                                    mlir::cast<SymbolRefAttr>(actions[i])});
+    }
+    // Collect forwarded inputs (if any).
+    for (Value input : foreachMatchOp.getForwardedInputs()) {
+      if (llvm::find(forwardedInputs, input) == forwardedInputs.end()) {
+        forwardedInputs.push_back(input); // Avoid duplicates
+      }
+    }
+  }
+
+  SmallVector<Attribute> mergedMatchers;
+  SmallVector<Attribute> mergedActions;
+
+  for (const auto &pair : matcherActionPairs) {
+    mergedMatchers.push_back(pair.first);
+    mergedActions.push_back(pair.second);
+  }
```

**Comment:**
You can make this more readable with structured bindings

---


---


## [PR #20081](https://github.com/iree-org/iree/pull/20081):  [Codegen][Tuner]: remove attrs inside decomposeConfig for attention op

### Review Summary

**COMMENTED** (2025-02-24)


**COMMENTED** (2025-02-25)

Looks good overall but let's wait from an approval from Kunwar


**APPROVED** (2025-02-25)


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:74`

```diff
@@ -69,8 +69,19 @@ struct StripAttentionOpCompilationInfo final
       eraseLoweringConfig(attentionOp);
     }
 
-    if (attentionOp.getDecompositionConfigAttr()) {
-      attentionOp.removeDecompositionConfigAttr();
+    DictionaryAttr decompositionConfig =
+        attentionOp.getDecompositionConfigAttr();
+    if (decompositionConfig) {
```

**Comment:**
```suggestion
    if (DictionaryAttr decompositionConfig =
        attentionOp.getDecompositionConfigAttr()) {
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:75`

```diff
@@ -69,8 +69,19 @@ struct StripAttentionOpCompilationInfo final
       eraseLoweringConfig(attentionOp);
     }
 
-    if (attentionOp.getDecompositionConfigAttr()) {
-      attentionOp.removeDecompositionConfigAttr();
+    DictionaryAttr decompositionConfig =
+        attentionOp.getDecompositionConfigAttr();
+    if (decompositionConfig) {
+      decompositionConfig = DictionaryAttr::get(
```

**Comment:**
It seems a little bit confusing to me to reuse the same variable even though it's not a reference and won't update the attribute itself. Can we create a new one instead?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:77`

```diff
@@ -69,8 +69,19 @@ struct StripAttentionOpCompilationInfo final
       eraseLoweringConfig(attentionOp);
     }
 
-    if (attentionOp.getDecompositionConfigAttr()) {
-      attentionOp.removeDecompositionConfigAttr();
+    DictionaryAttr decompositionConfig =
+        attentionOp.getDecompositionConfigAttr();
+    if (decompositionConfig) {
+      decompositionConfig = DictionaryAttr::get(
+          decompositionConfig.getContext(),
+          llvm::to_vector(llvm::make_filter_range(
```

**Comment:**
can we use `llvm::filter_to_vector`?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:77`

```diff
@@ -69,8 +69,19 @@ struct StripAttentionOpCompilationInfo final
       eraseLoweringConfig(attentionOp);
     }
 
-    if (attentionOp.getDecompositionConfigAttr()) {
-      attentionOp.removeDecompositionConfigAttr();
+    DictionaryAttr decompositionConfig =
+        attentionOp.getDecompositionConfigAttr();
+    if (decompositionConfig) {
+      decompositionConfig = DictionaryAttr::get(
+          decompositionConfig.getContext(),
+          llvm::to_vector(llvm::make_filter_range(
```

**Comment:**
I'd make this vector a local variable to reduce the overall nesting.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir:106`

```diff
@@ -103,7 +103,7 @@ func.func @attention(%arg0: tensor<2x10x6x4xf16>, %arg1 : tensor<2x10x4x4xf16>,
 #compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
 func.func @attention_1(%arg0: tensor<2x10x6x4xf16>, %arg1 : tensor<2x10x4x4xf16>, %arg2 : tensor<2x10x4x4xf16>, %arg3 : f16) -> tensor<2x10x6x4xf16> attributes {translation_info = #iree_codegen.translation_info<pipeline = None subgroup_size = 32>} {
   %init = tensor.empty() : tensor<2x10x6x4xf16>
-  %result = iree_linalg_ext.attention {decomposition_config = {x}, indexing_maps = [#map, #map1, #map2, #map3, #map4], compilation_info = #compilation} ins(%arg0, %arg1, %arg2, %arg3 : tensor<2x10x6x4xf16>, tensor<2x10x4x4xf16>, tensor<2x10x4x4xf16>, f16) outs(%init : tensor<2x10x6x4xf16>) {
+  %result = iree_linalg_ext.attention {decomposition_config = {pv_attrs = {x}, qk_attrs = {y}}, indexing_maps = [#map, #map1, #map2, #map3, #map4], compilation_info = #compilation} ins(%arg0, %arg1, %arg2, %arg3 : tensor<2x10x6x4xf16>, tensor<2x10x4x4xf16>, tensor<2x10x4x4xf16>, f16) outs(%init : tensor<2x10x6x4xf16>) {
```

**Comment:**
Should we check have a check with `use_exp2` to make sure we don't drop it? I assume that this is accomplished by the `z` unit attr above, but this seems like something easy to miss to me.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:76`

```diff
@@ -69,19 +69,17 @@ struct StripAttentionOpCompilationInfo final
       eraseLoweringConfig(attentionOp);
     }
 
-    DictionaryAttr decompositionConfig =
-        attentionOp.getDecompositionConfigAttr();
-    if (decompositionConfig) {
-      decompositionConfig = DictionaryAttr::get(
+    if (DictionaryAttr decompositionConfig =
+            attentionOp.getDecompositionConfigAttr()) {
+      DictionaryAttr modifiedDecompositionConfig = DictionaryAttr::get(
           decompositionConfig.getContext(),
-          llvm::to_vector(llvm::make_filter_range(
-              decompositionConfig, [&](NamedAttribute attr) {
-                return attr.getName() !=
-                           IREE::LinalgExt::AttentionOp::getQKAttrStr() &&
-                       attr.getName() !=
-                           IREE::LinalgExt::AttentionOp::getPVAttrStr();
-              })));
-      attentionOp.setDecompositionConfigAttr(decompositionConfig);
+          llvm::filter_to_vector(decompositionConfig, [&](NamedAttribute attr) {
```

**Comment:**
```suggestion
          llvm::filter_to_vector(decompositionConfig, [](NamedAttribute attr) {
```
AFAICT we don't need to capture anything

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:74`

```diff
@@ -69,19 +69,17 @@ struct StripAttentionOpCompilationInfo final
       eraseLoweringConfig(attentionOp);
     }
 
-    DictionaryAttr decompositionConfig =
-        attentionOp.getDecompositionConfigAttr();
-    if (decompositionConfig) {
-      decompositionConfig = DictionaryAttr::get(
+    if (DictionaryAttr decompositionConfig =
+            attentionOp.getDecompositionConfigAttr()) {
+      DictionaryAttr modifiedDecompositionConfig = DictionaryAttr::get(
```

**Comment:**
nit: this is a bit of a mouthful and I don't think we need to be that descriptive here
```suggestion
      DictionaryAttr newConfig = DictionaryAttr::get(
```

---


---


## [PR #20072](https://github.com/iree-org/iree/pull/20072): [Codegen][Tuner] add support for attention op in the StripCompilationInfoPass

### Review Summary

**COMMENTED** (2025-02-24)


**COMMENTED** (2025-02-24)


**COMMENTED** (2025-02-24)


**APPROVED** (2025-02-24)

LGTM


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:60`

```diff
@@ -57,13 +56,34 @@ struct StripLinalgOpCompilationInfo final
   }
 };
 
+struct StripAttentionOpCompilationInfo
+    : public OpRewritePattern<IREE::LinalgExt::AttentionOp> {
```

**Comment:**
```suggestion
struct StripAttentionOpCompilationInfo final
    : OpRewritePattern<IREE::LinalgExt::AttentionOp> {
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:86`

```diff
@@ -57,13 +56,34 @@ struct StripLinalgOpCompilationInfo final
   }
 };
 
+struct StripAttentionOpCompilationInfo
+    : public OpRewritePattern<IREE::LinalgExt::AttentionOp> {
+  using OpRewritePattern::OpRewritePattern;
+  LogicalResult matchAndRewrite(IREE::LinalgExt::AttentionOp attentionOp,
+                                PatternRewriter &rewriter) const override {
+    if (getCompilationInfo(attentionOp)) {
+      eraseCompilationInfo(attentionOp);
+    }
+
+    if (getLoweringConfig(attentionOp)) {
+      eraseLoweringConfig(attentionOp);
+    }
+
+    if (attentionOp.getDecompositionConfigAttr()) {
+      attentionOp.removeDecompositionConfigAttr();
+    }
+    return success();
+  }
+};
+
 struct StripCompilationInfoPass final
     : impl::StripCompilationInfoPassBase<StripCompilationInfoPass> {
   void runOnOperation() override {
     MLIRContext *ctx = &getContext();
     RewritePatternSet patterns(ctx);
     patterns.add<StripFuncOpTranslationInfo>(ctx);
     patterns.add<StripLinalgOpCompilationInfo>(ctx);
+    patterns.add<StripAttentionOpCompilationInfo>(ctx);
```

**Comment:**
```suggestion
    patterns.add<StripFuncOpTranslationInfo, StripLinalgOpCompilationInfo, StripAttentionOpCompilationInfo>(ctx);
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:66`

```diff
@@ -57,13 +56,34 @@ struct StripLinalgOpCompilationInfo final
   }
 };
 
+struct StripAttentionOpCompilationInfo
+    : public OpRewritePattern<IREE::LinalgExt::AttentionOp> {
+  using OpRewritePattern::OpRewritePattern;
+  LogicalResult matchAndRewrite(IREE::LinalgExt::AttentionOp attentionOp,
+                                PatternRewriter &rewriter) const override {
+    if (getCompilationInfo(attentionOp)) {
+      eraseCompilationInfo(attentionOp);
+    }
```

**Comment:**
There's no test for this

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir:111`

```diff
@@ -82,5 +82,33 @@ func.func @attention(%arg0: tensor<2x10x6x4xf16>, %arg1 : tensor<2x10x4x4xf16>,
 
 // CHECK-LABEL: func.func @attention
 // CHECK-NOT:   iree_codegen.translation_info
+// CHECK-NOT:   iree_codegen.lowering_config
 // CHECK-NOT:   decomposition_config =
+
+
+// -----
+
+#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
+#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
+#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5)>
+#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
+#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
+#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
+#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
+#config = #iree_codegen.lowering_config<tile_sizes = [[128, 256], [16, 16]]>
+#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [16, 8, 1] subgroup_size = 64>
+#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
+func.func @attention_1(%arg0: tensor<2x10x6x4xf16>, %arg1 : tensor<2x10x4x4xf16>, %arg2 : tensor<2x10x4x4xf16>, %arg3 : f16) -> tensor<2x10x6x4xf16> attributes {translation_info = #iree_codegen.translation_info<pipeline = None subgroup_size = 32>} {
+  %init = tensor.empty() : tensor<2x10x6x4xf16>
+  %result = iree_linalg_ext.attention {decomposition_config = {x}, indexing_maps = [#map, #map1, #map2, #map3, #map4], compilation_info = #compilation} ins(%arg0, %arg1, %arg2, %arg3 : tensor<2x10x6x4xf16>, tensor<2x10x4x4xf16>, tensor<2x10x4x4xf16>, f16) outs(%init : tensor<2x10x6x4xf16>) {
+        ^bb0(%arg: f32):
+          iree_linalg_ext.yield %arg : f32
+        } -> tensor<2x10x6x4xf16>
+  return %result : tensor<2x10x6x4xf16>
+}
+
+// CHECK-LABEL: func.func @attention_1
+// CHECK-NOT:   iree_codegen.compilation_info
```

**Comment:**
This checks for IREE attributes instead of keys in the attribute dictionary. A potential issue is that without `--mlir-print-ir-scope`, these attributes may be outlined **above the function** just like in the input IR.

Have you tried adding a new attribute, say `foo = #compilation`, and checking that the test does fail when the attribute remains in the output?

I'd think that we should check for the dictionary keys and/or print with local scope.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir:112`

```diff
@@ -60,3 +60,60 @@ func.func @matmul_128x1024x256_1(%lhs : tensor<128x256xf32>, %rhs: tensor<256x10
 
 // CHECK-LABEL: func.func @matmul_128x1024x256_1
 // CHECK-NOT:   iree_codegen.lowering_config
+
+// -----
+
+#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
+#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
+#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5)>
+#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
+#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
+#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
+#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
+#config = #iree_codegen.lowering_config<tile_sizes = [[128, 256], [16, 16]]>
+func.func @attention(%arg0: tensor<2x10x6x4xf16>, %arg1 : tensor<2x10x4x4xf16>, %arg2 : tensor<2x10x4x4xf16>, %arg3 : f16) -> tensor<2x10x6x4xf16> attributes {translation_info = #iree_codegen.translation_info<pipeline = None subgroup_size = 32>} {
+  %init = tensor.empty() : tensor<2x10x6x4xf16>
+  %result = iree_linalg_ext.attention {decomposition_config = {x}, indexing_maps = [#map, #map1, #map2, #map3, #map4], lowering_config = #config} ins(%arg0, %arg1, %arg2, %arg3 : tensor<2x10x6x4xf16>, tensor<2x10x4x4xf16>, tensor<2x10x4x4xf16>, f16) outs(%init : tensor<2x10x6x4xf16>) {
+        ^bb0(%arg: f32):
+          iree_linalg_ext.yield %arg : f32
+        } -> tensor<2x10x6x4xf16>
+  return %result : tensor<2x10x6x4xf16>
+}
+
+// CHECK-LABEL: func.func @attention
+// CHECK-NOT:   iree_codegen.translation_info
+// CHECK-NOT:   iree_codegen.lowering_config
+// CHECK-NOT:   translation_info =
+// CHECK-NOT:   lowering_config =
+// CHECK-NOT:   decomposition_config =
+
+
+// -----
+
+#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
+#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
+#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5)>
+#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
+#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
+#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
+#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
+#config = #iree_codegen.lowering_config<tile_sizes = [[128, 256], [16, 16]]>
+#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [16, 8, 1] subgroup_size = 64>
+#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
+func.func @attention_1(%arg0: tensor<2x10x6x4xf16>, %arg1 : tensor<2x10x4x4xf16>, %arg2 : tensor<2x10x4x4xf16>, %arg3 : f16) -> tensor<2x10x6x4xf16> attributes {translation_info = #iree_codegen.translation_info<pipeline = None subgroup_size = 32>} {
+  %init = tensor.empty() : tensor<2x10x6x4xf16>
+  %result = iree_linalg_ext.attention {decomposition_config = {x}, indexing_maps = [#map, #map1, #map2, #map3, #map4], compilation_info = #compilation} ins(%arg0, %arg1, %arg2, %arg3 : tensor<2x10x6x4xf16>, tensor<2x10x4x4xf16>, tensor<2x10x4x4xf16>, f16) outs(%init : tensor<2x10x6x4xf16>) {
+        ^bb0(%arg: f32):
+          iree_linalg_ext.yield %arg : f32
+        } -> tensor<2x10x6x4xf16>
+  return %result : tensor<2x10x6x4xf16>
+}
+
+// CHECK-LABEL: func.func @attention_1
```

**Comment:**
Can you also match the attention op?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir:76`

```diff
@@ -60,3 +60,60 @@ func.func @matmul_128x1024x256_1(%lhs : tensor<128x256xf32>, %rhs: tensor<256x10
 
 // CHECK-LABEL: func.func @matmul_128x1024x256_1
 // CHECK-NOT:   iree_codegen.lowering_config
+
+// -----
+
+#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
+#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
+#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5)>
+#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
+#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
+#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
+#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
+#config = #iree_codegen.lowering_config<tile_sizes = [[128, 256], [16, 16]]>
+func.func @attention(%arg0: tensor<2x10x6x4xf16>, %arg1 : tensor<2x10x4x4xf16>, %arg2 : tensor<2x10x4x4xf16>, %arg3 : f16) -> tensor<2x10x6x4xf16> attributes {translation_info = #iree_codegen.translation_info<pipeline = None subgroup_size = 32>} {
+  %init = tensor.empty() : tensor<2x10x6x4xf16>
+  %result = iree_linalg_ext.attention {decomposition_config = {x}, indexing_maps = [#map, #map1, #map2, #map3, #map4], lowering_config = #config} ins(%arg0, %arg1, %arg2, %arg3 : tensor<2x10x6x4xf16>, tensor<2x10x4x4xf16>, tensor<2x10x4x4xf16>, f16) outs(%init : tensor<2x10x6x4xf16>) {
```

**Comment:**
Also here, it would be nice to check that the attention op is there

---


---


## [PR #20039](https://github.com/iree-org/iree/pull/20039): [Codegen][Tuner] add attention op into default tuning spec

### Review Summary

**CHANGES_REQUESTED** (2025-02-20)

This needs tests (both correctness and the expected performance improvements)


**COMMENTED** (2025-02-21)


**COMMENTED** (2025-02-21)


**COMMENTED** (2025-02-21)


**COMMENTED** (2025-02-21)


**COMMENTED** (2025-02-21)


**COMMENTED** (2025-02-24)

Looks good overall, thanks for the fixes. I wonder if we can reduce the test IR further.

Also, let's wait for review from @Groverkss before landing.


**COMMENTED** (2025-02-24)


**APPROVED** (2025-02-25)

Thanks for the cleanup, the tests look much more maintainable now.

Just a couple of remaining nits.


### Code Comments

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir:34`

```diff
@@ -19,6 +19,40 @@ transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.read
   transform.yield
 }
 
+transform.named_sequence @apply_attn_op_config(%attention: !transform.any_op {transform.readonly},
+                                                %config: !transform.any_param {transform.readonly},
+                                                %decomposition_config: !transform.any_param {transform.readonly}) {
+  transform.annotate %attention "compilation_info" = %config : !transform.any_op, !transform.any_param
+  transform.annotate %attention "decomposition_config" = %decomposition_config : !transform.any_op, !transform.any_param
+  transform.annotate %attention "__tuning_spec_applied__" : !transform.any_op
+  transform.yield
+}
+
+transform.named_sequence @match_attention_f16(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param, !transform.any_param) {
+    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
+    %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
+    transform.iree.match.cast_compatible_type %in0 = tensor<?x?x?x?xf16> : !transform.any_value
```

**Comment:**
Does this work for any attention size?

---

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir:34`

```diff
@@ -19,6 +19,40 @@ transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.read
   transform.yield
 }
 
+transform.named_sequence @apply_attn_op_config(%attention: !transform.any_op {transform.readonly},
+                                                %config: !transform.any_param {transform.readonly},
+                                                %decomposition_config: !transform.any_param {transform.readonly}) {
+  transform.annotate %attention "compilation_info" = %config : !transform.any_op, !transform.any_param
+  transform.annotate %attention "decomposition_config" = %decomposition_config : !transform.any_op, !transform.any_param
+  transform.annotate %attention "__tuning_spec_applied__" : !transform.any_op
+  transform.yield
+}
+
+transform.named_sequence @match_attention_f16(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param, !transform.any_param) {
+    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
+    %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
+    transform.iree.match.cast_compatible_type %in0 = tensor<?x?x?x?xf16> : !transform.any_value
```

**Comment:**
That's the source of my concern -- it applies to any attention size

---

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir:34`

```diff
@@ -19,6 +19,40 @@ transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.read
   transform.yield
 }
 
+transform.named_sequence @apply_attn_op_config(%attention: !transform.any_op {transform.readonly},
+                                                %config: !transform.any_param {transform.readonly},
+                                                %decomposition_config: !transform.any_param {transform.readonly}) {
+  transform.annotate %attention "compilation_info" = %config : !transform.any_op, !transform.any_param
+  transform.annotate %attention "decomposition_config" = %decomposition_config : !transform.any_op, !transform.any_param
+  transform.annotate %attention "__tuning_spec_applied__" : !transform.any_op
+  transform.yield
+}
+
+transform.named_sequence @match_attention_f16(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param, !transform.any_param) {
+    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
+    %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
+    transform.iree.match.cast_compatible_type %in0 = tensor<?x?x?x?xf16> : !transform.any_value
```

**Comment:**
> What is the expectation/goal here ?

To get good performance out of the box on attention in models we care about and don't error out on other attention variants we haven't seen.

This is the first step towards learning how to pick good default specs and how to test them, so that we figure out some process that we can use later on to add more default tuning specs for key contractions etc.

---

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir:34`

```diff
@@ -19,6 +19,40 @@ transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.read
   transform.yield
 }
 
+transform.named_sequence @apply_attn_op_config(%attention: !transform.any_op {transform.readonly},
+                                                %config: !transform.any_param {transform.readonly},
+                                                %decomposition_config: !transform.any_param {transform.readonly}) {
+  transform.annotate %attention "compilation_info" = %config : !transform.any_op, !transform.any_param
+  transform.annotate %attention "decomposition_config" = %decomposition_config : !transform.any_op, !transform.any_param
+  transform.annotate %attention "__tuning_spec_applied__" : !transform.any_op
+  transform.yield
+}
+
+transform.named_sequence @match_attention_f16(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param, !transform.any_param) {
+    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
+    %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
+    transform.iree.match.cast_compatible_type %in0 = tensor<?x?x?x?xf16> : !transform.any_value
```

**Comment:**
My intuition is to start small and match exactly the attention shapes we've tested this on. Later on we can do a sweep of attention sizes to see if we can have something more general.

---

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir:140`

```diff
@@ -66,7 +137,8 @@ transform.named_sequence
 @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) -> !transform.any_op
   attributes { iree_codegen.tuning_spec_entrypoint } {
   %res = transform.foreach_match in %variant_op
-    @match_mmt_2048x1280x5120_f16_f16_f32 -> @apply_op_config
+    @match_attention_2x10x4096x64x64x64_f16 -> @apply_attn_op_config
```

**Comment:**
Can you add a comment with the expected speedup (for posterity)?

---

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir:74`

```diff
@@ -19,6 +19,77 @@ transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.read
   transform.yield
 }
 
+transform.named_sequence @apply_attn_op_config(%attention: !transform.any_op {transform.readonly},
+                                                %config: !transform.any_param {transform.readonly},
+                                                %decomposition_config: !transform.any_param {transform.readonly}) {
+  transform.annotate %attention "compilation_info" = %config : !transform.any_op, !transform.any_param
+  transform.annotate %attention "decomposition_config" = %decomposition_config : !transform.any_op, !transform.any_param
+  transform.annotate %attention "__tuning_spec_applied__" : !transform.any_op
+  transform.yield
+}
+
+transform.named_sequence @match_attention_f16(%root: !transform.any_op {transform.readonly})
+  -> !transform.any_op {
+  transform.match.operation_name %root ["iree_linalg_ext.attention"] : !transform.any_op
+  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
+    ^bb0(%query: tensor<?x?x?x?xf16>,
+         %key: tensor<?x?x?x?xf16>,
+         %value: tensor<?x?x?x?xf16>,
+         %softmax_scale: f16,
+         %out: tensor<?x?x?x?xf16>):
+
+      %attn = iree_linalg_ext.attention {indexing_maps = [
+                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>,
+                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>,
+                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5)>,
+                                          affine_map<(d0, d1, d2, d3, d4, d5) -> ()>,
+                                          affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>]}
+        ins(%query, %key, %value, %softmax_scale :
+            tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, f16)
+        outs(%out : tensor<?x?x?x?xf16>){
+          ^bb0(%arg0: f32):
+            iree_linalg_ext.yield %arg0 : f32
+        } -> tensor<?x?x?x?xf16>
+  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
+
+  transform.yield %root : !transform.any_op
+}
+
+transform.named_sequence
+@match_attention_2x10x4096x64x64x64_f16(%attention: !transform.any_op {transform.readonly})
+  -> (!transform.any_op, !transform.any_param, !transform.any_param) {
+
+  %matched = transform.include @match_attention_f16 failures(propagate) (%attention)
+    : (!transform.any_op) -> !transform.any_op
+
+  %query = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
+  %key = transform.get_operand %attention[1] : (!transform.any_op) -> !transform.any_value
+  %value = transform.get_operand %attention[2] : (!transform.any_op) -> !transform.any_value
+
+  transform.iree.match.cast_compatible_type %query = tensor<2x10x4096x64xf16> : !transform.any_value
+  transform.iree.match.cast_compatible_type %key = tensor<2x10x64x64xf16> : !transform.any_value
+  transform.iree.match.cast_compatible_type %value = tensor<2x10x64x64xf16> : !transform.any_value
+
+  %config = transform.param.constant #iree_codegen.compilation_info<
+          lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 1, 64, 0, 0, 0], reduction=[0, 0, 0, 0, 0, 64], promote_operands = [1, 2]}>,
```

**Comment:**
I thought we wanted to do 1, 1, 128 for this shape?

---

**File:** `compiler/plugins/target/ROCM/builtins/tuning/test/spec_gfx942.mlir:88`

```diff
@@ -54,3 +54,58 @@ hal.executable public @main {
     }
   }
 }
+
+// -----
+
+// CHECK-LABEL:  func.func @attention_2x10x4096x64x64x64_f16
+// CHECK:          iree_linalg_ext.attention
+// CHECK-SAME:       __tuning_spec_applied__
+
+// MI300X-LABEL:  func.func @attention_2x10x4096x64x64x64_f16
+// MI300X:          iree_linalg_ext.attention
+// MI300X-SAME:       __tuning_spec_applied__
+
+hal.executable public @main {
+  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
+    hal.executable.export public @attention ordinal(0) layout(#hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) {
+    ^bb0(%arg0: !hal.device):
+      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
+      hal.return %x, %y, %z : index, index, index
+    }
+    builtin.module {
+      // expected-remark@+1 {{Applied transform configuration strategy @iree_default_tuning_spec_gfx942::@__kernel_config}}
+      func.func @attention_2x10x4096x64x64x64_f16() {
+        %c85251584 = arith.constant 85251584 : index
+        %c283904 = arith.constant 283904 : index
+        %cst = arith.constant 1.250000e-01 : f16
+        %0 = hal.interface.constant.load layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
+        %1 = hal.interface.constant.load layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
+        %2 = arith.index_castui %0 : i32 to index
+        %3 = arith.index_castui %1 : i32 to index
+        %4:2 = util.assume.int
+            %2[<umin = 101640704, umax = 101640704, udiv = 101640704>, <umin = 101640704, umax = 101640704, udiv = 101640704>, <umin = 74765824, umax = 74765824, udiv = 74765824>],
+            %3[<umin = 91154944, umax = 91154944, udiv = 91154944>, <umin = 91154944, umax = 91154944, udiv = 91154944>, <umin = 64280064, umax = 64280064, udiv = 64280064>]
+          : index, index
```

**Comment:**
Can we simplify this test and remove parts that are not necessary?

---

**File:** `compiler/plugins/target/ROCM/builtins/tuning/test/spec_gfx942.mlir:77`

```diff
@@ -54,3 +54,58 @@ hal.executable public @main {
     }
   }
 }
+
+// -----
+
+// CHECK-LABEL:  func.func @attention_2x10x4096x64x64x64_f16
+// CHECK:          iree_linalg_ext.attention
+// CHECK-SAME:       __tuning_spec_applied__
+
+// MI300X-LABEL:  func.func @attention_2x10x4096x64x64x64_f16
+// MI300X:          iree_linalg_ext.attention
+// MI300X-SAME:       __tuning_spec_applied__
+
+hal.executable public @main {
+  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
+    hal.executable.export public @attention ordinal(0) layout(#hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) {
+    ^bb0(%arg0: !hal.device):
+      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
+      hal.return %x, %y, %z : index, index, index
+    }
+    builtin.module {
+      // expected-remark@+1 {{Applied transform configuration strategy @iree_default_tuning_spec_gfx942::@__kernel_config}}
+      func.func @attention_2x10x4096x64x64x64_f16() {
```

**Comment:**
I think we should also have at least one negative test where the attention config is expected *not to apply*

---

**File:** `compiler/plugins/target/ROCM/builtins/tuning/test/spec_gfx942.mlir:98`

```diff
@@ -54,3 +54,107 @@ hal.executable public @main {
     }
   }
 }
+
+// -----
+
+// CHECK-LABEL:  func.func @attention_2x10x4096x64x64x64_f16
+// CHECK:          iree_linalg_ext.attention
+// CHECK-SAME:       __tuning_spec_applied__
+
+// MI300X-LABEL:  func.func @attention_2x10x4096x64x64x64_f16
+// MI300X:          iree_linalg_ext.attention
+// MI300X-SAME:       __tuning_spec_applied__
+
+#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
+#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
+#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5)>
+#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
+#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
+#pipeline_layout = #hal.pipeline.layout<bindings = [
+  #hal.pipeline.binding<storage_buffer>,
+  #hal.pipeline.binding<storage_buffer>,
+  #hal.pipeline.binding<storage_buffer>,
+  #hal.pipeline.binding<storage_buffer>
+]>
+hal.executable public @main {
+  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
+    hal.executable.export public @attention ordinal(0) layout(#hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) {
+    ^bb0(%arg0: !hal.device):
+      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
+      hal.return %x, %y, %z : index, index, index
+    }
+    builtin.module {
+      // expected-remark@+1 {{Applied transform configuration strategy @iree_default_tuning_spec_gfx942::@__kernel_config}}
+      func.func @attention_2x10x4096x64x64x64_f16() {
+        %cst = arith.constant 1.250000e-01 : f16
+        %c0 = arith.constant 0 : index
+        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x10x4096x64xf16>>
+        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x10x64x64xf16>>
+        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x10x64x64xf16>>
+        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x10x4096x64xf16>>
+        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 10, 4096, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x10x4096x64xf16>> -> tensor<2x10x4096x64xf16>
+        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [2, 10, 64, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x10x64x64xf16>> -> tensor<2x10x64x64xf16>
+        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [2, 10, 64, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x10x64x64xf16>> -> tensor<2x10x64x64xf16>
+        %7 = tensor.empty() : tensor<2x10x4096x64xf16>
```

**Comment:**
I wonder if we could further reduce this by making these function arguments (even if it doesn't make sense in full compilation) and drop all of these hal and flow ops?

---

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir:140`

```diff
@@ -66,7 +137,9 @@ transform.named_sequence
 @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) -> !transform.any_op
   attributes { iree_codegen.tuning_spec_entrypoint } {
   %res = transform.foreach_match in %variant_op
-    @match_mmt_2048x1280x5120_f16_f16_f32 -> @apply_op_config
+    // Notice approximately 1.22x speedup over the baseline by using this atten_op_config
```

**Comment:**
```suggestion
    // Expected speedup: 1.22x.
```

---

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir:22`

```diff
@@ -19,6 +19,77 @@ transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.read
   transform.yield
 }
 
+transform.named_sequence @apply_attn_op_config(%attention: !transform.any_op {transform.readonly},
```

**Comment:**
We have to start somewhere. I'd remove the warning or reword it to something along the lines that this is work in progress.

---

**File:** `compiler/plugins/target/ROCM/builtins/tuning/test/spec_gfx942.mlir:97`

```diff
@@ -54,3 +54,113 @@ hal.executable public @main {
     }
   }
 }
+
+// -----
+
+// CHECK-LABEL:  func.func @attention_2x10x4096x64x64x64_f16
+// CHECK:          iree_linalg_ext.attention
+// CHECK-SAME:       __tuning_spec_applied__
+
+// MI300X-LABEL:  func.func @attention_2x10x4096x64x64x64_f16
+// MI300X:          iree_linalg_ext.attention
+// MI300X-SAME:       __tuning_spec_applied__
+
+#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
+#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
+#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5)>
+#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
+#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
+#pipeline_layout = #hal.pipeline.layout<bindings = [
+  #hal.pipeline.binding<storage_buffer>,
+  #hal.pipeline.binding<storage_buffer>,
+  #hal.pipeline.binding<storage_buffer>,
+  #hal.pipeline.binding<storage_buffer>
+]>
+hal.executable public @main {
+  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
+    hal.executable.export public @attention ordinal(0) layout(#pipeline_layout) {
+    ^bb0(%arg0: !hal.device):
+      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
+      hal.return %x, %y, %z : index, index, index
+    }
+    builtin.module {
+      // expected-remark@+1 {{Applied transform configuration strategy @iree_default_tuning_spec_gfx942::@__kernel_config}}
+      func.func @attention_2x10x4096x64x64x64_f16(
+        %query: tensor<2x10x4096x64xf16>,
+        %key: tensor<2x10x64x64xf16>,
+        %value: tensor<2x10x64x64xf16>
+      ) -> tensor<2x10x4096x64xf16> {
+
+        %cst = arith.constant 1.250000e-01 : f16
+        %output = tensor.empty() : tensor<2x10x4096x64xf16>
+
+        // Apply the attention operation directly to function inputs
```

**Comment:**
```suggestion
        // Apply the attention operation directly to function inputs.
```
https://llvm.org/docs/CodingStandards.html#commenting

---

**File:** `compiler/plugins/target/ROCM/builtins/tuning/test/spec_gfx942.mlir:152`

```diff
@@ -54,3 +54,113 @@ hal.executable public @main {
     }
   }
 }
+
+// -----
+
+// CHECK-LABEL:  func.func @attention_2x10x4096x64x64x64_f16
+// CHECK:          iree_linalg_ext.attention
+// CHECK-SAME:       __tuning_spec_applied__
+
+// MI300X-LABEL:  func.func @attention_2x10x4096x64x64x64_f16
+// MI300X:          iree_linalg_ext.attention
+// MI300X-SAME:       __tuning_spec_applied__
+
+#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
+#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
+#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5)>
+#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
+#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
+#pipeline_layout = #hal.pipeline.layout<bindings = [
+  #hal.pipeline.binding<storage_buffer>,
+  #hal.pipeline.binding<storage_buffer>,
+  #hal.pipeline.binding<storage_buffer>,
+  #hal.pipeline.binding<storage_buffer>
+]>
+hal.executable public @main {
+  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
+    hal.executable.export public @attention ordinal(0) layout(#pipeline_layout) {
+    ^bb0(%arg0: !hal.device):
+      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
+      hal.return %x, %y, %z : index, index, index
+    }
+    builtin.module {
+      // expected-remark@+1 {{Applied transform configuration strategy @iree_default_tuning_spec_gfx942::@__kernel_config}}
+      func.func @attention_2x10x4096x64x64x64_f16(
+        %query: tensor<2x10x4096x64xf16>,
+        %key: tensor<2x10x64x64xf16>,
+        %value: tensor<2x10x64x64xf16>
+      ) -> tensor<2x10x4096x64xf16> {
+
+        %cst = arith.constant 1.250000e-01 : f16
+        %output = tensor.empty() : tensor<2x10x4096x64xf16>
+
+        // Apply the attention operation directly to function inputs
+        %result = iree_linalg_ext.attention {
+            indexing_maps = [#map, #map1, #map2, #map3, #map4]
+        } ins(%query, %key, %value, %cst :
+            tensor<2x10x4096x64xf16>, tensor<2x10x64x64xf16>, tensor<2x10x64x64xf16>, f16)
+          outs(%output : tensor<2x10x4096x64xf16>) {
+            ^bb0(%arg0: f32):
+              iree_linalg_ext.yield %arg0 : f32
+          } -> tensor<2x10x4096x64xf16>
+
+        return %result : tensor<2x10x4096x64xf16>
+      }
+    }
+  }
+}
+
+// -----
+
+// CHECK-LABEL:  func.func @attention_3x10x4096x64x64x32_f16
+// CHECK:          iree_linalg_ext.attention
+// CHECK-NOT:       __tuning_spec_applied__
+
+// MI300X-LABEL:  func.func @attention_3x10x4096x64x64x32_f16
+// MI300X:          iree_linalg_ext.attention
+// MI300X-NOT:       __tuning_spec_applied__
+
+#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
+#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
+#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5)>
+#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
+#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
+#pipeline_layout = #hal.pipeline.layout<bindings = [
+  #hal.pipeline.binding<storage_buffer>,
+  #hal.pipeline.binding<storage_buffer>,
+  #hal.pipeline.binding<storage_buffer>,
+  #hal.pipeline.binding<storage_buffer>
+]>
+hal.executable public @main {
+  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
+    hal.executable.export public @attention ordinal(0) layout(#pipeline_layout) {
+    ^bb0(%arg0: !hal.device):
+      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
+      hal.return %x, %y, %z : index, index, index
+    }
+    builtin.module {
+      // expected-remark@+1 {{Applied transform configuration strategy @iree_default_tuning_spec_gfx942::@__kernel_config}}
+      func.func @attention_3x10x4096x64x64x32_f16(
+        %query: tensor<3x10x4096x64xf16>,
+        %key: tensor<3x10x32x64xf16>,
+        %value: tensor<3x10x64x32xf16>
+      ) -> tensor<3x10x4096x64xf16> {
+
+        %cst = arith.constant 1.250000e-01 : f16
+        %output = tensor.empty() : tensor<3x10x4096x64xf16>
+
+        // Apply the attention operation directly to function inputs
```

**Comment:**
```suggestion
        // Apply the attention operation directly to function inputs.
```

---


---


## [PR #19762](https://github.com/iree-org/iree/pull/19762): [Codegen][Tuner] Add support for per-sku tuning spec

### Review Summary

**CHANGES_REQUESTED** (2025-01-22)


**COMMENTED** (2025-01-22)


**COMMENTED** (2025-01-22)


**COMMENTED** (2025-01-22)


**COMMENTED** (2025-01-23)


**COMMENTED** (2025-01-23)

Looks much better now


**COMMENTED** (2025-01-23)


**COMMENTED** (2025-01-23)


**COMMENTED** (2025-01-23)


**APPROVED** (2025-01-23)

This LGTM % the minor issue above. But let's wait for one more approval before landing, maybe from @MaheshRavishankar or @qedawkins.


**COMMENTED** (2025-01-23)


**COMMENTED** (2025-01-24)


**APPROVED** (2025-01-24)


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp:130`

```diff
@@ -125,12 +125,25 @@ getDefaultTuningSpec(ModuleOp module,
 
   // Try to look up the default tuning spec for this architecture, if any.
   StringRef arch = gpuTarget.getArch();
+  std::optional<std::string> sku = gpuTarget.getSKU();
   std::string defaultTuningSpecName =
       llvm::formatv("iree_default_tuning_spec_{}.mlir", arch);
```

**Comment:**
This can now be moved into the code block where it's used.

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.cpp:1097`

```diff
@@ -1094,6 +1094,23 @@ bool TargetAttr::supportsSyncMMAOps() const {
   return false;
 }
 
+std::optional<std::string> TargetAttr::getSKU() const {
```

**Comment:**
This is not the best location for this code -- generic gpu attributes shouldn't know about the exact hardware details.

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.cpp:1097`

```diff
@@ -1094,6 +1094,23 @@ bool TargetAttr::supportsSyncMMAOps() const {
   return false;
 }
 
+std::optional<std::string> TargetAttr::getSKU() const {
```

**Comment:**
Why not return `optional<StringRef>`?

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.cpp:1111`

```diff
@@ -1094,6 +1094,23 @@ bool TargetAttr::supportsSyncMMAOps() const {
   return false;
 }
 
+std::optional<std::string> TargetAttr::getSKU() const {
+  StringRef arch = getArch();
+  if (arch == "gfx942") {
+    TargetChipAttr chip = getChip();
+    if (chip) {
+      if (chip.getWgpCount() == 304) {
+        return "mi300x";
+      } else if (chip.getWgpCount() == 228) {
+        return "mi300a";
+      } else if (chip.getWgpCount() == 80) {
+        return "mi308x";
+      }
+    }
+  }
+  return std::nullopt;
```

**Comment:**
Please update this code to follow the llvm coding standards: https://llvm.org/docs/CodingStandards.html

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp:958`

```diff
@@ -954,6 +954,8 @@ IREE::GPU::TargetAttr getCLGPUTarget(MLIRContext *context) {
       backend = "cuda";
     else if (StringRef(clTestTarget).starts_with("gfx"))
       backend = "hip";
+    else if (StringRef(clTestTarget).starts_with("mi"))
+      backend = "hip";
```

**Comment:**
I don't think we need to change this -- we can use the existing way of specifying targets and backends separately.

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.td:504`

```diff
@@ -499,6 +499,28 @@ def IREEGPU_TargetAttr : AttrDef<IREEGPU_Dialect, "Target"> {
     bool supportsTF32InputMMAOps() const;
     // Returns true if this target supports TensorCore synchronized MMA ops.
     bool supportsSyncMMAOps() const;
+    // Returns the SKU of the target GPU if available.
+    std::optional<StringRef> getSKU() const {
+        llvm::StringRef arch = getArch();
```

**Comment:**
I think we should move this code somewhere close to or in `KnownTargets.cpp`; The generic GPU attributes shouldn't know anything about the rocm backend details and hip targets.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp:146`

```diff
@@ -123,14 +123,28 @@ getDefaultTuningSpec(ModuleOp module,
     return failure();
   }
 
-  // Try to look up the default tuning spec for this architecture, if any.
-  StringRef arch = gpuTarget.getArch();
-  std::string defaultTuningSpecName =
-      llvm::formatv("iree_default_tuning_spec_{}.mlir", arch);
+  std::optional<StringRef> sku = gpuTarget.getSKU();
+  std::string defaultTuningSpecName;
   std::optional<StringRef> defaultTuningSpecSource;
-  EmbeddedDataDirectory::withGlobal([&](EmbeddedDataDirectory &dir) {
-    defaultTuningSpecSource = dir.getFile(defaultTuningSpecName);
-  });
+  if (sku) {
+    // Try to look up the default tuning spec for this sku.
+    defaultTuningSpecName =
+        llvm::formatv("iree_default_tuning_spec_{}.mlir", sku);
+    EmbeddedDataDirectory::withGlobal([&](EmbeddedDataDirectory &dir) {
+      defaultTuningSpecSource = dir.getFile(defaultTuningSpecName);
+    });
+  }
+  if (!defaultTuningSpecSource) {
+    // If SKU-specific spec is not found, fall back to the default
+    // architecture-based tuning spec.
+    StringRef arch = gpuTarget.getArch();
+    defaultTuningSpecName =
+        llvm::formatv("iree_default_tuning_spec_{}.mlir", arch);
+    EmbeddedDataDirectory::withGlobal([&](EmbeddedDataDirectory &dir) {
+      defaultTuningSpecSource = dir.getFile(defaultTuningSpecName);
+    });
+  }
```

**Comment:**
This function is getting a bit long, I think it'd be better to outline this code to a helper function.

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.td:506`

```diff
@@ -499,6 +499,28 @@ def IREEGPU_TargetAttr : AttrDef<IREEGPU_Dialect, "Target"> {
     bool supportsTF32InputMMAOps() const;
     // Returns true if this target supports TensorCore synchronized MMA ops.
     bool supportsSyncMMAOps() const;
+    // Returns the SKU of the target GPU if available.
+    std::optional<StringRef> getSKU() const {
+        llvm::StringRef arch = getArch();
+
+        if (arch == "gfx942") {
```

**Comment:**
Invert this condition and use an early return: https://llvm.org/docs/CodingStandards.html#use-early-exits-and-continue-to-simplify-code

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp:708`

```diff
@@ -693,6 +693,30 @@ StringRef normalizeHIPTarget(StringRef target) {
   return normalizeAMDGPUTarget(target);
 }
 
+std::optional<StringRef> getAMDSKU(TargetAttr target) {
+  StringRef arch = target.getArch();
+
+  if (arch != "gfx942") {
+    return std::nullopt;
+  }
+
+  TargetChipAttr chip = target.getChip();
+  if (chip) {
+    uint32_t wgpCount = chip.getWgpCount();
+    if (wgpCount == 304) {
+      return "mi300x";
+    }
```

**Comment:**
Thanks for all the fixes around this code. I spend some time thinking about this approach of raising the target attribute back to the sku and wasn't sure if it's a good idea or not. It seems simple and thought that it may be good enough for amdgpu for now, but I found a counterexample: mi325. It has the same number of CUs as mi300, but the performance characteristics are different.

I think that to make this robust, we have to go back to [what I suggested previously](https://github.com/iree-org/iree/pull/19748#discussion_r1924336701) and record the sku in the target attribute itself, similar to how we keep the target arch around today.

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp:57`

```diff
@@ -54,6 +54,7 @@ struct WgpDetails {
 // Chip level feature/limit details
 struct ChipDetails {
   uint32_t wgpCount;
+  std::optional<llvm::StringRef> sku;
```

**Comment:**
```suggestion
  std::optional<StringRef> sku;
```

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp:123`

```diff
@@ -116,9 +117,13 @@ TargetAttr createTargetAttr(const TargetDetails &details, StringRef arch,
       DictionaryAttr{});
 
   TargetChipAttr targetChip;
-  if (details.chip)
-    targetChip =
-        TargetChipAttr::get(context, details.chip->wgpCount, DictionaryAttr{});
+  if (details.chip) {
+    StringAttr skuAttr = details.chip->sku
+                             ? StringAttr::get(context, *(details.chip->sku))
+                             : StringAttr::get(context, "");
```

**Comment:**
optional has a `.value_or` function, we should use it here. https://en.cppreference.com/w/cpp/utility/optional/value_or

---

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp:140`

```diff
@@ -123,14 +135,28 @@ getDefaultTuningSpec(ModuleOp module,
     return failure();
   }
 
-  // Try to look up the default tuning spec for this architecture, if any.
-  StringRef arch = gpuTarget.getArch();
-  std::string defaultTuningSpecName =
-      llvm::formatv("iree_default_tuning_spec_{}.mlir", arch);
+  std::optional<StringRef> sku;
+  if (IREE::GPU::TargetChipAttr chip = gpuTarget.getChip()) {
+    StringAttr chipSku = chip.getSku();
```

**Comment:**
Since chipSKU is already an optional attribute, I'd not expect to also find empty strings here. We can add an assertion for this.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp:142`

```diff
@@ -123,14 +135,28 @@ getDefaultTuningSpec(ModuleOp module,
     return failure();
   }
 
-  // Try to look up the default tuning spec for this architecture, if any.
-  StringRef arch = gpuTarget.getArch();
-  std::string defaultTuningSpecName =
-      llvm::formatv("iree_default_tuning_spec_{}.mlir", arch);
+  std::optional<StringRef> sku;
+  if (IREE::GPU::TargetChipAttr chip = gpuTarget.getChip()) {
+    std::optional<StringAttr> chipSku = chip.getSku();
+    if (chipSku) {
+      sku = (*chipSku).getValue();
```

**Comment:**
```suggestion
    if (std::optional<StringAttr> chipSku = chip.getSku()) {
      sku = chipSku->getValue();
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp:149`

```diff
@@ -123,14 +135,28 @@ getDefaultTuningSpec(ModuleOp module,
     return failure();
   }
 
-  // Try to look up the default tuning spec for this architecture, if any.
-  StringRef arch = gpuTarget.getArch();
-  std::string defaultTuningSpecName =
-      llvm::formatv("iree_default_tuning_spec_{}.mlir", arch);
+  std::optional<StringRef> sku;
+  if (IREE::GPU::TargetChipAttr chip = gpuTarget.getChip()) {
+    std::optional<StringAttr> chipSku = chip.getSku();
+    if (chipSku) {
+      sku = (*chipSku).getValue();
+    }
+  }
+
+  std::string defaultTuningSpecName;
   std::optional<StringRef> defaultTuningSpecSource;
-  EmbeddedDataDirectory::withGlobal([&](EmbeddedDataDirectory &dir) {
-    defaultTuningSpecSource = dir.getFile(defaultTuningSpecName);
-  });
+  if (sku) {
+    // Try to look up the default tuning spec for this sku.
```

**Comment:**
This comment doesn't clarify much beyond what the code does. Focus on **why** when writing comments, not **what**.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp:155`

```diff
@@ -123,14 +135,28 @@ getDefaultTuningSpec(ModuleOp module,
     return failure();
   }
 
-  // Try to look up the default tuning spec for this architecture, if any.
-  StringRef arch = gpuTarget.getArch();
-  std::string defaultTuningSpecName =
-      llvm::formatv("iree_default_tuning_spec_{}.mlir", arch);
+  std::optional<StringRef> sku;
+  if (IREE::GPU::TargetChipAttr chip = gpuTarget.getChip()) {
+    std::optional<StringAttr> chipSku = chip.getSku();
+    if (chipSku) {
+      sku = (*chipSku).getValue();
+    }
+  }
+
+  std::string defaultTuningSpecName;
   std::optional<StringRef> defaultTuningSpecSource;
-  EmbeddedDataDirectory::withGlobal([&](EmbeddedDataDirectory &dir) {
-    defaultTuningSpecSource = dir.getFile(defaultTuningSpecName);
-  });
+  if (sku) {
+    // Try to look up the default tuning spec for this sku.
+    defaultTuningSpecSource = fetchDefaultTuningSpec(*sku);
+  }
+
+  if (!defaultTuningSpecSource) {
+    // If SKU-specific spec is not found, fall back to the default
+    // architecture-based tuning spec.
```

**Comment:**
Here, the comment is very useful because it explains why we are attempting to fetch this spec.

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp:125`

```diff
@@ -116,9 +117,15 @@ TargetAttr createTargetAttr(const TargetDetails &details, StringRef arch,
       DictionaryAttr{});
 
   TargetChipAttr targetChip;
-  if (details.chip)
-    targetChip =
-        TargetChipAttr::get(context, details.chip->wgpCount, DictionaryAttr{});
+  if (details.chip) {
+    std::optional<StringAttr> skuAttr =
+        details.chip->sku && !details.chip->sku->empty()
+            ? std::optional<StringAttr>(
+                  StringAttr::get(context, *details.chip->sku))
+            : std::nullopt;
```

**Comment:**
This ternary is very complex. Imo this should be an if statement.

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.td:420`

```diff
@@ -416,6 +416,8 @@ def IREEGPU_TargetChipAttr : AttrDef<IREEGPU_Dialect, "TargetChip"> {
   let parameters = (ins
     "uint32_t":$wgp_count,
 
+    // An optional SKU identifier to distinguish different models.
+    OptionalParameter<"std::optional<StringAttr>">:$sku,
```

**Comment:**
I don't think this should be double-optional

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp:122`

```diff
@@ -116,9 +117,14 @@ TargetAttr createTargetAttr(const TargetDetails &details, StringRef arch,
       DictionaryAttr{});
 
   TargetChipAttr targetChip;
-  if (details.chip)
-    targetChip =
-        TargetChipAttr::get(context, details.chip->wgpCount, DictionaryAttr{});
+  if (details.chip) {
+    std::optional<StringAttr> skuAttr = std::nullopt;
+    if (details.chip->sku && !details.chip->sku->empty()) {
```

**Comment:**
When can the sku be present but empty? I don't think this happens with the current code.

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp:122`

```diff
@@ -116,9 +117,14 @@ TargetAttr createTargetAttr(const TargetDetails &details, StringRef arch,
       DictionaryAttr{});
 
   TargetChipAttr targetChip;
-  if (details.chip)
-    targetChip =
-        TargetChipAttr::get(context, details.chip->wgpCount, DictionaryAttr{});
+  if (details.chip) {
+    std::optional<StringAttr> skuAttr = std::nullopt;
+    if (details.chip->sku && !details.chip->sku->empty()) {
```

**Comment:**
I don't think this is something we have to check at all

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp:123`

```diff
@@ -116,9 +117,12 @@ TargetAttr createTargetAttr(const TargetDetails &details, StringRef arch,
       DictionaryAttr{});
 
   TargetChipAttr targetChip;
-  if (details.chip)
-    targetChip =
-        TargetChipAttr::get(context, details.chip->wgpCount, DictionaryAttr{});
+  if (details.chip) {
+    StringAttr skuAttr =
+        StringAttr::get(context, details.chip->sku.value_or(""));
+    targetChip = TargetChipAttr::get(context, details.chip->wgpCount, skuAttr,
```

**Comment:**
If sku is optional, why are we setting it with an empty string? Is empty string considered the same as `nullptr` in `StringAttr`?

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp:122`

```diff
@@ -116,9 +117,12 @@ TargetAttr createTargetAttr(const TargetDetails &details, StringRef arch,
       DictionaryAttr{});
 
   TargetChipAttr targetChip;
-  if (details.chip)
-    targetChip =
-        TargetChipAttr::get(context, details.chip->wgpCount, DictionaryAttr{});
+  if (details.chip) {
+    StringAttr skuAttr =
+        StringAttr::get(context, details.chip->sku.value_or(""));
```

**Comment:**
```suggestion
    auto skuAttr =
        StringAttr::get(context, details.chip->sku.value_or(""));
```
The type is obvious based on the RHS: https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---


---


## [PR #19756](https://github.com/iree-org/iree/pull/19756): [Codegen] add mi308x target

### Review Summary

**COMMENTED** (2025-01-22)


**COMMENTED** (2025-01-22)

Can you link to the relevant issue in the PR description?


**APPROVED** (2025-01-22)

LGTM


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp:343`

```diff
@@ -338,7 +340,7 @@ StringRef normalizeAMDGPUTarget(StringRef target) {
     return target;
 
   return llvm::StringSwitch<StringRef>(target.lower())
-      .Cases("mi300x", "mi300a", "gfx942")
+      .Cases("mi300x", "mi300a", "mi308x", "gfx942")
```

**Comment:**
I'd think we should append them to keep consistent with `.Cases("cdna3", "gfx940", "gfx941", "gfx942",` above where newer targets are placed towards the end.

Ultimately I don't care as long as we maintain some consistent ordering.

---


---


## [PR #19748](https://github.com/iree-org/iree/pull/19748): [Codegen][Tuner] default tuning spec available per-SKU

### Review Summary

**CHANGES_REQUESTED** (2025-01-21)

We should split this into three (or more) PRs:
* Add a known target for mi308
* Implement support for per-sku tuning specs
* Populate tuning specs for mi308


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp:37`

```diff
@@ -34,6 +34,8 @@
 #define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
 #define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
 
+extern llvm::cl::opt<std::string> clTestTarget;
```

**Comment:**
In general, we should no reference global variables like this. This makes the code hard to maintain and LLVM converged on external storage for flags meant to be external, and having these flag storage variables declared in headers.

Here specifically, we should not rely on test flags in this code. All the information we use must come from the gpu target attr. If the information we need there is not available, we should work on adding it.

---

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_mi308x.mlir:243`

```diff
@@ -0,0 +1,263 @@
+// RUN: iree-opt %s
+
+// This is just an initial tuning spec for mi308x and is not intended for
+// production use.
+// TODO(https://github.com/iree-org/iree/issues/19214): Add missing
+// configurations to this spec.
+
+module @iree_default_tuning_spec_mi308x attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint} {
+//===----------------------------------------------------------------------===//
+// Tuning infra
+//===----------------------------------------------------------------------===//
+
+transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly},
+                                          %config: !transform.any_param {transform.readonly}) {
+  transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
+  transform.annotate %op "__tuning_spec_applied__" : !transform.any_op
+  transform.yield
+}
+
+transform.named_sequence @apply_attn_op_config(%attention: !transform.any_op {transform.readonly},
+                                                %config: !transform.any_param {transform.readonly},
+                                                %decomposition_config: !transform.any_param {transform.readonly}) {
+  transform.annotate %attention "compilation_info" = %config : !transform.any_op, !transform.any_param
+  transform.annotate %attention "decomposition_config" = %decomposition_config : !transform.any_op, !transform.any_param
+  // transform.print %attention {name = "Applied attention config"} : !transform.any_op
+  transform.yield
+}
+
+transform.named_sequence @match_attention_f16(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param, !transform.any_param) {
+  transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
+  %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
+  transform.iree.match.cast_compatible_type %in0 = tensor<?x?x?x?xf16> : !transform.any_value
+
+  %config = transform.param.constant #iree_codegen.compilation_info<
+          lowering_config = #iree_gpu.lowering_config<{workgroup = [1, 1, 64, 0, 0, 0], reduction=[0, 0, 0, 0, 0, 64], promote_operands = [1, 2]}>,
+          translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
+                                                            workgroup_size = [64, 4]
+                                                            subgroup_size = 64 ,
+            {llvm_func_attrs = { "amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign" }}>>
+  -> !transform.any_param
+
+  %decomposition_config = transform.param.constant {
+    qk_attrs = {attention_qk_matmul,
+                lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.virtual_mma_layout<intrinsic = VMFMA_F32_32x32x16_F16>,
+                                                              subgroup_m_count = 4, subgroup_n_count = 1, promote_operands = [1] }>},
+    pv_attrs = {attention_pv_matmul,
+                lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
+                                                              subgroup_m_count = 4, subgroup_n_count = 1, promote_operands = [1] }>}
+  } -> !transform.any_param
+
+  transform.yield %attention, %config, %decomposition_config : !transform.any_op, !transform.any_param, !transform.any_param
+}
+
+transform.named_sequence @match_mmt_f16_f16_f32(%root: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
+  transform.match.operation_name %root ["linalg.generic"] : !transform.any_op
+  // transform.print %root {name = "Generic"} : !transform.any_op
+  %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
+    ^bb0(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>, %out: tensor<?x?xf32>):
+    %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
+                                          affine_map<(d0, d1, d2) -> (d1, d2)>,
+                                          affine_map<(d0, d1, d2) -> (d0, d1)>],
+                         iterator_types = ["parallel", "parallel", "reduction"]}
+        ins(%lhs, %rhs : tensor<?x?xf16>, tensor<?x?xf16>) outs(%out : tensor<?x?xf32>) {
+      ^bb0(%in: f16, %in_0: f16, %acc: f32):
+        %18 = arith.extf %in : f16 to f32
+        %19 = arith.extf %in_0 : f16 to f32
+        %20 = arith.mulf %18, %19 : f32
+        %21 = arith.addf %acc, %20 : f32
+        linalg.yield %21 : f32
+      } -> tensor<?x?xf32>
+  } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
+  transform.yield %root : !transform.any_op
+}
+
+// TUNING_SPEC_BEGIN DO NOT REMOVE
+
+//===----------------------------------------------------------------------===//
+// Matmul tuning
+//===----------------------------------------------------------------------===//
+
+transform.named_sequence @match_mmt_1920x10240x1280(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
+  %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
+  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
+  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
+  transform.iree.match.cast_compatible_type %lhs = tensor<1920x1280xf16> : !transform.any_value
+  transform.iree.match.cast_compatible_type %rhs = tensor<10240x1280xf16> : !transform.any_value
+  %config = transform.param.constant #iree_codegen.compilation_info<
+  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
+                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
+                                                subgroup_m_count = 4, subgroup_n_count = 2,
+                                                reduction = [0, 0, 32],
+                                                workgroup = [128, 128, 0]}>,
+  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
+    workgroup_size = [128, 4, 1] subgroup_size = 64,
+    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>,
+     llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}
+    }>> -> !transform.any_param
+  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
+}
+
+transform.named_sequence @match_mmt_1920x1280x1280(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
+  %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
+  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
+  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
+  transform.iree.match.cast_compatible_type %lhs = tensor<1920x1280xf16> : !transform.any_value
+  transform.iree.match.cast_compatible_type %rhs = tensor<1280x1280xf16> : !transform.any_value
+  %config = transform.param.constant #iree_codegen.compilation_info<
+  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
+                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
+                                                subgroup_m_count = 4, subgroup_n_count = 2,
+                                                reduction = [0, 0, 32],
+                                                workgroup = [128, 128, 0]}>,
+  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
+    workgroup_size = [128, 4, 1] subgroup_size = 64,
+    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>,
+     llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}
+    }>> -> !transform.any_param
+  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
+}
+
+transform.named_sequence @match_mmt_1920x1280x5120(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
+  %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
+  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
+  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
+  transform.iree.match.cast_compatible_type %lhs = tensor<1920x5120xf16> : !transform.any_value
+  transform.iree.match.cast_compatible_type %rhs = tensor<1280x5120xf16> : !transform.any_value
+  %config = transform.param.constant #iree_codegen.compilation_info<
+  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
+                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
+                                                subgroup_m_count = 4, subgroup_n_count = 2,
+                                                reduction = [0, 0, 32],
+                                                workgroup = [128, 128, 0]}>,
+  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
+    workgroup_size = [128, 4, 1] subgroup_size = 64,
+    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>,
+     llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}
+    }>> -> !transform.any_param
+  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
+}
+
+transform.named_sequence @match_mmt_7680x5120x640(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
+  %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
+  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
+  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
+  transform.iree.match.cast_compatible_type %lhs = tensor<7680x640xf16> : !transform.any_value
+  transform.iree.match.cast_compatible_type %rhs = tensor<5120x640xf16> : !transform.any_value
+  %config = transform.param.constant #iree_codegen.compilation_info<
+  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
+                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
+                                                subgroup_m_count = 2, subgroup_n_count = 4,
+                                                reduction = [0, 0, 32],
+                                                workgroup = [128, 256, 0]}>,
+  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
+    workgroup_size = [256, 2, 1] subgroup_size = 64,
+    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>,
+     llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}
+    }>> -> !transform.any_param
+  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
+}
+
+transform.named_sequence @match_mmt_128x1280x2048(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
+  %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
+  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
+  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
+  transform.iree.match.cast_compatible_type %lhs = tensor<1280x2048xf16> : !transform.any_value
+  transform.iree.match.cast_compatible_type %rhs = tensor<1280x2048xf16> : !transform.any_value
+  %config = transform.param.constant #iree_codegen.compilation_info<
+  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
+                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
+                                                subgroup_m_count = 2, subgroup_n_count = 1,
+                                                reduction = [0, 0, 128],
+                                                workgroup = [64, 16, 0]}>,
+  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
+    workgroup_size = [64, 2, 1] subgroup_size = 64,
+    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>,
+     llvm_func_attrs = {"amdgpu-waves-per-eu" = "2"}
+    }>> -> !transform.any_param
+  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
+}
+
+transform.named_sequence @match_mmt_7680x640x640(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
+  %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
+  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
+  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
+  transform.iree.match.cast_compatible_type %lhs = tensor<7680x640xf16> : !transform.any_value
+  transform.iree.match.cast_compatible_type %rhs = tensor<640x640xf16> : !transform.any_value
+  %config = transform.param.constant #iree_codegen.compilation_info<
+  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
+                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
+                                                subgroup_m_count = 1, subgroup_n_count = 4,
+                                                reduction = [0, 0, 32],
+                                                workgroup = [256, 128, 0]}>,
+  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
+    workgroup_size = [256, 1, 1] subgroup_size = 64,
+    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>,
+     llvm_func_attrs = {"amdgpu-waves-per-eu" = "1"}
+    }>> -> !transform.any_param
+  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
+}
+
+transform.named_sequence @match_mmt_7680x640x2560(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
+  %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
+  %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
+  %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
+  transform.iree.match.cast_compatible_type %lhs = tensor<7680x2560xf16> : !transform.any_value
+  transform.iree.match.cast_compatible_type %rhs = tensor<640x2560xf16> : !transform.any_value
+  %config = transform.param.constant #iree_codegen.compilation_info<
+  lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1],
+                                                mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
+                                                subgroup_m_count = 4, subgroup_n_count = 2,
+                                                reduction = [0, 0, 32],
+                                                workgroup = [256, 128, 0]}>,
+  translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUVectorDistribute
+    workgroup_size = [128, 4, 1] subgroup_size = 64,
+    {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_shared_memory = true>,
+     llvm_func_attrs = {"amdgpu-waves-per-eu" = "4"}
+    }>> -> !transform.any_param
+  transform.yield %matmul, %config : !transform.any_op, !transform.any_param
+}
+
+//===----------------------------------------------------------------------===//
+// Convolution tuning
+//===----------------------------------------------------------------------===//
+
+//===----------------------------------------------------------------------===//
+// Batch matmul tuning
+//===----------------------------------------------------------------------===//
+
+//===----------------------------------------------------------------------===//
+// Broadcast rhs mmt tuning
+//===----------------------------------------------------------------------===//
+
+//===----------------------------------------------------------------------===//
+// Contraction tuning
+//===----------------------------------------------------------------------===//
+
+// TUNING_SPEC_END DO NOT REMOVE
+
+//===----------------------------------------------------------------------===//
+// Entry point
+//===----------------------------------------------------------------------===//
+
+  transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) -> !transform.any_op
```

**Comment:**
We can stage this PR such that we first commit a simple tuning spec to make sure everything works, and then separately work on adding configs we care about.

These configs should be tested to make sure they apply on the intended code, and we should quantify the improvements making sure we don't populate these specs with configs that give us only marginal improvements. This needs to be backed by data.

---


---


## [PR #19603](https://github.com/iree-org/iree/pull/19603): [Codegen][Tuner] skip linking based on the default entry point attribute

### Review Summary

**CHANGES_REQUESTED** (2025-01-06)


**APPROVED** (2025-01-06)

LGTM


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp:201`

```diff
@@ -197,14 +197,30 @@ struct MaterializeTuningSpecsPass final
       return;
     }
 
-    // If only the default tuning spec is available, use it directly and skip
-    // the linking stage.
-    if (!hasUserTuningSpec) {
-      if (failed(dumpFinalTuningSpecToDir(*defaultTuningSpec))) {
+    // Check if the user-provided tuning spec has the default entry point
+    // attribute.
```

**Comment:**
The code below is self-explanatory. I think the comment a few lines below should be enough.
```suggestion
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp:202`

```diff
@@ -197,14 +197,30 @@ struct MaterializeTuningSpecsPass final
       return;
     }
 
-    // If only the default tuning spec is available, use it directly and skip
-    // the linking stage.
-    if (!hasUserTuningSpec) {
-      if (failed(dumpFinalTuningSpecToDir(*defaultTuningSpec))) {
+    // Check if the user-provided tuning spec has the default entry point
+    // attribute.
+    bool userTuningSpecWithDefaultAttr =
```

**Comment:**
```suggestion
    bool isUserTuningSpecWithDefaultAttr =
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp:212`

```diff
@@ -197,14 +197,30 @@ struct MaterializeTuningSpecsPass final
       return;
     }
 
-    // If only the default tuning spec is available, use it directly and skip
-    // the linking stage.
-    if (!hasUserTuningSpec) {
-      if (failed(dumpFinalTuningSpecToDir(*defaultTuningSpec))) {
+    // Check if the user-provided tuning spec has the default entry point
+    // attribute.
+    bool userTuningSpecWithDefaultAttr =
+        hasUserTuningSpec &&
+        (*userTuningSpec)->hasAttr(kTuningSpecDefaultEntrypointAttrName);
+
+    // Determine if the linking pass should be skipped.
+    // Skip if there is a user-provided spec with the default attribute but no
+    // default tuning spec, or if there is no user-provided spec but a default
+    // tuning spec is available.
+    bool skipLinkPass = (hasUserTuningSpec && !hasDefaultTuningSpec &&
+                         userTuningSpecWithDefaultAttr) ||
+                        (!hasUserTuningSpec && hasDefaultTuningSpec);
```

**Comment:**
Instead of this logic, can we move up some code from down below to perform this check?
```c++
    SmallVector<ModuleOp, 2> allSpecs = {*userTuningSpec};
    if (hasDefaultTuningSpec) {
      allSpecs.push_back(*defaultTuningSpec);
    }
```
(and handle the missing user spec case too.)

Then we can check that there's a single element in this vector and that it has the default entrypoint attr.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/materialize_tuning_specs.mlir:33`

```diff
@@ -19,6 +24,16 @@
 // CHECK-SAME:     iree_codegen.tuning_spec_mlirbc = dense<{{.+}}> : vector<{{[0-9]+}}xi8>
 // CHECK-LABEL:    func.func @main_0
 
+
+// CHECK that the user-provided tuning spec is materized without linking when default tuing spec
+// is missing and the user-provided tuning spec is marked the default attribute.
+
+// SKIPLINK-LABEL: module  @user_spec
+// SKIPLINK-SAME:    iree_codegen.tuning_spec_with_default_entrypoint
+// SKIPLINK-SAME:    transform.with_named_sequence
```

**Comment:**
Can we also check that there are no nested modules?

---


---


## [PR #19525](https://github.com/iree-org/iree/pull/19525): [Codegen][Tuner] verifier for the default tuning spec

### Review Summary

**CHANGES_REQUESTED** (2024-12-19)

Can you explain what are the alternatives you considered and why you decided on this way of verifying the default specs? IIRC, we first discussed a dedicated verification pass.

Also, please update the documentation in https://iree.dev/reference/tuning/


**COMMENTED** (2025-01-02)


**COMMENTED** (2025-01-02)


**COMMENTED** (2025-01-03)


**APPROVED** (2025-01-03)

LGTM


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h:49`

```diff
@@ -45,6 +45,8 @@ namespace mlir::iree_compiler {
 // Constant names.
 //===----------------------------------------------------------------------===//
 constexpr StringLiteral kConfigAttrName = "lowering_config";
+constexpr StringLiteral kTuningDefaultSpecAttrName =
+    "iree_codegen.default_tuning_spec";
```

**Comment:**
I'm not sure this is the best name to use -- we should also allow user specs to specify that they have a single entry point. Maybe `iree_codegen.tuning_spec_with_default_entrypoint`?

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp:77`

```diff
@@ -63,6 +66,19 @@ IREECodegenDialect::verifyOperationAttribute(Operation *op,
   //         b. It must have exactly one argument type, and the argument must be
   //         of type `transform::AnyOpType`.
 
+  if (symbol == kTuningDefaultSpecAttrName) {
+    if (auto moduleOp = dyn_cast<ModuleOp>(op)) {
+      if (!llvm::any_of(moduleOp.getOps(), [](auto &op) {
+            return SymbolTable::getSymbolName(&op).getValue() ==
+                   kKernelConfigSpecName;
+          })) {
+        return moduleOp.emitError()
+               << "The default tuning specification must include an "
+                  "operation with the symbol name '__kernel_config'.";
```

**Comment:**
Don't hardcode the name here in case we want to rename it in the future.

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp:73`

```diff
@@ -63,6 +66,19 @@ IREECodegenDialect::verifyOperationAttribute(Operation *op,
   //         b. It must have exactly one argument type, and the argument must be
   //         of type `transform::AnyOpType`.
 
+  if (symbol == kTuningDefaultSpecAttrName) {
+    if (auto moduleOp = dyn_cast<ModuleOp>(op)) {
+      if (!llvm::any_of(moduleOp.getOps(), [](auto &op) {
+            return SymbolTable::getSymbolName(&op).getValue() ==
+                   kKernelConfigSpecName;
```

**Comment:**
Can you enumerate the named sequence ops and check only these instead? The check here doesn't guarantee that the symbol is a named sequence, it could be something else.

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp:58`

```diff
@@ -51,8 +51,11 @@ IREECodegenDialect::verifyOperationAttribute(Operation *op,
                                              NamedAttribute attribute) {
   StringRef symbol = attribute.getName().strref();
   Attribute attr = attribute.getValue();
-
   // This function verifies the validity of a specific operation attribute.
+  // - If the attribute's name matches `kTuningDefaultSpecAttrName` :
+  //   - For the `ModuleOp` operation ( representing the default spec):
+  //     - Ensure the module contains one operation with the symbol
+  //       name `__kernel_config`. If not, emit an error.
```

**Comment:**
```suggestion
  // - If the attribute's name matches `kTuningDefaultSpecAttrName`, make
  //   sure it contains a single named sequence op with name `__kernel_config`.
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:59`

```diff
@@ -51,3 +51,9 @@ module @foo_module attributes { transform.with_named_sequence } {
   transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly})
     attributes { iree_codegen.tuning_spec_entrypoint } {}
 }
+
+// -----
+
+// expected-error @+1{{The default tuning specification must include an operation with the symbol name '__kernel_config'}}
+module @iree_default_tuning_spec attributes { iree_codegen.default_tuning_spec } {
+}
```

**Comment:**
We should also have a test for when there's some other op with named `__kernel_config`, e.g., `func.func`.

---

**File:** `compiler/plugins/target/ROCM/test/default_tuning_specs_amdgpu.mlir:36`

```diff
@@ -33,11 +33,11 @@
 // Check that both the user tuning spec and the default spec get linked and
 // materialized. The user spec should have precedence over the default one.
 
-// BOTH-LABEL: module @iree_linked_tuning_spec attributes {transform.with_named_sequence}
+// BOTH-LABEL: module @iree_linked_tuning_spec attributes {iree_codegen.tuning_spec_with_default_entrypoint, transform.with_named_sequence}
```

**Comment:**
Can you use `BOTH-SAME` to make this match work for any ordering of these module-level attributes?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:86`

```diff
@@ -81,6 +81,10 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
       0, hasConsumedSequences ? kArgConsumedAttrName : kArgReadOnlyAttrName,
       builder.getUnitAttr());
   newSpec->setAttr(kTuningSpecEntrypointAttrName, builder.getUnitAttr());
+  // As the newSpec is a named sequence operation with the symbol name
+  // '__kernel_config', the module should add the unit attribute
+  // 'iree_codegen.tuning_spec_with_default_entrypoint' to indicate this change.
```

**Comment:**
It's not clear to me what `this change` refers to. Instead, I'd add a comment higher up that this will create a named sequence op that conforms to the requirements of tuning specs with default entrypoint (not just the name).

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp:72`

```diff
@@ -68,9 +68,8 @@ IREECodegenDialect::verifyOperationAttribute(Operation *op,
     if (auto moduleOp = dyn_cast<ModuleOp>(op)) {
       if (!llvm::any_of(moduleOp.getOps(), [](auto &op) {
             if (auto namedSeqOp = dyn_cast<transform::NamedSequenceOp>(&op)) {
-              return SymbolTable::getSymbolName(namedSeqOp)
-                  .getValue()
-                  .contains(kKernelConfigSpecName);
+              return SymbolTable::getSymbolName(namedSeqOp).getValue() ==
+                     kKernelConfigSpecName;
```

**Comment:**
Can we check the name directly (`.getName` or similar) instead of using the symbol table?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:88`

```diff
@@ -81,6 +83,10 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
       0, hasConsumedSequences ? kArgConsumedAttrName : kArgReadOnlyAttrName,
       builder.getUnitAttr());
   newSpec->setAttr(kTuningSpecEntrypointAttrName, builder.getUnitAttr());
+  // As the newSpec is a named sequence operation with the symbol name
+  // '__kernel_config', the module should add the unit attribute
+  // 'iree_codegen.tuning_spec_with_default_entrypoint' to indicate this change.
```

**Comment:**
I don't think we need this comment anymore
```suggestion
```

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp:69`

```diff
@@ -63,6 +64,22 @@ IREECodegenDialect::verifyOperationAttribute(Operation *op,
   //         b. It must have exactly one argument type, and the argument must be
   //         of type `transform::AnyOpType`.
 
+  if (symbol == kTuningSpecDefaultEntrypointAttrName) {
+    if (auto moduleOp = dyn_cast<ModuleOp>(op)) {
+      if (!llvm::any_of(moduleOp.getOps(), [](auto &op) {
```

**Comment:**
We don't need this to be generic. Also, let's not shadow the `op` variable from the parent scope.
```suggestion
      if (!llvm::any_of(moduleOp.getOps(), [](Operation *nestedOp) {
```

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp:69`

```diff
@@ -63,6 +64,22 @@ IREECodegenDialect::verifyOperationAttribute(Operation *op,
   //         b. It must have exactly one argument type, and the argument must be
   //         of type `transform::AnyOpType`.
 
+  if (symbol == kTuningSpecDefaultEntrypointAttrName) {
+    if (auto moduleOp = dyn_cast<ModuleOp>(op)) {
+      if (!llvm::any_of(moduleOp.getOps(), [](auto &op) {
```

**Comment:**
Isn't there a helper that gets the ops of the specified type? Something like `getOps<SomeOp>()`?

---

**File:** `docs/website/docs/reference/tuning.md:127`

```diff
@@ -123,6 +123,8 @@ that conform to the following format:
   `!transform.any_op`.
 * All entry points in the final tuning specs must either read
   (`transform.readonly`) or consume (`transform.consumed`) the argument.
+* The `iree_codegen.tuning_spec_with_default_entrypoint` attribute ensures that
+  the tuning spec includes a named sequence op marked with `__kernel_config`.
```

**Comment:**
```suggestion
  the tuning spec includes a named sequence op with name `__kernel_config`.
```

---


---


## [PR #19486](https://github.com/iree-org/iree/pull/19486): [Codegen][Tuner] attr verifier for tuning specs

### Review Summary

**CHANGES_REQUESTED** (2024-12-17)


**COMMENTED** (2024-12-18)


**COMMENTED** (2024-12-18)


**APPROVED** (2024-12-18)

Mostly LGTM now


**CHANGES_REQUESTED** (2024-12-18)


**COMMENTED** (2024-12-18)


**COMMENTED** (2024-12-18)


**APPROVED** (2024-12-18)

LGTM % one nit


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp:118`

```diff
@@ -107,6 +107,16 @@ getUserTuningSpec(ModuleOp module, IREE::Codegen::IREECodegenDialect &dialect) {
            << clCodegenTuningSpecPath;
   }
 
+  // Iterate through all operations in the module to verify attributes
+  for (Operation &op : (*maybeTransformLibrary).getBody()->getOperations()) {
+    for (NamedAttribute attr : op.getAttrs()) {
+      if (failed(dialect.verifyOperationAttribute(&op, attr))) {
+        return op.emitError() << "Attribute verification failed for operation "
+                                 "in the user tuning spec";
+      }
+    }
+  }
```

**Comment:**
We should call verify on the whole module. You can see this used here: https://github.com/llvm/llvm-project/blob/57c161a6479fb70a31553e2f9bc1efa46262aa92/mlir/lib/Dialect/Transform/Transforms/TransformInterpreterUtils.cpp#L118

---

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp:110`

```diff
@@ -107,6 +107,16 @@ getUserTuningSpec(ModuleOp module, IREE::Codegen::IREECodegenDialect &dialect) {
            << clCodegenTuningSpecPath;
   }
 
+  // Iterate through all operations in the module to verify attributes
```

**Comment:**
Use proper punctuation.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp:118`

```diff
@@ -107,6 +107,16 @@ getUserTuningSpec(ModuleOp module, IREE::Codegen::IREECodegenDialect &dialect) {
            << clCodegenTuningSpecPath;
   }
 
+  // Iterate through all operations in the module to verify attributes
+  for (Operation &op : (*maybeTransformLibrary).getBody()->getOperations()) {
+    for (NamedAttribute attr : op.getAttrs()) {
+      if (failed(dialect.verifyOperationAttribute(&op, attr))) {
+        return op.emitError() << "Attribute verification failed for operation "
+                                 "in the user tuning spec";
+      }
+    }
+  }
```

**Comment:**
Actually, I don't think this is needed along this code path because the user spec verification happens in `parseTransformModuleFromFile`. We should verify linked specs after linking though.l

---

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp:167`

```diff
@@ -138,8 +148,25 @@ getDefaultTuningSpec(ModuleOp module,
 
   // Load the library through the codegen dialect so that we cache the parsed
   // module.
-  return dialect.getOrParseTransformLibraryModule(defaultTuningSpecName,
-                                                  *defaultTuningSpecSource);
+  FailureOr<ModuleOp> defaultTransformLibrary =
+      dialect.getOrParseTransformLibraryModule(defaultTuningSpecName,
+                                               *defaultTuningSpecSource);
+  if (failed(defaultTransformLibrary)) {
+    return module->emitError()
+           << "Failed to parse default tuning spec transform library for '"
+           << arch << "'";
+  }
+  // Iterate through operations and validate their attributes
+  for (Operation &op : (*defaultTransformLibrary).getBody()->getOperations()) {
+    for (NamedAttribute attr : op.getAttrs()) {
+      if (failed(dialect.verifyOperationAttribute(&op, attr))) {
+        return op.emitError() << "Attribute verification failed for operation "
+                                 "in default tuning spec";
+      }
+    }
+  }
```

**Comment:**
I don't think we should verify default specs -- these are already verified by our tests when building the compiler. We could do that but under debug builds only.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:1`

```diff
@@ -0,0 +1,18 @@
+// RUN: iree-opt --no-implicit-module --verify-diagnostics -split-input-file --mlir-disable-threading %s
```

**Comment:**
```suggestion
// RUN: iree-opt  --verify-diagnostics  %s
```

This test does not rely on the other flags AFAICT

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:18`

```diff
@@ -0,0 +1,18 @@
+// RUN: iree-opt --no-implicit-module --verify-diagnostics -split-input-file --mlir-disable-threading %s
+
+module @td_module attributes { transform.with_named_sequence } {
+  module @foo_module attributes { transform.with_named_sequence } {
+    // expected-error @+1{{'iree_codegen.tuning_spec_entrypoint' attribute must be a UnitAttr}}
+    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+      attributes { iree_codegen.tuning_spec_entrypoint = "foo" } {
+      transform.yield %arg0 : !transform.any_op
+    }
+    transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+      attributes { iree_codegen.tuning_spec_entrypoint } {
+      transform.yield %arg0 : !transform.any_op
+    }
+    func.func @baz(%arg0: i32) -> () {
+      return
+    }
+  }
+}
```

**Comment:**
We should also check what happens there's some other attribute attached to `named_sequence` and documents whether that's allowed or not (by the virtue of having a testcase).

We should also add tests that check that the attribute is present but the `named_sequence` signature is wrong.

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp:76`

```diff
@@ -45,4 +46,39 @@ void IREECodegenDialect::initialize() {
       >();
 }
 
+LogicalResult
+IREECodegenDialect::verifyOperationAttribute(Operation *op,
+                                             NamedAttribute attribute) {
+  StringRef symbol = attribute.getName().strref();
+  Attribute attr = attribute.getValue();
+
+  // Early return if the symbol is not "iree_codegen.tuning_spec_entrypoint"
+  if (symbol != kTuningSpecEntrypointAttrName)
+    return success();
+
+  // Verify that the attribute is a UnitAttr
+  if (!llvm::isa<UnitAttr>(attr)) {
+    return op->emitError("'") << symbol << "' attribute must be a UnitAttr";
+  }
+
+  if (auto namedSeqOp = dyn_cast<transform::NamedSequenceOp>(op)) {
+    ArrayRef<Type> resTypes = namedSeqOp.getFunctionType().getResults();
+    if (resTypes.size() != 1 || !isa<transform::AnyOpType>(resTypes[0])) {
+      return namedSeqOp.emitWarning()
+             << "Tuning spec entry point expected to return any_op";
+    }
+
+    ArrayRef<Type> argTypes = namedSeqOp.getArgumentTypes();
+    if (argTypes.size() != 1 || !isa<transform::AnyOpType>(argTypes[0])) {
+      return namedSeqOp.emitWarning()
+             << "Tuning spec entry point expected to have a "
+                "single any_op argument";
+    }
```

**Comment:**
We should remove the validation from the other code where this was copied from -- no need to validate twice.

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp:55`

```diff
@@ -45,4 +46,39 @@ void IREECodegenDialect::initialize() {
       >();
 }
 
+LogicalResult
+IREECodegenDialect::verifyOperationAttribute(Operation *op,
+                                             NamedAttribute attribute) {
+  StringRef symbol = attribute.getName().strref();
+  Attribute attr = attribute.getValue();
+
+  // Early return if the symbol is not "iree_codegen.tuning_spec_entrypoint"
```

**Comment:**
I don't think this comment adds clarity, I'd drop it. Instead, you can summarize the validity criteria in one longer comment.

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp:59`

```diff
@@ -45,4 +46,39 @@ void IREECodegenDialect::initialize() {
       >();
 }
 
+LogicalResult
+IREECodegenDialect::verifyOperationAttribute(Operation *op,
+                                             NamedAttribute attribute) {
+  StringRef symbol = attribute.getName().strref();
+  Attribute attr = attribute.getValue();
+
+  // Early return if the symbol is not "iree_codegen.tuning_spec_entrypoint"
+  if (symbol != kTuningSpecEntrypointAttrName)
+    return success();
+
+  // Verify that the attribute is a UnitAttr
```

**Comment:**
Same here

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp:60`

```diff
@@ -45,4 +46,39 @@ void IREECodegenDialect::initialize() {
       >();
 }
 
+LogicalResult
+IREECodegenDialect::verifyOperationAttribute(Operation *op,
+                                             NamedAttribute attribute) {
+  StringRef symbol = attribute.getName().strref();
+  Attribute attr = attribute.getValue();
+
+  // Early return if the symbol is not "iree_codegen.tuning_spec_entrypoint"
+  if (symbol != kTuningSpecEntrypointAttrName)
+    return success();
+
+  // Verify that the attribute is a UnitAttr
+  if (!llvm::isa<UnitAttr>(attr)) {
```

**Comment:**
```suggestion
  if (!isa<UnitAttr>(attr)) {
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:150`

```diff
@@ -165,17 +145,13 @@ struct LinkTuningSpecsPass final
 FailureOr<NamedSequenceOp> linkTuningSpecs(ModuleOp module) {
   SmallVector<NamedSequenceOp> tuningSpecs;
 
+  if (failed(mlir::verify(module)))
+    return failure();
+
```

**Comment:**
We should verify the result of linking, not the input. It is assumed that the input would have been verifier by the parser or something else that created it.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp:147`

```diff
@@ -138,8 +139,17 @@ getDefaultTuningSpec(ModuleOp module,
 
   // Load the library through the codegen dialect so that we cache the parsed
   // module.
-  return dialect.getOrParseTransformLibraryModule(defaultTuningSpecName,
-                                                  *defaultTuningSpecSource);
+  FailureOr<ModuleOp> defaultTransformLibrary =
+      dialect.getOrParseTransformLibraryModule(defaultTuningSpecName,
+                                               *defaultTuningSpecSource);
+
+#ifndef NDEBUG
+  if (failed(mlir::verify(*defaultTransformLibrary)))
```

**Comment:**
This doesn't handle the case then the loaded module is a failure or nullptr

---

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp:257`

```diff
@@ -240,6 +250,13 @@ struct MaterializeTuningSpecsPass final
       return signalPassFailure();
     }
 
+    if (failed(mlir::verify(linkedTuningSpec.get()))) {
+      linkedTuningSpec.get().emitError(
+          "Attribute verification failed for operation in linked "
+          "tuning spec");
+      return signalPassFailure();
```

**Comment:**
This should be checked in the code that does linking

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:1`

```diff
@@ -0,0 +1,64 @@
+// RUN: iree-opt  --verify-diagnostics --split-input-file  %s
```

**Comment:**
```suggestion
// RUN: iree-opt --verify-diagnostics --split-input-file %s
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:39`

```diff
@@ -0,0 +1,64 @@
+// RUN: iree-opt  --verify-diagnostics --split-input-file  %s
+
+module @td_module attributes { transform.with_named_sequence } {
+  module @foo_module attributes { transform.with_named_sequence } {
+    // expected-error @+1{{'iree_codegen.tuning_spec_entrypoint' attribute must be a UnitAttr}}
+    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+      attributes { iree_codegen.tuning_spec_entrypoint = "foo" } {
+      transform.yield %arg0 : !transform.any_op
+    }
+    transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+      attributes { iree_codegen.something } {
+      transform.yield %arg0 : !transform.any_op
+    }
+    func.func @baz(%arg0: i32) -> () {
+      return
+    }
+  }
+}
+
+// -----
+
+module @td_module attributes { transform.with_named_sequence } {
+  module @foo_module attributes { transform.with_named_sequence } {
+    // expected-error @+1{{Tuning spec entry point expected to have a single any_op argument}}
+    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_op {transform.readonly}) -> !transform.any_op
+      attributes { iree_codegen.tuning_spec_entrypoint } {
+      transform.yield %arg0 : !transform.any_op
+    }
+    func.func @baz(%arg0: i32) -> () {
+      return
+    }
+  }
+}
+
+// -----
+
+module @td_module attributes { transform.with_named_sequence } {
+  module @foo_module attributes { transform.with_named_sequence } {
+    // expected-error @+1{{Tuning spec entry point expected to return any_op}}
```

**Comment:**
We should have a testcase for when there are no return values

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:6`

```diff
@@ -0,0 +1,64 @@
+// RUN: iree-opt  --verify-diagnostics --split-input-file  %s
+
+module @td_module attributes { transform.with_named_sequence } {
+  module @foo_module attributes { transform.with_named_sequence } {
+    // expected-error @+1{{'iree_codegen.tuning_spec_entrypoint' attribute must be a UnitAttr}}
+    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
```

**Comment:**
We don't need the nested module in this test -- a single level of nesting is sufficient

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:23`

```diff
@@ -0,0 +1,64 @@
+// RUN: iree-opt  --verify-diagnostics --split-input-file  %s
+
+module @td_module attributes { transform.with_named_sequence } {
+  module @foo_module attributes { transform.with_named_sequence } {
+    // expected-error @+1{{'iree_codegen.tuning_spec_entrypoint' attribute must be a UnitAttr}}
+    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+      attributes { iree_codegen.tuning_spec_entrypoint = "foo" } {
+      transform.yield %arg0 : !transform.any_op
+    }
+    transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+      attributes { iree_codegen.something } {
+      transform.yield %arg0 : !transform.any_op
+    }
+    func.func @baz(%arg0: i32) -> () {
+      return
+    }
+  }
+}
+
+// -----
+
+module @td_module attributes { transform.with_named_sequence } {
+  module @foo_module attributes { transform.with_named_sequence } {
```

**Comment:**
Also here, we don't need to nest

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:31`

```diff
@@ -0,0 +1,64 @@
+// RUN: iree-opt  --verify-diagnostics --split-input-file  %s
+
+module @td_module attributes { transform.with_named_sequence } {
+  module @foo_module attributes { transform.with_named_sequence } {
+    // expected-error @+1{{'iree_codegen.tuning_spec_entrypoint' attribute must be a UnitAttr}}
+    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+      attributes { iree_codegen.tuning_spec_entrypoint = "foo" } {
+      transform.yield %arg0 : !transform.any_op
+    }
+    transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+      attributes { iree_codegen.something } {
+      transform.yield %arg0 : !transform.any_op
+    }
+    func.func @baz(%arg0: i32) -> () {
+      return
+    }
+  }
+}
+
+// -----
+
+module @td_module attributes { transform.with_named_sequence } {
+  module @foo_module attributes { transform.with_named_sequence } {
+    // expected-error @+1{{Tuning spec entry point expected to have a single any_op argument}}
+    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_op {transform.readonly}) -> !transform.any_op
+      attributes { iree_codegen.tuning_spec_entrypoint } {
+      transform.yield %arg0 : !transform.any_op
+    }
+    func.func @baz(%arg0: i32) -> () {
+      return
+    }
```

**Comment:**
Other tests already check that function ops are allowed, I don't think we need this here

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:31`

```diff
@@ -0,0 +1,64 @@
+// RUN: iree-opt  --verify-diagnostics --split-input-file  %s
+
+module @td_module attributes { transform.with_named_sequence } {
+  module @foo_module attributes { transform.with_named_sequence } {
+    // expected-error @+1{{'iree_codegen.tuning_spec_entrypoint' attribute must be a UnitAttr}}
+    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+      attributes { iree_codegen.tuning_spec_entrypoint = "foo" } {
+      transform.yield %arg0 : !transform.any_op
+    }
+    transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+      attributes { iree_codegen.something } {
+      transform.yield %arg0 : !transform.any_op
+    }
+    func.func @baz(%arg0: i32) -> () {
+      return
+    }
+  }
+}
+
+// -----
+
+module @td_module attributes { transform.with_named_sequence } {
+  module @foo_module attributes { transform.with_named_sequence } {
+    // expected-error @+1{{Tuning spec entry point expected to have a single any_op argument}}
+    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_op {transform.readonly}) -> !transform.any_op
+      attributes { iree_codegen.tuning_spec_entrypoint } {
+      transform.yield %arg0 : !transform.any_op
+    }
+    func.func @baz(%arg0: i32) -> () {
+      return
+    }
```

**Comment:**
Same in the other test cases below

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:53`

```diff
@@ -0,0 +1,64 @@
+// RUN: iree-opt  --verify-diagnostics --split-input-file  %s
+
+module @td_module attributes { transform.with_named_sequence } {
+  module @foo_module attributes { transform.with_named_sequence } {
+    // expected-error @+1{{'iree_codegen.tuning_spec_entrypoint' attribute must be a UnitAttr}}
+    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+      attributes { iree_codegen.tuning_spec_entrypoint = "foo" } {
+      transform.yield %arg0 : !transform.any_op
+    }
+    transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+      attributes { iree_codegen.something } {
+      transform.yield %arg0 : !transform.any_op
+    }
+    func.func @baz(%arg0: i32) -> () {
+      return
+    }
+  }
+}
+
+// -----
+
+module @td_module attributes { transform.with_named_sequence } {
+  module @foo_module attributes { transform.with_named_sequence } {
+    // expected-error @+1{{Tuning spec entry point expected to have a single any_op argument}}
+    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_op {transform.readonly}) -> !transform.any_op
+      attributes { iree_codegen.tuning_spec_entrypoint } {
+      transform.yield %arg0 : !transform.any_op
+    }
+    func.func @baz(%arg0: i32) -> () {
+      return
+    }
+  }
+}
+
+// -----
+
+module @td_module attributes { transform.with_named_sequence } {
+  module @foo_module attributes { transform.with_named_sequence } {
+    // expected-error @+1{{Tuning spec entry point expected to return any_op}}
+    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> i32
+      attributes { iree_codegen.tuning_spec_entrypoint } {
+      %0 = arith.constant 0 : i32
+      transform.yield %0 : i32
+    }
+    func.func @baz(%arg0: i32) -> () {
+      return
+    }
+  }
+}
+
+// -----
+
+module @td_module attributes { transform.with_named_sequence } {
```

**Comment:**
I'd move this up just after the first testcase that checked for the wrong number of arguments

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp:88`

```diff
@@ -45,4 +46,49 @@ void IREECodegenDialect::initialize() {
       >();
 }
 
+LogicalResult
+IREECodegenDialect::verifyOperationAttribute(Operation *op,
+                                             NamedAttribute attribute) {
+  StringRef symbol = attribute.getName().strref();
+  Attribute attr = attribute.getValue();
+
+  // This function verifies the validity of a specific operation attribute.
+  // - If the attribute's name matches `kTuningSpecEntrypointAttrName`
+  // ("iree_codegen.tuning_spec_entrypoint"):
+  //   1. The attribute value must be a UnitAttr.
+  //   2. If the operation is a transform::NamedSequenceOp:
+  //      - The operation's function signature must satisfy the following:
+  //         a. It must have exactly one result type, and the result must be of
+  //         type `transform::AnyOpType`.
+  //         b. It must have exactly one argument type, and the argument must be
+  //         of type `transform::AnyOpType`.
+
+  if (symbol != kTuningSpecEntrypointAttrName)
+    return success();
+
+  // Verify that the attribute is a UnitAttr.
+  if (!isa<UnitAttr>(attr)) {
+    return op->emitError("'") << symbol << "' attribute must be a UnitAttr";
+  }
+
+  if (auto namedSeqOp = dyn_cast<transform::NamedSequenceOp>(op)) {
+    ArrayRef<Type> resTypes = namedSeqOp.getFunctionType().getResults();
+    if (resTypes.size() != 1 || !isa<transform::AnyOpType>(resTypes[0])) {
+      return namedSeqOp.emitError()
+             << "Tuning spec entry point expected to return any_op";
+    }
+
+    ArrayRef<Type> argTypes = namedSeqOp.getArgumentTypes();
+    if (argTypes.size() != 1 || !isa<transform::AnyOpType>(argTypes[0])) {
+      return namedSeqOp.emitError()
+             << "Tuning spec entry point expected to have a "
+                "single any_op argument";
+    }
+
+    return success();
```

**Comment:**
```suggestion
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:130`

```diff
@@ -124,6 +124,12 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   }
 
   builder.create<transform::YieldOp>(loc, operand);
+
+  if (failed(mlir::verify(module))) {
+    module.emitError("verification failed for operation in linked "
+                     "tuning spec");
```

**Comment:**
We should return failure so that nothing uses this module.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp:130`

```diff
@@ -144,6 +124,12 @@ emitLinkedTuningSpec(ModuleOp module, ArrayRef<NamedSequenceOp> specsToLink) {
   }
 
   builder.create<transform::YieldOp>(loc, operand);
+
+  if (failed(mlir::verify(module))) {
+    module.emitError("verification failed for operation in linked "
+                     "tuning spec");
```

**Comment:**
```suggestion
    module.emitError("Linked tuning spec failed to verify");
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp:154`

```diff
@@ -138,8 +139,22 @@ getDefaultTuningSpec(ModuleOp module,
 
   // Load the library through the codegen dialect so that we cache the parsed
   // module.
-  return dialect.getOrParseTransformLibraryModule(defaultTuningSpecName,
-                                                  *defaultTuningSpecSource);
+  FailureOr<ModuleOp> defaultTransformLibrary =
+      dialect.getOrParseTransformLibraryModule(defaultTuningSpecName,
+                                               *defaultTuningSpecSource);
+
+  if (failed(defaultTransformLibrary)) {
+    return module->emitError()
+           << "Failed to load  default tuning spec" << defaultTuningSpecName;
+  }
+
+#ifndef NDEBUG
+  if (failed(mlir::verify(*defaultTransformLibrary)))
+    return (*defaultTransformLibrary).emitError()
+           << "Verification failed for default tuning spec";
```

**Comment:**
```suggestion
           << "Default tuning spec " << defaultTuningSpecName << " failed to verify";
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp:148`

```diff
@@ -138,8 +139,22 @@ getDefaultTuningSpec(ModuleOp module,
 
   // Load the library through the codegen dialect so that we cache the parsed
   // module.
-  return dialect.getOrParseTransformLibraryModule(defaultTuningSpecName,
-                                                  *defaultTuningSpecSource);
+  FailureOr<ModuleOp> defaultTransformLibrary =
+      dialect.getOrParseTransformLibraryModule(defaultTuningSpecName,
+                                               *defaultTuningSpecSource);
+
+  if (failed(defaultTransformLibrary)) {
+    return module->emitError()
+           << "Failed to load  default tuning spec" << defaultTuningSpecName;
```

**Comment:**
We shouldn't emit this error here. The reason is that `getOrParseTransformLibraryModule` already does the reporting and it knows do it only when the parsing fails for the first time.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp:152`

```diff
@@ -138,8 +139,22 @@ getDefaultTuningSpec(ModuleOp module,
 
   // Load the library through the codegen dialect so that we cache the parsed
   // module.
-  return dialect.getOrParseTransformLibraryModule(defaultTuningSpecName,
-                                                  *defaultTuningSpecSource);
+  FailureOr<ModuleOp> defaultTransformLibrary =
+      dialect.getOrParseTransformLibraryModule(defaultTuningSpecName,
+                                               *defaultTuningSpecSource);
+
+  if (failed(defaultTransformLibrary)) {
+    return module->emitError()
+           << "Failed to load  default tuning spec" << defaultTuningSpecName;
+  }
+
+#ifndef NDEBUG
+  if (failed(mlir::verify(*defaultTransformLibrary)))
```

**Comment:**
Instead of the check above, we can do this:
```suggestion
  if (succeded(defaultTransformLibrary) && failed(mlir::verify(*defaultTransformLibrary)))
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:35`

```diff
@@ -0,0 +1,57 @@
+// RUN: iree-opt --verify-diagnostics --split-input-file %s
+
+module @foo_module attributes { transform.with_named_sequence } {
+  // expected-error @+1{{'iree_codegen.tuning_spec_entrypoint' attribute must be a UnitAttr}}
+  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+    attributes { iree_codegen.tuning_spec_entrypoint = "foo" } {
+    transform.yield %arg0 : !transform.any_op
+  }
+  transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+    attributes { iree_codegen.something } {
+    transform.yield %arg0 : !transform.any_op
+  }
+  func.func @baz(%arg0: i32) -> () {
+    return
+  }
+}
+
+// -----
+
+module @foo_module attributes { transform.with_named_sequence } {
+  // expected-error @+1{{Tuning spec entry point expected to have a single any_op argument}}
+  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_op {transform.readonly}) -> !transform.any_op
+    attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @foo_module attributes { transform.with_named_sequence } {
+  // expected-error @+1{{Tuning spec entry point expected to have a single any_op argument}}
+  transform.named_sequence @foo(%arg0: i32) -> !transform.any_op
+    attributes { iree_codegen.tuning_spec_entrypoint } {
+
+  }
```

**Comment:**
```suggestion
    attributes { iree_codegen.tuning_spec_entrypoint } {}
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:56`

```diff
@@ -0,0 +1,57 @@
+// RUN: iree-opt --verify-diagnostics --split-input-file %s
+
+module @foo_module attributes { transform.with_named_sequence } {
+  // expected-error @+1{{'iree_codegen.tuning_spec_entrypoint' attribute must be a UnitAttr}}
+  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+    attributes { iree_codegen.tuning_spec_entrypoint = "foo" } {
+    transform.yield %arg0 : !transform.any_op
+  }
+  transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+    attributes { iree_codegen.something } {
+    transform.yield %arg0 : !transform.any_op
+  }
+  func.func @baz(%arg0: i32) -> () {
+    return
+  }
+}
+
+// -----
+
+module @foo_module attributes { transform.with_named_sequence } {
+  // expected-error @+1{{Tuning spec entry point expected to have a single any_op argument}}
+  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_op {transform.readonly}) -> !transform.any_op
+    attributes { iree_codegen.tuning_spec_entrypoint } {
+    transform.yield %arg0 : !transform.any_op
+  }
+}
+
+// -----
+
+module @foo_module attributes { transform.with_named_sequence } {
+  // expected-error @+1{{Tuning spec entry point expected to have a single any_op argument}}
+  transform.named_sequence @foo(%arg0: i32) -> !transform.any_op
+    attributes { iree_codegen.tuning_spec_entrypoint } {
+
+  }
+}
+
+// -----
+
+module @foo_module attributes { transform.with_named_sequence } {
+  // expected-error @+1{{Tuning spec entry point expected to return any_op}}
+  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> i32
+    attributes { iree_codegen.tuning_spec_entrypoint } {
+    %0 = arith.constant 0 : i32
+    transform.yield %0 : i32
+  }
+}
+
+// -----
+
+module @foo_module attributes { transform.with_named_sequence } {
+  // expected-error @+1{{Tuning spec entry point expected to return any_op}}
+  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly})
+    attributes { iree_codegen.tuning_spec_entrypoint } {
+
+  }
```

**Comment:**
```suggestion
    attributes { iree_codegen.tuning_spec_entrypoint } {}
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:12`

```diff
@@ -0,0 +1,57 @@
+// RUN: iree-opt --verify-diagnostics --split-input-file %s
+
+module @foo_module attributes { transform.with_named_sequence } {
+  // expected-error @+1{{'iree_codegen.tuning_spec_entrypoint' attribute must be a UnitAttr}}
+  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+    attributes { iree_codegen.tuning_spec_entrypoint = "foo" } {
+    transform.yield %arg0 : !transform.any_op
+  }
+  transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+    attributes { iree_codegen.something } {
+    transform.yield %arg0 : !transform.any_op
+  }
```

**Comment:**
This won't be verified anyway because of the previous error. We should move it before the erroneous spec.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir:15`

```diff
@@ -0,0 +1,57 @@
+// RUN: iree-opt --verify-diagnostics --split-input-file %s
+
+module @foo_module attributes { transform.with_named_sequence } {
+  // expected-error @+1{{'iree_codegen.tuning_spec_entrypoint' attribute must be a UnitAttr}}
+  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+    attributes { iree_codegen.tuning_spec_entrypoint = "foo" } {
+    transform.yield %arg0 : !transform.any_op
+  }
+  transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
+    attributes { iree_codegen.something } {
+    transform.yield %arg0 : !transform.any_op
+  }
+  func.func @baz(%arg0: i32) -> () {
+    return
+  }
```

**Comment:**
Also here

---


---


## [PR #19376](https://github.com/iree-org/iree/pull/19376): [tuner]: add property functions to lowering config python binding

### Review Summary

**COMMENTED** (2024-12-05)


**CHANGES_REQUESTED** (2024-12-05)


**CHANGES_REQUESTED** (2024-12-05)


**COMMENTED** (2024-12-06)


**COMMENTED** (2024-12-06)


**COMMENTED** (2024-12-06)


**APPROVED** (2024-12-07)

LGTM % one nit


**COMMENTED** (2024-12-07)


### Code Comments

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h:102`

```diff
@@ -93,6 +93,23 @@ MLIR_CAPI_EXPORTED MlirAttribute ireeGPULoweringConfigAttrGet(
 MLIR_CAPI_EXPORTED MlirAttribute
 ireeGPULoweringConfigAttrGetAttributes(MlirAttribute attr);
 
+MLIR_CAPI_EXPORTED void
+ireeGPULoweringConfigAttrGetReductionTileSizes(MlirAttribute attr, size_t *len,
+                                               int64_t *reductionTileSizes);
+
+MLIR_CAPI_EXPORTED void
+ireeGPULoweringConfigAttrGetWorkgroupTileSizes(MlirAttribute attr, size_t *len,
+                                               int64_t *workgroupTileSizes);
```

**Comment:**
I'd put these in a single function that returns all tile sizes in a struct. You can see an example above (`ireeGPUMMAAttrGetInfo`). The reason is that this is expands the API surface area by quite a lot, and adding more levels of tiling would exacerbate this further. With a struct, we can keep extending it with more fields.

---

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h:108`

```diff
@@ -93,6 +93,23 @@ MLIR_CAPI_EXPORTED MlirAttribute ireeGPULoweringConfigAttrGet(
 MLIR_CAPI_EXPORTED MlirAttribute
 ireeGPULoweringConfigAttrGetAttributes(MlirAttribute attr);
 
+MLIR_CAPI_EXPORTED void
+ireeGPULoweringConfigAttrGetReductionTileSizes(MlirAttribute attr, size_t *len,
+                                               int64_t *reductionTileSizes);
+
+MLIR_CAPI_EXPORTED void
+ireeGPULoweringConfigAttrGetWorkgroupTileSizes(MlirAttribute attr, size_t *len,
+                                               int64_t *workgroupTileSizes);
+
+MLIR_CAPI_EXPORTED MlirAttribute
+ireeGPULoweringConfigAttrGetSubgroupMCount(MlirAttribute attr);
+
+MLIR_CAPI_EXPORTED MlirAttribute
+ireeGPULoweringConfigAttrGetSubgroupNCount(MlirAttribute attr);
```

**Comment:**
This can be a single function that returns two integers.

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:63`

```diff
@@ -48,6 +48,36 @@ ireeCodegenQueryMMAIntrinsicsBinding(MlirOperation op) {
   return mmaList;
 }
 
+static std::optional<std::vector<int64_t>>
+ireeGPULoweringConfigAttrGetReductionTileSizesBinding(
+    MlirAttribute lowering_config) {
+  size_t tileSizes = 0;
+  ireeGPULoweringConfigAttrGetReductionTileSizes(lowering_config, &tileSizes,
+                                                 nullptr);
+  if (tileSizes == -1) {
+    return std::nullopt;
+  }
+  std::vector<int64_t> reductionTileSizes(tileSizes);
+  ireeGPULoweringConfigAttrGetReductionTileSizes(lowering_config, &tileSizes,
+                                                 reductionTileSizes.data());
+  return reductionTileSizes;
```

**Comment:**
We don't need the buffer to be caller-allocated -- we can return a pointer to the attribute storage.

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:77`

```diff
@@ -48,6 +48,36 @@ ireeCodegenQueryMMAIntrinsicsBinding(MlirOperation op) {
   return mmaList;
 }
 
+static std::optional<std::vector<int64_t>>
+ireeGPULoweringConfigAttrGetReductionTileSizesBinding(
+    MlirAttribute lowering_config) {
+  size_t tileSizes = 0;
+  ireeGPULoweringConfigAttrGetReductionTileSizes(lowering_config, &tileSizes,
+                                                 nullptr);
+  if (tileSizes == -1) {
+    return std::nullopt;
+  }
+  std::vector<int64_t> reductionTileSizes(tileSizes);
+  ireeGPULoweringConfigAttrGetReductionTileSizes(lowering_config, &tileSizes,
+                                                 reductionTileSizes.data());
+  return reductionTileSizes;
+}
+
+static std::optional<std::vector<int64_t>>
+ireeGPULoweringConfigAttrGetWorkgroupTileSizesBinding(
+    MlirAttribute lowering_config) {
+  size_t tileSizes = 0;
+  ireeGPULoweringConfigAttrGetWorkgroupTileSizes(lowering_config, &tileSizes,
+                                                 nullptr);
+  if (tileSizes == -1) {
+    return std::nullopt;
+  }
+  std::vector<int64_t> workgroupTileSizes(tileSizes);
+  ireeGPULoweringConfigAttrGetWorkgroupTileSizes(lowering_config, &tileSizes,
+                                                 workgroupTileSizes.data());
```

**Comment:**
Also here

---

**File:** `compiler/bindings/python/test/ir/dialects_test.py:231`

```diff
@@ -210,11 +210,40 @@ def mma_intrinsic_attr():
 
 @run
 def lowering_config_attr():
-    attributes = ir.DictAttr.get({"reduction": ir.ArrayAttr.get([])})
+    attributes = ir.DictAttr.get(
+        {
+            "reduction": ir.ArrayAttr.get([]),
+        }
+    )
     lowering_config = iree_gpu.LoweringConfigAttr.get(attributes)
     assert lowering_config is not None
 
     assert lowering_config.attributes == attributes
+    assert lowering_config.workgroup_tile_sizes == None
+    assert lowering_config.reduction_tile_sizes == []
+    assert lowering_config.subgroup_m_count == None
+    assert lowering_config.mma_kind == None
+
+    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
+    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
+    attributes = ir.DictAttr.get(
+        {
+            "reduction": ir.ArrayAttr.get([ir.IntegerAttr.get(ir.IndexType.get(), 1)]),
```

**Comment:**
You can add a helper to `tuner_ctx` to help with this, e.g.: `tuner_ctx.type.getIndexArray([1])`

---

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h:109`

```diff
@@ -93,6 +93,32 @@ MLIR_CAPI_EXPORTED MlirAttribute ireeGPULoweringConfigAttrGet(
 MLIR_CAPI_EXPORTED MlirAttribute
 ireeGPULoweringConfigAttrGetAttributes(MlirAttribute attr);
 
+struct ireeGPUTileSizes {
+  const int64_t *workgroupTileSizes;
+  size_t numWorkgroupTileSizes;
+  const int64_t *reductionTileSizes;
+  size_t numReductionTileSizes;
+};
+
+MLIR_CAPI_EXPORTED ireeGPUTileSizes
+ireeGPULoweringConfigAttrGetTileSizes(MlirAttribute attr);
+
+// MLIR_CAPI_EXPORTED void
+// ireeGPULoweringConfigAttrGetWorkgroupTileSizes(MlirAttribute attr, size_t
+// *len,
+//                                                int64_t *workgroupTileSizes);
```

**Comment:**
Some leftover comment

---

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h:106`

```diff
@@ -93,6 +93,32 @@ MLIR_CAPI_EXPORTED MlirAttribute ireeGPULoweringConfigAttrGet(
 MLIR_CAPI_EXPORTED MlirAttribute
 ireeGPULoweringConfigAttrGetAttributes(MlirAttribute attr);
 
+struct ireeGPUTileSizes {
+  const int64_t *workgroupTileSizes;
+  size_t numWorkgroupTileSizes;
+  const int64_t *reductionTileSizes;
+  size_t numReductionTileSizes;
+};
+
+MLIR_CAPI_EXPORTED ireeGPUTileSizes
+ireeGPULoweringConfigAttrGetTileSizes(MlirAttribute attr);
+
+// MLIR_CAPI_EXPORTED void
+// ireeGPULoweringConfigAttrGetWorkgroupTileSizes(MlirAttribute attr, size_t
+// *len,
+//                                                int64_t *workgroupTileSizes);
+
+struct ireeGPUSubgroupCountInfo {
+  MlirAttribute subgroupMCountAttr;
+  MlirAttribute subgroupNCountAttr;
```

**Comment:**
I'd return these as integers int64_t

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:224`

```diff
@@ -213,3 +213,61 @@ MlirAttribute ireeGPULoweringConfigAttrGetAttributes(MlirAttribute attr) {
                   unwrap(attr))
                   .getAttributes());
 }
+
+ireeGPUTileSizes ireeGPULoweringConfigAttrGetTileSizes(MlirAttribute attr) {
+  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
+  ireeGPUTileSizes tilesizes = {};
+  auto loweringConfigAttr =
+      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
+          unwrap(attr));
+
+  auto workgroups = loweringConfigAttr.getWorkgroupTileSizes();
```

**Comment:**
Don't use `auto` when the type is not obvious based on the RHS only

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:228`

```diff
@@ -213,3 +213,61 @@ MlirAttribute ireeGPULoweringConfigAttrGetAttributes(MlirAttribute attr) {
                   unwrap(attr))
                   .getAttributes());
 }
+
+ireeGPUTileSizes ireeGPULoweringConfigAttrGetTileSizes(MlirAttribute attr) {
+  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
+  ireeGPUTileSizes tilesizes = {};
+  auto loweringConfigAttr =
+      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
+          unwrap(attr));
+
+  auto workgroups = loweringConfigAttr.getWorkgroupTileSizes();
+  tilesizes.workgroupTileSizes = workgroups.data();
+  tilesizes.numWorkgroupTileSizes = workgroups.size();
+
+  auto reduction = loweringConfigAttr.getStaticTilingLevelSizes(
```

**Comment:**
Also here

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:252`

```diff
@@ -213,3 +213,61 @@ MlirAttribute ireeGPULoweringConfigAttrGetAttributes(MlirAttribute attr) {
                   unwrap(attr))
                   .getAttributes());
 }
+
+ireeGPUTileSizes ireeGPULoweringConfigAttrGetTileSizes(MlirAttribute attr) {
+  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
+  ireeGPUTileSizes tilesizes = {};
+  auto loweringConfigAttr =
+      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
+          unwrap(attr));
+
+  auto workgroups = loweringConfigAttr.getWorkgroupTileSizes();
+  tilesizes.workgroupTileSizes = workgroups.data();
+  tilesizes.numWorkgroupTileSizes = workgroups.size();
+
+  auto reduction = loweringConfigAttr.getStaticTilingLevelSizes(
+      static_cast<int64_t>(
+          mlir::iree_compiler::IREE::GPU::TilingLevel::Reduction),
+      nullptr);
+  tilesizes.reductionTileSizes = reduction.data();
+  tilesizes.numReductionTileSizes = reduction.size();
+
+  return tilesizes;
+}
+
+ireeGPUSubgroupCountInfo
+ireeGPULoweringConfigAttrGetSubgroupCount(MlirAttribute attr) {
+  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
+  mlir::DictionaryAttr dict =
+      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
+          unwrap(attr))
+          .getAttributes();
+
+  constexpr mlir::StringLiteral kSubgroupMCountName = "subgroup_m_count";
+  constexpr mlir::StringLiteral kSubgroupNCountName = "subgroup_n_count";
+
+  mlir::IntegerAttr subgroup_m_count_attr =
+      dict.getAs<mlir::IntegerAttr>(kSubgroupMCountName);
+  mlir::IntegerAttr subgroup_n_count_attr =
+      dict.getAs<mlir::IntegerAttr>(kSubgroupNCountName);
```

**Comment:**
Don't we have helpers for this in the dialect headers or the attr interface?

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:252`

```diff
@@ -213,3 +213,61 @@ MlirAttribute ireeGPULoweringConfigAttrGetAttributes(MlirAttribute attr) {
                   unwrap(attr))
                   .getAttributes());
 }
+
+ireeGPUTileSizes ireeGPULoweringConfigAttrGetTileSizes(MlirAttribute attr) {
+  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
+  ireeGPUTileSizes tilesizes = {};
+  auto loweringConfigAttr =
+      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
+          unwrap(attr));
+
+  auto workgroups = loweringConfigAttr.getWorkgroupTileSizes();
+  tilesizes.workgroupTileSizes = workgroups.data();
+  tilesizes.numWorkgroupTileSizes = workgroups.size();
+
+  auto reduction = loweringConfigAttr.getStaticTilingLevelSizes(
+      static_cast<int64_t>(
+          mlir::iree_compiler::IREE::GPU::TilingLevel::Reduction),
+      nullptr);
+  tilesizes.reductionTileSizes = reduction.data();
+  tilesizes.numReductionTileSizes = reduction.size();
+
+  return tilesizes;
+}
+
+ireeGPUSubgroupCountInfo
+ireeGPULoweringConfigAttrGetSubgroupCount(MlirAttribute attr) {
+  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
+  mlir::DictionaryAttr dict =
+      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
+          unwrap(attr))
+          .getAttributes();
+
+  constexpr mlir::StringLiteral kSubgroupMCountName = "subgroup_m_count";
+  constexpr mlir::StringLiteral kSubgroupNCountName = "subgroup_n_count";
+
+  mlir::IntegerAttr subgroup_m_count_attr =
+      dict.getAs<mlir::IntegerAttr>(kSubgroupMCountName);
+  mlir::IntegerAttr subgroup_n_count_attr =
+      dict.getAs<mlir::IntegerAttr>(kSubgroupNCountName);
```

**Comment:**
Here: https://github.com/iree-org/iree/blob/a1664e30a6a53850f689f32086a8b0c45bed327b/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h#L22-L23

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:269`

```diff
@@ -213,3 +213,61 @@ MlirAttribute ireeGPULoweringConfigAttrGetAttributes(MlirAttribute attr) {
                   unwrap(attr))
                   .getAttributes());
 }
+
+ireeGPUTileSizes ireeGPULoweringConfigAttrGetTileSizes(MlirAttribute attr) {
+  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
+  ireeGPUTileSizes tilesizes = {};
+  auto loweringConfigAttr =
+      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
+          unwrap(attr));
+
+  auto workgroups = loweringConfigAttr.getWorkgroupTileSizes();
+  tilesizes.workgroupTileSizes = workgroups.data();
+  tilesizes.numWorkgroupTileSizes = workgroups.size();
+
+  auto reduction = loweringConfigAttr.getStaticTilingLevelSizes(
+      static_cast<int64_t>(
+          mlir::iree_compiler::IREE::GPU::TilingLevel::Reduction),
+      nullptr);
+  tilesizes.reductionTileSizes = reduction.data();
+  tilesizes.numReductionTileSizes = reduction.size();
+
+  return tilesizes;
+}
+
+ireeGPUSubgroupCountInfo
+ireeGPULoweringConfigAttrGetSubgroupCount(MlirAttribute attr) {
+  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
+  mlir::DictionaryAttr dict =
+      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
+          unwrap(attr))
+          .getAttributes();
+
+  constexpr mlir::StringLiteral kSubgroupMCountName = "subgroup_m_count";
+  constexpr mlir::StringLiteral kSubgroupNCountName = "subgroup_n_count";
+
+  mlir::IntegerAttr subgroup_m_count_attr =
+      dict.getAs<mlir::IntegerAttr>(kSubgroupMCountName);
+  mlir::IntegerAttr subgroup_n_count_attr =
+      dict.getAs<mlir::IntegerAttr>(kSubgroupNCountName);
+
+  ireeGPUSubgroupCountInfo info = {};
+  info.subgroupMCountAttr = wrap(subgroup_m_count_attr);
+  info.subgroupNCountAttr = wrap(subgroup_n_count_attr);
+  return info;
+}
+
+MlirAttribute ireeGPULoweringConfigAttrGetMmaKind(MlirAttribute attr) {
+  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
+  mlir::DictionaryAttr dict =
+      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
+          unwrap(attr))
+          .getAttributes();
+
+  constexpr mlir::StringLiteral kMmaKindName = "mma_kind";
+  mlir::iree_compiler::IREE::GPU::MmaInterfaceAttr mma_attr =
+      dict.getAs<mlir::iree_compiler::IREE::GPU::MmaInterfaceAttr>(
```

**Comment:**
We also have a helper for this

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:373`

```diff
@@ -352,7 +352,49 @@ PYBIND11_MODULE(_ireeCompilerDialects, m) {
           "cls"_a, "value"_a, "ctx"_a = py::none(),
           "Gets an #iree_gpu.lowering_config from parameters.")
       .def_property_readonly("attributes",
-                             ireeGPULoweringConfigAttrGetAttributes);
+                             ireeGPULoweringConfigAttrGetAttributes)
+      .def_property_readonly("workgroup_tile_sizes",
+                             [](MlirAttribute self) -> std::vector<int64_t> {
+                               auto tilesizes =
+                                   ireeGPULoweringConfigAttrGetTileSizes(self);
+                               return {tilesizes.workgroupTileSizes,
+                                       tilesizes.workgroupTileSizes +
+                                           tilesizes.numWorkgroupTileSizes};
+                             })
+      .def_property_readonly("reduction_tile_sizes",
+                             [](MlirAttribute self) -> std::vector<int64_t> {
+                               auto tilesizes =
+                                   ireeGPULoweringConfigAttrGetTileSizes(self);
+                               return {tilesizes.reductionTileSizes,
+                                       tilesizes.reductionTileSizes +
+                                           tilesizes.numReductionTileSizes};
+                             })
+      .def_property_readonly(
+          "subgroup_count",
```

**Comment:**
To me it's not clear how these map to mnk dimensions. Maybe call it `subgroup_count_mn`?

---

**File:** `compiler/bindings/python/test/ir/dialects_test.py:13`

```diff
@@ -10,6 +10,16 @@
 from iree.compiler.dialects import flow, hal, stream, vm, util, iree_codegen, iree_gpu
 
 
+def get_array_attr(vals: list[int]) -> ir.ArrayAttr:
```

**Comment:**
This function name doesn't specify the element type

---

**File:** `compiler/bindings/python/test/ir/dialects_test.py:19`

```diff
@@ -10,6 +10,16 @@
 from iree.compiler.dialects import flow, hal, stream, vm, util, iree_codegen, iree_gpu
 
 
+def get_array_attr(vals: list[int]) -> ir.ArrayAttr:
+    return ir.ArrayAttr.get(
+        [ir.IntegerAttr.get(ir.IndexType.get(), val) for val in vals]
+    )
+
+
+def get_integer_attr(val: int) -> ir.IntegerAttr:
```

**Comment:**
This returns the index type

---

**File:** `compiler/bindings/python/test/ir/dialects_test.py:15`

```diff
@@ -10,6 +10,16 @@
 from iree.compiler.dialects import flow, hal, stream, vm, util, iree_codegen, iree_gpu
 
 
+def get_array_attr(vals: list[int]) -> ir.ArrayAttr:
+    return ir.ArrayAttr.get(
+        [ir.IntegerAttr.get(ir.IndexType.get(), val) for val in vals]
```

**Comment:**
You can move the `get_index_attr` helper above and use here

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:388`

```diff
@@ -352,7 +352,49 @@ PYBIND11_MODULE(_ireeCompilerDialects, m) {
           "cls"_a, "value"_a, "ctx"_a = py::none(),
           "Gets an #iree_gpu.lowering_config from parameters.")
       .def_property_readonly("attributes",
-                             ireeGPULoweringConfigAttrGetAttributes);
+                             ireeGPULoweringConfigAttrGetAttributes)
+      .def_property_readonly("workgroup_tile_sizes",
+                             [](MlirAttribute self) -> std::vector<int64_t> {
+                               auto tilesizes =
+                                   ireeGPULoweringConfigAttrGetTileSizes(self);
+                               return {tilesizes.workgroupTileSizes,
+                                       tilesizes.workgroupTileSizes +
+                                           tilesizes.numWorkgroupTileSizes};
+                             })
+      .def_property_readonly("reduction_tile_sizes",
+                             [](MlirAttribute self) -> std::vector<int64_t> {
+                               auto tilesizes =
+                                   ireeGPULoweringConfigAttrGetTileSizes(self);
+                               return {tilesizes.reductionTileSizes,
+                                       tilesizes.reductionTileSizes +
+                                           tilesizes.numReductionTileSizes};
+                             })
+      .def_property_readonly(
+          "subgroup_count_mn",
+          [](MlirAttribute self) -> py::tuple {
+            ireeGPUSubgroupCountInfo info =
+                ireeGPULoweringConfigAttrGetSubgroupCount(self);
+            MlirAttribute mCountAttr = info.subgroupMCountAttr;
+            MlirAttribute nCountAttr = info.subgroupNCountAttr;
+            std::optional<int64_t> mCount =
+                mlirAttributeIsNull(mCountAttr)
+                    ? std::nullopt
+                    : std::optional<int64_t>(
+                          mlirIntegerAttrGetValueInt(mCountAttr));
+            std::optional<int64_t> nCount =
+                mlirAttributeIsNull(nCountAttr)
+                    ? std::nullopt
+                    : std::optional<int64_t>(
+                          mlirIntegerAttrGetValueInt(nCountAttr));
```

**Comment:**
These ternaries get pretty lengthy. Instead, I'd put the assignment in an if condition.

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:233`

```diff
@@ -213,3 +214,65 @@ MlirAttribute ireeGPULoweringConfigAttrGetAttributes(MlirAttribute attr) {
                   unwrap(attr))
                   .getAttributes());
 }
+
+ireeGPUTileSizes ireeGPULoweringConfigAttrGetTileSizes(MlirAttribute attr) {
+  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
+  ireeGPUTileSizes tilesizes = {};
+  auto loweringConfigAttr =
+      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
+          unwrap(attr));
+
+  llvm::SmallVector<int64_t> workgroups =
+      loweringConfigAttr.getWorkgroupTileSizes();
+  tilesizes.workgroupTileSizes = workgroups.data();
+  tilesizes.numWorkgroupTileSizes = workgroups.size();
+
+  llvm::SmallVector<int64_t> reductions =
+      loweringConfigAttr.getStaticTilingLevelSizes(
+          static_cast<int64_t>(
+              mlir::iree_compiler::IREE::GPU::TilingLevel::Reduction),
```

**Comment:**
use `llvm::to_underlying` to make sure we get the type right

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:236`

```diff
@@ -213,3 +214,65 @@ MlirAttribute ireeGPULoweringConfigAttrGetAttributes(MlirAttribute attr) {
                   unwrap(attr))
                   .getAttributes());
 }
+
+ireeGPUTileSizes ireeGPULoweringConfigAttrGetTileSizes(MlirAttribute attr) {
+  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
+  ireeGPUTileSizes tilesizes = {};
+  auto loweringConfigAttr =
+      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
+          unwrap(attr));
+
+  llvm::SmallVector<int64_t> workgroups =
+      loweringConfigAttr.getWorkgroupTileSizes();
+  tilesizes.workgroupTileSizes = workgroups.data();
+  tilesizes.numWorkgroupTileSizes = workgroups.size();
+
+  llvm::SmallVector<int64_t> reductions =
+      loweringConfigAttr.getStaticTilingLevelSizes(
+          llvm::to_underlying(
+              mlir::iree_compiler::IREE::GPU::TilingLevel::Reduction),
+          nullptr);
+  tilesizes.reductionTileSizes = reductions.data();
+  tilesizes.numReductionTileSizes = reductions.size();
```

**Comment:**
This returns dangling pointers. We should use the data from the attr storage -- these two should be array attributes.

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:367`

```diff
@@ -352,7 +352,66 @@ PYBIND11_MODULE(_ireeCompilerDialects, m) {
           "cls"_a, "value"_a, "ctx"_a = py::none(),
           "Gets an #iree_gpu.lowering_config from parameters.")
       .def_property_readonly("attributes",
-                             ireeGPULoweringConfigAttrGetAttributes);
+                             ireeGPULoweringConfigAttrGetAttributes)
+      .def_property_readonly(
+          "workgroup_tile_sizes",
+          [](MlirAttribute self) -> std::vector<int64_t> {
+            auto tilesizes = ireeGPULoweringConfigAttrGetTileSizes(self);
+            MlirAttribute workgroupAttr = tilesizes.workgroupAttr;
+            if (mlirAttributeIsNull(workgroupAttr)) {
+              return {};
+            }
+
+            size_t len = mlirArrayAttrGetNumElements(workgroupAttr);
+            std::vector<int64_t> workgroup(len);
+            for (size_t i = 0, e = len; i < e; ++i) {
```

**Comment:**
```suggestion
            for (size_t i = 0; i < len; ++i) {
```

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:384`

```diff
@@ -352,7 +352,66 @@ PYBIND11_MODULE(_ireeCompilerDialects, m) {
           "cls"_a, "value"_a, "ctx"_a = py::none(),
           "Gets an #iree_gpu.lowering_config from parameters.")
       .def_property_readonly("attributes",
-                             ireeGPULoweringConfigAttrGetAttributes);
+                             ireeGPULoweringConfigAttrGetAttributes)
+      .def_property_readonly(
+          "workgroup_tile_sizes",
+          [](MlirAttribute self) -> std::vector<int64_t> {
+            auto tilesizes = ireeGPULoweringConfigAttrGetTileSizes(self);
+            MlirAttribute workgroupAttr = tilesizes.workgroupAttr;
+            if (mlirAttributeIsNull(workgroupAttr)) {
+              return {};
+            }
+
+            size_t len = mlirArrayAttrGetNumElements(workgroupAttr);
+            std::vector<int64_t> workgroup(len);
+            for (size_t i = 0, e = len; i < e; ++i) {
+              MlirAttribute attr = mlirArrayAttrGetElement(workgroupAttr, i);
+              workgroup[i] = mlirIntegerAttrGetValueInt(attr);
+            }
+            return workgroup;
+          })
+      .def_property_readonly(
+          "reduction_tile_sizes",
+          [](MlirAttribute self) -> std::vector<int64_t> {
+            auto tilesizes = ireeGPULoweringConfigAttrGetTileSizes(self);
+            MlirAttribute reductionAttr = tilesizes.reductionAttr;
+            if (mlirAttributeIsNull(reductionAttr)) {
+              return {};
+            }
+
+            size_t len = mlirArrayAttrGetNumElements(reductionAttr);
+            std::vector<int64_t> reduction(len);
+            for (size_t i = 0, e = len; i < e; ++i) {
```

**Comment:**
```suggestion
            for (size_t i = 0; i < len; ++i) {
```

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:230`

```diff
@@ -213,3 +214,62 @@ MlirAttribute ireeGPULoweringConfigAttrGetAttributes(MlirAttribute attr) {
                   unwrap(attr))
                   .getAttributes());
 }
+
+ireeGPUTileSizes ireeGPULoweringConfigAttrGetTileSizes(MlirAttribute attr) {
+  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
+  ireeGPUTileSizes tilesizes = {};
+  mlir::DictionaryAttr dict =
+      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
+          unwrap(attr))
+          .getAttributes();
+
+  constexpr mlir::StringLiteral workgroupName = "workgroup";
+  if (auto workgroupArray = dict.getAs<mlir::ArrayAttr>(workgroupName)) {
```

**Comment:**
We should expose these string literals in the dialect headers.

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:231`

```diff
@@ -213,3 +214,62 @@ MlirAttribute ireeGPULoweringConfigAttrGetAttributes(MlirAttribute attr) {
                   unwrap(attr))
                   .getAttributes());
 }
+
+ireeGPUTileSizes ireeGPULoweringConfigAttrGetTileSizes(MlirAttribute attr) {
+  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
+  ireeGPUTileSizes tilesizes = {};
+  mlir::DictionaryAttr dict =
+      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
+          unwrap(attr))
+          .getAttributes();
+
+  constexpr mlir::StringLiteral workgroupName = "workgroup";
+  if (auto workgroupArray = dict.getAs<mlir::ArrayAttr>(workgroupName)) {
+    tilesizes.workgroupAttr = wrap(workgroupArray);
+  }
+
+  constexpr mlir::StringLiteral reductionName = "reduction";
```

**Comment:**
Also this one.

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h:89`

```diff
@@ -84,6 +84,11 @@ struct OpaqueMmaLayout {
 OpaqueMmaLayout getOpaqueMMALayout(MLIRContext *context,
                                    IREE::GPU::MMAIntrinsic intrinsic);
 
+/// Maps a GPU tiling level to its corresponding string representation. (e.g.,
+/// workgroup, reduction, etc.). If an invalid or unknown tiling level is
+/// provided, the function triggers an assertion failure.
```

**Comment:**
I'd say something more concise like:
```c++
/// Returns the name of the tilling `level`, as used in the `lowering_config` attribute.
```

---

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp:228`

```diff
@@ -213,3 +214,67 @@ MlirAttribute ireeGPULoweringConfigAttrGetAttributes(MlirAttribute attr) {
                   unwrap(attr))
                   .getAttributes());
 }
+
+ireeGPUTileSizes ireeGPULoweringConfigAttrGetTileSizes(MlirAttribute attr) {
+  assert(ireeAttributeIsAGPULoweringConfigAttr(attr));
+  ireeGPUTileSizes tilesizes = {};
+  mlir::DictionaryAttr dict =
+      llvm::cast<mlir::iree_compiler::IREE::GPU::LoweringConfigAttr>(
+          unwrap(attr))
+          .getAttributes();
+
+  llvm::StringRef workgroupName =
+      mlir::iree_compiler::IREE::GPU::getTilingLevelName(
+          mlir::iree_compiler::IREE::GPU::TilingLevel::Workgroup);
```

**Comment:**
looks much cleaner now!

---


---


## [PR #19218](https://github.com/iree-org/iree/pull/19218): [tuner]: add c/python binding for querying mma intrinsic

### Review Summary

**CHANGES_REQUESTED** (2024-11-20)

Nice. Looks good, just need to clean up the code a bit.

Let's keep the test around for now to make sure it continues working as we iterate on this code, but I think we should drop it before landing.


**COMMENTED** (2024-11-20)


**COMMENTED** (2024-11-20)


**COMMENTED** (2024-11-20)


**COMMENTED** (2024-11-20)


**APPROVED** (2024-11-20)

LGTM. Have you checked locally that your previous python test still works?


### Code Comments

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_codegen.h:71`

```diff
@@ -66,6 +66,14 @@ MLIR_CAPI_EXPORTED MlirAttribute ireeCodegenCompilationInfoAttrGet(
 MLIR_CAPI_EXPORTED ireeCodegenCompilationInfoParameters
 ireeCodegenCompilationInfoAttrGetParameters(MlirAttribute attr);
 
+MLIR_CAPI_EXPORTED void
+ireeCodegenGetExecutableVariantOps(MlirModule module, int *num_ops,
+                                   MlirOperation *executableOps);
```

**Comment:**
Let's keep the types consistent with how the other functions handle containers in this file.
```suggestion
ireeCodegenGetExecutableVariantOps(MlirModule module,  size_t *numOps,
                                   MlirOperation *executableOps);
```
also use `thisCase` for function arguments instead of the `snake_case`

---

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_codegen.h:75`

```diff
@@ -66,6 +66,14 @@ MLIR_CAPI_EXPORTED MlirAttribute ireeCodegenCompilationInfoAttrGet(
 MLIR_CAPI_EXPORTED ireeCodegenCompilationInfoParameters
 ireeCodegenCompilationInfoAttrGetParameters(MlirAttribute attr);
 
+MLIR_CAPI_EXPORTED void
+ireeCodegenGetExecutableVariantOps(MlirModule module, int *num_ops,
+                                   MlirOperation *executableOps);
+
+MLIR_CAPI_EXPORTED void ireeCodegenQueryMMAIntrinsics(MlirOperation op,
+                                                      int *num_intrinsics,
+                                                      uint32_t *mma_intrinsics);
```

**Comment:**
same here

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:24`

```diff
@@ -21,6 +21,38 @@ static const char *kGpuModuleImportPath =
 namespace py = pybind11;
 using namespace mlir::python::adaptors;
 
+py::list ireeCodegenGetExecutableVariantOpsBinding(MlirModule module) {
```

**Comment:**
You can return a vector: `std::vector<MlirOperation>`

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:53`

```diff
@@ -21,6 +21,38 @@ static const char *kGpuModuleImportPath =
 namespace py = pybind11;
 using namespace mlir::python::adaptors;
 
+py::list ireeCodegenGetExecutableVariantOpsBinding(MlirModule module) {
+  int numOps = 0;
+  ireeCodegenGetExecutableVariantOps(module, &numOps, nullptr);
+
+  std::vector<MlirOperation> ops(numOps);
+
+  ireeCodegenGetExecutableVariantOps(module, &numOps, ops.data());
+
+  py::list opsList;
+  for (int i = 0; i < numOps; ++i) {
+    opsList.append(ops[i]);
+  }
+
+  return opsList;
+}
+
+py::list ireeCodegenQueryMMAIntrinsicsBinding(MlirOperation op) {
+  int numMMAs = 0;
+  ireeCodegenQueryMMAIntrinsics(op, &numMMAs, nullptr);
+
+  std::vector<uint32_t> mmaIntrinsics(numMMAs);
+
+  ireeCodegenQueryMMAIntrinsics(op, &numMMAs, mmaIntrinsics.data());
+
+  py::list opsList;
+  for (int i = 0; i < numMMAs; ++i) {
+    opsList.append(mmaIntrinsics[i]);
+  }
+
+  return opsList;
```

**Comment:**
Here, we should return a list of enums, not integers. You can see how to construct an enum in the code that handles enum attributes below.

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:377`

```diff
@@ -326,4 +358,21 @@ PYBIND11_MODULE(_ireeCompilerDialects, m) {
           "Gets an #iree_gpu.lowering_config from parameters.")
       .def_property_readonly("attributes",
                              ireeGPULoweringConfigAttrGetAttributes);
+
+  //===-------------------------------------------------------------------===//
+  // Binding to utility function getExecutableVariantOps
+  //===-------------------------------------------------------------------===//
+
+  iree_codegen_module.def(
+      "get_executable_variant_ops", &ireeCodegenGetExecutableVariantOpsBinding,
+      "Gets the executable variant operations from a module.",
+      py::arg("module"));
+
+  //===-------------------------------------------------------------------===//
+  // Binding to utility function queryMMAIntrinsics
+  //===-------------------------------------------------------------------===//
+
+  iree_codegen_module.def(
+      "query_mma_intrinsics", &ireeCodegenQueryMMAIntrinsicsBinding,
+      "Querys the mma intrinsics from a executable variant op.", py::arg("op"));
```

**Comment:**
Please run your PR through a spell checker. I have a vscode extension for that.

---

**File:** `compiler/bindings/python/test/ir/dialects_test.py:240`

```diff
@@ -232,3 +232,44 @@ def compilation_info():
     assert compilation_info is not None
     assert compilation_info.lowering_config == lowering_config
     assert compilation_info.translation_info == translation_info
+
+
+@run
+def test_query_mma():
+    test_module = ir.Module.parse(
+        """
```

**Comment:**
I'm concerned this is a 'change detector' test. Every time we update any of the related dialects, we will have to come back to this test and also update it. It's worse than lit tests were at least you have the lsp helping you with syntax highlighting etc. IMO we can do without a test here -- the logic is tested on the C++ side with your test pass.

---

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp:157`

```diff
@@ -149,3 +150,41 @@ ireeCodegenCompilationInfoAttrGetParameters(MlirAttribute attr) {
   parameters.translationInfo = wrap(compilationInfo.getTranslationInfo());
   return parameters;
 }
+
+void ireeCodegenGetExecutableVariantOps(MlirModule module, int *num_ops,
+                                        MlirOperation *executableOps) {
+  mlir::ModuleOp moduleOp = unwrap(module);
+  llvm::SmallVector<mlir::iree_compiler::IREE::HAL::ExecutableVariantOp>
```

**Comment:**
You can make a typedef for this op type to define these longs namespaces

---

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp:156`

```diff
@@ -149,3 +150,41 @@ ireeCodegenCompilationInfoAttrGetParameters(MlirAttribute attr) {
   parameters.translationInfo = wrap(compilationInfo.getTranslationInfo());
   return parameters;
 }
+
+void ireeCodegenGetExecutableVariantOps(MlirModule module, int *num_ops,
+                                        MlirOperation *executableOps) {
+  mlir::ModuleOp moduleOp = unwrap(module);
```

**Comment:**
We should check that `num_ops` is not `nullptr`

---

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp:166`

```diff
@@ -149,3 +150,41 @@ ireeCodegenCompilationInfoAttrGetParameters(MlirAttribute attr) {
   parameters.translationInfo = wrap(compilationInfo.getTranslationInfo());
   return parameters;
 }
+
+void ireeCodegenGetExecutableVariantOps(MlirModule module, int *num_ops,
+                                        MlirOperation *executableOps) {
+  mlir::ModuleOp moduleOp = unwrap(module);
+  llvm::SmallVector<mlir::iree_compiler::IREE::HAL::ExecutableVariantOp>
+      executableVariantOps =
+          mlir::iree_compiler::getExecutableVariantOps(moduleOp);
+
+  if (!executableOps) {
+    *num_ops = executableVariantOps.size();
+    return;
+  }
+
+  for (size_t i = 0; i < executableVariantOps.size(); i++) {
```

**Comment:**
We should check that `num_ops` matches the number of variant ops.

---

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp:172`

```diff
@@ -149,3 +150,41 @@ ireeCodegenCompilationInfoAttrGetParameters(MlirAttribute attr) {
   parameters.translationInfo = wrap(compilationInfo.getTranslationInfo());
   return parameters;
 }
+
+void ireeCodegenGetExecutableVariantOps(MlirModule module, int *num_ops,
+                                        MlirOperation *executableOps) {
+  mlir::ModuleOp moduleOp = unwrap(module);
+  llvm::SmallVector<mlir::iree_compiler::IREE::HAL::ExecutableVariantOp>
+      executableVariantOps =
+          mlir::iree_compiler::getExecutableVariantOps(moduleOp);
+
+  if (!executableOps) {
+    *num_ops = executableVariantOps.size();
+    return;
+  }
+
+  for (size_t i = 0; i < executableVariantOps.size(); i++) {
+    executableOps[i] = wrap(executableVariantOps[i]);
+  }
+}
+
+void ireeCodegenQueryMMAIntrinsics(MlirOperation op, int *num_intrinsics,
+                                   uint32_t *mma_intrinsics) {
```

**Comment:**
Similar in this function

---

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp:175`

```diff
@@ -149,3 +150,41 @@ ireeCodegenCompilationInfoAttrGetParameters(MlirAttribute attr) {
   parameters.translationInfo = wrap(compilationInfo.getTranslationInfo());
   return parameters;
 }
+
+void ireeCodegenGetExecutableVariantOps(MlirModule module, int *num_ops,
+                                        MlirOperation *executableOps) {
+  mlir::ModuleOp moduleOp = unwrap(module);
+  llvm::SmallVector<mlir::iree_compiler::IREE::HAL::ExecutableVariantOp>
+      executableVariantOps =
+          mlir::iree_compiler::getExecutableVariantOps(moduleOp);
+
+  if (!executableOps) {
+    *num_ops = executableVariantOps.size();
+    return;
+  }
+
+  for (size_t i = 0; i < executableVariantOps.size(); i++) {
+    executableOps[i] = wrap(executableVariantOps[i]);
+  }
+}
+
+void ireeCodegenQueryMMAIntrinsics(MlirOperation op, int *num_intrinsics,
+                                   uint32_t *mma_intrinsics) {
+  mlir::Operation *mlirOp = unwrap(op);
+  auto variantOp =
+      llvm::dyn_cast<mlir::iree_compiler::IREE::HAL::ExecutableVariantOp>(
```

**Comment:**
```suggestion
      llvm::dyn_cast_if_present<mlir::iree_compiler::IREE::HAL::ExecutableVariantOp>(
```

---

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp:189`

```diff
@@ -149,3 +150,41 @@ ireeCodegenCompilationInfoAttrGetParameters(MlirAttribute attr) {
   parameters.translationInfo = wrap(compilationInfo.getTranslationInfo());
   return parameters;
 }
+
+void ireeCodegenGetExecutableVariantOps(MlirModule module, int *num_ops,
+                                        MlirOperation *executableOps) {
+  mlir::ModuleOp moduleOp = unwrap(module);
+  llvm::SmallVector<mlir::iree_compiler::IREE::HAL::ExecutableVariantOp>
+      executableVariantOps =
+          mlir::iree_compiler::getExecutableVariantOps(moduleOp);
+
+  if (!executableOps) {
+    *num_ops = executableVariantOps.size();
+    return;
+  }
+
+  for (size_t i = 0; i < executableVariantOps.size(); i++) {
+    executableOps[i] = wrap(executableVariantOps[i]);
+  }
+}
+
+void ireeCodegenQueryMMAIntrinsics(MlirOperation op, int *num_intrinsics,
+                                   uint32_t *mma_intrinsics) {
+  mlir::Operation *mlirOp = unwrap(op);
+  auto variantOp =
+      llvm::dyn_cast<mlir::iree_compiler::IREE::HAL::ExecutableVariantOp>(
+          mlirOp);
+
+  assert(variantOp && "operation is not a ExecutableVariantOp");
+
+  llvm::SmallVector<mlir::iree_compiler::IREE::GPU::MMAIntrinsic>
+      mmaIntrinsics = mlir::iree_compiler::queryMMAIntrinsics(variantOp);
+  if (!mma_intrinsics) {
+    *num_intrinsics = mmaIntrinsics.size();
+    return;
+  }
+
+  for (size_t i = 0; i < mmaIntrinsics.size(); i++) {
+    mma_intrinsics[i] = static_cast<uint32_t>(mmaIntrinsics[i]);
+  }
```

**Comment:**
Follow the llvm coding style for loops: https://llvm.org/docs/CodingStandards.html#don-t-evaluate-end-every-time-through-a-loop

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp:1033`

```diff
@@ -1030,7 +1030,7 @@ std::optional<int> getGPUSubgroupSize(mlir::FunctionOpInterface func) {
 
 SmallVector<IREE::HAL::ExecutableVariantOp>
 getExecutableVariantOps(mlir::ModuleOp moduleOp) {
-  llvm::SmallVector<IREE::HAL::ExecutableVariantOp> executableVariantOps;
+  SmallVector<IREE::HAL::ExecutableVariantOp> executableVariantOps;
```

**Comment:**
This looks like an old change?

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp:1042`

```diff
@@ -1039,7 +1039,7 @@ getExecutableVariantOps(mlir::ModuleOp moduleOp) {
 
 SmallVector<IREE::GPU::MMAIntrinsic>
 queryMMAIntrinsics(IREE::HAL::ExecutableVariantOp executableOp) {
-  llvm::SmallVector<IREE::GPU::MMAIntrinsic> mmaIntrinsics;
+  SmallVector<IREE::GPU::MMAIntrinsic> mmaIntrinsics;
```

**Comment:**
also here

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:41`

```diff
@@ -21,36 +21,33 @@ static const char *kGpuModuleImportPath =
 namespace py = pybind11;
 using namespace mlir::python::adaptors;
 
-py::list ireeCodegenGetExecutableVariantOpsBinding(MlirModule module) {
-  int numOps = 0;
+std::vector<MlirOperation>
+ireeCodegenGetExecutableVariantOpsBinding(MlirModule module) {
+  size_t numOps = 0;
   ireeCodegenGetExecutableVariantOps(module, &numOps, nullptr);
 
   std::vector<MlirOperation> ops(numOps);
 
   ireeCodegenGetExecutableVariantOps(module, &numOps, ops.data());
 
-  py::list opsList;
-  for (int i = 0; i < numOps; ++i) {
-    opsList.append(ops[i]);
-  }
-
-  return opsList;
+  return ops;
 }
 
-py::list ireeCodegenQueryMMAIntrinsicsBinding(MlirOperation op) {
-  int numMMAs = 0;
+std::vector<py::object> ireeCodegenQueryMMAIntrinsicsBinding(MlirOperation op) {
+  size_t numMMAs = 0;
   ireeCodegenQueryMMAIntrinsics(op, &numMMAs, nullptr);
 
   std::vector<uint32_t> mmaIntrinsics(numMMAs);
 
```

**Comment:**
```suggestion
  std::vector<uint32_t> mmaIntrinsics(numMMAs);
```

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:48`

```diff
@@ -21,36 +21,33 @@ static const char *kGpuModuleImportPath =
 namespace py = pybind11;
 using namespace mlir::python::adaptors;
 
-py::list ireeCodegenGetExecutableVariantOpsBinding(MlirModule module) {
-  int numOps = 0;
+std::vector<MlirOperation>
+ireeCodegenGetExecutableVariantOpsBinding(MlirModule module) {
+  size_t numOps = 0;
   ireeCodegenGetExecutableVariantOps(module, &numOps, nullptr);
 
   std::vector<MlirOperation> ops(numOps);
 
   ireeCodegenGetExecutableVariantOps(module, &numOps, ops.data());
 
-  py::list opsList;
-  for (int i = 0; i < numOps; ++i) {
-    opsList.append(ops[i]);
-  }
-
-  return opsList;
+  return ops;
 }
 
-py::list ireeCodegenQueryMMAIntrinsicsBinding(MlirOperation op) {
-  int numMMAs = 0;
+std::vector<py::object> ireeCodegenQueryMMAIntrinsicsBinding(MlirOperation op) {
+  size_t numMMAs = 0;
   ireeCodegenQueryMMAIntrinsics(op, &numMMAs, nullptr);
 
   std::vector<uint32_t> mmaIntrinsics(numMMAs);
 
   ireeCodegenQueryMMAIntrinsics(op, &numMMAs, mmaIntrinsics.data());
 
-  py::list opsList;
-  for (int i = 0; i < numMMAs; ++i) {
-    opsList.append(mmaIntrinsics[i]);
-  }
-
-  return opsList;
+  std::vector<py::object> mmaList(numMMAs);
+  std::transform(mmaIntrinsics.begin(), mmaIntrinsics.end(), mmaList.begin(),
+                 [&](uint32_t rawValue) {
+                   return py::module_::import(kGpuModuleImportPath)
+                       .attr("MMAIntrinsic")(rawValue);
```

**Comment:**
We should get the enum att once and reuse it instead of importing the module N times.

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:30`

```diff
@@ -21,6 +21,35 @@ static const char *kGpuModuleImportPath =
 namespace py = pybind11;
 using namespace mlir::python::adaptors;
 
+std::vector<MlirOperation>
+ireeCodegenGetExecutableVariantOpsBinding(MlirModule module) {
+  size_t numOps = 0;
+  ireeCodegenGetExecutableVariantOps(module, &numOps, nullptr);
+
+  std::vector<MlirOperation> ops(numOps);
+
```

**Comment:**
```suggestion
  std::vector<MlirOperation> ops(numOps);
```

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:42`

```diff
@@ -21,6 +21,35 @@ static const char *kGpuModuleImportPath =
 namespace py = pybind11;
 using namespace mlir::python::adaptors;
 
+std::vector<MlirOperation>
+ireeCodegenGetExecutableVariantOpsBinding(MlirModule module) {
+  size_t numOps = 0;
+  ireeCodegenGetExecutableVariantOps(module, &numOps, nullptr);
+
+  std::vector<MlirOperation> ops(numOps);
+
+  ireeCodegenGetExecutableVariantOps(module, &numOps, ops.data());
+
+  return ops;
+}
+
+std::vector<py::object> ireeCodegenQueryMMAIntrinsicsBinding(MlirOperation op) {
+  size_t numMMAs = 0;
+  ireeCodegenQueryMMAIntrinsics(op, &numMMAs, nullptr);
+
+  std::vector<uint32_t> mmaIntrinsics(numMMAs);
+
+  ireeCodegenQueryMMAIntrinsics(op, &numMMAs, mmaIntrinsics.data());
```

**Comment:**
```suggestion
  ireeCodegenQueryMMAIntrinsics(op, &numMMAs, nullptr);
  std::vector<uint32_t> mmaIntrinsics(numMMAs);
  ireeCodegenQueryMMAIntrinsics(op, &numMMAs, mmaIntrinsics.data());
```

---

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp:158`

```diff
@@ -149,3 +152,49 @@ ireeCodegenCompilationInfoAttrGetParameters(MlirAttribute attr) {
   parameters.translationInfo = wrap(compilationInfo.getTranslationInfo());
   return parameters;
 }
+
+void ireeCodegenGetExecutableVariantOps(MlirModule module, size_t *numOps,
+                                        MlirOperation *executableOps) {
+  assert(numOps && "numOps cannot be nullptr");
```

**Comment:**
We should also assert that `module` is not null

---

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp:184`

```diff
@@ -149,3 +152,49 @@ ireeCodegenCompilationInfoAttrGetParameters(MlirAttribute attr) {
   parameters.translationInfo = wrap(compilationInfo.getTranslationInfo());
   return parameters;
 }
+
+void ireeCodegenGetExecutableVariantOps(MlirModule module, size_t *numOps,
+                                        MlirOperation *executableOps) {
+  assert(numOps && "numOps cannot be nullptr");
+
+  mlir::ModuleOp moduleOp = unwrap(module);
+  llvm::SmallVector<ExecutableVariantOp> executableVariantOps =
+      mlir::iree_compiler::getExecutableVariantOps(moduleOp);
+
+  if (!executableOps) {
+    *numOps = executableVariantOps.size();
+    return;
+  }
+
+  assert(
+      *numOps == executableVariantOps.size() &&
+      "*numOps must match the number of elements in the executableVariantOps");
+
+  for (size_t i = 0, e = executableVariantOps.size(); i < e; ++i) {
+    executableOps[i] = wrap(executableVariantOps[i]);
+  }
+}
+
+void ireeCodegenQueryMMAIntrinsics(MlirOperation op, size_t *numIntrinsics,
+                                   uint32_t *mmaIntrinsics) {
+  assert(numIntrinsics && "numIntrinsics cannot be nullptr");
+
+  mlir::Operation *mlirOp = unwrap(op);
+  auto variantOp = llvm::dyn_cast_if_present<ExecutableVariantOp>(mlirOp);
+
```

**Comment:**
```suggestion
```

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp:1033`

```diff
@@ -1030,7 +1030,7 @@ std::optional<int> getGPUSubgroupSize(mlir::FunctionOpInterface func) {
 
 SmallVector<IREE::HAL::ExecutableVariantOp>
 getExecutableVariantOps(mlir::ModuleOp moduleOp) {
-  llvm::SmallVector<IREE::HAL::ExecutableVariantOp> executableVariantOps;
+  SmallVector<IREE::HAL::ExecutableVariantOp> executableVariantOps;
```

**Comment:**
This is not resolved. If these are old changes, we should rebase this PR to be reasonably up-to-date with main.

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp:1042`

```diff
@@ -1039,7 +1039,7 @@ getExecutableVariantOps(mlir::ModuleOp moduleOp) {
 
 SmallVector<IREE::GPU::MMAIntrinsic>
 queryMMAIntrinsics(IREE::HAL::ExecutableVariantOp executableOp) {
-  llvm::SmallVector<IREE::GPU::MMAIntrinsic> mmaIntrinsics;
+  SmallVector<IREE::GPU::MMAIntrinsic> mmaIntrinsics;
```

**Comment:**
same here

---

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp:158`

```diff
@@ -155,6 +155,7 @@ ireeCodegenCompilationInfoAttrGetParameters(MlirAttribute attr) {
 
 void ireeCodegenGetExecutableVariantOps(MlirModule module, size_t *numOps,
                                         MlirOperation *executableOps) {
+  assert(module.ptr && "module cannot be nullptr");
```

**Comment:**
use the c function to check this instead of accessing `.ptr` directly

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:46`

```diff
@@ -21,6 +21,32 @@ static const char *kGpuModuleImportPath =
 namespace py = pybind11;
 using namespace mlir::python::adaptors;
 
+std::vector<MlirOperation>
+ireeCodegenGetExecutableVariantOpsBinding(MlirModule module) {
+  size_t numOps = 0;
+  ireeCodegenGetExecutableVariantOps(module, &numOps, nullptr);
+
+  std::vector<MlirOperation> ops(numOps);
+
+  ireeCodegenGetExecutableVariantOps(module, &numOps, ops.data());
+
+  return ops;
+}
+
+std::vector<py::object> ireeCodegenQueryMMAIntrinsicsBinding(MlirOperation op) {
+  size_t numMMAs = 0;
+  ireeCodegenQueryMMAIntrinsics(op, &numMMAs, nullptr);
+  std::vector<uint32_t> mmaIntrinsics(numMMAs);
+  ireeCodegenQueryMMAIntrinsics(op, &numMMAs, mmaIntrinsics.data());
+
+  py::object mmaIntrinsicEnum =
+      py::module_::import(kGpuModuleImportPath).attr("MMAIntrinsic");
+  std::vector<py::object> mmaList(numMMAs);
+  std::transform(mmaIntrinsics.begin(), mmaIntrinsics.end(), mmaList.begin(),
+                 [&](uint32_t rawValue) { return mmaIntrinsicEnum(rawValue); });
```

**Comment:**
nit: I'd make this a for loop, I don't think the transform helps the readability much...

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:24`

```diff
@@ -21,6 +21,32 @@ static const char *kGpuModuleImportPath =
 namespace py = pybind11;
 using namespace mlir::python::adaptors;
 
+std::vector<MlirOperation>
```

**Comment:**
```suggestion
static std::vector<MlirOperation>
```

---

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp:36`

```diff
@@ -21,6 +21,32 @@ static const char *kGpuModuleImportPath =
 namespace py = pybind11;
 using namespace mlir::python::adaptors;
 
+std::vector<MlirOperation>
+ireeCodegenGetExecutableVariantOpsBinding(MlirModule module) {
+  size_t numOps = 0;
+  ireeCodegenGetExecutableVariantOps(module, &numOps, nullptr);
+
+  std::vector<MlirOperation> ops(numOps);
+
+  ireeCodegenGetExecutableVariantOps(module, &numOps, ops.data());
+
+  return ops;
+}
+
+std::vector<py::object> ireeCodegenQueryMMAIntrinsicsBinding(MlirOperation op) {
```

**Comment:**
```suggestion
static std::vector<py::object> ireeCodegenQueryMMAIntrinsicsBinding(MlirOperation op) {
```

---


---


## [PR #19199](https://github.com/iree-org/iree/pull/19199): [tuner]: two new utility functions which are more friendly for c binding

### Review Summary

**COMMENTED** (2024-11-19)


**APPROVED** (2024-11-19)

LGTM % one nit


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp:1034`

```diff
@@ -1028,22 +1028,25 @@ std::optional<int> getGPUSubgroupSize(mlir::FunctionOpInterface func) {
   return std::nullopt;
 }
 
-llvm::SmallDenseMap<IREE::HAL::ExecutableVariantOp,
-                    SmallVector<IREE::GPU::MMAIntrinsic>>
-queryMMAIntrinsics(mlir::ModuleOp moduleOp) {
-  llvm::SmallDenseMap<IREE::HAL::ExecutableVariantOp,
-                      SmallVector<IREE::GPU::MMAIntrinsic>>
-      mmaAttributesMap;
-  moduleOp.walk([&](IREE::HAL::ExecutableVariantOp executableOp) {
-    if (IREE::GPU::TargetAttr target = getGPUTargetAttr(executableOp)) {
-      auto mmaIntrinsics = llvm::map_to_vector(
-          target.getWgp().getMma(), [](IREE::GPU::MMAAttr attr) {
-            return attr.getIntrinsic().getValue();
-          });
-      mmaAttributesMap[executableOp] = std::move(mmaIntrinsics);
-    }
-  });
-  return mmaAttributesMap;
+llvm::SmallVector<IREE::HAL::ExecutableVariantOp>
+getExecutableVariantOps(mlir::ModuleOp moduleOp) {
+  llvm::SmallVector<IREE::HAL::ExecutableVariantOp> executableVariantOps;
+  moduleOp.walk<WalkOrder::PreOrder>(
```

**Comment:**
Why did you change it to pre-order?

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp:1033`

```diff
@@ -1028,22 +1028,25 @@ std::optional<int> getGPUSubgroupSize(mlir::FunctionOpInterface func) {
   return std::nullopt;
 }
 
-llvm::SmallDenseMap<IREE::HAL::ExecutableVariantOp,
-                    SmallVector<IREE::GPU::MMAIntrinsic>>
-queryMMAIntrinsics(mlir::ModuleOp moduleOp) {
-  llvm::SmallDenseMap<IREE::HAL::ExecutableVariantOp,
-                      SmallVector<IREE::GPU::MMAIntrinsic>>
-      mmaAttributesMap;
-  moduleOp.walk([&](IREE::HAL::ExecutableVariantOp executableOp) {
-    if (IREE::GPU::TargetAttr target = getGPUTargetAttr(executableOp)) {
-      auto mmaIntrinsics = llvm::map_to_vector(
-          target.getWgp().getMma(), [](IREE::GPU::MMAAttr attr) {
-            return attr.getIntrinsic().getValue();
-          });
-      mmaAttributesMap[executableOp] = std::move(mmaIntrinsics);
-    }
-  });
-  return mmaAttributesMap;
+llvm::SmallVector<IREE::HAL::ExecutableVariantOp>
+getExecutableVariantOps(mlir::ModuleOp moduleOp) {
+  llvm::SmallVector<IREE::HAL::ExecutableVariantOp> executableVariantOps;
```

**Comment:**
```suggestion
SmallVector<IREE::HAL::ExecutableVariantOp>
getExecutableVariantOps(mlir::ModuleOp moduleOp) {
  SmallVector<IREE::HAL::ExecutableVariantOp> executableVariantOps;
```

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h:211`

```diff
@@ -207,13 +207,16 @@ IREE::GPU::TargetAttr getGPUTargetAttr(Operation *op);
 /// Returns std::nullopt if none found.
 std::optional<int> getGPUSubgroupSize(mlir::FunctionOpInterface func);
 
-/// Returns a map of supported MMA intrinsic instructions based on the
-/// GPU target descriptions in `moduleOp`. Each entry in the map associates
-/// an `IREE::HAL::ExecutableVariantOp` with a vector of
-/// `IREE::GPU::MMAIntrinsic` attributes.
-llvm::SmallDenseMap<IREE::HAL::ExecutableVariantOp,
-                    SmallVector<IREE::GPU::MMAIntrinsic>>
-queryMMAIntrinsics(mlir::ModuleOp moduleOp);
+/// Returns all `IREE::HAL::ExecutableVariantOp` operations from the
+/// given `mlir::ModuleOp`,  ensuring they are returned in their original IR
```

**Comment:**
```suggestion
/// given `mlir::ModuleOp`, ensuring they are returned in their original IR
```

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h:213`

```diff
@@ -207,13 +207,16 @@ IREE::GPU::TargetAttr getGPUTargetAttr(Operation *op);
 /// Returns std::nullopt if none found.
 std::optional<int> getGPUSubgroupSize(mlir::FunctionOpInterface func);
 
-/// Returns a map of supported MMA intrinsic instructions based on the
-/// GPU target descriptions in `moduleOp`. Each entry in the map associates
-/// an `IREE::HAL::ExecutableVariantOp` with a vector of
-/// `IREE::GPU::MMAIntrinsic` attributes.
-llvm::SmallDenseMap<IREE::HAL::ExecutableVariantOp,
-                    SmallVector<IREE::GPU::MMAIntrinsic>>
-queryMMAIntrinsics(mlir::ModuleOp moduleOp);
+/// Returns all `IREE::HAL::ExecutableVariantOp` operations from the
+/// given `mlir::ModuleOp`,  ensuring they are returned in their original IR
+/// order.
+llvm::SmallVector<IREE::HAL::ExecutableVariantOp>
```

**Comment:**
```suggestion
SmallVector<IREE::HAL::ExecutableVariantOp>
```

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h:218`

```diff
@@ -207,13 +207,16 @@ IREE::GPU::TargetAttr getGPUTargetAttr(Operation *op);
 /// Returns std::nullopt if none found.
 std::optional<int> getGPUSubgroupSize(mlir::FunctionOpInterface func);
 
-/// Returns a map of supported MMA intrinsic instructions based on the
-/// GPU target descriptions in `moduleOp`. Each entry in the map associates
-/// an `IREE::HAL::ExecutableVariantOp` with a vector of
-/// `IREE::GPU::MMAIntrinsic` attributes.
-llvm::SmallDenseMap<IREE::HAL::ExecutableVariantOp,
-                    SmallVector<IREE::GPU::MMAIntrinsic>>
-queryMMAIntrinsics(mlir::ModuleOp moduleOp);
+/// Returns all `IREE::HAL::ExecutableVariantOp` operations from the
+/// given `mlir::ModuleOp`,  ensuring they are returned in their original IR
+/// order.
+llvm::SmallVector<IREE::HAL::ExecutableVariantOp>
+getExecutableVariantOps(mlir::ModuleOp moduleOp);
+
+// Returns the MMA intrinsics associated with the given
+// `IREE::HAL::ExecutableVariantOp`.
+llvm::SmallVector<IREE::GPU::MMAIntrinsic>
```

**Comment:**
```suggestion
SmallVector<IREE::GPU::MMAIntrinsic>
```

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/TestLLVMGPUQueryMMAPass.cpp:28`

```diff
@@ -23,15 +23,16 @@ struct TestLLVMGPUQueryMMAPass final
     : impl::TestLLVMGPUQueryMMAPassBase<TestLLVMGPUQueryMMAPass> {
   void runOnOperation() override {
     ModuleOp moduleOp = getOperation();
-    llvm::SmallDenseMap<IREE::HAL::ExecutableVariantOp,
-                        SmallVector<IREE::GPU::MMAIntrinsic>>
-        mmaMap = queryMMAIntrinsics(moduleOp);
-    for (const auto &[op, mmaAttrs] : mmaMap) {
+    SmallVector<IREE::HAL::ExecutableVariantOp> executableVariantOps =
+        getExecutableVariantOps(moduleOp);
+    for (const auto &op : executableVariantOps) {
```

**Comment:**
You can use values for IR constructs -- they are designed to be cheap to pass by value. Also use the actual type, since the type is not obvious based on the RHS only.

---


---


## [PR #19124](https://github.com/iree-org/iree/pull/19124): [tuner]: Add a utility function to query supported MMA intrinsics

### Review Summary

**CHANGES_REQUESTED** (2024-11-13)

> and expose it to C API and python

This doesn't expose the new helper to C or python. Did you forget to add some files when pushing or is that coming in a future PR?


**COMMENTED** (2024-11-13)


**COMMENTED** (2024-11-14)


**COMMENTED** (2024-11-14)

Looks pretty good now, just a few more comments.

We should also update the title of the PR so that we don't advertise C API  and Python bindings which are not in the PR.


**COMMENTED** (2024-11-14)


**COMMENTED** (2024-11-14)


**COMMENTED** (2024-11-14)


**COMMENTED** (2024-11-14)


**COMMENTED** (2024-11-14)


**APPROVED** (2024-11-14)

LGTM


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/TestLLVMGPUQueryMMAPass.cpp:27`

```diff
@@ -0,0 +1,48 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
+#include "iree/compiler/Codegen/Utils/GPUUtils.h"
+#include "mlir/Dialect/Func/IR/FuncOps.h"
+
+#include "llvm/Support/Debug.h"
+
+#define DEBUG_TYPE "iree-test-llvmgpu-query-mma"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_TESTLLVMGPUQUERYMMAPASS
+#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"
+
+static void printMMAVector(SmallVector<IREE::GPU::MMAAttr> &mmaAttrs,
+                           const std::string &extraMessage = {}) {
+  llvm::outs() << "Printing MMA Collection" << extraMessage
+               << ", size: " << mmaAttrs.size() << "\n";
+  for (const auto &mma : mmaAttrs) {
+    llvm::outs() << mma << " ";
+  }
+  llvm::outs() << "\n";
```

**Comment:**
This helper doesn't really do anything -- we can inline it into the pass use `llvm::interleaveComma` instead of the for loop.

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/TestLLVMGPUQueryMMAPass.cpp:39`

```diff
@@ -0,0 +1,48 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
+#include "iree/compiler/Codegen/Utils/GPUUtils.h"
+#include "mlir/Dialect/Func/IR/FuncOps.h"
+
+#include "llvm/Support/Debug.h"
+
+#define DEBUG_TYPE "iree-test-llvmgpu-query-mma"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_TESTLLVMGPUQUERYMMAPASS
+#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"
+
+static void printMMAVector(SmallVector<IREE::GPU::MMAAttr> &mmaAttrs,
+                           const std::string &extraMessage = {}) {
+  llvm::outs() << "Printing MMA Collection" << extraMessage
+               << ", size: " << mmaAttrs.size() << "\n";
+  for (const auto &mma : mmaAttrs) {
+    llvm::outs() << mma << " ";
+  }
+  llvm::outs() << "\n";
+}
+
+namespace {
+
+struct TestLLVMGPUQueryMMAPass final
+    : impl::TestLLVMGPUQueryMMAPassBase<TestLLVMGPUQueryMMAPass> {
+  void runOnOperation() override {
+    ModuleOp moduleOp = getOperation();
+    SmallVector<IREE::GPU::MMAAttr> mmaCollecton;
+    // Print mma vector before collection.
+    printMMAVector(mmaCollecton,
+                   " Before querying supported mma instrinsic instructions");
```

**Comment:**
There's no point in printing an empty vector

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/test_query_mma.mlir:5`

```diff
@@ -0,0 +1,60 @@
+// RUN: iree-opt --split-input-file --iree-test-llvmgpu-query-mma %s | FileCheck %s
+
+#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
+{abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "",
+wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8,
```

**Comment:**
We don't need an exhaustive list all the other wgp properties -- we can trim it down to something minimal like `compute = int32, storage = b32, ...`

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp:1037`

```diff
@@ -1028,4 +1029,27 @@ std::optional<int> getGPUSubgroupSize(mlir::FunctionOpInterface func) {
   return std::nullopt;
 }
 
+void QueryMMAIntrinsics(mlir::ModuleOp moduleOp,
+                        SmallVector<IREE::GPU::MMAAttr> &mmaAttrs) {
+  IREE::GPU::TargetAttr target;
+
+  // Walk through all `func::FuncOp` operations in `moduleOp`.
+  moduleOp.walk([&](func::FuncOp funcOp) {
```

**Comment:**
Wouldn't it be enough to look up `hal.executable.variant` only? This is where the attribute is attached to.

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp:1036`

```diff
@@ -1028,4 +1029,27 @@ std::optional<int> getGPUSubgroupSize(mlir::FunctionOpInterface func) {
   return std::nullopt;
 }
 
+void QueryMMAIntrinsics(mlir::ModuleOp moduleOp,
+                        SmallVector<IREE::GPU::MMAAttr> &mmaAttrs) {
+  IREE::GPU::TargetAttr target;
+
+  // Walk through all `func::FuncOp` operations in `moduleOp`.
```

**Comment:**
This comment doesn't clarify anything -- `moduleOp.walk` is a very basic function and the intention is obvious here.

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp:1033`

```diff
@@ -1028,4 +1029,27 @@ std::optional<int> getGPUSubgroupSize(mlir::FunctionOpInterface func) {
   return std::nullopt;
 }
 
+void QueryMMAIntrinsics(mlir::ModuleOp moduleOp,
+                        SmallVector<IREE::GPU::MMAAttr> &mmaAttrs) {
```

**Comment:**
I think this should probably return a mapping of executable variants to their mma attrs. Should should also have a test of a module with two variants.

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp:1050`

```diff
@@ -1028,4 +1029,27 @@ std::optional<int> getGPUSubgroupSize(mlir::FunctionOpInterface func) {
   return std::nullopt;
 }
 
+void QueryMMAIntrinsics(mlir::ModuleOp moduleOp,
+                        SmallVector<IREE::GPU::MMAAttr> &mmaAttrs) {
+  IREE::GPU::TargetAttr target;
+
+  // Walk through all `func::FuncOp` operations in `moduleOp`.
+  moduleOp.walk([&](func::FuncOp funcOp) {
+    if (auto attr = getGPUTargetAttr(funcOp)) {
+      // Store the target attribute if found.
+      target = attr;
+      return WalkResult::interrupt();
+    }
+    return WalkResult::advance();
+  });
+
+  if (target) {
+    // Append each MMA attribute from the target's `Wgp` configuration to
+    // `mmaAttrs`.
+    for (IREE::GPU::MMAAttr mma : target.getWgp().getMma()) {
+      mmaAttrs.emplace_back(mma);
```

**Comment:**
we should be able to append all of them at once with `llvm::append_range`

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h:211`

```diff
@@ -206,6 +206,11 @@ IREE::GPU::TargetAttr getGPUTargetAttr(Operation *op);
 /// Returns std::nullopt if none found.
 std::optional<int> getGPUSubgroupSize(mlir::FunctionOpInterface func);
 
+/// Returns supported MMA intrinsic instructions based on the GPU target
+/// description stored in `moduleOp` and populates them in `mmaAttrs`.
+void QueryMMAIntrinsics(mlir::ModuleOp moduleOp,
```

**Comment:**
```suggestion
void queryMMAIntrinsics(mlir::ModuleOp moduleOp,
```

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h:212`

```diff
@@ -206,6 +206,11 @@ IREE::GPU::TargetAttr getGPUTargetAttr(Operation *op);
 /// Returns std::nullopt if none found.
 std::optional<int> getGPUSubgroupSize(mlir::FunctionOpInterface func);
 
+/// Returns supported MMA intrinsic instructions based on the GPU target
+/// description stored in `moduleOp` and populates them in `mmaAttrs`.
+void QueryMMAIntrinsics(mlir::ModuleOp moduleOp,
+                        SmallVector<IREE::GPU::MMAAttr> &mmaAttrs);
```

**Comment:**
Can we return mma intrinsics attrs instead of mma attr? I think we can always go from an intrsic to an mma but not the other way round?

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/TestLLVMGPUQueryMMAPass.cpp:31`

```diff
@@ -17,31 +17,31 @@ namespace mlir::iree_compiler {
 #define GEN_PASS_DEF_TESTLLVMGPUQUERYMMAPASS
 #include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"
 
-static void printMMAVector(SmallVector<IREE::GPU::MMAAttr> &mmaAttrs,
-                           const std::string &extraMessage = {}) {
-  llvm::outs() << "Printing MMA Collection" << extraMessage
-               << ", size: " << mmaAttrs.size() << "\n";
-  for (const auto &mma : mmaAttrs) {
-    llvm::outs() << mma << " ";
-  }
-  llvm::outs() << "\n";
-}
-
 namespace {
 
 struct TestLLVMGPUQueryMMAPass final
     : impl::TestLLVMGPUQueryMMAPassBase<TestLLVMGPUQueryMMAPass> {
   void runOnOperation() override {
     ModuleOp moduleOp = getOperation();
-    SmallVector<IREE::GPU::MMAAttr> mmaCollecton;
-    // Print mma vector before collection.
-    printMMAVector(mmaCollecton,
-                   " Before querying supported mma instrinsic instructions");
-    // Collect mma intrinsic instructions.
-    QueryMMAIntrinsics(moduleOp, mmaCollecton);
-    // Print mma vector after collection.
-    printMMAVector(mmaCollecton,
-                   " After querying supported mma instrinsic instructions");
+    llvm::SmallDenseMap<Operation *, SmallVector<IREE::GPU::MMAIntrinsic>>
+        mmaMap;
+    queryMMAIntrinsics(moduleOp, mmaMap);
+    for (const auto &entry : mmaMap) {
+      Operation *op = entry.first;
+      const SmallVector<IREE::GPU::MMAIntrinsic> &mmaAttrs = entry.second;
```

**Comment:**
You can use structured bindings to unpack these two members to variables

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/TestLLVMGPUQueryMMAPass.cpp:42`

```diff
@@ -17,31 +17,31 @@ namespace mlir::iree_compiler {
 #define GEN_PASS_DEF_TESTLLVMGPUQUERYMMAPASS
 #include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"
 
-static void printMMAVector(SmallVector<IREE::GPU::MMAAttr> &mmaAttrs,
-                           const std::string &extraMessage = {}) {
-  llvm::outs() << "Printing MMA Collection" << extraMessage
-               << ", size: " << mmaAttrs.size() << "\n";
-  for (const auto &mma : mmaAttrs) {
-    llvm::outs() << mma << " ";
-  }
-  llvm::outs() << "\n";
-}
-
 namespace {
 
 struct TestLLVMGPUQueryMMAPass final
     : impl::TestLLVMGPUQueryMMAPassBase<TestLLVMGPUQueryMMAPass> {
   void runOnOperation() override {
     ModuleOp moduleOp = getOperation();
-    SmallVector<IREE::GPU::MMAAttr> mmaCollecton;
-    // Print mma vector before collection.
-    printMMAVector(mmaCollecton,
-                   " Before querying supported mma instrinsic instructions");
-    // Collect mma intrinsic instructions.
-    QueryMMAIntrinsics(moduleOp, mmaCollecton);
-    // Print mma vector after collection.
-    printMMAVector(mmaCollecton,
-                   " After querying supported mma instrinsic instructions");
+    llvm::SmallDenseMap<Operation *, SmallVector<IREE::GPU::MMAIntrinsic>>
+        mmaMap;
+    queryMMAIntrinsics(moduleOp, mmaMap);
+    for (const auto &entry : mmaMap) {
+      Operation *op = entry.first;
+      const SmallVector<IREE::GPU::MMAIntrinsic> &mmaAttrs = entry.second;
+      if (auto variantOp = llvm::dyn_cast<IREE::HAL::ExecutableVariantOp>(op)) {
+        llvm::outs() << "Executable Variant Name: " << variantOp.getName()
+                     << "\n";
+      } else {
+        llvm::outs() << "Executable Variant Name: " << "Unnamed Operation"
+                     << "\n";
+      }
+      llvm::outs() << "MMA Intrinsics: ";
+      for (const auto &mma : mmaAttrs) {
+        llvm::outs() << mma << " ";
+      }
```

**Comment:**
use  `llvm::interleave` -- you can give it the desired separator

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/test_query_mma.mlir:40`

```diff
@@ -25,18 +22,65 @@ module {
   }
 }
 
-// CHECK: Printing MMA Collection Before querying supported mma instrinsic instructions, size: 0
-// CHECK: Printing MMA Collection After querying supported mma instrinsic instructions, size: 9
-// CHECK: MFMA_F32_16x16x4_F32
+// CHECK:       Executable Variant Name
+// CHECK-SAME:  main
+// CHECK: MMA   Intrinsics
+// CHECK-SAME:  MFMA_F32_16x16x4_F32
+// CHECK-SAME:  MFMA_F32_16x16x16_F16
+// CHECK-LABEL: func.func @fn
+
+// -----
+
+#executable_target_rocm_hsaco_fb0 = #hal.executable.target<"rocm", "rocm-hsaco-fb",
+{abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "",
+wgp = <compute = int32, storage =  b32,
+subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32,
+mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>],
+subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024],
+max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536,
```

**Comment:**
could we further trim this down but skipping some of these `wgp` properties out entirely?

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h:214`

```diff
@@ -207,9 +207,11 @@ IREE::GPU::TargetAttr getGPUTargetAttr(Operation *op);
 std::optional<int> getGPUSubgroupSize(mlir::FunctionOpInterface func);
 
 /// Returns supported MMA intrinsic instructions based on the GPU target
-/// description stored in `moduleOp` and populates them in `mmaAttrs`.
-void QueryMMAIntrinsics(mlir::ModuleOp moduleOp,
-                        SmallVector<IREE::GPU::MMAAttr> &mmaAttrs);
+/// description stored in `moduleOp` and populates them in `MMAIntrinsic`.
+void queryMMAIntrinsics(
+    mlir::ModuleOp moduleOp,
+    llvm::SmallDenseMap<Operation *, SmallVector<IREE::GPU::MMAIntrinsic>>
+        &mmaAttributesMap);
```

**Comment:**
Why not return this map?

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp:1041`

```diff
@@ -1028,4 +1029,22 @@ std::optional<int> getGPUSubgroupSize(mlir::FunctionOpInterface func) {
   return std::nullopt;
 }
 
+void queryMMAIntrinsics(
+    mlir::ModuleOp moduleOp,
+    llvm::SmallDenseMap<Operation *, SmallVector<IREE::GPU::MMAIntrinsic>>
+        &mmaAttributesMap) {
+  moduleOp.walk([&](IREE::HAL::ExecutableVariantOp executableOp) {
+    if (IREE::GPU::TargetAttr target = getGPUTargetAttr(executableOp)) {
+      SmallVector<IREE::GPU::MMAIntrinsic> mmaIntrinsics;
+      llvm::append_range(
+          mmaIntrinsics,
+          llvm::map_range(target.getWgp().getMma(),
```

**Comment:**
Instead of appending to an empty range, use `llvm::map_to_vector`.

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h:213`

```diff
@@ -206,6 +206,13 @@ IREE::GPU::TargetAttr getGPUTargetAttr(Operation *op);
 /// Returns std::nullopt if none found.
 std::optional<int> getGPUSubgroupSize(mlir::FunctionOpInterface func);
 
+/// Returns a map of supported MMA intrinsic instructions based on the
+/// GPU target descriptions in `moduleOp`. Each entry in the map associates
+/// an `Operation*` ( an `IREE::HAL::ExecutableVariantOp`) with a
+/// vector of `IREE::GPU::MMAIntrinsic` attributes.
+llvm::SmallDenseMap<Operation *, SmallVector<IREE::GPU::MMAIntrinsic>>
```

**Comment:**
Why do we want to return `Operation *` instead of `ExecutableVariantOp`? I think this would simplify both code and the documentation.

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/TestLLVMGPUQueryMMAPass.cpp:35`

```diff
@@ -0,0 +1,43 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
+#include "iree/compiler/Codegen/Utils/GPUUtils.h"
+#include "mlir/Dialect/Func/IR/FuncOps.h"
+
+#include "llvm/Support/Debug.h"
+
+#define DEBUG_TYPE "iree-test-llvmgpu-query-mma"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_TESTLLVMGPUQUERYMMAPASS
+#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"
+
+namespace {
+
+struct TestLLVMGPUQueryMMAPass final
+    : impl::TestLLVMGPUQueryMMAPassBase<TestLLVMGPUQueryMMAPass> {
+  void runOnOperation() override {
+    ModuleOp moduleOp = getOperation();
+    llvm::SmallDenseMap<Operation *, SmallVector<IREE::GPU::MMAIntrinsic>>
+        mmaMap = queryMMAIntrinsics(moduleOp);
+    for (const auto &[op, mmaAttrs] : mmaMap) {
+      if (auto variantOp = llvm::dyn_cast<IREE::HAL::ExecutableVariantOp>(op)) {
+        llvm::outs() << "Executable Variant Name: " << variantOp.getName()
+                     << "\n";
+      } else {
+        llvm::outs() << "Executable Variant Name: " << "Unnamed Operation"
+                     << "\n";
+      }
```

**Comment:**
Can this actually happen? I don't see it in the LIT test.

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp:1042`

```diff
@@ -1028,4 +1029,20 @@ std::optional<int> getGPUSubgroupSize(mlir::FunctionOpInterface func) {
   return std::nullopt;
 }
 
+llvm::SmallDenseMap<Operation *, SmallVector<IREE::GPU::MMAIntrinsic>>
+queryMMAIntrinsics(mlir::ModuleOp moduleOp) {
+  llvm::SmallDenseMap<Operation *, SmallVector<IREE::GPU::MMAIntrinsic>>
+      mmaAttributesMap;
+  moduleOp.walk([&](IREE::HAL::ExecutableVariantOp executableOp) {
+    if (IREE::GPU::TargetAttr target = getGPUTargetAttr(executableOp)) {
+      auto mmaIntrinsics = llvm::map_to_vector(
+          target.getWgp().getMma(), [](IREE::GPU::MMAAttr attr) {
+            return attr.getIntrinsic().getValue();
+          });
+      mmaAttributesMap[executableOp] = mmaIntrinsics;
```

**Comment:**
```suggestion
      mmaAttributesMap[executableOp] = std::move(mmaIntrinsics);
```
to avoid needless copying

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp:21`

```diff
@@ -18,6 +18,7 @@
 #include "llvm/Support/ErrorHandling.h"
 #include "mlir/Dialect/Affine/IR/AffineOps.h"
 #include "mlir/Dialect/Arith/IR/Arith.h"
+#include "mlir/Dialect/Func/IR/FuncOps.h"
```

**Comment:**
Is this include still necessary?

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/TestLLVMGPUQueryMMAPass.cpp:31`

```diff
@@ -23,16 +23,13 @@ struct TestLLVMGPUQueryMMAPass final
     : impl::TestLLVMGPUQueryMMAPassBase<TestLLVMGPUQueryMMAPass> {
   void runOnOperation() override {
     ModuleOp moduleOp = getOperation();
-    llvm::SmallDenseMap<Operation *, SmallVector<IREE::GPU::MMAIntrinsic>>
+    llvm::SmallDenseMap<IREE::HAL::ExecutableVariantOp,
+                        SmallVector<IREE::GPU::MMAIntrinsic>>
         mmaMap = queryMMAIntrinsics(moduleOp);
     for (const auto &[op, mmaAttrs] : mmaMap) {
-      if (auto variantOp = llvm::dyn_cast<IREE::HAL::ExecutableVariantOp>(op)) {
-        llvm::outs() << "Executable Variant Name: " << variantOp.getName()
-                     << "\n";
-      } else {
-        llvm::outs() << "Executable Variant Name: " << "Unnamed Operation"
-                     << "\n";
-      }
+      llvm::outs() << "Executable Variant Name: "
+                   << cast<IREE::HAL::ExecutableVariantOp>(*op).getName()
```

**Comment:**
Why do we need the cast? I'd think the type of op is already executable variant op, no?

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h:213`

```diff
@@ -210,7 +211,8 @@ std::optional<int> getGPUSubgroupSize(mlir::FunctionOpInterface func);
 /// GPU target descriptions in `moduleOp`. Each entry in the map associates
 /// an `Operation*` ( an `IREE::HAL::ExecutableVariantOp`) with a
 /// vector of `IREE::GPU::MMAIntrinsic` attributes.
```

**Comment:**
This is out of date

---


---


## [PR #19069](https://github.com/iree-org/iree/pull/19069): [tuner] add an iree-opt pass to strip configuration from executable sources

### Review Summary

**COMMENTED** (2024-11-08)

Thanks for taking on this task. It seems to be functional in the current state already, now just need to clean it up a bit and make it landable.


**COMMENTED** (2024-11-08)


**COMMENTED** (2024-11-09)


**COMMENTED** (2024-11-09)


**COMMENTED** (2024-11-09)


**COMMENTED** (2024-11-11)

Just one remaining issue, looks good to me otherwise


**COMMENTED** (2024-11-11)


**COMMENTED** (2024-11-11)


**APPROVED** (2024-11-11)

LGTM, thanks for the fixes.


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Common/Passes.td:495`

```diff
@@ -490,6 +490,11 @@ def RemoveSingleIterationLoopPass :
   let summary = "Remove distributed loop with single iteration.";
 }
 
+def StripConfigInfoPass :
+    InterfacePass<"iree-codegen-strip-config-info", "mlir::FunctionOpInterface">{
+   let summary = "Remove all the the lowering configuration and translation info.";
```

**Comment:**
I'd call it `StripCompilationInfo`. The reason is that the `#iree_codegen.compilation_info` attribute contains both lowering config and translation info, so that'd encompass all 3 attributes

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripConfigInfoPass.cpp:20`

```diff
@@ -0,0 +1,44 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/Common/Passes.h"
+
+#define DEBUG_TYPE "iree-codegen-strip-config-info"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_STRIPCONFIGINFOPASS
+#include "iree/compiler/Codegen/Common/Passes.h.inc"
+
+namespace {
+class StripConfigInfoPass final
+    : public impl::StripConfigInfoPassBase<StripConfigInfoPass> {
+  using impl::StripConfigInfoPassBase<
+      StripConfigInfoPass>::StripConfigInfoPassBase;
```

**Comment:**
```suggestion
  using impl::StripConfigInfoPassBass::StripConfigInfoPassBase;
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripConfigInfoPass.cpp:18`

```diff
@@ -0,0 +1,44 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/Common/Passes.h"
+
+#define DEBUG_TYPE "iree-codegen-strip-config-info"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_STRIPCONFIGINFOPASS
+#include "iree/compiler/Codegen/Common/Passes.h.inc"
+
+namespace {
+class StripConfigInfoPass final
+    : public impl::StripConfigInfoPassBase<StripConfigInfoPass> {
```

**Comment:**
nit: `struct` will save us some typing, since we don't need to hide data members from anybody anyway -- this pass is defined in an anonymous namespace, so only this file knows its full type anyway
```suggestion
struct StripConfigInfoPass final
    : impl::StripConfigInfoPassBase<StripConfigInfoPass> {
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripConfigInfoPass.cpp:22`

```diff
@@ -0,0 +1,44 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/Common/Passes.h"
+
+#define DEBUG_TYPE "iree-codegen-strip-config-info"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_STRIPCONFIGINFOPASS
+#include "iree/compiler/Codegen/Common/Passes.h.inc"
+
+namespace {
+class StripConfigInfoPass final
+    : public impl::StripConfigInfoPassBase<StripConfigInfoPass> {
+  using impl::StripConfigInfoPassBase<
+      StripConfigInfoPass>::StripConfigInfoPassBase;
+
+public:
```

**Comment:**
```suggestion
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripConfigInfoPass.cpp:27`

```diff
@@ -0,0 +1,44 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/Common/Passes.h"
+
+#define DEBUG_TYPE "iree-codegen-strip-config-info"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_STRIPCONFIGINFOPASS
+#include "iree/compiler/Codegen/Common/Passes.h.inc"
+
+namespace {
+class StripConfigInfoPass final
+    : public impl::StripConfigInfoPassBase<StripConfigInfoPass> {
+  using impl::StripConfigInfoPassBase<
+      StripConfigInfoPass>::StripConfigInfoPassBase;
+
+public:
+  void runOnOperation() override;
+};
+} // namespace
+
+void StripConfigInfoPass::runOnOperation() {
```

**Comment:**
I'd define this inline -- I don't think we gain much by outlining this function

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripConfigInfoPass.cpp:31`

```diff
@@ -0,0 +1,44 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/Common/Passes.h"
+
+#define DEBUG_TYPE "iree-codegen-strip-config-info"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_STRIPCONFIGINFOPASS
+#include "iree/compiler/Codegen/Common/Passes.h.inc"
+
+namespace {
+class StripConfigInfoPass final
+    : public impl::StripConfigInfoPassBase<StripConfigInfoPass> {
+  using impl::StripConfigInfoPassBase<
+      StripConfigInfoPass>::StripConfigInfoPassBase;
+
+public:
+  void runOnOperation() override;
+};
+} // namespace
+
+void StripConfigInfoPass::runOnOperation() {
+  auto funcOp = getOperation();
+  IREE::Codegen::TranslationInfoAttr translationInfo =
+      getTranslationInfo(funcOp);
+  if (translationInfo) {
```

**Comment:**
We don't have to keep `translationInfo` as a local variable -- we never use it beyond checking that it's there. I'd do the same thing as you do with lowering config below.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripConfigInfoPass.cpp:37`

```diff
@@ -0,0 +1,44 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/Common/Passes.h"
+
+#define DEBUG_TYPE "iree-codegen-strip-config-info"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_STRIPCONFIGINFOPASS
+#include "iree/compiler/Codegen/Common/Passes.h.inc"
+
+namespace {
+class StripConfigInfoPass final
+    : public impl::StripConfigInfoPassBase<StripConfigInfoPass> {
+  using impl::StripConfigInfoPassBase<
+      StripConfigInfoPass>::StripConfigInfoPassBase;
+
+public:
+  void runOnOperation() override;
+};
+} // namespace
+
+void StripConfigInfoPass::runOnOperation() {
+  auto funcOp = getOperation();
+  IREE::Codegen::TranslationInfoAttr translationInfo =
+      getTranslationInfo(funcOp);
+  if (translationInfo) {
+    // Erase the translation info from function if it exists.
+    eraseTranslationInfo(funcOp);
+  }
+
+  funcOp->walk([&](Operation *op) {
+    if (getLoweringConfig(op)) {
```

**Comment:**
We should be also stripping compilation info attributes for completeness sake IMO. These won't be produced by the compiler in the default flow, but would appear as the result of applying the tuning specs (transform dialect library).

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_config_info.mlir:3`

```diff
@@ -0,0 +1,103 @@
+// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-strip-config-info)))))' %s | FileCheck %s
+
+#config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, promote_operands = [0, 1], reduction = [0, 0, 64], subgroup_m_count = 2 : i64, subgroup_n_count = 2 : i64, workgroup = [64, 128, 0]}>
```

**Comment:**
+1, we can write a small test case by hand that doesn't use the actual dumps from iree-opt. For example, setting `lowering_config` to a `StringAttr` should exercise the code just as well.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_config_info.mlir:1`

```diff
@@ -0,0 +1,103 @@
+// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-strip-config-info)))))' %s | FileCheck %s
```

**Comment:**
Ah, now I think we should make this pass over `Operation` so that we don't have to manually specify the nesting. We can walk whatever the input op is and find functions ops first.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:22`

```diff
@@ -0,0 +1,80 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/Common/Passes.h"
+#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
+
+#define DEBUG_TYPE "iree-codegen-strip-compilation-info"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_STRIPCOMPILATIONINFOPASS
+#include "iree/compiler/Codegen/Common/Passes.h.inc"
+
+namespace {
+
+// Checks whether the funcOp has any compilation info.
+LogicalResult hasCompilationInfo(func::FuncOp funcOp) {
+  if (getTranslationInfo(funcOp))
+    return success();
```

**Comment:**
This assumption is not correct. We may have lowering config / compilation info but not translation info. This would be true for modules configured by TD scripts.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:20`

```diff
@@ -0,0 +1,80 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/Common/Passes.h"
+#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
+
+#define DEBUG_TYPE "iree-codegen-strip-compilation-info"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_STRIPCOMPILATIONINFOPASS
+#include "iree/compiler/Codegen/Common/Passes.h.inc"
+
+namespace {
+
+// Checks whether the funcOp has any compilation info.
+LogicalResult hasCompilationInfo(func::FuncOp funcOp) {
```

**Comment:**
IMO this can return `bool` -- it's very clear what the meaning of `true`/`false` is.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:37`

```diff
@@ -0,0 +1,80 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/Common/Passes.h"
+#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
+
+#define DEBUG_TYPE "iree-codegen-strip-compilation-info"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_STRIPCOMPILATIONINFOPASS
+#include "iree/compiler/Codegen/Common/Passes.h.inc"
+
+namespace {
+
+// Checks whether the funcOp has any compilation info.
+LogicalResult hasCompilationInfo(func::FuncOp funcOp) {
+  if (getTranslationInfo(funcOp))
+    return success();
+
+  bool hasAttrConfig = false;
+  funcOp.walk([&](Operation *op) {
+    if (getCompilationInfo(op) || getLoweringConfig(op)) {
+      hasAttrConfig = true;
+      return;
+    }
+  });
+
+  // Return success if any relevant attributes were found.
+  return hasAttrConfig ? success() : failure();
+}
+
+class StripCompilationInfo : public OpRewritePattern<func::FuncOp> {
+public:
```

**Comment:**
Previous comments not addressed

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:45`

```diff
@@ -0,0 +1,80 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/Common/Passes.h"
+#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
+
+#define DEBUG_TYPE "iree-codegen-strip-compilation-info"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_STRIPCOMPILATIONINFOPASS
+#include "iree/compiler/Codegen/Common/Passes.h.inc"
+
+namespace {
+
+// Checks whether the funcOp has any compilation info.
+LogicalResult hasCompilationInfo(func::FuncOp funcOp) {
+  if (getTranslationInfo(funcOp))
+    return success();
+
+  bool hasAttrConfig = false;
+  funcOp.walk([&](Operation *op) {
+    if (getCompilationInfo(op) || getLoweringConfig(op)) {
+      hasAttrConfig = true;
+      return;
+    }
+  });
+
+  // Return success if any relevant attributes were found.
+  return hasAttrConfig ? success() : failure();
+}
+
+class StripCompilationInfo : public OpRewritePattern<func::FuncOp> {
+public:
+  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
+  LogicalResult matchAndRewrite(func::FuncOp funcOp,
+                                PatternRewriter &rewriter) const final {
+    if (failed(hasCompilationInfo(funcOp)))
+      return failure();
+
+    func::FuncOp newFuncOp =
+        dyn_cast<func::FuncOp>(rewriter.clone(*funcOp.getOperation()));
```

**Comment:**
Do we really need to clone the function to make this work? Can't we modify it in place?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:75`

```diff
@@ -0,0 +1,80 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/Common/Passes.h"
+#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
+
+#define DEBUG_TYPE "iree-codegen-strip-compilation-info"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_STRIPCOMPILATIONINFOPASS
+#include "iree/compiler/Codegen/Common/Passes.h.inc"
+
+namespace {
+
+// Checks whether the funcOp has any compilation info.
+LogicalResult hasCompilationInfo(func::FuncOp funcOp) {
+  if (getTranslationInfo(funcOp))
+    return success();
+
+  bool hasAttrConfig = false;
+  funcOp.walk([&](Operation *op) {
+    if (getCompilationInfo(op) || getLoweringConfig(op)) {
+      hasAttrConfig = true;
+      return;
+    }
+  });
+
+  // Return success if any relevant attributes were found.
+  return hasAttrConfig ? success() : failure();
+}
+
+class StripCompilationInfo : public OpRewritePattern<func::FuncOp> {
+public:
+  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
+  LogicalResult matchAndRewrite(func::FuncOp funcOp,
+                                PatternRewriter &rewriter) const final {
+    if (failed(hasCompilationInfo(funcOp)))
+      return failure();
+
+    func::FuncOp newFuncOp =
+        dyn_cast<func::FuncOp>(rewriter.clone(*funcOp.getOperation()));
+
+    // if the cloned function has translation info, erase it
+    if (getTranslationInfo(newFuncOp)) {
+      eraseTranslationInfo(newFuncOp);
+    }
+
+    newFuncOp->walk([&](Operation *op) {
+      if (getCompilationInfo(op)) {
+        // Erase the compilation info configuration if it exists
+        eraseCompilationInfo(op);
+      }
+      if (getLoweringConfig(op)) {
+        // Erase the lowering configuration from root operation if it
+        // exists.
+        eraseLoweringConfig(op);
+      }
+    });
+
+    rewriter.replaceOp(funcOp, newFuncOp);
+    return success();
+  }
+};
+
+struct StripCompilationInfoPass final
+    : impl::StripCompilationInfoPassBase<StripCompilationInfoPass> {
+  void runOnOperation() override {
+    RewritePatternSet patterns(&getContext());
+    patterns.add<StripCompilationInfo>(&getContext());
+    if (failed(
+            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
```

**Comment:**
You don't need the greedy rewriter here, you can use `walkAndApplyPatterns` which should be much cheaper.

But I'm not sure we need it any case -- the previous approach with a manual `walk` seemed fine to me

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir:57`

```diff
@@ -0,0 +1,64 @@
+// RUN: iree-opt --split-input-file --iree-codegen-strip-compilation-info %s | FileCheck %s
+
+#translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64>
+func.func @main() attributes {translation_info = #translation_info} {
+    return
+}
+
+// CHECK-LABEL: func.func @main
+// CHECK-NOT:   #translation_info =
+// CHECK-NOT:   LLVMGPUVectorDistribute
+
+#pipeline_layout = #hal.pipeline.layout<bindings = [
+  #hal.pipeline.binding<storage_buffer>
+]>
+hal.executable private @strip_main {
+  hal.executable.variant public @strip_main target(#hal.executable.target<"", "", {}>) {
+    hal.executable.export public @entry_point layout(#pipeline_layout)
+    builtin.module {
+      func.func @fn1() attributes {translation_info = #iree_codegen.translation_info<None subgroup_size = 32>}  {
+        return
+      }
+      func.func @fn2() attributes {translation_info = #iree_codegen.translation_info<None subgroup_size = 32>} {
+        return
+      }
+    }
+  }
+}
+
+// CHECK-LABEL: hal.executable private @strip_main
+// CHECK: @fn1
+// CHECK-NOT:   #translation_info =
+// CHECK: @fn2
+// CHECK-NOT:   #translation_info =
+// CHECK: return
+
+#layout = #hal.pipeline.layout<bindings = [
+  #hal.pipeline.binding<storage_buffer>,
+  #hal.pipeline.binding<storage_buffer>,
+  #hal.pipeline.binding<storage_buffer>
+]>
+#config = #iree_codegen.lowering_config<tile_sizes = [[128, 256], [16, 16]]>
+#translation = #iree_codegen.translation_info<None workgroup_size = [16, 8, 1] subgroup_size = 64>
+#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
+func.func @matmul_128x1024x256() {
+  %cst = arith.constant 0.000000e+00 : f32
+  %c128 = arith.constant 128 : index
+  %c1024 = arith.constant 1024 : index
+  %c0 = arith.constant 0 : index
+  %0 = hal.interface.binding.subspan layout(#layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<128x256xf32>>
+  %1 = hal.interface.binding.subspan layout(#layout) binding(1) : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>>
+  %2 = hal.interface.binding.subspan layout(#layout) binding(2) : !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
+  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x256xf32>> -> tensor<128x256xf32>
+  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x1024xf32>> -> tensor<256x1024xf32>
+  %5 = tensor.empty() : tensor<128x1024xf32>
+  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
+  %7 = linalg.matmul {compilation_info = #compilation} ins(%3, %4 : tensor<128x256xf32>, tensor<256x1024xf32>) outs(%6 : tensor<128x1024xf32>) -> tensor<128x1024xf32>
+  flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [128, 1024], strides = [1, 1] : tensor<128x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x1024xf32>>
```

**Comment:**
We can further simplify this. We can have a function with just a couple of ops that accepts tensors and does matmul on the operands.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:36`

```diff
@@ -0,0 +1,73 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/Common/Passes.h"
+#include "mlir/Transforms/WalkPatternRewriteDriver.h"
+
+#define DEBUG_TYPE "iree-codegen-strip-compilation-info"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_STRIPCOMPILATIONINFOPASS
+#include "iree/compiler/Codegen/Common/Passes.h.inc"
+
+namespace {
+
+// Checks whether the funcOp has any compilation info.
+bool hasCompilationInfo(func::FuncOp funcOp) {
+  if (getTranslationInfo(funcOp))
+    return true;
+
+  bool hasAttrConfig = false;
+  funcOp.walk([&](Operation *op) {
+    if (getCompilationInfo(op) || getLoweringConfig(op)) {
+      hasAttrConfig = true;
+      return;
+    }
+  });
+
+  // Return success if any relevant attributes were found.
+  return hasAttrConfig;
+}
+
+struct StripCompilationInfo : public OpRewritePattern<func::FuncOp> {
```

**Comment:**
```suggestion
struct StripCompilationInfo final : OpRewritePattern<func::FuncOp> {
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:48`

```diff
@@ -0,0 +1,73 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/Common/Passes.h"
+#include "mlir/Transforms/WalkPatternRewriteDriver.h"
+
+#define DEBUG_TYPE "iree-codegen-strip-compilation-info"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_STRIPCOMPILATIONINFOPASS
+#include "iree/compiler/Codegen/Common/Passes.h.inc"
+
+namespace {
+
+// Checks whether the funcOp has any compilation info.
+bool hasCompilationInfo(func::FuncOp funcOp) {
+  if (getTranslationInfo(funcOp))
+    return true;
+
+  bool hasAttrConfig = false;
+  funcOp.walk([&](Operation *op) {
+    if (getCompilationInfo(op) || getLoweringConfig(op)) {
+      hasAttrConfig = true;
+      return;
+    }
+  });
+
+  // Return success if any relevant attributes were found.
+  return hasAttrConfig;
+}
+
+struct StripCompilationInfo : public OpRewritePattern<func::FuncOp> {
+  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
+  LogicalResult matchAndRewrite(func::FuncOp funcOp,
+                                PatternRewriter &rewriter) const final {
+    if (!hasCompilationInfo(funcOp))
+      return failure();
+
+    // if the function has translation info, erase it
+    if (getTranslationInfo(funcOp)) {
+      eraseTranslationInfo(funcOp);
+    }
+
+    funcOp->walk([&](Operation *op) {
```

**Comment:**
The `hasCompilationInfo` check seems redundant -- we may just as well remember if any modification were performed or not.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:43`

```diff
@@ -0,0 +1,73 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/Common/Passes.h"
+#include "mlir/Transforms/WalkPatternRewriteDriver.h"
+
+#define DEBUG_TYPE "iree-codegen-strip-compilation-info"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_STRIPCOMPILATIONINFOPASS
+#include "iree/compiler/Codegen/Common/Passes.h.inc"
+
+namespace {
+
+// Checks whether the funcOp has any compilation info.
+bool hasCompilationInfo(func::FuncOp funcOp) {
+  if (getTranslationInfo(funcOp))
+    return true;
+
+  bool hasAttrConfig = false;
+  funcOp.walk([&](Operation *op) {
+    if (getCompilationInfo(op) || getLoweringConfig(op)) {
+      hasAttrConfig = true;
+      return;
+    }
+  });
+
+  // Return success if any relevant attributes were found.
+  return hasAttrConfig;
+}
+
+struct StripCompilationInfo : public OpRewritePattern<func::FuncOp> {
+  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
+  LogicalResult matchAndRewrite(func::FuncOp funcOp,
+                                PatternRewriter &rewriter) const final {
+    if (!hasCompilationInfo(funcOp))
+      return failure();
+
+    // if the function has translation info, erase it
```

**Comment:**
Use proper casing and punctuation

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir:45`

```diff
@@ -0,0 +1,52 @@
+// RUN: iree-opt --split-input-file --iree-codegen-strip-compilation-info %s | FileCheck %s
+
+#translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64>
+func.func @main() attributes {translation_info = #translation_info} {
+    return
+}
+
+// CHECK-LABEL: func.func @main
+// CHECK-NOT:   #translation_info =
+// CHECK-NOT:   LLVMGPUVectorDistribute
+
+#pipeline_layout = #hal.pipeline.layout<bindings = [
+  #hal.pipeline.binding<storage_buffer>
+]>
+hal.executable private @strip_main {
+  hal.executable.variant public @strip_main target(#hal.executable.target<"", "", {}>) {
+    hal.executable.export public @entry_point layout(#pipeline_layout)
+    builtin.module {
+      func.func @fn1() attributes {translation_info = #iree_codegen.translation_info<None subgroup_size = 32>}  {
+        return
+      }
+      func.func @fn2() attributes {translation_info = #iree_codegen.translation_info<None subgroup_size = 32>} {
+        return
+      }
+    }
+  }
+}
+
+// CHECK-LABEL: hal.executable private @strip_main
+// CHECK: @fn1
+// CHECK-NOT:   #translation_info =
+// CHECK: @fn2
+// CHECK-NOT:   #translation_info =
+// CHECK: return
+
+#layout = #hal.pipeline.layout<bindings = [
+  #hal.pipeline.binding<storage_buffer>,
+  #hal.pipeline.binding<storage_buffer>,
+  #hal.pipeline.binding<storage_buffer>
+]>
+#config = #iree_codegen.lowering_config<tile_sizes = [[128, 256], [16, 16]]>
+#translation = #iree_codegen.translation_info<None workgroup_size = [16, 8, 1] subgroup_size = 64>
+#compilation = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>
+func.func @matmul_128x1024x256(%lhs : tensor<128x256xf32>, %rhs: tensor<256x1024xf32>, %init: tensor<128x1024xf32>) -> tensor<128x1024xf32> {
+    %result =  linalg.matmul {compilation_info = #compilation} ins(%lhs, %rhs : tensor<128x256xf32>, tensor<256x1024xf32>) outs(%init : tensor<128x1024xf32>) -> tensor<128x1024xf32>
```

**Comment:**
We should also add one test with just lowering config attached to an op

---

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir:9`

```diff
@@ -0,0 +1,52 @@
+// RUN: iree-opt --split-input-file --iree-codegen-strip-compilation-info %s | FileCheck %s
+
+#translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [64, 1, 1] subgroup_size = 64>
+func.func @main() attributes {translation_info = #translation_info} {
+    return
+}
+
+// CHECK-LABEL: func.func @main
+// CHECK-NOT:   #translation_info =
```

**Comment:**
Most of the checks in this file seem off: there's no `#` symbol on the attribute name

---

**File:** `compiler/src/iree/compiler/Codegen/Common/Passes.td:495`

```diff
@@ -490,6 +490,11 @@ def RemoveSingleIterationLoopPass :
   let summary = "Remove distributed loop with single iteration.";
 }
 
+def StripCompilationInfoPass :
+    Pass<"iree-codegen-strip-compilation-info", "">{
+   let summary = "Remove all the the lowering configuration and translation info.";
```

**Comment:**
```suggestion
   let summary = "Remove all the the lowering configuration and translation info attributes.";
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:36`

```diff
@@ -0,0 +1,73 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/Common/Passes.h"
+#include "mlir/Transforms/WalkPatternRewriteDriver.h"
+
+#define DEBUG_TYPE "iree-codegen-strip-compilation-info"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_STRIPCOMPILATIONINFOPASS
+#include "iree/compiler/Codegen/Common/Passes.h.inc"
+
+namespace {
+
+// Checks whether the funcOp has any compilation info.
+bool hasCompilationInfo(func::FuncOp funcOp) {
+  if (getTranslationInfo(funcOp))
+    return true;
+
+  bool hasAttrConfig = false;
+  funcOp.walk([&](Operation *op) {
+    if (getCompilationInfo(op) || getLoweringConfig(op)) {
+      hasAttrConfig = true;
+      return;
+    }
+  });
+
+  // Return success if any relevant attributes were found.
+  return hasAttrConfig;
+}
+
+struct StripCompilationInfo : public OpRewritePattern<func::FuncOp> {
```

**Comment:**
Can we make this pattern be over the func op interface?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:53`

```diff
@@ -0,0 +1,73 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/Common/Passes.h"
+#include "mlir/Transforms/WalkPatternRewriteDriver.h"
+
+#define DEBUG_TYPE "iree-codegen-strip-compilation-info"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_STRIPCOMPILATIONINFOPASS
+#include "iree/compiler/Codegen/Common/Passes.h.inc"
+
+namespace {
+
+// Checks whether the funcOp has any compilation info.
+bool hasCompilationInfo(func::FuncOp funcOp) {
+  if (getTranslationInfo(funcOp))
+    return true;
+
+  bool hasAttrConfig = false;
+  funcOp.walk([&](Operation *op) {
+    if (getCompilationInfo(op) || getLoweringConfig(op)) {
+      hasAttrConfig = true;
+      return;
+    }
+  });
+
+  // Return success if any relevant attributes were found.
+  return hasAttrConfig;
+}
+
+struct StripCompilationInfo : public OpRewritePattern<func::FuncOp> {
+  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
+  LogicalResult matchAndRewrite(func::FuncOp funcOp,
+                                PatternRewriter &rewriter) const final {
+    if (!hasCompilationInfo(funcOp))
+      return failure();
+
+    // if the function has translation info, erase it
+    if (getTranslationInfo(funcOp)) {
+      eraseTranslationInfo(funcOp);
+    }
+
+    funcOp->walk([&](Operation *op) {
+      if (getCompilationInfo(op)) {
+        // Erase the compilation info configuration if it exists
+        eraseCompilationInfo(op);
+      }
+      if (getLoweringConfig(op)) {
+        // Erase the lowering configuration from root operation if it
+        // exists.
+        eraseLoweringConfig(op);
+      }
```

**Comment:**
Also, in rewrite patterns, it's invalid to mutate the IR without going through the rewriter: https://mlir.llvm.org/docs/PatternRewriter/#common-pattern-drivers. You'd need to use `modifyOpInPlace`.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:40`

```diff
@@ -0,0 +1,57 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/Common/Passes.h"
+#include "mlir/Transforms/WalkPatternRewriteDriver.h"
+
+#define DEBUG_TYPE "iree-codegen-strip-compilation-info"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_STRIPCOMPILATIONINFOPASS
+#include "iree/compiler/Codegen/Common/Passes.h.inc"
+
+namespace {
+
+struct StripCompilationInfo final
+    : OpInterfaceRewritePattern<mlir::FunctionOpInterface> {
+  using OpInterfaceRewritePattern<
+      mlir::FunctionOpInterface>::OpInterfaceRewritePattern;
+  LogicalResult matchAndRewrite(mlir::FunctionOpInterface funcOp,
+                                PatternRewriter &rewriter) const final {
+    rewriter.modifyOpInPlace(funcOp, [&]() {
+      // If the function has translation info, erase it.
+      if (getTranslationInfo(funcOp)) {
+        eraseTranslationInfo(funcOp);
+      }
+
+      funcOp->walk([&](Operation *op) {
+        if (getCompilationInfo(op)) {
+          // Erase the compilation info configuration if it exists
+          eraseCompilationInfo(op);
+        }
+        if (getLoweringConfig(op)) {
+          // Erase the lowering configuration from root operation if it
+          // exists.
+          eraseLoweringConfig(op);
+        }
```

**Comment:**
I think the nested walk makes this pass quadratic. Maybe we should have a separate pattern for these?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:22`

```diff
@@ -0,0 +1,57 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/Common/Passes.h"
+#include "mlir/Transforms/WalkPatternRewriteDriver.h"
+
+#define DEBUG_TYPE "iree-codegen-strip-compilation-info"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_STRIPCOMPILATIONINFOPASS
+#include "iree/compiler/Codegen/Common/Passes.h.inc"
+
+namespace {
+
+struct StripCompilationInfo final
+    : OpInterfaceRewritePattern<mlir::FunctionOpInterface> {
+  using OpInterfaceRewritePattern<
+      mlir::FunctionOpInterface>::OpInterfaceRewritePattern;
```

**Comment:**
```suggestion
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp:33`

```diff
@@ -0,0 +1,57 @@
+// Copyright 2024 The IREE Authors
+//
+// Licensed under the Apache License v2.0 with LLVM Exceptions.
+// See https://llvm.org/LICENSE.txt for license information.
+// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
+
+#include "iree/compiler/Codegen/Common/Passes.h"
+#include "mlir/Transforms/WalkPatternRewriteDriver.h"
+
+#define DEBUG_TYPE "iree-codegen-strip-compilation-info"
+
+namespace mlir::iree_compiler {
+
+#define GEN_PASS_DEF_STRIPCOMPILATIONINFOPASS
+#include "iree/compiler/Codegen/Common/Passes.h.inc"
+
+namespace {
+
+struct StripCompilationInfo final
+    : OpInterfaceRewritePattern<mlir::FunctionOpInterface> {
+  using OpInterfaceRewritePattern<
+      mlir::FunctionOpInterface>::OpInterfaceRewritePattern;
+  LogicalResult matchAndRewrite(mlir::FunctionOpInterface funcOp,
+                                PatternRewriter &rewriter) const final {
+    rewriter.modifyOpInPlace(funcOp, [&]() {
+      // If the function has translation info, erase it.
+      if (getTranslationInfo(funcOp)) {
+        eraseTranslationInfo(funcOp);
+      }
+
+      funcOp->walk([&](Operation *op) {
+        if (getCompilationInfo(op)) {
+          // Erase the compilation info configuration if it exists
```

**Comment:**
```suggestion
          // Erase the compilation info configuration if it exists.
```

---


---


## [PR #18952](https://github.com/iree-org/iree/pull/18952): [VectorDistribution] Add distribution pattern for vector::ContractionOp

### Review Summary

**COMMENTED** (2024-10-30)


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:573`

```diff
@@ -563,6 +563,223 @@ struct DistributeMultiReduction final
   int64_t maxBitsPerShuffle;
 };
 
+/// The lowering for Contract is performed in three steps (similar to above
+/// multi_reduction):
+///   1. Local Contract: Each thread performs operations on its locally
+///   distributed elements.
+///   2. Thread Reduction: Threads collectively reduce the results from step 1
+///   across threads,
+///      using a butterfly shuffle if distribution occurs along the reduction
+///      dimension.
```

**Comment:**
Butterfly shuffle is an implementation detail that's not observable from this pattern. I'd just say we perform subgroup reduction.

Also nit: the formatting is weird.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:571`

```diff
@@ -563,6 +563,223 @@ struct DistributeMultiReduction final
   int64_t maxBitsPerShuffle;
 };
 
+/// The lowering for Contract is performed in three steps (similar to above
+/// multi_reduction):
+///   1. Local Contract: Each thread performs operations on its locally
+///   distributed elements.
+///   2. Thread Reduction: Threads collectively reduce the results from step 1
+///   across threads,
```

**Comment:**
I'd call it something like: subgroup reduction or inter-thread reduction.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:576`

```diff
@@ -563,6 +563,223 @@ struct DistributeMultiReduction final
   int64_t maxBitsPerShuffle;
 };
 
+/// The lowering for Contract is performed in three steps (similar to above
+/// multi_reduction):
+///   1. Local Contract: Each thread performs operations on its locally
+///   distributed elements.
+///   2. Thread Reduction: Threads collectively reduce the results from step 1
+///   across threads,
+///      using a butterfly shuffle if distribution occurs along the reduction
+///      dimension.
+///   3. Accumulator Reduction: Each thread combines its intermediate results
+///   with its held accumulator.
+///
+/// Currently, reduction across multiple warps is not supported.
```

**Comment:**
let's try to stick to the portable naming like in the gpu dialect

```suggestion
/// Currently, reduction across multiple subgroups is not supported.
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:578`

```diff
@@ -563,6 +563,223 @@ struct DistributeMultiReduction final
   int64_t maxBitsPerShuffle;
 };
 
+/// The lowering for Contract is performed in three steps (similar to above
+/// multi_reduction):
+///   1. Local Contract: Each thread performs operations on its locally
+///   distributed elements.
+///   2. Thread Reduction: Threads collectively reduce the results from step 1
+///   across threads,
+///      using a butterfly shuffle if distribution occurs along the reduction
+///      dimension.
+///   3. Accumulator Reduction: Each thread combines its intermediate results
+///   with its held accumulator.
+///
+/// Currently, reduction across multiple warps is not supported.
+/// TODO: Add support for reductions across multiple warps.
```

**Comment:**
Can you describe why we'd need it?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:598`

```diff
@@ -563,6 +563,223 @@ struct DistributeMultiReduction final
   int64_t maxBitsPerShuffle;
 };
 
+/// The lowering for Contract is performed in three steps (similar to above
+/// multi_reduction):
+///   1. Local Contract: Each thread performs operations on its locally
+///   distributed elements.
+///   2. Thread Reduction: Threads collectively reduce the results from step 1
+///   across threads,
+///      using a butterfly shuffle if distribution occurs along the reduction
+///      dimension.
+///   3. Accumulator Reduction: Each thread combines its intermediate results
+///   with its held accumulator.
+///
+/// Currently, reduction across multiple warps is not supported.
+/// TODO: Add support for reductions across multiple warps.
+struct DistributeContract final : OpDistributionPattern<vector::ContractionOp> {
+  using OpDistributionPattern::OpDistributionPattern;
+
+  DistributeContract(MLIRContext *context, int64_t subgroupSize,
+                     int64_t maxBitsPerShuffle, int64_t benefit = 1)
+      : OpDistributionPattern(context, benefit), subgroupSize(subgroupSize),
+        maxBitsPerShuffle(maxBitsPerShuffle) {}
+
+  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
+                                DistributionSignature &signature,
+                                PatternRewriter &rewriter) const override {
+    FailureOr<VectorContractOpInfo> maybeOpInfo =
+        VectorContractOpInfo::inferFromIndexingMaps(
+            contractOp.getIndexingMapsArray());
+    if (failed(maybeOpInfo)) {
+      return rewriter.notifyMatchFailure(contractOp, "not a contraction");
+    }
+    // If mmaAttr exists, defer the lowering to use MMA.
+    // Notify failure if the "iree.amdgpu.mma" intrinsic attribute is present.
+    IREE::GPU::MMAAttr mmaAttr =
```

**Comment:**
```suggestion
    auto mmaAttr =
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:605`

```diff
@@ -563,6 +563,223 @@ struct DistributeMultiReduction final
   int64_t maxBitsPerShuffle;
 };
 
+/// The lowering for Contract is performed in three steps (similar to above
+/// multi_reduction):
+///   1. Local Contract: Each thread performs operations on its locally
+///   distributed elements.
+///   2. Thread Reduction: Threads collectively reduce the results from step 1
+///   across threads,
+///      using a butterfly shuffle if distribution occurs along the reduction
+///      dimension.
+///   3. Accumulator Reduction: Each thread combines its intermediate results
+///   with its held accumulator.
+///
+/// Currently, reduction across multiple warps is not supported.
+/// TODO: Add support for reductions across multiple warps.
+struct DistributeContract final : OpDistributionPattern<vector::ContractionOp> {
+  using OpDistributionPattern::OpDistributionPattern;
+
+  DistributeContract(MLIRContext *context, int64_t subgroupSize,
+                     int64_t maxBitsPerShuffle, int64_t benefit = 1)
+      : OpDistributionPattern(context, benefit), subgroupSize(subgroupSize),
+        maxBitsPerShuffle(maxBitsPerShuffle) {}
+
+  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
+                                DistributionSignature &signature,
+                                PatternRewriter &rewriter) const override {
+    FailureOr<VectorContractOpInfo> maybeOpInfo =
+        VectorContractOpInfo::inferFromIndexingMaps(
+            contractOp.getIndexingMapsArray());
+    if (failed(maybeOpInfo)) {
+      return rewriter.notifyMatchFailure(contractOp, "not a contraction");
+    }
+    // If mmaAttr exists, defer the lowering to use MMA.
+    // Notify failure if the "iree.amdgpu.mma" intrinsic attribute is present.
+    IREE::GPU::MMAAttr mmaAttr =
+        contractOp->getAttrOfType<IREE::GPU::MMAAttr>("iree.amdgpu.mma");
+    if (mmaAttr) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "iree.amdgpu.mma intrinsic attribute exists");
+    }
+
+    NestedLayoutAttr lhsLayout =
```

**Comment:**
```suggestion
    auto lhsLayout =
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:611`

```diff
@@ -563,6 +563,223 @@ struct DistributeMultiReduction final
   int64_t maxBitsPerShuffle;
 };
 
+/// The lowering for Contract is performed in three steps (similar to above
+/// multi_reduction):
+///   1. Local Contract: Each thread performs operations on its locally
+///   distributed elements.
+///   2. Thread Reduction: Threads collectively reduce the results from step 1
+///   across threads,
+///      using a butterfly shuffle if distribution occurs along the reduction
+///      dimension.
+///   3. Accumulator Reduction: Each thread combines its intermediate results
+///   with its held accumulator.
+///
+/// Currently, reduction across multiple warps is not supported.
+/// TODO: Add support for reductions across multiple warps.
+struct DistributeContract final : OpDistributionPattern<vector::ContractionOp> {
+  using OpDistributionPattern::OpDistributionPattern;
+
+  DistributeContract(MLIRContext *context, int64_t subgroupSize,
+                     int64_t maxBitsPerShuffle, int64_t benefit = 1)
+      : OpDistributionPattern(context, benefit), subgroupSize(subgroupSize),
+        maxBitsPerShuffle(maxBitsPerShuffle) {}
+
+  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
+                                DistributionSignature &signature,
+                                PatternRewriter &rewriter) const override {
+    FailureOr<VectorContractOpInfo> maybeOpInfo =
+        VectorContractOpInfo::inferFromIndexingMaps(
+            contractOp.getIndexingMapsArray());
+    if (failed(maybeOpInfo)) {
+      return rewriter.notifyMatchFailure(contractOp, "not a contraction");
+    }
+    // If mmaAttr exists, defer the lowering to use MMA.
+    // Notify failure if the "iree.amdgpu.mma" intrinsic attribute is present.
+    IREE::GPU::MMAAttr mmaAttr =
+        contractOp->getAttrOfType<IREE::GPU::MMAAttr>("iree.amdgpu.mma");
+    if (mmaAttr) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "iree.amdgpu.mma intrinsic attribute exists");
+    }
+
+    NestedLayoutAttr lhsLayout =
+        dyn_cast<NestedLayoutAttr>(signature[contractOp.getLhs()]);
+    if (!lhsLayout) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "missing nested layout for contraction lhs");
+    }
+    NestedLayoutAttr rhsLayout =
```

**Comment:**
```suggestion
    auto rhsLayout =
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:624`

```diff
@@ -563,6 +563,223 @@ struct DistributeMultiReduction final
   int64_t maxBitsPerShuffle;
 };
 
+/// The lowering for Contract is performed in three steps (similar to above
+/// multi_reduction):
+///   1. Local Contract: Each thread performs operations on its locally
+///   distributed elements.
+///   2. Thread Reduction: Threads collectively reduce the results from step 1
+///   across threads,
+///      using a butterfly shuffle if distribution occurs along the reduction
+///      dimension.
+///   3. Accumulator Reduction: Each thread combines its intermediate results
+///   with its held accumulator.
+///
+/// Currently, reduction across multiple warps is not supported.
+/// TODO: Add support for reductions across multiple warps.
+struct DistributeContract final : OpDistributionPattern<vector::ContractionOp> {
+  using OpDistributionPattern::OpDistributionPattern;
+
+  DistributeContract(MLIRContext *context, int64_t subgroupSize,
+                     int64_t maxBitsPerShuffle, int64_t benefit = 1)
+      : OpDistributionPattern(context, benefit), subgroupSize(subgroupSize),
+        maxBitsPerShuffle(maxBitsPerShuffle) {}
+
+  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
+                                DistributionSignature &signature,
+                                PatternRewriter &rewriter) const override {
+    FailureOr<VectorContractOpInfo> maybeOpInfo =
+        VectorContractOpInfo::inferFromIndexingMaps(
+            contractOp.getIndexingMapsArray());
+    if (failed(maybeOpInfo)) {
+      return rewriter.notifyMatchFailure(contractOp, "not a contraction");
+    }
+    // If mmaAttr exists, defer the lowering to use MMA.
+    // Notify failure if the "iree.amdgpu.mma" intrinsic attribute is present.
+    IREE::GPU::MMAAttr mmaAttr =
+        contractOp->getAttrOfType<IREE::GPU::MMAAttr>("iree.amdgpu.mma");
+    if (mmaAttr) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "iree.amdgpu.mma intrinsic attribute exists");
+    }
+
+    NestedLayoutAttr lhsLayout =
+        dyn_cast<NestedLayoutAttr>(signature[contractOp.getLhs()]);
+    if (!lhsLayout) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "missing nested layout for contraction lhs");
+    }
+    NestedLayoutAttr rhsLayout =
+        dyn_cast<NestedLayoutAttr>(signature[contractOp.getRhs()]);
+    if (!rhsLayout) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "missing nested layout for contraction rhs");
+    }
+
+    Value disLhs = getDistributed(rewriter, contractOp.getLhs(), lhsLayout);
+    Value disRhs = getDistributed(rewriter, contractOp.getRhs(), rhsLayout);
+
+    Value acc = contractOp.getAcc();
+    Value res = contractOp.getResult();
+    VectorValue accVector = dyn_cast<VectorValue>(acc);
+    VectorValue resVector = dyn_cast<VectorValue>(res);
```

**Comment:**
```suggestion
    auto accVector = dyn_cast<VectorValue>(acc);
    auto resVector = dyn_cast<VectorValue>(res);
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:636`

```diff
@@ -563,6 +563,223 @@ struct DistributeMultiReduction final
   int64_t maxBitsPerShuffle;
 };
 
+/// The lowering for Contract is performed in three steps (similar to above
+/// multi_reduction):
+///   1. Local Contract: Each thread performs operations on its locally
+///   distributed elements.
+///   2. Thread Reduction: Threads collectively reduce the results from step 1
+///   across threads,
+///      using a butterfly shuffle if distribution occurs along the reduction
+///      dimension.
+///   3. Accumulator Reduction: Each thread combines its intermediate results
+///   with its held accumulator.
+///
+/// Currently, reduction across multiple warps is not supported.
+/// TODO: Add support for reductions across multiple warps.
+struct DistributeContract final : OpDistributionPattern<vector::ContractionOp> {
+  using OpDistributionPattern::OpDistributionPattern;
+
+  DistributeContract(MLIRContext *context, int64_t subgroupSize,
+                     int64_t maxBitsPerShuffle, int64_t benefit = 1)
+      : OpDistributionPattern(context, benefit), subgroupSize(subgroupSize),
+        maxBitsPerShuffle(maxBitsPerShuffle) {}
+
+  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
+                                DistributionSignature &signature,
+                                PatternRewriter &rewriter) const override {
+    FailureOr<VectorContractOpInfo> maybeOpInfo =
+        VectorContractOpInfo::inferFromIndexingMaps(
+            contractOp.getIndexingMapsArray());
+    if (failed(maybeOpInfo)) {
+      return rewriter.notifyMatchFailure(contractOp, "not a contraction");
+    }
+    // If mmaAttr exists, defer the lowering to use MMA.
+    // Notify failure if the "iree.amdgpu.mma" intrinsic attribute is present.
+    IREE::GPU::MMAAttr mmaAttr =
+        contractOp->getAttrOfType<IREE::GPU::MMAAttr>("iree.amdgpu.mma");
+    if (mmaAttr) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "iree.amdgpu.mma intrinsic attribute exists");
+    }
+
+    NestedLayoutAttr lhsLayout =
+        dyn_cast<NestedLayoutAttr>(signature[contractOp.getLhs()]);
+    if (!lhsLayout) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "missing nested layout for contraction lhs");
+    }
+    NestedLayoutAttr rhsLayout =
+        dyn_cast<NestedLayoutAttr>(signature[contractOp.getRhs()]);
+    if (!rhsLayout) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "missing nested layout for contraction rhs");
+    }
+
+    Value disLhs = getDistributed(rewriter, contractOp.getLhs(), lhsLayout);
+    Value disRhs = getDistributed(rewriter, contractOp.getRhs(), rhsLayout);
+
+    Value acc = contractOp.getAcc();
+    Value res = contractOp.getResult();
+    VectorValue accVector = dyn_cast<VectorValue>(acc);
+    VectorValue resVector = dyn_cast<VectorValue>(res);
+    Value disAcc;
+    if (accVector) {
+      disAcc = getDistributed(rewriter, accVector, signature[accVector]);
+    } else {
+      disAcc = contractOp.getAcc();
+    }
+
+    Location loc = contractOp.getLoc();
+    int64_t rank = lhsLayout.getRank();
+
+    SmallVector<bool> reducedDims(rank, false);
+    auto maps = contractOp.getIndexingMapsArray();
```

**Comment:**
Use the actual type since it's not obvious based on the RHS only

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:639`

```diff
@@ -563,6 +563,223 @@ struct DistributeMultiReduction final
   int64_t maxBitsPerShuffle;
 };
 
+/// The lowering for Contract is performed in three steps (similar to above
+/// multi_reduction):
+///   1. Local Contract: Each thread performs operations on its locally
+///   distributed elements.
+///   2. Thread Reduction: Threads collectively reduce the results from step 1
+///   across threads,
+///      using a butterfly shuffle if distribution occurs along the reduction
+///      dimension.
+///   3. Accumulator Reduction: Each thread combines its intermediate results
+///   with its held accumulator.
+///
+/// Currently, reduction across multiple warps is not supported.
+/// TODO: Add support for reductions across multiple warps.
+struct DistributeContract final : OpDistributionPattern<vector::ContractionOp> {
+  using OpDistributionPattern::OpDistributionPattern;
+
+  DistributeContract(MLIRContext *context, int64_t subgroupSize,
+                     int64_t maxBitsPerShuffle, int64_t benefit = 1)
+      : OpDistributionPattern(context, benefit), subgroupSize(subgroupSize),
+        maxBitsPerShuffle(maxBitsPerShuffle) {}
+
+  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
+                                DistributionSignature &signature,
+                                PatternRewriter &rewriter) const override {
+    FailureOr<VectorContractOpInfo> maybeOpInfo =
+        VectorContractOpInfo::inferFromIndexingMaps(
+            contractOp.getIndexingMapsArray());
+    if (failed(maybeOpInfo)) {
+      return rewriter.notifyMatchFailure(contractOp, "not a contraction");
+    }
+    // If mmaAttr exists, defer the lowering to use MMA.
+    // Notify failure if the "iree.amdgpu.mma" intrinsic attribute is present.
+    IREE::GPU::MMAAttr mmaAttr =
+        contractOp->getAttrOfType<IREE::GPU::MMAAttr>("iree.amdgpu.mma");
+    if (mmaAttr) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "iree.amdgpu.mma intrinsic attribute exists");
+    }
+
+    NestedLayoutAttr lhsLayout =
+        dyn_cast<NestedLayoutAttr>(signature[contractOp.getLhs()]);
+    if (!lhsLayout) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "missing nested layout for contraction lhs");
+    }
+    NestedLayoutAttr rhsLayout =
+        dyn_cast<NestedLayoutAttr>(signature[contractOp.getRhs()]);
+    if (!rhsLayout) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "missing nested layout for contraction rhs");
+    }
+
+    Value disLhs = getDistributed(rewriter, contractOp.getLhs(), lhsLayout);
+    Value disRhs = getDistributed(rewriter, contractOp.getRhs(), rhsLayout);
+
+    Value acc = contractOp.getAcc();
+    Value res = contractOp.getResult();
+    VectorValue accVector = dyn_cast<VectorValue>(acc);
+    VectorValue resVector = dyn_cast<VectorValue>(res);
+    Value disAcc;
+    if (accVector) {
+      disAcc = getDistributed(rewriter, accVector, signature[accVector]);
+    } else {
+      disAcc = contractOp.getAcc();
+    }
+
+    Location loc = contractOp.getLoc();
+    int64_t rank = lhsLayout.getRank();
+
+    SmallVector<bool> reducedDims(rank, false);
+    auto maps = contractOp.getIndexingMapsArray();
+
+    // Identify the reduction dimension and apply it for cross-thread reduction.
+    MLIRContext *ctx = maps[0].getContext();
```

**Comment:**
I'd put it just before loc and extract from `contractOp` instead.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:657`

```diff
@@ -563,6 +563,223 @@ struct DistributeMultiReduction final
   int64_t maxBitsPerShuffle;
 };
 
+/// The lowering for Contract is performed in three steps (similar to above
+/// multi_reduction):
+///   1. Local Contract: Each thread performs operations on its locally
+///   distributed elements.
+///   2. Thread Reduction: Threads collectively reduce the results from step 1
+///   across threads,
+///      using a butterfly shuffle if distribution occurs along the reduction
+///      dimension.
+///   3. Accumulator Reduction: Each thread combines its intermediate results
+///   with its held accumulator.
+///
+/// Currently, reduction across multiple warps is not supported.
+/// TODO: Add support for reductions across multiple warps.
+struct DistributeContract final : OpDistributionPattern<vector::ContractionOp> {
+  using OpDistributionPattern::OpDistributionPattern;
+
+  DistributeContract(MLIRContext *context, int64_t subgroupSize,
+                     int64_t maxBitsPerShuffle, int64_t benefit = 1)
+      : OpDistributionPattern(context, benefit), subgroupSize(subgroupSize),
+        maxBitsPerShuffle(maxBitsPerShuffle) {}
+
+  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
+                                DistributionSignature &signature,
+                                PatternRewriter &rewriter) const override {
+    FailureOr<VectorContractOpInfo> maybeOpInfo =
+        VectorContractOpInfo::inferFromIndexingMaps(
+            contractOp.getIndexingMapsArray());
+    if (failed(maybeOpInfo)) {
+      return rewriter.notifyMatchFailure(contractOp, "not a contraction");
+    }
+    // If mmaAttr exists, defer the lowering to use MMA.
+    // Notify failure if the "iree.amdgpu.mma" intrinsic attribute is present.
+    IREE::GPU::MMAAttr mmaAttr =
+        contractOp->getAttrOfType<IREE::GPU::MMAAttr>("iree.amdgpu.mma");
+    if (mmaAttr) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "iree.amdgpu.mma intrinsic attribute exists");
+    }
+
+    NestedLayoutAttr lhsLayout =
+        dyn_cast<NestedLayoutAttr>(signature[contractOp.getLhs()]);
+    if (!lhsLayout) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "missing nested layout for contraction lhs");
+    }
+    NestedLayoutAttr rhsLayout =
+        dyn_cast<NestedLayoutAttr>(signature[contractOp.getRhs()]);
+    if (!rhsLayout) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "missing nested layout for contraction rhs");
+    }
+
+    Value disLhs = getDistributed(rewriter, contractOp.getLhs(), lhsLayout);
+    Value disRhs = getDistributed(rewriter, contractOp.getRhs(), rhsLayout);
+
+    Value acc = contractOp.getAcc();
+    Value res = contractOp.getResult();
+    VectorValue accVector = dyn_cast<VectorValue>(acc);
+    VectorValue resVector = dyn_cast<VectorValue>(res);
+    Value disAcc;
+    if (accVector) {
+      disAcc = getDistributed(rewriter, accVector, signature[accVector]);
+    } else {
+      disAcc = contractOp.getAcc();
+    }
+
+    Location loc = contractOp.getLoc();
+    int64_t rank = lhsLayout.getRank();
+
+    SmallVector<bool> reducedDims(rank, false);
+    auto maps = contractOp.getIndexingMapsArray();
+
+    // Identify the reduction dimension and apply it for cross-thread reduction.
+    MLIRContext *ctx = maps[0].getContext();
+    for (auto [index, iteratorType] :
+         llvm::enumerate(contractOp.getIteratorTypes())) {
+      if (vector::isReductionIterator(iteratorType)) {
+        int64_t redIdx =
+            *maps[0].getResultPosition(getAffineDimExpr(index, ctx));
+        reducedDims[redIdx] = true;
+      }
+    }
+
+    ArrayRef<Attribute> iteratorTypes =
+        contractOp.getIteratorTypes().getValue();
+    SmallVector<Attribute> newIterators;
+
+    // Given that the distribution format is <BATCH x OUTER x ELEMENT>,
+    // the iterations and affine maps need to be replicated three times.
+
+    // Replicate the iterators for local vector.contract
+    for (int i = 0; i < 3; i++) {
```

**Comment:**
https://llvm.org/docs/CodingStandards.html#prefer-preincrement

```suggestion
    for (int i = 0; i < 3; ++i) {
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:663`

```diff
@@ -563,6 +563,223 @@ struct DistributeMultiReduction final
   int64_t maxBitsPerShuffle;
 };
 
+/// The lowering for Contract is performed in three steps (similar to above
+/// multi_reduction):
+///   1. Local Contract: Each thread performs operations on its locally
+///   distributed elements.
+///   2. Thread Reduction: Threads collectively reduce the results from step 1
+///   across threads,
+///      using a butterfly shuffle if distribution occurs along the reduction
+///      dimension.
+///   3. Accumulator Reduction: Each thread combines its intermediate results
+///   with its held accumulator.
+///
+/// Currently, reduction across multiple warps is not supported.
+/// TODO: Add support for reductions across multiple warps.
+struct DistributeContract final : OpDistributionPattern<vector::ContractionOp> {
+  using OpDistributionPattern::OpDistributionPattern;
+
+  DistributeContract(MLIRContext *context, int64_t subgroupSize,
+                     int64_t maxBitsPerShuffle, int64_t benefit = 1)
+      : OpDistributionPattern(context, benefit), subgroupSize(subgroupSize),
+        maxBitsPerShuffle(maxBitsPerShuffle) {}
+
+  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
+                                DistributionSignature &signature,
+                                PatternRewriter &rewriter) const override {
+    FailureOr<VectorContractOpInfo> maybeOpInfo =
+        VectorContractOpInfo::inferFromIndexingMaps(
+            contractOp.getIndexingMapsArray());
+    if (failed(maybeOpInfo)) {
+      return rewriter.notifyMatchFailure(contractOp, "not a contraction");
+    }
+    // If mmaAttr exists, defer the lowering to use MMA.
+    // Notify failure if the "iree.amdgpu.mma" intrinsic attribute is present.
+    IREE::GPU::MMAAttr mmaAttr =
+        contractOp->getAttrOfType<IREE::GPU::MMAAttr>("iree.amdgpu.mma");
+    if (mmaAttr) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "iree.amdgpu.mma intrinsic attribute exists");
+    }
+
+    NestedLayoutAttr lhsLayout =
+        dyn_cast<NestedLayoutAttr>(signature[contractOp.getLhs()]);
+    if (!lhsLayout) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "missing nested layout for contraction lhs");
+    }
+    NestedLayoutAttr rhsLayout =
+        dyn_cast<NestedLayoutAttr>(signature[contractOp.getRhs()]);
+    if (!rhsLayout) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "missing nested layout for contraction rhs");
+    }
+
+    Value disLhs = getDistributed(rewriter, contractOp.getLhs(), lhsLayout);
+    Value disRhs = getDistributed(rewriter, contractOp.getRhs(), rhsLayout);
+
+    Value acc = contractOp.getAcc();
+    Value res = contractOp.getResult();
+    VectorValue accVector = dyn_cast<VectorValue>(acc);
+    VectorValue resVector = dyn_cast<VectorValue>(res);
+    Value disAcc;
+    if (accVector) {
+      disAcc = getDistributed(rewriter, accVector, signature[accVector]);
+    } else {
+      disAcc = contractOp.getAcc();
+    }
+
+    Location loc = contractOp.getLoc();
+    int64_t rank = lhsLayout.getRank();
+
+    SmallVector<bool> reducedDims(rank, false);
+    auto maps = contractOp.getIndexingMapsArray();
+
+    // Identify the reduction dimension and apply it for cross-thread reduction.
+    MLIRContext *ctx = maps[0].getContext();
+    for (auto [index, iteratorType] :
+         llvm::enumerate(contractOp.getIteratorTypes())) {
+      if (vector::isReductionIterator(iteratorType)) {
+        int64_t redIdx =
+            *maps[0].getResultPosition(getAffineDimExpr(index, ctx));
+        reducedDims[redIdx] = true;
+      }
+    }
+
+    ArrayRef<Attribute> iteratorTypes =
+        contractOp.getIteratorTypes().getValue();
+    SmallVector<Attribute> newIterators;
+
+    // Given that the distribution format is <BATCH x OUTER x ELEMENT>,
+    // the iterations and affine maps need to be replicated three times.
+
+    // Replicate the iterators for local vector.contract
+    for (int i = 0; i < 3; i++) {
+      newIterators.append(iteratorTypes.begin(), iteratorTypes.end());
+    }
+
+    // Replicate the affine maps for local vector.contract
+    SmallVector<AffineMap> newMaps;
+    for (auto map : maps) {
```

**Comment:**
Use the actual type since it's not obvious based on the RHS only

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:667`

```diff
@@ -563,6 +563,223 @@ struct DistributeMultiReduction final
   int64_t maxBitsPerShuffle;
 };
 
+/// The lowering for Contract is performed in three steps (similar to above
+/// multi_reduction):
+///   1. Local Contract: Each thread performs operations on its locally
+///   distributed elements.
+///   2. Thread Reduction: Threads collectively reduce the results from step 1
+///   across threads,
+///      using a butterfly shuffle if distribution occurs along the reduction
+///      dimension.
+///   3. Accumulator Reduction: Each thread combines its intermediate results
+///   with its held accumulator.
+///
+/// Currently, reduction across multiple warps is not supported.
+/// TODO: Add support for reductions across multiple warps.
+struct DistributeContract final : OpDistributionPattern<vector::ContractionOp> {
+  using OpDistributionPattern::OpDistributionPattern;
+
+  DistributeContract(MLIRContext *context, int64_t subgroupSize,
+                     int64_t maxBitsPerShuffle, int64_t benefit = 1)
+      : OpDistributionPattern(context, benefit), subgroupSize(subgroupSize),
+        maxBitsPerShuffle(maxBitsPerShuffle) {}
+
+  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
+                                DistributionSignature &signature,
+                                PatternRewriter &rewriter) const override {
+    FailureOr<VectorContractOpInfo> maybeOpInfo =
+        VectorContractOpInfo::inferFromIndexingMaps(
+            contractOp.getIndexingMapsArray());
+    if (failed(maybeOpInfo)) {
+      return rewriter.notifyMatchFailure(contractOp, "not a contraction");
+    }
+    // If mmaAttr exists, defer the lowering to use MMA.
+    // Notify failure if the "iree.amdgpu.mma" intrinsic attribute is present.
+    IREE::GPU::MMAAttr mmaAttr =
+        contractOp->getAttrOfType<IREE::GPU::MMAAttr>("iree.amdgpu.mma");
+    if (mmaAttr) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "iree.amdgpu.mma intrinsic attribute exists");
+    }
+
+    NestedLayoutAttr lhsLayout =
+        dyn_cast<NestedLayoutAttr>(signature[contractOp.getLhs()]);
+    if (!lhsLayout) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "missing nested layout for contraction lhs");
+    }
+    NestedLayoutAttr rhsLayout =
+        dyn_cast<NestedLayoutAttr>(signature[contractOp.getRhs()]);
+    if (!rhsLayout) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "missing nested layout for contraction rhs");
+    }
+
+    Value disLhs = getDistributed(rewriter, contractOp.getLhs(), lhsLayout);
+    Value disRhs = getDistributed(rewriter, contractOp.getRhs(), rhsLayout);
+
+    Value acc = contractOp.getAcc();
+    Value res = contractOp.getResult();
+    VectorValue accVector = dyn_cast<VectorValue>(acc);
+    VectorValue resVector = dyn_cast<VectorValue>(res);
+    Value disAcc;
+    if (accVector) {
+      disAcc = getDistributed(rewriter, accVector, signature[accVector]);
+    } else {
+      disAcc = contractOp.getAcc();
+    }
+
+    Location loc = contractOp.getLoc();
+    int64_t rank = lhsLayout.getRank();
+
+    SmallVector<bool> reducedDims(rank, false);
+    auto maps = contractOp.getIndexingMapsArray();
+
+    // Identify the reduction dimension and apply it for cross-thread reduction.
+    MLIRContext *ctx = maps[0].getContext();
+    for (auto [index, iteratorType] :
+         llvm::enumerate(contractOp.getIteratorTypes())) {
+      if (vector::isReductionIterator(iteratorType)) {
+        int64_t redIdx =
+            *maps[0].getResultPosition(getAffineDimExpr(index, ctx));
+        reducedDims[redIdx] = true;
+      }
+    }
+
+    ArrayRef<Attribute> iteratorTypes =
+        contractOp.getIteratorTypes().getValue();
+    SmallVector<Attribute> newIterators;
+
+    // Given that the distribution format is <BATCH x OUTER x ELEMENT>,
+    // the iterations and affine maps need to be replicated three times.
+
+    // Replicate the iterators for local vector.contract
+    for (int i = 0; i < 3; i++) {
+      newIterators.append(iteratorTypes.begin(), iteratorTypes.end());
+    }
+
+    // Replicate the affine maps for local vector.contract
+    SmallVector<AffineMap> newMaps;
+    for (auto map : maps) {
+      int64_t numDims = map.getNumDims();
+      int64_t numResults = map.getNumResults();
+      SmallVector<AffineExpr> exprs;
+      for (int i = 0; i < 3; i++) {
```

**Comment:**
also here

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:669`

```diff
@@ -563,6 +563,223 @@ struct DistributeMultiReduction final
   int64_t maxBitsPerShuffle;
 };
 
+/// The lowering for Contract is performed in three steps (similar to above
+/// multi_reduction):
+///   1. Local Contract: Each thread performs operations on its locally
+///   distributed elements.
+///   2. Thread Reduction: Threads collectively reduce the results from step 1
+///   across threads,
+///      using a butterfly shuffle if distribution occurs along the reduction
+///      dimension.
+///   3. Accumulator Reduction: Each thread combines its intermediate results
+///   with its held accumulator.
+///
+/// Currently, reduction across multiple warps is not supported.
+/// TODO: Add support for reductions across multiple warps.
+struct DistributeContract final : OpDistributionPattern<vector::ContractionOp> {
+  using OpDistributionPattern::OpDistributionPattern;
+
+  DistributeContract(MLIRContext *context, int64_t subgroupSize,
+                     int64_t maxBitsPerShuffle, int64_t benefit = 1)
+      : OpDistributionPattern(context, benefit), subgroupSize(subgroupSize),
+        maxBitsPerShuffle(maxBitsPerShuffle) {}
+
+  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
+                                DistributionSignature &signature,
+                                PatternRewriter &rewriter) const override {
+    FailureOr<VectorContractOpInfo> maybeOpInfo =
+        VectorContractOpInfo::inferFromIndexingMaps(
+            contractOp.getIndexingMapsArray());
+    if (failed(maybeOpInfo)) {
+      return rewriter.notifyMatchFailure(contractOp, "not a contraction");
+    }
+    // If mmaAttr exists, defer the lowering to use MMA.
+    // Notify failure if the "iree.amdgpu.mma" intrinsic attribute is present.
+    IREE::GPU::MMAAttr mmaAttr =
+        contractOp->getAttrOfType<IREE::GPU::MMAAttr>("iree.amdgpu.mma");
+    if (mmaAttr) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "iree.amdgpu.mma intrinsic attribute exists");
+    }
+
+    NestedLayoutAttr lhsLayout =
+        dyn_cast<NestedLayoutAttr>(signature[contractOp.getLhs()]);
+    if (!lhsLayout) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "missing nested layout for contraction lhs");
+    }
+    NestedLayoutAttr rhsLayout =
+        dyn_cast<NestedLayoutAttr>(signature[contractOp.getRhs()]);
+    if (!rhsLayout) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "missing nested layout for contraction rhs");
+    }
+
+    Value disLhs = getDistributed(rewriter, contractOp.getLhs(), lhsLayout);
+    Value disRhs = getDistributed(rewriter, contractOp.getRhs(), rhsLayout);
+
+    Value acc = contractOp.getAcc();
+    Value res = contractOp.getResult();
+    VectorValue accVector = dyn_cast<VectorValue>(acc);
+    VectorValue resVector = dyn_cast<VectorValue>(res);
+    Value disAcc;
+    if (accVector) {
+      disAcc = getDistributed(rewriter, accVector, signature[accVector]);
+    } else {
+      disAcc = contractOp.getAcc();
+    }
+
+    Location loc = contractOp.getLoc();
+    int64_t rank = lhsLayout.getRank();
+
+    SmallVector<bool> reducedDims(rank, false);
+    auto maps = contractOp.getIndexingMapsArray();
+
+    // Identify the reduction dimension and apply it for cross-thread reduction.
+    MLIRContext *ctx = maps[0].getContext();
+    for (auto [index, iteratorType] :
+         llvm::enumerate(contractOp.getIteratorTypes())) {
+      if (vector::isReductionIterator(iteratorType)) {
+        int64_t redIdx =
+            *maps[0].getResultPosition(getAffineDimExpr(index, ctx));
+        reducedDims[redIdx] = true;
+      }
+    }
+
+    ArrayRef<Attribute> iteratorTypes =
+        contractOp.getIteratorTypes().getValue();
+    SmallVector<Attribute> newIterators;
+
+    // Given that the distribution format is <BATCH x OUTER x ELEMENT>,
+    // the iterations and affine maps need to be replicated three times.
+
+    // Replicate the iterators for local vector.contract
+    for (int i = 0; i < 3; i++) {
+      newIterators.append(iteratorTypes.begin(), iteratorTypes.end());
+    }
+
+    // Replicate the affine maps for local vector.contract
+    SmallVector<AffineMap> newMaps;
+    for (auto map : maps) {
+      int64_t numDims = map.getNumDims();
+      int64_t numResults = map.getNumResults();
+      SmallVector<AffineExpr> exprs;
+      for (int i = 0; i < 3; i++) {
+        AffineMap shiftedMap = map.shiftDims(i * numDims);
+        for (int j = 0; j < numResults; j++) {
```

**Comment:**
also here

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:701`

```diff
@@ -563,6 +563,223 @@ struct DistributeMultiReduction final
   int64_t maxBitsPerShuffle;
 };
 
+/// The lowering for Contract is performed in three steps (similar to above
+/// multi_reduction):
+///   1. Local Contract: Each thread performs operations on its locally
+///   distributed elements.
+///   2. Thread Reduction: Threads collectively reduce the results from step 1
+///   across threads,
+///      using a butterfly shuffle if distribution occurs along the reduction
+///      dimension.
+///   3. Accumulator Reduction: Each thread combines its intermediate results
+///   with its held accumulator.
+///
+/// Currently, reduction across multiple warps is not supported.
+/// TODO: Add support for reductions across multiple warps.
+struct DistributeContract final : OpDistributionPattern<vector::ContractionOp> {
+  using OpDistributionPattern::OpDistributionPattern;
+
+  DistributeContract(MLIRContext *context, int64_t subgroupSize,
+                     int64_t maxBitsPerShuffle, int64_t benefit = 1)
+      : OpDistributionPattern(context, benefit), subgroupSize(subgroupSize),
+        maxBitsPerShuffle(maxBitsPerShuffle) {}
+
+  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
+                                DistributionSignature &signature,
+                                PatternRewriter &rewriter) const override {
+    FailureOr<VectorContractOpInfo> maybeOpInfo =
+        VectorContractOpInfo::inferFromIndexingMaps(
+            contractOp.getIndexingMapsArray());
+    if (failed(maybeOpInfo)) {
+      return rewriter.notifyMatchFailure(contractOp, "not a contraction");
+    }
+    // If mmaAttr exists, defer the lowering to use MMA.
+    // Notify failure if the "iree.amdgpu.mma" intrinsic attribute is present.
+    IREE::GPU::MMAAttr mmaAttr =
+        contractOp->getAttrOfType<IREE::GPU::MMAAttr>("iree.amdgpu.mma");
+    if (mmaAttr) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "iree.amdgpu.mma intrinsic attribute exists");
+    }
+
+    NestedLayoutAttr lhsLayout =
+        dyn_cast<NestedLayoutAttr>(signature[contractOp.getLhs()]);
+    if (!lhsLayout) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "missing nested layout for contraction lhs");
+    }
+    NestedLayoutAttr rhsLayout =
+        dyn_cast<NestedLayoutAttr>(signature[contractOp.getRhs()]);
+    if (!rhsLayout) {
+      return rewriter.notifyMatchFailure(
+          contractOp, "missing nested layout for contraction rhs");
+    }
+
+    Value disLhs = getDistributed(rewriter, contractOp.getLhs(), lhsLayout);
+    Value disRhs = getDistributed(rewriter, contractOp.getRhs(), rhsLayout);
+
+    Value acc = contractOp.getAcc();
+    Value res = contractOp.getResult();
+    VectorValue accVector = dyn_cast<VectorValue>(acc);
+    VectorValue resVector = dyn_cast<VectorValue>(res);
+    Value disAcc;
+    if (accVector) {
+      disAcc = getDistributed(rewriter, accVector, signature[accVector]);
+    } else {
+      disAcc = contractOp.getAcc();
+    }
+
+    Location loc = contractOp.getLoc();
+    int64_t rank = lhsLayout.getRank();
+
+    SmallVector<bool> reducedDims(rank, false);
+    auto maps = contractOp.getIndexingMapsArray();
+
+    // Identify the reduction dimension and apply it for cross-thread reduction.
+    MLIRContext *ctx = maps[0].getContext();
+    for (auto [index, iteratorType] :
+         llvm::enumerate(contractOp.getIteratorTypes())) {
+      if (vector::isReductionIterator(iteratorType)) {
+        int64_t redIdx =
+            *maps[0].getResultPosition(getAffineDimExpr(index, ctx));
+        reducedDims[redIdx] = true;
+      }
+    }
+
+    ArrayRef<Attribute> iteratorTypes =
+        contractOp.getIteratorTypes().getValue();
+    SmallVector<Attribute> newIterators;
+
+    // Given that the distribution format is <BATCH x OUTER x ELEMENT>,
+    // the iterations and affine maps need to be replicated three times.
+
+    // Replicate the iterators for local vector.contract
+    for (int i = 0; i < 3; i++) {
+      newIterators.append(iteratorTypes.begin(), iteratorTypes.end());
+    }
+
+    // Replicate the affine maps for local vector.contract
+    SmallVector<AffineMap> newMaps;
+    for (auto map : maps) {
+      int64_t numDims = map.getNumDims();
+      int64_t numResults = map.getNumResults();
+      SmallVector<AffineExpr> exprs;
+      for (int i = 0; i < 3; i++) {
+        AffineMap shiftedMap = map.shiftDims(i * numDims);
+        for (int j = 0; j < numResults; j++) {
+          exprs.push_back(shiftedMap.getResult(j));
+        }
+      }
+      AffineMap newMap =
+          AffineMap::get(/*dimCount=*/3 * numDims,
+                         /*symbolCount=*/map.getNumSymbols(), exprs, ctx);
+      newMaps.push_back(newMap);
+    }
+
+    Type accElemTy = getElementTypeOrSelf(acc.getType());
+    Value localInit = getCombiningIdentityValue(
+        loc, rewriter, contractOp.getKind(), disAcc.getType());
+
+    auto localContractOp = rewriter.create<vector::ContractionOp>(
+        loc, disLhs, disRhs, localInit, rewriter.getAffineMapArrayAttr(newMaps),
+        rewriter.getArrayAttr(newIterators), contractOp.getKind());
+    localContractOp->setDiscardableAttrs(
+        contractOp->getDiscardableAttrDictionary());
+
+    VectorValue localContractValue;
+    if (accVector) {
+      localContractValue = dyn_cast<VectorValue>(localContractOp.getResult());
+    } else {
+      VectorType vecType = VectorType::get(ArrayRef{int64_t(1)}, accElemTy);
+      localContractValue = rewriter.create<vector::BroadcastOp>(
+          loc, vecType, localContractOp.getResult());
+    }
+
+    assert(localContractValue && "result should have been a vector");
+
+    // Flatten the locally result value.
+    VectorType shaped = localContractValue.getType();
+    int64_t numElements = shaped.getNumElements();
+    SmallVector<int64_t> flatShape(1, numElements);
+    VectorType flatVecType = VectorType::get(flatShape, accElemTy);
+    VectorValue flat = rewriter.create<vector::ShapeCastOp>(loc, flatVecType,
+                                                            localContractValue);
+
+    // Do inter-thread/warp reduce.
+    FailureOr<VectorValue> threadReduced = doThreadReduction(
+        rewriter, lhsLayout, flat, contractOp.getKind(), reducedDims);
+    if (failed(threadReduced)) {
+      return failure();
+    }
+
+    // Do reduction against accumulator, which needs to be done after thread
+    // reduction.
+    VectorValue unflattened = rewriter.create<vector::ShapeCastOp>(
+        loc, shaped, threadReduced.value());
+
+    if (!accVector) {
+      disAcc = rewriter.create<vector::BroadcastOp>(loc, shaped, disAcc);
+    }
+
+    Value accReduction = vector::makeArithReduction(
+        rewriter, loc, contractOp.getKind(), unflattened, disAcc);
+    auto accReduced = dyn_cast<VectorValue>(accReduction);
+    if (!accReduced) {
+      return failure();
+    }
+
+    if (resVector) {
+      replaceOpWithDistributedValues(rewriter, contractOp, accReduced);
+    } else {
+      Value accReducedVal = rewriter.create<vector::ExtractOp>(
+          loc, accReduction, SmallVector<int64_t>{0});
+      replaceOpWithDistributedValues(rewriter, contractOp, accReducedVal);
+    }
+
+    return success();
```

**Comment:**
Would it be possible to split this function into a couple smaller helpers matching the 3-stage lowering described in the comment? This implementation is a bit long and hard to follow.

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPUConfigureTensorLayouts.cpp:344`

```diff
@@ -341,8 +341,10 @@ struct LLVMGPUConfigureTensorLayoutsPass final
     llvm::StringLiteral scheduleAttrName =
         IREE::GPU::MMAScheduleAttr::getMnemonic();
     DictionaryAttr configDict = getTranslationInfo(func).getConfiguration();
-    auto scheduleAttr = dyn_cast_or_null<IREE::GPU::MMAScheduleAttr>(
-        configDict.get(scheduleAttrName));
+    IREE::GPU::MMAScheduleAttr scheduleAttr = nullptr;
```

**Comment:**
```suggestion
    IREE::GPU::MMAScheduleAttr scheduleAttr;
```

---


---


## [PR #18825](https://github.com/iree-org/iree/pull/18825): [VectorDistribution] Add kernel configs for root reduction operations (4/4)

### Review Summary

**COMMENTED** (2024-10-18)

Looks OK


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp:932`

```diff
@@ -814,6 +820,156 @@ setAttentionVectorDistributionConfig(IREE::GPU::TargetAttr target,
                                                targetSubgroupSize, configDict);
 }
 
+// Checks if 'outputOperand' is a reduction with a single combiner operation.
+// Returns the combiner operation of the reduction, or nullptr if it is not a
+// valid reduction. This function is adapted from the implementation of Linalg
+// vectorization.
+static Operation *matchLinalgReduction(OpOperand *outputOperand) {
+  auto linalgOp = cast<linalg::LinalgOp>(outputOperand->getOwner());
+  unsigned outputPos =
+      outputOperand->getOperandNumber() - linalgOp.getNumDpsInputs();
+  // Only single combiner operations are supported for now.
+  SmallVector<Operation *, 4> combinerOps;
+  if (!matchReduction(linalgOp.getRegionOutputArgs(), outputPos, combinerOps) ||
+      combinerOps.size() != 1)
+    return nullptr;
+
+  // Return the combiner operation.
+  return combinerOps[0];
+}
+
+static LogicalResult reductionPrecondition(linalg::LinalgOp op) {
+  SmallVector<unsigned> parallelDims;
+  SmallVector<unsigned> reductionDims;
+  op.getParallelDims(parallelDims);
+  op.getReductionDims(reductionDims);
+  if (reductionDims.empty())
+    return failure();
+
+  SmallVector<int64_t, 4> bounds = op.getStaticLoopRanges();
+  int64_t numParallelDims = op.getNumParallelLoops();
+
+  // Make sure reduction dimensions are static and innermost ones.
+  for (unsigned dim : reductionDims) {
+    if (ShapedType::isDynamic(bounds[dim]))
+      return failure();
+    if (dim < numParallelDims) {
+      LDBG("Not innermost ones");
+      return failure();
+    }
+  }
+
+  // Make sure parallel dimensions are static
+  for (unsigned dim : parallelDims) {
+    if (ShapedType::isDynamic(bounds[dim]))
+      return failure();
+  }
+
+  for (OpOperand &opOperand : op.getDpsInitsMutable()) {
+    AffineMap indexingMap = op.getMatchingIndexingMap(&opOperand);
+    if (indexingMap.isPermutation())
+      continue;
+
+    Operation *reduceOp = matchLinalgReduction(&opOperand);
+    if (!reduceOp || !linalg::getCombinerOpKind(reduceOp)) {
+      LDBG("reduction precondition failed: reduction detection failed\n");
+      return failure();
+    }
+  }
+  return success();
+}
+
+static LogicalResult
+setReductionVectorDistributionConfig(IREE::GPU::TargetAttr target,
+                                     mlir::FunctionOpInterface entryPoint,
+                                     linalg::LinalgOp op) {
+  if (!target.supportsSubgroupShuffle())
+    return failure();
+
+  SmallVector<unsigned> parallelDims;
+  SmallVector<unsigned> reductionDims;
+  op.getParallelDims(parallelDims);
+  op.getReductionDims(reductionDims);
+
+  SmallVector<int64_t, 4> bounds = op.getStaticLoopRanges();
+
+  int64_t reductionSize = 1;
+  for (int64_t dim : reductionDims)
+    reductionSize *= bounds[dim];
+
+  int64_t subgroupSize = 0;
+  for (int s : target.getWgp().getSubgroupSizeChoices().asArrayRef()) {
+    if (reductionSize % s == 0) {
+      subgroupSize = s;
+      break;
+    }
+  }
+  if (subgroupSize == 0)
+    return failure();
+
+  Value init = op.getDpsInitOperand(0)->get();
+  Value src = op.getDpsInputOperand(0)->get();
+  Type initElemType = getElementTypeOrSelf(init);
+  Type srcElemType = getElementTypeOrSelf(src);
+
+  if (auto initOp = init.getDefiningOp<linalg::GenericOp>()) {
+    if (IREE::LinalgExt::isBitExtendOp(initOp))
+      initElemType = getElementTypeOrSelf(initOp.getDpsInputs()[0]);
+  }
+
+  if (auto srcOp = src.getDefiningOp<linalg::GenericOp>()) {
+    if (IREE::LinalgExt::isBitExtendOp(srcOp))
+      srcElemType = getElementTypeOrSelf(srcOp.getDpsInputs()[0]);
+  }
+
+  if (!initElemType.isIntOrFloat() || !srcElemType.isIntOrFloat())
+    return failure();
+
+  unsigned bitWidth = std::min(initElemType.getIntOrFloatBitWidth(),
+                               srcElemType.getIntOrFloatBitWidth());
+
+  // Reduction distribution only supports 8/16/32 bit types now.
+  if (bitWidth != 32 && bitWidth != 16 && bitWidth != 8)
```

**Comment:**
nit: you can use `llvm::is_contained`

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp:1103`

```diff
@@ -814,6 +820,156 @@ setAttentionVectorDistributionConfig(IREE::GPU::TargetAttr target,
                                                targetSubgroupSize, configDict);
 }
 
+// Checks if 'outputOperand' is a reduction with a single combiner operation.
+// Returns the combiner operation of the reduction, or nullptr if it is not a
+// valid reduction. This function is adapted from the implementation of Linalg
+// vectorization.
+static Operation *matchLinalgReduction(OpOperand *outputOperand) {
+  auto linalgOp = cast<linalg::LinalgOp>(outputOperand->getOwner());
+  unsigned outputPos =
+      outputOperand->getOperandNumber() - linalgOp.getNumDpsInputs();
+  // Only single combiner operations are supported for now.
+  SmallVector<Operation *, 4> combinerOps;
+  if (!matchReduction(linalgOp.getRegionOutputArgs(), outputPos, combinerOps) ||
+      combinerOps.size() != 1)
+    return nullptr;
+
+  // Return the combiner operation.
+  return combinerOps[0];
+}
+
+static LogicalResult reductionPrecondition(linalg::LinalgOp op) {
+  SmallVector<unsigned> parallelDims;
+  SmallVector<unsigned> reductionDims;
+  op.getParallelDims(parallelDims);
+  op.getReductionDims(reductionDims);
+  if (reductionDims.empty())
+    return failure();
+
+  SmallVector<int64_t, 4> bounds = op.getStaticLoopRanges();
+  int64_t numParallelDims = op.getNumParallelLoops();
+
+  // Make sure reduction dimensions are static and innermost ones.
+  for (unsigned dim : reductionDims) {
+    if (ShapedType::isDynamic(bounds[dim]))
+      return failure();
+    if (dim < numParallelDims) {
+      LDBG("Not innermost ones");
+      return failure();
+    }
+  }
+
+  // Make sure parallel dimensions are static
+  for (unsigned dim : parallelDims) {
+    if (ShapedType::isDynamic(bounds[dim]))
+      return failure();
+  }
+
+  for (OpOperand &opOperand : op.getDpsInitsMutable()) {
+    AffineMap indexingMap = op.getMatchingIndexingMap(&opOperand);
+    if (indexingMap.isPermutation())
+      continue;
+
+    Operation *reduceOp = matchLinalgReduction(&opOperand);
+    if (!reduceOp || !linalg::getCombinerOpKind(reduceOp)) {
+      LDBG("reduction precondition failed: reduction detection failed\n");
+      return failure();
+    }
+  }
+  return success();
+}
+
+static LogicalResult
+setReductionVectorDistributionConfig(IREE::GPU::TargetAttr target,
+                                     mlir::FunctionOpInterface entryPoint,
+                                     linalg::LinalgOp op) {
+  if (!target.supportsSubgroupShuffle())
+    return failure();
+
+  SmallVector<unsigned> parallelDims;
+  SmallVector<unsigned> reductionDims;
+  op.getParallelDims(parallelDims);
+  op.getReductionDims(reductionDims);
+
+  SmallVector<int64_t, 4> bounds = op.getStaticLoopRanges();
+
+  int64_t reductionSize = 1;
+  for (int64_t dim : reductionDims)
+    reductionSize *= bounds[dim];
+
+  int64_t subgroupSize = 0;
+  for (int s : target.getWgp().getSubgroupSizeChoices().asArrayRef()) {
+    if (reductionSize % s == 0) {
+      subgroupSize = s;
+      break;
+    }
+  }
+  if (subgroupSize == 0)
+    return failure();
+
+  Value init = op.getDpsInitOperand(0)->get();
+  Value src = op.getDpsInputOperand(0)->get();
+  Type initElemType = getElementTypeOrSelf(init);
+  Type srcElemType = getElementTypeOrSelf(src);
+
+  if (auto initOp = init.getDefiningOp<linalg::GenericOp>()) {
+    if (IREE::LinalgExt::isBitExtendOp(initOp))
+      initElemType = getElementTypeOrSelf(initOp.getDpsInputs()[0]);
+  }
+
+  if (auto srcOp = src.getDefiningOp<linalg::GenericOp>()) {
+    if (IREE::LinalgExt::isBitExtendOp(srcOp))
+      srcElemType = getElementTypeOrSelf(srcOp.getDpsInputs()[0]);
+  }
+
+  if (!initElemType.isIntOrFloat() || !srcElemType.isIntOrFloat())
+    return failure();
+
+  unsigned bitWidth = std::min(initElemType.getIntOrFloatBitWidth(),
+                               srcElemType.getIntOrFloatBitWidth());
+
+  // Reduction distribution only supports 8/16/32 bit types now.
+  if (bitWidth != 32 && bitWidth != 16 && bitWidth != 8)
+    return failure();
+
+  const int64_t targetSubgroupSize = target.getPreferredSubgroupSize();
+
+  const unsigned largestLoadSizeInBits = 128;
+  unsigned vectorSize = largestLoadSizeInBits / bitWidth;
```

**Comment:**
We should probably make this a target property like discussed here: https://discord.com/channels/689900678990135345/1254843174111678555/1296841473249251439.
This is not a blocker IMO.

---


---


## [PR #18822](https://github.com/iree-org/iree/pull/18822): [VectorDistribution] Plumb the VectorDistribute pipeline to support reduction operations (3/4)

### Review Summary

**COMMENTED** (2024-10-17)

Why do we need the new pipeline option?


**APPROVED** (2024-11-04)


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h:47`

```diff
@@ -44,6 +44,7 @@ struct GPUPipelineOptions {
   bool enableReduceSharedMemoryBankConflicts = true;
   bool prefetchSharedMemory = false;
   bool enableUkernels = false;
+  bool generateContract = true;
```

**Comment:**
I think changes to these options need to be reflected on the tablegen side (in the matching attribute)? @Max191

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp:232`

```diff
@@ -227,6 +228,9 @@ static void addGPUVectorizationPasses(OpPassManager &funcPassManager) {
   options.vectorizeGatherAccesses = true;
   options.enableCleanup = false;
   options.foldCastIntoContract = true;
+  // used for supporting reduction along VectorDistribute pipeline
+  // disable conversion from reduction ops to contraction ops.
```

**Comment:**
Use proper capitalization

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp:867`

```diff
@@ -857,7 +861,12 @@ void addGPUVectorDistributePassPipeline(OpPassManager &funcPassManager,
   funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
 
   // Linalg -> Vector
-  addGPUVectorizationPasses(funcPassManager);
+  if (options.generateContract) {
+    addGPUVectorizationPasses(funcPassManager);
+  } else {
+    // disable conversion from reductions ops to contraction ops.
```

**Comment:**
Also here

---


---


## [PR #18800](https://github.com/iree-org/iree/pull/18800): [VectorDistribution] Add vector distribution support multi-dim reduction with scalars

### Review Summary

**COMMENTED** (2024-10-17)


**COMMENTED** (2024-10-23)


**APPROVED** (2024-10-25)

LGTM % one comment


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h:112`

```diff
@@ -105,6 +107,10 @@ FailureOr<SmallVector<int64_t>> getGPUTileSize(mlir::FunctionOpInterface funcOp,
 FailureOr<scf::SCFTileSizeComputationFunction>
 getGPUScfTileSizeComputeFn(mlir::FunctionOpInterface funcOp, int tilingLevel);
 
+// Returns a boolean flag indicating whether the input value 'val' is a
+// vector, determined by checking its rank.
+bool isVector(VectorValue val);
```

**Comment:**
This strikes me as an odd helper: you give 'vector' a new meaning without introducing a name. Instead, I'd either flip it and add a helper like `isRank0(VectorValue val)`, or just expand the check where you need it.

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp:219`

```diff
@@ -212,6 +212,12 @@ getGPUScfTileSizeComputeFn(mlir::FunctionOpInterface funcOp, int tilingLevel) {
   return computeFn;
 }
 
+bool isVector(VectorValue val) {
+  if (val.getType().getRank() != 0)
+    return true;
+  return false;
+}
```

**Comment:**
```suggestion
bool isVector(VectorValue val) {
  return val.getType().getRank() != 0;
}
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp:1063`

```diff
@@ -1027,6 +1040,47 @@ void EnforceLayout::visitRegionSuccessors(RegionBranchOpInterface branch,
   }
 }
 
+void EnforceLayout::visitRegionBranchTerminatorOpInterface(
+    RegionBranchOpInterface branch, RegionBranchPoint branchPoint) {
+  SmallVector<RegionSuccessor> successors;
+  branch.getSuccessorRegions(branchPoint, successors);
+  if (!branch.hasLoop())
+    return;
+  SmallVector<DistributionLayout *> resultLattices;
+  for (Value result : branch->getResults()) {
+    DistributionLayout *resultLattice = getLatticeElement(result);
+    if (resultLattice->isUninitialized())
+      continue;
+    resultLattices.push_back(resultLattice);
+  }
+
+  // Result lattice not has a layout yet.
+  if (resultLattices.empty())
+    return;
+
+  // We do not support multiple results yet.
+  if (resultLattices.size() != 1)
+    return;
```

**Comment:**
The first check is redundant

---

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp:1066`

```diff
@@ -1027,6 +1040,47 @@ void EnforceLayout::visitRegionSuccessors(RegionBranchOpInterface branch,
   }
 }
 
+void EnforceLayout::visitRegionBranchTerminatorOpInterface(
+    RegionBranchOpInterface branch, RegionBranchPoint branchPoint) {
+  SmallVector<RegionSuccessor> successors;
+  branch.getSuccessorRegions(branchPoint, successors);
+  if (!branch.hasLoop())
+    return;
+  SmallVector<DistributionLayout *> resultLattices;
+  for (Value result : branch->getResults()) {
+    DistributionLayout *resultLattice = getLatticeElement(result);
+    if (resultLattice->isUninitialized())
+      continue;
+    resultLattices.push_back(resultLattice);
+  }
+
+  // Result lattice not has a layout yet.
+  if (resultLattices.empty())
+    return;
+
+  // We do not support multiple results yet.
+  if (resultLattices.size() != 1)
+    return;
+
+  for (RegionSuccessor successor : successors) {
+    if (auto succ = successor.getSuccessor()) {
```

**Comment:**
Can you use the actual type here instead of `auto`? It's not clear based on the RHS

---

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp:713`

```diff
@@ -699,6 +706,9 @@ static void enforceLayoutToBroadcastOp(
 
   auto resultShape = broadcast.getResultVectorType().getShape();
   auto inputType = broadcast.getSourceType();
+  if (!isa<VectorType>(inputType)) {
+    return;
+  }
   assert(isa<VectorType>(inputType) &&
          "Scalar broadcast not supported for now.");
```

**Comment:**
This assertion is obsolete now.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.cpp:136`

```diff
@@ -132,7 +132,8 @@ void DistributionPattern::replaceOpWithDistributedValues(
   for (auto [opResult, replacement] :
        llvm::zip_equal(op->getOpResults(), values)) {
     // If this value is a vector type, it must be converted back to simd.
-    if (isa<VectorType>(replacement.getType())) {
+    if (isa<VectorType>(replacement.getType()) &&
+        cast<ShapedType>(replacement.getType()).getRank() != 0) {
```

**Comment:**
use dyn_cast instead:
```c++
if (auto x = dyn_cast<Y>(y)) {
  if (x.something() == Z) {
 ```
 
 See https://llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates:~:text=Note%20that%20you%20should%20not%20use%20an%20isa%3C%3E%20test%20followed%20by%20a%20cast%3C%3E%2C%20for%20that%20use%20the%20dyn_cast%3C%3E%20operator.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:535`

```diff
@@ -485,15 +516,27 @@ struct DistributeMultiReduction final
     // reduction.
     VectorValue unflattened = rewriter.create<vector::ShapeCastOp>(
         loc, shaped, threadReduced.value());
+
+    if (!accVector) {
+      disAcc = rewriter.create<vector::BroadcastOp>(loc, shaped, disAcc);
+    }
+
     Value accReduction = vector::makeArithReduction(
         rewriter, loc, multiReduceOp.getKind(), unflattened, disAcc);
     auto accReduced = dyn_cast<VectorValue>(accReduction);
     if (!accReduced) {
       return failure();
     }
-    replaceOpWithDistributedValues(rewriter, multiReduceOp, accReduced);
 
-    return failure();
+    if (resVector) {
+      replaceOpWithDistributedValues(rewriter, multiReduceOp, accReduced);
+    } else {
+      Value accReducedVal = rewriter.create<vector::ExtractOp>(
+          loc, accReduction, SmallVector<int64_t>{0});
```

**Comment:**
You don't need a vector here, you can do: `ArrayRef{int64_t(0)}`

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:493`

```diff
@@ -462,7 +485,15 @@ struct DistributeMultiReduction final
     auto localReduction = rewriter.create<vector::MultiDimReductionOp>(
         loc, disSrc, localInit, distributedReductionMask,
         multiReduceOp.getKind());
-    auto locallyReduced = dyn_cast<VectorValue>(localReduction.getResult());
+
+    VectorValue locallyReduced;
+    if (accVector) {
+      locallyReduced = dyn_cast<VectorValue>(localReduction.getResult());
+    } else {
+      VectorType vecType = VectorType::get(SmallVector<int64_t>{1}, elemTy);
```

**Comment:**
Also here, no need to use a vector

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:428`

```diff
@@ -413,15 +417,29 @@ struct DistributeMultiReduction final
                                 DistributionSignature &signature,
                                 PatternRewriter &rewriter) const override {
     VectorValue srcVector = multiReduceOp.getSource();
-    auto accVector = dyn_cast<VectorValue>(multiReduceOp.getAcc());
-    if (!accVector) {
-      return rewriter.notifyMatchFailure(
-          multiReduceOp, "unimplemented: scalar accumulator distribution");
-    }
-    auto resVector = dyn_cast<VectorValue>(multiReduceOp.getResult());
-    if (!resVector) {
-      return rewriter.notifyMatchFailure(
-          multiReduceOp, "unimplemented: scalar result distribution");
+    Value acc = multiReduceOp.getAcc();
+    Value res = multiReduceOp.getResult();
+    auto accVector = dyn_cast<VectorValue>(acc);
+    auto resVector = dyn_cast<VectorValue>(res);
+    Type accType = acc.getType();
+    Type resType = res.getType();
+    Type accElemTy;
+    if (accVector) {
+      accElemTy = accVector.getType().getElementType();
```

**Comment:**
You can use `getElementTypeOrSelf`. Also below.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:312`

```diff
@@ -305,7 +305,11 @@ struct DistributeBroadcast final : OpDistributionPattern<vector::BroadcastOp> {
     auto vectorType = VectorType::get(distShape, elementType);
 
     VectorValue srcVector = dyn_cast<VectorValue>(broadcastOp.getSource());
-    if (!srcVector) {
+    // Types such as vector<f32> can return a valid pointer. An additional
+    // rank check is added to ensure that the type is indeed a vector
+    // value.
+    bool isSrcVector = (srcVector) && (isVector(srcVector));
+    if (!isSrcVector) {
```

**Comment:**
This is the only use of `isSrcVector`, so it makes sense to inline it
```suggestion
    if (!srcVector || !isVector(srcVector)) {
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUDistributionPatterns.cpp:588`

```diff
@@ -582,8 +584,12 @@ struct DistributeScfFor final : OpDistributionPattern<scf::ForOp> {
     SmallVector<Value> operands;
     for (Value operand : yieldOp->getOperands()) {
       if (auto vectorOperand = dyn_cast<VectorValue>(operand)) {
-        operand = DistributionPattern::getDistributed(rewriter, vectorOperand,
-                                                      signature[vectorOperand]);
+        // Types such as vector<f32> can pass this condition. An additional rank
+        // check is added here to ensure that the type is indeed a vector value.
```

**Comment:**
`vector<f32>` is a vector value so the comment doesn't make sense to me

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:310`

```diff
@@ -305,7 +305,11 @@ struct DistributeBroadcast final : OpDistributionPattern<vector::BroadcastOp> {
     auto vectorType = VectorType::get(distShape, elementType);
 
     VectorValue srcVector = dyn_cast<VectorValue>(broadcastOp.getSource());
-    if (!srcVector) {
+    // Types such as vector<f32> can return a valid pointer. An additional
+    // rank check is added to ensure that the type is indeed a vector
+    // value.
```

**Comment:**
Similar here -- the comment doesn't make sense to me. What does it mean for `vector<f32>` to return a pointer?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:312`

```diff
@@ -305,7 +305,11 @@ struct DistributeBroadcast final : OpDistributionPattern<vector::BroadcastOp> {
     auto vectorType = VectorType::get(distShape, elementType);
 
     VectorValue srcVector = dyn_cast<VectorValue>(broadcastOp.getSource());
-    if (!srcVector) {
+    // Types such as vector<f32> can return a valid pointer. An additional
+    // rank check is added to ensure that the type is indeed a vector
+    // value.
+    bool isSrcVector = (srcVector) && (isNonZeroRank(srcVector));
+    if (!isSrcVector) {
```

**Comment:**
We should inline this condition into the `if`

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:426`

```diff
@@ -413,15 +417,17 @@ struct DistributeMultiReduction final
                                 DistributionSignature &signature,
                                 PatternRewriter &rewriter) const override {
     VectorValue srcVector = multiReduceOp.getSource();
-    auto accVector = dyn_cast<VectorValue>(multiReduceOp.getAcc());
-    if (!accVector) {
-      return rewriter.notifyMatchFailure(
-          multiReduceOp, "unimplemented: scalar accumulator distribution");
-    }
-    auto resVector = dyn_cast<VectorValue>(multiReduceOp.getResult());
-    if (!resVector) {
-      return rewriter.notifyMatchFailure(
-          multiReduceOp, "unimplemented: scalar result distribution");
+    Value acc = multiReduceOp.getAcc();
+    Value res = multiReduceOp.getResult();
+    auto accVector = dyn_cast<VectorValue>(acc);
+    auto resVector = dyn_cast<VectorValue>(res);
+
+    Type accElemTy = getElementTypeOrSelf(acc.getType());
+    Type resElemTy = getElementTypeOrSelf(res.getType());
```

**Comment:**
Either check that the `dyn_cast`s succeeded (and return an error if not) or use `cast` to assert on cast failures

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.cpp:135`

```diff
@@ -132,14 +132,17 @@ void DistributionPattern::replaceOpWithDistributedValues(
   for (auto [opResult, replacement] :
        llvm::zip_equal(op->getOpResults(), values)) {
     // If this value is a vector type, it must be converted back to simd.
-    if (isa<VectorType>(replacement.getType())) {
-      auto oldResult = cast<VectorValue>(opResult);
-      // Create a toSIMD op to convert the value back to the simd.
-      rewriter.setInsertionPointAfterValue(oldResult);
-      Value toSIMD = rewriter.create<IREE::VectorExt::ToSIMDOp>(
-          oldResult.getLoc(), oldResult.getType(), replacement);
-      // Add to replacements.
-      replacement = toSIMD;
+    if (VectorType replacementType =
```

**Comment:**
```suggestion
    if (auto replacementType =
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp:712`

```diff
@@ -699,8 +706,10 @@ static void enforceLayoutToBroadcastOp(
 
   auto resultShape = broadcast.getResultVectorType().getShape();
   auto inputType = broadcast.getSourceType();
-  assert(isa<VectorType>(inputType) &&
-         "Scalar broadcast not supported for now.");
+  if (!isa<VectorType>(inputType)) {
+    return;
+  }
+
```

**Comment:**
Follow the llvm coding style and use `dyn_cast` here to avoid repeated type checking below

---

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp:948`

```diff
@@ -935,6 +944,9 @@ void EnforceLayout::visitOperation(Operation *op) {
   if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
     visitRegionSuccessors(branch, RegionBranchPoint::parent(),
                           branch->getOpOperands());
+
+    // Handle the propation from scf.for to yield op
```

**Comment:**
typo and punctuation
```suggestion
    // Handle the propagation from scf.for to yield op.
```

---

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h:111`

```diff
@@ -105,6 +107,10 @@ FailureOr<SmallVector<int64_t>> getGPUTileSize(mlir::FunctionOpInterface funcOp,
 FailureOr<scf::SCFTileSizeComputationFunction>
 getGPUScfTileSizeComputeFn(mlir::FunctionOpInterface funcOp, int tilingLevel);
 
+// Determines whether the rank of the input value 'val' is non-zero.
+// Returns true if the rank is non-zero; otherwise, returns false.
```

**Comment:**
```suggestion
// Returns true iff the rank of the input value 'val' is non-zero.
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUDistributionPatterns.cpp:592`

```diff
@@ -584,8 +584,12 @@ struct DistributeScfFor final : OpDistributionPattern<scf::ForOp> {
     SmallVector<Value> operands;
     for (Value operand : yieldOp->getOperands()) {
       if (auto vectorOperand = dyn_cast<VectorValue>(operand)) {
-        // Types such as vector<f32> can pass this condition. An additional rank
-        // check is added here to ensure that the type is indeed a vector value.
+        // Check if the operand is a vector type (e.g., vector<f32>), which
+        // passes this condition as it is indeed a vector. However, distributing
+        // the operand requires it to have a non-zero rank, meaning it must have
+        // at least one dimension. To ensure this, we add a necessary rank
+        // check. If the vector has a non-zero rank, the operand is distributed
+        // according to the provided layout signature.
```

**Comment:**
Thanks for revising this. Now the comment is very clear but also very verbose. I think we can simplify this.

```suggestion
        // Distributing the operand requires it to have a non-zero rank, meaning it must have
        // at least one dimension. If the vector has a non-zero rank, the operand is distributed
        // according to the provided layout signature.
```

---


---


## [PR #18784](https://github.com/iree-org/iree/pull/18784): [VectorDistribution] Add scalar support for distributing multi-dim reduction (1/4)

### Review Summary

**COMMENTED** (2024-10-17)

I reviewed your PRs out of ordered and left the comments here: https://github.com/iree-org/iree/pull/18800


**COMMENTED** (2024-10-30)


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:484`

```diff
@@ -462,7 +476,15 @@ struct DistributeMultiReduction final
     auto localReduction = rewriter.create<vector::MultiDimReductionOp>(
         loc, disSrc, localInit, distributedReductionMask,
         multiReduceOp.getKind());
-    auto locallyReduced = dyn_cast<VectorValue>(localReduction.getResult());
+
+    VectorValue locallyReduced;
+    if (accVector) {
+      locallyReduced = dyn_cast<VectorValue>(localReduction.getResult());
+    } else {
+      VectorType vecType = VectorType::get(SmallVector<int64_t>{1}, elemTy);
```

**Comment:**
You should be able to use ArrayRef here instead of constructing a vector

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp:526`

```diff
@@ -485,15 +507,27 @@ struct DistributeMultiReduction final
     // reduction.
     VectorValue unflattened = rewriter.create<vector::ShapeCastOp>(
         loc, shaped, threadReduced.value());
+
+    if (!accVector) {
+      disAcc = rewriter.create<vector::BroadcastOp>(loc, shaped, disAcc);
+    }
+
     Value accReduction = vector::makeArithReduction(
         rewriter, loc, multiReduceOp.getKind(), unflattened, disAcc);
     auto accReduced = dyn_cast<VectorValue>(accReduction);
     if (!accReduced) {
       return failure();
     }
-    replaceOpWithDistributedValues(rewriter, multiReduceOp, accReduced);
 
-    return failure();
+    if (resVector) {
+      replaceOpWithDistributedValues(rewriter, multiReduceOp, accReduced);
+    } else {
+      Value accReducedVal = rewriter.create<vector::ExtractOp>(
+          loc, accReduction, SmallVector<int64_t>{0});
```

**Comment:**
also here

---


---


## [PR #18012](https://github.com/iree-org/iree/pull/18012): use tile 16

### Review Summary

**APPROVED** (2024-07-26)

I tested this locally and it looks fine. We also checked that this only applies to those two convs across the whole unet.



---


## [PR #17811](https://github.com/iree-org/iree/pull/17811): Add workgroup chipletgroup strategy to workgroup reordering pass

### Review Summary

**CHANGES_REQUESTED** (2024-07-08)


**COMMENTED** (2024-07-10)


**COMMENTED** (2024-07-10)

Just some remaining issues with comments. Looks good otherwise.


**COMMENTED** (2024-07-10)


**COMMENTED** (2024-07-10)


**COMMENTED** (2024-07-11)


**COMMENTED** (2024-07-11)


**APPROVED** (2024-07-11)

LGTM, thanks for all the fixes


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/Passes.td:205`

```diff
@@ -199,10 +199,11 @@ def ReorderWorkgroupsPass :
   let dependentDialects = ["::mlir::affine::AffineDialect"];
   let options = [
     Option<"strategy", "strategy", "std::string", /*default=*/"",
-           "Workgroup reordering strategy, one of: '' (none),  'transpose', 'swizzle'">,
+           "Workgroup reordering strategy, one of: '' (none),  'transpose', 'swizzle', 'chipletgroup'">,
     Option<"logTile", "logTile", "unsigned",
             /*default=*/"0",
-           "The log2 of the tile size used for swizzling. (0: disabled, non-0: swizzling enabled)">,
+           "The log2 of the tile size used for swizzling and chipletgroup. "
```

**Comment:**
The name doesn't match the chiplet-group strategy. We should either rename it or add a second option if that makes more sense.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp:71`

```diff
@@ -68,6 +68,95 @@ makeSwizzledIds(Location loc, OpBuilder b, Value workgroupIdX,
   return {swizzledIdX, swizzledIdY};
 }
 
+// reoredering to make workgroup ids move slowly between chiplet groups
```

**Comment:**
```suggestion
// Reordering to make workgroup ids move slowly between chiplet groups.
```

Could you also give an example? IE pick some topology and show how the math works out.

Also say what the return value is.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp:76`

```diff
@@ -68,6 +68,95 @@ makeSwizzledIds(Location loc, OpBuilder b, Value workgroupIdX,
   return {swizzledIdX, swizzledIdY};
 }
 
+// reoredering to make workgroup ids move slowly between chiplet groups
+static Value chipletAwareWorkgroupReordering(Location loc, OpBuilder b,
+                                             Value linearizedId,
+                                             Value workgroupCountX,
+                                             Value workgroupCountY,
+                                             int64_t numChipletsPerGroup) {
```

**Comment:**
Is this the number of chiplets per *work*group or something else?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp:90`

```diff
@@ -68,6 +68,95 @@ makeSwizzledIds(Location loc, OpBuilder b, Value workgroupIdX,
   return {swizzledIdX, swizzledIdY};
 }
 
+// reoredering to make workgroup ids move slowly between chiplet groups
+static Value chipletAwareWorkgroupReordering(Location loc, OpBuilder b,
+                                             Value linearizedId,
+                                             Value workgroupCountX,
+                                             Value workgroupCountY,
+                                             int64_t numChipletsPerGroup) {
+  Value numChipletsVal =
+      b.createOrFold<arith::ConstantIndexOp>(loc, numChipletsPerGroup);
+  Value workgroupCount =
+      b.create<arith::MulIOp>(loc, workgroupCountX, workgroupCountY);
+  Value workgroupCountPerChiplet =
+      b.create<arith::DivUIOp>(loc, workgroupCount, numChipletsVal);
+  Value chipletId = b.create<arith::RemUIOp>(loc, linearizedId, numChipletsVal);
+  Value wgIdWithinChiplet =
+      b.create<arith::DivUIOp>(loc, linearizedId, numChipletsVal);
+  Value reorderedId = b.create<arith::AddIOp>(
+      loc, wgIdWithinChiplet,
+      b.create<arith::MulIOp>(loc, chipletId, workgroupCountPerChiplet));
+
+  // The following code is used to handle the remainder part
```

**Comment:**
```suggestion
  // Handle the remainder part.
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp:100`

```diff
@@ -68,6 +68,95 @@ makeSwizzledIds(Location loc, OpBuilder b, Value workgroupIdX,
   return {swizzledIdX, swizzledIdY};
 }
 
+// reoredering to make workgroup ids move slowly between chiplet groups
+static Value chipletAwareWorkgroupReordering(Location loc, OpBuilder b,
+                                             Value linearizedId,
+                                             Value workgroupCountX,
+                                             Value workgroupCountY,
+                                             int64_t numChipletsPerGroup) {
+  Value numChipletsVal =
+      b.createOrFold<arith::ConstantIndexOp>(loc, numChipletsPerGroup);
+  Value workgroupCount =
+      b.create<arith::MulIOp>(loc, workgroupCountX, workgroupCountY);
+  Value workgroupCountPerChiplet =
+      b.create<arith::DivUIOp>(loc, workgroupCount, numChipletsVal);
+  Value chipletId = b.create<arith::RemUIOp>(loc, linearizedId, numChipletsVal);
+  Value wgIdWithinChiplet =
+      b.create<arith::DivUIOp>(loc, linearizedId, numChipletsVal);
+  Value reorderedId = b.create<arith::AddIOp>(
+      loc, wgIdWithinChiplet,
+      b.create<arith::MulIOp>(loc, chipletId, workgroupCountPerChiplet));
+
+  // The following code is used to handle the remainder part
+
+  Value constOne = b.createOrFold<arith::ConstantIndexOp>(loc, 1);
+  Value lastWorkgroupId =
+      b.create<arith::SubIOp>(loc, workgroupCount, constOne);
+  Value modulatedLastWorkgroupId = b.create<arith::SubIOp>(
+      loc, lastWorkgroupId,
+      b.create<arith::RemUIOp>(loc, workgroupCount, numChipletsVal));
+  Value isGreaterThanFinalWorkgroupId = b.create<arith::CmpIOp>(
+      loc, arith::CmpIPredicate::ugt, linearizedId, modulatedLastWorkgroupId);
+  linearizedId = b.create<arith::SelectOp>(loc, isGreaterThanFinalWorkgroupId,
```

**Comment:**
nit: Do not reassign the function arguments, it makes the logic harder to follow

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp:110`

```diff
@@ -68,6 +68,95 @@ makeSwizzledIds(Location loc, OpBuilder b, Value workgroupIdX,
   return {swizzledIdX, swizzledIdY};
 }
 
+// reoredering to make workgroup ids move slowly between chiplet groups
+static Value chipletAwareWorkgroupReordering(Location loc, OpBuilder b,
+                                             Value linearizedId,
+                                             Value workgroupCountX,
+                                             Value workgroupCountY,
+                                             int64_t numChipletsPerGroup) {
+  Value numChipletsVal =
+      b.createOrFold<arith::ConstantIndexOp>(loc, numChipletsPerGroup);
+  Value workgroupCount =
+      b.create<arith::MulIOp>(loc, workgroupCountX, workgroupCountY);
+  Value workgroupCountPerChiplet =
+      b.create<arith::DivUIOp>(loc, workgroupCount, numChipletsVal);
+  Value chipletId = b.create<arith::RemUIOp>(loc, linearizedId, numChipletsVal);
+  Value wgIdWithinChiplet =
+      b.create<arith::DivUIOp>(loc, linearizedId, numChipletsVal);
+  Value reorderedId = b.create<arith::AddIOp>(
+      loc, wgIdWithinChiplet,
+      b.create<arith::MulIOp>(loc, chipletId, workgroupCountPerChiplet));
+
+  // The following code is used to handle the remainder part
+
+  Value constOne = b.createOrFold<arith::ConstantIndexOp>(loc, 1);
+  Value lastWorkgroupId =
+      b.create<arith::SubIOp>(loc, workgroupCount, constOne);
+  Value modulatedLastWorkgroupId = b.create<arith::SubIOp>(
+      loc, lastWorkgroupId,
+      b.create<arith::RemUIOp>(loc, workgroupCount, numChipletsVal));
+  Value isGreaterThanFinalWorkgroupId = b.create<arith::CmpIOp>(
+      loc, arith::CmpIPredicate::ugt, linearizedId, modulatedLastWorkgroupId);
+  linearizedId = b.create<arith::SelectOp>(loc, isGreaterThanFinalWorkgroupId,
+                                           linearizedId, reorderedId);
+
+  return linearizedId;
+}
+
+// Chiplet-aware workgroup reordering strategy: reordering + super-grouping.
+// Step 1: Reorder the workgroup grid to move slowly between
+// chiplet groups (Function: chipletAwareWorkgroupReordering).
+// Step 2: Implement 'super-grouping' of workgroups before switching to the next
+// column.
```

**Comment:**
Say what the return value is.

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp:115`

```diff
@@ -68,6 +68,95 @@ makeSwizzledIds(Location loc, OpBuilder b, Value workgroupIdX,
   return {swizzledIdX, swizzledIdY};
 }
 
+// reoredering to make workgroup ids move slowly between chiplet groups
+static Value chipletAwareWorkgroupReordering(Location loc, OpBuilder b,
+                                             Value linearizedId,
+                                             Value workgroupCountX,
+                                             Value workgroupCountY,
+                                             int64_t numChipletsPerGroup) {
+  Value numChipletsVal =
+      b.createOrFold<arith::ConstantIndexOp>(loc, numChipletsPerGroup);
+  Value workgroupCount =
+      b.create<arith::MulIOp>(loc, workgroupCountX, workgroupCountY);
+  Value workgroupCountPerChiplet =
+      b.create<arith::DivUIOp>(loc, workgroupCount, numChipletsVal);
+  Value chipletId = b.create<arith::RemUIOp>(loc, linearizedId, numChipletsVal);
+  Value wgIdWithinChiplet =
+      b.create<arith::DivUIOp>(loc, linearizedId, numChipletsVal);
+  Value reorderedId = b.create<arith::AddIOp>(
+      loc, wgIdWithinChiplet,
+      b.create<arith::MulIOp>(loc, chipletId, workgroupCountPerChiplet));
+
+  // The following code is used to handle the remainder part
+
+  Value constOne = b.createOrFold<arith::ConstantIndexOp>(loc, 1);
+  Value lastWorkgroupId =
+      b.create<arith::SubIOp>(loc, workgroupCount, constOne);
+  Value modulatedLastWorkgroupId = b.create<arith::SubIOp>(
+      loc, lastWorkgroupId,
+      b.create<arith::RemUIOp>(loc, workgroupCount, numChipletsVal));
+  Value isGreaterThanFinalWorkgroupId = b.create<arith::CmpIOp>(
+      loc, arith::CmpIPredicate::ugt, linearizedId, modulatedLastWorkgroupId);
+  linearizedId = b.create<arith::SelectOp>(loc, isGreaterThanFinalWorkgroupId,
+                                           linearizedId, reorderedId);
+
+  return linearizedId;
+}
+
+// Chiplet-aware workgroup reordering strategy: reordering + super-grouping.
+// Step 1: Reorder the workgroup grid to move slowly between
+// chiplet groups (Function: chipletAwareWorkgroupReordering).
+// Step 2: Implement 'super-grouping' of workgroups before switching to the next
+// column.
+static std::pair<Value, Value>
+makeChipletGroupedIds(Location loc, OpBuilder b, Value workgroupIdX,
+                      Value workgroupIdY, Value workgroupCountX,
+                      Value workgroupCountY, unsigned chipletGroupTile) {
+  // Create one dimension ID for workgroup
```

**Comment:**
```suggestion
  // Create one dimension ID for workgroup.
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp:120`

```diff
@@ -68,6 +68,95 @@ makeSwizzledIds(Location loc, OpBuilder b, Value workgroupIdX,
   return {swizzledIdX, swizzledIdY};
 }
 
+// reoredering to make workgroup ids move slowly between chiplet groups
+static Value chipletAwareWorkgroupReordering(Location loc, OpBuilder b,
+                                             Value linearizedId,
+                                             Value workgroupCountX,
+                                             Value workgroupCountY,
+                                             int64_t numChipletsPerGroup) {
+  Value numChipletsVal =
+      b.createOrFold<arith::ConstantIndexOp>(loc, numChipletsPerGroup);
+  Value workgroupCount =
+      b.create<arith::MulIOp>(loc, workgroupCountX, workgroupCountY);
+  Value workgroupCountPerChiplet =
+      b.create<arith::DivUIOp>(loc, workgroupCount, numChipletsVal);
+  Value chipletId = b.create<arith::RemUIOp>(loc, linearizedId, numChipletsVal);
+  Value wgIdWithinChiplet =
+      b.create<arith::DivUIOp>(loc, linearizedId, numChipletsVal);
+  Value reorderedId = b.create<arith::AddIOp>(
+      loc, wgIdWithinChiplet,
+      b.create<arith::MulIOp>(loc, chipletId, workgroupCountPerChiplet));
+
+  // The following code is used to handle the remainder part
+
+  Value constOne = b.createOrFold<arith::ConstantIndexOp>(loc, 1);
+  Value lastWorkgroupId =
+      b.create<arith::SubIOp>(loc, workgroupCount, constOne);
+  Value modulatedLastWorkgroupId = b.create<arith::SubIOp>(
+      loc, lastWorkgroupId,
+      b.create<arith::RemUIOp>(loc, workgroupCount, numChipletsVal));
+  Value isGreaterThanFinalWorkgroupId = b.create<arith::CmpIOp>(
+      loc, arith::CmpIPredicate::ugt, linearizedId, modulatedLastWorkgroupId);
+  linearizedId = b.create<arith::SelectOp>(loc, isGreaterThanFinalWorkgroupId,
+                                           linearizedId, reorderedId);
+
+  return linearizedId;
+}
+
+// Chiplet-aware workgroup reordering strategy: reordering + super-grouping.
+// Step 1: Reorder the workgroup grid to move slowly between
+// chiplet groups (Function: chipletAwareWorkgroupReordering).
+// Step 2: Implement 'super-grouping' of workgroups before switching to the next
+// column.
+static std::pair<Value, Value>
+makeChipletGroupedIds(Location loc, OpBuilder b, Value workgroupIdX,
+                      Value workgroupIdY, Value workgroupCountX,
+                      Value workgroupCountY, unsigned chipletGroupTile) {
+  // Create one dimension ID for workgroup
+  Value linearized =
+      b.create<arith::MulIOp>(loc, workgroupIdY, workgroupCountX);
+  linearized = b.create<arith::AddIOp>(loc, linearized, workgroupIdX);
+
+  // This value is hardcoded for cdna3(mi300x)
```

**Comment:**
Should we plumb this through and add to the target description attribute? Do we have an issue for this?

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.td:280`

```diff
@@ -269,6 +269,27 @@ def IREEGPU_MmaScheduleAttr : AttrDef<IREEGPU_Dialect, "MMASchedule"> {
   }];
 }
 
+//===----------------------------------------------------------------------===//
+// Workgroup Reordering Attr
+
+def IREEGPU_WorkGroupReorderAttr: AttrDef<IREEGPU_Dialect, "WorkgroupReorderOptions">{
+  let mnemonic = "reorder_workgroups";
+  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
+
+  string description = [{
+    options for workgroup reordering strategies to improve L2 cache hit rate
```

**Comment:**
```suggestion
    Options for workgroup reordering strategies to improve L2 cache hit rate.
```

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp:95`

```diff
@@ -92,21 +92,31 @@ getPipelineOptions(FunctionOpInterface funcOp,
       // Get the workgroups reorder config and enable the workgroup reordering.
       Attribute reorderWorkgroupOption =
           config.get(LLVMGPUAttrNames::kReorderWorkgroups);
-      if (!isa<StringAttr>(reorderWorkgroupOption))
-        funcOp.emitOpError() << "'" << LLVMGPUAttrNames::kReorderWorkgroups
-                             << "' is expected to be a string attribute";
-      StringRef reorderStr = llvm::cast<StringAttr>(reorderWorkgroupOption);
-      if (reorderStr == "transpose") {
-        pipelineOptions.reorderStrategy = ReorderWorkgroupsStrategy::Transpose;
-      } else if (reorderStr == "swizzle") {
-        pipelineOptions.reorderStrategy = ReorderWorkgroupsStrategy::Swizzle;
-      } else {
-        if (reorderStr != "none")
-          funcOp.emitOpError()
-              << "Unknown " << LLVMGPUAttrNames::kReorderWorkgroups
-              << "value: " << reorderWorkgroupOption;
-        else
+      if (llvm::isa<IREE::GPU::WorkgroupReorderOptionsAttr>(
```

**Comment:**
nit: Do we need the `llvm::`?

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp:98`

```diff
@@ -92,21 +92,31 @@ getPipelineOptions(FunctionOpInterface funcOp,
       // Get the workgroups reorder config and enable the workgroup reordering.
       Attribute reorderWorkgroupOption =
           config.get(LLVMGPUAttrNames::kReorderWorkgroups);
-      if (!isa<StringAttr>(reorderWorkgroupOption))
-        funcOp.emitOpError() << "'" << LLVMGPUAttrNames::kReorderWorkgroups
-                             << "' is expected to be a string attribute";
-      StringRef reorderStr = llvm::cast<StringAttr>(reorderWorkgroupOption);
-      if (reorderStr == "transpose") {
-        pipelineOptions.reorderStrategy = ReorderWorkgroupsStrategy::Transpose;
-      } else if (reorderStr == "swizzle") {
-        pipelineOptions.reorderStrategy = ReorderWorkgroupsStrategy::Swizzle;
-      } else {
-        if (reorderStr != "none")
-          funcOp.emitOpError()
-              << "Unknown " << LLVMGPUAttrNames::kReorderWorkgroups
-              << "value: " << reorderWorkgroupOption;
-        else
+      if (llvm::isa<IREE::GPU::WorkgroupReorderOptionsAttr>(
+              reorderWorkgroupOption)) {
+        IREE::GPU::WorkgroupReorderOptionsAttr ReorderOption =
+            llvm::dyn_cast<IREE::GPU::WorkgroupReorderOptionsAttr>(
```

**Comment:**
also here

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/config_user_vector_distribute.mlir:90`

```diff
@@ -87,11 +87,11 @@ hal.executable public @main_0_dispatch_0 {
 
 // OPT-OUT:       #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64
 // OPT-OUT-SAME:    mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
-// OPT-OUT-SAME:    reorder_workgroups = "transpose"
+// OPT-OUT-SAME:    reorder_workgroups = #iree_gpu.reorder_workgroups<reorder_option = transpose>
```

**Comment:**
Should we add the (optional) parameter to the same attribute?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/Passes.td:210`

```diff
@@ -199,10 +199,15 @@ def ReorderWorkgroupsPass :
   let dependentDialects = ["::mlir::affine::AffineDialect"];
   let options = [
     Option<"strategy", "strategy", "std::string", /*default=*/"",
-           "Workgroup reordering strategy, one of: '' (none),  'transpose', 'swizzle'">,
-    Option<"logTile", "logTile", "unsigned",
+           "Workgroup reordering strategy, one of: '' (none),  'transpose', 'swizzle', 'chipletgroup'">,
+    Option<"logSwTile", "logSwTile", "unsigned",
             /*default=*/"0",
-           "The log2 of the tile size used for swizzling. (0: disabled, non-0: swizzling enabled)">,
+           "The log2 of the tile size used for swizzling. "
+           "(0: swizzling disabled, non-0: swizzling enabled)">,
+    Option<"logCgTile", "logCgTile", "unsigned",
+            /*default=*/"0",
+           "The log2 of the tile size used for chipletgroup. "
+           "(0: chipletgroup disabled, non-0: chipletgroup enabled)">,
```

**Comment:**
Let's not reuse the flag for swizzle tile size to enable / disable chiplet-based reordering.  Do we need it at all to control chiplet-aware reordering?

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp:115`

```diff
@@ -68,6 +68,95 @@ makeSwizzledIds(Location loc, OpBuilder b, Value workgroupIdX,
   return {swizzledIdX, swizzledIdY};
 }
 
+// reoredering to make workgroup ids move slowly between chiplet groups
+static Value chipletAwareWorkgroupReordering(Location loc, OpBuilder b,
+                                             Value linearizedId,
+                                             Value workgroupCountX,
+                                             Value workgroupCountY,
+                                             int64_t numChipletsPerGroup) {
+  Value numChipletsVal =
+      b.createOrFold<arith::ConstantIndexOp>(loc, numChipletsPerGroup);
+  Value workgroupCount =
+      b.create<arith::MulIOp>(loc, workgroupCountX, workgroupCountY);
+  Value workgroupCountPerChiplet =
+      b.create<arith::DivUIOp>(loc, workgroupCount, numChipletsVal);
+  Value chipletId = b.create<arith::RemUIOp>(loc, linearizedId, numChipletsVal);
+  Value wgIdWithinChiplet =
+      b.create<arith::DivUIOp>(loc, linearizedId, numChipletsVal);
+  Value reorderedId = b.create<arith::AddIOp>(
+      loc, wgIdWithinChiplet,
+      b.create<arith::MulIOp>(loc, chipletId, workgroupCountPerChiplet));
+
+  // The following code is used to handle the remainder part
+
+  Value constOne = b.createOrFold<arith::ConstantIndexOp>(loc, 1);
+  Value lastWorkgroupId =
+      b.create<arith::SubIOp>(loc, workgroupCount, constOne);
+  Value modulatedLastWorkgroupId = b.create<arith::SubIOp>(
+      loc, lastWorkgroupId,
+      b.create<arith::RemUIOp>(loc, workgroupCount, numChipletsVal));
+  Value isGreaterThanFinalWorkgroupId = b.create<arith::CmpIOp>(
+      loc, arith::CmpIPredicate::ugt, linearizedId, modulatedLastWorkgroupId);
+  linearizedId = b.create<arith::SelectOp>(loc, isGreaterThanFinalWorkgroupId,
+                                           linearizedId, reorderedId);
+
+  return linearizedId;
+}
+
+// Chiplet-aware workgroup reordering strategy: reordering + super-grouping.
+// Step 1: Reorder the workgroup grid to move slowly between
+// chiplet groups (Function: chipletAwareWorkgroupReordering).
+// Step 2: Implement 'super-grouping' of workgroups before switching to the next
+// column.
+static std::pair<Value, Value>
+makeChipletGroupedIds(Location loc, OpBuilder b, Value workgroupIdX,
+                      Value workgroupIdY, Value workgroupCountX,
+                      Value workgroupCountY, unsigned chipletGroupTile) {
+  // Create one dimension ID for workgroup
```

**Comment:**
not addressed

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp:160`

```diff
@@ -68,6 +68,117 @@ makeSwizzledIds(Location loc, OpBuilder b, Value workgroupIdX,
   return {swizzledIdX, swizzledIdY};
 }
 
+// Reoredering to make workgroup ids move slowly between chiplet groups.
+
+// The following example illustrates the concept behind this function:
+// Currently, the GPU launches workgroups in a round-robin fashion across
+// each XCD partition on the GPU.
+// Assume we have 16 workgroups and XCDPartitionsOnGPU is 4.
+// The default GPU schedule will launch workgroups {0, 1, 2, 3, ..., 15} in
+// the following round-robin fashion:
+// Partition 0: {0, 4, 8, 12}
+// Partition 1: {1, 5, 9, 13}
+// Partition 2: {2, 6, 10, 14}
+// Partition 3: {3, 7, 11, 15}
+
+// After reordering, the workgroup IDs are {0, 4, 8, 12, 1, ..., 15},
+// resulting in the round-robin launching fashion:
+// Partition 0: {0, 1, 2, 3}
+// Partition 1: {4, 5, 6, 7}
+// Partition 2: {8, 9, 10, 11}
+// Partition 3: {12, 13, 14, 15}
+
+// The return value is each workgroup's permuted Id
+// In the above example:
+// linearedId 0's permuted Id is still 0
+// linearedId 1's permiuted Id is 4
+static Value chipletAwareWorkgroupReordering(Location loc, OpBuilder b,
+                                             Value linearizedId,
+                                             Value workgroupCountX,
+                                             Value workgroupCountY,
+                                             int64_t XCDParitionsOnGPU) {
+  Value numChipletsVal =
+      b.createOrFold<arith::ConstantIndexOp>(loc, XCDParitionsOnGPU);
+  Value workgroupCount =
+      b.create<arith::MulIOp>(loc, workgroupCountX, workgroupCountY);
+  Value workgroupCountPerChiplet =
+      b.create<arith::DivUIOp>(loc, workgroupCount, numChipletsVal);
+  Value chipletId = b.create<arith::RemUIOp>(loc, linearizedId, numChipletsVal);
+  Value wgIdWithinChiplet =
+      b.create<arith::DivUIOp>(loc, linearizedId, numChipletsVal);
+  Value reorderedId = b.create<arith::AddIOp>(
+      loc, wgIdWithinChiplet,
+      b.create<arith::MulIOp>(loc, chipletId, workgroupCountPerChiplet));
+
+  // Handle the remainder part.
+  Value constOne = b.createOrFold<arith::ConstantIndexOp>(loc, 1);
+  Value lastWorkgroupId =
+      b.create<arith::SubIOp>(loc, workgroupCount, constOne);
+  Value modulatedLastWorkgroupId = b.create<arith::SubIOp>(
+      loc, lastWorkgroupId,
+      b.create<arith::RemUIOp>(loc, workgroupCount, numChipletsVal));
+  Value isGreaterThanFinalWorkgroupId = b.create<arith::CmpIOp>(
+      loc, arith::CmpIPredicate::ugt, linearizedId, modulatedLastWorkgroupId);
+  Value finalId = b.create<arith::SelectOp>(loc, isGreaterThanFinalWorkgroupId,
+                                            linearizedId, reorderedId);
+
+  return finalId;
+}
+
+// Chiplet-aware workgroup reordering strategy: reordering + super-grouping.
+// Step 1: Reorder the workgroup grid to move slowly between
+// chiplet groups (Function: chipletAwareWorkgroupReordering).
+// Step 2: Implement 'super-grouping' of workgroups before switching to the next
+// column.
+static std::pair<Value, Value>
+makeChipletGroupedIds(Location loc, OpBuilder b, Value workgroupIdX,
+                      Value workgroupIdY, Value workgroupCountX,
+                      Value workgroupCountY, unsigned chipletGroupTile,
+                      unsigned numXCDs) {
+  // Create one dimension ID for workgroup
+  Value linearized =
+      b.create<arith::MulIOp>(loc, workgroupIdY, workgroupCountX);
+  linearized = b.create<arith::AddIOp>(loc, linearized, workgroupIdX);
+
+  assert(numXCDs > 1);
+  // Map chiplets to perform a spatially local tile operation.
+  // Reorder the linearized ID such that every consecutive group of chiplets
+  // is the slowest-changing dimension in the grid.
+  // Emphircally found that two chiplets as a group has better locality
+  // throughout.
+  linearized = chipletAwareWorkgroupReordering(
+      loc, b, linearized, workgroupCountX, workgroupCountY, numXCDs / 2);
+
+  // Detailed explaination about the idea behind the below implementation:
+  // the L2 Cache Optimizations subsection in
+  // https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#
+  // Emphircally, found rowGroupSize=16 for mi300x achieves good performance
+  unsigned rowGroupSize = chipletGroupTile;
+  Value rowGroupSizeVal =
+      b.createOrFold<arith::ConstantIndexOp>(loc, rowGroupSize);
+  // group every 16 workgroups along Y dimension
+  // Number of workgroups in the group
```

**Comment:**
nit: it's weird that the comments are interleaved with code mid-sentence
```suggestion
  unsigned rowGroupSize = chipletGroupTile;
  Value rowGroupSizeVal =
      b.createOrFold<arith::ConstantIndexOp>(loc, rowGroupSize);
  
  // Empirically, found rowGroupSize=16 for mi300x achieves good performance
  // group every 16 workgroups along Y dimension.
  
  // Number of workgroups in the group.
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp:92`

```diff
@@ -68,6 +68,119 @@ makeSwizzledIds(Location loc, OpBuilder b, Value workgroupIdX,
   return {swizzledIdX, swizzledIdY};
 }
 
+// Reoredering to make workgroup ids move slowly between chiplet groups.
+
+// The following example illustrates the concept behind this function:
+// Currently, the GPU launches workgroups in a round-robin fashion across
+// each XCD partition on the GPU.
+// Assume we have 16 workgroups and XCDPartitionsOnGPU is 4.
+// The default GPU schedule will launch workgroups {0, 1, 2, 3, ..., 15} in
+// the following round-robin fashion:
+// Partition 0: {0, 4, 8, 12}
+// Partition 1: {1, 5, 9, 13}
+// Partition 2: {2, 6, 10, 14}
+// Partition 3: {3, 7, 11, 15}
+
+// After reordering, the workgroup IDs are {0, 4, 8, 12, 1, ..., 15},
+// resulting in the round-robin launching fashion:
+// Partition 0: {0, 1, 2, 3}
+// Partition 1: {4, 5, 6, 7}
+// Partition 2: {8, 9, 10, 11}
+// Partition 3: {12, 13, 14, 15}
+
+// The return value is each workgroup's permuted Id
+// In the above example:
```

**Comment:**
```suggestion
// Returns permuted workgroup id (X and Y dimensions).
// In the above example:
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp:73`

```diff
@@ -68,6 +68,119 @@ makeSwizzledIds(Location loc, OpBuilder b, Value workgroupIdX,
   return {swizzledIdX, swizzledIdY};
 }
 
+// Reoredering to make workgroup ids move slowly between chiplet groups.
+
+// The following example illustrates the concept behind this function:
```

**Comment:**
```suggestion
// Example:
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp:78`

```diff
@@ -68,6 +68,119 @@ makeSwizzledIds(Location loc, OpBuilder b, Value workgroupIdX,
   return {swizzledIdX, swizzledIdY};
 }
 
+// Reoredering to make workgroup ids move slowly between chiplet groups.
+
+// The following example illustrates the concept behind this function:
+// Currently, the GPU launches workgroups in a round-robin fashion across
+// each XCD partition on the GPU.
+// Assume we have 16 workgroups and XCDPartitionsOnGPU is 4.
+// The default GPU schedule will launch workgroups {0, 1, 2, 3, ..., 15} in
+// the following round-robin fashion:
```

**Comment:**
```suggestion
// the following order:
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp:85`

```diff
@@ -68,6 +68,119 @@ makeSwizzledIds(Location loc, OpBuilder b, Value workgroupIdX,
   return {swizzledIdX, swizzledIdY};
 }
 
+// Reoredering to make workgroup ids move slowly between chiplet groups.
+
+// The following example illustrates the concept behind this function:
+// Currently, the GPU launches workgroups in a round-robin fashion across
+// each XCD partition on the GPU.
+// Assume we have 16 workgroups and XCDPartitionsOnGPU is 4.
+// The default GPU schedule will launch workgroups {0, 1, 2, 3, ..., 15} in
+// the following round-robin fashion:
+// Partition 0: {0, 4, 8, 12}
+// Partition 1: {1, 5, 9, 13}
+// Partition 2: {2, 6, 10, 14}
+// Partition 3: {3, 7, 11, 15}
+
+// After reordering, the workgroup IDs are {0, 4, 8, 12, 1, ..., 15},
+// resulting in the round-robin launching fashion:
```

**Comment:**
```suggestion
// resulting in the launch order:
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp:160`

```diff
@@ -68,6 +68,119 @@ makeSwizzledIds(Location loc, OpBuilder b, Value workgroupIdX,
   return {swizzledIdX, swizzledIdY};
 }
 
+// Reoredering to make workgroup ids move slowly between chiplet groups.
+
+// The following example illustrates the concept behind this function:
+// Currently, the GPU launches workgroups in a round-robin fashion across
+// each XCD partition on the GPU.
+// Assume we have 16 workgroups and XCDPartitionsOnGPU is 4.
+// The default GPU schedule will launch workgroups {0, 1, 2, 3, ..., 15} in
+// the following round-robin fashion:
+// Partition 0: {0, 4, 8, 12}
+// Partition 1: {1, 5, 9, 13}
+// Partition 2: {2, 6, 10, 14}
+// Partition 3: {3, 7, 11, 15}
+
+// After reordering, the workgroup IDs are {0, 4, 8, 12, 1, ..., 15},
+// resulting in the round-robin launching fashion:
+// Partition 0: {0, 1, 2, 3}
+// Partition 1: {4, 5, 6, 7}
+// Partition 2: {8, 9, 10, 11}
+// Partition 3: {12, 13, 14, 15}
+
+// The return value is each workgroup's permuted Id
+// In the above example:
+// linearedId 0's permuted Id is still 0
+// linearedId 1's permiuted Id is 4
+static Value chipletAwareWorkgroupReordering(Location loc, OpBuilder b,
+                                             Value linearizedId,
+                                             Value workgroupCountX,
+                                             Value workgroupCountY,
+                                             int64_t XCDParitionsOnGPU) {
+  Value numChipletsVal =
+      b.createOrFold<arith::ConstantIndexOp>(loc, XCDParitionsOnGPU);
+  Value workgroupCount =
+      b.create<arith::MulIOp>(loc, workgroupCountX, workgroupCountY);
+  Value workgroupCountPerChiplet =
+      b.create<arith::DivUIOp>(loc, workgroupCount, numChipletsVal);
+  Value chipletId = b.create<arith::RemUIOp>(loc, linearizedId, numChipletsVal);
+  Value wgIdWithinChiplet =
+      b.create<arith::DivUIOp>(loc, linearizedId, numChipletsVal);
+  Value reorderedId = b.create<arith::AddIOp>(
+      loc, wgIdWithinChiplet,
+      b.create<arith::MulIOp>(loc, chipletId, workgroupCountPerChiplet));
+
+  // Handle the remainder part.
+  Value constOne = b.createOrFold<arith::ConstantIndexOp>(loc, 1);
+  Value lastWorkgroupId =
+      b.create<arith::SubIOp>(loc, workgroupCount, constOne);
+  Value modulatedLastWorkgroupId = b.create<arith::SubIOp>(
+      loc, lastWorkgroupId,
+      b.create<arith::RemUIOp>(loc, workgroupCount, numChipletsVal));
+  Value isGreaterThanFinalWorkgroupId = b.create<arith::CmpIOp>(
+      loc, arith::CmpIPredicate::ugt, linearizedId, modulatedLastWorkgroupId);
+  Value finalId = b.create<arith::SelectOp>(loc, isGreaterThanFinalWorkgroupId,
+                                            linearizedId, reorderedId);
+
+  return finalId;
+}
+
+// Chiplet-aware workgroup reordering strategy: reordering + super-grouping.
+// Step 1: Reorder the workgroup grid to move slowly between
+// chiplet groups (Function: chipletAwareWorkgroupReordering).
+// Step 2: Implement 'super-grouping' of workgroups before switching to the next
+// column.
+static std::pair<Value, Value>
+makeChipletGroupedIds(Location loc, OpBuilder b, Value workgroupIdX,
+                      Value workgroupIdY, Value workgroupCountX,
+                      Value workgroupCountY, unsigned chipletGroupTile,
+                      unsigned numXCDs) {
+  // Create one dimension ID for workgroup.
+  Value linearized =
+      b.create<arith::MulIOp>(loc, workgroupIdY, workgroupCountX);
+  linearized = b.create<arith::AddIOp>(loc, linearized, workgroupIdX);
+
+  assert(numXCDs > 1);
+  // Map chiplets to perform a spatially local tile operation.
+  // Reorder the linearized ID such that every consecutive group of chiplets
+  // is the slowest-changing dimension in the grid.
+  // Emphircally found that two chiplets as a group has better locality
+  // throughout.
+  linearized = chipletAwareWorkgroupReordering(
+      loc, b, linearized, workgroupCountX, workgroupCountY, numXCDs / 2);
+
+  // Detailed explaination about the idea behind the below implementation:
+  // the L2 Cache Optimizations subsection in
+  // https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#
+  unsigned rowGroupSize = chipletGroupTile;
+  Value rowGroupSizeVal =
+      b.createOrFold<arith::ConstantIndexOp>(loc, rowGroupSize);
+
+  // Emphircally, found rowGroupSize=16 for mi300x achieves good performance
+  // group every 16 workgroups along Y dimension.
```

**Comment:**
```suggestion
  // Empirically, found rowGroupSize=16 for MI300X achieves good performance
  // group every 16 workgroups along Y dimension.
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp:154`

```diff
@@ -68,6 +68,119 @@ makeSwizzledIds(Location loc, OpBuilder b, Value workgroupIdX,
   return {swizzledIdX, swizzledIdY};
 }
 
+// Reoredering to make workgroup ids move slowly between chiplet groups.
+
+// The following example illustrates the concept behind this function:
+// Currently, the GPU launches workgroups in a round-robin fashion across
+// each XCD partition on the GPU.
+// Assume we have 16 workgroups and XCDPartitionsOnGPU is 4.
+// The default GPU schedule will launch workgroups {0, 1, 2, 3, ..., 15} in
+// the following round-robin fashion:
+// Partition 0: {0, 4, 8, 12}
+// Partition 1: {1, 5, 9, 13}
+// Partition 2: {2, 6, 10, 14}
+// Partition 3: {3, 7, 11, 15}
+
+// After reordering, the workgroup IDs are {0, 4, 8, 12, 1, ..., 15},
+// resulting in the round-robin launching fashion:
+// Partition 0: {0, 1, 2, 3}
+// Partition 1: {4, 5, 6, 7}
+// Partition 2: {8, 9, 10, 11}
+// Partition 3: {12, 13, 14, 15}
+
+// The return value is each workgroup's permuted Id
+// In the above example:
+// linearedId 0's permuted Id is still 0
+// linearedId 1's permiuted Id is 4
+static Value chipletAwareWorkgroupReordering(Location loc, OpBuilder b,
+                                             Value linearizedId,
+                                             Value workgroupCountX,
+                                             Value workgroupCountY,
+                                             int64_t XCDParitionsOnGPU) {
+  Value numChipletsVal =
+      b.createOrFold<arith::ConstantIndexOp>(loc, XCDParitionsOnGPU);
+  Value workgroupCount =
+      b.create<arith::MulIOp>(loc, workgroupCountX, workgroupCountY);
+  Value workgroupCountPerChiplet =
+      b.create<arith::DivUIOp>(loc, workgroupCount, numChipletsVal);
+  Value chipletId = b.create<arith::RemUIOp>(loc, linearizedId, numChipletsVal);
+  Value wgIdWithinChiplet =
+      b.create<arith::DivUIOp>(loc, linearizedId, numChipletsVal);
+  Value reorderedId = b.create<arith::AddIOp>(
+      loc, wgIdWithinChiplet,
+      b.create<arith::MulIOp>(loc, chipletId, workgroupCountPerChiplet));
+
+  // Handle the remainder part.
+  Value constOne = b.createOrFold<arith::ConstantIndexOp>(loc, 1);
+  Value lastWorkgroupId =
+      b.create<arith::SubIOp>(loc, workgroupCount, constOne);
+  Value modulatedLastWorkgroupId = b.create<arith::SubIOp>(
+      loc, lastWorkgroupId,
+      b.create<arith::RemUIOp>(loc, workgroupCount, numChipletsVal));
+  Value isGreaterThanFinalWorkgroupId = b.create<arith::CmpIOp>(
+      loc, arith::CmpIPredicate::ugt, linearizedId, modulatedLastWorkgroupId);
+  Value finalId = b.create<arith::SelectOp>(loc, isGreaterThanFinalWorkgroupId,
+                                            linearizedId, reorderedId);
+
+  return finalId;
+}
+
+// Chiplet-aware workgroup reordering strategy: reordering + super-grouping.
+// Step 1: Reorder the workgroup grid to move slowly between
+// chiplet groups (Function: chipletAwareWorkgroupReordering).
+// Step 2: Implement 'super-grouping' of workgroups before switching to the next
+// column.
+static std::pair<Value, Value>
+makeChipletGroupedIds(Location loc, OpBuilder b, Value workgroupIdX,
+                      Value workgroupIdY, Value workgroupCountX,
+                      Value workgroupCountY, unsigned chipletGroupTile,
+                      unsigned numXCDs) {
+  // Create one dimension ID for workgroup.
+  Value linearized =
+      b.create<arith::MulIOp>(loc, workgroupIdY, workgroupCountX);
+  linearized = b.create<arith::AddIOp>(loc, linearized, workgroupIdX);
+
+  assert(numXCDs > 1);
+  // Map chiplets to perform a spatially local tile operation.
+  // Reorder the linearized ID such that every consecutive group of chiplets
+  // is the slowest-changing dimension in the grid.
+  // Emphircally found that two chiplets as a group has better locality
+  // throughout.
+  linearized = chipletAwareWorkgroupReordering(
+      loc, b, linearized, workgroupCountX, workgroupCountY, numXCDs / 2);
+
+  // Detailed explaination about the idea behind the below implementation:
+  // the L2 Cache Optimizations subsection in
+  // https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#
```

**Comment:**
```suggestion
  // Map chiplets to perform a spatially local tile operation.
  // Reorder the linearized ID such that every consecutive group of chiplets
  // is the slowest-changing dimension in the grid.
  // Empirically found that two chiplets as a group has better locality
  // throughout.
  linearized = chipletAwareWorkgroupReordering(
      loc, b, linearized, workgroupCountX, workgroupCountY, numXCDs / 2);

  // Detailed explanation about the idea behind the below implementation:
  // the L2 Cache Optimizations subsection in
  // https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#
```

---

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.td:281`

```diff
@@ -269,6 +269,27 @@ def IREEGPU_MmaScheduleAttr : AttrDef<IREEGPU_Dialect, "MMASchedule"> {
   }];
 }
 
+//===----------------------------------------------------------------------===//
+// Workgroup Reordering Attr
+
+def IREEGPU_WorkGroupReorderAttr: AttrDef<IREEGPU_Dialect, "WorkgroupReorderOptions">{
+  let mnemonic = "reorder_workgroups";
+  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
+
+  string description = [{
+    Options for workgroup reordering strategies to improve L2 cache hit rate
+    and thus provide performance improvement.
```

**Comment:**
```suggestion
    Options for workgroup reordering strategies to improve L2 cache hit rate.
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp:147`

```diff
@@ -144,19 +144,19 @@ makeChipletGroupedIds(Location loc, OpBuilder b, Value workgroupIdX,
   // Map chiplets to perform a spatially local tile operation.
   // Reorder the linearized ID such that every consecutive group of chiplets
   // is the slowest-changing dimension in the grid.
-  // Emphircally found that two chiplets as a group has better locality
+  // Emphirically found that two chiplets as a group has better locality
```

**Comment:**
```suggestion
  // Empirically found that two chiplets as a group has better locality
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp:159`

```diff
@@ -144,19 +144,19 @@ makeChipletGroupedIds(Location loc, OpBuilder b, Value workgroupIdX,
   // Map chiplets to perform a spatially local tile operation.
   // Reorder the linearized ID such that every consecutive group of chiplets
   // is the slowest-changing dimension in the grid.
-  // Emphircally found that two chiplets as a group has better locality
+  // Emphirically found that two chiplets as a group has better locality
   // throughout.
   linearized = chipletAwareWorkgroupReordering(
       loc, b, linearized, workgroupCountX, workgroupCountY, numXCDs / 2);
 
-  // Detailed explaination about the idea behind the below implementation:
+  // Detailed explanation about the idea behind the below implementation:
   // the L2 Cache Optimizations subsection in
   // https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#
   unsigned rowGroupSize = chipletGroupTile;
   Value rowGroupSizeVal =
       b.createOrFold<arith::ConstantIndexOp>(loc, rowGroupSize);
 
-  // Emphircally, found rowGroupSize=16 for mi300x achieves good performance
+  // Emphirically, found rowGroupSize=16 for MI300X achieves good performance
```

**Comment:**
```suggestion
  // Empirically, found rowGroupSize=16 for MI300X achieves good performance
```

---

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp:110`

```diff
@@ -68,6 +68,95 @@ makeSwizzledIds(Location loc, OpBuilder b, Value workgroupIdX,
   return {swizzledIdX, swizzledIdY};
 }
 
+// reoredering to make workgroup ids move slowly between chiplet groups
+static Value chipletAwareWorkgroupReordering(Location loc, OpBuilder b,
+                                             Value linearizedId,
+                                             Value workgroupCountX,
+                                             Value workgroupCountY,
+                                             int64_t numChipletsPerGroup) {
+  Value numChipletsVal =
+      b.createOrFold<arith::ConstantIndexOp>(loc, numChipletsPerGroup);
+  Value workgroupCount =
+      b.create<arith::MulIOp>(loc, workgroupCountX, workgroupCountY);
+  Value workgroupCountPerChiplet =
+      b.create<arith::DivUIOp>(loc, workgroupCount, numChipletsVal);
+  Value chipletId = b.create<arith::RemUIOp>(loc, linearizedId, numChipletsVal);
+  Value wgIdWithinChiplet =
+      b.create<arith::DivUIOp>(loc, linearizedId, numChipletsVal);
+  Value reorderedId = b.create<arith::AddIOp>(
+      loc, wgIdWithinChiplet,
+      b.create<arith::MulIOp>(loc, chipletId, workgroupCountPerChiplet));
+
+  // The following code is used to handle the remainder part
+
+  Value constOne = b.createOrFold<arith::ConstantIndexOp>(loc, 1);
+  Value lastWorkgroupId =
+      b.create<arith::SubIOp>(loc, workgroupCount, constOne);
+  Value modulatedLastWorkgroupId = b.create<arith::SubIOp>(
+      loc, lastWorkgroupId,
+      b.create<arith::RemUIOp>(loc, workgroupCount, numChipletsVal));
+  Value isGreaterThanFinalWorkgroupId = b.create<arith::CmpIOp>(
+      loc, arith::CmpIPredicate::ugt, linearizedId, modulatedLastWorkgroupId);
+  linearizedId = b.create<arith::SelectOp>(loc, isGreaterThanFinalWorkgroupId,
+                                           linearizedId, reorderedId);
+
+  return linearizedId;
+}
+
+// Chiplet-aware workgroup reordering strategy: reordering + super-grouping.
+// Step 1: Reorder the workgroup grid to move slowly between
+// chiplet groups (Function: chipletAwareWorkgroupReordering).
+// Step 2: Implement 'super-grouping' of workgroups before switching to the next
+// column.
```

**Comment:**
Not addressed.

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp:223`

```diff
@@ -206,6 +217,12 @@ static ReorderWorkgroupsStrategy getReorderWorkgroupsStrategy(
   return option.value_or(clReorderWorkgroupsStrategy);
 }
 
+// Reconciles log2 of the workgroup reordering tile size based on the pipeline
+// `option` and the CLI flag.
+static unsigned
+getReorderWorkgroupsLogTileSize(const std::optional<int64_t> &option) {
```

**Comment:**
```suggestion
getReorderWorkgroupsLogTileSize(std::optional<int64_t> option) {
```

This is a very small type, we can pass by value

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp:224`

```diff
@@ -206,6 +217,12 @@ static ReorderWorkgroupsStrategy getReorderWorkgroupsStrategy(
   return option.value_or(clReorderWorkgroupsStrategy);
 }
 
+// Reconciles log2 of the workgroup reordering tile size based on the pipeline
+// `option` and the CLI flag.
+static unsigned
+getReorderWorkgroupsLogTileSize(const std::optional<int64_t> &option) {
+  return (unsigned)option.value_or(clReorderWorkgroupsLogTile);
```

**Comment:**
```suggestion
  int64_t logTile = option.value_or(clReorderWorkgroupsLogTile);
  assert(logTile >= 0);
  return static_cast<unsigned>(logTile);
```

---


---


## [PR #17645](https://github.com/iree-org/iree/pull/17645): Enable Workgroup Reordering Based on Translation Info Config Entries

### Review Summary

**CHANGES_REQUESTED** (2024-06-11)


**COMMENTED** (2024-06-12)


**COMMENTED** (2024-06-12)


**COMMENTED** (2024-06-12)

LGTM % nits. I will give it a try before approving.


**COMMENTED** (2024-06-12)


**APPROVED** (2024-06-12)

LGTM. I can see this improves performance on SDXL convs. Thanks for all the fixes.


### Code Comments

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp:101`

```diff
@@ -86,8 +86,26 @@ getPipelineOptions(FunctionOpInterface funcOp,
   if (DictionaryAttr config = translationInfo.getConfiguration()) {
     if (config.contains(LLVMGPUAttrNames::kNoReduceSharedMemoryBankConflicts))
       pipelineOptions.enableReduceSharedMemoryBankConflicts = false;
-    if (config.contains(LLVMGPUAttrNames::kNoReorderWorkgroups))
-      pipelineOptions.enableReorderWorkgroups = false;
+    if (config.contains(LLVMGPUAttrNames::kReorderWorkgroups)) {
+      pipelineOptions.enableReorderWorkgroups = true;
+      // Get the workgroups reorder config and enable the workgroup reordering
+      Attribute reorderGroupOption =
+          config.get(LLVMGPUAttrNames::kReorderWorkgroups);
+      assert(mlir::isa<mlir::StringAttr>(reorderGroupOption) &&
+             "reorder strategy should be a StringAttr");
+      StringRef reorderStr =
+          llvm::cast<StringAttr>(reorderGroupOption).getValue();
+      if (reorderStr == "transpose" || reorderStr == "Transpose") {
+        pipelineOptions.reorderOption =
+            LLVMGPUPipelineOptions::reorderWorkGroupOption::Transpose;
+      } else if (reorderStr == "swizzle" || reorderStr == "Swizzle") {
```

**Comment:**
I don't think we need to support both variants -- lowercase is fine.

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp:105`

```diff
@@ -86,8 +86,26 @@ getPipelineOptions(FunctionOpInterface funcOp,
   if (DictionaryAttr config = translationInfo.getConfiguration()) {
     if (config.contains(LLVMGPUAttrNames::kNoReduceSharedMemoryBankConflicts))
       pipelineOptions.enableReduceSharedMemoryBankConflicts = false;
-    if (config.contains(LLVMGPUAttrNames::kNoReorderWorkgroups))
-      pipelineOptions.enableReorderWorkgroups = false;
+    if (config.contains(LLVMGPUAttrNames::kReorderWorkgroups)) {
+      pipelineOptions.enableReorderWorkgroups = true;
+      // Get the workgroups reorder config and enable the workgroup reordering
+      Attribute reorderGroupOption =
+          config.get(LLVMGPUAttrNames::kReorderWorkgroups);
+      assert(mlir::isa<mlir::StringAttr>(reorderGroupOption) &&
+             "reorder strategy should be a StringAttr");
+      StringRef reorderStr =
+          llvm::cast<StringAttr>(reorderGroupOption).getValue();
+      if (reorderStr == "transpose" || reorderStr == "Transpose") {
+        pipelineOptions.reorderOption =
+            LLVMGPUPipelineOptions::reorderWorkGroupOption::Transpose;
+      } else if (reorderStr == "swizzle" || reorderStr == "Swizzle") {
+        pipelineOptions.reorderOption =
+            LLVMGPUPipelineOptions::reorderWorkGroupOption::Swizzle;
+      } else {
+        pipelineOptions.reorderOption =
```

**Comment:**
Here, should we check that the string is `none`? This is to diagnose cases when the value is misspelt.

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp:81`

```diff
@@ -77,7 +77,8 @@ static llvm::cl::opt<bool> clLLVMGPUEnablePrefetch(
 
 llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                               const LLVMGPUPipelineOptions &options) {
-  return os << "{" << "enableReduceSharedMemoryBankConflicts = "
+  return os << "{"
+            << "enableReduceSharedMemoryBankConflicts = "
```

**Comment:**
Why is this changed?

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.h:34`

```diff
@@ -25,15 +25,19 @@ namespace mlir::iree_compiler {
 /// attribute. These are used to override default pass heuristics at the
 /// function granularity.
 namespace LLVMGPUAttrNames {
-inline constexpr StringLiteral kNoReorderWorkgroups = "no_reorder_workgroups";
+inline constexpr StringLiteral kReorderWorkgroups = "reorder_workgroups";
 inline constexpr StringLiteral kNoReduceSharedMemoryBankConflicts =
     "no_reduce_shared_memory_bank_conflicts";
 } //  namespace LLVMGPUAttrNames
 
 struct LLVMGPUPipelineOptions {
+  enum reorderWorkGroupOption { None, Transpose, Swizzle };
```

**Comment:**
```suggestion
  enum ReorderWorkgroupsOption { None, Transpose, Swizzle };
```

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.h:37`

```diff
@@ -25,15 +25,19 @@ namespace mlir::iree_compiler {
 /// attribute. These are used to override default pass heuristics at the
 /// function granularity.
 namespace LLVMGPUAttrNames {
-inline constexpr StringLiteral kNoReorderWorkgroups = "no_reorder_workgroups";
+inline constexpr StringLiteral kReorderWorkgroups = "reorder_workgroups";
 inline constexpr StringLiteral kNoReduceSharedMemoryBankConflicts =
     "no_reduce_shared_memory_bank_conflicts";
 } //  namespace LLVMGPUAttrNames
 
 struct LLVMGPUPipelineOptions {
+  enum reorderWorkGroupOption { None, Transpose, Swizzle };
+
   bool enableReduceSharedMemoryBankConflicts = true;
-  bool enableReorderWorkgroups = true;
+  bool enableReorderWorkgroups = false;
```

**Comment:**
We don't need this anymore

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp:337`

```diff
@@ -322,10 +323,24 @@ void addGPUMatmulSimtPassPipeline(OpPassManager &funcPassManager,
   }
 
   if (options.enableReorderWorkgroups) {
+    ReorderWorkgrupsStrategy reorderStrategy;
+    if (options.reorderOption ==
+        LLVMGPUPipelineOptions::reorderWorkGroupOption::Transpose)
+      reorderStrategy = ReorderWorkgrupsStrategy::Transpose;
+    else if (options.reorderOption ==
+             LLVMGPUPipelineOptions::reorderWorkGroupOption::Swizzle)
+      reorderStrategy = ReorderWorkgrupsStrategy::Swizzle;
+    else
+      reorderStrategy = ReorderWorkgrupsStrategy::None;
+    funcPassManager.addPass(createReorderWorkgroups(
+        reorderStrategy, clReorderWorkgroupsLogSwizzleTile,
+        canReorderWorkgroups));
```

**Comment:**
This should probably go to a helper function

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp:83`

```diff
@@ -77,7 +77,8 @@ static llvm::cl::opt<bool> clLLVMGPUEnablePrefetch(
 
 llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                               const LLVMGPUPipelineOptions &options) {
-  return os << "{" << "enableReduceSharedMemoryBankConflicts = "
+  return os << "{"
+            << "enableReduceSharedMemoryBankConflicts = "
             << options.enableReduceSharedMemoryBankConflicts
             << ", enableReorderWorkgroups = " << options.enableReorderWorkgroups
```

**Comment:**
The enum value is missing, we should also print it.

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp:326`

```diff
@@ -322,10 +323,24 @@ void addGPUMatmulSimtPassPipeline(OpPassManager &funcPassManager,
   }
 
   if (options.enableReorderWorkgroups) {
+    ReorderWorkgrupsStrategy reorderStrategy;
```

**Comment:**
Can we use the same enum type and assign it directly?

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.h:34`

```diff
@@ -25,15 +25,19 @@ namespace mlir::iree_compiler {
 /// attribute. These are used to override default pass heuristics at the
 /// function granularity.
 namespace LLVMGPUAttrNames {
-inline constexpr StringLiteral kNoReorderWorkgroups = "no_reorder_workgroups";
+inline constexpr StringLiteral kReorderWorkgroups = "reorder_workgroups";
 inline constexpr StringLiteral kNoReduceSharedMemoryBankConflicts =
     "no_reduce_shared_memory_bank_conflicts";
 } //  namespace LLVMGPUAttrNames
 
 struct LLVMGPUPipelineOptions {
+  enum reorderWorkGroupOption { None, Transpose, Swizzle };
```

**Comment:**
We should be careful so that we can distinguish the case when this disables workgroup reordering (when enabled globally) and when it's not set. I think we can use `std::optional<ReorderWorkgroupsStrategy>`

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/config_user_vector_distribute.mlir:74`

```diff
@@ -71,7 +71,7 @@ hal.executable public @main_0_dispatch_0 {
 
 // CHECK:       #[[$TRANSLATION:.+]] = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64
 // CHECK-SAME:    mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
-// CHECK-SAME:    no_reorder_workgroups
+// CHECK-SAME:    reorder_workgroups = "transpose"
```

**Comment:**
I don't think this tests what it appears to since the attribute matches the global default. IMO we should maintain a test that disables globally-enables reordering (through the CLI flag) and then add a new test that enabled reordering when disabled globally.

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp:102`

```diff
@@ -87,23 +87,20 @@ getPipelineOptions(FunctionOpInterface funcOp,
     if (config.contains(LLVMGPUAttrNames::kNoReduceSharedMemoryBankConflicts))
       pipelineOptions.enableReduceSharedMemoryBankConflicts = false;
     if (config.contains(LLVMGPUAttrNames::kReorderWorkgroups)) {
-      pipelineOptions.enableReorderWorkgroups = true;
       // Get the workgroups reorder config and enable the workgroup reordering
       Attribute reorderGroupOption =
           config.get(LLVMGPUAttrNames::kReorderWorkgroups);
       assert(mlir::isa<mlir::StringAttr>(reorderGroupOption) &&
              "reorder strategy should be a StringAttr");
       StringRef reorderStr =
           llvm::cast<StringAttr>(reorderGroupOption).getValue();
-      if (reorderStr == "transpose" || reorderStr == "Transpose") {
-        pipelineOptions.reorderOption =
-            LLVMGPUPipelineOptions::reorderWorkGroupOption::Transpose;
-      } else if (reorderStr == "swizzle" || reorderStr == "Swizzle") {
-        pipelineOptions.reorderOption =
-            LLVMGPUPipelineOptions::reorderWorkGroupOption::Swizzle;
+      if (reorderStr == "transpose") {
+        pipelineOptions.reorderStrategy = ReorderWorkgrupsStrategy::Transpose;
+      } else if (reorderStr == "swizzle") {
+        pipelineOptions.reorderStrategy = ReorderWorkgrupsStrategy::Swizzle;
       } else {
-        pipelineOptions.reorderOption =
-            LLVMGPUPipelineOptions::reorderWorkGroupOption::None;
+        assert(reorderStr == "none" && "Unhandled reorder option");
```

**Comment:**
This should be an op error IMO -- the compiler shouldn't crash on invalid IR.

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp:91`

```diff
@@ -77,9 +77,28 @@ static llvm::cl::opt<bool> clLLVMGPUEnablePrefetch(
 
 llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                               const LLVMGPUPipelineOptions &options) {
+  if (options.reorderStrategy) {
+    StringRef reorderStr;
+    if (options.reorderStrategy.value() == ReorderWorkgrupsStrategy::Transpose)
+      reorderStr = "transpose";
+    else if (options.reorderStrategy.value() ==
+             ReorderWorkgrupsStrategy::Swizzle)
+      reorderStr = "swizzle";
+    else {
+      assert(options.reorderStrategy.value() ==
+                 ReorderWorkgrupsStrategy::None &&
+             "Unhandled reorder option");
+      reorderStr = "none";
+    }
+
+    return os << "{" << "enableReduceSharedMemoryBankConflicts = "
+              << options.enableReduceSharedMemoryBankConflicts
+              << ", ReorderWorkgroupsStrategy = " << reorderStr
+              << ", enableUkernels = " << options.enableUkernels << "}";
+  }
```

**Comment:**
We should always print the strategy here. If the value is `std::nullopt` we can print something like `<not set>`

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.h:35`

```diff
@@ -25,15 +26,19 @@ namespace mlir::iree_compiler {
 /// attribute. These are used to override default pass heuristics at the
 /// function granularity.
 namespace LLVMGPUAttrNames {
-inline constexpr StringLiteral kNoReorderWorkgroups = "no_reorder_workgroups";
+inline constexpr StringLiteral kReorderWorkgroups = "reorder_workgroups";
 inline constexpr StringLiteral kNoReduceSharedMemoryBankConflicts =
     "no_reduce_shared_memory_bank_conflicts";
 } //  namespace LLVMGPUAttrNames
 
 struct LLVMGPUPipelineOptions {
+  //   enum reorderWorkGroupsOption { None, Transpose, Swizzle };
```

**Comment:**
Remove this

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.h:40`

```diff
@@ -25,15 +26,19 @@ namespace mlir::iree_compiler {
 /// attribute. These are used to override default pass heuristics at the
 /// function granularity.
 namespace LLVMGPUAttrNames {
-inline constexpr StringLiteral kNoReorderWorkgroups = "no_reorder_workgroups";
+inline constexpr StringLiteral kReorderWorkgroups = "reorder_workgroups";
 inline constexpr StringLiteral kNoReduceSharedMemoryBankConflicts =
     "no_reduce_shared_memory_bank_conflicts";
 } //  namespace LLVMGPUAttrNames
 
 struct LLVMGPUPipelineOptions {
+  //   enum reorderWorkGroupsOption { None, Transpose, Swizzle };
+
   bool enableReduceSharedMemoryBankConflicts = true;
-  bool enableReorderWorkgroups = true;
   bool enableUkernels = false;
+
+  //   reorderWorkGroupOption reorderOption = None;
```

**Comment:**
Remove this

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp:93`

```diff
@@ -86,8 +86,23 @@ getPipelineOptions(FunctionOpInterface funcOp,
   if (DictionaryAttr config = translationInfo.getConfiguration()) {
     if (config.contains(LLVMGPUAttrNames::kNoReduceSharedMemoryBankConflicts))
       pipelineOptions.enableReduceSharedMemoryBankConflicts = false;
-    if (config.contains(LLVMGPUAttrNames::kNoReorderWorkgroups))
-      pipelineOptions.enableReorderWorkgroups = false;
+    if (config.contains(LLVMGPUAttrNames::kReorderWorkgroups)) {
+      // Get the workgroups reorder config and enable the workgroup reordering
+      Attribute reorderGroupOption =
+          config.get(LLVMGPUAttrNames::kReorderWorkgroups);
+      assert(mlir::isa<mlir::StringAttr>(reorderGroupOption) &&
```

**Comment:**
```suggestion
      assert(isa<StringAttr>(reorderGroupOption) &&
```

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp:96`

```diff
@@ -86,8 +86,23 @@ getPipelineOptions(FunctionOpInterface funcOp,
   if (DictionaryAttr config = translationInfo.getConfiguration()) {
     if (config.contains(LLVMGPUAttrNames::kNoReduceSharedMemoryBankConflicts))
       pipelineOptions.enableReduceSharedMemoryBankConflicts = false;
-    if (config.contains(LLVMGPUAttrNames::kNoReorderWorkgroups))
-      pipelineOptions.enableReorderWorkgroups = false;
+    if (config.contains(LLVMGPUAttrNames::kReorderWorkgroups)) {
+      // Get the workgroups reorder config and enable the workgroup reordering
+      Attribute reorderGroupOption =
+          config.get(LLVMGPUAttrNames::kReorderWorkgroups);
+      assert(mlir::isa<mlir::StringAttr>(reorderGroupOption) &&
+             "reorder strategy should be a StringAttr");
+      StringRef reorderStr =
+          llvm::cast<StringAttr>(reorderGroupOption).getValue();
```

**Comment:**
```suggestion
         cast<StringAttr>(reorderGroupOption).getValue();
```

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp:82`

```diff
@@ -77,9 +77,28 @@ static llvm::cl::opt<bool> clLLVMGPUEnablePrefetch(
 
 llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                               const LLVMGPUPipelineOptions &options) {
+  if (options.reorderStrategy) {
+    StringRef reorderStr;
+    if (options.reorderStrategy.value() == ReorderWorkgrupsStrategy::Transpose)
```

**Comment:**
nit: you can compare directly with `==`
```suggestion
    if (options.reorderStrategy == ReorderWorkgrupsStrategy::Transpose)
```

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp:88`

```diff
@@ -77,9 +77,28 @@ static llvm::cl::opt<bool> clLLVMGPUEnablePrefetch(
 
 llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                               const LLVMGPUPipelineOptions &options) {
+  if (options.reorderStrategy) {
+    StringRef reorderStr;
+    if (options.reorderStrategy.value() == ReorderWorkgrupsStrategy::Transpose)
+      reorderStr = "transpose";
+    else if (options.reorderStrategy.value() ==
+             ReorderWorkgrupsStrategy::Swizzle)
+      reorderStr = "swizzle";
+    else {
+      assert(options.reorderStrategy.value() ==
```

**Comment:**
nit: If one of the if-else branches has braces around it, the other ones should have them too

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/config_user_vector_distribute.mlir:5`

```diff
@@ -2,11 +2,20 @@
 // RUN:   --iree-codegen-reorder-workgroups-strategy=transpose \
 // RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target)))))" %s | FileCheck %s
 
+// A new test that disables the global CLI flag (--iree-codegen-reorder-workgroups-strategy) and checks that applying reorder_workgroups = "transpose" enables workgroup reordering.
```

**Comment:**
```suggestion
// Check that applying `reorder_workgroups = "transpose"` enables workgroup reordering.
```

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/config_user_vector_distribute.mlir:8`

```diff
@@ -2,11 +2,20 @@
 // RUN:   --iree-codegen-reorder-workgroups-strategy=transpose \
 // RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target)))))" %s | FileCheck %s
 
+// A new test that disables the global CLI flag (--iree-codegen-reorder-workgroups-strategy) and checks that applying reorder_workgroups = "transpose" enables workgroup reordering.
+
+// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --iree-codegen-llvmgpu-use-vector-distribution \
+// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target)))))" %s | FileCheck %s --check-prefix=RWP
```

**Comment:**
What does RWP stand for?

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/config_user_vector_distribute.mlir:100`

```diff
@@ -90,14 +110,24 @@ hal.executable public @main_0_dispatch_0 {
       // CHECK-LABEL: func.func @main_0_dispatch_0_matmul_transpose_b
       // CHECK:         memref.alloc() : memref<128x36xf16, #gpu.address_space<workgroup>>
       // CHECK:         memref.alloc() : memref<128x36xf16, #gpu.address_space<workgroup>>
-      // CHECK-DAG:     hal.interface.workgroup.id[1] : index
-      // CHECK-DAG:     hal.interface.workgroup.id[0] : index
-      // CHECK-NEXT:    scf.for
+      // CHECK-DAG:     %[[WG_Y:.+]] = hal.interface.workgroup.id[1] : index
+      // CHECK-DAG:     %[[WG_X:.+]] = hal.interface.workgroup.id[0] : index
+      // CHECK-DAG:     arith.muli %[[WG_Y]], %{{.+}} : index
+      // CHECK-DAG:     arith.addi %{{.+}}, %[[WG_X]] : index
+      // CHECK:         scf.for
 
+      // RWP-LABEL: func.func @main_0_dispatch_0_matmul_transpose_b
+      // RWP:         memref.alloc() : memref<128x36xf16, #gpu.address_space<workgroup>>
+      // RWP:         memref.alloc() : memref<128x36xf16, #gpu.address_space<workgroup>>
+      // RWP-DAG:     %[[WG_Y:.+]] = hal.interface.workgroup.id[1] : index
+      // RWP-DAG:     %[[WG_X:.+]] = hal.interface.workgroup.id[0] : index
+      // RWP-DAG:     arith.muli %[[WG_Y]], %{{.+}} : index
+      // RWP-DAG:     arith.addi %{{.+}}, %[[WG_X]] : index
+      // RWP:         scf.for  
       func.func @main_0_dispatch_0_matmul_transpose_b_2048x10240x1280_f16xf16xf32()
         attributes {translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [128, 2, 1] subgroup_size = 64, {
           mma_schedule = #iree_gpu.mma_schedule<intrinsic = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>, subgroup_m_count = 2, subgroup_n_count = 2>,
-          no_reorder_workgroups  // Disable the 'reorderWorkgroups' pass.
```

**Comment:**
We should maintain a test where global reordering is disabled through this attribute

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp:96`

```diff
@@ -86,8 +86,23 @@ getPipelineOptions(FunctionOpInterface funcOp,
   if (DictionaryAttr config = translationInfo.getConfiguration()) {
     if (config.contains(LLVMGPUAttrNames::kNoReduceSharedMemoryBankConflicts))
       pipelineOptions.enableReduceSharedMemoryBankConflicts = false;
-    if (config.contains(LLVMGPUAttrNames::kNoReorderWorkgroups))
-      pipelineOptions.enableReorderWorkgroups = false;
+    if (config.contains(LLVMGPUAttrNames::kReorderWorkgroups)) {
+      // Get the workgroups reorder config and enable the workgroup reordering
+      Attribute reorderGroupOption =
+          config.get(LLVMGPUAttrNames::kReorderWorkgroups);
+      assert(mlir::isa<mlir::StringAttr>(reorderGroupOption) &&
+             "reorder strategy should be a StringAttr");
+      StringRef reorderStr =
+          llvm::cast<StringAttr>(reorderGroupOption).getValue();
```

**Comment:**
not resolved

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp:103`

```diff
@@ -86,8 +86,25 @@ getPipelineOptions(FunctionOpInterface funcOp,
   if (DictionaryAttr config = translationInfo.getConfiguration()) {
     if (config.contains(LLVMGPUAttrNames::kNoReduceSharedMemoryBankConflicts))
       pipelineOptions.enableReduceSharedMemoryBankConflicts = false;
-    if (config.contains(LLVMGPUAttrNames::kNoReorderWorkgroups))
-      pipelineOptions.enableReorderWorkgroups = false;
+    if (config.contains(LLVMGPUAttrNames::kReorderWorkgroups)) {
+      // Get the workgroups reorder config and enable the workgroup reordering
+      Attribute reorderGroupOption =
+          config.get(LLVMGPUAttrNames::kReorderWorkgroups);
+      assert(isa<StringAttr>(reorderGroupOption) &&
+             "reorder strategy should be a StringAttr");
+      StringRef reorderStr =
+          llvm::cast<StringAttr>(reorderGroupOption).getValue();
+      if (reorderStr == "transpose") {
+        pipelineOptions.reorderStrategy = ReorderWorkgrupsStrategy::Transpose;
+      } else if (reorderStr == "swizzle") {
+        pipelineOptions.reorderStrategy = ReorderWorkgrupsStrategy::Swizzle;
+      } else {
+        if (reorderStr != "none")
+          funcOp.emitOpError("Unhandled reorder option");
```

**Comment:**
I don't remember the exact syntax, but this error can be made more helpful and precise:
```suggestion
          funcOp.emitOpError() << "Unknown " << LLVMGPUAttrNames::kReorderWorkgroups << "value: " << reorderGroupOpton;
```

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp:91`

```diff
@@ -86,8 +86,25 @@ getPipelineOptions(FunctionOpInterface funcOp,
   if (DictionaryAttr config = translationInfo.getConfiguration()) {
     if (config.contains(LLVMGPUAttrNames::kNoReduceSharedMemoryBankConflicts))
       pipelineOptions.enableReduceSharedMemoryBankConflicts = false;
-    if (config.contains(LLVMGPUAttrNames::kNoReorderWorkgroups))
-      pipelineOptions.enableReorderWorkgroups = false;
+    if (config.contains(LLVMGPUAttrNames::kReorderWorkgroups)) {
+      // Get the workgroups reorder config and enable the workgroup reordering
+      Attribute reorderGroupOption =
```

**Comment:**
```suggestion
      Attribute reorderWorkgroupOption =
```

---

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp:90`

```diff
@@ -86,8 +86,25 @@ getPipelineOptions(FunctionOpInterface funcOp,
   if (DictionaryAttr config = translationInfo.getConfiguration()) {
     if (config.contains(LLVMGPUAttrNames::kNoReduceSharedMemoryBankConflicts))
       pipelineOptions.enableReduceSharedMemoryBankConflicts = false;
-    if (config.contains(LLVMGPUAttrNames::kNoReorderWorkgroups))
-      pipelineOptions.enableReorderWorkgroups = false;
+    if (config.contains(LLVMGPUAttrNames::kReorderWorkgroups)) {
+      // Get the workgroups reorder config and enable the workgroup reordering
```

**Comment:**
```suggestion
      // Get the workgroups reorder config and enable the workgroup reordering.
```

---


---


## [PR #17539](https://github.com/iree-org/iree/pull/17539): Add support for Conv2D in new filter layout (Fhwc) : (NchwFchw => NhwcFhwc)

### Review Summary

**COMMENTED** (2024-06-05)


### Code Comments

**File:** `compiler/src/iree/compiler/Preprocessing/Common/ConvertConvToChannelsLast.cpp:486`

```diff
@@ -450,34 +476,53 @@ namespace {
 
 struct ConvertLinalgConvNchwFchw : OpRewritePattern<linalg::Conv2DNchwFchwOp> {
   using OpRewritePattern::OpRewritePattern;
-  ConvertLinalgConvNchwFchw(MLIRContext *context, PatternBenefit benefit = 2)
-      : OpRewritePattern<linalg::Conv2DNchwFchwOp>(context, benefit) {}
+  ConvertLinalgConvNchwFchw(MLIRContext *context, bool enableFHWC = false,
+                            PatternBenefit benefit = 2)
+      : OpRewritePattern<linalg::Conv2DNchwFchwOp>(context, benefit),
+        enableFHWCFilter(enableFHWC) {}
 
   LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                 PatternRewriter &rewriter) const override {
-    return transposeConvLikeLinalgOp(
-        rewriter, convOp, /*tilingFactor=*/-1,
-        namedConvBuilderFn<linalg::Conv2DNchwFchwOp, linalg::Conv2DNhwcHwcfOp>);
+    auto strides = convOp.getStrides();
```

**Comment:**
Don't use auto when the type is not obvious based on the RHS only.

---

**File:** `compiler/src/iree/compiler/Preprocessing/Common/ConvertConvToChannelsLast.cpp:495`

```diff
@@ -450,34 +476,53 @@ namespace {
 
 struct ConvertLinalgConvNchwFchw : OpRewritePattern<linalg::Conv2DNchwFchwOp> {
   using OpRewritePattern::OpRewritePattern;
-  ConvertLinalgConvNchwFchw(MLIRContext *context, PatternBenefit benefit = 2)
-      : OpRewritePattern<linalg::Conv2DNchwFchwOp>(context, benefit) {}
+  ConvertLinalgConvNchwFchw(MLIRContext *context, bool enableFHWC = false,
+                            PatternBenefit benefit = 2)
+      : OpRewritePattern<linalg::Conv2DNchwFchwOp>(context, benefit),
+        enableFHWCFilter(enableFHWC) {}
 
   LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                 PatternRewriter &rewriter) const override {
-    return transposeConvLikeLinalgOp(
-        rewriter, convOp, /*tilingFactor=*/-1,
-        namedConvBuilderFn<linalg::Conv2DNchwFchwOp, linalg::Conv2DNhwcHwcfOp>);
+    auto strides = convOp.getStrides();
+    bool hasAllOneStrides =
+        strides.isSplat() && strides.getSplatValue<int64_t>() == 1;
+    // Only enable this new filter layout when all strides are 1.
+    if (enableFHWCFilter && hasAllOneStrides) {
+      return transposeConvLikeLinalgOp(
+          rewriter, convOp, /*tilingFactor=*/-1, enableFHWCFilter,
+          namedConvBuilderFn<linalg::Conv2DNchwFchwOp,
+                             linalg::Conv2DNhwcFhwcOp>);
+    } else {
```

**Comment:**
no else after return: https://www.llvm.org/docs/CodingStandards.html#don-t-use-else-after-a-return

---


---
