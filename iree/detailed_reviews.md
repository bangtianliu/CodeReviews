# Code Review Comments

**Repository:** iree-org/iree

**Generated:** 2026-03-05

---

## PR #23440: [VectorExt] Add vectorization support for iree_linalg_ext.arg_compare

**URL:** https://github.com/iree-org/iree/pull/23440
**State:** MERGED

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 585

**Comment:**

> function 'vectorizeLinalgExtArgCompare' can be made static or moved into an anonymous namespace to enforce internal linkage

Is there a problem with clang-tidy? @kuhar is a known issue?

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 584

**Comment:**

Maybe remove this or make it less verbose since this is already documented in the header.

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 640

**Comment:**

 The input tensor includes the reduction dimension, which can be dynamic even when the output shape is static. Do we need to check the inputs, too?

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 624

**Comment:**

I think `GenericVectorization` ensures full tiles, but should we still add an explicit check for this and return failure when the output type doesn't match the vector sizes?

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 670

**Comment:**

Use vectorSizes as the single source of truth for vector dimensions:

```suggestion
    auto initVecTy =
        VectorType::get(vectorSizes, initTy.getElementType());
```

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 708

**Comment:**

```suggestion
  rewriter.eraseOp(oldYield);
```

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 699

**Comment:**

```suggestion
  rewriter.eraseBlock(&dstRegion.front());
```

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 707

**Comment:**

```suggestion
  IREE::VectorExt::YieldOp::create(rewriter, oldYield.getLoc(),
                                   oldYield.getOperands());
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 585

**Comment:**

yeah 

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GenericVectorization.cpp`

**Line:** 226

**Comment:**

This can be a typeswitch

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 588

**Comment:**

nit: type

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 592

**Comment:**

```suggestion
  auto inputValTy =
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 681

**Comment:**

can we zip_equal?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 689

**Comment:**

Use dyn_cast instead and then continue when it's null. See https://llvm.org/docs/CodingStandards.html#use-early-exits-and-continue-to-simplify-code and https://llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 691

**Comment:**

nit: type

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 707

**Comment:**

Can we zip_equal results and dps inits instead?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GenericVectorization.cpp`

**Line:** 211

**Comment:**

We use this style across llvm/mlir/iree:
```suggestion
        .Case([&](linalg::GenericOp genericOp) {
```
also elsewhere

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GenericVectorization.cpp`

**Line:** 216

**Comment:**

you can return early and drop the else: https://llvm.org/docs/CodingStandards.html#use-early-exits-and-continue-to-simplify-code

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 687

**Comment:**

hoist this outside of the if  and flip the condition so that it checks for null -- this will be more in-line with https://llvm.org/docs/CodingStandards.html#use-early-exits-and-continue-to-simplify-code

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 693

**Comment:**

type

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 670

**Comment:**

Doesn't this have to go through the rewriter?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 598

**Comment:**

we can check this higher up

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 670

**Comment:**

 The block is auto-created by `SingleBlockImplicitTerminator `after ArgCompareOp::create() returns, so the                                                                                                                                                                                                                                                                                  
rewriter never tracks it. Using `rewriter.eraseBlock()` would crash. 

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 670

**Comment:**

But the arg compare op is created through the rewriter. I'd expect that we have to start an in-place update and then clear it inside the lambda.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 669

**Comment:**

I think this comment is out of date now, no?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 669

**Comment:**

Yes, removed it.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/Transforms/VectorizeIREEVectorExtOps.cpp`

**Line:** 671

**Comment:**

The second update should be outside of this lambda, since it's done directly through the rewriter

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/generic_vectorization.mlir`

**Line:** 890

**Comment:**

```suggestion
                                      %out_val: tensor<4xf32>,
                                      %out_idx: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>) {
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/generic_vectorization.mlir`

**Line:** 930

**Comment:**

```suggestion
func.func @arg_compare_explicit_index(%partial_vals: tensor<4x32xf32>,
                                      %partial_idxs: tensor<4x32xi32>,
                                      %out_val: tensor<4xf32>,
                                      %out_idx: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>) {
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/generic_vectorization.mlir`

**Line:** 972

**Comment:**

```suggestion
func.func @arg_compare_with_index_base(%input: tensor<4x128xf32>,
                                       %out_val: tensor<4xf32>,
                                       %out_idx: tensor<4xi32>) -> (tensor<4xf32>, tensor<4xi32>) {
```

---

## PR #23386: [VectorExt] Add iree_vector_ext.arg_compare operation

**URL:** https://github.com/iree-org/iree/pull/23386
**State:** MERGED

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 275

**Comment:**

You need `SingleBlockImplicitTerminator<"::mlir::iree_compiler::IREE::LinalgExt::YieldOp">` or to manually verify.


Actually, this would require adding LinalgExt as a dependency which would be a bit strange. This might need a new terminator.

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/test/invalid.mlir`

**Line:** 96

**Comment:**

It's a bit strange to use an iree_linalg_ext op here.

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 315

**Comment:**

Can these be renamed since this isn't a DPS op. Maybe `acc` or `init` instead of outs?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 275

**Comment:**

Yeah, I added `IREEVectorExt_YieldOp` in the latest commit. 

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 332

**Comment:**

Would it be possible to have named inputs and inits instead of using variadic (with optional input). This may cut down some of the verification logic and reduce the number of `extraClassDeclaration`.

---

### Comment by qedawkins

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 285

**Comment:**

This comment is about the lowering, we don't need it as a part of the op's semantics description.

---

### Comment by qedawkins

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 290

**Comment:**

Can you expand on the semantics for the execution for the comparator region here? I'd assume there are restrictions on the kinds of operations allowed inside (only pure ops maybe?)

---

### Comment by qedawkins

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 324

**Comment:**

Can you describe what `index_base` means here without referencing the tensor version.

---

### Comment by qedawkins

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 332

**Comment:**

+1 lets definitely do that. The LinalgExt approach to having unnamed variadics for all operands is really poor.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp`

**Line:** 727

**Comment:**

```suggestion
  if (expectedShape != initValueShape) {
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp`

**Line:** 754

**Comment:**

Can we move shape verification to ODS?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp`

**Line:** 736

**Comment:**

same here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp`

**Line:** 747

**Comment:**

Also here -- can we move to ODS?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp`

**Line:** 773

**Comment:**

Can we move to ODS?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp`

**Line:** 824

**Comment:**

Can we move to ODS?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp`

**Line:** 831

**Comment:**

Can we move to ODS?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 253

**Comment:**

I think there's also another trait to check the parent op, something like `ParentOneOf`

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 276

**Comment:**

Is this always pure? For example, what if the comparator performs division?

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 283

**Comment:**

I think this is done just to be safe, but it should be possible to use the normal namespace style. e.g.
```suggestion
          "cast<VectorType>($input_value.getType()).getRank()">>,
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp`

**Line:** 763

**Comment:**

I wonder if yield should verify this itself -- how do other upstream ops handle this?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 303

**Comment:**

```suggestion
    CPred<"::llvm::isa<::mlir::IntegerType>("
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 305

**Comment:**

```suggestion
          "::mlir::llvm<::mlir::IndexType>("
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/test/invalid.mlir`

**Line:** 127

**Comment:**

```suggestion
  // expected-error @+1 {{failed to verify that init value rank must be input_rank - 1}}
```

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 387

**Comment:**

I think you can remove the cast. It should already be a VectorType.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp`

**Line:** 763

**Comment:**

Good catch! 

Mainly two approaches there:
  - `scf.yield`: No custom verifier,  parent ops verify everything
  - `linalg.yield`: has `hasVerifier = 1` and implements its own verification:
  
so that I can follow the approach from `linalg.yield` (closest to my use case here), I can move this verification to `iree_vector_ext.yield`.

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 405

**Comment:**

This should probably be asserted or return nullptr. Is this an ordering issue with the tablegen verification?

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 400

**Comment:**

Should `initType` be a VectorType?

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 306

**Comment:**

Not blocking, but I don't see the benefit of doing verification in the td file instead of in the verify method. It makes sense in cases where the checks are easily reusable (`SingleBlockImplicitTerminator`). This just seems more difficult to read, especially since there is a mix of c++ and table gen.

I'd be open to hearing the benefits of this approach. I could be missing something.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 306

**Comment:**

cc @kuhar here

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp`

**Line:** 763

**Comment:**

Ok here is linalg.yield 
```C++
LogicalResult linalg::YieldOp::verify() {
  auto *parentOp = (*this)->getParentOp();
  if (parentOp->getNumRegions() != 1 || parentOp->getRegion(0).empty())
    return emitOpError("expected single non-empty parent region");

  if (auto linalgOp = dyn_cast<LinalgOp>(parentOp))
    return verifyYield(*this, linalgOp);

  return emitOpError("expected parent op with LinalgOp interface");
}
```
 linalg.yield verification is split into two parts:
  - General (in YieldOp::verify()): Structural sanity checks (parent has regions, parent is correct type)
  - Specific (in verifyYield() helper): Parent-specific requirements (operand counts, types)
  
 so for my case, it is kind of similar. We have general cases for` iree_vector_ext.yield`,  `yield i1` is specific to ArgCompare Op.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 405

**Comment:**

Yes, it is about  TableGen verification ordering. 

` PredOpTrait<"dimension must be in range [0, input_rank)", ...>` runs before   `TypesMatchWith<"init value shape must match...", "ArgCompareOp::getExpectedInitType(...)">`.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp`

**Line:** 763

**Comment:**

This split into general and specific makes most sense to me

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td`

**Line:** 306

**Comment:**

I prefer using traits when available, if we have to write CPred and won’t be reusing them across multiple ops it probably makes more sense to keep in ::verify. I don’t remember which traits are generic and available here, so moving some of these back to .cpp may make more sense.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.cpp`

**Line:** 719

**Comment:**

Isn't this already checked in ODS? `ParentOneOf`

---

## PR #23320: Add C API support for --iree-codegen-tuning-spec-path flag

**URL:** https://github.com/iree-org/iree/pull/23320
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/CompilerDriver.cpp`

**Line:** 1016

**Comment:**

naming nit: these don't strictly need tuning specs, should we instead say these *support* tuning specs?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 69

**Comment:**

Relying on global variables seems like a major red flag. Is this even thread-safe? I'm thinking of having multiple compiler objects created by different threads, each setting its own options.

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/API/test/tuning_spec_flags_test.c`

**Line:** 1

**Comment:**

This test should probably be removed + it will fail if building without the rocm backend.

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/API/Internal/BUILD.bazel`

**Line:** 26

**Comment:**

This can be deleted

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/API/Internal/CompilerDriver.cpp`

**Line:** 44

**Comment:**

Same here

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Codegen/Common/Options.h`

**Line:** 1

**Comment:**

This file can be deleted

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/API/Internal/CompilerDriver.cpp`

**Line:** 1094

**Comment:**

This isn't a bad change, but given the current state of the PR it's a bit random. This PR shouldn't require any changes to `API/`.  I'm not sure if it should be kept or not.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/API/test/tuning_spec_flags_test.c`

**Line:** 1

**Comment:**

Then follow-up Q is that How can we add a test for this?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/API/test/tuning_spec_flags_test.c`

**Line:** 1

**Comment:**

Ok moved the test to `compiler/plugins/target/ROCM/test/`

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/test/tuning_spec_c_api_test.c`

**Line:** 14

**Comment:**

these seem unused
```suggestion
int main() {
```

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/test/tuning_spec_c_api_test.c`

**Line:** 21

**Comment:**

nit: there's only one flag, so no need for the array
```suggestion
  const char *flag = "--iree-codegen-tuning-spec-path=/tmp/spec.mlir";
  iree_compiler_error_t *err = ireeCompilerSessionSetFlags(session, 1, &flag);
  if (err) {
```

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/ROCMTarget.cpp`

**Line:** 96

**Comment:**

Maybe move it higher up closer to the other string options?

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/ROCMTarget.cpp`

**Line:** 187

**Comment:**

similar here

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/ROCMTarget.cpp`

**Line:** 189

**Comment:**

```suggestion
        cl::desc("Path to a module containing a tuning spec (transform "
```

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/ROCMTarget.cpp`

**Line:** 190

**Comment:**

Maybe it would be worth documenting if this needs to be a text file or if mlir bytecode is also accepted

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 173

**Comment:**

the `Base` type alias should be tablegened for us
```suggestion
struct MaterializeTuningSpecsPass final
: impl::MaterializeTuningSpecsPassBase<MaterializeTuningSpecsPass> {
  using Base::Base;
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 281

**Comment:**

nit: undo this?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/Passes.td`

**Line:** 798

**Comment:**

also here: explain if text/bytecode is accepted
```suggestion
           "Path to a module containing a tuning spec (transform dialect library).">,
```

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/test/BUILD.bazel`

**Line:** 19

**Comment:**

In the future we can add more c api tests specific to rocm, so I wouldn't make the name specific to the flag we are testing right now

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/test/tuning_spec_c_api_test.c`

**Line:** 7

**Comment:**


AFAICT, this test is only compiled and executed, but nothing checks that the 'PASSED` message is printed, right?

Why aren't we using gtest here?  We can implement C api tests in C++, that's completely fine.



---

### Comment by kuhar

**File:** `docs/website/docs/reference/tuning.md`

**Line:** 261

**Comment:**

```suggestion
    The `--iree-codegen-tuning-spec-path` flag is a part of `ROCMOptions` and
```

---

### Comment by kuhar

**File:** `docs/website/docs/reference/tuning.md`

**Line:** 262

**Comment:**

```suggestion
    only available for the ROCM/HIP backends. It can be set programmatically via
```

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/test/tuning_spec_c_api_test.c`

**Line:** 7

**Comment:**

For example, here you have unit tests for runtime code (IREE runtime is implemented in pure C): https://github.com/iree-org/iree/blob/main/runtime/src/iree/tokenizer/wordpiece_test.cc

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/test/rocm_c_api_test.cc`

**Line:** 28

**Comment:**

Maybe worth adding a second testcase that uses some non-existing flag? Just to make sure we this API produces errors as expected. If there already other tests for this, maybe worth adding a comment that explains it's tested elsewhere.

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/test/rocm_c_api_test.cc`

**Line:** 11

**Comment:**

```suggestion
#include "iree/testing/gtest.h"
```

---

### Comment by bangtianliu

**File:** `compiler/plugins/target/ROCM/test/rocm_c_api_test.cc`

**Line:** 11

**Comment:**

After applying this change, it causes bazel test error: https://github.com/iree-org/iree/actions/runs/21682118908/job/62519644825?pr=23320#step:5:1707

The `iree/testing/gtest.h` header is in the runtime directory (runtime/src/iree/testing/), so using it here would require a compiler to runtime dependency.

I checked other compiler unit tests like [compiler/src/iree/compiler/Utils/unittests/UtilsTest.cpp](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Utils/unittests/UtilsTest.cpp), [compiler/src/iree/compiler/Codegen/Dialect/Codegen/Utils/unittests/UtilsTest.cpp](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Codegen/Dialect/Codegen/Utils/unittests/UtilsTest.cpp), they all use <gtest/gtest.h> directly.

so I will still use `#include <gtest/gtest.h>`

---

## PR #23284: Set CMAKE_CXX_EXTENSIONS to OFF to align with LLVM

**URL:** https://github.com/iree-org/iree/pull/23284
**State:** MERGED

### Comment by kuhar

**File:** `CMakeLists.txt`

**Line:** 26

**Comment:**

```suggestion
set(CMAKE_CXX_EXTENSIONS OFF)

set(IREE_IDE_FOLDER IREE)
```

---

## PR #23218: [LinalgExt] Support and use arg_compare with explicit-index mode in split reduction

**URL:** https://github.com/iree-org/iree/pull/23218
**State:** MERGED

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1788

**Comment:**

Could just use `value_or` here:

```suggestion
    std::optional<int64_t> size = getConstantIntValue(initSizes[i])
    resultValShape.push_back(size.value_or(ShapedType::kDynamic));
    resultIdxShape.push_back(size.value_or(ShapedType::kDynamic));
```

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1862

**Comment:**

```suggestion
  int64_t reductionDim = reductionDims.front();
```

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/test/tiling.mlir`

**Line:** 2931

**Comment:**

nit: for readability some of these lines could be split up and use `CHECK-SAME:`

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/test/tiling.mlir`

**Line:** 2910

**Comment:**

Could you reduce some of the whitespace? I think its fine before the comments but it doesn't look needed here.

---

## PR #23193: [LinalgExt] Extend arg_compare tiling interface for explicit-index mode

**URL:** https://github.com/iree-org/iree/pull/23193
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1664

**Comment:**

```suggestion
    // Implicit-index mode: compute from induction variable.
```

? If that's not it, maybe it's worth renaming this function argument.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1671

**Comment:**

also here

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1658

**Comment:**

If we don't use index base when in explicit index mode, should we make it illegal in the verifier?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1658

**Comment:**

Absolutely, that is what I plan to do in the next PR. 

---

## PR #23153: [LinalgExt] Extend arg_compare to support both value and index provided ( explicit-index mode)

**URL:** https://github.com/iree-org/iree/pull/23153
**State:** MERGED

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td`

**Line:** 766

**Comment:**

Here's an alternative design that I think might be cleaner and more general. Instead of special-casing a second "index input," what if we treat all inputs uniformly? In the case of merge reduction, the arg compare op would receive the partial values and indices and treat them both as normal inputs:


```mlir
%0:3 = iree_linalg_ext.arg_compare dimension(1)
  ins(%partial_vals, %partial_idxs : tensor<2x4xf32>, tensor<2x4xi32>)
  outs(%max_val, %max_idx, %out_idx : tensor<2xf32>, tensor<2xi32>, tensor<2xi32>) {
  ^bb0(%val_a: f32, %idx_a: i32, %val_b: f32, %idx_b: i32):
    %cmp = arith.cmpf ogt, %val_a, %val_b : f32
    iree_linalg_ext.yield %cmp : i1
}
```

The third output (`%out_idx`) would be left unused because we don't care about the index into the partial reduction tensors. This approach seems more natural to me because in the merge step we are just applying an arg max over a tuple of values. Also, it might be easier to write code for this op since handling two different modes might be tricky. However, the trade off is extra block args and an unused result. What do you think?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td`

**Line:** 766

**Comment:**

  > what if we treat all inputs uniformly

 No, that's not the semantics of `arg_compare`. The inputs should be either a value array (implicit-index mode), or a value-index pair with matching shapes (explicit-index mode). The index input has special semantics as position tracking - it's not just another arbitrary input.
  
  I'd also like to clarify the semantics of the merge step. It's not about applying an argmax over a tuple of values. Rather, each tile produces a single (value, index) pair, and the merge step performs a pairwise reduction using the same predicate logic on the values. The indices are simply carried along - whichever value wins the comparison, its corresponding index is selected.

  Regarding the region: in our design, the comparator always takes 2 arguments (just values), keeping it uniform across both modes:
  ```mlir
  ^bb0(%val_a: f32, %val_b: f32):
    %cmp = arith.cmpf ogt, %val_a, %val_b : f32
    iree_linalg_ext.yield %cmp : i1
   ```
The region is just used to define predication logic. why would we need to expose indices to the comparator? For argmax/argmin, the comparison is purely value-based.
  
I think making it too general would blur its purpose and potentially complicate both the implementation and understanding.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 1254

**Comment:**

```suggestion
  if (numInputs == 1 || numInputs == 2) {
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 1253

**Comment:**

Maybe add this as an extra class declaration instead?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 1274

**Comment:**

Also here: maybe add this as a helper in ODS?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 1279

**Comment:**

does compatible == the same, or do we allow for some wiggle room?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 1293

**Comment:**

Can we define these through ODS?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 1308

**Comment:**

Also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 1259

**Comment:**

You already have a helper for this: `hasExplicitIndexInput`

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td`

**Line:** 859

**Comment:**

alternatively you could return nullptr when there's none and not assert

---

## PR #23102: [LinalgExt] Add OuterReduction tiling strategy for ArgCompareOp

**URL:** https://github.com/iree-org/iree/pull/23102
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1782

**Comment:**

You can also use matchers to get the constant value directly (`m_Constant`)

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1815

**Comment:**

```suggestion
    auto identityMap =
```
See https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1925

**Comment:**

Any change we can break this up? This function is getting really long and complicated

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1981

**Comment:**

You can do `for (auto [i, size, offset] : llvm::enumerate(sizes, offsets))` 

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1782

**Comment:**

You can use this method: [getConstantIntValue](https://github.com/llvm/llvm-project/blob/007f1af30eeb5604c04b0fd563af86c23dedbd5c/mlir/include/mlir/Dialect/Utils/StaticValueUtils.h#L120C24-L120C43) It does the matching for you.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1925

**Comment:**

Yes, I agree this function is getting large.

I'm currently working on revisiting the split-reduction (OuterParallel) tiling strategy in a separate PR, where I'll implement merge reduction using arg_compare with explicit index input.

After that PR lands, I'll come back to this one and refactor tileToPartialReduction to better handle both strategies with the explicit index model in mind. I'll likely split the OuterReduction and OuterParallel implementations into separate helper functions.


---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 2016

**Comment:**

This comment seems to have some code mixed with it.

---

## PR #23015: [LinalgExt] Generalize ArgCompareOp in GPUGeneralizeNamedOpsPass

**URL:** https://github.com/iree-org/iree/pull/23015
**State:** MERGED

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp`

**Line:** 1383

**Comment:**

Shouldn't we be using the pre-existing values from `iree_linalg_ext.arg_compare`'s outs operands?

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp`

**Line:** 1418

**Comment:**

Why can't we just inline `iree_linalg_ext.arg_compare`'s region here and give that value to an `arith.select` to choose either `%in` or `%out`. That way, this transformation doesn't need to worry about the contents of the region at all.


Using your test case "arg_compare_argmax_f32" as an example:

```
func.func @arg_compare_argmax_f32(%input: tensor<4x128xf32>,
                                   %out_val: tensor<4xf32>,
                                   %out_idx: tensor<4xi32>)
    -> (tensor<4xf32>, tensor<4xi32>) {
  %result:2 = iree_linalg_ext.arg_compare
      dimension(1)
      ins(%input : tensor<4x128xf32>)
      outs(%out_val, %out_idx : tensor<4xf32>, tensor<4xi32>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
      iree_linalg_ext.yield %cmp : i1
  } -> tensor<4xf32>, tensor<4xi32>
  return %result#0, %result#1 : tensor<4xf32>, tensor<4xi32>
}
```

This would produce the following
```
%cmp = arith.cmpf ogt, %in, %out : f32
%result = arith.select %cmp, %in, %out : f32
```

Instead of what is currently being generated:

```
%result = arith.maximumf %in, %out : f32
```



The `arith.cmpf` + `arith.select` -> `arith.maximumf` optimization seems like a transformation that should be owned by another pass. It looks like a straight forward pattern match, so I don't think we are losing information this way. But I could be missing something here, so let me know if there is a reason you choose not to do it this way.





---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp`

**Line:** 1418

**Comment:**

https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmaximumf-arithmaximumfop

>Returns the maximum of the two arguments, treating -0.0 as less than +0.0. If one of the arguments is NaN, then the result is also NaN.

Actually, `arith.cmpf` + `arith.select` isn't equal to `arith.maximumf` in the two cases mentioned above. Maybe this wouldn't be a legal canonicalization, then.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp`

**Line:** 1418

**Comment:**

Good question! The reason we generate arith.maximumf + arith.cmpf ogt instead of just inlining the region is that we want the decomposed linalg.generic to match the pattern expected by [isArgmaxOp](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp#L904)

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp`

**Line:** 1383

**Comment:**

Yes, we should use values from outs operands.

The reason behind that was also to match the requirement from isArgMaxOp: https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp#L931

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp`

**Line:** 1418

**Comment:**

+1 on just inlining the region of the arg_compare if possible. Is there a reason why the form that matches `isArgmaxOp` is preferred at this point?

My understanding was that the point of having the arg_compare op was so that we don't need to implement the complex matching logic that would be required. If we are decomposing the arg_compare operation, then we are intentionally giving up the matching convenience, so we shouldn't care about matching anymore at the point of decomposition. Also, the form that isArgmaxOp expects is already somewhat arbitrary, and if something changes in either that matcher or in this decomposition, then the connection here will be lost.

The reason I like inlining is that it saves a lot of complexity and boilerplate from the decomposition, since you'd be able to get rid of all the switch cases. Unless there is a good reason for keeping the form of the `isArgmaxOp` function, then I think we should go for inlining.

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/test/decompose_aggregate_op.mlir`

**Line:** 499

**Comment:**

If you go with the inlining idea, then I think you can also significantly reduce the number of tests, since there will be fewer functionally different cases.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp`

**Line:** 1418

**Comment:**

> https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmaximumf-arithmaximumfop
> 
> > Returns the maximum of the two arguments, treating -0.0 as less than +0.0. If one of the arguments is NaN, then the result is also NaN.
> 
> Actually, `arith.cmpf` + `arith.select` isn't equal to `arith.maximumf` in the two cases mentioned above. Maybe this wouldn't be a legal canonicalization, then.

Nobody cares about positive/negative zero in ML and most implementations don't respect it anyway: https://discourse.llvm.org/t/rfc-a-consistent-set-of-semantics-for-the-floating-point-minimum-and-maximum-operations/89006

This minor difference is safe to ignore.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp`

**Line:** 1418

**Comment:**

Thanks for your comments @IanWood1 @Max191 @kuhar, it makes sense very much.

I’m not defending the current approach, this discussion made me realize there’s a NaN-semantics concern.
`arith.maximumf` is NaN-propagating by definition. A lowering via `arith.cmpf + arith.select` is not equivalent: with ordered predicates (e.g., ogt), NaNs make the compare false, so whether NaN propagates becomes operand-order dependent (and ±0.0 handling differs too).

So the question is: do we want to preserve NaN-propagation semantics, or can we assume NaNs don’t appear (typical ML assumption)? If we can ignore NaNs, inlining the region is simpler.




---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp`

**Line:** 1418

**Comment:**

> +1 on just inlining the region of the arg_compare if possible. Is there a reason why the form that matches `isArgmaxOp` is preferred at this point?

The original intention behind this design choice was to first plumb through the argmax/argmin path along the VectorDistribute pipeline, then extend it to general cases. The existing `isArgMaxOp `is primarily used for ukernel support, so it's fine to inline the region here and implement a new matcher for my own purposes later if needed.



---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp`

**Line:** 1299

**Comment:**

This will create an illegal index cast if the output type is already `index`.

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/test/decompose_aggregate_op.mlir`

**Line:** 542

**Comment:**

Could you add a test for an op with `index_base` (check that it either doesn't get decomposed or that it is decomposed correctly)?

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp`

**Line:** 1317

**Comment:**

This doesn't handle the `index_base` arg. It should be easy to just create an `arith::AddIOP` to support it.

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/test/decompose_aggregate_op.mlir`

**Line:** 542

**Comment:**

Also, maybe a test for https://github.com/iree-org/iree/pull/23015#discussion_r2665733876 should be added too.

---

### Comment by krzysz00

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/AggregatedOpInterfaceImpl.cpp`

**Line:** 1418

**Comment:**

Re maximumf, I'd suggest taking a peek at what various backends of interest implement and wondering if we want `maxnumf` (or maybe it's spelled `maximumnumf`, I forget) instead

---

### Comment by Max191

**File:** `tests/external/iree-test-suites/torch_models/llama_8b_fp16/modules/scheduler_gfx942.json`

**Line:** 6

**Comment:**

Why is this changing to O3?

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/GeneralizeArgCompare.cpp`

**Line:** 1

**Comment:**

nit: Maybe change the filename to GeneralizeLinalgExtOps.cpp in case we have more ops that can be generalized like this in the future?

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUGeneralizeNamedOps.cpp`

**Line:** 13

**Comment:**

Can you also update the description in the pass docs to indicate that this generalizes LinalgExt ops too?

https://github.com/iree-org/iree/blob/b71b345821ff8e50d4122f94545d5aa68df5027e/compiler/src/iree/compiler/Codegen/Common/GPU/Passes.td#L156-L162

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/test/gpu_generalize_named_ops.mlir`

**Line:** 275

**Comment:**

I don't think we need this: index to index cast would fail to verify anyway

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUGeneralizeNamedOps.cpp`

**Line:** 79

**Comment:**

nit: spell out this type, per https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/GeneralizeArgCompare.cpp`

**Line:** 1

**Comment:**

this should be 2026 now

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h`

**Line:** 196

**Comment:**

```suggestion
/// ```mlir
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h`

**Line:** 206

**Comment:**

```suggestion
/// ```mlir
```

---

### Comment by kuhar

**File:** `tests/external/iree-test-suites/torch_models/llama_8b_fp16/modules/scheduler_gfx942.json`

**Line:** 6

**Comment:**

+1, this should be kept out

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/test/gpu_generalize_named_ops.mlir`

**Line:** 257

**Comment:**

Can we also have a testcase for at least one integer case?

---

### Comment by bangtianliu

**File:** `tests/external/iree-test-suites/torch_models/llama_8b_fp16/modules/scheduler_gfx942.json`

**Line:** 6

**Comment:**

For testing purpose (not belong to this PR), will undo it before landing. 

See [[CI][Torch]](https://github.com/iree-org/iree/pull/22379) for context, and discussion here: https://github.com/iree-org/iree/pull/22379#issuecomment-3437815189. Previously, we saw argmax compilation issues when using -O3.

I temporarily enabled -O3 to check whether the changes in the current PR introduce any compilation or numerical correctness regressions under that optimization level.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/GeneralizeArgCompare.cpp`

**Line:** 1

**Comment:**

yeah, now it is 2026, sign

---

## PR #23012: [DispatchCreation] Fold extract_slice of broadcast during split reduction tiling

**URL:** https://github.com/iree-org/iree/pull/23012
**State:** MERGED

### Comment by bangtianliu

**File:** `tests/external/iree-test-suites/torch_models/llama_8b_fp16/modules/scheduler_gfx942.json`

**Line:** 10

**Comment:**

This is mainly for testing purposes. I will remove this change from the current PR and land it in a separate one.

---

### Comment by MaheshRavishankar

**File:** `tests/external/iree-test-suites/torch_models/llama_8b_fp16/modules/scheduler_gfx942.json`

**Line:** 10

**Comment:**

Ok sounds good. Please make sure you land that separately.

---

## PR #22953: [Codegen] add FoldExtractSliceOfBroadcast pattern to TileAndDistributeToWorkgroups

**URL:** https://github.com/iree-org/iree/pull/22953
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/tile_and_distribute_workgroups_using_forall.mlir`

**Line:** 1380

**Comment:**

Would it be possible to turn these into function arguments to simplify the test?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/Transforms.cpp`

**Line:** 377

**Comment:**

```suggestion
  using Base::Base;
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/Transforms.cpp`

**Line:** 416

**Comment:**

```suggestion
    if (!llvm::all_of(offsets, isZeroInteger)) {
```
I think this aligns better with the match failure message.

(I realize this code is moved over, so fine to keep as-is in this PR)

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/Transforms.cpp`

**Line:** 429

**Comment:**

Similar here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/Transforms.cpp`

**Line:** 435

**Comment:**

`llvm::filter_vector`

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/Transforms.cpp`

**Line:** 377

**Comment:**

this used to use `Base::Base` but got dropped after the move for some reason

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/tile_and_distribute_workgroups_using_forall.mlir`

**Line:** 1384

**Comment:**

Could these also be function arguments? Do we care about the lowering config (I don't know)

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/test/tile_and_distribute_workgroups_using_forall.mlir`

**Line:** 1384

**Comment:**

Good catch! This test IR was derived from a real case where the race condition occurred and can certainly be simplified. Since this pattern (FoldExtractSliceOfBroadcast) doesn't specifically handle fill operations, the lowering_config isn't required. Will remove.

---

## PR #22883: Update shark/SHARK to amd-shark/AMD-SHARK in documentation and URLs in IREE

**URL:** https://github.com/iree-org/iree/pull/22883
**State:** MERGED

### Comment by ScottTodd

**File:** `.github/workflows/pkgci_test_riscv64.yml`

**Line:** 77

**Comment:**

Does this `amdsharkpublic` storage account exist? I can't access files like https://amdsharkpublic.blob.core.windows.net/amdsharkpublic/GCP-Migration-Files/toolchain_iree_manylinux_2_28_20231012.tar.gz

---

### Comment by ScottTodd

**File:** `compiler/plugins/target/ROCM/CMakeLists.txt`

**Line:** 86

**Comment:**

The `shark-infra` github organization has not been renamed [yet?], so this link now 404's

---

### Comment by ScottTodd

**File:** `docs/website/docs/developers/general/release-management.md`

**Line:** 45

**Comment:**

Also update the link text, not just the url

---

### Comment by ScottTodd

**File:** `docs/website/docs/developers/general/release-management.md`

**Line:** 190

**Comment:**

More link text that should be updated, not just url

---

### Comment by ScottTodd

**File:** `docs/website/docs/developers/general/release-management.md`

**Line:** 179

**Comment:**

404 for these pypi links
* https://pypi.org/project/amdsharktank
* https://pypi.org/project/amdshark-ai

---

### Comment by ScottTodd

**File:** `samples/colab/pytorch_huggingface_whisper.ipynb`

**Line:** 1

**Comment:**

more link text that should be updated in this file

---

### Comment by ScottTodd

**File:** `tests/external/iree-test-suites/sharktank_models/quality_tests/llama/8b_f16_decode_data_tiling_rocm.json`

**Line:** 21

**Comment:**

More 404s for this bucket

---

### Comment by bangtianliu

**File:** `tests/external/iree-test-suites/sharktank_models/quality_tests/llama/8b_f16_decode_data_tiling_rocm.json`

**Line:** 21

**Comment:**

Sure, it seems that all the links to data repo should not be updated, cc @pdhirajkumarprasad here. 

---

### Comment by kuhar

**File:** `samples/colab/pytorch_huggingface_whisper.ipynb`

**Line:** 2

**Comment:**

Can we update this doc without changing the structure so much?

---

### Comment by bangtianliu

**File:** `samples/colab/pytorch_huggingface_whisper.ipynb`

**Line:** 2

**Comment:**

I actually only updated the link there, but it seems to have rendered this way. I’ll double check.

---

### Comment by kuhar

**File:** `samples/colab/pytorch_huggingface_whisper.ipynb`

**Line:** 2

**Comment:**

You can edit this in a text editor to avoid surprises

---

## PR #22809: [python] use pytest for tests

**URL:** https://github.com/iree-org/iree/pull/22809
**State:** OPEN

### Comment by ScottTodd

**File:** `build_tools/cmake/iree_python.cmake`

**Line:** 111

**Comment:**

Have you evaluated https://python-cmake.github.io/pytest-cmake/ ?

---

### Comment by bangtianliu

**File:** `build_tools/cmake/iree_python.cmake`

**Line:** 111

**Comment:**

not yet

---

### Comment by bangtianliu

**File:** `build_tools/cmake/iree_python.cmake`

**Line:** 111

**Comment:**

Thanks for pointing this, I will see how to use pytest-cmake. 

---

## PR #22694: [DispatchCreation] Add FoldExtractSliceOfBroadcast Pattern

**URL:** https://github.com/iree-org/iree/pull/22694
**State:** MERGED

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/DispatchCreation/FormSplitReductionDispatches.cpp`

**Line:** 37

**Comment:**

Please update this based on our discussion that this is really hiding the race condition and there is a race condition somewhere in the lower levels of the stack. The tiling itself seems ok.

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/DispatchCreation/FormSplitReductionDispatches.cpp`

**Line:** 66

**Comment:**

Nit: It is useful to add a `rewriter.notifyMatchFailure()` here with a line that says why the match failed. That will be printed out in the debug log.

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/DispatchCreation/FormSplitReductionDispatches.cpp`

**Line:** 107

**Comment:**

You can just use `isOneInteger` here.

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/DispatchCreation/FormSplitReductionDispatches.cpp`

**Line:** 122

**Comment:**

You can use `isConstantIntValue` here.

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/DispatchCreation/FormSplitReductionDispatches.cpp`

**Line:** 93

**Comment:**

You can use `isConstantIntValue` here.

---

## PR #22683: [tuner][docs] update sharktuner readme

**URL:** https://github.com/iree-org/iree/pull/22683
**State:** MERGED

### Comment by kuhar

**File:** `docs/website/docs/reference/tuning.md`

**Line:** 308

**Comment:**

```suggestion
* IREE provides transform match operations (e.g.,
```

---

## PR #22598: [Codegen][Tuner] expose python binding for getIGEMMGenericConvDetails

**URL:** https://github.com/iree-org/iree/pull/22598
**State:** MERGED

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_codegen.h`

**Line:** 103

**Comment:**

What does this mean? Would it make sense to have a separate helper function for checking if IGEMM details can be queried for a given op in the first place?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp`

**Line:** 256

**Comment:**

Yeah, I think this would be better handled inside python bindings, and to assert here instead.

---

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_codegen.h`

**Line:** 103

**Comment:**

The rationale is that we generally prefer to invalid states not to be representable: https://geeklaunch.io/blog/make-invalid-states-unrepresentable/

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp`

**Line:** 256

**Comment:**

I can do it in this way as shown below:
```C
  ireeCodegenIGEMMGenericConvDetails result{};
  auto linalgOp = llvm::dyn_cast<mlir::linalg::LinalgOp>(unwrap(op));
  if (!linalgOp) {
    return result;
  }

  llvm::FailureOr<mlir::iree_compiler::IREE::LinalgExt::IGEMMGenericConvDetails>
      maybeDetails =
          mlir::iree_compiler::IREE::LinalgExt::getIGEMMGenericConvDetails(
              linalgOp);
  if (failed(maybeDetails)) {
    return result;
  }
```
Similar to : https://github.com/llvm/llvm-project/pull/135253/files#diff-77af7f1bca93ed6c9c25c48a91a73d279a6fac6f9b26c06ad5937258a475a96dR88-R96

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 33

**Comment:**

What does this do?

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/api/tuner_api_test.py`

**Line:** 305

**Comment:**

I think we should either switch these to pytest and get these prints for free or repeating the expected values in error messages

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp`

**Line:** 264

**Comment:**

This doesn't remove the invalid state, it just changes the representation.

My suggestion is to have a new public C API function that checks if something has igemm details, and separately a get function that asserts that the igemm details can be queried.

---

### Comment by bangtianliu

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 33

**Comment:**

mainly for the convenience of below C++ function:

```C++
static std::optional<ireeCodegenIGEMMGenericConvDetails>
GetIGEMMGenericConvDetails(MlirOperation op)
```

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 33

**Comment:**

I don't understand what it does

---

### Comment by bangtianliu

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 33

**Comment:**

just allow C++ code to use the C struct without `struct` keyword. But nvm, it will be fixed in the next commit. 

---

### Comment by bangtianliu

**File:** `compiler/bindings/python/test/api/tuner_api_test.py`

**Line:** 305

**Comment:**

Changed it to 
```
    assert details.igemm_loop_bounds == [
        96,
        3,
        96,
        98304,
    ], f"got {details.igemm_loop_bounds}"
```

The expected values are already visible in the assertion itself. 

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp`

**Line:** 260

**Comment:**

```suggestion
  return succeeded(mlir::iree_compiler::IREE::LinalgExt::getIGEMMGenericConvDetails(
              linalgOp));
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp`

**Line:** 283

**Comment:**

```suggestion
    return llvm::map_to_vector(
        vec, llvm::StaticCastTo<int64_t>);
```

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp`

**Line:** 283

**Comment:**

Thanks! good to know that we can write in this way!

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp`

**Line:** 283

**Comment:**

It landed in llvm a week ago

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/api/tuner_api_test.py`

**Line:** 195

**Comment:**

```suggestion
        not details.is_output_channel_first
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp`

**Line:** 279

**Comment:**

```suggestion
  auto toInt64 = [](ArrayRef<unsigned> vec) {
```
to avoid over-relying on the exact small vector size

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/api/tuner_api_test.py`

**Line:** 311

**Comment:**

You can make %1 a third function argument to make this test case more concise

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 728

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

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp`

**Line:** 298

**Comment:**

can you spell out this type? https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

## PR #22490: [Codegen][GPU] Generalize linalg.reduce operations

**URL:** https://github.com/iree-org/iree/pull/22490
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/TypePropagationPass.cpp`

**Line:** 434

**Comment:**

```suggestion
struct ReduceOpTypePropagation final
    : TypePropagationPattern<linalg::ReduceOp> {
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/TypePropagationPass.cpp`

**Line:** 435

**Comment:**

```suggestion
  using TypePropagationPattern::TypePropagationPattern;
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/TypePropagationPass.cpp`

**Line:** 441

**Comment:**

spell out types that are not obvious based on the immediate context: https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/TypePropagationPass.cpp`

**Line:** 443

**Comment:**

what if type conversion fails?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/TypePropagationPass.cpp`

**Line:** 459

**Comment:**

also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/TypePropagationPass.cpp`

**Line:** 497

**Comment:**

```suggestion
      Value valueToYield = yieldValue; // Default to original value.
```
https://llvm.org/docs/CodingStandards.html#commenting

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/Codegen/Common/TypePropagationPass.cpp`

**Line:** 453

**Comment:**

Hmm... This should actually already be checked by the conversion pattern driver. If you get here. it pretty much means that the conversion is needed.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/TypePropagationPass.cpp`

**Line:** 453

**Comment:**

Yes, correct.

---

## PR #22466: [DispatchCreation] Set split reduction size for ArgCompare

**URL:** https://github.com/iree-org/iree/pull/22466
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/DispatchCreation/test/set_split_reduction_sizes_outer_reduction.mlir`

**Line:** 188

**Comment:**

Maybe it would be also worth adding a test case for when split reduction is not applied because the reduction dimension is below the threshold?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/DispatchCreation/test/set_split_reduction_sizes_outer_reduction.mlir`

**Line:** 188

**Comment:**

Sure, will do!

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp`

**Line:** 486

**Comment:**

I don't follow split reduction work for a while, but why not make it configurable from CLI?

The other question, that may be resolved if we make it configurable, is that these methods look more like static methods to me. I don't know why we put them within the class.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp`

**Line:** 486

**Comment:**

Thanks for the feedback!
1. On making minSizeToSplit configurable: I just noticed we can simply reuse the existing `splitReductionTargetSize` option for this case. 
2. On refactoring to static methods: I agree with you. I can address this in a separate PR to keep the scope of this change focused.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp`

**Line:** 486

**Comment:**

You may want different option for different ops, but we can go with this for now. It is easy to extend.

> On refactoring to static methods: I agree with you. I can address this in a separate PR to keep the scope of this change focused.

My guess is that the original author wanted to use splitReductionTargetSize in the method, but we can pass the value to method as well. I'm +1 on moving them to static local methods. And yeah, let's address it in a separate PR.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp`

**Line:** 482

**Comment:**

Prefer `ShapedType::isDynamic(reductionSize)`. We also need a test for dynamic case.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp`

**Line:** 491

**Comment:**

optional: I'd drop blank lines because this is not a large function and they belong to the same block to me.

https://google.github.io/styleguide/cppguide.html#Vertical_Whitespace

> Use vertical whitespace sparingly; unnecessary blank lines make it harder to see overall code structure. Use blank lines only where they aid the reader in understanding the structure.
>
> Do not add blank lines where indentation already provides clear delineation, such as at the start or end of a code block. Do use blank lines to separate code into closely related chunks, analogous to paragraph breaks in prose. Within a statement or declaration, usually only insert line breaks to stay within the [line length limit](https://google.github.io/styleguide/cppguide.html#Line_Length), or to attach a comment to only part of the contents.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp`

**Line:** 491

**Comment:**

Sure, thanks for the link.

---

## PR #22409: [Codegen][Tuner] solve name conflicts for merging td specs

**URL:** https://github.com/iree-org/iree/pull/22409
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 189

**Comment:**

There's no need to build a string attribute here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 153

**Comment:**

Could we make it simpler by appending unique suffix only? Not sure how much value there is in having these prefixes since usually the names we append come from tiny modules with a single matcher only.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 159

**Comment:**

nit: You can turn this into a for loop since `suffix` is not used anywhere else

```suggestion
  for (unsigned suffix = 0; seenNames.contains(uniqueNewSpecName); ++suffix) {
    uniqueNewSpecName = llvm::formatv("{}_{}", specName, suffix).str();
  }
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 226

**Comment:**

We don't need the `if` statement above

---

## PR #22348: [Codegen][Tuner] Add root_op for matvec and reduction along VectorDistribute pipeline 

**URL:** https://github.com/iree-org/iree/pull/22348
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/config_root_op_attribute.mlir`

**Line:** 22

**Comment:**

Can we use linalg.matmul like in the test cases above? This should make the IR more concise.

---

## PR #22249: [Codegen] Update the td spec using the contraction matcher op

**URL:** https://github.com/iree-org/iree/pull/22249
**State:** MERGED

### Comment by kuhar

**File:** `tests/external/iree-test-suites/test_suite_files/attention_and_matmul_spec_punet_mi300.mlir`

**Line:** 9

**Comment:**

Can you undo this? These prints are useful for local debugging and it's nice to able to just uncomment them.

---

### Comment by kuhar

**File:** `tests/external/iree-test-suites/test_suite_files/attention_and_matmul_spec_punet_mi300.mlir`

**Line:** 18

**Comment:**

also here

---

### Comment by kuhar

**File:** `tests/external/iree-test-suites/test_suite_files/attention_and_matmul_spec_flux_mi300.mlir`

**Line:** 1

**Comment:**

How do you know this is unused?

---

### Comment by bangtianliu

**File:** `tests/external/iree-test-suites/test_suite_files/attention_and_matmul_spec_flux_mi300.mlir`

**Line:** 1

**Comment:**

Just search across iree, I do not see it is used. If you think we should keep it, I can keep it. 

---

### Comment by kuhar

**File:** `tests/external/iree-test-suites/test_suite_files/attention_and_matmul_spec_flux_mi300.mlir`

**Line:** 1

**Comment:**

I also checked iree-test-suites and I don't see it being used there either: https://github.com/search?q=repo%3Airee-org%2Firee-test-suites%20attention_and_matmul&type=code

---

## PR #22227: [python] Set up binding for preprocessing transform ops

**URL:** https://github.com/iree-org/iree/pull/22227
**State:** MERGED

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 650

**Comment:**

I don't understand why we have this test case -- I think it will parse correctly no matter how we set up python bindings, won't it? 

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 693

**Comment:**

Can we check that this contraction matcher has the expected indexing maps?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 239

**Comment:**

Can we also use this syntax in all tests and add an op example in `let description = ` above?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 239

**Comment:**

Here is my comment: https://github.com/iree-org/iree/pull/22227#issuecomment-3382224903

Should we do it in this PR? 

---

## PR #22199: [Codegen] add transform op for matching attention op

**URL:** https://github.com/iree-org/iree/pull/22199
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 377

**Comment:**

Can you also add an example?

---

## PR #22194: [Codegen] Add transform op for matching convolution ops

**URL:** https://github.com/iree-org/iree/pull/22194
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 318

**Comment:**

Don't print the whole op, this can be very verbose. We should also remove that from the matcher for contractions.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 323

**Comment:**

also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 283

**Comment:**

Can we pick an assembly format that doesn't print `!transform.param<i64>` 8 times? This was still OK with contractions but with convs it got very verbose... Maybe we can do:
`!transform.any_op -> !transform.param<i64>`
?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 245

**Comment:**

I don't think we need to print the op name either -- we already pass in the location

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 326

**Comment:**

Also here, we don't need the op name

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 371

**Comment:**

We should avoid copying vectors inside convDims: https://llvm.org/docs/CodingStandards.html#beware-unnecessary-copies-with-auto
```suggestion
  auto buildI64Attrs = [&builder](const auto& values, const auto& transform) {
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 368

**Comment:**

```suggestion
    return llvm::map_to_vector(values, [&](unsigned val) -> Attribute {
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 368

**Comment:**

ah no, this can be int64_t for dilations, disregard

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 245

**Comment:**

Solved in another separate PR specific for contraction op. 

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 377

**Comment:**

Can you hoist this lambda to a local variable? It's used 6 time total

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 366

**Comment:**

```suggestion
  Builder builder(getContext());
```
`ctx` is not used anywhere else AFAICT

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 277

**Comment:**

Can you also add an example?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 277

**Comment:**

Sure, will do.

---

## PR #22178: [Codegen][GPU] Disable MMA Intrinsics Sorting

**URL:** https://github.com/iree-org/iree/pull/22178
**State:** CLOSED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.cpp`

**Line:** 630

**Comment:**

We should also remove the sorting functions. Just commenting out the code will lead to unused private function warnings.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.cpp`

**Line:** 630

**Comment:**

Sure

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.cpp`

**Line:** 580

**Comment:**

The comparison function also needs to be removed

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.cpp`

**Line:** 580

**Comment:**

It is also used here:
https://github.com/iree-org/iree/blob/8e46a4f9191cc52d82a6c6672a0bb11e73dd49eb/compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.cpp#L831-L834



---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.cpp`

**Line:** 580

**Comment:**

So this function can not be removed

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.cpp`

**Line:** 580

**Comment:**

Gotcha. BTW, the code where this is used performs another unstable sorting -- I wonder if that's also asking for trouble

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.cpp`

**Line:** 580

**Comment:**

> Gotcha. BTW, the code where this is used performs another unstable sorting -- I wonder if that's also asking for trouble

It is used for Attention. so cc @Groverkss for awareness. FYI, Jakub sent PR https://github.com/iree-org/iree/pull/22141 to ensure consistent ordering across platforms.  

---

## PR #22154: [DispatchCreation] infer split-reduction sizes for ArgCompare

**URL:** https://github.com/iree-org/iree/pull/22154
**State:** MERGED

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp`

**Line:** 121

**Comment:**

```suggestion
    if (!maybeSizes) {
      return std::nullopt;
    }
```

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp`

**Line:** 32

**Comment:**

This might be overkill but you could consider implementing `LinalgFusionInterface` for `ArgCompareOp` since it will give you `getStaticLoopRanges`. Then there would be no need for the switch. Instead, it can just be done on the interface.

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp`

**Line:** 36

**Comment:**

`getStaticReductionDimSizes()` shouldn't be needed anymore right? This function (`getReductionDimSizes`) should handle linalg ops too.

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp`

**Line:** 53

**Comment:**

This is a bit confusing since it's indirectly checking that `op` implements the `TilingInterface`. These should always be the same size, right? So maybe you can just early return nullopt if `!tilingInterfaceOp`. Then the two if statements just get reduced to 1.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 1248

**Comment:**

nit: you can use brace initialization
```suggestion
  return {getInputType().getShape()};
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp`

**Line:** 45

**Comment:**

```suggestion
  SmallVector<utils::IteratorType> iters = tilingInterfaceOp.getLoopIteratorTypes();
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 1227

**Comment:**

nit: this is only used in one place, I'd inline it and use brace initialization
```suggestion
  return {b.getMultiDimIdentityMap(getInputRank())};
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 1244

**Comment:**

nit: you can return an initializer list
```suggestion
  AffineMap resultMap = AffineMap::get(rank, 0, proj, ctx);
  return {resultMap, resultMap};
```

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 1248

**Comment:**

`getInputType().getShape()` returns `ArrayRef<int64_t>`, SmallVector’s ArrayRef constructor is explicit, so return {getInputType().getShape()} can’t not be used here. 

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 1248

**Comment:**

But it can be used for other cases you pointed to, thanks. 

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 1224

**Comment:**

`getIndexingMapsForOperands()` should also include the dps inits. Then `getIndexingMapsForResults` can be very simple (see `AttentionOp::getIndexingMapsForResults`)

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp`

**Line:** 36

**Comment:**

The following `zip_equal` will throw an assert of the `IREE::LinalgExt::LinalgFusionOpInterface` is not implemented since `loopRanges` will be empty. Maybe return nullopt here too if it isn't implemented?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp`

**Line:** 36

**Comment:**

Would it be possible to have a test that covers this? 

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/DispatchCreation/SetSplitReductionSizes.cpp`

**Line:** 36

**Comment:**

I don't think we have any ops that implement `TilingInterface` and `PartialReductionOpInterface` but not `LinalgFusionOpInterface` currently.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 1224

**Comment:**

It seems that AttentionOp has one helper function `getIndexingMapsArray` there, I will implement that too. 

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 1224

**Comment:**

The Attention op gets its indexing maps from an indexing_maps attribute defined in TableGen: 
https://github.com/iree-org/iree/blob/2978fa3297acfe236b288526ebec4f96f27f5d55/compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td#L840.
For arg_compare, we don’t have that attribute, so we need to build up the maps.


---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 1248

**Comment:**

the alternative is to use `llvm::to_vector(someArrayRef)` -- you don't need the exact type. But fine to keep as-is.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 1235

**Comment:**

missing braces, see https://iree.dev/developers/general/contributing/#compiler

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 1225

**Comment:**

Move `b` closer to the first use and use the `ctx` variable

---

## PR #22149: [Codegen] support matching any values for dims_equal transform op

**URL:** https://github.com/iree-org/iree/pull/22149
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/Common/test/preprocessing_match_ops.mlir`

**Line:** 716

**Comment:**

Instead of having each test cases in a different split, could we have multiple ops to match in the same function? These tests are currently quite verbose.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 339

**Comment:**

It would be simpler to have a helper lambda/function that takes a range of attrs and returns something like `FailureOr<SmallVector<int64_t>>`. Then you don't have to repeat the same extraction logic for lhs and rhs and can do the comparison with a simple `!=`.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 339

**Comment:**

Or alternatively map to vector of `std::optional<int64_t>`, I think that's even simpler

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 357

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

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 357

**Comment:**

And then you can check for equality with https://github.com/llvm/llvm-project/blob/9ce0dae54e7d34ef4e0266069c0d3f1ae5968612/llvm/include/llvm/ADT/STLExtras.h#L2073-L2077:

```c++
if (!llvm::equal(actual, target, [](std::optional<int64_t> lhs, std::optional<int64_t> rhs) {
    return rhs == -1 || lhs == rhs;
  }) {
  return emitSilenceableError() << ...;
}

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 320

**Comment:**

nit: We could skip the size check as `llvm::equal` is going to do it anyway. I don't anticipate more than a few dims, so this should fit well within the small vector static size and not allocate any memory. But you can also keep as-is if you want.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 340

**Comment:**

Do we need to check for the `nullopt` state explicitly? I thought that `std::optional`'s comparison operators handle it like you would expect: https://en.cppreference.com/w/cpp/utility/optional/operator_cmp.html

I wrote a quick test and it seems to be the case: https://godbolt.org/z/hbfvbjK98

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 326

**Comment:**

If you don't want to deal with arbitrary values, you can also tighten the op definition in tablegen / move this to the verifier. That's probably a better way to go about this error mode. (This is a side comment, we don't have to fix it in this PR.)

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 326

**Comment:**

For example, you can make the target dims be `DenseI64ArrayAttr` instead of `ArrayAttr`. I don't think we can do anything about `currentDimAttrs` though.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 340

**Comment:**

Thanks, good to know about this.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 340

**Comment:**

This doesn't seem to have been addressed

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 315

**Comment:**

nit: since you this is called in `expected_values` in tablegen, I'd keep similar variables names in the code here. Then you can have `actualDimAttrs` and `expectedDims`, which is a common pairing in any test code that performs pairwise comparisons.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 340

**Comment:**

> This doesn't seem to have been addressed

It changes to comparison between `std::optional<int>` and `int `after using  DenseI64ArrayAttr. Should I use `optional<T> == T` directly (it is supported in C++20, not C++17)? 




---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 340

**Comment:**

It's supported in c++17 https://godbolt.org/z/WzMvh6eYP
<img width="820" height="145" alt="image" src="https://github.com/user-attachments/assets/f012f6b2-614e-48db-87b1-dd99b4f3c24c" />


---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 340

**Comment:**

Good, then I will go ahead and use it.


---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 340

**Comment:**

Thanks

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 327

**Comment:**

```suggestion
                     return rhs == -1 || lhs == rhs;
```

---

## PR #22137: [Codegen][GPU] Perfer MMA over VirtualMMA in Sorting. 

**URL:** https://github.com/iree-org/iree/pull/22137
**State:** CLOSED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.cpp`

**Line:** 579

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

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.cpp`

**Line:** 574

**Comment:**

An easier way to write this comparison is to come up with a feature tuple for `GPUInstrinsicType` and then delegate the exact comparison logic to the `std::tuple` comparison operator.

---

## PR #22090: [Codegen][GPU] Fix MMA Intrinsics Sorting

**URL:** https://github.com/iree-org/iree/pull/22090
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.h`

**Line:** 47

**Comment:**

Why not make `numHorizontallyFusedOps` a default constructor parameter to the existing consturctors?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.h`

**Line:** 40

**Comment:**

This is not super helpful IMO and the variable name in the comment doesn't match the code.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.h`

**Line:** 47

**Comment:**

Just wanted to do minor changes to be compatible with existing code. I can make it as default constructor parameter. 

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.h`

**Line:** 40

**Comment:**

Yeah, forgot to update it and will send a follow-up PR to fix it.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.h`

**Line:** 47

**Comment:**

I'm not super familiar with this code, does a default argument make it incompatible with the existing usage?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.h`

**Line:** 47

**Comment:**

nvm, I think I got your point. 

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.h`

**Line:** 47

**Comment:**

> I'm not super familiar with this code, does a default argument make it incompatible with the existing usage?

No, it should be fine. I will send a follow-up PR to apply your comment.

---

## PR #22040: [Codegen] Add transform op for matching dimension sizes.

**URL:** https://github.com/iree-org/iree/pull/22040
**State:** MERGED

### Comment by Max191

**File:** `compiler/src/iree/compiler/Preprocessing/Common/test/preprocessing_match_ops.mlir`

**Line:** 670

**Comment:**

What happens if the contraction op were to have multiple M/N/K dims?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Preprocessing/Common/test/preprocessing_match_ops.mlir`

**Line:** 670

**Comment:**

Good questions! I need to fix this PR. I want to keep size_equals simple, but I forgot to handle the case where %m could be an array.

In my current PR, I assumed %m could only be a single value, but I realize now that %m itself might be an array of parameters (like %m = [%M0, %M1, %M1]) according to previous PR #21981. I need to make this implementation concrete.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Preprocessing/Common/test/preprocessing_match_ops.mlir`

**Line:** 670

**Comment:**

I need to consider both two cases
```
%batch, [1024, 2048] // %batch stores one parameters
%batch [[1024], [2048]] // %batch stores one array of parameters
```
Current implementation supports first one, but the implementation should be flexible to support both.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 331

**Comment:**

How do you know you know each dim gets matched? For example, I think this would match:
expected: `[64, 64, 64]`
actual: `[64, 128, 256]`

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 243

**Comment:**

I don't understand why we need to support alternatives. What's the usecase?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 331

**Comment:**

OK, I see what's going on: https://github.com/iree-org/iree/pull/22040/files#r2370334264

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 331

**Comment:**

Here is the context for the design: https://github.com/iree-org/iree/pull/22040#discussion_r2362950731

At the end, I decide to use nested array to support each %batch/%m/%n/%k may have multiple ones.
Expected will be [[64], [64], [64]] for your example.
The logic correctly matches each dimension position-by-position. In your example:
expected: [[64], [64], [64]] (nested array - each sub-array corresponds to one position)
actual: [64, 128, 256]

The `zip_equal `pairs each actual dimension with its corresponding allowed values:
1. actual[0] = 64 vs expected[0] = [64] → match
2. actual[1] = 128 vs expected[1] = [64] → no match

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 243

**Comment:**

The main use cases seems to allowing flexibility in matching operations but acceptable dimension sizes. For example:
``` mlir
// Allow M dimension to be either 4096 or 2048
transform.iree.match.size_equals %op, %m, [[4096, 2048]] : !transform.any_op, !transform.param<i64>
```
 you could match both (m = 4096) and (m=2048) matmuls.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 331

**Comment:**

Here, it returns array (multiple batch/m/n/k dims).

https://github.com/iree-org/iree/blob/b73bb9a2a8cde5cef08e60812a0f59683bbe492a/compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp#L294-L301


---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 243

**Comment:**

but why do we need this?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 243

**Comment:**

Ok, this is the first version design where I wanted to provide some flexibility. If you think we don't need this, I can simplify it to exact matching instead of supporting multiple options.


---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 243

**Comment:**

Yeah I don't think we will have to emit this from the tuner. Later on we could potentially allow for merging of matchers, but I don't think we will need it any time soon. For the v0, I'd like to be able to replace DAG matching only.

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 230

**Comment:**

I don't think this needs to be `SingleOpMatcher`, since it doesn't seem to be using the operation handle.

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 250

**Comment:**

It looks like this handle is not used. Should it be removed?

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 250

**Comment:**

If you need the `Location`, then you could get it from the transform op itself (i.e., just call `getLoc()`).

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 332

**Comment:**

Could we replace this with `if(!llvm::equal(currentDimSizes, targetDimSizes))`? I don't think we need a precise diagnostics beyond that the two arrays don't match

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 241

**Comment:**

Can we use an actual op to show the syntax?

Something like:
```
 `transform.iree.match.size_equals %m, [512, 256]` matches when `%m` has two dims, 512 and 256.
```



---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 332

**Comment:**

Yes doable.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 322

**Comment:**

`llvm::equal` below already checks the sizes for you: https://en.cppreference.com/w/cpp/algorithm/equal.html#:~:text=of%20BinaryPredicate.-,Return%20value,-1%2D4)

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 241

**Comment:**

Code blocks render nicely on the website
```suggestion
    ```milr
    transform.iree.match.size_equals %m, [512, 256] : !transform.param<i64>
    ```

    This succeeds when `%m` has exactly two dimensions, 512 and 256.
```


---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 228

**Comment:**

I wonder if we should call it `iree.match.dims_equal` or something like that -- we are no longer taking a handle to the op as an argument, we are only looking at some of its dimension. I think sizes would imply we are looking at the overall size.

We already have two custom dim matching ops: https://github.com/iree-org/iree/blob/2c5b07daa588cd81c41e1ff19aa4130c7e41e6ce/compiler/src/iree/compiler/Preprocessing/Common/test/preprocessing_match_ops.mlir#L180-L181

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 228

**Comment:**

Another example: https://mlir.llvm.org/docs/Dialects/Transform/#transformmatchparamcmpi-transformmatchparamcmpiop

https://github.com/search?q=repo%3Allvm%2Fllvm-project%20transform.match.param.cmpi&type=code

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 11

**Comment:**

nit: Is this include needed?

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 262

**Comment:**

nit: Is this `extraClassDeclaration` needed?

---

## PR #22027: [Codegen][Tuner] update lowering config binding for subgroup basis

**URL:** https://github.com/iree-org/iree/pull/22027
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 378

**Comment:**

Do we know what's the cost of creating the builder just to create some attributes? I know we've been doing that for a while in this file, but this stood out for me just now. No need to change this here, just something to check separately.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 378

**Comment:**

I've just checked and it's cheap: https://github.com/llvm/llvm-project/blob/1a172b9924948f10f1bd3db07a83fe5e884f7b64/mlir/include/mlir/IR/Builders.h#L51-L55

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 378

**Comment:**

Thanks for checking this, and good to know about it!

---

## PR #21981: [Codegen] Add transform ops for matching contraction ops

**URL:** https://github.com/iree-org/iree/pull/21981
**State:** MERGED

### Comment by Max191

**File:** `compiler/src/iree/compiler/Preprocessing/Common/test/preprocessing_match_ops.mlir`

**Line:** 498

**Comment:**

Maybe combine this test with the one for `Verify that operations with exact same matching indexing maps are matched correctly.`?

You could put both matmul ops in the same func.func, but make one of them regular matmul, and the other transpose_b.

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 196

**Comment:**

Maybe just make this `element_types`, and have it provide `[LHS, RHS, ACC]` to reduce the number of named fields?

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 203

**Comment:**

I think this supports multiple m/n/k/batch dims right? Maybe update the naming to `batch_dims/m_dims/n_dims/k_dims` to make it more clear that there can be multiple? It's hard to tell unless you look at the implementation.

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Preprocessing/Common/test/preprocessing_match_ops.mlir`

**Line:** 274

**Comment:**

Could you add a test where the M/N/K/Batch dims are used in the matcher (ideally with multiple M/N/K/B dims)? You could use some matcher ops like either of these (maybe the first one, not sure if the second ones will work for params):
https://github.com/iree-org/iree/blob/9ad9f06bd5f03bb25ff65052f9465e03bcf43044/samples/custom_dispatch/vulkan/shaders/example_transform_spec.mlir#L67
https://github.com/iree-org/iree/blob/9ad9f06bd5f03bb25ff65052f9465e03bcf43044/compiler/src/iree/compiler/Preprocessing/Common/test/preprocessing_match_ops.mlir#L180-L181

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 196

**Comment:**

Actually, nvm, I forgot we want these to be optional. Maybe we should split up the LHS and RHS types into 2 separate optional fields then for more flexibility in matching?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 205

**Comment:**

Should we make these non-optional? I don't think we have any use for matching contraction ops with arbitrary types

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 202

**Comment:**

Why not define a few variants like `NN`, `NT`, etc., like we discussed before? I'm not saying indexing maps are a bad option, but these are quite verbose and I'd like to understand what motivates this extra complexity.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 256

**Comment:**

Use `zip_equal` for ranges of equal lengths. See https://llvm.org/docs/ProgrammersManual.html#iterating-over-ranges

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 248

**Comment:**

Why not compare two array attributes? I don't think there's value in producing verbose silenceable failures -- nobody looks at them

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 236

**Comment:**

Since the `current->getLoc()` expression appears often in this function, I'd hoist it to a local variable

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 317

**Comment:**

Use `map_to_vector`

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 202

**Comment:**

You can also see how we defined these in named matmul ops in linalg: https://github.com/llvm/llvm-project/blob/105fc90b6b96e0edb7529062fcba513a3a347820/mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp#L4053-L4076

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 205

**Comment:**

Sure, I can make it non-optional. 

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 202

**Comment:**

Yeah, I did think about it. The only concern is NN, NT stuff are quite matmul-specific, implicitly assuming the number of dimension is 3 (correct me if my understanding is wrong). Here I am still trying to provide a general and flexible support for matching contractions. 
So the thinking is first implement it based on indexing maps and then add NN, NT stuff as shortcuts for matching matmul based on indexing maps (missing, but I can add in the current PR). 

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 202

**Comment:**

I think the main criteria has to be that we should be able to easily emit these matchers inside the tuner, given a linalg.generic. In this setting, getting indexing maps is easier than deciding the matmul variant. Let's leave this as-is then -- I think we don't need these `NN`/`NT`/etc. variants after all.

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td`

**Line:** 202

**Comment:**

Yeah, I think for the tuner it's actually easier to just set the indexing maps. The `NN`/`NT`/etc variants would be more for readability of tuning specs + if anyone wants to hand-write them, which seems like low priority to me. We have the option if we want it, but I like having indexing_maps.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensions.cpp`

**Line:** 232

**Comment:**

This function is unused now

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/Common/test/preprocessing_match_ops.mlir`

**Line:** 447

**Comment:**

Can you add a testcase that shows if we match contractions that already have some attribute?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Preprocessing/Common/test/preprocessing_match_ops.mlir`

**Line:** 447

**Comment:**

Good point!

---

## PR #21977: [NFC] remove unused header files

**URL:** https://github.com/iree-org/iree/pull/21977
**State:** MERGED

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Codegen/Common/ConvertBf16ToUInt16Buffers.cpp`

**Line:** 14

**Comment:**

I'm not sure if it is expected because all the passes include the header. If we're going to remove it, we should be able to remove all of them?

There might be missing includes (about dialect) in the passes' implementation because sometimes the `Passes.h` include the headers for them.

https://github.com/search?q=repo%3Airee-org%2Firee+%22iree%2Fcompiler%2FCodegen%2FCommon%2FPasses.h%22+path%3A%2F%5Ecompiler%5C%2Fsrc%5C%2Firee%5C%2Fcompiler%5C%2FCodegen%5C%2FCommon%5C%2F%2F&type=code

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/ConvertBf16ToUInt16Buffers.cpp`

**Line:** 14

**Comment:**

I will try locally first.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/ConvertBf16ToUInt16Buffers.cpp`

**Line:** 14

**Comment:**

most of "iree/compiler/Codegen/Common/Passes.h" are dropped in the new commit except some necessary ones (if I remove, it will cause compilation errors). @hanhanW  

command: `find . -type f -name "*.cpp" -exec sed -i '/#include "iree\/compiler\/Codegen\/Common\/Passes\.h"/d' {} +`

---

## PR #21903: [DispatchCreation]: Add FormSplitReductionDispatchesPass support for ArgCompare op

**URL:** https://github.com/iree-org/iree/pull/21903
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1458

**Comment:**

These are deprecated, use free create functions instead: https://discourse.llvm.org/t/psa-opty-create-now-with-100-more-tab-complete/87339

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1468

**Comment:**

use `llvm::to_vector_of<int64_t>`

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1532

**Comment:**

missing braces

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1536

**Comment:**

```suggestion
    if (auto sizeAttr = dyn_cast<IntegerAttr>(initSizes[i])) {
      int64_t size = sizeAttr.getInt();
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1543

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

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1566

**Comment:**

also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1572

**Comment:**

This can be an array of exactly two elements

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1581

**Comment:**

also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1603

**Comment:**

use `llvm::to_vector`

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1560

**Comment:**

also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1586

**Comment:**

Use the `/*argName=*/ paramName` format. Clang tidy and other linters can check that the names are up to date

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/DispatchCreation/test/form_split_reduction_dispatches.mlir`

**Line:** 161

**Comment:**

```suggestion
    ^bb0(%in: f32, %out_val: f32):  // Only 2 arguments: input and output value.
```

See https://llvm.org/docs/CodingStandards.html#commenting

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1507

**Comment:**

nit reflow this comment

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1454

**Comment:**

```suggestion
  SmallVector<OpFoldResult> partialResultShape(sizes);
```

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1536

**Comment:**

 We can not directly use `dyn_cast<IntegerAttr>(initSizes[i])` here because initSizes[i] is of type `OpFoldResult`, which is a union of Attribute and Value (`PointerUnion<Attribute, Value>`).

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1466

**Comment:**

```suggestion
  auto broadcastDims =
      llvm::to_vector_of<int64_t>(reductionDims);
```
See https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1519

**Comment:**

Do these have to be const references?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1600

**Comment:**

```suggestion
  auto mergeReductionDims =
      llvm::to_vector_of<int64_t>(reductionDims);
```
See https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1580

**Comment:**

This is deprecated

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1483

**Comment:**

Just curious, is there a reason why this is an assert instead of returning failure?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1483

**Comment:**

Thanks, it is better to return failure. 

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/DispatchCreation/test/form_split_reduction_dispatches.mlir`

**Line:** 140

**Comment:**

Shouldn't this test be in `LinalgExt`? I think `form_split_reduction_dispatches.mlir` is meant to test the logic of `FormSplitReductionDispatchesPass` and not implementations of the tiling interface.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1540

**Comment:**

Since this is the only use of these two types, you can query them here where they are needed

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/DispatchCreation/test/form_split_reduction_dispatches.mlir`

**Line:** 140

**Comment:**

I think the transform dialect op here can be used to test: https://github.com/llvm/llvm-project/blob/2a2296b1aab4614bf6c95c3003000832c9d43de5/mlir/test/Dialect/Linalg/transform-op-split-reduction.mlir#L4

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/DispatchCreation/test/form_split_reduction_dispatches.mlir`

**Line:** 140

**Comment:**

Sure, I can move the tests to: https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/test/tiling.mlir.


---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1471

**Comment:**

I am not sure why we need the `broadcast`. We have never needed for other use cases, so wondering what I am missing. Hard to say from the tests here cause the tests does not have the slices that are extracted from this broadcast, but this seems like a pessimization. Can you post the actual IR (or update the tests).

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1471

**Comment:**

Thanks for asking about the broadcast operation. Let me explain the reasoning and I’m happy to revisit if there’s a better approach.

We run arg_compare on tensor<64x4096xf32> and tile the reduction dimension by 128, yielding 4096 / 128 = 32 tiles in scf.forall. Unlike standard reductions, arg_compare has no fixed neutral element—the identity depends on the comparator region. Per comment https://github.com/iree-org/iree/pull/21138#discussion_r2162706779, we assume the outputs already carry the per-row initial value/index:

- init_value : tensor<64xf32>
- init_index : tensor<64xi32> 

To seed split-k correctly, we broadcast these to the tiled partial-result shapes so each tile gets its own initialization, e.g.:
- init_value → tensor<64x32xf32>
- init_index → tensor<64x32xi32>

This ensures each parallel tile accumulates into a disjoint chunk with the correct per-tile start state, rather than sharing a single set of inits.

Here is the link to the example input and output mlir: https://gist.github.com/bangtianliu/34406f8491c4187a34b33c546713a670
(In this case, fusion sinks the broadcast inside scf.forall.)

Of course, I’m open to a more efficient alternative if we can materialize per-tile initialization without the explicit broadcast.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/DispatchCreation/test/form_split_reduction_dispatches.mlir`

**Line:** 140

**Comment:**

> I think the transform dialect op here can be used to test: https://github.com/llvm/llvm-project/blob/2a2296b1aab4614bf6c95c3003000832c9d43de5/mlir/test/Dialect/Linalg/transform-op-split-reduction.mlir#L4

We should use `transform.structured.tile_reduction_using_forall` for this, though I still need to learn how to apply it.

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1471

**Comment:**

My question is I see 

```
%broadcasted = linalg.broadcast ins(%2 : tensor<64xf32>) outs(%extracted_slice_0 : tensor<64x1xf32>) dimensions = [1] 
%extracted_slice_1 = tensor.extract_slice %broadcasted[0, 0] [64, 1] [1, 1] : tensor<64x1xf32> to tensor<64xf32>
```

It seems like `%extracted_slice_1` is same as `%broadcasted`. Similarly

```
%broadcasted_3 = linalg.broadcast ins(%3 : tensor<64xi32>) outs(%extracted_slice_2 : tensor<64x1xi32>) dimensions = [1] 
%extracted_slice_4 = tensor.extract_slice %broadcasted_3[0, 0] [64, 1] [1, 1] : tensor<64x1xi32> to tensor<64xi32>
```

`%extract_slice_4` seems to be same as `%3`. So we dont need the broadcast and extract_slice, correct?



---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1471

**Comment:**

Ok, I see what is happening here. I think `broadcast` might work. its kind a neat that it does. For Linalg ops (here : https://github.com/llvm/llvm-project/blob/f3b7ad4859037254afc0ee6d938018376c7c03d3/mlir/lib/Dialect/Linalg/Transforms/TilingInterfaceImpl.cpp#L521) I ended up getting an identity element and creating a new `FillOp`. Here your issue is where to get the identity element from. After thinking about this, I think broadcast this way is a pretty good solution if you can enforce that it will always get fused into the created `scf.forall` loop. So this makes sense! Thanks for explaining.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/DispatchCreation/test/form_split_reduction_dispatches.mlir`

**Line:** 140

**Comment:**

> We should use `transform.structured.tile_reduction_using_forall` for this, though I still need to learn how to apply it.

I'd like to provide some context on why transform dialect op(`transform.structured.tile_reduction_using_forall`) cannot be used now.

Previous PR https://github.com/iree-org/iree/pull/19684 implements PartialReductionOpInterface for `LinalgExt::OnlineAttentionOp` using the `PartialReductionOuterReduction `strategy in the tiling interface, which enables the use of `transform.structured.tile_reduction_using_for` (whose default strategy is `PartialReductionOuterReduction`, refer to https://github.com/llvm/llvm-project/blob/127d77d279de6547324c87f393164c604297c4d4/mlir/lib/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp#L2987). However, this is not split-k reduction, as explained here: https://github.com/iree-org/iree/pull/19684#issuecomment-2589388827.

For `LinalgExt::ArgCompareOp`, we are using the `PartialReductionOuterParallel `strategy in the tiling interface, which would naturally suggest using `transform.structured.tile_reduction_using_forall`. However, TileReductionUsingForallOp is currently constrained to only accept `LinalgOp `targets, as defined here: https://github.com/llvm/llvm-project/blob/127d77d279de6547324c87f393164c604297c4d4/mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td#L2045.
This limitation prevents us from directly applying `tile_reduction_using_forall` to  support LinalgExt::ArgCompareOp operations. 
 We would need an LLVM PR similar to this one: https://github.com/llvm/llvm-project/pull/120118, to extend TileReductionUsingForallOp (`transform.structured.tile_reduction_using_forall`) to support general target. 

Since this PR implements a split-k through the flow of `FormSplitReductionDispatchesPass`, I think the current tests   are fine and don't need to use transform ops for testing again. What do you think? @IanWood1

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/DispatchCreation/test/form_split_reduction_dispatches.mlir`

**Line:** 140

**Comment:**

I dont know if the upstream operation is built into IREE. They are part of test dialect upstream which is not built into IREE. 
You can copy that op in IREE  and use that if that helps.

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/DispatchCreation/test/form_split_reduction_dispatches.mlir`

**Line:** 140

**Comment:**

Thank you for the explanation, I now see what the issue is with using transform dialect ops. Using `transform.structured.tile_reduction_using_forall` might make sense in the future but for now I agree that the tests are fine where they are.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/DispatchCreation/test/form_split_reduction_dispatches.mlir`

**Line:** 140

**Comment:**

> I dont know if the upstream operation is built into IREE. They are part of test dialect upstream which is not built into IREE. You can copy that op in IREE and use that if that helps.

I think it is already built into IREE since `transform.structured.tile_reduction_using_for` had been used inside IREE. If required, I can send an upstream llvm PR similar to this: https://github.com/llvm/llvm-project/pull/120118 to modify the target from LinalgOp to a general operation.

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/DispatchCreation/test/form_split_reduction_dispatches.mlir`

**Line:** 140

**Comment:**

That'd be good too. I think its good to have stand alone tests for the tiling of this operation instead of tying it to the dispatch creation pass solely.

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/DispatchCreation/test/form_split_reduction_dispatches.mlir`

**Line:** 140

**Comment:**

Can you take a stab at doing that. If we can manage to move the tests to be decoupled from the pass, that would be good! 

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/DispatchCreation/test/form_split_reduction_dispatches.mlir`

**Line:** 140

**Comment:**

> That'd be good too. I think its good to have stand alone tests for the tiling of this operation instead of tying it to the dispatch creation pass solely.

Got it! Should I land this PR first and then send a follow-up PR to implement the standalone tests after the upstream PR is merged, or should I keep this PR here?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/DispatchCreation/test/form_split_reduction_dispatches.mlir`

**Line:** 140

**Comment:**

> Can you take a stab at doing that. If we can manage to move the tests to be decoupled from the pass, that would be good!

Sure, will do!

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1471

**Comment:**

(opening the discussion because it is useful.)

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1485

**Comment:**

Do we bail out when the reduction dimensions mismatch, i.e., `reductionDims.size() > 1 && reducionDims[0] != getDimension`?

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1541

**Comment:**

I'm not sure if we can handle encodings or not. They are tricky. I think we can ignore the encodings like other operations, so it becomes simpler.

```suggestion
  auto sliceValResultType = RankedTensorType::get(
      resultValShape, getOutputValueType().getElementType());
  auto sliceIdxResultType = RankedTensorType::get(
      resultIdxShape, getOutputIndexType().getElementType());
```

It looks like it can be simpler with using the [clone](https://github.com/llvm/llvm-project/blob/4eadb45f83cef00165055f8038f179ca5c3e88ef/mlir/include/mlir/IR/BuiltinTypeInterfaces.td#L246-L251) method. E.g.,

```cpp
ShapedType sliceValResultType = getOutputValueType().clone(resultValShape);
// ...
```

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1583

**Comment:**

optional: we don't need the function argument comment if they are obvious.

```suggestion
  Operation *tiledArgmaxOp = ArgCompareOp::create(
      b, loc, resultTypes,
      /*inputs=*/operands[0],
      /*outputs=*/ValueRange{operands[1], operands[2]},
      /*indexBase=*/operands[3], reductionDim);
```

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1588

**Comment:**

optional: just a minor note. `takeBody` takes over the region, so the original op must be deleted before exiting the transformation. Otherwise, the old op becomes an invalid op because it does not have a body.

To reflect your comment (i.e., `Copy the region`) better, you can use `cloneInto` method. It does not take over the ownership of the region, but it requires declaring a `mapper`. E.g.,

```cpp
IRMapping mapper;
getRegion().cloneInto(targetRegion, mapper);
```

https://github.com/iree-org/iree/blob/fd7c6507ac0125658e84e3b471adc5ed399b65ec/compiler/src/iree/compiler/Codegen/Common/PatchFuncOps.cpp#L89-L95

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1658

**Comment:**

optional nit: I'd remove the blank lines because it does not help the readability to me. 

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1585

**Comment:**

I'd keep the comment for `dimension=` here

---

## PR #21812: [Codegen][Tuner]: improve python binding to query target info

**URL:** https://github.com/iree-org/iree/pull/21812
**State:** MERGED

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 86

**Comment:**

Let's initialize this since it's a C struct and we don't want to accidentally end up with uninitialized fields.

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 81

**Comment:**

Can you explain why we need this function? I'd expect this to be queried by C bindings which return a complete `ireeGPUTargetInfo` struct

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 600

**Comment:**

Could we do this without throwing any exceptions?

---

### Comment by bangtianliu

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 81

**Comment:**

The C binding `ireeHALExecutableTargetAttrGetGPUTargetInfo` can return a complete `ireeGPUTargetInfo ` struct from the executable variant op. The `createGPUTargetInfo` is mainly for the constructor of `TargetInfo` python class. 

```C++
  py::class_<ireeGPUTargetInfo>(iree_gpu_module, "TargetInfo")
      .def_static("get", &createGPUTargetInfo, "context"_a, "arch"_a,
                  "subgroup_size_choices"_a, "max_workgroup_sizes"_a,
                  "max_thread_count_per_workgroup"_a,
                  "max_workgroup_memory_bytes"_a,
                  "mma_intrinsics"_a = py::list{},
                  "Create a GPUTargetInfo with the given parameters")
  ```

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 81

**Comment:**

This is understand, I just don't think we should have so much work to do on the python binding side. We should only do minimal work to take data from C structs and put them in a format friendly to python.

---

### Comment by bangtianliu

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 81

**Comment:**

This function is mainly about constructing an `ireeGPUTargetInfo `struct, which is then used to initialize the Python data class `TargetInfo`. The idea is to let users create a `TargetInfo `instance directly from Python, for example:
```python
    target_info = iree_gpu.TargetInfo.get(
        context=context,
        arch="gfx942",
        subgroup_size_choices=[32, 64],
        max_workgroup_sizes=[256, 512, 1024],
        max_thread_count_per_workgroup=1024,
        max_workgroup_memory_bytes=65536,
        mma_intrinsics=[
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x4_F32,
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.VirtualMMAIntrinsic.VMFMA_F32_16x16x32_F16,
        ],
    )
```
If we only want to query existing data from the struct, then this `createGPUTargetInfo `function could indeed be removed. I’m not entirely sure how to achieve that through the C binding, since unlike something like `LowerConfigAttr`, this isn’t represented as an IREE attribute. But I can look into how to make it work that way.

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 81

**Comment:**

you can have two constructors (or one constructor and one static method): one to allow construction in python, and one to query target details and return you `TargetInfo`: https://nanobind.readthedocs.io/en/latest/classes.html . Not every python class has to be backed by a struct in the C api

> But I can look into how to make it work that way.

+1

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 469

**Comment:**

We also need a test for the cases when the constructor is given values of wrong type

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 7

**Comment:**

I don't think we use pytest anywhere else in IREE -- would be worth checking with other folks if we want to add it and then clean up other pieces of the infra.

If you grep python files, you will notice that some use `untitest` instead -- can we switch to that?

---

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 164

**Comment:**

How do you know `mmaIntrinsics` map to `int32_t`?

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 522

**Comment:**

1. How do we know this is the underlying enum type?
2. How do we know that this is an mma intrinsic type and not some other enum?

---

### Comment by bangtianliu

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 7

**Comment:**

yeah, I noticed that it caused CI errors although it worked locally.

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 541

**Comment:**

Do we need the lambda?

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 575

**Comment:**

Can we move this to the C API instead?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 478

**Comment:**

Do we need to handle the empty case in a separate code path?

---

### Comment by bangtianliu

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 164

**Comment:**

I checked the builded file (_iree_gpu_enum_gen.py):

```
class MMAIntrinsic(IntEnum):
    """Descriptor for different MMA intrinsics"""

    MFMA_F32_16x16x4_F32 = 4112
    MFMA_F32_16x16x16_F16 = 4128
    MFMA_F32_32x32x8_F16 = 4129
    MFMA_I32_16x16x16_I8 = 4288
    MFMA_I32_32x32x8_I8 = 4289
```

---

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 164

**Comment:**

How can I tell it's i32 instead of u32? Also, what happens if this changes in the future? I think we need to have a typedef on the C api side, static_assert in the C implementation, and then use this typedef on the python side.

---

### Comment by bangtianliu

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 164

**Comment:**

ok, I found we have this https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp#L151:

```
static_assert(
    std::is_same_v<uint32_t, std::underlying_type_t<
                                 mlir::iree_compiler::IREE::GPU::MMAIntrinsic>>,
    "Enum type changed");
```

sorry for missing this.

---

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 20

**Comment:**

```suggestion
typedef uint32_t mma_intrinsic_enum_t;
```

---

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 175

**Comment:**

Could you make this a plural so that it's clear there is more than one value behind this pointer?

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 532

**Comment:**

Could we make these share the same base class?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 528

**Comment:**

llvm prefers C++ casts: https://llvm.org/docs/CodingStandards.html#prefer-c-style-casts

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 537

**Comment:**

Use early exits: https://llvm.org/docs/CodingStandards.html#use-early-exits-and-continue-to-simplify-code

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 527

**Comment:**

The memory should be allocated and freed on the python side. We have a few examples of this already.

---

### Comment by bangtianliu

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 532

**Comment:**

I think it comes from IREE side: https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.td#L172-L173
and  https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.td#L278-L279

Not sure how to make it share the same base class. 

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 532

**Comment:**

There are defined on the Python side though: https://github.com/iree-org/iree/blob/933f798046a817dcff48d84df8fd987c5cb9e72b/compiler/bindings/python/IREECompilerDialectsModule.cpp#L312 so I'd think we can add a python base clase that doesn't exist in tablegen/c++

Fine to leave as is though for now

---

### Comment by bangtianliu

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 532

**Comment:**

ok, then I will keep this as a todo and address it in a separate PR.

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 526

**Comment:**

https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 176

**Comment:**

This name doesn't make sense to me, I'd call it something like `virtualMmaIntrinsicTags` and add a comment explaining that it's to distinguish virtual from non-virtual intrinsics

---

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 20

**Comment:**

This needs a static assert on the C bindings implementation side

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 524

**Comment:**

we can expose these names in C bindings so that this is not hardcoded

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 595

**Comment:**

also here

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 602

**Comment:**

https://llvm.org/docs/CodingStandards.html#use-early-exits-and-continue-to-simplify-code

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 430

**Comment:**

https://llvm.org/docs/CodingStandards.html#use-early-exits-and-continue-to-simplify-code

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 437

**Comment:**

you can inline this into the push back on the next line, this variable does not seem to be used anywhere else

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 444

**Comment:**

also here

---

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 176

**Comment:**

I don't think we need to pass in the number of elements -- this can be derived from the size of the `mmaIntrinsics` array, no?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 522

**Comment:**

Should we assert that `mmaIntrinsics` is an array attr?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 522

**Comment:**

We have checks before calling this function as shown below:

```
       if (mlirAttributeIsNull(self.mmaIntrinsics) ||
                !mlirAttributeIsAArray(self.mmaIntrinsics)) {
              return py::list();
            }
```
so if the `mmaIntrinsics ` is not an array attr, the function will not be called.


---

### Comment by bangtianliu

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 20

**Comment:**

It's already there: https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp#L151-L160

---

### Comment by bangtianliu

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 524

**Comment:**

I've implemented a solution using static constants at the beginning of IREECompilerDialectsModule.cpp:
```
static const char *kMMAIntrinsicEnumName = "MMAIntrinsic";
static const char *kVirtualMMAIntrinsicEnumName = "VirtualMMAIntrinsic";
```
And then used these constants throughout the file instead of hardcoded strings.

However, I understand this may not be the ideal approach you're asking for. I'm still trying to figure out how to obtain the `enum class MMAIntrinsic`'s class name directly from the MLIR-generated code rather than hardcoding it.
Could you point me to the correct approach like correct MLIR API or generated function, which would allow me to get the name dynamically. 

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 522

**Comment:**

This can be called from C api without any python, bypassing this check

---

### Comment by bangtianliu

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 524

**Comment:**

ping here @kuhar 

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 524

**Comment:**

Ideally this should be emitted by tablegen, but if it's not possible, you can expose a helper function that will survive refactoring:
```c++
#define GET_ENUM_NAME(X) strrchr(#X, ':') + 1

const char *ireeGPUGetMMAIntrinsicCName() {
  static const char *kTxt =
      GET_ENUM_NAME(mlir::iree_compiler::IREE::GPU::MMAIntrinsic);
  return kTxt;
}

#undef GET_ENUM_NAME
```

What you wrote is probably good enough though -- we can keep it in `IREECompilerDialectsModule.cpp` and make sure there's a test that checks the names are in sync.

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 534

**Comment:**

can we use the string constants here too?

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 538

**Comment:**

we can inline this into the push back, since this is the only use of this variable

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 587

**Comment:**

Do we need a special case for zero elements?

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 613

**Comment:**

this continue doesn't do anything

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 482

**Comment:**

Can we use a range for loop here? I don't see `i` being used anywhere else.

Also https://llvm.org/docs/CodingStandards.html#prefer-preincrement

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 508

**Comment:**

This won't work -- the assertion message it decided at compilation time

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 482

**Comment:**

No, `mmaIntrinsics` is a pointer, not a container. And I will change `i++` to `++i`.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 518

**Comment:**

The formatting is weird here because of the comment, I'd move it to the `.h` file and above the function definition

---

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 20

**Comment:**

This is not resolved. Nothing checks `mma_intrinsic_enum_t`

---

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 180

**Comment:**

llvm uses the C++ comment style: https://llvm.org/docs/CodingStandards.html#comment-formatting

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 537

**Comment:**

Do you know if this is valid in C++? I'm afraid this won't work because the string is a temporary whose lifetime won't be extended until the corresponding catch. Maybe we just simplify this and say `"All items must be MMA atributes"`

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 404

**Comment:**

This is not the right way to test this. We only need to make sure that the function that uses these names can look up the correct attributes in python.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 524

**Comment:**

Do we need this special case?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 152

**Comment:**

Instead of this, we should update the two enums below. Our code is correct when `mma_intrinsic_enum_t` matches the underlying type of both mma enums

---

## PR #21782: [Codegen][Tuner] expose python binding to query target info

**URL:** https://github.com/iree-org/iree/pull/21782
**State:** MERGED

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 148

**Comment:**

```suggestion
  MlirIdentifier arch;                 // E.g., "gfx942".
```

---

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 152

**Comment:**

Can this be int64_t?

---

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 151

**Comment:**

Can we make this int64_t?

---

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 155

**Comment:**

This doesn't add any functions, this is a function to query something
```suggestion
// Queries GPU target info from the given `ExecutableTargetAttr` attribute |attr|.
```

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 542

**Comment:**

Can we use `ireeHALExecutableTargetAttrGetGPUTargetInfo` directly instead of wrapping it in a lambda?

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 410

**Comment:**

Could we set the attributes we don't care about to the empty value, e.g., `none`, to keep this minimal?

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 413

**Comment:**

Can you add a testcase that has more than one value for subgroup size coices?

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 417

**Comment:**

For max wokrgroup sizes and counts, I would pick 3 distinct values so that we can test they appear in the correct order on the python side

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 435

**Comment:**

I think failed assertions are going to print the actual and expected values -- could you check if this is the case, and remove those messages if it works like I described? Also below.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 403

**Comment:**

Please spell out the types here and below, where the type is not obvious based on the RHS only. See https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 149

**Comment:**

Also use the mlir naming standard for all fields in this class

---

### Comment by bangtianliu

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 435

**Comment:**

I tested it using `assert max_thread_count==1025` . The ctest (my command: ctest -R dialects --output-on-failure) only shows AssertationError as below:
```
TEST: gpu_target_info_attribute_parsing
Traceback (most recent call last):
  File "/home/bangtliu/iree/compiler/bindings/python/test/ir/dialects_test.py", line 396, in <module>
    @run
     ^^^
  File "/home/bangtliu/iree/compiler/bindings/python/test/ir/dialects_test.py", line 68, in run
    fn()
  File "/home/bangtliu/iree/compiler/bindings/python/test/ir/dialects_test.py", line 447, in gpu_target_info_attribute_parsing
    max_thread_count == 1025
**AssertionError**
```
the actual value of `max_thread_count` is 1024.

---

### Comment by bangtianliu

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 417

**Comment:**

Thanks for the suggestion! I made `max_workgroup_sizes` change to test ordering. Note that `max_workgroup_counts`  (The maximum number of workgroups per X/Y/Z dimension in a dispatch.) isn't included in the Python binding as it's not required for the tuner.

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 413

**Comment:**

Can we populate this with at least two values and make sure they are in the correct order?

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 417

**Comment:**

Can we make these distinct?

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 478

**Comment:**

IIRC, this should always match the x / y / z dimensions, so this config seems invalid

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 460

**Comment:**

Why do we need a second test input? I think one should be enough to exercise everything we need?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 406

**Comment:**

Use an early return instead: https://llvm.org/docs/CodingStandards.html#use-early-exits-and-continue-to-simplify-code

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 399

**Comment:**

This assert will never trigger because `llvm::cast` checks that the types match and asserts internally (unlike `llvm::dyn_cast`). See https://llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 399

**Comment:**

We can delete this assert

---

## PR #21454: [Codegen][Tuner]: expose python binding for mma single subgroup layout

**URL:** https://github.com/iree-org/iree/pull/21454
**State:** MERGED

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 548

**Comment:**

nit: this string is broken in a weird way -- could you reflow this? I'd expect something like
```suggestion
      "tstrides) for a given MMA or VirtualMMA intrinsic and "
      "fragment.",
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 390

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

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 383

**Comment:**

We can start by zero-initializing the whole struct in case we forgot to initialize some field later on. Zero values are usually easier to track down than uninitialized garbage.
```suggestion
  ireeGPUMMASingleSubgroupLayout result = {};
```

---

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 141

**Comment:**

Can you comment on the underlying data type? I think this will be an arrayattr of integers/index?

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 528

**Comment:**

Should we return an arrayattr or a python list of integers? I think the latter will be easier to work with, especially since all of these are read only

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 66

**Comment:**

Use the C API for the type check instead -- notice how the whole file uses C APIs only.

---

## PR #21408: Integrate LLVM to llvm/llvm-project@5f53182

**URL:** https://github.com/iree-org/iree/pull/21408
**State:** MERGED

### Comment by krzysz00

**File:** `compiler/src/iree/compiler/Codegen/Common/FoldTensorExtractOp.td`

**Line:** 20

**Comment:**

That might want to be a native code call producing the null attribute to make it optional?

---

## PR #21403: [Codegen][Tuner] add python binding for VirtualMMAIntrinsic

**URL:** https://github.com/iree-org/iree/pull/21403
**State:** MERGED

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 106

**Comment:**

Could we use the same functions to handle virtual and non-virtual mma intrinsics? I mean one function that would subsume both `ireeGPUVirtualMMAAttrGetInfo` and  `ireeGPUMMAAttrGetInfo`.

---

### Comment by bangtianliu

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 106

**Comment:**

Yeah, good suggestion! 

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 249

**Comment:**

you can use `llvm::TypeSwitch`

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 249

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

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 223

**Comment:**

This assertion os redundant to the one in the default case

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 230

**Comment:**

```suggestion
          [](auto mma) {
```

The return type appears in the typeswitch definition and in side the lamdba body already 

---

## PR #21364: Integrate LLVM to llvm/llvm-project@3ed3a33

**URL:** https://github.com/iree-org/iree/pull/21364
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td`

**Line:** 1662

**Comment:**

Delete this instead of commenting out?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td`

**Line:** 1782

**Comment:**

Also here

---

## PR #21345: [Codegen][Tuner] remove decomposition attr for attention op

**URL:** https://github.com/iree-org/iree/pull/21345
**State:** MERGED

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 83

**Comment:**

Is there a reason we don't always just remove the decomposition config entirely? I thought it was just used as additional compilation info, which seems like something this pass should always remove.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 83

**Comment:**

Here from Kunwar: https://github.com/iree-org/iree/pull/20072#discussion_r1968187342

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 86

**Comment:**

nit: flip this condition to avoid negation

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 83

**Comment:**

I see, thanks!

---

## PR #21216: [Codegen][Tuner] expose python binding isa_attention_op

**URL:** https://github.com/iree-org/iree/pull/21216
**State:** MERGED

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 509

**Comment:**

Do we need the lambda here? If the types match, you shouldn't need a wrapper around `ireeCodegenMlirOperationIsACodegenAttentionOp`

---

## PR #21170: [Codegen][Tuner] expose python binding for attention op details

**URL:** https://github.com/iree-org/iree/pull/21170
**State:** MERGED

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_codegen.h`

**Line:** 94

**Comment:**

What about the rank?

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 25

**Comment:**

Let's keep the use of mlir C++ apis explicit here

---

### Comment by bangtianliu

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_codegen.h`

**Line:** 94

**Comment:**

This refers to the function `mlir::iree_compiler::IREE::LinalgExt::AttentionOpDetail::get(QMap, KMap, VMap, OMap)`.
The rank information is included in the returned struct `ireeCodegenAttentionOpDetail`.

---

### Comment by bangtianliu

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_codegen.h`

**Line:** 94

**Comment:**

it corresponds to https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h#L64, I will keep the name consistent. 

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 16

**Comment:**

What do we use this for?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp`

**Line:** 250

**Comment:**

Don't use auto here, the type is not obvious without IDE

---

### Comment by bangtianliu

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 16

**Comment:**

used for unwrap in this line: https://github.com/iree-org/iree/pull/21170/files#diff-c7a24b0fe6eed0b576272379d2beb77420561eebd101a406b390c712d9b7e519R66

---

## PR #21138: [LinalgExt] support converting argcompare to loops.

**URL:** https://github.com/iree-org/iree/pull/21138
**State:** MERGED

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/test/convert_to_loops.mlir`

**Line:** 750

**Comment:**

I don't think the lowering here is valid. From my understanding of the op definition, `%0` and `%1` are loading from a potentially uninitialized memref.

```mlir
func.func @arg_compare_memref(%arg0: memref<2x10xf32>, %arg1: memref<2xf32>, %arg2: memref<2xi32>) {                                                                                          %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0 to %c2 step %c1 {
    scf.for %arg4 = %c0 to %c10 step %c1 {
      %0 = memref.load %arg1[%arg3] : memref<2xf32>
      %1 = memref.load %arg2[%arg3] : memref<2xi32>
      %2 = memref.load %arg0[%arg3, %arg4] : memref<2x10xf32>
      %3 = arith.cmpf ogt, %2, %0 : f32
      %4 = arith.select %3, %2, %0 : f32
      %5 = arith.index_cast %arg4 : index to i32
      %6 = arith.select %3, %5, %1 : i32
      memref.store %4, %arg1[%arg3] : memref<2xf32>
      memref.store %6, %arg2[%arg3] : memref<2xi32>
    }
  }
  return
}
```

This might not be possible due to limitations of the interface, but outside of the loop you want to insert the first elems of `%arg0` into `%arg2` and `0 + index_base` into `%arg3`. It might be possible to do this as a conditional inside the loop? Or the op definition could be clarified to say that the outs must be initialized in this way.

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1363

**Comment:**

```suggestion
  uint64_t dim = getDimension();
```

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1363

**Comment:**

or `reductionDim`

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1365

**Comment:**

Since the non-reduction dims may not be the outermost:

```suggestion
  SmallVector<Value> parallelIndices;
```

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1367

**Comment:**

I don't think this cast is needed. `i` can be `uint64_t` to match `kDim`

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/test/convert_to_loops.mlir`

**Line:** 750

**Comment:**

here is one example about generic op version for argmax, FYI 

```
util.func public @argmax(%arg0: tensor<?x131072xbf16>, %arg1: index) -> tensor<?xi64> {
  %cst = arith.constant 0xFF80 : bf16
  %c0_i64 = arith.constant 0 : i64
  %0 = tensor.empty(%arg1) : tensor<?xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<?xbf16>) -> tensor<?xbf16>
  %2 = tensor.empty(%arg1) : tensor<?xi64>
  %3 = linalg.fill ins(%c0_i64 : i64) outs(%2 : tensor<?xi64>) -> tensor<?xi64>
  %4:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<?x131072xbf16>) outs(%1, %3 : tensor<?xbf16>, tensor<?xi64>) {
  ^bb0(%in: bf16, %out: bf16, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : bf16
    %8 = arith.cmpf ogt, %in, %out : bf16
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : bf16, i64
  } -> (tensor<?xbf16>, tensor<?xi64>)
  util.return %4#1 : tensor<?xi64>
}
```
But I think you are right. This lowering currently assumes that the outs are pre-initialized with the first candidate value and corresponding index before the loop starts, which isn't guaranteed and can lead to undefined behavior.
I will see how to fix this issue.


---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1365

**Comment:**

nit: do not recalculate the end index at each loop iteration

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1377

**Comment:**

```suggestion
  Type indexType = getOutputIndexType().getElementType();
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1406

**Comment:**

nit: The types are not obvious in these two places without using an IDE, we shouldn't use `auto` here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1403

**Comment:**

What is `bvm`? I don't know this TLA.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1403

**Comment:**

I guess it stands for "Block Value Mapping" (bvm), and I followed the naming convention used in similar implementations—for example, in the TopK lowering.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1366

**Comment:**

FYI, you can also do:
```suggestion
  for (size_t i = 0, rank = ivs.size(); i < rank; ++i) {
```
if you don't use `rank` outside of the loop

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/test/convert_to_loops.mlir`

**Line:** 870

**Comment:**

I dont think this should be `J` . This should be `OFFSET` ?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/test/convert_to_loops.mlir`

**Line:** 870

**Comment:**

This is only checking whether it's the first iteration, so it shouldn't involve any offset.

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/test/convert_to_loops.mlir`

**Line:** 870

**Comment:**

that seems off to me. Actually we dont even need that if statement. Its like a matmul op. It doesnt check for index being 0 to do a reduction. You load what is there and perform the "reduction" and store it back. It is expected that you prefill it such that this is valid.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/test/convert_to_loops.mlir`

**Line:** 870

**Comment:**

```
func.func @arg_compare_memref(%arg0: memref<2x10xf32>, %arg1: memref<2xf32>, %arg2: memref<2xi32>) {                                                                                          %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  scf.for %arg3 = %c0 to %c2 step %c1 {
    scf.for %arg4 = %c0 to %c10 step %c1 {
      %0 = memref.load %arg1[%arg3] : memref<2xf32>
      %1 = memref.load %arg2[%arg3] : memref<2xi32>
      %2 = memref.load %arg0[%arg3, %arg4] : memref<2x10xf32>
      %3 = arith.cmpf ogt, %2, %0 : f32
      %4 = arith.select %3, %2, %0 : f32
      %5 = arith.index_cast %arg4 : index to i32
      %6 = arith.select %3, %5, %1 : i32
      memref.store %4, %arg1[%arg3] : memref<2xf32>
      memref.store %6, %arg2[%arg3] : memref<2xi32>
    }
  }
  return
}
```

Does this version make sense?  refer to https://github.com/iree-org/iree/pull/21138#discussion_r2157667682 for more info. 

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/test/convert_to_loops.mlir`

**Line:** 870

**Comment:**

Yeah, this makes mroe sense. @IanWood1  the expectation is that the `outs` are pre-initialized, just like `linalg` ops. If you dont pre-initialize them then it will have garbage and your answer is going to be off. That is the expectation/semantics of the op. 


---

## PR #21106: [LinalgExt] fix arg_compare op with region and start index

**URL:** https://github.com/iree-org/iree/pull/21106
**State:** MERGED

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td`

**Line:** 689

**Comment:**

The op description need to be updated to reflect the new region.

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 931

**Comment:**

This isn't needed right? `llvm::size(getInputs())` should equal `getNumDpsInputs()`


edit: I think my reasoning above is wrong but still don't think this check is needed since `index_base` is an optional arg. So, if `size(inputs)` is 1 then `getNumDpsInputs()` must be 1 or 2. 

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 1003

**Comment:**

This causes a crash with the following IR:

```
func.func @argmax_static(
    %input : tensor<2x6xf32>,
    %outv : tensor<2xf32>, %outi : tensor<2xindex>
) -> (tensor<2xf32>, tensor<2xindex>) {
  %0:2 = iree_linalg_ext.argmax
    dimension(1)
    ins(%input : tensor<2x6xf32>)
    outs(%outv, %outi : tensor<2xf32>, tensor<2xindex>) {
    ^bb0(%a: f32, %b: f32):
      %cmp = arith.cmpf ogt, %a, %b : f32
  } -> tensor<2xf32>, tensor<2xindex>
  return %0#0, %0#1 : tensor<2xf32>, tensor<2xindex>
}

```

I think this is because the `SingleBlockImplicitTerminator` trait will create a `LinalgExt::YieldOp` with no operands when its implicit. Also, this doesn't handle >1 operand. You can probably just copy what the attention verifier does and check that `getNumOperands() == 1`

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 931

**Comment:**

> This isn't needed right? `llvm::size(getInputs())` should equal `getNumDpsInputs()`
> 
> edit: I think my reasoning above is wrong but still don't think this check is needed since `index_base` is an optional arg. So, if `size(inputs)` is 1 then `getNumDpsInputs()` must be 1 or 2.

Yeah, you are correct here. 

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 933

**Comment:**

I missed this earlier, the tblgen definition `Optional<Index>:$index_base` should preform this check (that's why `indexBase` is `TypedValue<IndexType>` and not just Value). So, I think this can be removed.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td`

**Line:** 687

**Comment:**

It would be nice to add an example of the op here, I rely on these heavily when writing IR by hand. I think argmax specifically would make a nice sample.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td`

**Line:** 700

**Comment:**

Please put this in code blocks so that it renders nicely on the website

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td`

**Line:** 717

**Comment:**

Also here

---

## PR #21077: [LinalgExt] add TilingInterface support for ArgCompareOp

**URL:** https://github.com/iree-org/iree/pull/21077
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1352

**Comment:**

nit: I think a plain loop would be fine as well. If you keep as-is, I think there's an unary version of `seq` that would be more concise.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1392

**Comment:**

IREE requires braces around bodies of single-statements `if`s/loops

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1373

**Comment:**

Can you break this down into two separate assertions? This way we will know which one failed without having to start a debugger or recompile the source.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1392

**Comment:**

Also elsewhere below

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1352

**Comment:**

Do not recalculate the trip count: https://llvm.org/docs/CodingStandards.html#don-t-evaluate-end-every-time-through-a-loop

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1401

**Comment:**

I know this follows the convention of this file, but I don't think we need a NULL check here as it shouldn't fail. Maybe this could be done as a separate cleanup to the whole file.

```suggestion
  Operation *outputValSlice = getSlice(
      builder, loc, outputValue(), outputOffsets, outputSizes, outputStrides);
```

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1449

**Comment:**

This doesn't look needed. It should be true as long as the op is valid.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1401

**Comment:**

https://github.com/iree-org/iree/blob/d867a94aaf35d551905786253c4912a914312a12/compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp#L89-L105

Just curious — why shouldn't this fail? Based on the code above, if a type other than RankedTensorType or MemRefType is provided, it looks like the function could return nullptr.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1401

**Comment:**

drive-by notification: I think the function returns the nullptr just because the compiler would complain if it doesn't. 

My rule about handling a failure or not (i.e., use assert or ignore the check) is that if it is expected or not. If it is an expected failure and we are able to handle it, let's go with handling it. If not, we either assert or `(void)` the return value.

In the tiling implementation, the input should always be a valid LinalgExt op where the inputs and outputs can only be ranked tensor type or memref type. In this context, the failure handling is not required IMO.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1401

**Comment:**

Oh, it has `assert false ` there lol.

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp`

**Line:** 1401

**Comment:**

>I think the function returns the nullptr just because the compiler would complain if it doesn't.

I just looked and apparently you don't need a default case. `TypeSwitch` is convertible to `ResultT` and will assert when none of the cases are matched.

---

## PR #21021: [LinalgExt] Add argmax op with rountrip and invalid mlir test

**URL:** https://github.com/iree-org/iree/pull/21021
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 924

**Comment:**

It would be nice to also print the actual number (in addition to the expected one)

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 929

**Comment:**

Also here: print what the wrong value is

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 934

**Comment:**

```suggestion
  auto outputValueType = cast<ShapedType>(outputValue().getType());
  auto outputIndexType = cast<ShapedType>(outputIndex().getType());
```
See https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 936

**Comment:**

I don't think this comment adds much value -- it's pretty clear what is being check. In general, focus on the *why* when writing the comments, not on what the code does.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 941

**Comment:**

Same here...

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 961

**Comment:**

nit: I think `==` would work here?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 954

**Comment:**

I think it would help to print the expected and the actual shape

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td`

**Line:** 710

**Comment:**

Do we also want `getOutput*Type` for index and value?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 951

**Comment:**

use `llvm::interleaved_array`

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 955

**Comment:**

also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 974

**Comment:**

Also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 953

**Comment:**

```suggestion
           << llvm::interleaved_array(outputValueType.getShape())
           << ", output index shape: "
           << llvm::interleaved_array(outputIndexType.getShape());
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp`

**Line:** 966

**Comment:**

```suggestion
           << "Expected: " << llvm::interleaved_array(expectedShape)
           << ", but got: "
           << llvm::interleaved_array(outputValueType.getShape());
```

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td`

**Line:** 682

**Comment:**

The intent of this operation is that this should be tilable. To make that correct I think we need a couple of mor things.

1) I think defining an operation just for argmax is not scalable. I think we should define an operation that llows you to do "max" of any operation. So it might be better to add an operation that allows you to specify different functions to compute the "max".

Basically do 
```
%0:2 = iree_linalg_ext.arg_compare ins(%0 : tensor<?x?xf32>) outs(...) dimension(...) {
  ^bb0(%b0 : f32, %b1: f32):
    %1 = arith.cmpf ge, %b0, %b1 : f32
    iree_linalg_ext.yield %1 : f32
} -> tensor<?xf32>
```

(Think how you would do this with something like C++ std lib which generally allows you to take a comparator function, this is essentially that).

2. I think you also need an additional argument that represents the "global start index", that is by default 0. After tiling you will need to update the global start index appropriately.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td`

**Line:** 682

**Comment:**

Yes, I actually considered both adding a region and supporting a global start index.
My initial plan was to start with a basic implementation to get the core functionality in place, and then extend it to support custom comparators via regions for more flexibility . Similarly, the global start index will be important for correctness when enabling split-K.
Thanks for the suggestions, I'll make sure to address both aspects in the follow-up PR.

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td`

**Line:** 682

**Comment:**

That would create too much churn no? It will be a pain to update all the tests after the fact etc. I think it would be better to just add this from the get go.
If we find out later that we missed something then there is nothing we can do, but we might as well start with accounting for things we already know?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td`

**Line:** 682

**Comment:**

> It will be a pain to update all the tests after the fact etc

Ok, then I will add all the required changes directly to the second PR #21077. 

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td`

**Line:** 682

**Comment:**

Ok that's fine. Will be easier to stage it, i.e. fix the op and then the PR on top of that, but your call 

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td`

**Line:** 682

**Comment:**

> Ok that's fine. Will be easier to stage it, i.e. fix the op and then the PR on top of that, but your call

Sure, sounds good and I’ll follow your suggestion. I’ll first send a follow-up to address the argmax op fixes, then rebase and update the tiling PR accordingly. This should help keep the diffs cleaner and easier to review.


---

## PR #20906: [Codegen] split-k on argmax to ensure ukernel support

**URL:** https://github.com/iree-org/iree/pull/20906
**State:** MERGED

### Comment by Max191

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 674

**Comment:**

Is the fill necessary here? It doesn't seem to be reading these values in the generic op.

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp`

**Line:** 902

**Comment:**

Can we add a small test for this new logic in `Codegen/Common/GPU/test/gpu_lower_to_ukernels.mlir`?

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/test/gpu_lower_to_ukernels.mlir`

**Line:** 137

**Comment:**

nit: These could be passed as function arguments to make the test simpler.

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 644

**Comment:**

nit: I think the comment is a little unclear to me. What do you think about this suggestion?
```
// Step 1: Create a pure argmax to partially reduce the split dimension. The result will
// contain local indices within each reduction group, which need to be adjusted to the
// global index later.
```

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 651

**Comment:**

nit: rename to `reductionIdx`. IMO it is worth spelling out the full word here.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/test/gpu_lower_to_ukernels.mlir`

**Line:** 137

**Comment:**

Thanks for the suggestion! I’ve partially applied it—I added function arguments to avoid using` tensor.empty`. However, I kept the linalg.fill operations for init_val and init_idx inside the function because the `isArgmaxOp `implementation relies on these values being defined by `linalg.fill`:

```C++
  Value initVal = genericOp.getDpsInitOperand(0)->get();
  auto fillOp = initVal.getDefiningOp<linalg::FillOp>();
  if (!fillOp)
    return false;
 ```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/test/gpu_lower_to_ukernels.mlir`

**Line:** 147

**Comment:**

```suggestion
      // Breaks isArgmaxOp matching.
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 644

**Comment:**

What do you mean by 'pure argmax'?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 652

**Comment:**

nit: llvm discourages defining multiple variables on a single line.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 694

**Comment:**

also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp`

**Line:** 894

**Comment:**

Use `cast` if you require the result to be a select. Otherwise check if the result is not null. See https://llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 644

**Comment:**

By "pure argmax", I meant a structurally strict argmax pattern required by lowering it to ukernel (detailed in `isArgmaxOp` function). Typically looks like below IR (using maximumf, cmpf, and select instruction and with the index derived directly from a linalg.index optionally through index_cast)
```
  %8:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded : tensor<?x1024x128xbf16>) outs(%5, %7 : tensor<?x1024xbf16>, tensor<?x1024xi64>) {
    ^bb0(%in: bf16, %out: bf16, %out_6: i64):
      %10 = linalg.index 2 : index
      %11 = arith.index_cast %10 : index to i64
      %12 = arith.maximumf %in, %out : bf16
      %13 = arith.cmpf ogt, %in, %out : bf16
      %14 = arith.select %13, %11, %out_6 : i64
      linalg.yield %12, %14 : bf16, i64
    } -> (tensor<?x1024xbf16>, tensor<?x1024xi64>)
```
I'll clarify the comment to avoid ambiguity — perhaps change it to “strict argmax”.

---

## PR #20768: [Codegen] Add ukernel support for argmax on BF16 and enable optional max value return

**URL:** https://github.com/iree-org/iree/pull/20768
**State:** MERGED

### Comment by bjacob

**File:** `compiler/plugins/target/ROCM/builtins/ukernel/common.h`

**Line:** 127

**Comment:**

Why is that needed? Since this code is only ever compiled with in-tree Clang/LLVM to AMDGPU target, which always supports the built-in type `__bf16`, you should be able to just use that.  This seems to confirm that it results in LLVM generating that conversion code for you:
https://godbolt.org/z/6nnMaev6r

The LLVM documention seems out of date in not mentioning support on AMDGPU,
https://clang.llvm.org/docs/LanguageExtensions.html#half-precision-floating-point


---

### Comment by bjacob

**File:** `compiler/plugins/target/ROCM/builtins/ukernel/common.h`

**Line:** 132

**Comment:**

No need for utility functions that do nothing more than a cast between built-in types.

---

## PR #20717: [Codegen] split-k on argmax op

**URL:** https://github.com/iree-org/iree/pull/20717
**State:** MERGED

### Comment by pashu123

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp`

**Line:** 848

**Comment:**

`op.getNumReductionLoops() != 1`

---

### Comment by pashu123

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp`

**Line:** 864

**Comment:**

Why do we need this check?

---

### Comment by pashu123

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp`

**Line:** 829

**Comment:**

I suggest using `LogicalResult` with better error messages so that if there's a case we want to support, people will know exactly where to add the support.

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/DispatchCreation/test/split_argmax_reduction.mlir`

**Line:** 32

**Comment:**

I am assuming you didnt need to create these operations, i.e they are the same as what existed before.

---

### Comment by pashu123

**File:** `compiler/src/iree/compiler/DispatchCreation/SplitReduction.cpp`

**Line:** 78

**Comment:**

This has already been checked. It is with the `isArgmaxOp` check.

---

### Comment by pashu123

**File:** `compiler/src/iree/compiler/DispatchCreation/SplitReduction.cpp`

**Line:** 84

**Comment:**

same.

---

### Comment by pashu123

**File:** `compiler/src/iree/compiler/DispatchCreation/SplitReduction.cpp`

**Line:** 69

**Comment:**

You can pass `MutableArrayRef<Operation*> combinerOps`.

---

### Comment by pashu123

**File:** `compiler/src/iree/compiler/DispatchCreation/SplitReduction.cpp`

**Line:** 126

**Comment:**

I don't think you need this check again. This has already been checked by the isArgMaxOp helper.

---

### Comment by pashu123

**File:** `compiler/src/iree/compiler/DispatchCreation/SplitReduction.cpp`

**Line:** 149

**Comment:**

NIT: "invalid combiner for argmax"

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/DispatchCreation/test/split_argmax_reduction.mlir`

**Line:** 32

**Comment:**

These are identity (or initial) value and index used for reduction.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp`

**Line:** 864

**Comment:**

I didn't originally write this logic for this function. You can refer to the comment here for the reasoning: https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Codegen/Utils/Utils.cpp#L1347.

I just moved the code from iree/compiler/src/iree/compiler/Codegen/Utils/Utils.cpp to compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp, following Mahesh’s branch: https://github.com/MaheshRavishankar/iree/commit/2e87623675840153525d21248d41beb57d516f82.

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp`

**Line:** 829

**Comment:**

I think that would be inconsistent with other methods here.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp`

**Line:** 829

**Comment:**

> I suggest using LogicalResult with better error messages so that if there's a case we want to support, people will know exactly where to add the support.

I'm curious about how it works in your mind? I.e, how do we propagate the error message to listener (or whatever stuff that I'm not familiar with)?

My feeling is that people write code like below, and I don't connect how returning `LogicalResult` connects to the code properly. The "better error message" is dropped in this kind of implementation.

```cpp
if (failed(isArgmaxOp(op)) {
  return rewriter.noitfyMatchFailure(op, "not a argmax op");
}
```

If this is the most common usage, I'm -1 returning LogicalResult.

(I personally don't like returning LogicalResult when possible. Boolean is simple and covers the needs for many cases, IMO. I use LogicalResult when I see the values or for style consistency.)

---

### Comment by pashu123

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Utils/Utils.cpp`

**Line:** 829

**Comment:**

The isArgmaxOp function is too big and contains many if-else's, just returning the boolean doesn't pinpoint the exact location inside the isArgmaxOp where the matcher isn't working.

---

### Comment by pashu123

**File:** `compiler/src/iree/compiler/DispatchCreation/SplitReduction.cpp`

**Line:** 104

**Comment:**

I would rather create a struct combinerOps with Operation* max, ..., so you can directly query the maxOp and don't care about the extra bookkeeping.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/DispatchCreation/SplitReduction.cpp`

**Line:** 37

**Comment:**

What is default argmax split-k?

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/DispatchCreation/SplitReduction.cpp`

**Line:** 142

**Comment:**

We follow camelCase style.

```suggestion
  int64_t outputDimSize = reductionDimSize / ratio;
```

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/DispatchCreation/SplitReduction.cpp`

**Line:** 373

**Comment:**

style nit:

```suggestion
      /*initOrAlloc=*/nullptr, /*fillOp=*/nullptr,
```

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/DispatchCreation/SplitReduction.cpp`

**Line:** 37

**Comment:**

You don't need to address this comment now. It is a question when I saw the code. Default is ambiguous in the splitReduction context.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/DispatchCreation/test/split_argmax_reduction.mlir`

**Line:** 1

**Comment:**

I'd put the test to `split_reduction.mlir`. I only split the files when it has too many test cases. When it happens, I start thinking if the pass has more responsibilities than it should. There are exceptions, though. This case does not fit my mental model about the split. So I suggest moving the test to split_reduction.mlir, and it is easier for others to find the test. (I think it is also how we structure the tests.)

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/DispatchCreation/SplitReduction.cpp`

**Line:** 97

**Comment:**

Why do we need the additional flag, but not just making `iree-dispatch-creation-split-argmax-reduction` default to 128? The other question is that do we want it happening by default? If so, we should document how 128 is derived. If it is an experimental number, we can also just mention it in the comment. People will ask why when they see the magic number, IMO.

My expectation is that the default value is 1 (and we don't need the `enableStaticArgmaxSplit` variable). You use `--iree-dispatch-creation-split-argmax-reduction=128` in your compilation config. We can set the default value to a reasonable value when we are confident for this.

(I don't work on this area, so maybe we are already confident about this. Then it is okay to set it to 128 by default.)

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/DispatchCreation/SplitReduction.cpp`

**Line:** 132

**Comment:**

It is not your responsibility because the code was bad in the first place. Could you help remove the `int64_t()` cast since you're touching the code? I think we don't need the cast at all. The type is already int64_t.

(side note: ideally, if we really need the cast, we should use c++ style like static_cast. We should not use c style casting.)

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 444

**Comment:**

style nit 1: don't use blank lines when you don't have to. resist starting functions with a blank line. https://google.github.io/styleguide/cppguide.html#Vertical_Whitespace
nit 2: ops is not used until the end. we should move the declaration down.
style nit 3: I'd use `cast` because you are not handling the 'failure'. We expect it is castable and we already have the assertion in the first line.

```suggestion
static FailureOr<ArgmaxCombinerOps>
collectArgmaxCombinerOps(linalg::GenericOp genericOp) {
  assert(isArgmaxOp(genericOp) && "expected operation to be an argmax op");
  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());

  // Extract max value producer: arith.maximumf.
  Value maxResult = yieldOp.getOperand(0);
  auto maxOp = dyn_cast<arith::MaximumFOp>(maxResult.getDefiningOp());

  // Extract index result producer: arith.select.
  Value indexResult = yieldOp.getOperand(1);
  auto selectOp = dyn_cast<arith::SelectOp>(indexResult.getDefiningOp());

  // Extract the condition of the select, expected to be arith.cmpf with
  // predicate OGT.
  auto cmpOp = dyn_cast<arith::CmpFOp>(selectOp.getCondition().getDefiningOp());

  ArgmaxCombinerOps ops;
  ops.maxOp = maxOp;
  ops.selectOp = selectOp;
  ops.cmpOp = cmpOp;
  return ops;
}
```

(I'd suggest aggregate initialization, e.g., `return {.maxOp  = maxOp, ...}`, if IREE is built with c++20. Unfortunately, we are not using c++20 atm..)

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h`

**Line:** 176

**Comment:**

Optional nit: remove the markdown style block. I've seen that people use it in IREE and MLIR, but I really don't get the point. I'm not a fan of using it because it adds visual noise to me. Marking it optional because there are no convention, IMO. We can just replace ``` with blank, and have indents for examples. I took a stab at the formatting a bit, please make the change if you think it looks better.

```suggestion
/// Example: original argmax op reducing over dim=512
///
///   %4:2 = linalg.generic {
///     indexing_maps = [...],
///     iterator_types = ["parallel", "reduction"]
///   } ins(%arg0 : tensor<?x512xbf16>)
///     outs(%out_val, %out_idx : tensor<?xbf16>, tensor<?xi64>) {
///   ^bb0(%in: bf16, %out: bf16, %out_0: i64):
///     %idx = linalg.index 1 : index
///     %cast = arith.index_cast %idx : index to i64
///     %max = arith.maximumf %in, %out : bf16
///     %cmp = arith.cmpf ogt, %in, %out : bf16
///     %sel = arith.select %cmp, %cast, %out_0 : i64
///     linalg.yield %max, %sel : bf16, i64
///   } -> (tensor<?xbf16>, tensor<?xi64>)
///
/// To: splitting K=512 into 4 x 128 + final argmax over the tile dimension
///     (dim=1 of ?x4)
///
///   %expanded = tensor.expand_shape %arg0 [[0], [1, 2]] : tensor<?x512xbf16>
///       into tensor<?x4x128xbf16>
///   %init_val = linalg.fill ins(%cst : bf16) outs(%empty : tensor<?x4xbf16>)
///   %init_idx = linalg.fill ins(%zero : i64) outs(%empty : tensor<?x4xi64>)
///   %partial:2 = linalg.generic {
///     indexing_maps = [...],
///     iterator_types = ["parallel", "reduction"]
///   } ins(%expanded : tensor<?x4x128xbf16>)
///     outs(%init_val, %init_idx : tensor<?x4xbf16>, tensor<?x4xi64>) {
///     // compute global index: outer_idx * 128 + inner_idx
///     ...
///   }
///   %final:2 = linalg.generic {
///     indexing_maps = [...],
///     iterator_types = ["reduction"]
///   } ins(%partial#0, %partial#1)
///     outs(%out_val, %out_idx : tensor<?xbf16>, tensor<?xi64>) {
///     // same combiner: maximumf, cmpf, select
///     ...
///   }
```



---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 418

**Comment:**

nit: remove `FailureOr`, because the assumption is that it is always a argmax-style op. It is just a local method and we already have the sanity check. We don't need to handle the failure for this, and it makes code easier.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 492

**Comment:**

How about renaming it to `combinerOps`? We dont need the `maybe` prefix after we remove `FailureOr`, and `ops` seem to be too vague.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 483

**Comment:**

nit 1: I think we should move the `outputDimSize` to the chunk that it is used, i.e., the place that you compute the output shapes/maps/etc. It is weird that we do the computation before the check. It could be dynamic value. In this case, we don't need to compute `outputDimSize` at all, right?

nit 2: switch to new preferred style, `ShapedType::isDynamic(reductionDimSize)`

picky nit for the error message: We typically start the first sentence with a lower-case letter, because it matches error message styles commonly produced by other tools

https://llvm.org/docs/CodingStandards.html#error-and-warning-messages

> Also, to match error message styles commonly produced by other tools, start the first sentence with a lower-case letter, and finish the last sentence without a period, if it would end in one otherwise.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 610

**Comment:**

IIUC, this part is generating the expanded operands and the properties of the partial reduction op. Furthermore, it creates a "folded" linalg.fill op on the expanded domain.

I think the logic can be much simpler because we have few more patterns help a lot. First, you can declare a `expandValue()` method that applies the expand_shape for all the operands, and the partial reduction op takes them as inputs and outputs. Here, I think you'd have a question about how to fold the expand shape away.

1. We have [canonicalization patterns](https://github.com/llvm/llvm-project/blob/31fd77aa51a643245f8eb277483554509b771832/mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp#L976-L983) in FillOp that swaps the (fill, reshape) pair.
2. We can explicitly populate the [patterns](https://github.com/llvm/llvm-project/blob/65a6cbde5bb074ce377cb1aa6145241ad1719c17/mlir/lib/Dialect/Tensor/Transforms/EmptyOpPatterns.cpp#L130C20-L138) that fold reshapes into empty ops. I don't know why upstream people do not put them EmptyOp canonicalization patterns, but they are valid to use IMO.
3. For creating the expand_shape op, you don't need to compute the dynamic values yourself. You can use [this builder](https://github.com/llvm/llvm-project/blob/65a6cbde5bb074ce377cb1aa6145241ad1719c17/mlir/include/mlir/Dialect/Tensor/IR/TensorOps.td#L1144-L1146) that takes result type and reassociation map. The new type can be constructed easily, so I'm not going to dig into details. After you get the type, you can use [getReassociationIndicesForReshape](https://github.com/llvm/llvm-project/blob/0d5124775cace200f6f99905989ebeb65853b16e/mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h#L67-L71C1) method to get the map, and use them to create the expand shape op. I checked that the upstream builder infers the output shape for you and they build the dynamic values for you.

This way, we can generate the expanded operands cleanly.

The next problem is about the generic op properties. For indexing maps, I think you can use `argmaxOp.getIndexingMapsArray()` to get all the indexing maps. It returns `SmallVector<AffineMap>` vector. Then you can declare a `getExpandedIndexingMap()` and iterate on all the indexing map to get the expanded indexing maps.

Then you get all the properties, except iterators. The iterator change is in below, so I'll leave the comment there.

I played a bit with the IR, I think you will generate something like below. (please ignore the transform script, it is just for kicking in the patterns that fold expand_shape away for empty ops.)

```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
module {
  util.func public @argmax(%arg0: tensor<?x131072xbf16>, %arg1: index) -> tensor<?xi64> {
    %cst = arith.constant 0xFF80 : bf16
    %c0_i64 = arith.constant 0 : i64
    %0 = tensor.empty(%arg1) : tensor<?xbf16>
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<?xbf16>) -> tensor<?xbf16>
    %2 = tensor.empty(%arg1) : tensor<?xi64>
    %3 = linalg.fill ins(%c0_i64 : i64) outs(%2 : tensor<?xi64>) -> tensor<?xi64>
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x131072xbf16>
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [%dim, 1024, 128] : tensor<?x131072xbf16> into tensor<?x1024x128xbf16>
    %c1024 = arith.constant 1024 : index
    %4 = arith.divsi %dim, %c1024 : index
    %expanded_0 = tensor.expand_shape %1 [[0, 1]] output_shape [%dim, 1024] : tensor<?xbf16> into tensor<?x1024xbf16>
    %expanded_1 = tensor.expand_shape %3 [[0, 1]] output_shape [%dim, 1024] : tensor<?xi64> into tensor<?x1024xi64>
    %5:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded : tensor<?x1024x128xbf16>) outs(%expanded_0, %expanded_1 : tensor<?x1024xbf16>, tensor<?x1024xi64>) {
    ^bb0(%in: bf16, %out: bf16, %out_2: i64):
      %7 = linalg.index 1 : index
      %8 = linalg.index 2 : index
      %9 = arith.muli %7, %c128 : index
      %10 = arith.addi %9, %8 : index
      %11 = arith.index_cast %10 : index to i64
      %12 = arith.maximumf %in, %out : bf16
      %13 = arith.cmpf ogt, %in, %out : bf16
      %14 = arith.select %13, %11, %out_2 : i64
      linalg.yield %12, %14 : bf16, i64
    } -> (tensor<?x1024xbf16>, tensor<?x1024xi64>)
    %6:2 = linalg.generic {indexing_maps = [#map2, #map2, #map3, #map3], iterator_types = ["parallel", "reduction"]} ins(%5#0, %5#1 : tensor<?x1024xbf16>, tensor<?x1024xi64>) outs(%1, %3 : tensor<?xbf16>, tensor<?xi64>) {
    ^bb0(%in: bf16, %in_2: i64, %out: bf16, %out_3: i64):
      %7 = arith.maximumf %in, %out : bf16
      %8 = arith.cmpf ogt, %in, %out : bf16
      %9 = arith.select %8, %in_2, %out_3 : i64
      linalg.yield %7, %9 : bf16, i64
    } -> (tensor<?xbf16>, tensor<?xi64>)
    util.return %6#1 : tensor<?xi64>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["util.func"]} in %arg0 : (!transform.any_op) -> !transform.op<"util.func">
      transform.apply_patterns to %0 {
        transform.apply_patterns.tensor.fold_tensor_empty
      } : !transform.op<"util.func">
      transform.yield
    }
  }
}
```

And I verified that the expand ops are folded away, if you run `iree-opt --canonicalize -transform-interpreter ~/repro.mlir`.

Does it make sense?

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 628

**Comment:**

I think we can use the `insert` method from SmallVector. Something like

```cpp
SmallVector<utils::IteratorType> newIteratorTypes = genericOp.getIteratorTypesArray();
newIteratorTypes.insert(newIteratorTypes.begin() + index, utils::IteratorType::parallel);
```

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 665

**Comment:**

I think the construction of `reductionIteratorTypes` can be easier, and it makes the below loop simpler.

```cpp
SmallVector<utils::IteratorType> reductionIteratorTypes(intermRank, utils::IteratorType::parallel);
reductionIteratorTypes[insertSplitIndex] = utils::IteratorType::reduction;
```

Then you dont need the else statement in the below loop.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 470

**Comment:**

`innerParallel` seems to be always false for argmax case, should we bail out when it is true?

Also, I'd remove the blank line in the between. They belong the same block to me.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/DispatchCreation/SplitReduction.cpp`

**Line:** 97

**Comment:**

> Why do we need the additional flag, but not just making `iree-dispatch-creation-split-argmax-reduction` default to 128? The other question is that do we want it happening by default? If so, we should document how 128 is derived. If it is an experimental number, we can also just mention it in the comment. People will ask why when they see the magic number, IMO.
> 
> My expectation is that the default value is 1 (and we don't need the `enableStaticArgmaxSplit` variable). You use `--iree-dispatch-creation-split-argmax-reduction=128` in your compilation config. We can set the default value to a reasonable value when we are confident for this.
> 
> (I don't work on this area, so maybe we are already confident about this. Then it is okay to set it to 128 by default.)

Here is the context: https://github.com/iree-org/iree/pull/20717#issuecomment-2856229221

We want to enable it by default so that the performance issue in #20650 can be solved. 

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/DispatchCreation/test/split_argmax_reduction.mlir`

**Line:** 45

**Comment:**

I missed a comment I had in my mind, we may want to check indexing maps as well.

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/DispatchCreation/SplitReduction.cpp`

**Line:** 97

**Comment:**

I think we need to have two knobs, the size of the reduction at which to apply split-k on and one to select the split to apply.

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/DispatchCreation/SplitReduction.cpp`

**Line:** 32

**Comment:**

I don't think we want these options if we are trying to turn it on by default. You need to make it two values

1. The minimum size of the reduction dimension that triggers the split
2. The size of (one of the) the split dimension, I think the size of the reduction dimension in the first operation.

You can set the value of 1 to default to 128*1024 and the second to be 128.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/DispatchCreation/SplitReduction.cpp`

**Line:** 32

**Comment:**

I see the reason of having two flags now, it makes sense to me. Thanks!

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 610

**Comment:**

I am confused about how this 
```
    %expanded_0 = tensor.expand_shape %1 [[0, 1]] output_shape [%dim, 1024] : tensor<?xbf16> into tensor<?x1024xbf16>
```
is generated in your provided mlir since from my end I noticed that association can not be computed from <?xbf16> to <?x1024xbf16>. 


---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 610

**Comment:**

Especially its is converted from %1
```
    %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<?xbf16>) -> tensor<?xbf16>
```

For example, assume the case in which %1 is of <1xbf16>. cc @hanhanW 

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 610

**Comment:**

Sorry @bangtianliu and good catch! I misunderstood the outs operands part in the algorithm, and here is the suggestion and the reasons behind the suggestion.

In the beginning, I thought that we could reuse the destination tensors (i.e., inits) because we should reuse the inits as much as possible. They could be `tensor.empty()` or other buffers. However, I was wrong on this. Fundamentally, it is not possible because we always need a storage to hold the data until we do the final reduction. (Vectorization is a different story, which could claim that generating `tensor.empty` is always valid. But I'm not going to expand the context here.)

Thus, we always need to create a `tensor.empty()` op and fill in the corresponding values. We can do it with few lines to achieve what you're doing here. Your implementation is correct, and my suggestion is making the code simpler.

My suggestion is to have `getSplitReductionInit(Value origInit, TypeAttr identity, int64_t insertDimSize, unsigned insertSplitIndex)` method and reuse it for all the outs operands. I'm assuming that all the outs need the same behavior. This is true for argmax, but it may be false for some linalg ops that I'm not aware of. Can always generalize it later when we see a need.

In the implementation, we create `arith.constant`, `tensor.empty`, and `linalg.fill ins(%cst) outs(%empty)` ops. This assumption is valid to me because it is a reduction-based generic op. The `arith.constant` op creation looks good to me. For `tensor.empty` op, what matters is the shape and the element type. You can use [tensor::getMixedSizes()](https://github.com/llvm/llvm-project/blob/27983696a6d6caf8f90d77745598aeaec88b7009/mlir/include/mlir/Dialect/Tensor/IR/Tensor.h#L133-L135) to get the shape with `SmallVector<OpFoldResult>` type, use the insert trick to insert the dimension, and create the `tensor.empty()` op with the [builder](https://github.com/llvm/llvm-project/blob/50316c1eb8ed189dc3a979d552b8b04b0730b287/mlir/include/mlir/Dialect/Tensor/IR/TensorOps.td#L311-L313). This builder takes `SmallVector<OpFoldResult>` shape and the element type, so you don't need to create the shapes like the current implementation. The internal implementation is very similar to what you have. The final implementation could be something like:

```cpp
// The getElementTypeOrSelf util looks a little silly as it is always a RankedTensorType, but it is helpful.
Type elemType = getElementTypeOrSelf(origInit.getType());
SmallVector<OpFoldResult> shape = tensor::getMixedSizes(builder, loc, origInit);
shape.insert(shape.begin() + insertSplitIndex, builder.getIndexAttr(insertDimSize));
Value emptyValue =
    rewriter.create<tensor::EmptyOp>(loc, shape, elemType);
Value identityValue =
    rewriter.create<linalg::FillOp>(loc, valueConst, emptyValue).getResult(0);
```

I will review the other parts tomorrow morning, and thanks for addressing my comments!

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 450

**Comment:**

nit: I'd replace the error message with something like `failed to infer reassociation indices from types`.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 478

**Comment:**

style nit: prefer early-exit to simplify the code and reduce levels of indents. https://llvm.org/docs/CodingStandards.html#use-early-exits-and-continue-to-simplify-code

E.g.,

```cpp
if (dim != reductionDim) {
  unsigned shifted = (dim < insertSplitDim) ? dim : dim + 1;
  exprs.push_back(getAffineDimExpr(shifted, ctx));
  continue;
}
// other code
...
```

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 458

**Comment:**

style nit: please add a doc, it is not trivial to understand what it returns based on names.

> Almost every function declaration should have comments immediately preceding it that describe what the function does and how to use it. These comments may be omitted only if the function is simple and obvious (e.g., simple accessors for obvious properties of the class). 


https://google.github.io/styleguide/cppguide.html#Function_Comments

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 490

**Comment:**

style nit: please add a doc. This is not obvious to me.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 546

**Comment:**

optional style nit: I'd remove the blank line in the between. They are reasonable to belong to the same chunk to me. (Mark it optional because maybe you have different thought.)

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 561

**Comment:**

optional nit: Same here, I'd remove the blank line.

nit 2: Different from LLVM, IREE always wraps single statement with braces. IREE follows mixed LLVM/Google style,  and this is one of few difference from LLVM style.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 568

**Comment:**

optional nit: we can use `getElementTypeOrSelf`. Mark it optional because there are no convention. I personally like it better in this place because it could be one-line code.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 585

**Comment:**

ditto: prefer early-exit style and try to save levels of indents. (you'll use `if (dim != reductionDim)` if you apply the change.)

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 589

**Comment:**

style nit: use auto because `XXX::get` already spells the type.

https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 596

**Comment:**

style nit: I think you accidentally add one more leading whitespace in the comment. :)
nit 2: can we rename it to `outputDimSize` for consistency?

```suggestion
  // The total number of output elements along this new dimension is
  // reductionDimSize / ratio.
  int64_t outputDimSize = reductionDimSize / ratio;
```

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 711

**Comment:**

style nit: use auto because `llvm::seq<>` already spells the type.
style nit 2: wrap simple single statement with braces.

```suggestion
  for (auto i : llvm::seq<unsigned>(0, intermRank)) {
    if (i != insertSplitIndex) {
      resultExprs.push_back(rewriter.getAffineDimExpr(i));
    }
  }
```

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 579

**Comment:**

Wrap with braces.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 446

**Comment:**

style: wrap with braces.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 449

**Comment:**

Sorry, I missed this in the previous reviews. I think we usually drop `mlir::` prefix in transformation implementation. This is why we have `iree_compiler` under `mlir` namespace.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 519

**Comment:**

nit: drop `llvm::` prefix for the casting.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 593

**Comment:**

I think this `if` is not necessary, right?

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/Dialect/LinalgExt/Transforms/SplitReduction.cpp`

**Line:** 587

**Comment:**

nit: I missed it in the previous reviews, use `auto`.

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/DispatchCreation/test/split_reduction.mlir`

**Line:** 71

**Comment:**

```suggestion
// Check partial reduction.
```

---

### Comment by hanhanW

**File:** `compiler/src/iree/compiler/DispatchCreation/test/split_reduction.mlir`

**Line:** 94

**Comment:**

```suggestion
// Check final reduction.
```

---

## PR #20438: [tuner] expose python binding for getting the tuner root ops

**URL:** https://github.com/iree-org/iree/pull/20438
**State:** MERGED

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 329

**Comment:**

Could you add two more test cases:
1. no root ops
2. two or more root ops

This is so that we can exercise the vector population logic.

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 315

**Comment:**

This test is not a dialect test, we should move it to a different file

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp`

**Line:** 1055

**Comment:**

https://llvm.org/docs/CodingStandards.html#commenting

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/tuner_test.py`

**Line:** 39

**Comment:**

I don't think python is supposed to modify IR like this [1]: there may be some dangling references to IR objects. Instead, can you make it a few separate test cases with their own individual inputs?

[1] https://mlir.llvm.org/docs/Bindings/Python/#ownership-in-the-core-ir

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/tuner_test.py`

**Line:** 21

**Comment:**

I think this code should be in a directory outside of `ir`, like `compiler/bindings/python/test/api/tuner_api_test.py`

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/api/tuner_api_test.py`

**Line:** 23

**Comment:**

Can we use implicit modules here to make this a bit shorter?

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/api/tuner_api_test.py`

**Line:** 33

**Comment:**

assert that the module parsed so that we can quickly identify issues with mlir syntax changes. Also in other testcases

---

## PR #20207: Integrates/llvm 20250310: Bump to llvm/llvm-project@967ab7e

**URL:** https://github.com/iree-org/iree/pull/20207
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMCPU/ConvertToLLVM.cpp`

**Line:** 986

**Comment:**

How did you decide these options? Could we also pick the default values for these new enums, e.g., `vector::VectorTransposeLowering()`?

Could you add the mapping to the PR description?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/LLVMCPU/ConvertToLLVM.cpp`

**Line:** 986

**Comment:**

I typically just pick up the default value there, for example vector::VectorTransformsOptions().VectorTransposeLowering is  VectorTransposeLowering::EltWise as pointed here: https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Vector/Transforms/VectorTransforms.h#L50.

Sure, I will add the mapping to PR description.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMCPU/ConvertToLLVM.cpp`

**Line:** 986

**Comment:**

Why not do this then to pick up the defaults:
```c++
    VectorTransformsOptions defaultOptions;
    ...
    vector::populateVectorTransposeLoweringPatterns(
        patterns, defaultOptions.vectorTransposeLowering);
 ```
 


---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/LLVMCPU/ConvertToLLVM.cpp`

**Line:** 986

**Comment:**

> Why not do this then to pick up the defaults:
> 
> ```c++
>     VectorTransformsOptions defaultOptions;
>     ...
>     vector::populateVectorTransposeLoweringPatterns(
>         patterns, defaultOptions.vectorTransposeLowering);
> ```

Yeah, what you proposed is a good approach,  but my concern is that all the other variables inside the options are not be used.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMCPU/ConvertToLLVM.cpp`

**Line:** 986

**Comment:**

This is trivial to be optimized out for any c++ compiler -- just SROA and DCE

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/LLVMCPU/ConvertToLLVM.cpp`

**Line:** 986

**Comment:**

Sure, then I will do the changes you suggested for picking up default values. 

---

## PR #20173: [Codegen][Tuner] improve verifier for the default attribute

**URL:** https://github.com/iree-org/iree/pull/20173
**State:** MERGED

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 192

**Comment:**

Can you update the comments above?

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 193

**Comment:**

Should this also check that there is only a single named_sequence with the `kTuningSpecEntrypointAttrName` attr?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 261

**Comment:**

Do we need these to check for this error? Let's make sure we minimize the amount of unrelated code and keep the tests minimal

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 77

**Comment:**

b. should be on a new line

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 99

**Comment:**

What if there are other ops like `transform.include`?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 99

**Comment:**

In the current verification code, we only check the number of `foreach_match` ops. This means that if there are one `transform.include` and one foreach_match op, the verifier will still consider it legal.

What are your thoughts on this? Do you think we should add additional checks for `transform.include` as well?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 99

**Comment:**

I'm thinking whether requiring a single for_each only would make merging simpler

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 99

**Comment:**

 IMO, it should be. Currently, we do not encounter multiple foreach_match operations in our tuning scenario.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 99

**Comment:**

Let's check for this then

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/materialize_tuning_specs.mlir`

**Line:** 14

**Comment:**

I think this effectively break this test. This is fine, but we shoulda add a TODO to fix it once the merging logic can handle foreach_match ops.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 134

**Comment:**

We should add a test that has a foreach_match op in `__kernel_config` and some other op like print or include.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 112

**Comment:**

What if we find ops that are not named sequences?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 123

**Comment:**

What if there's a single foreach_match but also some other ops like `transform.include`?

---

### Comment by kuhar

**File:** `docs/website/docs/reference/tuning.md`

**Line:** 129

**Comment:**

```suggestion
  the tuning spec includes a named sequence op with name `__kernel_config`, which
  must contain exactly one `foreach_match` op.
```
this seems redundant to me

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/test/default_tuning_specs_amdgpu.mlir`

**Line:** 39

**Comment:**

Can we add a TODO to revisit this test and re-add this CHECK when the new linking lands?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 86

**Comment:**

Also here: let's keep track of re-enabling this

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/tuning_spec_default.mlir`

**Line:** 14

**Comment:**

nit: put match on its own line -- I find it weird to have the whole loop with its body on a single line

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 95

**Comment:**

Can we add some clue as to where this rule comes from? This is related to the attributes you set. Otherwise the `__kernel_config` name is not special on its own.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 85

**Comment:**

We don't need this named sequence in this test -- we can check the same thing by having two foreach_match ops that use the same match function

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 109

**Comment:**

Also here: we should print which attribute adds this verification rule. It's not `tuning_spec_entrypoint` that's at issue

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 140

**Comment:**

Two yields don't make sense -- I'd expect this to be a terminator. Instead of checking that there is exact one yield and one foreach_match, we can check that the only two ops are foreach_match and yield without counting how many there are.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 170

**Comment:**

The quote is not closed

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 193

**Comment:**

Something is wrong with this error message -- the closing quote is missing and there's no space before 'but'.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 220

**Comment:**

Also here: let's print where this rule comes from

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 245

**Comment:**

This looks like an ill-formed forech_match op. It should check that the return type makes sense. Should we fix the `transform_match` verifier instead?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 263

**Comment:**

Also here: we should either tighten the verifier on `transform.foreach_match` or say which attribute adds this verification rule

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 290

**Comment:**

Also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 305

**Comment:**

Print which attribute adds this rule

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 326

**Comment:**

Also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 134

**Comment:**

https://llvm.org/docs/CodingStandards.html#prefer-preincrement

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 133

**Comment:**

Instead of counting these ops, we should check that there are exactly two ops: one foreach_match, one yield.

---

### Comment by kuhar

**File:** `docs/website/docs/reference/tuning.md`

**Line:** 130

**Comment:**

Why would someone add these attributes? To me it sounds like too much of an implementation detail to mention here.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 93

**Comment:**

Now what this logic became quite complex, I think we should pull this into a separate function with helper function that can check each condition. For example, one function can check that's there's exactly one default entrypoint while another one can check its contents.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 92

**Comment:**

In the other PR you used this:
```c++
module.getBody()->getOps<T>()
```
I think this should also work here?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 86

**Comment:**

```suggestion
  // TODO: Re-enable default attribute as below once new linking lands.
```
nit: Use proper capitalization per https://llvm.org/docs/CodingStandards.html#commenting.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/tuning_spec_default.mlir`

**Line:** 6

**Comment:**

We can make match or apply_op_config print the 'hello' message -- I found it useful when debugging this for the first time 

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 150

**Comment:**

yield is a terminator so this should come before

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 169

**Comment:**

This is already covered by the previous test case that checks for print. We can drop this one.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 194

**Comment:**

Also here, we already have a test like this before

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 57

**Comment:**

Return `numTuningEntrypoints` instead of using an out parameter

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 57

**Comment:**

Only use the `k` prefix for compile-time constants. Also, you can pass this as StringRef

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 97

**Comment:**

Also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 100

**Comment:**

You can reduce the nesting here by erroring out when there's more than one block

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 102

**Comment:**

I'd drop this loop and use the iterators directly to make sure there are exactly two ops.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 204

**Comment:**

The variable name is misleading in two ways: it's not a compile time constant, so we shouldn't use the `k` prefix, and this is a string instead of an attribute. I'd call it something like `requiredByDefaultAttrMessage` or `requiredByMessage` or `requiredByErrorSuffix`.

You can also construct this with `llvm::formatv`, but either way is fine IMO.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 204

**Comment:**

Also, we typically don't end error messages with `.` in mlir

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 109

**Comment:**

Use `hasNItems` from https://llvm.org/doxygen/STLExtras_8h.html for size checks in linked lists. This way it won't iterate over the whole thing once you know there are more than 2 instructions.

---

## PR #20127: [Codegen][Tuner] merge the default td specs

**URL:** https://github.com/iree-org/iree/pull/20127
**State:** MERGED

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/test/default_tuning_specs_amdgpu.mlir`

**Line:** 49

**Comment:**

nit: indent these to match the print format
```suggestion
// BOTH:           @match_mmt -> @apply_mmt_op_config
// BOTH-NEXT:      @match_attention_2x10x4096x64x64x64_f16 -> @apply_attn_op_config
// BOTH-NEXT:      @match_mmt_2048x1280x5120_f16_f16_f32 -> @apply_op_config
```

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/test/default_tuning_specs_amdgpu.mlir`

**Line:** 49

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

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/test/tuning_spec_mmt_tile_and_fuse.mlir`

**Line:** 4

**Comment:**

Why are we changing this? Do you think we should disallow tuning specs without default entrypoints?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 90

**Comment:**

Do we remove it because the entrypoint name can change? If that's the case, I think we'd want to iterate over the parents of `specsToLink` instead, since this functions doesn't assume a specific nesting structure.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 405

**Comment:**

Prefer pre-increment: https://llvm.org/docs/CodingStandards.html#prefer-preincrement

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 407

**Comment:**

also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 401

**Comment:**

I don't understand this variable name. What do these modules match?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 402

**Comment:**

I'd put this variable first since you update it unconditionally in the loop below.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 401

**Comment:**

Maybe `numDefaultEntrypoint`?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 152

**Comment:**

Can we call the variable something like `namedSequenceToForeach` so that we don't need this comment?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 250

**Comment:**

This function is very long -- can we outline this loop to a helper function?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 163

**Comment:**

You don't need to explicitly initialize IR types to nullptr.
```suggestion
        transform::ForeachMatchOp foreachMatch;
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 164

**Comment:**

This needs a more descriptive name

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 170

**Comment:**

also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 179

**Comment:**

```suggestion
        if (matchCount == 0 || matchCount > 1) {
          return failure();
        }
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 188

**Comment:**

Use `.contains(...)` for map types

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 199

**Comment:**

also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 187

**Comment:**

Can this ever not be a named sequence op?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 214

**Comment:**

```suggestion
  auto expectedResultTypes =
      llvm::to_vector_of<Type, 4>(foreachMatchOps.front()->getResultTypes());
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 257

**Comment:**

This could also be a helper function

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 262

**Comment:**

Instead, should we check that the result type is exactly what we expect? I think it must take any_op and return any_op.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 228

**Comment:**

also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 270

**Comment:**

We have the same logic elsewhere: could we make it a helper function?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 269

**Comment:**

also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 267

**Comment:**

What do you mean by 'if there's a reference'?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 324

**Comment:**

Make this a helper function

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 345

**Comment:**

Do not re-evaluate the end iterator: https://llvm.org/docs/CodingStandards.html#don-t-evaluate-end-every-time-through-a-loop

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 351

**Comment:**

Use `llvm::is_contained`

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 344

**Comment:**

You can make this more readable with structured bindings

---

### Comment by bangtianliu

**File:** `compiler/plugins/target/ROCM/test/tuning_spec_mmt_tile_and_fuse.mlir`

**Line:** 4

**Comment:**

No,  I think we need both. I will add a tuning_spec file with a default attribute to do the test.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 187

**Comment:**

No, it is always a NamedSequenceOp.

---

### Comment by Max191

**File:** `compiler/plugins/target/ROCM/test/default_tuning_specs_amdgpu.mlir`

**Line:** 18

**Comment:**

Why does this need `--mlir-disable-threading`?

---

### Comment by Max191

**File:** `compiler/plugins/target/ROCM/test/tuning_spec_mmt_tile_and_fuse.mlir`

**Line:** 3

**Comment:**

Was this accidental?

---

### Comment by Max191

**File:** `compiler/plugins/target/ROCM/test/default_tuning_specs_amdgpu.mlir`

**Line:** 66

**Comment:**

```suggestion
// materialized, in which nested structure should not be present, and a merged foreach_match op
// should exist. The user spec should have precedence over the default one.
```

---

### Comment by Max191

**File:** `compiler/plugins/target/ROCM/test/default_tuning_specs_amdgpu.mlir`

**Line:** 81

**Comment:**

nit: use `MERGE-DAG` for these, or note in a comment that the order does matter (it seems like order does matter to me, but someone else looking at the code might not realize this).

---

### Comment by Max191

**File:** `compiler/plugins/target/ROCM/test/default_tuning_specs_amdgpu.mlir`

**Line:** 88

**Comment:**

nit: No need to use `MERGE-LABEL` here, since there are no checks following it.

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 419

**Comment:**

nit: move `tuningSpecs` to where it is used below.

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 202

**Comment:**

nit: Save a level of nesting with:
```
if (namedSequenceOp.getSymName() != kKernelConfigSpecName) {
  namedSequenceOpsToMove.push_back(namedSequenceOp);
  continue;
}
```

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 202

**Comment:**

Also, it seems safer to also check that the names_sequence has the `iree_codegen.tuning_spec_entrypoint` attr.

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 220

**Comment:**

This could just be:

```
auto foreachOpIter = namedSequenceOp.getOps<transform::ForeachMatchOp>();
if (!llvm::hasSingleElement(foreachOpIter)) {
  // Warning + return failure
}
foreachMatchOps.push_back(*foreachOpIter.begin());
```

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 226

**Comment:**

nit: You can use `SymbolTable::lookupNearestSymbolFrom<transform::NamedSequenceOp>`

Also, if the symbol table might not find the reference, then this shouldn't use cast (using the templated lookup in my suggestion will take care of this).

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 223

**Comment:**

nit: Save some nesting here with early return/continue.

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 234

**Comment:**

Same here. Use early returns to save nesting

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 237

**Comment:**

`SymbolTable::lookupNearestSymbolFrom<transform::NamedSequenceOp>`

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 92

**Comment:**

nit: Move `resultTypes` to where it is used below.

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 90

**Comment:**

nit: I think you can just do `SmallVector<Type> argTypes(foreachMatchOp.getOperandTypes())`?

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 412

**Comment:**

This can be something like:
```
if (!llvm::all_equal(llvm::map_range(foreachMatchOps,
    [](transform::ForeachMatchOp matchOp){ return matchOp.getRestrictRootAttr(); }))) {
  // Emit warning
  return failure();
}
if (!llvm::all_equal(llvm::map_range(foreachMatchOps,
    [](transform::ForeachMatchOp matchOp){ return matchOp.getFlattenResultsAttr(); }))) {
  // Emit warning
  return failure();
}
```

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 263

**Comment:**

I think this could have conflicts if we do multiple consecutive merges. Say you have:

```
module @outer_module {
  // From merged spec
  module @inner_module_from_previous_merge {
    transform.named_sequence @conflicting_reference ...
    // The `_1` was added from the previous merge.
    transform.named_sequence @conflicting_reference_1 ...
  }
  module @new_inner_module {
    // This will become `@conflicting_reference_1`, which will conflict with the reference from the above module.
    transform.named_sequence @conflicting_reference ...
  }
}
```

When you form a new unique name, you need to check that it doesn't collide with any existing names, and handle the case when it does.

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 263

**Comment:**

Also please add a test for this case.

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 301

**Comment:**

nit: Save a level of nesting with early return.

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 342

**Comment:**

nit: Move to a helper function and reuse it above too.

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 355

**Comment:**

```suggestion
  // Step 3-C: Create a new block inside the NamedSequenceOp and merge the
  // ForeachMatchOp from each inner module into one ForachMatchOp.
```

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 351

**Comment:**

```suggestion
  // Indicate that the output module is a default tuning spec after merging.
```

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 367

**Comment:**

nit: Omit the `mlir` namespace from cast.

---

### Comment by bangtianliu

**File:** `compiler/plugins/target/ROCM/test/default_tuning_specs_amdgpu.mlir`

**Line:** 18

**Comment:**

No, no need and it can be removed. I follow the example from the above test here. Should we remove all of them from this test file?

---

### Comment by bangtianliu

**File:** `compiler/plugins/target/ROCM/test/tuning_spec_mmt_tile_and_fuse.mlir`

**Line:** 3

**Comment:**

yes, lol


---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/test/default_tuning_specs_amdgpu.mlir`

**Line:** 83

**Comment:**

```suggestion
// NOTE: The order matters above because `foreach_match` ops performs matching from top to bottom.
```

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/test/tuning_spec_mmt_tile_and_fuse_default.mlir`

**Line:** 21

**Comment:**

Why do we need full compilation info in this test?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 84

**Comment:**

Use `.contains` with set/map

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 86

**Comment:**

This seem wasteful: not only we keep looping, but each time we also construct a new string. We should be able to come up with a unique name on the first try.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 69

**Comment:**

```suggestion
            dyn_cast<transform::NamedSequenceOp>(parentOp)) {
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 66

**Comment:**

```suggestion
  if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 96

**Comment:**

Can we check this in the verifier?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 149

**Comment:**

Could you explain **why** we are trying to do this? 

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 250

**Comment:**

Could you make it a helper function to improve readability?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 257

**Comment:**

Can we check this in the verifier?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 266

**Comment:**

Can we check this in the verifier?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/link_tuning_specs.mlir`

**Line:** 202

**Comment:**

This test shouldn't have to do anything related to attention. Let's try to keep the linking tests minimal.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 266

**Comment:**

I think no. this check is intended for merging foreach_map operations across modules, whereas the verifier operates at the single-module level. However, this is just my understanding—please correct me if I'm mistaken.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 266

**Comment:**

You won't check that all are the same but you can check that each is exactly what you expect it to be. What is the value of `restrictRoot` and `flattenResults` that we want? Are these just unit attributes? If yes, we can require them to be present/absent.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 266

**Comment:**

Yes, they are unit attributes. We can require them to be absent for now based on the generated specs generated in tuner. 

---

### Comment by andfau-amd

**File:** `compiler/plugins/target/ROCM/test/default_tuning_specs_amdgpu.mlir`

**Line:** 4

**Comment:**

Removing threading seems like an unrelated change?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 101

**Comment:**

nit: use and `if` statement instead. This way you can put a breakpoint on the return statement of interest

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 114

**Comment:**

Could you define a struct with this info that can be directly returned. This should improve readability.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 126

**Comment:**

We should assert that this is actually a non-null op. In case the verifier changes in the future and we find some other op here.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 169

**Comment:**

Outline this to a helper function that takes a single foreach_match and returns its info

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 248

**Comment:**

Can you add a comment with an example of the op that's being returned?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 289

**Comment:**

I'd take this logic out of the push back to make debugging easier. It's simpler to print a variable than to print an element in a vector

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 246

**Comment:**

I don't think we need to store it in a vector -- we could as well query the foreachMap ops as we iterate, no?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 247

**Comment:**

Also here: can't we query this at runtime?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 362

**Comment:**

Could we move this to a helper function?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 315

**Comment:**

nit: why capitalize C here but not in the two previous steps: 2-a and 2-b?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 318

**Comment:**

Do we need the `TypeRange`?
```suggestion
  SmallVector<Type, 4> resultTypes = {anyOpType};
```

This should work, no?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 333

**Comment:**

When would this create duplicates? Can you give an example?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 329

**Comment:**

```suggestion
    for (auto [matcher, action] : llvm::zip_equal(foreachMatchOp.getMatchers(), foreachMatchOp.getActions())) {
      matcherActionPairs.push_back(
          {cast<SymbolRefAttr>(matcher), cast<SymbolRefAttr>(action)});
    }
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 343

**Comment:**

Wait, why did we put these in a vector of pairs to decompose it to matchers and actions again? Couldn't we keep them separate from the start?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 246

**Comment:**

As we iterate, the IR structure changes—for example, some NamedSequenceOps are moved outside. This means iterating again would require additional handling, making the code more complex. Keeping track of everything at the beginning avoids redundant queries and simplifies the logic.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 246

**Comment:**

But it's always the first op in the named sequence -- should be very quick to look up

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 246

**Comment:**

Sure, I will apply this comment too. 

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 79

**Comment:**

```suggestion
// - If `specName` ends with `_<number>`, the base name is everything before
//   `_`.
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 90

**Comment:**

Use proper punctuation: https://llvm.org/docs/CodingStandards.html#commenting

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 114

**Comment:**

What do you mean by `updates`?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 107

**Comment:**

Why do we need the `ForeachMatchOp` as values? Couldn't we query them when needed?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 157

**Comment:**

```suggestion
      auto foreachMatch =
```
See https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 179

**Comment:**

What creates these `_{num}` suffixes? Could we not generate them and then not have to run `getBaseName`?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 193

**Comment:**

Update would suggest this performs some actions and modifies the attribute. But because attributes are immutable, I'd expect it to create a new one instead. Maybe `getUpdatedSymbol` instead?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 200

**Comment:**

```suggestion
  SmallVector<Attribute> updatedMatchers;
  SmallVector<Attribute> updatedActions;
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 412

**Comment:**

Could we make this a helper function that given a named sequence op returns its unique foreach match op?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 354

**Comment:**

Write the full type here since it's not obvious based on the context: https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 441

**Comment:**

use `zip_equal`: https://llvm.org/docs/ProgrammersManual.html#iterating-over-ranges

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 107

**Comment:**

If we don’t store `ForeachMatchOp` as values, we would always need to first traverse to the parent ModuleOp, locate the `__kernel_config` named sequence op, and then find the corresponding `ForeachMatchOp` to update its name when resolving name conflicts.

Maybe you have a better solution?


---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 107

**Comment:**

We will only have to traverse NamedSequenceOp and find the first op inside, no?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 107

**Comment:**

Yes, but each query requires one traveral for locating the `__kernel_config` named sequence op.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 107

**Comment:**

I can't reconcile this with the name: does this map the named sequence op to the foreach map op inside or to a use in some other foreach op outside?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 107

**Comment:**

For this example IR 
```
module @inner_module_a
    attributes { transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint } {
    transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
      transform.yield %arg : !transform.any_op
    }

    transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}) {
      transform.yield
    }

    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.consumed})
      -> (!transform.any_op) attributes { iree_codegen.tuning_spec_entrypoint } {
      %res = transform.foreach_match in %arg0 @match -> @apply_op_config
        : (!transform.any_op) -> (!transform.any_op)
      transform.yield %res : !transform.any_op
    }
  }
  ```
  
  mapping  the named sequence op `match` to ` transform.foreach_match` inside `__kernel_config`
  and  the named sequence op `apply_op_config` to  ` transform.foreach_match` inside `__kernel_config`.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 107

**Comment:**

I'd call it something like `namedSequenceToUser` where the user is a foreach_match op (based on the type). The details can be in a comment.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 179

**Comment:**

The way we generate names to resolve conflicts results in `_{num}` suffixes:
`llvm::formatv("{}_{}", specBaseName, specNameSeenCount).str();
`
If we want to avoid these suffixes, we need to explore an alternative naming strategy.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 179

**Comment:**

What creates these _{num} suffixes? Can you link to the code?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 179

**Comment:**

https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp#L115

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 179

**Comment:**

FYI, this code is added to address this comment:

https://github.com/iree-org/iree/pull/20127#discussion_r1981657536


---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 179

**Comment:**

The comment above says this is only needed to work around a `transform.include` bug: https://github.com/iree-org/iree/blob/4984d92991ee03a1e14e510b539e0fea8fdbec34/compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp#L95-L97. If we don't use `transform.include` anyway, can we avoid creating these suffixes?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 179

**Comment:**

Yes, but we still need to propose some stuff there to address the name conflict. 

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 179

**Comment:**

Ok,  I remember the code generation cannot handle it automatically, But I will have a try first to see whether the name conflicts really exist. Then think about how to address the name conflict if it exits. 

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 179

**Comment:**

I'd think we can add suffixes when we move named sequences from nested modules to parent modules.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 179

**Comment:**

Then it is what the code does now: `getUniqueSpecName` (the code you mark here) is used inside  function `updateNamedSequenceOp`, which are used for the named sequences to be moved from nested modules to parent modules.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 179

**Comment:**

I'd like to avoid setting the suffix in one place just to forget it and try to recover this information in `getBaseName` and update again in `getUniqueSpecName`. This is bad design.

I'd rather do one of:
1. only set it under a flag in  `emitLinkedTuningSpec` and decide it in the code that does merging if it has different naming requirements
2. add a different name uniqing mode to `emitLinkedTuningSpec` to do the right thing for merging
3. update all names as a pre-processing steps that's common across both code paths (linking and merging)


---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 236

**Comment:**

You can use `hasSingleElement` in the assert, or replace the whole thing with `getSingleElement` if our llvm is recent enough: https://github.com/llvm/llvm-project/pull/131508 . I see there was an integrate earlier today so this should be available when you rebase this PR.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 103

**Comment:**

These two functions appear unused now. Can we delete them or did I miss something?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 149

**Comment:**

I think this is already covered by the comments in the struct definition. Do we need it here?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 162

**Comment:**

Use `getSingleElement`?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 161

**Comment:**

I don't think this comments adds much value. Focus on documenting the __why__ instead of the __how__.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 236

**Comment:**

If you don't add/remove elements, pass this as `ArrayRef`

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 279

**Comment:**

This is already checked by `getSingleElement`, no need to assert again.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 410

**Comment:**

How do we know the first op is the correct named sequence?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/link_tuning_specs.mlir`

**Line:** 260

**Comment:**

Can you update this test and change the order of named sequences? match doesn't have to be the first op, while kernel_config doesn't always have to be the last one IIUC.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 103

**Comment:**

yeah, you are right.

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 156

**Comment:**

nit: The verifier of the module guarantees that there is only one op with `kTuningSpecEntrypointAttrName`, so we can omit the SymName check.

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 140

**Comment:**

nit: These loops are doing the same thing. Can you just make this a single loop that iterates over the combined matchers and actions?

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 405

**Comment:**

I think something like this would be more descriptive:
```suggestion
  // Collect the `ForeachMatchOp`s from the entry point named sequences of
  // each inner module to merge into the new default entry point in the top
  // module.
```

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 402

**Comment:**

```suggestion
  // Step 2-b: Create a new entry point NamedSequenceOp called `__kernel_config` in
  // the top-level module.
```

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 411

**Comment:**

```suggestion
    foreachMatchOps.push_back(getForeachMatchOpFromKernelConfig(namedSequenceOp));
```

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 417

**Comment:**

nit: rename `newNamedSequence` to `newEntryPoint`

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 426

**Comment:**

```suggestion
  // Indicate that the outer module is a default tuning spec after merging.
  module->setAttr(kTuningSpecDefaultEntrypointAttrName, builder.getUnitAttr());
```

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 138

**Comment:**

Please add some more docs explaining the conflict resolution strategy (i.e., prefixing the symbol names with the symbol name of its containing module, or an incrementing counter for unnamed modules).

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 81

**Comment:**

The format is weird -- why repeat the field name and make it a bullet point?
```suggestion
  // Contains `NamedSequenceOp`s that either:
  //  - Are not named `__kernel_config`.
  //  - Do not have the `iree_codegen.tuning_spec_entrypoint` attribute.
  SmallVector<NamedSequenceOp> namedSequenceOpsToMove;
```

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 103

**Comment:**

Please remove this, since it seems to be unused.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 99

**Comment:**

```suggestion
    if (auto matcherSymRef = dyn_cast<SymbolRefAttr>(matcher)) {
```

---

### Comment by Max191

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 103

**Comment:**

Please remove this, since it seems to be unused.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 98

**Comment:**

Use the actual type here, since it's not obvious based on the context: https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 109

**Comment:**

```suggestion
    if (auto actionSymRef = dyn_cast<SymbolRefAttr>(action)) {
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 108

**Comment:**

Use the actual type here, since it's not obvious based on the context: https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 134

**Comment:**

I don't think this assertion will ever trigger -- if there are no foreach match ops, the list will be empty, not filled with nullptr values

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 190

**Comment:**

Use the actual type here, since it's not obvious based on the context: https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 195

**Comment:**

Use the actual type here, since it's not obvious based on the context: https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 407

**Comment:**

Use the actual type here, since it's not obvious based on the context: https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

## PR #20081:  [Codegen][Tuner]: remove attrs inside decomposeConfig for attention op

**URL:** https://github.com/iree-org/iree/pull/20081
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 74

**Comment:**

```suggestion
    if (DictionaryAttr decompositionConfig =
        attentionOp.getDecompositionConfigAttr()) {
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 75

**Comment:**

It seems a little bit confusing to me to reuse the same variable even though it's not a reference and won't update the attribute itself. Can we create a new one instead?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 77

**Comment:**

can we use `llvm::filter_to_vector`?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 77

**Comment:**

I'd make this vector a local variable to reduce the overall nesting.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir`

**Line:** 106

**Comment:**

Should we check have a check with `use_exp2` to make sure we don't drop it? I assume that this is accomplished by the `z` unit attr above, but this seems like something easy to miss to me.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 76

**Comment:**

```suggestion
          llvm::filter_to_vector(decompositionConfig, [](NamedAttribute attr) {
```
AFAICT we don't need to capture anything

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 74

**Comment:**

nit: this is a bit of a mouthful and I don't think we need to be that descriptive here
```suggestion
      DictionaryAttr newConfig = DictionaryAttr::get(
```


---

## PR #20072: [Codegen][Tuner] add support for attention op in the StripCompilationInfoPass

**URL:** https://github.com/iree-org/iree/pull/20072
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 60

**Comment:**

```suggestion
struct StripAttentionOpCompilationInfo final
    : OpRewritePattern<IREE::LinalgExt::AttentionOp> {
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 86

**Comment:**

```suggestion
    patterns.add<StripFuncOpTranslationInfo, StripLinalgOpCompilationInfo, StripAttentionOpCompilationInfo>(ctx);
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 66

**Comment:**

There's no test for this

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir`

**Line:** 115

**Comment:**

This checks for IREE attributes instead of keys in the attribute dictionary. A potential issue is that without `--mlir-print-ir-scope`, these attributes may be outlined **above the function** just like in the input IR.

Have you tried adding a new attribute, say `foo = #compilation`, and checking that the test does fail when the attribute remains in the output?

I'd think that we should check for the dictionary keys and/or print with local scope.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir`

**Line:** 115

**Comment:**

> Have you tried adding a new attribute, say `foo = #compilation`, and checking that the test does fail when the attribute remains in the output?

No, I haven’t tried that yet. I’ll test it first to gain a better understanding.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir`

**Line:** 115

**Comment:**

Yes, if using `foo = #compilation`, the attribute remains in the output.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir`

**Line:** 113

**Comment:**

Can you also match the attention op?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir`

**Line:** 76

**Comment:**

Also here, it would be nice to check that the attention op is there

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir`

**Line:** 76

**Comment:**

 I added a check for this one.  

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 74

**Comment:**

This is wrong. DecompositionConfig is part of the operation's definition. You can only drop qk_attrs/pv_attrs in it. This can cause miscompiles.

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 74

**Comment:**

For example, the decomposition config can contain a "use_exp2" attribute, which specifies that the operation must use exp2 during lowering. If you use this patch, you would get numerically incorrect result.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 74

**Comment:**

Thanks for pointing it out and I will send another PR to fix it.

---

## PR #20039: [Codegen][Tuner] add attention op into default tuning spec

**URL:** https://github.com/iree-org/iree/pull/20039
**State:** MERGED

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 34

**Comment:**

Does this work for any attention size?

---

### Comment by bangtianliu

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 34

**Comment:**

I think it should handle any sizes as the size constraint relies on `tensor<?x?x?x?xf16>`. But I need to add the correctness tests. 

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 34

**Comment:**

That's the source of my concern -- it applies to any attention size

---

### Comment by bangtianliu

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 34

**Comment:**

I have no idea about it. The context is that I borrowed it from https://github.com/iree-org/iree/blob/main/build_tools/pkgci/external_test_suite/attention_and_matmul_spec_punet_mi300.mlir#L48.

---

### Comment by manupak

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 34

**Comment:**

What is the expectation/goal here ? 
a) improve the chances this is applied
b) if its applied, make sure its correct.

Note that this will only work for 6 dimensional attention variant which is usually b1, b2, m, n, k1, k2 where each input has [b1, b2, x, x].  Therefore, I think b) would work for most such cases while the rest it wont be applied.

For a),
talking about the rest, lately, I ve been seeing b1, b2 being collapsed to a single b.
So its b, m, n, k1, k2 where each input is [b, x, x].
If we want to cover that, we need another matcher with partially collapsed #iree_gpu.lowering_config.

cc: @MaheshRavishankar @Groverkss 

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 34

**Comment:**

> What is the expectation/goal here ?

To get good performance out of the box on attention in models we care about and don't error out on other attention variants we haven't seen.

This is the first step towards learning how to pick good default specs and how to test them, so that we figure out some process that we can use later on to add more default tuning specs for key contractions etc.

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 34

**Comment:**

My intuition is to start small and match exactly the attention shapes we've tested this on. Later on we can do a sweep of attention sizes to see if we can have something more general.

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 163

**Comment:**

Can you add a comment with the expected speedup (for posterity)?

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 74

**Comment:**

I thought we wanted to do 1, 1, 128 for this shape?

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/builtins/tuning/test/spec_gfx942.mlir`

**Line:** 88

**Comment:**

Can we simplify this test and remove parts that are not necessary?

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/builtins/tuning/test/spec_gfx942.mlir`

**Line:** 77

**Comment:**

I think we should also have at least one negative test where the attention config is expected *not to apply*

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/builtins/tuning/test/spec_gfx942.mlir`

**Line:** 98

**Comment:**

I wonder if we could further reduce this by making these function arguments (even if it doesn't make sense in full compilation) and drop all of these hal and flow ops?

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 140

**Comment:**

```suggestion
    // Expected speedup: 1.22x.
```

---

### Comment by Groverkss

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 46

**Comment:**

can you rename these dimensions? The op becomes much easier to read that way:

d0, d1: B0, B1
d2: M
d3: N
d4: K1
d5: K2

---

### Comment by Groverkss

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 46

**Comment:**

I'm not sure why we are starting out with a variant that has 2 batches. This points to the fact that we need more transform matching maps than just cast_comptible_dag_from_root. It's okay for this patch, but if we want to add more attention variants, we need something better.

---

### Comment by Groverkss

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 71

**Comment:**

This spec works for:

query: tensor<?x?x[128]x[16]xf16>
key: tensor<?x?x[64]x[16]xf16>
value: tensor<?x?x[16]x[64]xf16>

Here, [x] means the dimension is a multiple of x and `?` means it can be anything. Can you check for this instead here?

---

### Comment by Groverkss

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 76

**Comment:**

nit: use `workgroup_size = [256]` instead.

---

### Comment by Groverkss

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 92

**Comment:**

Can you add some comments on why some specific things work here? I wrote explanations for the spec in comments here.

---

### Comment by Groverkss

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 72

**Comment:**

This lowering spec only promotes K and V tensors to shared memory. It does not promote Q to shared memory. This works, because we use VMFMA instructions for QK matmul which take vector<8xf16>, which means we don't loose out on good global memory reads despite not using shared memory for Q.

---

### Comment by Groverkss

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 97

**Comment:**

amdgpu-waves-per-eu reasoning:

Our attention implementation on MI300X uses a large number of registers. Setting this flag makes the compiler less conservative when assigning registers. (@kuhar may know a better reasoning actually).

denormal-fp-math-f32:

This asks the llvm backend to disable denorm flushing for exp2/exp, which consumes less instructions on doing exp/exp2.

---

### Comment by Groverkss

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 22

**Comment:**

This file says this file is just an example tuning spec and not for production use. Is this still valid?

---

### Comment by Groverkss

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 29

**Comment:**

Can you add a comment here `//* Tuning Configurations Start *//` or something like that seprating utilities and tuning specs.

---

### Comment by Groverkss

**File:** `compiler/plugins/target/ROCM/builtins/tuning/test/spec_gfx942.mlir`

**Line:** 133

**Comment:**

No need to specify "ReadOnly|Indirect" on pipeline.binding for tests.

---

### Comment by bangtianliu

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 22

**Comment:**

My understanding is that it is not ready for production yet, but needs input from @kuhar about it.

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_gfx942.mlir`

**Line:** 22

**Comment:**

We have to start somewhere. I'd remove the warning or reword it to something along the lines that this is work in progress.

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/builtins/tuning/test/spec_gfx942.mlir`

**Line:** 97

**Comment:**

```suggestion
        // Apply the attention operation directly to function inputs.
```
https://llvm.org/docs/CodingStandards.html#commenting

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/builtins/tuning/test/spec_gfx942.mlir`

**Line:** 152

**Comment:**

```suggestion
        // Apply the attention operation directly to function inputs.
```

---

## PR #19762: [Codegen][Tuner] Add support for per-sku tuning spec

**URL:** https://github.com/iree-org/iree/pull/19762
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 130

**Comment:**

This can now be moved into the code block where it's used.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.cpp`

**Line:** 1097

**Comment:**

This is not the best location for this code -- generic gpu attributes shouldn't know about the exact hardware details.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.cpp`

**Line:** 1097

**Comment:**

Why not return `optional<StringRef>`?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.cpp`

**Line:** 1111

**Comment:**

Please update this code to follow the llvm coding standards: https://llvm.org/docs/CodingStandards.html

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp`

**Line:** 958

**Comment:**

I don't think we need to change this -- we can use the existing way of specifying targets and backends separately.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.td`

**Line:** 504

**Comment:**

I think we should move this code somewhere close to or in `KnownTargets.cpp`; The generic GPU attributes shouldn't know anything about the rocm backend details and hip targets.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 146

**Comment:**

This function is getting a bit long, I think it'd be better to outline this code to a helper function.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.td`

**Line:** 506

**Comment:**

Invert this condition and use an early return: https://llvm.org/docs/CodingStandards.html#use-early-exits-and-continue-to-simplify-code

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp`

**Line:** 708

**Comment:**

Thanks for all the fixes around this code. I spend some time thinking about this approach of raising the target attribute back to the sku and wasn't sure if it's a good idea or not. It seems simple and thought that it may be good enough for amdgpu for now, but I found a counterexample: mi325. It has the same number of CUs as mi300, but the performance characteristics are different.

I think that to make this robust, we have to go back to [what I suggested previously](https://github.com/iree-org/iree/pull/19748#discussion_r1924336701) and record the sku in the target attribute itself, similar to how we keep the target arch around today.


---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp`

**Line:** 57

**Comment:**

```suggestion
  std::optional<StringRef> sku;
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp`

**Line:** 123

**Comment:**

optional has a `.value_or` function, we should use it here. https://en.cppreference.com/w/cpp/utility/optional/value_or

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 140

**Comment:**

Since chipSKU is already an optional attribute, I'd not expect to also find empty strings here. We can add an assertion for this.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 142

**Comment:**

```suggestion
    if (std::optional<StringAttr> chipSku = chip.getSku()) {
      sku = chipSku->getValue();
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 149

**Comment:**

This comment doesn't clarify much beyond what the code does. Focus on **why** when writing comments, not **what**.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 155

**Comment:**

Here, the comment is very useful because it explains why we are attempting to fetch this spec.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp`

**Line:** 125

**Comment:**

This ternary is very complex. Imo this should be an if statement.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.td`

**Line:** 420

**Comment:**

I don't think this should be double-optional

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp`

**Line:** 122

**Comment:**

When can the sku be present but empty? I don't think this happens with the current code.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp`

**Line:** 122

**Comment:**

just to deal with the corner case "", use assert instead?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp`

**Line:** 122

**Comment:**

I don't think this is something we have to check at all

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp`

**Line:** 122

**Comment:**

It will not happen in current code, maye I just add assert.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp`

**Line:** 122

**Comment:**

> I don't think this is something we have to check at all

Ok, got it.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp`

**Line:** 123

**Comment:**

If sku is optional, why are we setting it with an empty string? Is empty string considered the same as `nullptr` in `StringAttr`?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp`

**Line:** 122

**Comment:**

```suggestion
    auto skuAttr =
        StringAttr::get(context, details.chip->sku.value_or(""));
```
The type is obvious based on the RHS: https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable

---

### Comment by qedawkins

**File:** `compiler/plugins/target/ROCM/builtins/tuning/test/spec_gfx942.mlir`

**Line:** 10

**Comment:**

I missed when this flag was added, but it should be named `--iree-codegen-test` and/or be hidden or something. Not for this PR though.

---

## PR #19756: [Codegen] add mi308x target

**URL:** https://github.com/iree-org/iree/pull/19756
**State:** MERGED

### Comment by ScottTodd

**File:** `docs/website/docs/guides/deployment-configurations/gpu-rocm.md`

**Line:** 133

**Comment:**

Keep sorted (at least don't put this between the two `MI300A` rows)

```suggestion
| AMD MI300A (early units) | `gfx941`    | `cdna3`
| AMD MI300A               | `gfx942`    | `cdna3`
| AMD MI300X               | `gfx942`    | `cdna3`
| AMD MI308X               | `gfx942`    | `cdna3`
```

---

### Comment by ScottTodd

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp`

**Line:** 343

**Comment:**

keep sorted?
```suggestion
      .Cases("mi308x", "mi300x", "mi300a", "gfx942")
```

@kuhar do you want new SKUs to go at the start of the list or the end of the list?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp`

**Line:** 343

**Comment:**

I'd think we should append them to keep consistent with `.Cases("cdna3", "gfx940", "gfx941", "gfx942",` above where newer targets are placed towards the end.

Ultimately I don't care as long as we maintain some consistent ordering.

---

## PR #19748: [Codegen][Tuner] default tuning spec available per-SKU

**URL:** https://github.com/iree-org/iree/pull/19748
**State:** CLOSED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 37

**Comment:**

In general, we should no reference global variables like this. This makes the code hard to maintain and LLVM converged on external storage for flags meant to be external, and having these flag storage variables declared in headers.

Here specifically, we should not rely on test flags in this code. All the information we use must come from the gpu target attr. If the information we need there is not available, we should work on adding it.


---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/builtins/tuning/iree_default_tuning_spec_mi308x.mlir`

**Line:** 243

**Comment:**

We can stage this PR such that we first commit a simple tuning spec to make sure everything works, and then separately work on adding configs we care about.

These configs should be tested to make sure they apply on the intended code, and we should quantify the improvements making sure we don't populate these specs with configs that give us only marginal improvements. This needs to be backed by data.

---

## PR #19603: [Codegen][Tuner] skip linking based on the default entry point attribute

**URL:** https://github.com/iree-org/iree/pull/19603
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 201

**Comment:**

The code below is self-explanatory. I think the comment a few lines below should be enough.
```suggestion
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 202

**Comment:**

```suggestion
    bool isUserTuningSpecWithDefaultAttr =
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 212

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

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/materialize_tuning_specs.mlir`

**Line:** 33

**Comment:**

Can we also check that there are no nested modules?

---

## PR #19525: [Codegen][Tuner] verifier for the default tuning spec

**URL:** https://github.com/iree-org/iree/pull/19525
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h`

**Line:** 49

**Comment:**

I'm not sure this is the best name to use -- we should also allow user specs to specify that they have a single entry point. Maybe `iree_codegen.tuning_spec_with_default_entrypoint`?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 77

**Comment:**

Don't hardcode the name here in case we want to rename it in the future.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 73

**Comment:**

Can you enumerate the named sequence ops and check only these instead? The check here doesn't guarantee that the symbol is a named sequence, it could be something else.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 58

**Comment:**

```suggestion
  // - If the attribute's name matches `kTuningDefaultSpecAttrName`, make
  //   sure it contains a single named sequence op with name `__kernel_config`.
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 68

**Comment:**

We should also have a test for when there's some other op with named `__kernel_config`, e.g., `func.func`.

---

### Comment by kuhar

**File:** `compiler/plugins/target/ROCM/test/default_tuning_specs_amdgpu.mlir`

**Line:** 36

**Comment:**

Can you use `BOTH-SAME` to make this match work for any ordering of these module-level attributes?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 86

**Comment:**

It's not clear to me what `this change` refers to. Instead, I'd add a comment higher up that this will create a named sequence op that conforms to the requirements of tuning specs with default entrypoint (not just the name).

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 72

**Comment:**

Can we check the name directly (`.getName` or similar) instead of using the symbol table?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 88

**Comment:**

I don't think we need this comment anymore
```suggestion
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 69

**Comment:**

We don't need this to be generic. Also, let's not shadow the `op` variable from the parent scope.
```suggestion
      if (!llvm::any_of(moduleOp.getOps(), [](Operation *nestedOp) {
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 69

**Comment:**

Isn't there a helper that gets the ops of the specified type? Something like `getOps<SomeOp>()`?


---

### Comment by kuhar

**File:** `docs/website/docs/reference/tuning.md`

**Line:** 127

**Comment:**

```suggestion
  the tuning spec includes a named sequence op with name `__kernel_config`.
```

---

## PR #19486: [Codegen][Tuner] attr verifier for tuning specs

**URL:** https://github.com/iree-org/iree/pull/19486
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 118

**Comment:**

We should call verify on the whole module. You can see this used here: https://github.com/llvm/llvm-project/blob/57c161a6479fb70a31553e2f9bc1efa46262aa92/mlir/lib/Dialect/Transform/Transforms/TransformInterpreterUtils.cpp#L118

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 110

**Comment:**

Use proper punctuation.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 118

**Comment:**

Actually, I don't think this is needed along this code path because the user spec verification happens in `parseTransformModuleFromFile`. We should verify linked specs after linking though.l

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 167

**Comment:**

I don't think we should verify default specs -- these are already verified by our tests when building the compiler. We could do that but under debug builds only.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 1

**Comment:**

```suggestion
// RUN: iree-opt  --verify-diagnostics  %s
```

This test does not rely on the other flags AFAICT

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 16

**Comment:**

We should also check what happens there's some other attribute attached to `named_sequence` and documents whether that's allowed or not (by the virtue of having a testcase).

We should also add tests that check that the attribute is present but the `named_sequence` signature is wrong.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 76

**Comment:**

We should remove the validation from the other code where this was copied from -- no need to validate twice.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 55

**Comment:**

I don't think this comment adds clarity, I'd drop it. Instead, you can summarize the validity criteria in one longer comment.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 59

**Comment:**

Same here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 60

**Comment:**

```suggestion
  if (!isa<UnitAttr>(attr)) {
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 150

**Comment:**

We should verify the result of linking, not the input. It is assumed that the input would have been verifier by the parser or something else that created it.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 147

**Comment:**

This doesn't handle the case then the loaded module is a failure or nullptr

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 257

**Comment:**

This should be checked in the code that does linking

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 1

**Comment:**

```suggestion
// RUN: iree-opt --verify-diagnostics --split-input-file %s
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 39

**Comment:**

We should have a testcase for when there are no return values

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 6

**Comment:**

We don't need the nested module in this test -- a single level of nesting is sufficient

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 23

**Comment:**

Also here, we don't need to nest

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 31

**Comment:**

Other tests already check that function ops are allowed, I don't think we need this here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 31

**Comment:**

Same in the other test cases below

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 53

**Comment:**

I'd move this up just after the first testcase that checked for the wrong number of arguments

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 88

**Comment:**

```suggestion
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 130

**Comment:**

We should return failure so that nothing uses this module.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 130

**Comment:**

```suggestion
    module.emitError("Linked tuning spec failed to verify");
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 154

**Comment:**

```suggestion
           << "Default tuning spec " << defaultTuningSpecName << " failed to verify";
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 148

**Comment:**

We shouldn't emit this error here. The reason is that `getOrParseTransformLibraryModule` already does the reporting and it knows do it only when the parsing fails for the first time.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 152

**Comment:**

Instead of the check above, we can do this:
```suggestion
  if (succeded(defaultTransformLibrary) && failed(mlir::verify(*defaultTransformLibrary)))
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 35

**Comment:**

```suggestion
    attributes { iree_codegen.tuning_spec_entrypoint } {}
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 56

**Comment:**

```suggestion
    attributes { iree_codegen.tuning_spec_entrypoint } {}
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 10

**Comment:**

This won't be verified anyway because of the previous error. We should move it before the erroneous spec.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/verify_tuning_specs.mlir`

**Line:** 15

**Comment:**

Also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/LinkTuningSpecsPass.cpp`

**Line:** 130

**Comment:**

I think this should work
```suggestion
    return module.emitError("Linked tuning spec failed to verify");
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 151

**Comment:**

Have you confirmed that this error gets emitted when the default spec is invalid?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 154

**Comment:**

We shouldn't dereference here -- this assumes no failures

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 151

**Comment:**

@bangtianliu can you please reply instead of marking this as resolved? I don't know what the outcome of this is.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/MaterializeTuningSpecsPass.cpp`

**Line:** 151

**Comment:**

Yes, I did check it and fixed the crashes.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.cpp`

**Line:** 69

**Comment:**

The check below is self-explanatory, I don't think this comment adds clarity

---

## PR #19376: [tuner]: add property functions to lowering config python binding

**URL:** https://github.com/iree-org/iree/pull/19376
**State:** MERGED

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 102

**Comment:**

I'd put these in a single function that returns all tile sizes in a struct. You can see an example above (`ireeGPUMMAAttrGetInfo`). The reason is that this is expands the API surface area by quite a lot, and adding more levels of tiling would exacerbate this further. With a struct, we can keep extending it with more fields.

---

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 108

**Comment:**

This can be a single function that returns two integers.

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 63

**Comment:**

We don't need the buffer to be caller-allocated -- we can return a pointer to the attribute storage.

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 77

**Comment:**

Also here

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 231

**Comment:**

You can add a helper to `tuner_ctx` to help with this, e.g.: `tuner_ctx.type.getIndexArray([1])`

---

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 109

**Comment:**

Some leftover comment

---

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_gpu.h`

**Line:** 106

**Comment:**

I'd return these as integers int64_t

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 224

**Comment:**

Don't use `auto` when the type is not obvious based on the RHS only

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 228

**Comment:**

Also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 252

**Comment:**

Don't we have helpers for this in the dialect headers or the attr interface?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 252

**Comment:**

Here: https://github.com/iree-org/iree/blob/a1664e30a6a53850f689f32086a8b0c45bed327b/compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h#L22-L23

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 269

**Comment:**

We also have a helper for this

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 373

**Comment:**

To me it's not clear how these map to mnk dimensions. Maybe call it `subgroup_count_mn`?

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 13

**Comment:**

This function name doesn't specify the element type

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 19

**Comment:**

This returns the index type

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 15

**Comment:**

You can move the `get_index_attr` helper above and use here

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 388

**Comment:**

These ternaries get pretty lengthy. Instead, I'd put the assignment in an if condition.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 233

**Comment:**

use `llvm::to_underlying` to make sure we get the type right

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 236

**Comment:**

This returns dangling pointers. We should use the data from the attr storage -- these two should be array attributes.

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 367

**Comment:**

```suggestion
            for (size_t i = 0; i < len; ++i) {
```

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 384

**Comment:**

```suggestion
            for (size_t i = 0; i < len; ++i) {
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 230

**Comment:**

We should expose these string literals in the dialect headers.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 231

**Comment:**

Also this one.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h`

**Line:** 89

**Comment:**

I'd say something more concise like:
```c++
/// Returns the name of the tilling `level`, as used in the `lowering_config` attribute.
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREEGPUDialectCAPI.cpp`

**Line:** 228

**Comment:**

looks much cleaner now!

---

## PR #19218: [tuner]: add c/python binding for querying mma intrinsic

**URL:** https://github.com/iree-org/iree/pull/19218
**State:** MERGED

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_codegen.h`

**Line:** 71

**Comment:**

Let's keep the types consistent with how the other functions handle containers in this file.
```suggestion
ireeCodegenGetExecutableVariantOps(MlirModule module,  size_t *numOps,
                                   MlirOperation *executableOps);
```
also use `thisCase` for function arguments instead of the `snake_case`

---

### Comment by kuhar

**File:** `compiler/bindings/c/iree/compiler/dialects/iree_codegen.h`

**Line:** 75

**Comment:**

same here

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 24

**Comment:**

You can return a vector: `std::vector<MlirOperation>`

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 53

**Comment:**

Here, we should return a list of enums, not integers. You can see how to construct an enum in the code that handles enum attributes below.

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 377

**Comment:**

Please run your PR through a spell checker. I have a vscode extension for that.

---

### Comment by kuhar

**File:** `compiler/bindings/python/test/ir/dialects_test.py`

**Line:** 240

**Comment:**

I'm concerned this is a 'change detector' test. Every time we update any of the related dialects, we will have to come back to this test and also update it. It's worse than lit tests were at least you have the lsp helping you with syntax highlighting etc. IMO we can do without a test here -- the logic is tested on the C++ side with your test pass.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp`

**Line:** 157

**Comment:**

You can make a typedef for this op type to define these longs namespaces

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp`

**Line:** 161

**Comment:**

We should check that `num_ops` is not `nullptr`

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp`

**Line:** 166

**Comment:**

We should check that `num_ops` matches the number of variant ops.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp`

**Line:** 172

**Comment:**

Similar in this function

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp`

**Line:** 175

**Comment:**

```suggestion
      llvm::dyn_cast_if_present<mlir::iree_compiler::IREE::HAL::ExecutableVariantOp>(
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp`

**Line:** 189

**Comment:**

Follow the llvm coding style for loops: https://llvm.org/docs/CodingStandards.html#don-t-evaluate-end-every-time-through-a-loop

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp`

**Line:** 1033

**Comment:**

This looks like an old change?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp`

**Line:** 1042

**Comment:**

also here

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp`

**Line:** 1033

**Comment:**

yes

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 41

**Comment:**

```suggestion
  std::vector<uint32_t> mmaIntrinsics(numMMAs);
```

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 48

**Comment:**

We should get the enum att once and reuse it instead of importing the module N times.

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 30

**Comment:**

```suggestion
  std::vector<MlirOperation> ops(numOps);
```

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 42

**Comment:**

```suggestion
  ireeCodegenQueryMMAIntrinsics(op, &numMMAs, nullptr);
  std::vector<uint32_t> mmaIntrinsics(numMMAs);
  ireeCodegenQueryMMAIntrinsics(op, &numMMAs, mmaIntrinsics.data());
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp`

**Line:** 159

**Comment:**

We should also assert that `module` is not null

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp`

**Line:** 184

**Comment:**

```suggestion
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp`

**Line:** 1033

**Comment:**

This is not resolved. If these are old changes, we should rebase this PR to be reasonably up-to-date with main.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp`

**Line:** 1042

**Comment:**

same here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/API/Internal/IREECodegenDialectCAPI.cpp`

**Line:** 158

**Comment:**

use the c function to check this instead of accessing `.ptr` directly

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 46

**Comment:**

nit: I'd make this a for loop, I don't think the transform helps the readability much...

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 24

**Comment:**

```suggestion
static std::vector<MlirOperation>
```

---

### Comment by kuhar

**File:** `compiler/bindings/python/IREECompilerDialectsModule.cpp`

**Line:** 36

**Comment:**

```suggestion
static std::vector<py::object> ireeCodegenQueryMMAIntrinsicsBinding(MlirOperation op) {
```

---

## PR #19199: [tuner]: two new utility functions which are more friendly for c binding

**URL:** https://github.com/iree-org/iree/pull/19199
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp`

**Line:** 1034

**Comment:**

Why did you change it to pre-order?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp`

**Line:** 1033

**Comment:**

```suggestion
SmallVector<IREE::HAL::ExecutableVariantOp>
getExecutableVariantOps(mlir::ModuleOp moduleOp) {
  SmallVector<IREE::HAL::ExecutableVariantOp> executableVariantOps;
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h`

**Line:** 211

**Comment:**

```suggestion
/// given `mlir::ModuleOp`, ensuring they are returned in their original IR
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h`

**Line:** 213

**Comment:**

```suggestion
SmallVector<IREE::HAL::ExecutableVariantOp>
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h`

**Line:** 218

**Comment:**

```suggestion
SmallVector<IREE::GPU::MMAIntrinsic>
```

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp`

**Line:** 1034

**Comment:**

Just to ensure it is in correct order.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/TestLLVMGPUQueryMMAPass.cpp`

**Line:** 28

**Comment:**

You can use values for IR constructs -- they are designed to be cheap to pass by value. Also use the actual type, since the type is not obvious based on the RHS only.

---

## PR #19124: [tuner]: Add a utility function to query supported MMA intrinsics

**URL:** https://github.com/iree-org/iree/pull/19124
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/TestLLVMGPUQueryMMAPass.cpp`

**Line:** 27

**Comment:**

This helper doesn't really do anything -- we can inline it into the pass use `llvm::interleaveComma` instead of the for loop.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/TestLLVMGPUQueryMMAPass.cpp`

**Line:** 39

**Comment:**

There's no point in printing an empty vector

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/test_query_mma.mlir`

**Line:** 5

**Comment:**

We don't need an exhaustive list all the other wgp properties -- we can trim it down to something minimal like `compute = int32, storage = b32, ...`

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp`

**Line:** 1037

**Comment:**

Wouldn't it be enough to look up `hal.executable.variant` only? This is where the attribute is attached to.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp`

**Line:** 1036

**Comment:**

This comment doesn't clarify anything -- `moduleOp.walk` is a very basic function and the intention is obvious here.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp`

**Line:** 1033

**Comment:**

I think this should probably return a mapping of executable variants to their mma attrs. Should should also have a test of a module with two variants.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp`

**Line:** 1050

**Comment:**

we should be able to append all of them at once with `llvm::append_range`

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h`

**Line:** 211

**Comment:**

```suggestion
void queryMMAIntrinsics(mlir::ModuleOp moduleOp,
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h`

**Line:** 212

**Comment:**

Can we return mma intrinsics attrs instead of mma attr? I think we can always go from an intrsic to an mma but not the other way round?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/TestLLVMGPUQueryMMAPass.cpp`

**Line:** 31

**Comment:**

You can use structured bindings to unpack these two members to variables

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/TestLLVMGPUQueryMMAPass.cpp`

**Line:** 42

**Comment:**

use  `llvm::interleave` -- you can give it the desired separator

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/test_query_mma.mlir`

**Line:** 40

**Comment:**

could we further trim this down but skipping some of these `wgp` properties out entirely?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h`

**Line:** 214

**Comment:**

Why not return this map?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp`

**Line:** 1041

**Comment:**

Instead of appending to an empty range, use `llvm::map_to_vector`.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h`

**Line:** 213

**Comment:**

Why do we want to return `Operation *` instead of `ExecutableVariantOp`? I think this would simplify both code and the documentation.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/TestLLVMGPUQueryMMAPass.cpp`

**Line:** 35

**Comment:**

Can this actually happen? I don't see it in the LIT test.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp`

**Line:** 1042

**Comment:**

```suggestion
      mmaAttributesMap[executableOp] = std::move(mmaIntrinsics);
```
to avoid needless copying

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp`

**Line:** 21

**Comment:**

Is this include still necessary?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/TestLLVMGPUQueryMMAPass.cpp`

**Line:** 31

**Comment:**

Why do we need the cast? I'd think the type of op is already executable variant op, no?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h`

**Line:** 213

**Comment:**

This is out of date

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/TestLLVMGPUQueryMMAPass.cpp`

**Line:** 31

**Comment:**

No, compiler assumes that it is operation* type. 

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h`

**Line:** 213

**Comment:**

yeah, forgot to update.

---

## PR #19069: [tuner] add an iree-opt pass to strip configuration from executable sources

**URL:** https://github.com/iree-org/iree/pull/19069
**State:** MERGED

### Comment by benvanik

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_config_info.mlir`

**Line:** 3

**Comment:**

You can trim this down significantly - as it is this is a "change detector" test in that any change to any attribute or op used in here will require someone to come fix the test. You should have an op count of 3 and the attributes should have the minimal stable configuration to pass verification.

The pass is 3 lines - the test should be correspondingly small.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/Passes.td`

**Line:** 495

**Comment:**

I'd call it `StripCompilationInfo`. The reason is that the `#iree_codegen.compilation_info` attribute contains both lowering config and translation info, so that'd encompass all 3 attributes

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripConfigInfoPass.cpp`

**Line:** 20

**Comment:**

```suggestion
  using impl::StripConfigInfoPassBass::StripConfigInfoPassBase;
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripConfigInfoPass.cpp`

**Line:** 18

**Comment:**

nit: `struct` will save us some typing, since we don't need to hide data members from anybody anyway -- this pass is defined in an anonymous namespace, so only this file knows its full type anyway
```suggestion
struct StripConfigInfoPass final
    : impl::StripConfigInfoPassBase<StripConfigInfoPass> {
```


---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripConfigInfoPass.cpp`

**Line:** 22

**Comment:**

```suggestion
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripConfigInfoPass.cpp`

**Line:** 27

**Comment:**

I'd define this inline -- I don't think we gain much by outlining this function

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripConfigInfoPass.cpp`

**Line:** 31

**Comment:**

We don't have to keep `translationInfo` as a local variable -- we never use it beyond checking that it's there. I'd do the same thing as you do with lowering config below.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripConfigInfoPass.cpp`

**Line:** 37

**Comment:**

We should be also stripping compilation info attributes for completeness sake IMO. These won't be produced by the compiler in the default flow, but would appear as the result of applying the tuning specs (transform dialect library).

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_config_info.mlir`

**Line:** 3

**Comment:**

+1, we can write a small test case by hand that doesn't use the actual dumps from iree-opt. For example, setting `lowering_config` to a `StringAttr` should exercise the code just as well.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_config_info.mlir`

**Line:** 1

**Comment:**

Ah, now I think we should make this pass over `Operation` so that we don't have to manually specify the nesting. We can walk whatever the input op is and find functions ops first.

---

### Comment by benvanik

**File:** `compiler/src/iree/compiler/Codegen/Common/Passes.td`

**Line:** 495

**Comment:**

good call

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 22

**Comment:**

This assumption is not correct. We may have lowering config / compilation info but not translation info. This would be true for modules configured by TD scripts.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 20

**Comment:**

IMO this can return `bool` -- it's very clear what the meaning of `true`/`false` is.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 37

**Comment:**

Previous comments not addressed

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 45

**Comment:**

Do we really need to clone the function to make this work? Can't we modify it in place?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 75

**Comment:**

You don't need the greedy rewriter here, you can use `walkAndApplyPatterns` which should be much cheaper.

But I'm not sure we need it any case -- the previous approach with a manual `walk` seemed fine to me 

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir`

**Line:** 57

**Comment:**

We can further simplify this. We can have a function with just a couple of ops that accepts tensors and does matmul on the operands.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 22

**Comment:**

For the case you proposed,  this condition cannot pass and then the function will continue to check whether it has lowering config/translation info. 

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 36

**Comment:**

```suggestion
struct StripCompilationInfo final : OpRewritePattern<func::FuncOp> {
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 48

**Comment:**

The `hasCompilationInfo` check seems redundant -- we may just as well remember if any modification were performed or not.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 43

**Comment:**

Use proper casing and punctuation

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir`

**Line:** 45

**Comment:**

We should also add one test with just lowering config attached to an op

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir`

**Line:** 9

**Comment:**

Most of the checks in this file seem off: there's no `#` symbol on the attribute name

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/Passes.td`

**Line:** 495

**Comment:**

```suggestion
   let summary = "Remove all the the lowering configuration and translation info attributes.";
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 36

**Comment:**

Can we make this pattern be over the func op interface?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 53

**Comment:**

Also, in rewrite patterns, it's invalid to mutate the IR without going through the rewriter: https://mlir.llvm.org/docs/PatternRewriter/#common-pattern-drivers. You'd need to use `modifyOpInPlace`.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 40

**Comment:**

I think the nested walk makes this pass quadratic. Maybe we should have a separate pattern for these?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 22

**Comment:**

```suggestion
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
```

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 40

**Comment:**

Good point!  I will separate the patterns. 

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 33

**Comment:**

```suggestion
          // Erase the compilation info configuration if it exists.
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 19

**Comment:**

```suggestion
struct StripFuncOpTranslationInfo final
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 32

**Comment:**

you should only return `success()` if the IR was actually modified. If not, return `failure()`. The way you can do it is to pull the `if (getTranslationInfo())` check out of the `modifyInPlace` and fail early if it's not present.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 56

**Comment:**

Similar here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/StripCompilationInfoPass.cpp`

**Line:** 59

**Comment:**

Put context in a local variable to CSE this a bit.

---

### Comment by benvanik

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir`

**Line:** 5

**Comment:**

```suggestion
  return
```

---

### Comment by benvanik

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir`

**Line:** 45

**Comment:**

```suggestion
  return %result : tensor<128x1024xf32>
```

---

### Comment by benvanik

**File:** `compiler/src/iree/compiler/Codegen/Common/test/strip_compilation_info.mlir`

**Line:** 58

**Comment:**

```suggestion
  return %result : tensor<128x1024xf32>
```

---

## PR #18952: [VectorDistribution] Add distribution pattern for vector::ContractionOp

**URL:** https://github.com/iree-org/iree/pull/18952
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 573

**Comment:**

Butterfly shuffle is an implementation detail that's not observable from this pattern. I'd just say we perform subgroup reduction.

Also nit: the formatting is weird.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 571

**Comment:**

I'd call it something like: subgroup reduction or inter-thread reduction.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 576

**Comment:**

let's try to stick to the portable naming like in the gpu dialect

```suggestion
/// Currently, reduction across multiple subgroups is not supported.
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 578

**Comment:**

Can you describe why we'd need it?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 598

**Comment:**

```suggestion
    auto mmaAttr =
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 605

**Comment:**

```suggestion
    auto lhsLayout =
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 611

**Comment:**

```suggestion
    auto rhsLayout =
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 624

**Comment:**

```suggestion
    auto accVector = dyn_cast<VectorValue>(acc);
    auto resVector = dyn_cast<VectorValue>(res);
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 636

**Comment:**

Use the actual type since it's not obvious based on the RHS only

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 639

**Comment:**

I'd put it just before loc and extract from `contractOp` instead.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 657

**Comment:**

https://llvm.org/docs/CodingStandards.html#prefer-preincrement

```suggestion
    for (int i = 0; i < 3; ++i) {
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 663

**Comment:**

Use the actual type since it's not obvious based on the RHS only

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 667

**Comment:**

also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 669

**Comment:**

also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 701

**Comment:**

Would it be possible to split this function into a couple smaller helpers matching the 3-stage lowering described in the comment? This implementation is a bit long and hard to follow.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPUConfigureTensorLayouts.cpp`

**Line:** 344

**Comment:**

```suggestion
    IREE::GPU::MMAScheduleAttr scheduleAttr;
```

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 578

**Comment:**

Now the implementation is limited to support only 1 subgroup, I don't know whether 1 subgroup for sure is always the optimal config for performance. if multiple subgroups can achieve more performance benefits, then adding support for reductions across multiple subgroups will be a TODO.  

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 578

**Comment:**

I think I added this comment for multi_reduction and it was copied over. Since we can describe reduction on subgroups on layouts, if we ever need workgroup reduce with atomic read/write we can do it. Can also just remove the comment if we dont plan on ever implementing that.

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 571

**Comment:**

```
Subgroup Reduction: Threads in each subgroup reduce the results from step 1 across threads
```

could work fine

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPUConfigureTensorLayouts.cpp`

**Line:** 497

**Comment:**

I think you can always assume there will be a configDict. What you cannot assume is there always being a scheduleAttr. The pass should return error if there is no translation_info / configuration.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 578

**Comment:**

Yes, I can remove it.

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPUConfigureVectorLayouts.cpp`

**Line:** 275

**Comment:**

I think this file got deleted actually. You might want to rebase.

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/test/gpu_nested_layout_vector_distribution.mlir`

**Line:** 1228

**Comment:**

Can we have a test where vector.contract has a parallel dimension as well?

---

## PR #18825: [VectorDistribution] Add kernel configs for root reduction operations (4/4)

**URL:** https://github.com/iree-org/iree/pull/18825
**State:** CLOSED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp`

**Line:** 932

**Comment:**

nit: you can use `llvm::is_contained`

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp`

**Line:** 1103

**Comment:**

We should probably make this a target property like discussed here: https://discord.com/channels/689900678990135345/1254843174111678555/1296841473249251439.
This is not a blocker IMO.

---

### Comment by IanWood1

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp`

**Line:** 1105

**Comment:**

@bangtianliu the SIGFPE is from here since `getMaxLoadInstructionBits()` was returning 0 when targeting `gfx1100`. See: https://github.com/iree-org/iree/issues/18849

---

## PR #18822: [VectorDistribution] Plumb the VectorDistribute pipeline to support reduction operations (3/4)

**URL:** https://github.com/iree-org/iree/pull/18822
**State:** CLOSED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h`

**Line:** 47

**Comment:**

I think changes to these options need to be reflected on the tablegen side (in the matching attribute)? @Max191 

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 232

**Comment:**

Use proper capitalization

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 867

**Comment:**

Also here

---

## PR #18800: [VectorDistribution] Add vector distribution support multi-dim reduction with scalars

**URL:** https://github.com/iree-org/iree/pull/18800
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h`

**Line:** 112

**Comment:**

This strikes me as an odd helper: you give 'vector' a new meaning without introducing a name. Instead, I'd either flip it and add a helper like `isRank0(VectorValue val)`, or just expand the check where you need it. 

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.cpp`

**Line:** 219

**Comment:**

```suggestion
bool isVector(VectorValue val) {
  return val.getType().getRank() != 0;
}
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp`

**Line:** 1063

**Comment:**

The first check is redundant

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp`

**Line:** 1066

**Comment:**

Can you use the actual type here instead of `auto`? It's not clear based on the RHS

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp`

**Line:** 713

**Comment:**

This assertion is obsolete now.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.cpp`

**Line:** 136

**Comment:**

use dyn_cast instead:
```c++
if (auto x = dyn_cast<Y>(y)) {
  if (x.something() == Z) {
 ```
 
 See https://llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates:~:text=Note%20that%20you%20should%20not%20use%20an%20isa%3C%3E%20test%20followed%20by%20a%20cast%3C%3E%2C%20for%20that%20use%20the%20dyn_cast%3C%3E%20operator.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 535

**Comment:**

You don't need a vector here, you can do: `ArrayRef{int64_t(0)}`

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 493

**Comment:**

Also here, no need to use a vector

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 428

**Comment:**

You can use `getElementTypeOrSelf`. Also below.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 312

**Comment:**

This is the only use of `isSrcVector`, so it makes sense to inline it
```suggestion
    if (!srcVector || !isVector(srcVector)) {
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUDistributionPatterns.cpp`

**Line:** 588

**Comment:**

`vector<f32>` is a vector value so the comment doesn't make sense to me

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 310

**Comment:**

Similar here -- the comment doesn't make sense to me. What does it mean for `vector<f32>` to return a pointer?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 312

**Comment:**

We should inline this condition into the `if`

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 426

**Comment:**

Either check that the `dyn_cast`s succeeded (and return an error if not) or use `cast` to assert on cast failures

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.cpp`

**Line:** 135

**Comment:**

```suggestion
    if (auto replacementType =
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp`

**Line:** 712

**Comment:**

Follow the llvm coding style and use `dyn_cast` here to avoid repeated type checking below

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp`

**Line:** 948

**Comment:**

typo and punctuation
```suggestion
    // Handle the propagation from scf.for to yield op.
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h`

**Line:** 111

**Comment:**

```suggestion
// Returns true iff the rank of the input value 'val' is non-zero.
```

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 426

**Comment:**

> Either check that the `dyn_cast`s succeeded (and return an error if not) or use `cast` to assert on cast failures

In the following code, we handle both the valid pointer (vector case) and null pointer cases (scalar case). So, there's no need to add an assertion or return an error.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUDistributionPatterns.cpp`

**Line:** 592

**Comment:**

Thanks for revising this. Now the comment is very clear but also very verbose. I think we can simplify this.

```suggestion
        // Distributing the operand requires it to have a non-zero rank, meaning it must have
        // at least one dimension. If the vector has a non-zero rank, the operand is distributed
        // according to the provided layout signature.
```

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h`

**Line:** 22

**Comment:**

We shouldn't have this in the header file.

---

## PR #18784: [VectorDistribution] Add scalar support for distributing multi-dim reduction (1/4)

**URL:** https://github.com/iree-org/iree/pull/18784
**State:** CLOSED

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 439

**Comment:**

You probably dont need to check this.

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 427

**Comment:**

Use getElementTypeOrSelf

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 434

**Comment:**

use getElementTypeOrSelf

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 457

**Comment:**

Add a comment here that scalars are always distributed to all threads already.

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 487

**Comment:**

Add a comment that we broadcast scalar accumulator to vector.

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 513

**Comment:**

Add a comment that we broadcast scalar to vector and why.

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp`

**Line:** 639

**Comment:**

This shoudnt be part of this patch?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp`

**Line:** 639

**Comment:**

It will crash for scalar case if this statement is not included, patch 2 mainly include the propagation from scf.for to scf.yield. 

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp`

**Line:** 639

**Comment:**

ah ok, i see.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 484

**Comment:**

You should be able to use ArrayRef here instead of constructing a vector

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 526

**Comment:**

also here

---

## PR #18660:  [VectorDistribution]Add distribution pattern and test mlir file for vector.gather

**URL:** https://github.com/iree-org/iree/pull/18660
**State:** MERGED

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp`

**Line:** 824

**Comment:**

We also need a propagation transfer function for vector.gather i think. You can look at other examples.

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp`

**Line:** 822

**Comment:**

Can you add tests for the layout analysis?

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp`

**Line:** 553

**Comment:**

vector.gather does not have multiple results. You can just assert this or remove this condition.

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp`

**Line:** 568

**Comment:**

You can remove this condition. resolve already takes care of it.

---

## PR #18519: [VectorDistribution]reduction support along LLVMGPUVectorDistribute pipeline

**URL:** https://github.com/iree-org/iree/pull/18519
**State:** CLOSED

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUDistributionPatterns.cpp`

**Line:** 110

**Comment:**

remove this comment

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUDistributionPatterns.cpp`

**Line:** 112

**Comment:**

nit: invert condition to be if (isScalar) and move the else branch

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUDistributionPatterns.cpp`

**Line:** 113

**Comment:**

nit: distirbuted -> distributed

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUDistributionPatterns.cpp`

**Line:** 113

**Comment:**

Operation * -> auto

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUDistributionPatterns.cpp`

**Line:** 119

**Comment:**

Operation * -> auto

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUDistributionPatterns.cpp`

**Line:** 118

**Comment:**

Are these branches different? They look the same to me

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 313

**Comment:**

nit: you can just check `srcVector && ...`

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUNestedLayoutDistributionPatterns.cpp`

**Line:** 313

**Comment:**

I would create a helper function "isScalar" and use that instead of checking rank everywhere.

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp`

**Line:** 141

**Comment:**

Can we move this fix to a seperate patch?

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp`

**Line:** 642

**Comment:**

nit: braces

---

### Comment by Groverkss

**File:** `compiler/src/iree/compiler/Codegen/Common/VectorLayoutAnalysis.cpp`

**Line:** 709

**Comment:**

nit: braces

---

## PR #18125: E2e gpu convolution test

**URL:** https://github.com/iree-org/iree/pull/18125
**State:** CLOSED

### Comment by benvanik

**File:** `tools/testing/e2e/iree-e2e-conv2d-test.cc`

**Line:** 12

**Comment:**

we don't use exceptions in runtime code so this should not be required

---

### Comment by bangtianliu

**File:** `tools/testing/e2e/iree-e2e-conv2d-test.cc`

**Line:** 12

**Comment:**

Yes, this is irrelevant. I used it for my local debugging, and am cleaning the code again. 

---

### Comment by ScottTodd

**File:** `tests/e2e/convolution/CMakeLists.txt`

**Line:** 333

**Comment:**

FYI I made it possible for some of this logic to be handled in the BUILD.bazel file in https://github.com/iree-org/iree/pull/17843. I didn't look specifically at folders like tests/e2e/matmul though. Generally speaking, while still using Bazel and CMake for tests, we should avoid using the `BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE` escape hatch if possible.

---

### Comment by ScottTodd

**File:** `tests/e2e/convolution/generate_e2e_conv2d_tests.py`

**Line:** 102

**Comment:**

Are these "compilation infos" part of the public compiler API? How are users expected to make use of these vectorize/vectordistribute MFMA/WMMA/CDNA/RDNA pipelines? I'm worried that this is reaching too deeply into implementation details, when test suites should be modeled more closely after how developers are realistically expected to use the tools.

I would expect:
* program definitions to have computations in them: "multiply these two tensors/matrices with these shapes, then do this other math"
* users/developers to enumerate the explicit devices they want to target, including which features those devices support (e.g. cpu intrinsics, gpu API extensions)
* users/developers to provide tuning configurations per each device that give hints about branches to take during compilation, heuristics to use, etc. (maybe that's this "compilation info"?)

---

### Comment by ScottTodd

**File:** `tools/testing/e2e/iree-e2e-conv2d-test.cc`

**Line:** 121

**Comment:**

All this math in C++ scares me, especially for lower precision types. Would like to at least validate the C++ against reference python framework code _then_ compare our own compiler/runtime/codegen systems against the C++ once we trust it

---

### Comment by ScottTodd

**File:** `tests/e2e/convolution/generate_e2e_conv2d_tests.py`

**Line:** 102

**Comment:**

I'm also sketched out by `get_rocm_test_compilation_infos()` needing to exist at all here - we shouldn't need anything specific to a single target in a test generator like this.

---

### Comment by Max191

**File:** `tests/e2e/convolution/generate_e2e_conv2d_tests.py`

**Line:** 102

**Comment:**

+1 on this. I would expect these e2e tests to look nearly the same as what we have done for winograd convolutions:
https://github.com/iree-org/iree/blob/31bfc932f1428bab08d42d954e6763c0d509376e/tests/e2e/convolution/CMakeLists.txt#L163-L189

You should be able to use the same test generator but choose a different target, and you can add additional compiler flags as needed with `COMPILER_FLAGS`. The test generator should simply create some source IR that performs a convolution, which already exists for winograd. Adding new e2e convolution tests should be able to use that same input IR, but with different compiler specifications. Setting the compilation info is the job of the specific backend codegen, and baking that into the test generator seems like duplicating logic unnecessarily.

---

## PR #17811: Add workgroup chipletgroup strategy to workgroup reordering pass

**URL:** https://github.com/iree-org/iree/pull/17811
**State:** CLOSED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/Passes.td`

**Line:** 205

**Comment:**

The name doesn't match the chiplet-group strategy. We should either rename it or add a second option if that makes more sense.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 71

**Comment:**

```suggestion
// Reordering to make workgroup ids move slowly between chiplet groups.
```

Could you also give an example? IE pick some topology and show how the math works out.

Also say what the return value is.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 76

**Comment:**

Is this the number of chiplets per *work*group or something else?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 90

**Comment:**

```suggestion
  // Handle the remainder part.
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 100

**Comment:**

nit: Do not reassign the function arguments, it makes the logic harder to follow

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 145

**Comment:**

Say what the return value is.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 115

**Comment:**

```suggestion
  // Create one dimension ID for workgroup.
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 120

**Comment:**

Should we plumb this through and add to the target description attribute? Do we have an issue for this?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.td`

**Line:** 280

**Comment:**

```suggestion
    Options for workgroup reordering strategies to improve L2 cache hit rate.
```



---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp`

**Line:** 95

**Comment:**

nit: Do we need the `llvm::`?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp`

**Line:** 97

**Comment:**

also here

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/config_user_vector_distribute.mlir`

**Line:** 90

**Comment:**

Should we add the (optional) parameter to the same attribute?

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/config_user_vector_distribute.mlir`

**Line:** 90

**Comment:**

Yes, should be better and I can make it optional too

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 76

**Comment:**

The number of partitions on GPU, value is typically either 4, with two XCDs per partition, or 8, with one XCD per partition. 

here I use the same naming convention as RocMLIR, which is misleading. I will adopt a more accurate name to better reflect its purpose.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 120

**Comment:**

Address all the other comments except this one. 

It requires the number of XCDs info available in #iree_gpu.target attribute, need to learn how to do it.  

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/Passes.td`

**Line:** 210

**Comment:**

Let's not reuse the flag for swizzle tile size to enable / disable chiplet-based reordering.  Do we need it at all to control chiplet-aware reordering?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 115

**Comment:**

not addressed

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 160

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

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 92

**Comment:**

```suggestion
// Returns permuted workgroup id (X and Y dimensions).
// In the above example:
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 73

**Comment:**

```suggestion
// Example:
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 78

**Comment:**

```suggestion
// the following order:
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 85

**Comment:**

```suggestion
// resulting in the launch order:
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 160

**Comment:**

```suggestion
  // Empirically, found rowGroupSize=16 for MI300X achieves good performance
  // group every 16 workgroups along Y dimension.
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 154

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

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.td`

**Line:** 281

**Comment:**

```suggestion
    Options for workgroup reordering strategies to improve L2 cache hit rate.
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 147

**Comment:**

```suggestion
  // Empirically found that two chiplets as a group has better locality
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 159

**Comment:**

```suggestion
  // Empirically, found rowGroupSize=16 for MI300X achieves good performance
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 145

**Comment:**

Not addressed.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 223

**Comment:**

```suggestion
getReorderWorkgroupsLogTileSize(std::optional<int64_t> option) {
```

This is a very small type, we can pass by value 

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 224

**Comment:**

```suggestion
  int64_t logTile = option.value_or(clReorderWorkgroupsLogTile);
  assert(logTile >= 0);
  return static_cast<unsigned>(logTile);
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 367

**Comment:**

Is there no helper function to look this up? Maybe this https://github.com/iree-org/iree/blob/d174e8bcec9f221082511c67111b3f995bdd54a0/compiler/src/iree/compiler/Codegen/Utils/GPUUtils.h#L153 or something in https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Codegen/Utils/Utils.h

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 367

**Comment:**

sure, I will see whether it is doable.


---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 104

**Comment:**

This won't work, `tilesizeStr` will form a dangling reference because `to_string` returns a temporary

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 101

**Comment:**

```suggestion
  StringRef tileSizeStr = "<not set>";
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 108

**Comment:**

Use the actual field names from `LLVMGPUPipelineOptions` (`reorderWgLogTileSize`)

---

### Comment by qedawkins

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 72

**Comment:**

```suggestion
// Reordering to make workgroup ids move slowly between chiplet groups.
```

---

### Comment by qedawkins

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 95

**Comment:**

```suggestion
// linearizedId 0's permuted Id is still 0.
// linearizedId 1's permuted Id is 4.
```

---

### Comment by qedawkins

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 134

**Comment:**

```suggestion
// Returns the permuted workgroup IDs (along X and Y dimension).
```

---

### Comment by qedawkins

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 136

**Comment:**

Can you add a comment explaining the math in plain text? Something like
```
l = linearizedId
x_dim = workgroupCountX
y_dim = workgroupCountY

Then the new workgroup ID is computed as follows.

wgp_count = x_dim * y_dim
reordered_id = l / xcd_count + (l % xcd_count) * (wgp_count) / xcd_count
final_id = l >= wgp_count - 1 - wgp_count % xcd_count ? l : reordered_id
```
It helps to have the math written out plainly in addition to the great example you wrote above.

---

### Comment by qedawkins

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 285

**Comment:**

Add a check here that returns failure if `numXCDs <= 1` to reflect the assert in `makeChipletGroupedIds`

---

### Comment by qedawkins

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 145

**Comment:**

Move this assert to the beginning of this helper and add a comment to the assert
```
assert(numXCDs > 1 && "expected more than one xcd for chiplet reordering");
```

---

### Comment by qedawkins

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 367

**Comment:**

nit: IREE style is to always include braces on multi-line nesting
```suggestion
    if (IREE::GPU::TargetAttr attr = getGPUTargetAttr(funcOp)) {
      if (IREE::GPU::TargetChipAttr chipAttr = attr.getChip()) {
        numXCDs = chipAttr.getChipletCount();
      }
    }
```

---

### Comment by qedawkins

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.td`

**Line:** 276

**Comment:**

nit: Can you make the naming between the IR mnemonic and the C++ class consistent? Maybe make both called `workgroup_reordering` and `WorkgroupReorderingAttr` respectively? It makes it easier to guess what the c++ class for a given attribute is later on.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 104

**Comment:**

```suggestion
            << ", reorderWorkgroupsTileSize = "
```

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 285

**Comment:**

MakeChipletGroupedIds is designed exclusively for the chiplet group strategy; so, it seems that an assertion for the swizzle method is unnecessary.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.td`

**Line:** 276

**Comment:**

Sure, will do 


---

### Comment by qedawkins

**File:** `compiler/src/iree/compiler/Codegen/Common/GPU/WorkgroupReordering.cpp`

**Line:** 285

**Comment:**

I'm mainly asking because there is an assert for it in `makeChipletGroupedIds`, meaning if the assert gets stripped and someone calls this function with the wrong number of xcds somewhere, then it's a compiler crash or something bad. Asserts should only be used to catch inconsistencies internal to the compiler.

---

## PR #17645: Enable Workgroup Reordering Based on Translation Info Config Entries

**URL:** https://github.com/iree-org/iree/pull/17645
**State:** MERGED

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp`

**Line:** 101

**Comment:**

I don't think we need to support both variants -- lowercase is fine.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp`

**Line:** 105

**Comment:**

Here, should we check that the string is `none`? This is to diagnose cases when the value is misspelt.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 81

**Comment:**

Why is this changed?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.h`

**Line:** 34

**Comment:**

```suggestion
  enum ReorderWorkgroupsOption { None, Transpose, Swizzle };
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.h`

**Line:** 37

**Comment:**

We don't need this anymore

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 337

**Comment:**

This should probably go to a helper function

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 83

**Comment:**

The enum value is missing, we should also print it.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 326

**Comment:**

Can we use the same enum type and assign it directly?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.h`

**Line:** 34

**Comment:**

We should be careful so that we can distinguish the case when this disables workgroup reordering (when enabled globally) and when it's not set. I think we can use `std::optional<ReorderWorkgroupsStrategy>`

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/config_user_vector_distribute.mlir`

**Line:** 74

**Comment:**

I don't think this tests what it appears to since the attribute matches the global default. IMO we should maintain a test that disables globally-enables reordering (through the CLI flag) and then add a new test that enabled reordering when disabled globally.

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 81

**Comment:**

adapting to changes in the data structure (LLVMGPUPipelineOptions). 

---

### Comment by qedawkins

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 81

**Comment:**

I think this just came from a different clang-format version. Same thing happened to me.

---

### Comment by ScottTodd

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 81

**Comment:**

Yep, clang-format wants this line unchanged: https://github.com/iree-org/iree/actions/runs/9472784149/job/26099808430?pr=17645#step:4:120

You can set up pre-commit to run formatting automatically or install a version close to what the CI uses: https://github.com/iree-org/iree/blob/c1e542d6370473244a8fa9178615cb8a6041b489/.pre-commit-config.yaml#L33-L41

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 81

**Comment:**

Thanks, I have set up pre-commit and it is very helpful. 

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp`

**Line:** 102

**Comment:**

This should be an op error IMO -- the compiler shouldn't crash on invalid IR.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 91

**Comment:**

We should always print the strategy here. If the value is `std::nullopt` we can print something like `<not set>`

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.h`

**Line:** 35

**Comment:**

Remove this

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.h`

**Line:** 40

**Comment:**

Remove this

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp`

**Line:** 93

**Comment:**

```suggestion
      assert(isa<StringAttr>(reorderGroupOption) &&
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp`

**Line:** 96

**Comment:**

```suggestion
         cast<StringAttr>(reorderGroupOption).getValue();
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 82

**Comment:**

nit: you can compare directly with `==`
```suggestion
    if (options.reorderStrategy == ReorderWorkgrupsStrategy::Transpose)
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 88

**Comment:**

nit: If one of the if-else branches has braces around it, the other ones should have them too

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/config_user_vector_distribute.mlir`

**Line:** 5

**Comment:**

```suggestion
// Check that applying `reorder_workgroups = "transpose"` enables workgroup reordering.
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/config_user_vector_distribute.mlir`

**Line:** 8

**Comment:**

What does RWP stand for?

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/config_user_vector_distribute.mlir`

**Line:** 100

**Comment:**

We should maintain a test where global reordering is disabled through this attribute

---

### Comment by bangtianliu

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/config_user_vector_distribute.mlir`

**Line:** 8

**Comment:**

Reorder-WorkgrouP

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp`

**Line:** 96

**Comment:**

not resolved

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp`

**Line:** 103

**Comment:**

I don't remember the exact syntax, but this error can be made more helpful and precise:
```suggestion
          funcOp.emitOpError() << "Unknown " << LLVMGPUAttrNames::kReorderWorkgroups << "value: " << reorderGroupOpton;
```


---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp`

**Line:** 91

**Comment:**

```suggestion
      Attribute reorderWorkgroupOption =
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp`

**Line:** 90

**Comment:**

```suggestion
      // Get the workgroups reorder config and enable the workgroup reordering.
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPULowerExecutableTarget.cpp`

**Line:** 94

**Comment:**

Same here, this should be an error instead an assertion.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 80

**Comment:**

uber nit:
```suggestion
  StringRef reorderStr = "<not set>";
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 95

**Comment:**

```suggestion
            << ", reorderWorkgroupsStrategy = " << reorderStr
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 339

**Comment:**

```suggestion
        *options.reorderStrategy, clReorderWorkgroupsLogSwizzleTile,
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 339

**Comment:**

We could have a helper function `getWokrgroupReoderingStrategy` that takes `options` and returns either the workgroup reordering strategy if set, or the global flag. This will reduce branching in this code and the code below.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/config_user_vector_distribute.mlir`

**Line:** 220

**Comment:**

missing newline

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 190

**Comment:**

```suggestion
  return option.value_or(clReorderWorkgroupsStrategy);
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 186

**Comment:**

```suggestion
// Reconciles workgroup reordering strategy based on the pipeline `option` and the CLI flag.
static ReorderWorkgrupsStrategy getWorkgroupsReoderStrategy(
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/test/ROCDL/config_user_vector_distribute.mlir`

**Line:** 191

**Comment:**

```suggestion
          reorder_workgroups = "none"  // Disable the 'reorderWorkgroups' pass.
```


---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.h`

**Line:** 37

**Comment:**

```suggestion
  bool enableUkernels = false;
  std::optional<ReorderWorkgroupsStrategy> reorderStrategy;
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 188

**Comment:**

```suggestion
static ReorderWorkgroupsStrategy getReorderWorkgroupsStrategy(
```

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Codegen/LLVMGPU/Passes.cpp`

**Line:** 191

**Comment:**

Delete this
```suggestion
```

---

## PR #17539: Add support for Conv2D in new filter layout (Fhwc) : (NchwFchw => NhwcFhwc)

**URL:** https://github.com/iree-org/iree/pull/17539
**State:** CLOSED

### Comment by benvanik

**File:** `compiler/src/iree/compiler/Preprocessing/Common/ConvertConvToChannelsLast.cpp`

**Line:** 474

**Comment:**

use {} around multi-line if statements (here and elsewhere)

---

### Comment by benvanik

**File:** `compiler/src/iree/compiler/Preprocessing/Common/ConvertConvToChannelsLast.cpp`

**Line:** 389

**Comment:**

style: avoid comments that are not explaining more than what the code itself does

---

### Comment by qedawkins

**File:** `compiler/src/iree/compiler/Preprocessing/Common/ConvertConvToChannelsLast.cpp`

**Line:** 36

**Comment:**

W.r.t. the flag, it shouldn't be necessary because this is a preprocessing pass and is already available as a pass option.

---

### Comment by MaheshRavishankar

**File:** `compiler/src/iree/compiler/Preprocessing/Common/ConvertConvToChannelsLast.cpp`

**Line:** 33

**Comment:**

I dont think you need this. All preprocessing passes are just stitched together just like you would with `mlir-opt --pass-pipeline="..."` . So adding a pass option is enough.

---

### Comment by qedawkins

**File:** `compiler/src/iree/compiler/Preprocessing/Common/Passes.td`

**Line:** 48

**Comment:**

```suggestion
           "prefer the output channel of the filter to be outer most (FCHW ==> FHWC)">,
```

---

### Comment by qedawkins

**File:** `compiler/src/iree/compiler/Preprocessing/Common/ConvertConvToChannelsLast.cpp`

**Line:** 32

**Comment:**

nit: trailing whitespace

---

### Comment by qedawkins

**File:** `compiler/src/iree/compiler/Preprocessing/Common/ConvertConvToChannelsLast.cpp`

**Line:** 405

**Comment:**

you only need to check `filterIndices` again here because `input` and `output` indices are unchanged. Maybe remember the result of `isInnerIdentityIndices` from above instead of calling it twice.

---

### Comment by qedawkins

**File:** `compiler/src/iree/compiler/Preprocessing/Common/test/conv_to_channels_last.mlir`

**Line:** 179

**Comment:**

nit: no newline at the end of the file

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/Common/ConvertConvToChannelsLast.cpp`

**Line:** 486

**Comment:**

Don't use auto when the type is not obvious based on the RHS only.

---

### Comment by kuhar

**File:** `compiler/src/iree/compiler/Preprocessing/Common/ConvertConvToChannelsLast.cpp`

**Line:** 495

**Comment:**

no else after return: https://www.llvm.org/docs/CodingStandards.html#don-t-use-else-after-a-return

---


---

**Total PRs with comments:** 79
**Total comments:** 1248
