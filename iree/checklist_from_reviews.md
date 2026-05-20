# Code Style Guide from Reviews - IREE

**Repository:** [iree-org/iree](https://github.com/iree-org/iree)

**Based on 1475 review comments across 118 PRs**

**Generated:** 2026-05-05 (last updated 2026-05-20 with 103 additional kuhar comments)

---

## LLVM Coding Standards

**Reference:** [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html) | [Programmer's Manual](https://llvm.org/docs/ProgrammersManual.html)

### Auto Type Deduction
- **Use `auto` when type is obvious from RHS** - e.g., `auto *Ptr = cast<Foo>(Bar);`
- **Spell out types when not obvious** - Don't use `auto` for function return values that aren't clear
- **Beware unnecessary copies with auto** - Use `const auto&` or `auto&` for range-based loops when iterating over containers
  ```cpp
  // Good - avoids copying vector
  for (const auto& value : container) { ... }

  // Bad - creates copies
  for (auto value : container) { ... }
  ```
- Reference: [Use auto Type Deduction](https://llvm.org/docs/CodingStandards.html#use-auto-type-deduction-to-make-code-more-readable)

### Early Exits and Code Simplification
- **Use early returns to reduce nesting** - Avoid deep if-else chains
- **Use `continue` in loops to simplify logic**
- **Use `dyn_cast` with null check instead of `isa` + `cast`**:
  ```cpp
  // Good - use dyn_cast and continue when null
  auto tensorTy = dyn_cast<TensorType>(type);
  if (!tensorTy)
    continue;

  // Avoid
  if (isa<TensorType>(type)) {
    auto tensorTy = cast<TensorType>(type);
    // ...
  }
  ```
- Reference: [Use Early Exits](https://llvm.org/docs/CodingStandards.html#use-early-exits-and-continue-to-simplify-code)

### Casts and Type Conversions
- **Prefer C++ style casts** over C-style casts (use `static_cast`, not `(int64_t)`)
- **Use `llvm::cast<T>`** when type is guaranteed (asserts on failure)
- **Use `llvm::dyn_cast<T>`** when type might not match (returns nullptr on failure)
- **Use `llvm::isa<T>`** for type checking without casting
- **Never assert after `llvm::cast`** - it already asserts internally
- **Never assert before `llvm::cast`** - the cast already asserts the same condition; the assert is redundant
- Reference: [isa/cast/dyn_cast](https://llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates)

### Integer Types
- **Avoid `unsigned`** unless representing bitfields or modular arithmetic
- **Prefer signed integer types** (`int64_t`, `int32_t`) over `unsigned`
- Mixing signedness causes bugs the compiler can't diagnose
- Reference: [Google C++ Style Guide - Integer Types](https://google.github.io/styleguide/cppguide.html#Integer_Types)

### Constructor Initialization
- **Do not use braced initializer lists to call constructors** - use `=` syntax instead:
  ```cpp
  // Good
  SmallVector<Value> reverseValues = {kValue, loopCarryValues[0]};

  // Avoid
  SmallVector<Value> reverseValues({kValue, loopCarryValues[0]});
  ```
- Reference: [Abseil Tip #88](https://abseil.io/tips/88) | [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html#do-not-use-braced-initializer-lists-to-call-a-constructor)

### Iterators and Ranges
- **Use `llvm::zip_equal`** for iterating over multiple ranges of equal length
- **Use range-based for loops** when possible
- **Use `llvm::enumerate`** when you need indices
- **Use structured bindings** with `llvm::enumerate`:
  ```cpp
  // Good - structured bindings
  for (auto [idx, value] : llvm::enumerate(container)) { ... }

  // Avoid - using .value() accessor
  for (auto item : llvm::enumerate(container)) {
    item.value().doSomething();  // Less readable
  }
  ```
- **Use `llvm::equal`** instead of manual size + element comparison
- Reference: [Iterating over Ranges](https://llvm.org/docs/ProgrammersManual.html#iterating-over-ranges)

### Comments and Style
- **Comment the "why" not the "what"** - Code should be self-documenting
- **Avoid obvious comments** - Don't comment self-explanatory code
- **Remove or simplify verbose comments** if documented elsewhere (e.g., in header)
- **Use `//` for single-line comments** in C++
- **Document non-obvious behavior** and edge cases
- **Don't use blank lines when you don't have to** - resist starting functions with a blank line
- **Start error messages with a lower-case letter** and finish without a period
- **Use descriptive variable names** - avoid single-letter names like `J`, `J1`, `vJ`; use `lhs`/`rhs` or semantically meaningful names
- **Preserve existing comments when refactoring** - don't silently remove comments in moved/restructured code
- **Don't refer to "previous" / "new" / "old" code in comments** - readers don't know the timeframe; always describe the *current* state of the codebase
- **Don't use non-ASCII characters** in source files (incl. test files)
- **Cite the source** for safety-critical data tables (e.g., hardware bank/phase tables) - a single wrong entry can silently shift codegen with no test failure
- Reference: [LLVM Coding Standards - Vertical Whitespace](https://google.github.io/styleguide/cppguide.html#Vertical_Whitespace)

### Function Parameters
- **Pass vectors as `ArrayRef`** instead of by value to avoid copies and allocations:
  ```cpp
  // Good - no copies
  static bool isConsumerCompatible(ArrayRef<unsigned> reductionDims) { ... }

  // Bad - causes copies and allocations
  static bool isConsumerCompatible(SmallVector<unsigned> reductionDims) { ... }
  ```
- **Use `ArrayRef` for read-only access** to contiguous memory
- **Use `MutableArrayRef`** when the function needs to modify elements

### Other LLVM Best Practices
- **Use `llvm::to_vector`** or `llvm::to_vector_of<T>` for converting ranges to vectors
- **Use `llvm::append_range`** instead of manual loops when appending
- **Use `llvm::filter_vector`** for filtering operations
- **Use `isZeroInteger`, `isOneInteger`, `isConstantIntValue`** instead of manual checks
- **Use `value_or`** with `std::optional` instead of explicit conditionals
- **Prefer `std::optional`** over nullable pointers when appropriate
- **Use LLVM ADT containers** (SmallVector, DenseMap, etc.) for better performance
- **Follow naming conventions** - UpperCamelCase for types, lowerCamelCase for variables/functions (camelCase style)
- **Use `llvm::is_contained({...}, value)`** for membership checks instead of chained `==` or switch:
  ```cpp
  // Good
  if (llvm::is_contained({Model::A, Model::B}, model)) return 64;
  return 32;
  ```
- **Use `llvm::all_equal`** to check that a range's elements are identical
- **Use existing LLVM string-join helpers** (`llvm::join`, `llvm::interleaveComma`) instead of hand-rolled loops
- **`static` is redundant on namespace-scope `constexpr` variables** - `constexpr` already implies internal linkage
- **Use `ShapedType::kDynamic` / `ShapedType::isDynamic(...)`** - never hardcode `-1` in comments or code as the dynamic sentinel; it is an implementation detail subject to change
- **Mark namespace-scope helpers `static`** if they have no external callers - otherwise they appear as dead external-linkage symbols

---

## MLIR Pattern Rewriting

### Pattern Best Practices
- **Never mutate IR directly** - always use rewriter methods
- **Use `modifyOpInPlace`** for in-place modifications
- **Operations must go through the rewriter** - even block operations
- **Use `rewriter.eraseOp()` and `rewriter.eraseBlock()`** properly
- **Use `notifyMatchFailure()`** to provide debug information when matches fail
- **Use `OpBuilder::create<OpType>`** pattern for creating ops

### TypeSwitch Pattern
- **Use TypeSwitch for type-based dispatch**:
  ```cpp
  // Good
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](linalg::GenericOp genericOp) {
        return handleGeneric(genericOp);
      })
      .Default([](Operation *) { return failure(); });
  ```

### Builder Convenience Methods
- **Use Builder convenience methods** instead of verbose attribute creation:
  ```cpp
  // Good - use Builder convenience methods
  Builder b(funcOp.getContext());
  correlatedIndices.push_back(b.getI32IntegerAttr(otherBinding.index()));

  // Verbose - avoid
  correlatedIndices.push_back(IntegerAttr::get(
      IntegerType::get(funcOp.getContext(), 32), otherBinding.index()));
  ```

---

## ODS (TableGen) Best Practices

### Verification
- **Move shape verification to ODS** when possible using traits
- **Use `TypesMatchWith`** for type constraints between operands/results
- **Use `ParentOneOf`** trait to check parent operation type
- **Prefer existing traits** over custom CPred when available
- **Keep custom CPred in ::verify()** if not reusable across ops
- **Move region checks to `verifyRegions()`** - use `cast` instead of `dyn_cast` in verifyRegions since the region is already validated
- **Avoid double negation** in conditionals:
  ```cpp
  // Good
  if (ShapedType::isStatic(inputDimSize) && ShapedType::isStatic(outputDimSize)) {

  // Avoid - double negation
  if (!ShapedType::isDynamic(inputDimSize) && !ShapedType::isDynamic(outputDimSize)) {
  ```
- **Prefer simple loops over complex lambdas** in verification code for readability
- **Keep operand order consistent** with the assembly format definition

### Extra Class Declarations
- **Define helper methods in ODS** as `extraClassDeclaration` instead of repeating logic:
  ```tablegen
  let extraClassDeclaration = [{
    bool hasExplicitIndexInput() { return getInputs().size() == 2; }
    Value getInputIndex() {
      assert(hasExplicitIndexInput());
      return getDpsInputOperand(1)->get();
    }
  }];
  ```

### Assembly Format
- **Use named inputs and inits** instead of unnamed variadics when possible
- **Avoid unnamed variadics** for all operands (poor pattern from LinalgExt)

---

## Testing Best Practices

### Test Organization
- **Put tests in existing files** rather than creating new test files for small additions
- **Use `--split-input-file`** for better test isolation
- **Put test cases in separate splits** so checks are local to the code under test
- **Organize tests logically** (error cases near related valid cases)
- **Check indexing maps** in addition to operations when testing tiling

### Test Quality
- **Simplify test IR** - turn constants into function arguments when they don't affect the test
- **Remove unnecessary attributes** (e.g., lowering_config) if not being tested
- **Use `CHECK-SAME:`** for readability when splitting long CHECK lines
- **Reduce whitespace** - don't add unnecessary blank lines in tests
- **Tests must capture dataflow** - CHECK lines should verify connections between operations, not just individual op existence
- **CHECK lines should verify actual values** - e.g., check layout element tile sizes, not just that a layout exists
- **Don't add dead code** - land interface implementations alongside e2e tests that exercise them
- **Remove tests that don't add value** - e.g., if a test only checks propagation already tested elsewhere
- **Watch for trivially-passing `CHECK-NOT`** - if the function under test exits early due to missing setup (e.g., no `hal.executable.target`), the test proves nothing. Always pair with a positive companion that exercises the pipeline.
- **Don't write tautological unit tests** - "checks that the constant is whatever the constant is" is not a test
- **Keep test fixtures consistent across a PR** - if you add a new test that pins `root_op` on `linalg.fill`, don't remove the same annotation from an existing sibling test; you silently change what the existing test exercises
- **Lock in the actual emitted IR**, not just structural surroundings - if the pass emits divisibility asserts on specific dims, CHECK those asserts, not just the surrounding dictionary
- **Use `--verify-diagnostics`** for expected-error tests instead of ad-hoc CHECK plumbing
- **Move target-specific tests to the target directory** (e.g., `gpu_nested_layout_*` rocdl-only tests go under `rocdl/`)
- **Multi-op comparator/region bodies need a multi-op test case** - a one-op body doesn't exercise the result-remapping loop
- **Cover dynamic-shape paths or explicitly bail** - if the pass can't handle dynamic shapes, add a static-shape guard *and* a dynamic-shape test that confirms the bail-out

### Test Coverage Gap Patterns
These review patterns repeat - audit new ops/passes against them:
- **Negative subgroup_size / element type / batch dim** mismatches when canTargetIntrinsic-style filters are involved
- **Multiple compatible options** to verify de-duplication and ordering
- **Dynamic shapes** for any pass that walks `staticLoopRanges`
- **Block-vs-non-block MMA intrinsics** when the pipeline only accepts one
- **Targeted unit/lit tests for "magic table" data** (phase groups, bank layouts) so a stray edit breaks the test, not silently the codegen
- **Conflict-free / no-op paths** must be distinguishable from genuine failures in tests (see Type Design below)

### C/C++ Tests
- **Use gtest for C API tests** - C API tests can be written in C++
- **Add negative test cases** - test error conditions and invalid inputs
- **Include static_asserts** for C binding type safety
- **Avoid system includes** (`<vector>`, `<string>`, etc.) when LLVM ADT alternatives exist (`SmallVector`, `StringRef`)
- **Use `testing::ElementsAreArray`** for gtest array comparisons instead of allocating a `SmallVector` to compare against
- **Use a for loop** instead of unrolled repeated `EXPECT_*` calls when checking a sequence

---

## Code Organization

### PR Structure
- **Split large PRs** into focused, reviewable chunks
- **Use `NFC` (No Functional Change)** in commit titles for refactoring
- **Land bindings separately** to keep PRs small
- **Separate unrelated changes** into different PRs

### Function Organization
- **Extract long functions into smaller helpers** matching conceptual stages
- **Move variable declarations close to use** - don't declare at function start
- **Move checks before dependent code** - validate inputs early
- **Mark helper functions as `static`**
- **Remove unused code completely** - don't leave commented-out code
- **Drop unused parameters** - an unused `Attribute attr` argument implies the function is "pipeline-aware" when it isn't; mislead removed by deleting the param
- **For fixed-position returns, use `std::tuple`** rather than out-params when the order (e.g., `M, N, K`) is part of the contract and isn't exposed publicly
- **When you split a target, actually split the source** - don't list the same `.cpp` in both the parent library and the new split library in CMake/Bazel; that defeats the split

### SmallVector Patterns
- **Use `SmallVector::insert`** for inserting elements:
  ```cpp
  SmallVector<utils::IteratorType> newIteratorTypes = genericOp.getIteratorTypesArray();
  newIteratorTypes.insert(newIteratorTypes.begin() + index, utils::IteratorType::parallel);
  ```
- **Initialize with size and value**:
  ```cpp
  SmallVector<utils::IteratorType> types(rank, utils::IteratorType::parallel);
  types[reductionIdx] = utils::IteratorType::reduction;
  ```

---

## C/Python Bindings

### C API
- **Prefer invalid states to be unrepresentable** - don't return "invalid" marker values
- **Handle failures in Python bindings** not C bindings (assert in C, check in Python)
- **Use typedefs with static_assert** for type safety
- **Document whether types are signed/unsigned** (i32 vs u32)

### Python Bindings
- **Allocate/free memory on Python side** (see existing examples)
- **Test bindings across iree/c/python layers**
- **Use distinct test values** to verify ordering (e.g., [1,2,3] not [1,1,1])

---

## Transform Dialect

### Matcher Operations
- **Return enums, not integers** from matcher operations
- **Look at existing transform ops** for patterns (e.g., `transform.match.param.cmpi`)
- **Use SingleOpMatcher trait** only when the operation handle is used
- **Remove unused handles** from transform op definitions
- **Provide clear documentation** with actual syntax examples in op descriptions

### Naming
- **Use descriptive names** that reflect semantics (e.g., `dims_equal` not `size_equals`)
- **Match existing naming conventions** in the transform dialect

---

## GPU Distribution and Reduction Patterns

### Code Reuse
- **Share helpers between similar patterns** - e.g., `DistributeArgCompare` and `DistributeMultiReduction` should share reduction logic with a parameterized combiner
- **Don't special-case** for trivial dimensions (e.g., `elementTile == 1`) - let canonicalization fold these naturally
- **Don't add redundant create_mask** operations with all-true inBounds

### Pattern Structure
- **Check invariants before creating IR** - validate all pattern preconditions before emitting any operations
- **Use early return instead of else** after failure checks
- **Use `continue` instead of `else`** in loop bodies
- **Inline single-use values** - don't create variables used only once
- **Store repeated expressions** in variables - e.g., `disInitValue3.getType().getElementType()` used multiple times

### Naming and Consistency
- **Function names should match their purpose** - `analyzeComparatorForSubgroupReduce` vs `doThreadReduction` is inconsistent
- **Similar helper functions should have distinguishable names** that reflect what they do differently
- **Be consistent with existing naming** in the codebase (e.g., `doThreadReduction` matches `DistributeMultiReduction`)

---

## Common Review Patterns

### Code Quality
- **Remove debug prints before merging**:
  ```cpp
  // Remove before committing
  llvm::errs() << "loweringConfig: " << loweringConfig << "\n";
  ```
- **Don't rely on global variables** - thread safety concerns
- **Check for needed includes** - remove unused, ensure needed are present
- **Simplify unnecessary arrays** - if only one element, don't use array

### Documentation
- **Update link text when updating URLs** in documentation
- **Edit notebooks in text editor** to avoid structural changes
- **Check that external links are valid** before committing

### Structs and Classes
- **`public` is default for structs** - don't redundantly inherit:
  ```cpp
  // Good
  struct TruncFToFP8 final : OpRewritePattern<arith::TruncFOp> { ... };

  // Unnecessary - public is already default for structs
  struct TruncFToFP8 final : public OpRewritePattern<arith::TruncFOp> { ... };
  ```
- **Use named structs** instead of tuples for better clarity:
  ```cpp
  // Good - named fields
  struct ArgmaxCombinerOps { Operation* maxOp; Operation* selectOp; Operation* cmpOp; };

  // Avoid - unnamed tuple fields
  std::tuple<Operation*, Operation*, Operation*> collectOps();
  ```

### Error Messages
- **Make error messages precise and helpful**
- **Use `emitOpError()`** for operation-specific errors
- **Use `notifyMatchFailure()`** to explain why pattern matches fail
- **Include actual values** in error messages when possible

---

## Type Design and Error Semantics

### Avoid Double-Nullable Types
- **Avoid `std::optional<T*>` / `std::optional<DyncCastable>`** - if the inner value can already be null, the optional layer is redundant *and* confusing. A `dyn_cast` that returns null already encodes "absent"; don't wrap it in `optional`.
- **Avoid `FailureOr<std::optional<T>>`** for the same reason.
- **If a function can return "no result needed" vs. "result couldn't be computed"**, encode that explicitly in the return type - don't let the caller infer it from `null`.

### Distinguish "Not Needed" from "Failed"
A persistent review pattern: **`failure()` must mean one thing per function.** Don't overload it to mean both "the requested transform isn't needed (use default)" *and* "something went wrong (bail)" - callers can't tell them apart and silently regress in the "not needed" case. Options:
- Return `std::optional<Result>` where `nullopt` = "not needed" and exceptions/diagnostics handle real failure
- Return a sentinel value with an explicit `isIdentity()` / `isNoOp()` helper (e.g., `XorShuffleParams{0,0}` with `isIdentity()`)
- Use LLVM-style RTTI with distinct subclasses for the conceptually different outcomes (e.g., `XorSwizzle`, `PadSwizzle`, `NoSwizzle`)

### Verifier Tightening
- **Tighten the verifier rather than relying on downstream pass assertions** - if a region is allowed to capture an external i1 that the lowering can't handle, fix the verifier (or the lowering's `IRMapping`), don't leave it as an internal-assert landmine
- **Use `lookupOrDefault`** when remapping values that may not have been cloned into the new region

---

## Hardware Modeling and GPU Targets

### Granularity
- **Coarse target enums (cdna3 / cdna4 / rdna3) hide per-generation differences** - bank counts, phase tables, and LDS layouts vary; don't assume one enum captures all relevant hw state
- **Derive related target params from a single source** - e.g., number of threads and LDS bank count should both follow from the generation attribute, not be hardcoded independently
- **Trim dead target entries** - drop enum cases for hardware IREE no longer targets (cdna1, rdna1-2, etc.)
- **Default configs should fall back to safe minimums** (e.g., bank width) rather than asserting/failing on unknown targets

### Safety-Critical Tables
- **Cite the spec/source** for hardware tables (phase groups, bank conflict maps) in a comment
- **Assert documented preconditions** (e.g., `assert(numThreads == 64)` for a CDNA4-only branch)
- **Validate the table covers the actual access widths** - if the table is built for `ds_read_b128`, document/guard against wider reads (e.g., 32-byte scaled-MMA operands)
- **Pin tables with targeted unit tests** so an accidental edit becomes a test failure rather than a silent codegen regression

### Wiring Constraints Through All Paths
- **If you add a constraint (e.g., DMA min-access-width) gated on a flag, pass that flag everywhere the constraint applies** - it's easy to leave a sibling call defaulted to `false` and silently disable the constraint on the only path that motivated it
- **Mirror filters between configs and constraint generators** - if the matmul/conv config skips block MMA intrinsics, the constraint generator must skip them too, or the tuner emits choices the pipeline never selects
- **Don't broaden compiler/test flags more than necessary** - `--iree-input-demote-f64-to-f32=false` for one test should not apply to the whole suite; split it into its own suite in both BUILD.bazel and CMakeLists.txt

### Lowering vs. Target Coverage
- **Prefer "make the lowering work everywhere" over "add a target-env entry to gate it"** - per kuhar: most lowerings can be made to work for all targets by decomposing to supported scalars, instead of gating with a new target capability bit
- **`gpu.shuffle` is target-agnostic** - it does not know native bitwidths; comments/asserts that claim otherwise are wrong

---

## Build, CI, and Tooling

### Bazel/CMake
- **Don't add `allow_empty` globs reflexively** - if there are no files to match, drop the whole `exports_files` / glob instead of papering over with `allow_empty = True`
- **A split library target must own its sources exclusively** - remove the split source from the parent's `SRCS` when you create the new target, otherwise the source compiles in both

### CI
- **Inline shell in `.github/workflows/*.yml` should be a checked-in script** when it's non-trivial - actions are not locally reproducible, but a script in the repo is. Bonus: the workflow can also call the script.
- **Don't drop `iree-hip` / `iree-rocm` duplicate flags** - one is sufficient (kuhar prefers `iree-rocm`)

### Linters
- **Don't add entries to `build_tools/linters/typos.toml`** to silence transient warnings - fix the typo or leave the file alone

---

## Review Etiquette / PR Hygiene

- **`LGTM % minor comments but wait for another approval`** is the common pattern for cross-area reviews; respect domain ownership
- **A `CHANGES_REQUESTED` for "I don't understand why we lost test coverage"** is binding - any PR that removes existing checks needs to justify the deletion in the PR body
- **Drive-by nits are explicitly labeled** - don't block PRs on `nit:` comments unless the author asks
- **Suggestion blocks** (` ```suggestion`) should be preferred over prose for one-line fixes - they're clickable
