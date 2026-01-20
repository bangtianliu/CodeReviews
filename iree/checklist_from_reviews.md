# Code Style Guide from Reviews

**Based on 828 review comments across 148 PRs**

**Generated:** 2026-01-19

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
- Flatten code structure for better readability
  ```cpp
  // Good
  if (!condition)
    return;
  doWork();

  // Bad
  if (condition) {
    doWork();
  }
  ```
- Reference: [Use Early Exits](https://llvm.org/docs/CodingStandards.html#use-early-exits-and-continue-to-simplify-code)

### Casts and Type Conversions
- **Prefer C++ style casts** over C-style casts
- **Use `llvm::cast<T>`** when type is guaranteed (asserts on failure)
- **Use `llvm::dyn_cast<T>`** when type might not match (returns nullptr on failure)
- **Use `llvm::isa<T>`** for type checking without casting
- **Never assert after `llvm::cast`** - it already asserts internally
- Reference: [isa/cast/dyn_cast](https://llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates)

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
- **Do not recalculate loop end** - cache the end value:
  ```cpp
  // Good
  for (size_t i = 1, e = basis.size(); i <= e; ++i) { ... }

  // Bad - recalculates size each iteration
  for (size_t i = 1; i <= basis.size(); ++i) { ... }
  ```
- Reference: [Iterating over Ranges](https://llvm.org/docs/ProgrammersManual.html#iterating-over-ranges)

### Comments
- **Comment the "why" not the "what"** - Code should be self-documenting for what it does
- **Avoid obvious comments** - Don't comment self-explanatory code
- **Use `//` for single-line comments** in C++
- **Document non-obvious behavior** and edge cases
- **End comment sentences with periods**:
  ```cpp
  // Good - proper punctuation
  // Same correlation group: same resource, different offsets (noalias).

  // Bad - missing period
  // Same correlation group: same resource, different offsets (noalias)
  ```
- **Use correct terminology** - "induction variable" not "iteration variable"
- Reference: [Commenting](https://llvm.org/docs/CodingStandards.html#commenting)

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
- **Use `llvm::to_vector`** or `llvm::to_vector_of<T>` for converting ranges to vectors:
  ```cpp
  // Good
  return llvm::to_vector(delinearizedLaneId);

  // Avoid - manual construction
  return SmallVector<Value>(delinearizedLaneId.begin(), delinearizedLaneId.end());
  ```
- **Use `llvm::is_detected`** instead of custom SFINAE helpers for type detection
- **Prefer `std::optional`** over nullable pointers when appropriate
- **Use LLVM ADT containers** (SmallVector, DenseMap, etc.) for better performance
- **Follow naming conventions** - UpperCamelCase for types, lowerCamelCase for variables/functions

---

## General Guidelines

### Documentation
- Add clear comments explaining non-obvious logic
- Document function parameters and return values
- Update related documentation when changing behavior
- Add examples in comments for complex operations

### Naming Conventions
- Use descriptive, self-explanatory variable names
- Follow project naming conventions consistently
- Avoid abbreviations unless widely understood
- Fix typos in variable/function names
- **Include units in variable names** to clarify meaning:
  ```cpp
  // Good - clear what the unit is
  int64_t numRowElems = kLDSBankWidthBits / elemBitwidth;
  int64_t numAccessElems = schedule->kSizes.back();

  // Ambiguous - "width" could mean bits, bytes, or elements
  int64_t rowWidth = kCacheLineSizeBits / bitwidth;
  int64_t accessWidth = schedule->kSizes.back();
  ```

### Code Structure
- Keep functions focused and single-purpose
- Extract complex logic into helper functions
- Avoid deep nesting - consider early returns
- Group related functionality together
- Use NFC (No Functional Change) commits for refactoring

### Testing
- Add tests for new functionality
- Include edge case testing
- Update existing tests when behavior changes
- Verify tests actually test the intended behavior

### Error Handling & Validation
- Add proper error messages for failure cases
- Validate inputs at function boundaries
- Use assertions for invariants
- Add verifiers for new operations/attributes
- Handle edge cases explicitly

## IREE/MLIR Specific

### Transform Dialect
- Use proper transform operation interfaces
- Document transform operation semantics clearly
- Add verifiers for transform operations
- Test transform sequences thoroughly

### Attributes & Operations
- Use appropriate attribute types (DictionaryAttr, ArrayAttr, etc.)
- Add proper assembly format for custom operations
- Implement necessary operation interfaces
- Validate attribute values in verifiers
- **Define helpers in ODS** as extra class declarations instead of repeating logic:
  ```tablegen
  // Define helper methods in ODS
  let extraClassDeclaration = [{
    bool hasExplicitIndexInput() { return getInputs().size() == 2; }
    Value getInputIndex() {
      assert(hasExplicitIndexInput());
      return getDpsInputOperand(1)->get();
    }
  }];
  ```
- **Consider returning nullptr** instead of asserting when appropriate:
  ```cpp
  // Alternative: return nullptr when condition not met
  Value getInputIndex() {
    return hasExplicitIndexInput() ? getDpsInputOperand(1)->get() : nullptr;
  }
  ```

### Python Bindings
- Use proper pybind11 patterns
- Add docstrings for Python-exposed functions
- Handle Python exceptions appropriately
- Test Python bindings separately

### Tuning & Configuration
- Validate lowering configurations
- Check for default attribute compatibility
- Verify tuning specs match target architecture
- Document configuration options
- **Use target environment attributes** instead of hardcoded constants:
  ```cpp
  // Bad - hardcoded constant
  constexpr int64_t kLDSBankWidthBits = 32 * 4 * 8;  // Wrong for some targets

  // Good - use target environment
  int64_t ldsBankWidth = targetEnv.getLDSBankWidth();  // MI300: 32, MI355: 64
  ```
- **Remember LDS != cache** - use correct terminology and constants for shared memory vs cache


## Common Review Patterns

### Performance Optimization
- Add performance tests alongside correctness tests
- Document empirically determined values (e.g., "rowGroupSize=16 for MI300X")
- Avoid premature optimization - trust compiler for simple operations (SROA, DCE)
- Profile before optimizing unclear cases

### MLIR Pattern Rewriting
- **Never mutate IR directly** - always use rewriter methods
- Use `modifyOpInPlace` for in-place modifications
- **Use Builder convenience methods** instead of verbose attribute creation:
  ```cpp
  // Good - use Builder convenience methods
  Builder b(funcOp.getContext());
  correlatedIndices.push_back(b.getI32IntegerAttr(otherBinding.index()));

  // Verbose - avoid
  correlatedIndices.push_back(IntegerAttr::get(
      IntegerType::get(funcOp.getContext(), 32), otherBinding.index()));
  ```
- **Use named structs instead of tuples** for better clarity:
  ```cpp
  // Good - named fields
  struct F32Fields { Value sign; Value biasedExp; Value mantissa; };
  F32Fields extractF32Fields(Value i32Val);

  // Avoid - unnamed tuple fields
  std::tuple<Value, Value, Value> extractF32Fields(Value i32Val);
  ```
- Reference: [MLIR Pattern Rewriter](https://mlir.llvm.org/docs/PatternRewriter/#common-pattern-drivers)

### Transform Dialect Patterns
- Return enums, not integers from matcher operations
- Look at existing transform ops for patterns (e.g., `transform.match.param.cmpi`)
- Use type switch pattern for type-erased types that need derived type info
- Avoid unnecessary lambda wrappers when types already match

### Testing Best Practices
- **Avoid "change detector" tests** - tests that break on any dialect update
- Test edge cases: wrong number of arguments, wrong types, null/nullopt states
- Test attributes on operations that already have attributes
- Organize tests logically (error cases near related valid cases)
- Keep disabled feature tests to prevent regression
- Split test files with `--split-input-file` for better isolation
- **Put test cases in separate splits** so checks are local to the code under test:
  ```mlir
  // CHECK-LABEL: @test_success
  func.func @test_success() { ... }

  // -----

  // Put failure case in a new split, not after success case
  // CHECK-LABEL: @test_failure
  func.func @test_failure() { ... }
  ```

### Code Organization
- Split large PRs into focused, reviewable chunks
- Use `NFC` (No Functional Change) in commit titles for refactoring
- Extract long functions into smaller helpers matching conceptual stages
- Remove commented-out code completely (causes compiler warnings)
- Move error checks before code that depends on them passing
- **Remove debug prints before merging**:
  ```cpp
  // Remove before committing
  llvm::errs() << "loweringConfig: " << loweringConfig << "\n";
  ```
- **Mark helper functions as `static`**:
  ```cpp
  // Good
  static void maybeAppendType(MLIRContext *ctx, SmallVectorImpl<Type> &types) { ... }

  // Bad - missing static
  void maybeAppendType(MLIRContext *ctx, SmallVectorImpl<Type> &types) { ... }
  ```
- **Remove unused variables** - don't leave declared but unused code
- **Land bindings separately** to keep PRs small and focused

### Error Messages
- Make error messages precise and helpful
- Use `emitOpError()` for operation-specific errors
- Check grammar: closing quotes, spacing, punctuation
- Include actual values in error messages when possible
  ```cpp
  funcOp.emitOpError() << "Unknown " << AttrName << " value: " << actualValue;
  ```

### Python Bindings Specific
- Allocate/free memory on Python side (see existing examples)
- Add static asserts on C bindings implementation
- Use distinct test values to verify ordering (e.g., [1,2,3] not [1,1,1])
- Test bindings across iree/c/python layers
- **Don't modify IR from Python** - creates dangling references
- Split test cases into separate inputs instead of modifying shared IR
- Use typedefs on C API side with static_assert for type safety
- Document whether types are signed/unsigned (i32 vs u32)

### Attribute Validation
- Check for nested modules if not supported
- Verify attribute symbols match (e.g., `#` prefix if required)
- Test behavior with unexpected attributes on operations
- Document allowed/disallowed attribute combinations

### Naming and Clarity
- Choose names that reflect actual semantics, not just one use case
- Simplify loops when loop variables aren't used elsewhere
  ```cpp
  // Prefer this when suffix isn't used:
  for (unsigned suffix = 0; check; ++suffix) { ... }
  ```
- Use descriptive names over generic ones
- Follow portable naming conventions (e.g., GPU dialect style)

### Memory and Resource Management
- Verify `std::optional` comparison operators handle nullopt correctly
- Remove redundant assertions (e.g., after `llvm::cast` which already asserts)
- Allocate memory on appropriate side of language bindings

### C++ Style
- **`public` is default for structs** - don't redundantly inherit:
  ```cpp
  // Good
  struct TruncFToFP8 final : OpRewritePattern<arith::TruncFOp> { ... };

  // Unnecessary - public is already default for structs
  struct TruncFToFP8 final : public OpRewritePattern<arith::TruncFOp> { ... };
  ```
- **Use `llvm::append_range`** instead of manual loops when appending:
  ```cpp
  // Good
  llvm::append_range(mergedMatchers, foreachMatchOp.getMatchers());

  // Unnecessary loop
  for (auto matcher : foreachMatchOp.getMatchers())
    mergedMatchers.push_back(matcher);
  ```


