# Code Style Guide from Reviews - IREE

**Repository:** [iree-org/iree](https://github.com/iree-org/iree)

**Based on 1248 review comments across 79 PRs**

**Generated:** 2026-03-05

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

### C/C++ Tests
- **Use gtest for C API tests** - C API tests can be written in C++
- **Add negative test cases** - test error conditions and invalid inputs
- **Include static_asserts** for C binding type safety

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
