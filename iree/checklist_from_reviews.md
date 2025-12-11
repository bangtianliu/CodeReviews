# Code Style Guide from Reviews

**Based on 640 review comments across 74 PRs**

**Generated:** 2025-12-10 18:51:45

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
- Reference: [Iterating over Ranges](https://llvm.org/docs/ProgrammersManual.html#iterating-over-ranges)

### Comments
- **Comment the "why" not the "what"** - Code should be self-documenting for what it does
- **Avoid obvious comments** - Don't comment self-explanatory code
- **Use `//` for single-line comments** in C++
- **Document non-obvious behavior** and edge cases
- Reference: [Commenting](https://llvm.org/docs/CodingStandards.html#commenting)

### Other LLVM Best Practices
- **Use `llvm::to_vector`** or `llvm::to_vector_of<T>` for converting ranges to vectors
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


## Common Review Patterns

### Performance Optimization
- Add performance tests alongside correctness tests
- Document empirically determined values (e.g., "rowGroupSize=16 for MI300X")
- Avoid premature optimization - trust compiler for simple operations (SROA, DCE)
- Profile before optimizing unclear cases

### MLIR Pattern Rewriting
- **Never mutate IR directly** - always use rewriter methods
- Use `modifyOpInPlace` for in-place modifications
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

### Code Organization
- Split large PRs into focused, reviewable chunks
- Use `NFC` (No Functional Change) in commit titles for refactoring
- Extract long functions into smaller helpers matching conceptual stages
- Remove commented-out code completely (causes compiler warnings)
- Move error checks before code that depends on them passing

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


