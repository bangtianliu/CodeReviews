# Code Style Guide from Reviews - Tuner

**Repository:** [nod-ai/amd-shark-ai](https://github.com/nod-ai/amd-shark-ai)

**Based on 326 review comments across 64 PRs**

**Generated:** 2026-01-20

---

## Documentation

### Docstrings and Comments
- **Explain what functions return**, especially for tuple returns:
  > "Could you explain what is being returned?"
  ```python
  def compute_next_aligned_bound(original_bound: int, alignment: int) -> int:
      """Pads a bound up to the next multiple of alignment if needed.

      Returns:
          The original bound if already aligned, or the next multiple of alignment.
      """
  ```

- **Add examples in docstrings** for non-obvious functions:
  > "Can you add some example?"
  ```python
  def is_affine_expr_function_of_dim(expr: ir.AffineExpr, position: int) -> bool:
      """Return True if the expression depends on the dimension at position.

      Example:
          d0 + d1 depends on both dim 0 and dim 1
          d1 * 2 depends only on dim 1
      """
  ```

- **Add comments explaining non-obvious code**:
  > "Can you add a one-line comment explaining the None element is for the baseline?"

### Naming
- **Start function names with active verbs**:
  > "We should start function names with active verbs"
  ```python
  # Good
  def compute_next_aligned_bound(...)
  def calculate_shared_memory_usage(...)

  # Avoid
  def maybe_padded_bounds(...)  # Not an active verb
  ```

- **Describe what values mean in help text**:
  > "Why `3 = new option`? Can you describe what 3+ means instead?"

---

## Code Style

### Comments
- **End comment sentences with periods**:
  ```python
  # Good
  # Tuning artifacts.

  # Missing period
  # Tuning artifacts
  ```

### Assertions and Boolean Checks
- **Use direct boolean assertions**, not `== True` or `== False`:
  ```python
  # Good
  assert common.is_affine_expr_function_of_dim(d0, 0)
  assert not common.is_affine_expr_function_of_dim(d0, 1)

  # Avoid
  assert common.is_affine_expr_function_of_dim(d0, 0) == True
  assert common.is_affine_expr_function_of_dim(d0, 1) == False
  ```

### Code Formatting
- **Keep arrays on single lines** when possible:
  > "This formatting is really weird, isn't there any way to keep the second array on a single line?"
  ```python
  # Good
  supported_promotions = ([0, 1], [0, 1, 2])
  assert promote_operands in supported_promotions

  # Awkward formatting
  assert promote_operands == [0, 1] or promote_operands == [
      0,
      1,
      2,
  ]
  ```

### Simplify Code
- **Replace tricks with clear if statements**:
  > "Can we replace this trick with a few if statements?"
  ```python
  # Good - clear
  total_memory = 0
  if 0 in promote_operands:
      total_memory += lhs_memory
  if 1 in promote_operands:
      total_memory += rhs_memory

  # Clever but unclear
  total_memory = (
      int(0 in promote_operands) * lhs_memory
      + int(1 in promote_operands) * rhs_memory
  )
  ```

- **Hoist invariant checks outside loops**:
  > "Why not hoist this check outside of the loop?"
  ```python
  # Good
  if padding_can_be_expensive:
      return list(dims)
  for dim in dims:
      # process dim

  # Inefficient
  for dim in dims:
      if padding_can_be_expensive:
          result.append(dim)
          continue
      # process dim
  ```

- **Don't guard for loops with if**:
  > "Why are you guarding a for loop with an if condition?"
  ```python
  # The for loop handles empty lists fine
  for solution in solutions:
      ...

  # Unnecessary
  if len(solutions) > 0:
      for solution in solutions:
          ...
  ```

---

## Testing

### Test File Hygiene
- **Don't include usage notes in tests** - the README explains how to run tests:
  > "We don't need this usage notes in tests -- all tests are supposed to be executed like this and the README explains it."

### Assertions
- **Compare lists directly** instead of element by element:
  > "You can compare lists"
  ```python
  # Good
  assert knob_assignments == [None, knob1, knob2, knob3]

  # Unnecessary
  assert len(result) == 4
  assert result[0] is None
  assert result[1] == knob1
  ```

- **Pytest prints expected/actual values** - don't add redundant messages:
  > "Doesn't pytest print expected and actual values?"
  ```python
  # Good - pytest will show the diff
  assert "padding =" in str(lowering_config)

  # Redundant message
  assert "padding =" in str(lowering_config), f"Missing padding: {lowering_config}"
  ```

### Test Quality
- **Tests must exercise actual code**:
  > "This test doesn't exercise any of the tuner code. If you change the tuner code, the test won't catch anything."

- **Test negative cases too**:
  > "Do we have any tests that exercises `allow_virtual_mma=False`?"

- **Move variables inside scope where used**:
  > "Can we move these inside the `with` statement, since they are not used anywhere else?"

- **Make output a function argument** in test helpers:
  > "You can make the output be another function argument"

- **Add spaces in MLIR strings**:
  ```python
  # Good
  module_str = """
      builtin.module {
  """

  # Missing space
  module_str = """
      builtin.module{
  """
  ```

---

## Architecture and Design

### PR Size
- **Split large PRs** into smaller focused ones:
  > "Can you split it up into a few smaller PRs? Nearly 3 kLOC is a lot to review, even if this is mostly code motion."

- **Separate unrelated changes**:
  > "Can you move all these logging changes to a separate PR?"

### Code Organization
- **Question if code is truly generic or target-specific**:
  > "Isn't this code generic?"

- **Don't leak implementation details**:
  > "I'm concerned we are leaking codegen pipeline and conv strategies to this generic code. The layering seems off to me."

- **Base classes shouldn't know about derived classes**:
  > "The base class shouldn't know about the derived classes -- can we make this a free function instead?"
  ```python
  # Bad - base knows about derived
  class ConvolutionTunerBase:
      @classmethod
      def get_tuner_for_strategy(cls, strategy):
          return {
              Strategy.IGEMM: IGEMMTuner,  # Base knows derived!
          }[strategy]

  # Good - use a free function
  def get_tuner_for_strategy(strategy):
      return {Strategy.IGEMM: IGEMMTuner}[strategy]
  ```

- **Don't add target-specific code to abstract base**:
  > "This doesn't belong in the abstract base -- not all dispatch parsers even know what a convolution is."

- **Functions shouldn't know about concrete types**:
  > "This function shouldn't know about concrete tuner classes."

### Avoid Unnecessary Complexity
- **Don't create helpers for trivial operations**:
  > "I don't think we need a helper function for this -- this can be as simple as prepending a `None`."

- **Use constants directly** instead of creating local variables:
  > "Can we use this constant directly instead of creating a local variable for it? I think it only hurts readability here"

- **Consider exposing as bindings** instead of duplicating IREE code:
  > "Is this something that we would rather keep entirely in IREE and expose as new bindings?"

---

## Python Best Practices

### Type Annotations
- **Avoid `Any` type** - it sidesteps type checking:
  > "What is the type? Using `Any` effectively sidesteps any type checking"
  ```python
  # Good - specific type
  igemm_details: Optional[IGEMMDetails] = None

  # Avoid
  igemm_details: Any = None
  ```

### Functional Style
- **Use `filter()` for simple filtering**:
  ```python
  # Good
  compatible_intrinsics = filter(
      lambda x: isinstance(x, iree_gpu.MMAIntrinsic),
      compatible_intrinsics
  )

  # Also fine - list comprehension
  compatible_intrinsics = [
      instr for instr in compatible_intrinsics
      if isinstance(instr, iree_gpu.MMAIntrinsic)
  ]
  ```

### Prefer Logic Where Testable
- **Move logic to where it's easier to test**:
  > "Why not handle this in `get_compatible_mfma_intrinsics`? It should be easier to test."

### Unnecessary Copies
- **Don't copy when already correct type**:
  > "dims is already a list"
  ```python
  # Good
  return dims

  # Unnecessary
  return list(dims)  # dims is already a list!
  ```

---

## Debug Code

- **Remove debug prints before merging**:
  > "Drop debug prints"
  ```python
  # Remove before committing
  print(f"matmul_size.K: {matmul_size.K}")
  ```

---

## PR Titles

- **Use descriptive titles** that convey what's changing:
  > "LGTM but consider updating the PR title: 'revisit' does not really convey what's changing, I'd call it something like 'Sync padding for TileAndFuse with IREE changes'"

---

## Common Patterns in Positive Reviews

- "Thanks"
- "Thanks for cleaning this up"
- "LGTM % nit" (LGTM except for a minor issue)
- "+1, especially as we start looking at NN and TN variants"

