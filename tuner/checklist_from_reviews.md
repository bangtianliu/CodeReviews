# Code Style Guide from Reviews - Tuner

**Repository:** [nod-ai/amd-shark-ai](https://github.com/nod-ai/amd-shark-ai)

**Based on 629 review comments across 58 PRs**

**Generated:** 2026-03-05

---

## Documentation

### Docstrings and Comments
- **Explain what functions return**, especially for tuple returns:
  ```python
  def compute_next_aligned_bound(original_bound: int, alignment: int) -> int:
      """Pads a bound up to the next multiple of alignment if needed.

      Returns:
          The original bound if already aligned, or the next multiple of alignment.
      """
  ```

- **Add examples in docstrings** for non-obvious functions:
  ```python
  def is_affine_expr_function_of_dim(expr: ir.AffineExpr, position: int) -> bool:
      """Return True if the expression depends on the dimension at position.

      Example:
          d0 + d1 depends on both dim 0 and dim 1
          d1 * 2 depends only on dim 1
      """
  ```

- **Add comments explaining non-obvious code** - especially placeholder values and workarounds
- **Update docstrings** when adding new parameters

### Naming
- **Start function names with active verbs**:
  ```python
  # Good
  def compute_next_aligned_bound(...)
  def calculate_shared_memory_usage(...)
  def get_candidates_ordered_by_speedup(...)

  # Avoid
  def maybe_padded_bounds(...)  # Not an active verb
  ```

- **Use descriptive parameter names** like `prune_slow_candidates` not just `prune`
- **File names should reflect content** - `rocm_utils.py` not `rocm_libtuner.py` for generic utilities

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

### Type Annotations
- **Avoid `Any` type** - it sidesteps type checking:
  ```python
  # Good - specific type
  igemm_details: Optional[IGEMMDetails] = None

  # Avoid
  igemm_details: Any = None
  ```
- **Add type hints to functions** - especially public functions
- **Use `dict[K, V]` syntax** (Python 3.9+) instead of `Dict[K, V]`:
  ```python
  # Good
  conv_to_igemm_dim: dict[int, int] = field(default_factory=dict)
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
  ```python
  # The for loop handles empty lists fine
  for solution in solutions:
      ...

  # Unnecessary
  if len(solutions) > 0:
      for solution in solutions:
          ...
  ```

- **Exit early when condition is met** instead of nesting:
  ```python
  # Good
  if returncode == 0:
      return result
  # handle error case

  # Avoid deep nesting
  if returncode != 0:
      # long error handling
  else:
      return result
  ```

### Unnecessary Code
- **Don't copy when already correct type**:
  ```python
  # Good
  return dims

  # Unnecessary
  return list(dims)  # dims is already a list!
  ```

- **Don't need local variables for one-time use**:
  ```python
  # Good
  return ir.StringAttr(func_op.name).value

  # Unnecessary
  name_attr = ir.StringAttr(func_op.name)
  return name_attr.value
  ```

- **Use `+=` for appending to lists**:
  ```python
  # Good
  args += extra_args

  # Verbose
  args = args + extra_args
  ```

### Python Patterns
- **Use `match` statement** (Python 3.10+) for multiple conditions:
  ```python
  # Good
  match pipeline:
      case Pipeline.TileAndFuse:
          return handle_tile_and_fuse()
      case Pipeline.VectorDistribute:
          return handle_vector_distribute()

  # Also fine - if/elif chain
  if pipeline == Pipeline.TileAndFuse:
      return handle_tile_and_fuse()
  elif pipeline == Pipeline.VectorDistribute:
      return handle_vector_distribute()
  ```

- **Use `filter()` or list comprehension** for filtering:
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

- **Turn loops into list comprehension** when appropriate

---

## Testing

### Test Quality
- **Tests must exercise actual code**:
  > "This test doesn't exercise any of the tuner code. If you change the tuner code, the test won't catch anything."

- **Test negative cases too** - not just happy path
- **Add tests with real-world data** that matches actual usage patterns
- **Test edge cases** - empty inputs, boundary conditions

### Assertions
- **Compare lists directly** instead of element by element:
  ```python
  # Good
  assert knob_assignments == [None, knob1, knob2, knob3]

  # Unnecessary
  assert len(result) == 4
  assert result[0] is None
  assert result[1] == knob1
  ```

- **Pytest prints expected/actual values** - don't add redundant messages:
  ```python
  # Good - pytest will show the diff
  assert "padding =" in str(lowering_config)

  # Redundant message
  assert "padding =" in str(lowering_config), f"Missing padding: {lowering_config}"
  ```

### Test Organization
- **Don't include usage notes in tests** - the README explains how to run tests
- **Move variables inside scope where used**
- **Make output a function argument** in test helpers for flexibility
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

### Avoiding Mocks
- **Prefer testable functions over mocks**:
  > "Instead of relying on mocks for this test, could we add a function that takes `candidate_results` and decides which candidates to keep?"

- **Make functions take argv as input** for easier testing
- **Extract pure logic** that can be tested without mocking

---

## Architecture and Design

### PR Size
- **Split large PRs** into smaller focused ones:
  > "Can you split it up into a few smaller PRs? Nearly 3 kLOC is a lot to review, even if this is mostly code motion."

- **Separate unrelated changes** into different PRs

### Code Organization
- **Question if code is truly generic or target-specific**
- **Don't leak implementation details** - watch layering
- **Base classes shouldn't know about derived classes**:
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

- **Functions shouldn't know about concrete types** when abstraction is intended
- **Don't add target-specific code to abstract base**

### Imports
- **Combine related imports**:
  ```python
  # Good
  from amdsharktuner import candidate_ordering, common

  # Verbose
  from amdsharktuner import candidate_ordering
  from amdsharktuner import common
  ```

- **Use relative imports consistently**:
  ```python
  from . import common, dispatch_constraints, dispatch_parser
  ```

### Avoid Unnecessary Complexity
- **Don't create helpers for trivial operations**
- **Use constants directly** instead of creating local variables when it hurts readability
- **Consider exposing as bindings** instead of duplicating IREE code
- **Query parent operations properly** - don't assume direct parent is correct type

---

## Debug Code

- **Remove debug prints before merging**:
  ```python
  # Remove before committing
  print(f"matmul_size.K: {matmul_size.K}")
  ```

- **Use `logging.exception()`** instead of `traceback.print_exc()`:
  ```python
  # Good
  logging.exception(f"Error tuning benchmark {benchmark_path}")

  # Avoid
  traceback.print_exc()
  ```

---

## Safety and Robustness

### Input Validation
- **Add checkers for assumptions**:
  ```python
  # Good
  if len(graph_dirs) != 1:
      raise ValueError(f"Expected exactly one graph dir, got {len(graph_dirs)}")
  ```

- **Handle edge cases in argument parsing** (e.g., `-o=foo.mlir` vs `-o foo.mlir`)
- **Validate directory operations** - don't blindly delete user-provided paths

### Error Handling
- **Bail out early** when required conditions aren't met
- **Provide clear error messages** that explain what went wrong

---

## PR Titles and Messages

- **Use descriptive titles** that convey what's changing:
  > "LGTM but consider updating the PR title: 'revisit' does not really convey what's changing, I'd call it something like 'Sync padding for TileAndFuse with IREE changes'"

---

## Comments from Reviewers

### Common Positive Patterns
- "Thanks"
- "Thanks for cleaning this up"
- "LGTM % nit" (LGTM except for a minor issue)
- "+1, especially as we start looking at NN and TN variants"

### Common Issues Caught
- Missing type hints
- Debug prints left in code
- Overly complex logic that could be simplified
- Functions that know too much about their callers/callees
- Tests that don't actually test the code
- Redundant local variables
- Missing docstrings on public functions
