# Code Review Comments

**Repository:** nod-ai/amd-shark-ai

**Generated:** 2026-03-05

---

## PR #2831: [tuner] Fix BOO tuner logging issues

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2831
**State:** MERGED

### Comment by kuhar

**File:** `amdsharktuner/boo_tuner/boo_tuner.py`

**Line:** 372

**Comment:**

Is this something we can test?

---

## PR #2828: [tuner] Refactor compilation info generation into pipeline-specific functions

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2828
**State:** MERGED

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_dispatch_constraints.py`

**Line:** 707

**Comment:**

Can you add type hints to this function?

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_solutions.py`

**Line:** 308

**Comment:**

nit: I think you can use the `match` statement

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_dispatch_constraints.py`

**Line:** 707

**Comment:**

This hasn't been resolved

---

### Comment by bangtianliu

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_dispatch_constraints.py`

**Line:** 707

**Comment:**

Ok, pushed again to the latest commit. 

---

## PR #2817: [Tuner] Add Fusilli tuner

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2817
**State:** OPEN

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 88

**Comment:**

We should make this more concise and explain what argv and return values are inline

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 130

**Comment:**

Can you explain what the default is?

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 136

**Comment:**

Is this for a single candidate or across all candidates?

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 140

**Comment:**

What does it mean that fusilli generates files internally?

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 141

**Comment:**

If you make argv the function argument you will be able to write unit tests for this funciton

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 186

**Comment:**

What if the output filename is specified within the same arg? `-o=foo.mlir`

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 192

**Comment:**

Similar here.

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 197

**Comment:**

nit: += ?

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 215

**Comment:**

It's obvious this is what's happening, but can you explain why we are doing this?

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 218

**Comment:**

same here

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/README.md`

**Line:** 26

**Comment:**

```suggestion
Set up `PYTHONPATH`:
```

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/README.md`

**Line:** 70

**Comment:**

Why did you decide to allow for this to be passed as arguments to fusilli_tuner instead of having it be a named argument like `--fusilli-args="conv -F 1 ..."`? I think the latter is less likely to break as each tool introduces new flags and may eventually have name collisions 

---

### Comment by bangtianliu

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 140

**Comment:**

Sorry, this code comment is kind of confusing and misleading. I will update it.

The placeholder `"fusilli.mlir"` is inserted here just to satisfy `libtuner.parse_arguments()` which expects an `input_file` positional argument.  The actual benchmark MLIR files are generated later (this is what "Fusilli generates files # internally" means). 

---

### Comment by kuhar

**File:** `amdsharktuner/tests/fusilli_tuner_test.py`

**Line:** 49

**Comment:**

Can we also have a testcase with no trailing newline?

---

### Comment by kuhar

**File:** `amdsharktuner/tests/fusilli_tuner_test.py`

**Line:** 97

**Comment:**

Can we have parse_args take sys.argv as input instead?

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 249

**Comment:**

This is somewhat confusing to me -- I'd expect `case_dir` to already be the full location of fusilli cache based on the function docstring

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 350

**Comment:**

This seems like a footgun;  what if someone passes in their home dir as tmp dir?

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 315

**Comment:**

Can we add types across the implementation?

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 214

**Comment:**

Also in this function -- can we add type hints?

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 230

**Comment:**

exit early when the returncode is 0 isntead

---

### Comment by RattataKing

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 287

**Comment:**

Better add a checker `if len(graph_dirs) != 1`

---

### Comment by RattataKing

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 466

**Comment:**

And drop `traceback` import
```suggestion
            logging.exception(f"Error tuning benchmark {benchmark_path}")
```

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 391

**Comment:**

Won't we run into issues with this directory being polluted by previous runs? Later on in `run_fusilli_benchmark_driver`, we do `env["FUSILLI_CACHE_DIR"] = str(cache_dir)` and `find_cached_artifacts` expects exactly one graph directory under `.cache/fusilli/`.

Can you confirm if running the tuner with 2+ commands through the commands file and `--tmp-dir` works?

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/__init__.py`

**Line:** 5

**Comment:**

Don't we want to import something here from fusilli?

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 492

**Comment:**

should we also bail out if neither is set?

---

### Comment by kuhar

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 359

**Comment:**

This is probably not critical

---

### Comment by bangtianliu

**File:** `amdsharktuner/fusilli_tuner/fusilli_tuner.py`

**Line:** 391

**Comment:**

Yes, good catch!. It cause errors according to my local test. 

---

## PR #2815: [Tune] Fix TypeError in shlex.join and fatal abort in batch negative indexing

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2815
**State:** MERGED

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_common.py`

**Line:** 203

**Comment:**

Do we have any tests that covers this?

---

### Comment by bangtianliu

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_common.py`

**Line:** 203

**Comment:**

Yes, https://github.com/nod-ai/amd-shark-ai/blob/7cac743766c9c97a8062162a77de2f19480f1f15/amdsharktuner/tests/rocm/rocm_common_test.py#L111

---

## PR #2809: Pin IREE to >=3.10.0rc20260110 for Tuner

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2809
**State:** CLOSED

### Comment by bangtianliu

**File:** `amdsharktuner/model_tuner/__main__.py`

**Line:** 9

**Comment:**

used to trigger ci, will delete it before landing

---

## PR #2792: [Tuner] Refactor: Extract ROCm-specific candidate ordering heuristics to rocm subdirectory (5/n)

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2792
**State:** MERGED

### Comment by RattataKing

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_candidate_ordering.py`

**Line:** 13

**Comment:**

```suggestion
from amdsharktuner import candidate_ordering, common
```

---

## PR #2790: [Tuner] Refactor: Extract ROCm-specific libtuner, constraint generator tests to rocm subdirectory (4/n)

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2790
**State:** MERGED

### Comment by RattataKing

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_libtuner.py`

**Line:** 10

**Comment:**

The filename `rocm_libtuner.py` may be confusing, as it suggests a rocm specific variant of libtuner.py. It might be clearer to rename it to something like rocm_utils.py.

---

### Comment by bangtianliu

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_libtuner.py`

**Line:** 10

**Comment:**

Yeah, Because it comes from lib_tuner.py originally. Sure, will rename it. 

---

### Comment by bangtianliu

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_libtuner.py`

**Line:** 10

**Comment:**

ok I decide to move to rcom_common.py, which serves the purpose of utilities. 

---

### Comment by RattataKing

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_common.py`

**Line:** 19

**Comment:**

Change `from .. import ` in rocm_*.py to `from amdsharktuner import `. I think relative path imports can be fragile if dir is moved around later.
```suggestion
from amdsharktuner import common, process_utils
```

---

### Comment by bangtianliu

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_common.py`

**Line:** 19

**Comment:**

Thanks for catching this! This is a good reminder about import consistency. I'll send a follow-up PR to address the import  issue fully. 

---

## PR #2775: [tuner] add gitignore patterns for boo tuner artifacts

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2775
**State:** MERGED

### Comment by kuhar

**File:** `amdsharktuner/.gitignore`

**Line:** 3

**Comment:**

```suggestion
# Tuning artifacts.
```

---

## PR #2772: [Tuner] Refactor: Extract rocm-specific tuners and constraint generators to rocm subdirectory (3/3)

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2772
**State:** MERGED

### Comment by RattataKing

**File:** `amdsharktuner/amdsharktuner/candidate_gen.py`

**Line:** 65

**Comment:**

Consider create a mapping dict in common file to make it easy to extend in future, something like:
```
TunersByPipeline = dict[
    iree_codegen.DispatchLoweringPassPipeline,
    list[type[DispatchTuner]],
]
TUNERS: dict[str, TunersByPipeline] = {
    "rocm": {
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUVectorDistribute: [
            rocm_tuners.ROCmContractionVectorDistributeTuner,
            ...,
        ],
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse: [
            ...,
        ],
    },
    # "xxx": {...}  # other backend
}
```
Then in `candidate_gen.py` call api:
```
TUNERS.get(backend, {}).get(codegen_pipeline, [])
```

---

### Comment by RattataKing

**File:** `amdsharktuner/amdsharktuner/libtuner.py`

**Line:** 900

**Comment:**

Function naming and usage here is a bit confusing, maybe rename to:
`get_dispatch_tuners()` -> `get_supported_dispatch_tuners()` 
`set_dispatch_tuner()` -> `instantiate_dispatch_tuner()`

---

### Comment by bangtianliu

**File:** `amdsharktuner/amdsharktuner/candidate_gen.py`

**Line:** 65

**Comment:**

Thanks for the suggestion! After trying, I opted for the simpler function-based approach instead of a registry dict because the backend is already determined before we lookup tuners.

The mapping is still declarative and centralized in `rocm_tuners.get_tuners_for_pipeline()`. When we add SPIR-V support, we'll create a similar `spirv_tuners.get_tuners_for_pipeline()` function and add an elif branch in `get_supported_dispatch_tuners()` to call it.

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_constraint_generators.py`

**Line:** 133

**Comment:**

```suggestion
    ROCm Constraint generator for the IREE LinalgExt AttentionOp.
```

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_constraint_generators.py`

**Line:** 172

**Comment:**

Why does this class have documentation while the other ones above don't?

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_solutions.py`

**Line:** 37

**Comment:**

Can we split this condition and return early instead? The logic inside this if statement is much more complicated than outside, so this will remove a lot of nesting

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_solutions.py`

**Line:** 75

**Comment:**

Do we want to put these in some base class to that users don't have to pass so many arguments? We could put it in a second base class that `ROCmContractionVectorDistributeConstraintGenerator` inherits from

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/candidate_gen.py`

**Line:** 50

**Comment:**

```suggestion
        tune_logger.warning(
            f"Target architecture '{target_arch}' is not tested. "
            f"Tested ROCm architectures: {rocm_common.ROCM_ARCHITECTURES}. "
            f"Proceeding with tuning anyway."
```

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/constraint_generator.py`

**Line:** 215

**Comment:**

```suggestion
    for a specific type of a tunable problem.
```

---

### Comment by bangtianliu

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_constraint_generators.py`

**Line:** 172

**Comment:**

The context is that I previously added attention support to the tuner, and I was asked to add some documentation at that time (see discussion here: https://github.com/nod-ai/amd-shark-ai/pull/1772#discussion_r2201498789). 

This PR only does code movement for this part. But sure, I will add documentation for the other ones. 

---

### Comment by bangtianliu

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_solutions.py`

**Line:** 75

**Comment:**

Yes, I kept this in mind: https://github.com/nod-ai/amd-shark-ai/blob/33d05779449a40aa2532927c438148b5ebf45238/amdsharktuner/amdsharktuner/constraint_generator.py#L906-L915

I can solve this in one separate PR, once refactor and IGEMM PR are landed. 

---

## PR #2771: [Tuner] Refactor: Extract rocm_common.py from common.py (2/3)

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2771
**State:** MERGED

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_common.py`

**Line:** 121

**Comment:**

Isn't this code generic?

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_common.py`

**Line:** 121

**Comment:**

Ah no, it uses iree_gpu lowering config, so it may be gpu-generic at best.

---

## PR #2769: [Tuner] Refactor: Move dispatch_constraints to rocm subdirectory (1/3)

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2769
**State:** MERGED

### Comment by kuhar

**File:** `amdsharktuner/tests/rocm/rocm_dispatch_constraints_test.py`

**Line:** 9

**Comment:**

We don't need this usage notes in tests -- all tests are supposed to be executed like this and the README explains it. We can drop it.

---

## PR #2747: [tuner] use virtual mma only for attention op

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2747
**State:** MERGED

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/dispatch_constraints.py`

**Line:** 88

**Comment:**

```suggestion
        compatible_intrinsics = filter(lambda x: isinstance(x, iree_gpu.MMAIntrinsic), compatible_intrinsics)
```

---

### Comment by kuhar

**File:** `amdsharktuner/tests/dispatch_constraints_test.py`

**Line:** 426

**Comment:**

Do we have any tests that exercises `allow_virtual_mma=False`? IE, does anything fail if you always pass `allow_virtual_mma=False`?

---

### Comment by bangtianliu

**File:** `amdsharktuner/tests/dispatch_constraints_test.py`

**Line:** 426

**Comment:**

Yes, we do have tests for `allow_virtual_mma=False` : [lines 162-164 in common_test.py]( https://github.com/nod-ai/amd-shark-ai/pull/2747/files#diff-d7787f1930ca7cf2cc378b576be218f3ca06cf1da05c910d4fbd544a6d6863b8R154-R160)

> If you always pass `allow_virtual_mma=False`:

- The tests at [lines 162-164 in common_test.py](https://github.com/nod-ai/amd-shark-ai/pull/2747/files#diff-d7787f1930ca7cf2cc378b576be218f3ca06cf1da05c910d4fbd544a6d6863b8R162-R164) would fail (expects virtual intrinsic, gets []).
- The tests at [lines 415-437 in dispatch_constraints_test.py](https://github.com/nod-ai/amd-shark-ai/pull/2747/files#diff-e79281b29c8d99e1c8ebd3f6bf96101582615c60b3b7c4ed74e76863e39c40a6R415-R437) would fail.

---

## PR #2744: [tuner] ensure correct number of TD specs are generated

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2744
**State:** MERGED

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/libtuner.py`

**Line:** 958

**Comment:**

I don't think we need a helper function for this -- this can be a s simple as prepending a `None`.

---

### Comment by kuhar

**File:** `amdsharktuner/tests/libtuner_test.py`

**Line:** 450

**Comment:**

You can compare lists

---

### Comment by kuhar

**File:** `amdsharktuner/tests/libtuner_test.py`

**Line:** 454

**Comment:**

also here

---

### Comment by bangtianliu

**File:** `amdsharktuner/amdsharktuner/libtuner.py`

**Line:** 958

**Comment:**

The helper function was originally added to make testing easier, but I agree it can be simplified to just inline code.

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/libtuner.py`

**Line:** 943

**Comment:**

Can you add a one-line comment explaining the None element is for the baseline?

---

### Comment by kuhar

**File:** `amdsharktuner/tests/libtuner_test.py`

**Line:** 452

**Comment:**

This doesn't test anything

---

### Comment by bangtianliu

**File:** `amdsharktuner/tests/libtuner_test.py`

**Line:** 452

**Comment:**

In this test, I just replicated what I did in the tuner. This test can be removed.

---

### Comment by kuhar

**File:** `amdsharktuner/tests/libtuner_test.py`

**Line:** 452

**Comment:**

This test doesn't exercise any of the tuner code. If you change the tuner code, the test won't catch anything.

---

## PR #2736: [tuner] enable igemm support for all conv layouts

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2736
**State:** MERGED

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/candidate_gen.py`

**Line:** 230

**Comment:**

This function shouldn't know about concrete tuner classes. Instead, could we have a property for the conv strategy and set it externally?

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/dispatch_parser.py`

**Line:** 198

**Comment:**

This doesn't belong in the abstract base -- not all dispatch parsers even know what a convolution is. I think the key design requirement should be that, in the future, we can tune across multiple conv lowering strategies within the same tuner context. This suggests to me that this should be stored somewhere outside of the parser.

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/dispatch_parser.py`

**Line:** 198

**Comment:**

Or alternatively, maybe we want to have two conv dispatch parsers, and allow for multiple parsers to matcha a single root op?

---

### Comment by bangtianliu

**File:** `amdsharktuner/amdsharktuner/dispatch_parser.py`

**Line:** 198

**Comment:**

Thanks for the suggestions. 

I've refactored to use two separate conv dispatch parsers:
- IGEMMConvolutionParser / IGEMMConvolutionTuner for IGEMM lowering
- InnerMNKConvolutionParser / InnerMNKConvolutionTuner for INNER_MNK lowering


---

### Comment by RattataKing

**File:** `amdsharktuner/amdsharktuner/libtuner.py`

**Line:** 913

**Comment:**

To me `candidate_gen.set_dispatch_tuner()` is a generic function, and passing `args.codegen_pipeline` instead of `conv_lowering_strategy` avoids making it look convolution specific.

Consider move:
`get_conv_lowering_strategy_for_pipeline()` from `libtuner.py` -> `candidate_gen.py`, 
`class CodegenPipelines` from `libtuner.py` -> `common.py`

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/candidate_gen.py`

**Line:** 138

**Comment:**

The base class shouldn't know about the derived classes -- can we make this a free function instead?

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/candidate_gen.py`

**Line:** 265

**Comment:**

I'm concerned we are leaking codegen pipeline and conv strategies to this generic code. The layering seems off to me.

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/rocm/rocm_common.py`

**Line:** 41

**Comment:**

```suggestion
    conv_to_igemm_dim: dict[int, int] = field(default_factory=dict)
```

---

## PR #2713: [tuner]: sync the change of using prefetch_num_stages to replace prefetch_shared_memory

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2713
**State:** MERGED

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/libtuner.py`

**Line:** 378

**Comment:**

Why `3 = new option`? Can you describe what 3+ means instead?

---

## PR #2701: [tuner] add padding_conv attribute along IGEMM supprot for conv 

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2701
**State:** MERGED

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/common.py`

**Line:** 255

**Comment:**

Can you add some example?

---

### Comment by kuhar

**File:** `amdsharktuner/tests/common_test.py`

**Line:** 578

**Comment:**

```suggestion
        assert common.is_affine_expr_function_of_dim(d0, 0)
        assert not common.is_affine_expr_function_of_dim(d0, 1)
```
also below

---

### Comment by kuhar

**File:** `amdsharktuner/tests/common_test.py`

**Line:** 598

**Comment:**

Can you add a comment explaining why we need this?

---

### Comment by kuhar

**File:** `amdsharktuner/tests/constraint_generator_test.py`

**Line:** 301

**Comment:**

Doesn't pytest print expected and actual values?

---

### Comment by kuhar

**File:** `amdsharktuner/tests/dispatch_parser_test.py`

**Line:** 396

**Comment:**

```suggestion
        builtin.module {
```

---

### Comment by kuhar

**File:** `amdsharktuner/tests/dispatch_parser_test.py`

**Line:** 400

**Comment:**

You can make the output be another function argument

---

### Comment by bangtianliu

**File:** `amdsharktuner/tests/constraint_generator_test.py`

**Line:** 301

**Comment:**

Thanks for the catching! 

The change followed the existing test style like `padding` blindly.  but you're right and I should clean up the redundant error messages. Pytest's assertion can shows the actual values when assertions fail (I checked it locally). 

---

### Comment by Max191

**File:** `amdsharktuner/amdsharktuner/common.py`

**Line:** 673

**Comment:**

The filter loop dimensions often get collapsed with the input channel dimensions when converting to IGEMM, so the outer `if` here will be executed for the filter loop dimensions as well. IIUC, this could cause the `igemm_pos` for the reduction dim to be added during the loop iteration for the filter loop dims, and then this will return `None` when it gets to the input channel dim because it thinks it already found an input channel dim.

For example, the conv_to_igemm_map could have the following:
```
conv_to_igemm_map = {
    ...
    5 : 3, // this is filterLoop0 -> K
    6 : 3, // this is filterLoop1 -> K
    7 : 3 // this is inputChannel -> K
}
```
`3` could be added to padded_igemm_dims when looking at `5 : 3`, and then it will be skipped when looking at `7 : 3`, even though `7` is the only inputChannel dim.

I think you can fix this by instead checking for reduction dimensions that are not inputChannel dimensions, and skipping them before anything else.

---

### Comment by Max191

**File:** `amdsharktuner/amdsharktuner/common.py`

**Line:** 659

**Comment:**

nit: If this is just meant to be checking for reduction dimensions, then can you instead pass in the iterator types for the igemm, instead of using the tile sizes. I know we don't generate it in the tuner, but it is valid for reduction dimensions to have a tile size of 0, so it would make more sense to me to be using the actual iterator types instead of the tile sizes for this.

---

### Comment by Max191

**File:** `amdsharktuner/amdsharktuner/common.py`

**Line:** 694

**Comment:**

Shouldn't this be using the `padding_sizes` to check whether or not the dimension is padded? Also, this is the only use of `workgroup_tile_sizes`, so if you make this change, then you can remove the function parameter.

---

### Comment by bangtianliu

**File:** `amdsharktuner/amdsharktuner/common.py`

**Line:** 673

**Comment:**

Yeah, I agree.  your suggestion makes the logic clearer and more robust.

The tuner-side implementation is essentially mirroring the behavior in IREE.
Does this mean the corresponding logic in IREE: https://github.com/iree-org/iree/blob/358543624e1132b31f2cbf51a4be70d19152af07/compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.cpp#L476-L509 should also be updated in the same way? I just want to ensure that everything works correctly.

---

### Comment by Max191

**File:** `amdsharktuner/tests/common_test.py`

**Line:** 598

**Comment:**

Could you add one test here that more closely matches a real convolution in its ConvToIgemmInfo? You could have one that matches the result from the `test_build_conv_to_igemm_info` test below, for example.

---

## PR #2698: [tuner] update the calculation of shared memory usage

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2698
**State:** MERGED

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/dispatch_constraints.py`

**Line:** 174

**Comment:**

This formatting is really weird, isn't there any way to keep the second array on a single line?
For example, you could do something like:

```py
supported_promotions = ([0, 1], [0, 1, 2])
assert promote_operands in supported_promotions
```

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/dispatch_constraints.py`

**Line:** 192

**Comment:**

Can we replace this trick with a few if statements?

---

### Comment by Max191

**File:** `amdsharktuner/amdsharktuner/dispatch_constraints.py`

**Line:** 171

**Comment:**

Any reason why we don't support arbitrary promotion types? In general it should be more likely that promoting both LHS and RHS would be better, but it could be interesting to see if any shapes benefit from promoting only one of the operands.

Perhaps it can be done as a follow up, since it isn't really related to this PR, but I think it could be useful to experiment with tuning over different operand promotion lists. Then you can remove the assert in the followup PR.

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/dispatch_constraints.py`

**Line:** 171

**Comment:**

+1, especially as we start looking at NN and TN variants

---

### Comment by bangtianliu

**File:** `amdsharktuner/amdsharktuner/dispatch_constraints.py`

**Line:** 171

**Comment:**

I'm keeping the assert here to be conservative for now. Currently, the tuner only manually sets promote_operands to either [0, 1] or [0, 1, 2], and we don't yet support exploring other promotion patterns during tuning.

I'll remove the assert in a follow-up when we either add promote_operands to the tuning search space or expand support for other operations (like NN and TN variants as Jakub mentioned above). 



---

## PR #2692: [tuner] Sync Padding for TileAndFuse with IREE changes

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2692
**State:** MERGED

### Comment by bangtianliu

**File:** `amdsharktuner/amdsharktuner/constraint_generator.py`

**Line:** 263

**Comment:**

https://github.com/nod-ai/amd-shark-ai/blob/9298ae5d1a94b624e144d0fe3d686d7e853589ed/amdsharktuner/amdsharktuner/dispatch_constraints.py#L161-L174

Spent some time investigating the failed specs, and most issues are related to shared memory usage. The current constraint for estimating shared memory assumes only operands 0 and 1 are promoted, which isn’t always correct. I’ll address this in a separate PR.

---

### Comment by Max191

**File:** `amdsharktuner/amdsharktuner/constraint_generator.py`

**Line:** 263

**Comment:**

This should actually be supported now, so you may be able to just get rid of the promotion for operand 2. Doing it in this PR or as a follow up would both be fine to me.

---

### Comment by Max191

**File:** `amdsharktuner/amdsharktuner/common.py`

**Line:** 556

**Comment:**

FYI, this logic is to allow over-padding to get larger tile sizes, which may result in better performance despite doing more padded computation. We may want to try making this overpadding a tunable parameter eventually. Can you add a TODO comment?

I wouldn't do it right now because it may blow up the tuning space, but we should be getting better candidate ordering soon, which will help with that, so the TODO comment is good for now.

---

### Comment by kuhar

**File:** `amdsharktuner/tests/constraint_generator_test.py`

**Line:** 357

**Comment:**

why are you guarding a for loop with an if condition?

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/common.py`

**Line:** 575

**Comment:**

Can you explain what is returned?

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/common.py`

**Line:** 540

**Comment:**

Also here: what does this do?

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/common.py`

**Line:** 530

**Comment:**

also here -- could you explain what is being returned?

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/common.py`

**Line:** 530

**Comment:**

we should start function names with active verbs

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/common.py`

**Line:** 547

**Comment:**

Why not hoist this check outside of the loop?

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/common.py`

**Line:** 552

**Comment:**

dims is already a list

---

### Comment by Max191

**File:** `amdsharktuner/amdsharktuner/constraint_generator.py`

**Line:** 243

**Comment:**

I'm not sure it's okay to just delete this. The `calculate_padded_dimensions` logic is only for overpadding, and also only for the M and N dims. Shapes with small M/N dims that are not aligned to the intrinsic sizes would still require padding, but I don't think they will be caught by `calculate_padded_dimensions`. I think the if below should be checking for `required_padding or padding_applied` (and small nit: maybe rename `padding_applied` to `overpadding_applied`).

It's also worth testing the tuner on a smaller shape that has M/N unaligned to the intrinsic, but less than 32 (i.e., `MxNxK = 30x30x30`. Maybe some of the transposed cases as well, since they will not do overpadding, but they should still pad to intrinsics.

---

### Comment by bangtianliu

**File:** `amdsharktuner/amdsharktuner/constraint_generator.py`

**Line:** 243

**Comment:**

The current changes in this PR follow the existing logic in KernelConfig.

I did notice that small-dimension cases aren’t producing any solutions. If we want to support those including the transposed and small case, I can update the logic and add corresponding tests.

---

## PR #2683: [tuner] use igemm bindings

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2683
**State:** MERGED

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/constraint_generator.py`

**Line:** 59

**Comment:**

Drop debug prints

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/dispatch_parser.py`

**Line:** 76

**Comment:**

What is the type? Using `Any` effectively sidesteps any type checking

---

### Comment by kuhar

**File:** `amdsharktuner/amdsharktuner/dispatch_parser.py`

**Line:** 273

**Comment:**

```suggestion
        # Get IGEMM details for potential use with the TileAndFuse pipeline.
```

---

### Comment by kuhar

**File:** `amdsharktuner/tests/constraint_generator_test.py`

**Line:** 396

**Comment:**

Can we use this constant directly instead of creating a local variable for it? I think it only hurts readability here

---

### Comment by kuhar

**File:** `amdsharktuner/tests/constraint_generator_test.py`

**Line:** 460

**Comment:**

Can we move these inside the `with` statement, since they are not used anywhere else?

---

### Comment by kuhar

**File:** `amdsharktuner/tests/constraint_generator_test.py`

**Line:** 493

**Comment:**

Would it help to assert that what the returned dimensions and sizes are? I think it won't ever change, so we don't have to worry about these getting out of sync with the compiler.

---

### Comment by kuhar

**File:** `amdsharktuner/tests/constraint_generator_test.py`

**Line:** 461

**Comment:**

I'd inline these shapes here, if you are not going to use these variables anywhere else

---

## PR #2641: [tuner] Create dedicated IREE requirements file for sharktuner

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2641
**State:** MERGED

### Comment by ScottTodd

**File:** `requirements-iree-unpinned.txt`

**Line:** 2

**Comment:**

Remove from here too? https://github.com/nod-ai/shark-ai/blob/8e9e3c4131d4bca0421d4867250d0fdcba33c4ed/requirements-iree-pinned.txt#L1-L2

These should be kept in sync.

---

### Comment by kuhar

**File:** `requirements-iree-unpinned.txt`

**Line:** 2

**Comment:**

I think we may want to have our own requirements for sharktuner independent of what the rest of shark-ai uses

---

### Comment by bangtianliu

**File:** `requirements-iree-unpinned.txt`

**Line:** 2

**Comment:**

> Remove from here too?
> 
> https://github.com/nod-ai/shark-ai/blob/8e9e3c4131d4bca0421d4867250d0fdcba33c4ed/requirements-iree-pinned.txt#L1-L2
> 
> These should be kept in sync.

But the rest of shark-ai may be dependent on wave.

---

### Comment by ScottTodd

**File:** `requirements-iree-unpinned.txt`

**Line:** 2

**Comment:**

If other projects depend on it, it should stay here.

> motivated by CI error: https://github.com/nod-ai/shark-ai/actions/runs/19075325301/job/54489756809?pr=2555.

I don't see any reference to "wave" in those logs. Is this even the right fix for that?

---

### Comment by bangtianliu

**File:** `requirements-iree-unpinned.txt`

**Line:** 2

**Comment:**

> I don't see any reference to "wave" in those logs. Is this even the right fix for that?

here: https://github.com/nod-ai/shark-ai/actions/runs/19075325301/job/54489756809?pr=2555#step:6:1

it mainly about iree 3.8 is installed after installing wave (override iree 3.9 installed at the beginning) and we need iree 3.9. 

FYI, After dropping wave, we have this  : https://github.com/nod-ai/shark-ai/actions/runs/19075707245/job/54490540104?pr=2627.

---

### Comment by ScottTodd

**File:** `requirements-iree-unpinned.txt`

**Line:** 2

**Comment:**

> it mainly about iree 3.8 is installed after installing wave (override iree 3.9 installed at the beginning) and we need iree 3.9.

So broken by https://github.com/iree-org/wave/pull/395 maybe? cc @xintin @Hardcode84 @harsh-nod

---

### Comment by xintin

**File:** `requirements-iree-unpinned.txt`

**Line:** 2

**Comment:**

> > it mainly about iree 3.8 is installed after installing wave (override iree 3.9 installed at the beginning) and we need iree 3.9.
> 
> So broken by [iree-org/wave#395](https://github.com/iree-org/wave/pull/395) maybe? cc @xintin @Hardcode84 @harsh-nod

As part of https://github.com/iree-org/wave/pull/395, we ensured that `wave-lang` stable pip pkg release depends on right version of `iree-base-compiler` and `runtime`. It is needed for SGLang. Else, let's say if iree 3.9.0 releases before wave-lang 3.9.0, then wave-lang will pick the latest stable release. This already broke sglang ci twice. 
After this change, now if someone does `pip install wave-lang`, it will pick the latest stable version of iree-base-c/r mentioned in the metadata of wave-lang. 
Otherwise, if nightly are required, one can build wave-lang from source.

---

### Comment by ScottTodd

**File:** `requirements-iree-unpinned.txt`

**Line:** 2

**Comment:**

Version management docs (which wave hasn't integrated with):
* https://iree.dev/developers/general/release-management/
* https://iree.dev/developers/general/versioning-scheme/

---

### Comment by xintin

**File:** `requirements-iree-unpinned.txt`

**Line:** 2

**Comment:**

We can discuss this in a separate channel, but can you suggest what needs to be done here to integrate wave into the version management docs? 

---

### Comment by bangtianliu

**File:** `requirements-iree-unpinned.txt`

**Line:** 2

**Comment:**

Ok, then what should we do for this PR? 
We could either remove wave from the unpinned IREE requirements (current approach) or create a separate requirements file specifically for the tuner. cc @ScottTodd @kuhar


---

### Comment by kuhar

**File:** `requirements-iree-unpinned.txt`

**Line:** 2

**Comment:**

https://github.com/nod-ai/shark-ai/pull/2641#discussion_r2491522614

---

### Comment by ScottTodd

**File:** `requirements-iree-unpinned.txt`

**Line:** 2

**Comment:**

Are other sharkai workflows (for developers and CI) also installing the old/stable 3.8.0 version of IREE now instead of unpinned 3.9.0rc versions too?

---

### Comment by bangtianliu

**File:** `requirements-iree-unpinned.txt`

**Line:** 2

**Comment:**

Yes, FYI: https://github.com/nod-ai/shark-ai/actions/runs/19077368699/job/54496619284?pr=2641#step:5:73

---

### Comment by kuhar

**File:** `requirements-iree-sharktuner.txt`

**Line:** 1

**Comment:**

Should we move it inside the shartuner dir?

---

### Comment by kuhar

**File:** `requirements-iree-sharktuner.txt`

**Line:** 1

**Comment:**

Can we delete this file now?

---

### Comment by bangtianliu

**File:** `requirements-iree-sharktuner.txt`

**Line:** 1

**Comment:**

yeah,  forgot to git add this one

---

## PR #2627: [tuner] exits gracefully for unsupported cases like mat-vec operations

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2627
**State:** MERGED

### Comment by kuhar

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 110

**Comment:**

This warning makes it sounds like something is wrong with the input. Instead, we should print that this contraction type is not supported by the tuner yet.

---

### Comment by kuhar

**File:** `sharktuner/tests/candidate_gen_test.py`

**Line:** 234

**Comment:**

Can we use named linalg ops like `linalg.matmul` or `linalg.matvec` to simplify this?

---

### Comment by kuhar

**File:** `sharktuner/tests/candidate_gen_test.py`

**Line:** 234

**Comment:**

Also in the other tests.

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 258

**Comment:**

```suggestion
            "The operation may not be supported by the tuner yet."
```

---

### Comment by kuhar

**File:** `sharktuner/tests/candidate_gen_test.py`

**Line:** 226

**Comment:**

I would make this more concise:

```suggestion
    # Make sure we do not crash on unsupported root ops (matvec).
```

---

### Comment by kuhar

**File:** `sharktuner/tests/candidate_gen_test.py`

**Line:** 255

**Comment:**

similar here

---

## PR #2621: [tuner][nfc]: merge imports and update the readme

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2621
**State:** MERGED

### Comment by kuhar

**File:** `sharktuner/tests/libtuner_test.py`

**Line:** 283

**Comment:**

Can we also remove this non-ascii character?

---

### Comment by kuhar

**File:** `sharktuner/README.md`

**Line:** 25

**Comment:**

We should rename the file instead -- the convention is to use `requirements-test.txt`

---

## PR #2613: [tuner] set prefetch shared memory option based on layout matching for attention ops

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2613
**State:** MERGED

### Comment by Groverkss

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 304

**Comment:**

This should be only turned on if the intrinsics have "matching" layout. I remember you implemented that right? You can just make this a constraint instead of searching it on True/False. This can be True if the output layout of intrinsicA matches the lhs or rhs layout of intrinsicB

---

## PR #2596: [tuner] use python binding to build td specs for attention

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2596
**State:** MERGED

### Comment by kuhar

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 576

**Comment:**

Should we move this comment down to `class AttentionOpInfo(OpInfo):`?

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 110

**Comment:**

Can we put them inline next to / on top of each property? This way it should be easier to keep it up to date and make sure the documentation matches the code. Probably also easier to read.

---

## PR #2592: [tuner] support phase-aware candidate pruning

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2592
**State:** MERGED

### Comment by kuhar

**File:** `sharktuner/model_tuner/model_tuner.py`

**Line:** 169

**Comment:**

```suggestion
                "No tuning specs to return: no dispatch candidate outperformed the baseline."
```

---

### Comment by Max191

**File:** `sharktuner/sharktuner/libtuner.py`

**Line:** 156

**Comment:**

I think it is better to make this function name more directly linked with its purpose (i.e., to prune the top_candidates list if there are no candidates below the baseline). Maybe something like `prune_slower_than_baseline`?

---

### Comment by Max191

**File:** `sharktuner/model_tuner/model_tuner.py`

**Line:** 52

**Comment:**

nit: IMO it's useful to explain why we want to keep slower candidates for the model phase. Maybe include that in the comment? (i.e., because slower dispatch candidates could still give improvement when run in the full model due to different fusions and/or concurrent dispatches).

---

### Comment by Max191

**File:** `sharktuner/sharktuner/libtuner.py`

**Line:** 164

**Comment:**

nit: This suggestion may be slightly misleading. I think the distinction is not about multi-phase vs single phase tuners, but more about whether slower candidates from the first phase could still show improvement in the later phases. Perhaps you can phrase this suggested implementation in this way instead.

---

### Comment by Max191

**File:** `sharktuner/sharktuner/libtuner.py`

**Line:** 160

**Comment:**

nit: When this says `returns empty list` and `returns top N candidates`, it's not clear what is returning these lists. Could you make it a bit more specific about what is returning the list (i.e., the `benchmark` function)?

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/libtuner.py`

**Line:** 1269

**Comment:**

Instead of relying on mocks for this test, could we add a function that takes `candidate_results` and the `baseline_handler` and decides which candidates to keep and which to drop?

I'd like to avoid mocks for things that can executed using the intended logic. The issue with mocks is that they require you to assume what a valid implementation may do and what its state is without exercising the surrounding logic, hence my preference is towards solutions that reduce the amount of state and don't have to mock. Some related articles that talks about this:
* https://the-dext.github.io/TDD_Mocking-vs-No-Mocking/
* https://www.robertopiva.pro/2017/02/15/the-mock-excuse.html

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/libtuner.py`

**Line:** 1219

**Comment:**

Have you considered passing a second parameter to this function, `should_prune`? I'm wondering if we need a new function at all

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/libtuner.py`

**Line:** 1222

**Comment:**

Usually there's no need for local variables only used once
```suggestion
    return all_candidates_with_speedup[:num_candidates]
```

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/libtuner.py`

**Line:** 1219

**Comment:**

Good suggestion! I created the separate function to address your earlier feedback about avoiding mocks, but didn't consider reusing the existing function. Adding the pruning parameters directly to `get_candidates_ordered_by_speedup()` seems to be to a better approach.


---

### Comment by kuhar

**File:** `sharktuner/sharktuner/libtuner.py`

**Line:** 1085

**Comment:**

We should update the docstring below and explain what this argument is responsible for

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/libtuner.py`

**Line:** 1085

**Comment:**

I would also call it something more descriptive like `prune_slow_candidates`

---

## PR #2543: [tuner] use python binding to build td specs for contraction

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2543
**State:** MERGED

### Comment by kuhar

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 21

**Comment:**

```suggestion
from iree.compiler.dialects import iree_codegen, iree_gpu, linalg  # type: ignore
```

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 139

**Comment:**

When can this fail?

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 18

**Comment:**

```suggestion
from . import common, dispatch_constraints, dispatch_parser
```

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 492

**Comment:**

```suggestion
        # TODO(Bangtian): Both root_op and op_info are kept as a temporary solution.
```

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 36

**Comment:**

Can we remove this member and query the nearest parent of type funcopinterface instead?

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 59

**Comment:**

Why don't we store the op in op info and drop the parent function name?

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 16

**Comment:**

also here, can we combine these?

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 139

**Comment:**

`linalg.isa_convolution_op` goes to `mlir::linalg::detail::isConvolutionInterfaceImpl` in low level, 
And `linalg.infer_convolution_dimensions` goes to `inferConvolutionDimsImpl` in low level.

isConvolutionInterfaceImpl does contain below code block:
https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Linalg/IR/LinalgInterfaces.cpp#L1006-L1012
```C++
if (dimensions) {
    FailureOr<ConvolutionDimensions> res = inferConvolutionDimsImpl(
        linalgOp, inputExprWalker, allowEmptyConvolvedDims);
    assert(succeeded(res) && "unexpected failure to infer convolution dims");
    *dimensions = *res;
  }
```
However, When `linalg.isa_convolution_op` is called, it passes dimensions=nullptr, so this block is not executed since dimension is nullptr. 

So from my understanding,` linalg.infer_convolution_dimensions` could fail even after `linalg.isa_convolution_op` succeeds, because it's the first time the actual dimension inference logic runs.


---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 59

**Comment:**

Yes, we can. Previously I just worried storing root_op into op_info was too heavy. 

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 139

**Comment:**

Right, it may fail when the output image is empty: https://github.com/llvm/llvm-project/blob/df2ff3a1b2c231f8ec78c244950687cdc54b507b/mlir/lib/Dialect/Linalg/IR/LinalgInterfaces.cpp#L784-L785.

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 48

**Comment:**

the `match_` prefix can be added by spec builders -- the op doesn't live in a named sequence and doesn't have to know about it

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 46

**Comment:**

Can we use function interface like we do in the compiler? For example, this could also be `util.func`

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 84

**Comment:**

Can you make this a helper function that doesn't live in any specific class?  I don't see why this has to be duplicated across dispatch parser and op_info

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 164

**Comment:**

```suggestion
            config_params.append(ransform.ParamConstantOp(
                transform.AnyParamType.get(),
                config.configuration
            ).result)
```

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 46

**Comment:**

I couldn't find a decent way to do it. Maybe just manually check both? maybe later we can expose the python binding for `isa<FunctionOpInterface>`.   

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 46

**Comment:**

ok fine to leave func.func for the time being

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 167

**Comment:**

actually you can turn this whole loop into list comprehension

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 27

**Comment:**

nit: We don't really need local variables for stuff that's only being used once 
```suggestion
    return ir.StringAttr(func_op.name).value
```

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 26

**Comment:**

How do you know the nearest parent is a function? The root op could be inside an `scr.if`, for example.  In C++ we'd check this with `op->getParentOfType<FunctionOpInterface>()`

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 26

**Comment:**

This part of the code wasn’t originally added in this PR and I just moved it here. But thanks for pointing it out. I can update it to make the check more robust.

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 26

**Comment:**

We already have an assertion, so I'd leave it as `# FIXME: ...` if we can't easily query this from python

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 244

**Comment:**

I'm not sure we need this comment -- it seems pretty clear what the code is doing

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 27

**Comment:**

This is not a function name though, is it? A matcher is a named sequence.

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 84

**Comment:**

```suggestion
        operand_name = operand.get_name()
```

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 196

**Comment:**

I would only add the `match_` prefix here, not in the helper

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 196

**Comment:**

Adding the `match_` prefix is currently used multiple times in both spec_builder.py and candidate_gen.py.

---

## PR #2527: [tuner][nfc] clean up the code

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2527
**State:** MERGED

### Comment by kuhar

**File:** `sharktuner/setup.py`

**Line:** 10

**Comment:**

Could we add the typing info for these to our requirements? https://pypi.org/project/types-setuptools/

---

### Comment by bangtianliu

**File:** `sharktuner/setup.py`

**Line:** 10

**Comment:**

My bad, it exists here: https://github.com/nod-ai/shark-ai/blob/b65fdae35ede615932734d6c204771a33cfdc7ec/sharktuner/requirements-dev.txt#L4

I just forgot to install it after rebase. 

---

## PR #2515: [tuner] use python bindings to build td specs

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2515
**State:** CLOSED

### Comment by kuhar

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 105

**Comment:**

```suggestion
        op_info = self.get_op_info()
        builder = spec_builder.ConvolutionSpecBuilder(op_info)
```

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 125

**Comment:**

```suggestion
        op_info = self.get_op_info()
        builder = spec_builder.AttentionSpecBuilder(opinfo)
```

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 105

**Comment:**

Can you also annotate op_info with a type?

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 492

**Comment:**

what's the type?
```suggestion
        self.op_info = op_info
```

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 35

**Comment:**

What does func_name refer to?

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 41

**Comment:**

Why does op info need context?

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 25

**Comment:**

```suggestion
    def __init__(self, op_info: OpInfo):
        self.op_info = op_info
```

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 31

**Comment:**

I don't understand this comment

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 43

**Comment:**

What's the benefit of building the base spec programmatically? If it's going to be the same for all inputs, having hardocded mlir sounds OK to me and is much more concise.

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 109

**Comment:**

maybe create a helper for getting these readonly attributes?

---

### Comment by kuhar

**File:** `sharktuner/setup.py`

**Line:** 10

**Comment:**

Can you move unrelated changes to a separate PR?

---

### Comment by kuhar

**File:** `sharktuner/tests/spec_builder_test.py`

**Line:** 268

**Comment:**

Why has the name changed?

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 31

**Comment:**

It is referring to 
```mlir
    %0 = transform.param.constant #iree_codegen.compilation_info<xx
```
I will rewrite the comment to make it clear.

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 43

**Comment:**

Yeah, make sense. I will switch back to textual string for baseline td spec.

---

### Comment by bangtianliu

**File:** `sharktuner/setup.py`

**Line:** 10

**Comment:**

NFC PR is sent https://github.com/nod-ai/shark-ai/pull/2527 

---

### Comment by bangtianliu

**File:** `sharktuner/tests/spec_builder_test.py`

**Line:** 268

**Comment:**

In original code:

https://github.com/nod-ai/shark-ai/blob/3fac73c1e335bba12f67e2b5ccfeb1a2b798856f/sharktuner/tests/spec_builder_test.py#L251-L265

the matcher name was manually set to 'match_batch_matmul'.

In this PR:
```python
        self._func_name = f"match_{ir.StringAttr(func_name_attr).value}"
```
because the funct_name is `bach_match_func` in the test, so it automatically become `match_batch_matmul_func`.
        

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 41

**Comment:**

This op info is used by the spec builder to create the td spec via Python bindings, so a context is required.

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 35

**Comment:**

`func_name` is the name of the matcher named sequence that will be generated for this operation.

It's derived from the function operation that contains the root operation being tuned:
- Extracted from the parent `func.FuncOp` of the root operation
- Prefixed with "match_" to create the matcher sequence name

For example, if the input mlir contains:
```mlir
func.func @batch_matmul_func(...) {
  // root operation here
}
```

Then `func_name` would be `"match_batch_matmul_func"`.

---

### Comment by bangtianliu

**File:** `sharktuner/tests/spec_builder_test.py`

**Line:** 268

**Comment:**

Just to clarify, the dispatch parser code already exists here:
https://github.com/nod-ai/shark-ai/blob/3fac73c1e335bba12f67e2b5ccfeb1a2b798856f/sharktuner/sharktuner/dispatch_parser.py#L31-L39

This PR extends the existing `OpInfo` dataclasses to include the additional metadata (lhs/rhs/res types, indexing maps, dimension sizes) required by the new matcher operations. 


---

### Comment by kuhar

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 35

**Comment:**

Maybe call it `parent_function_name` instead? Or just keep the function op as a field and query its name. I thought that this was the name of the op itself.

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 41

**Comment:**

Can we pass context as a parameter to `build_td_spec` instead?

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 31

**Comment:**

Maybe `Creates a constant parameter with #iree_codegen.compilation_info for each configuration`?

---

## PR #2482: [tuner] add pypi readme

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2482
**State:** MERGED

### Comment by kuhar

**File:** `sharktuner/PYPI_README.md`

**Line:** 9

**Comment:**

````suggestion
```

This will install all required dependencies including IREE compiler and runtime.
````

---

### Comment by kuhar

**File:** `sharktuner/PYPI_README.md`

**Line:** 11

**Comment:**

```suggestion
You can use the latest nightly IREE python bindings:
```

---

### Comment by kuhar

**File:** `sharktuner/PYPI_README.md`

**Line:** 25

**Comment:**

````suggestion
```

or
````

---

### Comment by kuhar

**File:** `sharktuner/PYPI_README.md`

**Line:** 29

**Comment:**

Use `###` for third-level headings 

---

### Comment by kuhar

**File:** `sharktuner/PYPI_README.md`

**Line:** 31

**Comment:**

why is this `bash` while the previous snippets used `shell`?

---

### Comment by kuhar

**File:** `sharktuner/PYPI_README.md`

**Line:** 43

**Comment:**

````suggestion
```

Refer to [Mode Tuner README](https://github.com/nod-ai/shark-ai/tree/main/sharktuner/model_tuner) for detailed information on flags and MLIR input files.
````

---

### Comment by kuhar

**File:** `sharktuner/PYPI_README.md`

**Line:** 42

**Comment:**

also here

---

### Comment by kuhar

**File:** `sharktuner/PYPI_README.md`

**Line:** 44

**Comment:**

also here

---

### Comment by kuhar

**File:** `sharktuner/PYPI_README.md`

**Line:** 54

**Comment:**

also here

---

### Comment by bangtianliu

**File:** `sharktuner/PYPI_README.md`

**Line:** 31

**Comment:**

yeah it should be shell everywhere


---

## PR #2362: [Tuner] use custom transform op for matching contraction op

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2362
**State:** MERGED

### Comment by kuhar

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 61

**Comment:**

```suggestion
    if linalg.isa_contraction_op(op):
```

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 95

**Comment:**

Can we get rid of this branch and match with an empty array? Just to keep things uniform.

---

### Comment by kuhar

**File:** `sharktuner/tests/spec_builder_test.py`

**Line:** 181

**Comment:**

Can we add a testcase that also exercises batch dim?

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 89

**Comment:**

Are you sure the lhs always knows about the batch dim? In existing tuning specs for Unet, we sometimes had the RHS being broadcast: https://github.com/nod-ai/sdxl-scripts/blob/4fa7ccbc3de4873c0751a48ef9b4b86a7f24428e/int8-model/specs/attention_and_matmul_spec.mlir#L632-L648

I think it could also be that LHS gets broadcast.

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 89

**Comment:**

https://github.com/llvm/llvm-project/blob/2936a2c882d76c719f9a96e443ad3f75b366bc8f/mlir/lib/Dialect/Linalg/IR/LinalgInterfaces.cpp#L457-L460

```C++
 // A & B & C are the "batch" dimensions.
  llvm::SmallDenseSet<int64_t> batches = a;
  llvm::set_intersect(batches, b);
  llvm::set_intersect(batches, c);
```
`inferContractionDimsImpl` requires 'batch' to be in lrhs, rhs and output.


---

## PR #2272: [tuner] use subgroup basis for lowering config

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2272
**State:** MERGED

### Comment by kuhar

**File:** `sharktuner/sharktuner/common.py`

**Line:** 272

**Comment:**

We don't need this control flow -- we can assert the negation of the `if` condition above

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/common.py`

**Line:** 268

**Comment:**

Maybe add a helper to create an `I64ArrayAttr`? 

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 235

**Comment:**

I think there's an upcoming PR to allow distributing subgroups on multiple m dims -- I wonder if the `sg_m_cnt` logic will get out of date very soon: https://github.com/iree-org/iree/pull/22000

cc: @Groverkss 

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/common.py`

**Line:** 263

**Comment:**

Wasn't this a `std::tuple` on the iree side? I'm surprised value became a list.

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/common.py`

**Line:** 263

**Comment:**

Good catch! This is about assembling the input from the Python side, which then gets used to create `iree_gpu.LoweringConfigAttr`. The Python input format is `[[counts], [mapping]]` (list of two lists), which gets converted to the appropriate MLIR attributes internally (I think).

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 235

**Comment:**

https://github.com/iree-org/iree/pull/22000 has been landed, so I will also sync in this PR directly after I understand how it works @kuhar, also cc @Groverkss here.

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 235

**Comment:**

Has this been done @bangtianliu ?

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 383

**Comment:**

How do we know it's always in this order?

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 235

**Comment:**

Not yet in this PR.

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 235

**Comment:**

Can you add a TODO comment then?

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 235

**Comment:**

Sure will do.

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 383

**Comment:**

Basically it follows code from here: https://github.com/iree-org/iree/pull/21912/files#diff-e491707653524e61bd22ba47ca5bd8414065b70bd1c6b85f45947929d7a49aafR1564-R1568

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 383

**Comment:**

The code for attention calls the `projectBasis` helper though: https://github.com/iree-org/iree/pull/21912/files#diff-e491707653524e61bd22ba47ca5bd8414065b70bd1c6b85f45947929d7a49aafR1579-R1586

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 383

**Comment:**

Oh, I think I see, you do it later on this code. Makes sense then.

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 230

**Comment:**

```suggestion
        # TODO(Bangtian): Sync changes from IREE PR: https://github.com/iree-org/iree/pull/22000.
```

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 230

**Comment:**

Sure, I will address this in the next PR. 

---

## PR #2122: [tuner]: add target info to tuner

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2122
**State:** MERGED

### Comment by kuhar

**File:** `sharktuner/sharktuner/common.py`

**Line:** 31

**Comment:**

Why do we need a separate definition on the python side in addition to the class defined by python bindings? Both represent contain the same information.

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/dispatch_constraints.py`

**Line:** 204

**Comment:**

Use a permanent link, otherwise this will get out of date eventually and won't point to the same code

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/common.py`

**Line:** 31

**Comment:**

As shown in the code below, the Python binding class iree_gpu.TargetInfo is read-only and designed to extract target info from existing GPU executables. For testing and configuration purposes, we need a mutable version that we can construct with specific values.
Furthermore, the `mma_intrinsics` field should be added to the struct ireeGPUTargetInfo, which would allow us to retire iree_codegen.query_mma_intrinsics.

```C++
  py::class_<ireeGPUTargetInfo>(iree_gpu_module, "TargetInfo")
      .def_prop_ro("arch",
                   [](const ireeGPUTargetInfo &self) -> std::string {
                     MlirStringRef strRef = mlirIdentifierStr(self.arch);
                     return std::string(strRef.data, strRef.length);
                   })
      .def_prop_ro("subgroup_size_choices",
                   [](const ireeGPUTargetInfo &self) -> std::vector<int64_t> {
                     return getIntArrayAttrValues(self.subgroupSizeChoices);
                   })
      .def_prop_ro("max_thread_count_per_workgroup",
                   [](const ireeGPUTargetInfo &self) -> int64_t {
                     return self.maxThreadCountPerWorkgroup;
                   })
      .def_prop_ro("max_workgroup_sizes",
                   [](const ireeGPUTargetInfo &self) -> std::vector<int64_t> {
                     return getIntArrayAttrValues(self.maxWorkgroupSizes);
                   })
      .def_prop_ro("max_workgroup_memory_bytes",
                   [](const ireeGPUTargetInfo &self) -> int64_t {
                     return self.maxWorkgroupMemoryBytes;
                   });
```

It means that we need another PR from IREE end to improve it.



---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/common.py`

**Line:** 31

**Comment:**

https://github.com/iree-org/iree/pull/21812 is sent to ensure that we can use the class defined by python binding. 

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 201

**Comment:**

Can you open an issue to also test this on gfx950 and gfx1201?

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/dispatch_constraints.py`

**Line:** 205

**Comment:**

Fine for now, but we should allow both in the future

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/dispatch_constraints.py`

**Line:** 293

**Comment:**

Also here

---

### Comment by kuhar

**File:** `sharktuner/tests/constraint_generator_test.py`

**Line:** 31

**Comment:**

Why do we have to specify scope? I haven't seen `scope` specified elsewhere. I think `function` is the default, no?

---

### Comment by kuhar

**File:** `sharktuner/tests/dispatch_constraints_test.py`

**Line:** 25

**Comment:**

also here

---

### Comment by bangtianliu

**File:** `sharktuner/tests/constraint_generator_test.py`

**Line:** 31

**Comment:**

Yes, you are correct. The context is that previously I tried to set module scope and it did not work. 

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 201

**Comment:**

Sure, will do!

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/dispatch_constraints.py`

**Line:** 205

**Comment:**

It seems that on the IREE side, gfx1100 only accepts 32. Do we know if using 64 will always lead to compilation failures? I’m not sure.

---

## PR #2104: [tuner] add acc layout match to constraint generation for attention

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/2104
**State:** MERGED

### Comment by kuhar

**File:** `sharktuner/sharktuner/dispatch_constraints.py`

**Line:** 478

**Comment:**

Won't this fit on a single line?
```suggestion
    constraints += [match_layout(qk_mma_acc_layout, pv_mma_acc_layout)]
```
?

---

## PR #1909: [tuner] improve support for attention op

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/1909
**State:** MERGED

### Comment by Max191

**File:** `sharktuner/tests/dispatch_constraints_test.py`

**Line:** 395

**Comment:**

I don't fully understand why the result is expected to be one of these options. Why are we able to select `(32, 32, 8)` and why are we not able to select `(32, 32, 16)`? (mostly just for my own understanding)

---

### Comment by bangtianliu

**File:** `sharktuner/tests/dispatch_constraints_test.py`

**Line:** 395

**Comment:**

due to type matching

---

### Comment by Max191

**File:** `sharktuner/tests/dispatch_constraints_test.py`

**Line:** 395

**Comment:**

Oh, right, we only look at the element type. Okay, thanks!

---

## PR #1772: [tuner] add support for attention op

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/1772
**State:** MERGED

### Comment by Groverkss

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 372

**Comment:**

We don't always want to promote Q operand. It should be a tuning option.

---

### Comment by Max191

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 348

**Comment:**

nit: inline `m/n/k2_dim` as `opinfo.m/n/k2_dims[-1]` and remove the extra variables.

---

### Comment by Max191

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 349

**Comment:**

Why is subgroup_n_count always 1 for the QK matmul?

---

### Comment by Max191

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 569

**Comment:**

nit: move these definitions to where they are used below

---

### Comment by Max191

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 608

**Comment:**

This makes sense for QK, if it is not captured by the attention detail, but why not just take the element type of the output for the PV matmul?

---

### Comment by Max191

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 553

**Comment:**

Can you add some docs about what the different fields are that are being set? The `transposed_q`, `transposed_k`, and `transposed_v` need definitions, for example, since it's not clear what it means to be transposed.

---

### Comment by Max191

**File:** `sharktuner/sharktuner/dispatch_constraints.py`

**Line:** 270

**Comment:**

I think these constraints are more tied to the VectorDistribute pipeline. Maybe rename this function to `is_valid_vector_distribute_mma_schedule`?

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 349

**Comment:**

I followed the logic defined here: 
https://github.com/iree-org/iree/blob/51d8006753ac44e1aa52e437a8204a6d3bfe3973/compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.cpp#L574

And detailed explanation in here:

https://github.com/iree-org/iree/blob/51d8006753ac44e1aa52e437a8204a6d3bfe3973/compiler/src/iree/compiler/Codegen/Common/GPU/GPUHeuristics.cpp#L489-L512

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 608

**Comment:**

This code is based on IREE's implementation:
https://github.com/iree-org/iree/blob/9e8691fdab06ee5c432bbf40ea70883b4b78d8bf/compiler/src/iree/compiler/Codegen/LLVMGPU/KernelConfig.cpp#L1398-L1412


---

### Comment by Max191

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 608

**Comment:**

Let's just use the output element type here in the tuner, since I don't really see a reason not to.

---

### Comment by Max191

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 577

**Comment:**

nice docs!

---

### Comment by Max191

**File:** `sharktuner/sharktuner/dispatch_constraints.py`

**Line:** 326

**Comment:**

This assumes that the C-matrix is not promoted. Maybe update the name of the function or add some docs to make this more clear? (maybe `calculate_schedule_input_operands_shared_memory_usage_in_bytes`?)

---

### Comment by Max191

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 372

**Comment:**

That's good to know, thanks! It's one of the TODOs to expose operand promotion as a tunable parameter, so I think we should leave it as a follow up, and we can expose it for attention too.

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/common.py`

**Line:** 184

**Comment:**

Can we take this class out of `common.py`? The unintuitive part to me is that all of these are z3 types, even though mma schedule is more general than that. I think this belongs to some file related strictly to constraint generation

---

## PR #1733: [tuner] add parser for attention op

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/1733
**State:** MERGED

### Comment by Max191

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 92

**Comment:**

nit: I think we do want to use `iree_codegen.isa_attention_op`, so maybe update the TODO to say that we should use it once it is available? (is it already available?)

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 92

**Comment:**

yeah, I will update this when https://github.com/iree-org/iree/pull/21216 is landed. 

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/dispatch_parser.py`

**Line:** 92

**Comment:**

```suggestion
        # TODO(Bangtian): Switch to `iree_codegen.isa_attention_op` once available.
```

---

## PR #1641: [tuner] use config list for constraint generators

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/1641
**State:** MERGED

### Comment by kuhar

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 48

**Comment:**

```suggestion
        Generates a transform dialect spec from a config list.
```
```suggestion
        Generate a transform dialect spec from a config list.
```

---

### Comment by kuhar

**File:** `sharktuner/tests/constraint_generator_test.py`

**Line:** 199

**Comment:**

I think we could simplify this code ba bunch by writing a for loop and then checking each pair separately (instead of relying on `all`). Once you name the pair elements the code will become more readable.

---

### Comment by kuhar

**File:** `sharktuner/tests/constraint_generator_test.py`

**Line:** 271

**Comment:**

Similar here

---

### Comment by Max191

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 88

**Comment:**

optional nit: I personally like the name `config_list` for the param.

---

### Comment by Max191

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 45

**Comment:**

nit: I had an idea to help readability. We could create a dataclass for the config type (`tuple[str, ir.Attribute]`), that contains a field for the name and the configuration:
```
@dataclass
class TuningConfiguration:
    """
    A TuningConfiguration contains an attribute that will be set on an op as a
    result of running a tuning spec, along with its name. For example, a common
    tuning configuration would have "compilation_info" as its name, and an
    `iree_codegen.CompilationInfoAttr` as the configuration.
    """

    name: str
    configuration: ir.Attribute
```

---

### Comment by kuhar

**File:** `sharktuner/tests/constraint_generator_test.py`

**Line:** 206

**Comment:**

IIRC assert already prints the expected and the actual values when you have `==`, could you check if we need this string?

---

### Comment by kuhar

**File:** `sharktuner/tests/constraint_generator_test.py`

**Line:** 285

**Comment:**

Also here

---

### Comment by bangtianliu

**File:** `sharktuner/tests/constraint_generator_test.py`

**Line:** 206

**Comment:**

yeah, you are right.

---

### Comment by kuhar

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 55

**Comment:**

I'm not sure what this example is trying to show here -- I think this belongs to the documentation for `common.TuningConfirguration`.

---

## PR #1602: [tuner] build td spec using config list

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/1602
**State:** MERGED

### Comment by kuhar

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 131

**Comment:**

How could we test this? I think it should be possible to generate some TD IR with more than one config and then make sure that is parses.

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/spec_builder.py`

**Line:** 131

**Comment:**

We don't currently have a test for this file. I’ll look into how I can add one.

---

### Comment by Max191

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 92

**Comment:**

The remaining part to connect this with the constraint generators is to make the generate_solutions function generate an iterator of these config lists:
https://github.com/nod-ai/shark-ai/blob/b5d0abf85bf4460d27dd019350673f122782e593/sharktuner/sharktuner/constraint_generator.py#L259-L270

i.e., `Iterator[list[tuple[str, ir.Attribute]]]`

This can be a follow-up, or could be part of this PR, since I think it adds helpful context about what is happening in this PR.

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 92

**Comment:**

I'll address this in the next PR, as it also involves updating the corresponding tests.

---

## PR #1511: [tuner]  create interface for constraint generation

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/1511
**State:** MERGED

### Comment by Max191

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 9

**Comment:**

```suggestion
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
```

---

### Comment by Max191

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 245

**Comment:**

Can you add some docs that describe what this class is for? Something like:
```
"""
Describes how to generate constraints and produce tuning candidates for a specific
type of tunable problem. Implementations of ConstraintGenerator will carry information
about the problem that is required for generating constraints (e.g., contraction
dimensions and sizes for a contraction op).
"""
```

---

### Comment by rkayaith

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 265

**Comment:**

any reason the arguments used for `kwargs` aren't explicit here in the interface?

---

### Comment by rkayaith

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 34

**Comment:**

`import *` is generally not a good idea, it makes it hard to tell where functions are coming from when scanning the code. 

`from .constraint_generator import foo, bar, ...` or `import .constraint_generator; constraint_generator.foo()` would be preferrable

---

### Comment by rkayaith

**File:** `sharktuner/tests/dispatch_parser_test.py`

**Line:** 295

**Comment:**

is this covered by some other test now?

---

### Comment by bangtianliu

**File:** `sharktuner/tests/dispatch_parser_test.py`

**Line:** 295

**Comment:**

This is because the function `get_problem_size` is just retired.

---

### Comment by rkayaith

**File:** `sharktuner/tests/dispatch_parser_test.py`

**Line:** 295

**Comment:**

it seems this information is moved to `ContractionOpInterfaceConstraintGenerator`/`ConvolutionOpInterfaceConstraintGenerator`, is it possible to get that object here and check it?

It might make sense to make that a separate test, but tbh I don't think these parsing tests are very useful with the checks removed, since they no longer ensure that the op is parsed _correctly_.

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 265

**Comment:**

We want it to be flexible supporting different ops. detailed reason you can find in the issue https://github.com/nod-ai/shark-ai/issues/1442
> The only required argument is the codegen pipeline, and anything else will be passed as kwargs, because the additional parameters may depend on which codegen pipeline is being used.

---

### Comment by rkayaith

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 265

**Comment:**

I see, maybe renaming `kwargs` to `codegen_pipeline_options` would help make that clearer?

---

### Comment by rkayaith

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 335

**Comment:**

If you want to just forward all the arguments, you can do this:
```suggestion
            codegen_pipeline=codegen_pipeline,
            **kwargs,
```
It seems `generate_generic_contraction_solutions` already has defaults for some arguments, so this approach also avoids duplicating the default values.

---

### Comment by bangtianliu

**File:** `sharktuner/tests/dispatch_parser_test.py`

**Line:** 295

**Comment:**

All relevant tests have been moved to `constraint_generator_test.py`. Successfully generating a configuration implies that the constraint generator is functioning correctly.

---

### Comment by bangtianliu

**File:** `sharktuner/sharktuner/candidate_gen.py`

**Line:** 34

**Comment:**

Thanks for your comments. I just formatted the tuner code based on this. 

---

### Comment by Max191

**File:** `sharktuner/model_tuner/model_tuner.py`

**Line:** 12

**Comment:**

It is okay for this PR, since I think there are already other changes mixed in, but for the future, it is better to make large formatting changes like this as a separate followup PR IMO. It helps reduce clutter for reviewers from all the NFC changes. 

---

### Comment by Max191

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 265

**Comment:**

I like the suggestion for renaming! Although, `codegen_pipeline_options` may be slightly misleading, since there is already an `iree_gpu.PipelineOptionsAttr`. Maybe something like `pipeline_constraint_options`?

---

### Comment by Max191

**File:** `sharktuner/tests/constraint_generator_test.py`

**Line:** 30

**Comment:**

Could we have some tests that are directly testing the `ContractionOpInterfaceConstraintGenerator` and the `ConvolutionOpInterfaceConstraintGenerator`, instead of the `generate_generic_contraction_solutions`? You can mock the problem specific info by creating a root op like is done in the dispatch_parser tests.

This makes it easier to refactor code in the future, since we don't rely on the `generate_generic_contraction_solutions` for our unit tests.

I think it may also make the tests shorter, since you won't need to pass all the default values for the extra kwargs.

---

### Comment by Max191

**File:** `sharktuner/tests/dispatch_constraints_test.py`

**Line:** 73

**Comment:**

Delete the comment?

---

### Comment by Max191

**File:** `sharktuner/tests/dispatch_constraints_test.py`

**Line:** 156

**Comment:**

Delete comment?

---

### Comment by Max191

**File:** `sharktuner/tests/dispatch_constraints_test.py`

**Line:** 223

**Comment:**

Delete

---

### Comment by Max191

**File:** `sharktuner/tests/dispatch_constraints_test.py`

**Line:** 274

**Comment:**

Delete comment

---

### Comment by Max191

**File:** `sharktuner/tests/dispatch_constraints_test.py`

**Line:** 287

**Comment:**

Delete comment

---

### Comment by Max191

**File:** `sharktuner/tests/dispatch_parser_test.py`

**Line:** 295

**Comment:**

It may still generate configurations even if it parses the convolution incorrectly. I think we should still be testing the logic for gathering the appropriate convolution/contraction dimensions/sizes. You could add tests in constraint_generator_test.py for this.

---

### Comment by rkayaith

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 56

**Comment:**

I think for this function you can make these regular arguments, rather than using `**kwargs`. That way extra arguments/typos will become an error at runtime.

---

### Comment by Max191

**File:** `sharktuner/tests/constraint_generator_test.py`

**Line:** 31

**Comment:**

```suggestion
def test_matmul_constraint_generator_dims(tuner_ctx: common.TunerContext) -> None:
```

---

### Comment by Max191

**File:** `sharktuner/tests/constraint_generator_test.py`

**Line:** 58

**Comment:**

I have a suggestion for shortening the tests. I think this section could be split out into a separate function, like
```
def build_func_with_matmul(module: ir.Module, m_size: int, n_size: int, k_size: int):
    # build a func op with a linalg.matmul op of size m_size x n_size x k_size.
    ...
```
(Element types could also be a parameter or just always f16xf16xf32)

Then, this section can be replaced with:
```
with ir.Location.unknown(context):
    module = ir.Module.create()
    build_func_with_matmul(module, 16, 32, 64):
```

And you can reuse the function below in `test_generate_solutions` and `test_generate_solutions_tile_and_fuse_contraction_padding` by creating a constraint generator like you did in this test, and then running `gen.generate_solutions(...)`.

You can also create a similar function for convolution, and use it in both `test_conv2d_constraint_generator_dims` and `test_generate_solutions_tile_and_fuse_conv_padding`

---

### Comment by rkayaith

**File:** `sharktuner/sharktuner/constraint_generator.py`

**Line:** 340

**Comment:**

I think the change here and below to just forward `pipeline_constraint_options` got dropped somehow
```suggestion
        return generate_generic_contraction_solutions(
            tuner_ctx=tuner_context,
            contraction_dims=self.dims,
            matmul_size=self.matmul_size,
            lhs_type=self.lhs_type,
            rhs_type=self.rhs_type,
            res_type=self.res_type,
            dispatch_kind=common.DispatchKind.contraction,
            codegen_pipeline=codegen_pipeline,
            **pipeline_constraint_options
        )
```

---

## PR #1310: [tuner] padding support along TileAndFuse pipeline

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/1310
**State:** MERGED

### Comment by Max191

**File:** `sharktuner/sharktuner/dispatch_constraints.py`

**Line:** 525

**Comment:**

Add a TODO here for removing operand `2` promotion when codegen can support it. We currently need to promote the C matrix when we have padding, but Nirvedh and Jerry are working on getting rid of this restriction. Let's add the TODO so we don't forget about this later.

---

### Comment by Max191

**File:** `sharktuner/sharktuner/dispatch_constraints.py`

**Line:** 310

**Comment:**

Let's refactor this a little and instead of passing a bool for required padding, we can pass the list of actual padding to add. This way, we could even make the amount of padding a tunable parameter, and include it as part of the constraints if we want in the future.

---

### Comment by Max191

**File:** `sharktuner/sharktuner/dispatch_constraints_test.py`

**Line:** 89

**Comment:**

Also add an assert that the `promote_operands` is `[0, 1, 2]`.

---

### Comment by Max191

**File:** `sharktuner/sharktuner/dispatch_constraints.py`

**Line:** 534

**Comment:**

nit: It is a bit more clear and concise if `padding` is defined outside the if.
```suggestion
        padding = None
        if required_padding:
            # TODO: Remove promotion of operand 2 once codegen supports handling padded outputs without promotion.
            promote_operands = [0, 1, 2]
            workgroup_tile_m, workgroup_tile_n, _ = workgroup_tile_sizes
            _, _, reduction_tile_k = reduction_tile_sizes
            _, _, mma_intrinsic_k = mma_attr.mnk_shape
            padding = [
                workgroup_tile_m,
                workgroup_tile_n,
                reduction_tile_k * mma_intrinsic_k,
            ]
```

---

## PR #1304: [tuner] get the function name from the func.func in the input module

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/1304
**State:** MERGED

### Comment by Max191

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 34

**Comment:**

Are there exposed bindings for FuncOp? If so, could you use the bindings to get the sym_name and use `isinstance()` instead of checking the `func_op.name`?

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 34

**Comment:**

This would be nice to have but I wouldn't wait for it here. If we are going to use it in one or two places, the ROI is kind of low IMO. That could be a good starter task if you want to open an issue for that.

---

### Comment by Max191

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 34

**Comment:**

Yeah, I don't mean to ask for going and adding the bindings. I'm just asking to use it here if the bindings already exist. If they don't exist yet then it is totally fine as is.

---

### Comment by bangtianliu

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 34

**Comment:**

here the func_op's type is `'iree.compiler._mlir_libs._mlir.ir.Operation`, so the binding for FuncOp cannot be used.

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 34

**Comment:**

You can cast to FuncOp though, right?

---

### Comment by bangtianliu

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 34

**Comment:**

Actually no, FuncOp only has constructor: 

```python
  def __init__(self, sym_name, function_type, *, sym_visibility=None, arg_attrs=None, res_attrs=None, no_inline=None, loc=None, ip=None):
```

so func.FuncOp(func_op) is unable to do cast, and also I think using above constructor is too much. 

---

### Comment by Max191

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 34

**Comment:**

Are you still able to call `isinstance()` for the op and use `func_op.sym_name`? The functions may still be registered on the op.

---

### Comment by bangtianliu

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 34

**Comment:**

No, the func_op is in the type of `iree.compiler._mlir_libs._mlir.ir.Operation`. I tried 'print(func_op.sym_name)' and got below error:

```
AttributeError: 'iree.compiler._mlir_libs._mlir.ir.Operation' object has no attribute 'sym_name'
```

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 34

**Comment:**

It's not a constructor though that we want to use, IIUC. cc: @makslevental 

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 34

**Comment:**

@bangtianliu Maks suggested `func_op.opview` should give us the function op object -- can you check that?

---

### Comment by bangtianliu

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 34

**Comment:**

Yeah, it works. Thanks.


---

## PR #1289: [tuner] add tests for named ops

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/1289
**State:** MERGED

### Comment by kuhar

**File:** `tuner/tuner/dispatch_parser_test.py`

**Line:** 159

**Comment:**

We can provide this as the third function argument

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_parser_test.py`

**Line:** 205

**Comment:**

Also here

---

## PR #1264: [tuner] retire op_matchers.py through IREE python bindings

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/1264
**State:** MERGED

### Comment by Max191

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 48

**Comment:**

I think this PR addresses the TODO here. I think you can either remove it in this PR and add some tests for named ops, or restrict to only `linalg.generic` ops, and address the TODO in a followup PR. WDYT?

---

### Comment by Max191

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 165

**Comment:**

Instead of an assert, how about just logging an error, and then returning an empty list of configs?

Perhaps 2 possible messages. One for when there are multiple root ops, and one for when there are no root ops. When there are no root ops, it can suggest to add the `--iree-config-add-tuner-attributes` flag during compilation with IREE.

---

### Comment by Max191

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 38

**Comment:**

I think renaming `supports` to `has_valid_root_op` is more clear.

---

### Comment by Max191

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 42

**Comment:**

I think we can rename this to `get_problem_size`. The ProblemSize struct already contains more than just the shapes.

---

### Comment by Max191

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 63

**Comment:**

I don't think there is any need to check the name of the func_op anymore. This logic used to be a hack, but we don't need it anymore now that we can directly check that the op is a contraction.

---

### Comment by Max191

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 69

**Comment:**

Can you add a TODO to expose some bindings for getting indexing_maps and remove the string matching here?

Also, this will not work for named ops, since the attribute is called `memoized_indexing_maps` for named ops. If we are supporting named ops in this PR, then this case needs to be supported.

---

### Comment by Max191

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 115

**Comment:**

Same here. I don't think we need to match against the func_op sym_name name anymore.

---

### Comment by Max191

**File:** `tuner/tuner/dispatch_parser_test.py`

**Line:** 149

**Comment:**

This used to be testing the convolution parsing logic. I think this test should continue to check the parsing logic by calling the `supports` function (you will probably need to remove the func_op sym_name matching part of `supports` in order to do this).

---

### Comment by Max191

**File:** `tuner/tuner/dispatch_parser_test.py`

**Line:** 149

**Comment:**

Same for the contraction op tests above. It should call the `supports` function to make sure the matching logic there is working properly.

---

### Comment by bangtianliu

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 48

**Comment:**

> I think this PR addresses the TODO here. I think you can either remove it in this PR and add some tests for named ops, or restrict to only `linalg.generic` ops, and address the TODO in a followup PR. WDYT?

I’ll go with the option to restrict support to only `linalg.generic` ops for now, so this PR stays focused and doesn’t become too heavy, as your above comment.

---

### Comment by Max191

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 117

**Comment:**

nit: This can be `return root_op.name in self.supported_ops` now.

---

## PR #1225: [tuner] deferring the link phase to multi-threaded compilation phase

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/1225
**State:** MERGED

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 483

**Comment:**

Could we move this to a helper function?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 749

**Comment:**

```suggestion
                td_specs: list[ir.Module] = [spec, starter_td_spec]
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 754

**Comment:**

```suggestion
                td_specs_to_link = determine_td_specs_to_link(
                    td_specs,
                    log_duplicates=(candidate_num == 0),
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 465

**Comment:**

Why are we writing the td spec to a file instead of using the path provided through CLI arguments?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 471

**Comment:**

Are we linking a single input only? I'm confused by this code.

---

### Comment by bangtianliu

**File:** `tuner/tuner/libtuner.py`

**Line:** 465

**Comment:**

Because linking has two steps, firstly it is about combining starter_td_spec and candidate spec into one module (single input) and then link through calling `--iree-codegen-link-tuning-specs`. 

---

### Comment by bangtianliu

**File:** `tuner/tuner/libtuner.py`

**Line:** 471

**Comment:**

yes, it is about serializing the td specs into one module and then link,  refer to examples in https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Codegen/Common/test/link_tuning_specs.mlir#L1.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 465

**Comment:**

Oh, I see what you mean. In this case, I think we'd need a different argument name. This is no longer just a `starter_td_spec_str`, it's a combined/nested spec that needs linking.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 465

**Comment:**

Maybe something like:
```py
def flatten_nested_td_spec(td_spec_str: str, output_path: Path) -> None:
```

---

## PR #1203: Revert "[shortfin] Enable building Rust deps on Linux (#619)"

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/1203
**State:** MERGED

### Comment by ScottTodd

**File:** `.github/workflows/ci-libshortfin.yml`

**Line:** 134

**Comment:**

Looks like the revert was not clean. Only the "Setup Rust" step should be removed. These other checkout / submodule steps should not be added.

---

### Comment by ScottTodd

**File:** `.github/workflows/ci-libshortfin.yml`

**Line:** 134

**Comment:**

this change should also remain

```diff
+              -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_LINKER_TYPE=LLD
-              -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_LINKER_TYPE=LLD -DSHORTFIN_ENABLE_TOKENIZERS=ON
```

---

### Comment by ScottTodd

**File:** `.github/workflows/ci-libshortfin.yml`

**Line:** 134

**Comment:**

Technically only the CMake flag change is needed to disabling the native tokenizers component that is causing issues with CMake 4.0.0, but the native tokenizers component is the only thing using Rust, so we can also remove the "setup Rust" step.

---

## PR #1184: [tuner] expose the function merge_td_specs as utility executable

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/1184
**State:** MERGED

### Comment by kuhar

**File:** `tuner/tuner/merge_td_specs.py`

**Line:** 1

**Comment:**

```suggestion
# Copyright 2025 Advanced Micro Devices, Inc.
```

---

### Comment by kuhar

**File:** `tuner/tuner/merge_td_specs.py`

**Line:** 10

**Comment:**

```suggestion
This script wraps the `iree-opt --iree-codegen-link-tuning-specs` pass.
```

---

### Comment by kuhar

**File:** `tuner/tuner/merge_td_specs.py`

**Line:** 102

**Comment:**

You can use the top-level docstring here to avoid duplication

---

### Comment by kuhar

**File:** `tuner/tuner/merge_td_specs.py`

**Line:** 130

**Comment:**

Is this printed by default? If one of your supported output options is to print to stdout, you we can't interleave that with random logs

---

### Comment by kuhar

**File:** `tuner/tuner/merge_td_specs.py`

**Line:** 139

**Comment:**

Also here

---

### Comment by kuhar

**File:** `tuner/tuner/merge_td_specs_test.py`

**Line:** 1

**Comment:**

```suggestion
# Copyright 2025 Advanced Micro Devices, Inc.
```

---

### Comment by kuhar

**File:** `tuner/tuner/merge_td_specs_test.py`

**Line:** 9

**Comment:**

We don't need to explain how to run unit tests in each file

---

### Comment by kuhar

**File:** `tuner/tuner/merge_td_specs_test.py`

**Line:** 28

**Comment:**

Can we put this and similar functions in a new file with common testing utilities? Say, `test_utils.py`.

---

### Comment by kuhar

**File:** `tuner/tuner/merge_td_specs.py`

**Line:** 101

**Comment:**

We should use the top-level docstring here to avoid duplications (`__doc__`).

---

### Comment by kuhar

**File:** `tuner/tuner/merge_td_specs.py`

**Line:** 70

**Comment:**

why doesn't this call the merge function?

---

### Comment by kuhar

**File:** `tuner/tuner/test_utils.py`

**Line:** 22

**Comment:**

```suggestion
def tuner_ctx() -> Generator[common.TunerContext, None, None]:
    mock_logger = MagicMock(spec=Logger)
```

---

### Comment by bangtianliu

**File:** `tuner/tuner/merge_td_specs.py`

**Line:** 70

**Comment:**

> why doesn't this call the merge function?

Since the pass is named `--iree-codegen-link-tuning-specs`, so I just keep using the name `link_*`.  However, for the default attribute cases, the link actually is about performing the merge of td specs annotated with default attributes. 
Maybe add a code comment here to explain?  Open to suggestions if there's a better way to handle this naming inconsistency.

---

### Comment by kuhar

**File:** `tuner/tuner/merge_td_specs.py`

**Line:** 70

**Comment:**

I meant the python function to determine the tuning specs to merge and print warnings in case of duplicate matchers -- it would be nice to emit the same warnings here

---

### Comment by bangtianliu

**File:** `tuner/tuner/merge_td_specs.py`

**Line:** 70

**Comment:**

yeah, you are fully correct. That function `determine_td_specs_to_link` should be reused here for emitting a warning. 

---

## PR #1171: [tuner] add the option of providing starter td spec

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/1171
**State:** MERGED

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 218

**Comment:**

This should be a function that comes with its own unit tests

---

### Comment by kuhar

**File:** `tuner/tuner/common_test.py`

**Line:** 358

**Comment:**

We should also have a test for IR with no matchers

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 246

**Comment:**

Debug print?

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 227

**Comment:**

```suggestion
    assert should_link
```

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 234

**Comment:**

```suggestion
    assert should_link
```

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 245

**Comment:**

```suggestion
    assert not should_link
```

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 211

**Comment:**

```suggestion
    if starter_td_spec is None:
```

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 230

**Comment:**

I think we should unify these two code paths to reduce the overall complexity of the control flow

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 265

**Comment:**

If you make `check_td_spec_matchers_overlap` accept td specs as arguments and call `get_matcher_names_from_td_spec` internally, you can make it take a list of tuning specs. For the case with no starter spec, the list will always have a single element.

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 266

**Comment:**

Similarly, linking should be able to handle a single tuning spec passed to it

---

### Comment by kuhar

**File:** `tuner/examples/simple/simple_tuner.py`

**Line:** 75

**Comment:**

Can you undo this formatting change?

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 251

**Comment:**

Make this list a local variable and append to it instead of having this branching logic

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 232

**Comment:**

The indentation is weird here

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 167

**Comment:**

The indentation is weird here

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 184

**Comment:**

I can't parse this sentence, the grammar seems off.

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 174

**Comment:**

nit: I'd put this assert on top and check for `len(td_specs) <= 2` and also allow for empty inputs. This way the preconditions are clear.

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 176

**Comment:**

nit: you can unpack it like this
```suggestion
    starter_td_spec, current_td_spec = td_specs
```

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 181

**Comment:**

```suggestion
    overlapping_matchers = starter_matchers & current_matchers
```

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 185

**Comment:**

`new_diplicate_matchers`?

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 152

**Comment:**

Instead of having the caller provide the known duplicate matchers and returning new duplicated, I think it would be easier to add a boolean flag that decides whether to print a warning about duplicates. This simplifies the interface. If you want to test the duplicate finding logic, you can make it a helper function and test in isolation.

---

### Comment by Max191

**File:** `tuner/tuner/common.py`

**Line:** 342

**Comment:**

Better to use something like `isinstance(op, transform.NamedSequenceOp)` instead of checking the string name of the operation.

---

### Comment by Max191

**File:** `tuner/tuner/common.py`

**Line:** 348

**Comment:**

Same here. Better to use `isinstance(inner_op, transform.ForeachMatchOp)`

---

### Comment by Max191

**File:** `tuner/tuner/libtuner.py`

**Line:** 699

**Comment:**

The flag from the `simple_tuner.py` example is leaking into the core libtuner implementation here. The flag should probably be moved to the arguments of the tuner (i.e., in `parse_arguments`)

---

### Comment by Max191

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 245

**Comment:**

This is going to spin up iree-opt, link the same starting spec, and write/read to a temp file for every single generated candidate. This is probably going to slow down candidate generation significantly. This is something that I expect is better to do selectively depending on whether we are doing model vs dispatch tuning. In dispatch tuning, the extra specs don't matter, so we don't even need to do any linking. During model benchmarking, it does matter, but there are generally fewer candidates to link.

One way to do this would be to perform the linking during the compile() step of tuning, and produce new linked specs for each model compilation. Then it can be an option to pass an initial spec for each invocation of `compile()` in the tuning client.

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 245

**Comment:**

It would be good to measure the candidate generation time before and after and decide based on this.

---

### Comment by Max191

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 245

**Comment:**

The other benefit of doing this is that we reduce file sizes in the tuning tmp directory, and it is easier to look through any generated dispatch candidates, since there is less clutter from unnecessary linked IR.

Measurements are always good to have, though.

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 245

**Comment:**

On the other hand having two different td script inherently increases the complexity. It may be annoying to have to deal with potential bugs where we get the expected attributes when compiling dispatches but not model caused by issues with linking etc. I'd rather learn whether the specs are working or not when dealing with candidates.

---

### Comment by bangtianliu

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 245

**Comment:**

I did a measurement, the linking time is around 0.047 seconds for linking matmul and attention specs. Maybe just skip linking when the number of td specs is 1. 

---

### Comment by bangtianliu

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 245

**Comment:**

And also I did another measurement about whole candidate generation time with `--num-candidates=20`
The command I used for measurement `python -m examples.simple double_mmt.mlir mmt_benchmark.mlir --simple-compile-flags-file=examples/simple/compile_flags.txt --simple-model-benchmark-flags-file=examples/simple/model_benchmark_flags.txt --devices=hip://0 --simple-num-dispatch-candidates=5 --simple-num-model-candidates=3 --num-candidates=20 --simple-starter-td-spec=attention.mlir ` 

It took 1.38 seconds before (with just the original code and no starter TD spec), and 2.40 seconds after applying the changes in this PR — including the time spent determining which TD specs to link and the overhead of linking them.

---

### Comment by Max191

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 245

**Comment:**

So, sounds like about 0.05s per candidate. With something like 8000 candidates, this adds close to 7 minutes to the candidate generation. Personally, I think it is worth skipping the linking for dispatch candidates, but I don't have a particularly strong opinion, since most of the tuning time is still in benchmarking anyway. If it becomes annoying to deal with, we can always optimize later on.

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 245

**Comment:**

> I did a measurement, the linking time is around 0.047 seconds for linking matmul and attention specs. Maybe just skip linking when the number of td specs is 1.

@bangtianliu with how many candidates generated?

> With something like 8000 candidates, this adds close to 7 minutes to the candidate generation.

This is something we should be able to parallelize easily -- (8000 * 0.05s) / 192 = 2s, which seems more than acceptable to me.


---

### Comment by bangtianliu

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 245

**Comment:**

> > I did a measurement, the linking time is around 0.047 seconds for linking matmul and attention specs. Maybe just skip linking when the number of td specs is 1.
> 
> @bangtianliu with how many candidates generated?
> 
> > With something like 8000 candidates, this adds close to 7 minutes to the candidate generation.
> 
> This is something we should be able to parallelize easily -- (8000 * 0.05s) / 192 = 2s, which seems more than acceptable to me.

The 0.047 seconds is the overhead for a single linking operation (i.e., one candidate). The second test above measures the total time for 20 candidates.

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 270

**Comment:**

You can put this micro-optimization into `link_tuning_specs` to simplify the logic and make this testable.

Actually, do we need this at all? The `continue` a few lines above already handles this case, doesn't it?

---

### Comment by bangtianliu

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 270

**Comment:**

We do need it, as this condition is caused by `determine_td_specs_to_link`, which will exclude the starter spec if all of its matchers are covered by a tuner-generated candidate spec. 

---

### Comment by Max191

**File:** `tuner/tuner/libtuner.py`

**Line:** 350

**Comment:**

Since this isn't part of the simple example anymore, please rename to `starter-td-spec`

---

### Comment by Max191

**File:** `tuner/tuner/libtuner.py`

**Line:** 351

**Comment:**

Since it is a path, can you use `Path` for the type instead of `str`?

---

### Comment by Max191

**File:** `tuner/tuner/libtuner.py`

**Line:** 353

**Comment:**

Can you add a short statement that the newly generated candidates take precedence over starter specs to the `help` description? Users won't want to look into the code and find the above comments.

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 246

**Comment:**

```suggestion
        # If starter td spec is not provided, use the generated td spec directly.
```

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 276

**Comment:**

Why did you change this? Passing the larger context around in the tuner code seems fine to me

---

## PR #1136: [tuner] merge default td specs

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/1136
**State:** MERGED

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 275

**Comment:**

Why `combine` instead of `merge`? I think we should stick with consistent naming if we can

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 313

**Comment:**

I don't think this is covered by any tests?

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 307

**Comment:**

I think this may be very annoying to reproduce if it ever fails. Why not serialize it to disk and put under a temp directory? 

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 308

**Comment:**

```suggestion
    return ir.Module.parse(result.stdout, tuner_ctx.mlir_ctx)
```

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 275

**Comment:**

Could you explain what this function does as a comment so that it's easy to understand the difference between combine/link/merge?

---

### Comment by kuhar

**File:** `tuner/tuner/common_test.py`

**Line:** 347

**Comment:**

We shouldn't need to repeat the same tests for the logic already tested in IREE. Here, I'd only check that that there are no nested modules and that there is a `__kernel_config` op.

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 274

**Comment:**

This should be a docstring

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 294

**Comment:**

This should be a docstring

---

### Comment by kuhar

**File:** `tuner/tuner/common_test.py`

**Line:** 267

**Comment:**

Could we reduce duplication by providing the same module twice?

---

### Comment by kuhar

**File:** `tuner/tuner/common_test.py`

**Line:** 341

**Comment:**

This assert should be inside the `with` statement

---

### Comment by kuhar

**File:** `tuner/tuner/common_test.py`

**Line:** 315

**Comment:**

Can you explain why you expect this to fail?

---

### Comment by kuhar

**File:** `tuner/tuner/common_test.py`

**Line:** 324

**Comment:**

```suggestion
        # by the `iree_codegen.tuning_spec_with_default_entrypoint` attribute.
```

---

### Comment by kuhar

**File:** `tuner/tuner/common_test.py`

**Line:** 313

**Comment:**

Do we need these two named sequences to make it fail because of missing __kernel_config?

---

## PR #1124: [Tuner] remove default attribute for the place holder tuning spec

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/1124
**State:** MERGED

### Comment by kuhar

**File:** `tuner/tuner/libtuner_test.py`

**Line:** 191

**Comment:**

Can you put this in a separate PR? Let's not mix these two things

---

### Comment by bangtianliu

**File:** `tuner/tuner/libtuner_test.py`

**Line:** 191

**Comment:**

Sure


---

## PR #815: [tuner] add the output file

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/815
**State:** MERGED

### Comment by kuhar

**File:** `tuner/examples/simple/simple_tuner.py`

**Line:** 78

**Comment:**

This appears to be a private function (it starts with an underscore), so we shouldn't call it directly in the current form.

---

### Comment by kuhar

**File:** `tuner/examples/simple/simple_tuner.py`

**Line:** 168

**Comment:**

We should write these as they become available, not at the very end. This is in case something fails / we interrupt the tuner -- in these situations partial results are still useful.

---

### Comment by kuhar

**File:** `tuner/examples/simple/simple_tuner.py`

**Line:** 78

**Comment:**

Also, can we give it a more descriptive name than `output`? It would not be clear to what to expect based on this name. Maybe `summary.log`?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 808

**Comment:**

Nice catch! I wish we had a test for this too...

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 90

**Comment:**

This is not a directory -- it's a path to a file

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 730

**Comment:**

This can cause division by zero, no? Can we add a unit test for this?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 730

**Comment:**

Why are we using `Any` for the element type?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 884

**Comment:**

What happens when we use the root logger here? Why do we have to explicitly log via the logger in the context?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner_test.py`

**Line:** 185

**Comment:**

Missing testcase: empty input list

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner_test.py`

**Line:** 29

**Comment:**

This appears unused

---

### Comment by bangtianliu

**File:** `tuner/tuner/libtuner.py`

**Line:** 730

**Comment:**

No, compiled_candidates will never be empty. Even if all attempts fail, it will still be a list containing only None values.

---

### Comment by bangtianliu

**File:** `tuner/tuner/libtuner.py`

**Line:** 884

**Comment:**

if we use root, it will not output to the summary log.

---

### Comment by bangtianliu

**File:** `tuner/tuner/libtuner_test.py`

**Line:** 29

**Comment:**

It does use for test_select_best_benchmark_results.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 884

**Comment:**

Why is that? I thought that if we register the logging handler for summary, it will be attached to the root logger.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 730

**Comment:**

I can see this happening if we for whatever reason select the max number of candidates to generate to 0

---

### Comment by bangtianliu

**File:** `tuner/tuner/libtuner.py`

**Line:** 884

**Comment:**

Yes, if we use root, it will go to the root logger but not to the summary log. Using tuner_context.logger can ensure that it goes to both the root and summary loggers. Not sure, I guess it is relevant to context management.  

---

### Comment by bangtianliu

**File:** `tuner/tuner/libtuner.py`

**Line:** 730

**Comment:**

Sure, I will see how to support this case. 

---

### Comment by kuhar

**File:** `tuner/examples/simple/simple_tuner.py`

**Line:** 125

**Comment:**

No need for fstring, we don't print any variables here. Also, the grammar is weird. The summary is not ready at this point, so how about we just print the same thing as the print above?

---

### Comment by kuhar

**File:** `tuner/examples/simple/simple_tuner.py`

**Line:** 157

**Comment:**

Same here

---

### Comment by kuhar

**File:** `tuner/examples/simple/simple_tuner.py`

**Line:** 177

**Comment:**

```suggestion
        print("Check the summary in:")
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 898

**Comment:**

Do we still need this?

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 50

**Comment:**

I don't think we need this helper, users can access the logger directly.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 100

**Comment:**

Do we need to keep track of the summary log here? I'd think that the simple tuner has the path and that `libtuner` doesn't have to know about it

---

### Comment by bangtianliu

**File:** `tuner/tuner/libtuner.py`

**Line:** 100

**Comment:**

Yeah, good point!

---

## PR #789: [tuner] Add BaselineResultHandler class

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/789
**State:** MERGED

### Comment by Max191

**File:** `tuner/tuner/libtuner.py`

**Line:** 1065

**Comment:**

nit: Use `logging.warning` here instead of `logging.info`.

---

### Comment by Max191

**File:** `tuner/tuner/libtuner.py`

**Line:** 1029

**Comment:**

Since we are running essentially the same code with just different candidates, could you outline this code into a separate function that takes the list of candidate indices, and returns the results of the benchmarking? Then you can call the same function for all 3 benchmark sets here.

i.e., for the baseline, the indices would be `[0] * len(args.devices)`, and for the main benchmarks the indices would be `[i for i in compiled_candidates if i != 0]`

---

### Comment by Max191

**File:** `tuner/tuner/libtuner.py`

**Line:** 891

**Comment:**

nit: This can probably be `logging.debug`.

---

### Comment by bangtianliu

**File:** `tuner/tuner/libtuner.py`

**Line:** 1029

**Comment:**

> Since we are running essentially the same code with just different candidates, could you outline this code into a separate function that takes the list of candidate indices, and returns the results of the benchmarking? Then you can call the same function for all 3 benchmark sets here.
> 
> i.e., for the baseline, the indices would be `[0] * len(args.devices)`, and for the main benchmarks the indices would be `[i for i in compiled_candidates if i != 0]`

Good point! Will do.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 734

**Comment:**

When writing comments, focus on **why** we are doing certain things, not **what** the code is doing. The **what** is usually redundant because it carries the same information content as the code.
```suggestion
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 764

**Comment:**

```suggestion
    task_list = [
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 756

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

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 889

**Comment:**

```suggestion
    # Benchmarking baselines again to check for performance regressions. These may indicate machine instability, overheating, etc.
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 904

**Comment:**

```suggestion
    ), "Device ID mismatch between baseline runs."
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 908

**Comment:**

What happens when benchmarking fails altogether and there's no time?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 921

**Comment:**

```suggestion
```

---

### Comment by Max191

**File:** `tuner/tuner/libtuner.py`

**Line:** 905

**Comment:**

Is this warning possible to run? I think it can be removed.

---

### Comment by bangtianliu

**File:** `tuner/tuner/libtuner.py`

**Line:** 905

**Comment:**

This was added to address Jakub's comment 
> What happens when benchmarking fails altogether and there's no time?

It addresses the corner case where benchmarking completely fails for a specific device ID (e.g., when one device is broken). While I haven't encountered this scenario, I think it's better to keep this safeguard in place.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 745

**Comment:**

Use proper punctuation.

```suggestion
    # Perform benchmarking.
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 885

**Comment:**

This seems to exceed the column limit, doesn't the formatter complain about it?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 903

**Comment:**

Have you tested this code? Make sure you exercise the failure conditions

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 905

**Comment:**

I think the correct way to handle this to issue a warning when any of the baseline benchmarks fail, but more importantly try to recover from that: try to use the time from the other baseline run, if that succeeded.

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 357

**Comment:**

Why do we want to remove the type hints?

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 371

**Comment:**

also here

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 227

**Comment:**

Based on the function name alone, it's impossible to tell what this code does... Could we call it something like: `get_valid_benchmark_results` and add a one-line docstring explaining what is considered valid?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 246

**Comment:**

Should we check that the device IDs are unique? Otherwise this will result in data loss.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 253

**Comment:**

I think it's better to expand this check inline, the helper function doesn't seem to help that much -- it's just a comparison

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 269

**Comment:**

Similar here, it's impossible to tell what this function does based on the name alone. We need a better name and some comment with an explanation.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 1028

**Comment:**

Should we put baseline benchmarking into it's own function that return a map from devices to results?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner_test.py`

**Line:** 262

**Comment:**

```suggestion
    assert not result
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner_test.py`

**Line:** 270

**Comment:**

```suggestion
    assert result
```

---

### Comment by bangtianliu

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 357

**Comment:**

It triggers mypy warning notes:
> By default the bodies of untyped functions are not checked, consider using --check-untyped-defs 



---

### Comment by bangtianliu

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 371

**Comment:**

same

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 357

**Comment:**

make this function typed then?

---

### Comment by bangtianliu

**File:** `tuner/tuner/libtuner.py`

**Line:** 246

**Comment:**

Yeah, good point!

---

### Comment by bangtianliu

**File:** `tuner/tuner/libtuner.py`

**Line:** 1028

**Comment:**

No, since the baseline results are used later on: https://github.com/nod-ai/shark-ai/blob/main/tuner/tuner/libtuner.py#L930.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 237

**Comment:**

Based on the function name, we can't tell if `True` indicates a success or a failure. (That's why we have the `LogicalResult` type in LLVM, BTW!).

I think we could use this name instead:
```suggestion
def are_benchmark_devices_unique(baseline_results: list[BenchmarkResult]) -> bool:
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 243

**Comment:**

```suggestion
    return len(baseline_results) == len(set(map(lambda r: r.device_id, baseline_results))
```
?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 271

**Comment:**

What is the result of this function?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 272

**Comment:**

```suggestion
                f"First baseline time = {first_baseline_time}, Second baseline time = {second_baseline_time}, "
```

Also, what's the time unit?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 264

**Comment:**

What's the time unit? We should make it a suffix of the variable name

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 1016

**Comment:**

Maybe move this uniqueness check to `map_baseline_by_device`? This function is already quite long.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 876

**Comment:**

It's not obvious what are the keys and what are the keys and the value. Can you either rename it to something like `self.device_to_run_results` or add a comment?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 881

**Comment:**

Why not store the whole results instead of just the times? Even if it didn't succeed, I think it might be useful to keep this information around instead of silently discarding it. The issue with discarding it here is that the results no longer align, i.e., you can't tell which of the runs succeeded.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 886

**Comment:**

What is the time unit? We should make it be a part of the function name, IMO. 
https://users.ece.cmu.edu/~eno/coding/CppCodingStandard.html#units
https://ruudvanasseldonk.com/2022/03/20/please-put-units-in-names

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 864

**Comment:**

What does it mean for this to be valid? Can you add a docstring?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 903

**Comment:**

```suggestion
        Returns a map from candidate_id to its speedup ratio.
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 909

**Comment:**

Why not use `get_average_result` function that you defined above?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 913

**Comment:**

Can we use an early return instead?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 1086

**Comment:**

Why not move it to the new class? It already knows how to deal with baseline results.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 1065

**Comment:**

This can also be moved to the new class IMO.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner_test.py`

**Line:** 321

**Comment:**

We can test multiple methods in one function to avoid repeating the same setup code. Not every function has to have its own test function.

---

### Comment by kuhar

**File:** `tuner/examples/simple/simple_tuner.py`

**Line:** 129

**Comment:**

Can you submit a separate PR for these unrelated changes?

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 371

**Comment:**

Also this: we can land this separately so that we don't have to wait for the benchmarking stuff to finalize.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 753

**Comment:**

The return type is missing

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 805

**Comment:**

Why do we keep the benchmark time only? I asked about this here but the discussion was resolved without any reply: https://github.com/nod-ai/shark-ai/pull/789#discussion_r1919429834

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 812

**Comment:**

This can be a static method, `self` is not used: https://docs.python.org/3/library/functions.html#staticmethod

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 838

**Comment:**

```suggestion
        if self.get_valid_time_ms(device_id):
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 846

**Comment:**

Can you add a docstring explaining what is being returned?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 824

**Comment:**

`self` is not used, this can be a staticmethod

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 820

**Comment:**

Can we add a method to `BenchmarkResult` that checks if that result is valid? Invalid benchmarks being represented with infinite values is an implementation detail that shouldn't leak here.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 824

**Comment:**

The function name would suggest to me that this tell whether the whole run was valid, not how many times device finished. 

Maybe we don't need this function anymore and can use `get_valid_time_ms` directly?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 860

**Comment:**

```suggestion
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 858

**Comment:**

```suggestion
        Return True iff at least a valid (finite) baseline time recorded.
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 868

**Comment:**

Maybe we can remove this function and do `num_successful_runs(device_id) != 0` instead?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 879

**Comment:**

nit: Use proper punctuation
```suggestion
            # Use the candidate time directly when no baselines are available.
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 885

**Comment:**

nit: Use proper punctuation
```suggestion
        # Calculate the fallback baseline as the average of all valid times across devices.
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 882

**Comment:**

I think in the absence of any baselines, the speedup should be 1.0, no?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 850

**Comment:**

The code that uses this function already prints a warning. Maybe we can drop this message?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 1020

**Comment:**

I don't think it's obvious what it means for baseline results to be not valid at this point. How about this?
```suggestion
        logging.warning("Baseline run failed")
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 1023

**Comment:**

I think we should move this check to the code that relies on baseline devices being unique. This code is no longer in this function.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 1050

**Comment:**

Same here

---

### Comment by bangtianliu

**File:** `tuner/tuner/libtuner.py`

**Line:** 882

**Comment:**

No, raw time data itself is used for sorting and obtaining top candidates.

---

### Comment by bangtianliu

**File:** `tuner/tuner/libtuner.py`

**Line:** 838

**Comment:**

```python
    def get_average_result_ms(self, device_id: str) -> Optional[float]:
        valid_times = self.get_valid_time_ms(device_id)
        if valid_times:
            return sum(valid_times) / len(valid_times)
        return None
```
The complete code is shown above, `valid_times `is reused multiple times and thus maybe it's better left as is.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 882

**Comment:**

Can we use some other metric instead like the average across all the other devices?

---

### Comment by bangtianliu

**File:** `tuner/tuner/libtuner.py`

**Line:** 882

**Comment:**

Under this condition, the context is all the baseline data (even on other devices) is not available. 



---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 845

**Comment:**

This doesn't really explain what we consider a regression
```suggestion
        Returns a list of device IDs where regressions were detected.
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 861

**Comment:**

```suggestion
        Returns True iff at least one valid (finite) baseline time was recorded.
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 875

**Comment:**

Can you explain what you understand by speedup? Is it `run_time / avg_baseline_time` ?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 1050

**Comment:**

I'd change this API such that `get_top_candidates` returns you **all** the candidates ordered by speedup, as well as the speedup values themselves. This way we don't leak implementation details around what is a speedup. In the code here, we can decide how many of the op candidates to print out to the log.

---

### Comment by bangtianliu

**File:** `tuner/tuner/libtuner.py`

**Line:** 875

**Comment:**

Yes, it is  the runing time divided by baseline time.
Here, I am following previous Max's design to maintain consistency of sorting numbers in ascending order based on either speedup or time.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 886

**Comment:**

typo
```suggestion
        If no valid baseline times are available, the candidate's runtime is used directly as:
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 888

**Comment:**

This looks wrong because for very short runtimes this can end up being less then other `candidate_runtime / avg_baseline_time` and dominate the list of top candidates. Please add a test for this.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 1057

**Comment:**

Speedup result should be calculated inside get_top_candidates. We should also rename this to something else now that it doesn't produce top N values and, instead, returns the full list. See the full suggestion here: https://github.com/nod-ai/shark-ai/pull/789#discussion_r1920717256

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner_test.py`

**Line:** 294

**Comment:**

Please remove debug prints

---

### Comment by bangtianliu

**File:** `tuner/tuner/libtuner.py`

**Line:** 888

**Comment:**

There are no mixing of results due to the presence of a fallback baseline. If no valid baseline times exist on all devices, we sort based on raw runtime instead fully. Relevant test cases for this scenario have already been added.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 913

**Comment:**

Add a time unit to `baseline_avg`. Also AFAICT, `get_average_result_ms` never returns a non-finite time, does it?

---

### Comment by bangtianliu

**File:** `tuner/tuner/libtuner.py`

**Line:** 913

**Comment:**

Yeah, good catch.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 1064

**Comment:**

We should only have to call `baseline_handler.sort_candidates_with_speedup`, it can calculate the speedup internally. This `speedup_result` variable is only ever used in this function. See the two prior comments that made the same suggestion: https://github.com/nod-ai/shark-ai/pull/789#discussion_r1920717256 and https://github.com/nod-ai/shark-ai/pull/789#discussion_r1920819793 .

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 1069

**Comment:**

Can you add the time unit to the variable name? I'd call it `candidate_id_to_time_ms`. 

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 1073

**Comment:**

What's the time unit?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 924

**Comment:**

```suggestion
    def get_candidates_ordered_by_speedup(
```
?

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 1065

**Comment:**

If `get_candidates_ordered_by_speedup` returned `list[tuple[BenchmarkResult, float]]:`, we could get rid of this logic.

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 881

**Comment:**

```suggestion
        Returns:
```

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 883

**Comment:**

We shouldn't change the return value type based on the input -- can we always return `list[tuple[BenchmarkResult, float]]`. When no baselines are available, you can make the speedup be 1.0, for example, as a placeholder value.

---

## PR #770: [Tuner] Fix context management

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/770
**State:** MERGED

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 184

**Comment:**

TuningClient shouldn't define these

---

### Comment by kuhar

**File:** `tuner/examples/dispatch/dispatch_tuner.py`

**Line:** 114

**Comment:**

I don't understand why we define these outside of the tuner class?

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 50

**Comment:**

Can we add type hints for these?

---

### Comment by kuhar

**File:** `tuner/examples/test/tuner_test.py`

**Line:** 8

**Comment:**

Do we need this import? I don't see it used.

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 33

**Comment:**

I don't think these comments help here

```suggestion
    mock_logger = MagicMock(spec=Logger)
```

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 7

**Comment:**

Why do we need this import? 

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 13

**Comment:**

Why do we need this?

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 53

**Comment:**

Can we add type hints for `exc_type`, `exc_value`, and `traceback`? Also, can we make this return whatever the nested exit returns?

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 55

**Comment:**

```suggestion
    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return self.mlir_ctx.__exit__(exc_type, exc_value, traceback)
```

---

### Comment by kuhar

**File:** `tuner/tuner/common_test.py`

**Line:** 28

**Comment:**

Also here

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_constraints_test.py`

**Line:** 30

**Comment:**

Also here

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_parser_test.py`

**Line:** 32

**Comment:**

Also here

---

### Comment by kuhar

**File:** `tuner/tuner/libtuner.py`

**Line:** 1061

**Comment:**

Do we need this after your other changes?

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 44

**Comment:**

```suggestion
        self.logger: logging.Logger = logger or logging.getLogger("tune")
```

I don't see how this comments helps

---

## PR #669: [tuner]: use translation_info binding

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/669
**State:** MERGED

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 61

**Comment:**

The pipeline should be llvmgpuvectordistribution. Also everywhere else

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 64

**Comment:**

Can you put the gpu_pipeline_options key name as a constant somewhere in `common.py`?

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 156

**Comment:**

Can we print the whole translation info attr here?

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 225

**Comment:**

Also here

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 292

**Comment:**

Also here

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 377

**Comment:**

Also here

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 449

**Comment:**

Also here

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 61

**Comment:**

this is not the right pipeline

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 129

**Comment:**

also here

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 210

**Comment:**

also here

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 273

**Comment:**

also here

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 339

**Comment:**

also here

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 403

**Comment:**

also here

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 491

**Comment:**

also here

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 122

**Comment:**

```suggestion
GPU_PIPELINE_OPTIONS_KEY = "gpu_pipeline_options"
```

---

### Comment by kuhar

**File:** `tuner/tuner/common_test.py`

**Line:** 89

**Comment:**

wrong pipeline

---

### Comment by kuhar

**File:** `tuner/tuner/common_test.py`

**Line:** 229

**Comment:**

wrong pipeline

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_constraints.py`

**Line:** 248

**Comment:**

this one is correct

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_parser_test.py`

**Line:** 55

**Comment:**

wrong pipeline

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_parser_test.py`

**Line:** 86

**Comment:**

wrong pipeline

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_parser_test.py`

**Line:** 116

**Comment:**

wrong pipeline

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 63

**Comment:**

```suggestion
    pipeline_options = iree_gpu.PipelineOptionsAttr.get(prefetch_shared_memory=True)
```

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 64

**Comment:**

```suggestion
    pipeline_options_dict = ir.DictAttr.get(
```

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 63

**Comment:**

also everywhere else

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 64

**Comment:**

also everywhere else

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 325

**Comment:**

```suggestion
    translation_info = {configuration.translation_info}
```

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 70

**Comment:**

You can create a helper function to build config_dict for you, similar to the helper function `get_lowering_config`

---

### Comment by bangtianliu

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 70

**Comment:**

Good suggestion! will do it.

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 167

**Comment:**

I think it would be easier to always require `waves_per_eu` to be an int

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 167

**Comment:**

Could you add a comment explaining what the purpose of this function is? Maybe show a piece of IR and where this attribute fits?

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 166

**Comment:**

This doesn't generate a dictionary **in** translation info. I'd say something like this instead:
```suggestion
# Generate a config dictionary used in the translation_info attribute.
```

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 187

**Comment:**

I don't think we need this, the usage pattern is pretty obvious

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 181

**Comment:**

Also this is now outdated

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 176

**Comment:**

```suggestion
                    {gpu_pipeline_options = #iree_gpu.pipeline_options<...>,
```

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 175

**Comment:**

```suggestion
                    pipeline = LLVMGPUVectorDistribute workgroup_size = [512, 1, 1] subgroup_size = 64,
```

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 174

**Comment:**

```suggestion
                    {gpu_pipeline_options = #iree_gpu.pipeline_options<...>,
```

---

## PR #662: [tuner]: use property function from iree lowering config python binding

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/662
**State:** MERGED

### Comment by kuhar

**File:** `tuner/tuner/dispatch_parser_test.py`

**Line:** 83

**Comment:**

nit: Will these arrays fit on the same line now?

---

## PR #629: [tuner]: use lowering config binding

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/629
**State:** MERGED

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 57

**Comment:**

We can add a helper functions to `tuner_ctx.type` like `tuner_ctx.type.getI32(8)` etc. This should save a lot of typing in code like this.

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 65

**Comment:**

We can create a helper function that accepts python types and returns a lowering config attribute to avoid having to create these dictionaries by hand.

For example, this could be something like: `common.getLoweringConfig(mma_kind=mma_attr, workgroup=[8,8,0], reduction=[0, 0, 8], subgroup_m_count=16, subgroup_n_count=16)`. 

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 60

**Comment:**

The tile sizes are incorrect here: `[8, 8, 8]` in the single-array format should be `workgroup=[8, 8, 0], reduction = [0, 0, 8]`

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 118

**Comment:**

I wouldn't make these properties because this can fail when the lowering config doesn't contain that attribute

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 134

**Comment:**

Adding these is a good idea, but we should make sure we have tests that exercise this code

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 65

**Comment:**

This could also be new method defined in python bindings, but I think it will be easier to define this in python as a new helper.

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 144

**Comment:**

We should print the lowering config attribute here instead of extracting stuff out of it.

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 93

**Comment:**

I don't these attributes have two levels of nesting -- can you check with the attributes that come out of iree?

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 155

**Comment:**

This list of attributes is not 'static' in the sense that some lowering configs can have a subset or a superset of these. Instead, can we handle this via kwargs (essentially a dictionary with a list of known attrs)?

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 123

**Comment:**

Don't use private methods (that start with an underscore). For Dictattr, I think you can use the `in` and `[...]` operators

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 123

**Comment:**

Also, I think this pattern would be supported as `my_dict.get(key, default)`, but I'm not sure if the python bindings support that

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 165

**Comment:**

I think it would be better to do the conversion based on known keys instead of just values. We can check if the passes value is an attribute (nothing to do then), or dispatch based on the type otherwise (I think you can use a `match` statement for that ([some examples](https://benhoyt.com/writings/python-pattern-matching/)).

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 156

**Comment:**

Please add tests for this and make sure the returned attributes roundtrip with the helpers from `Configuration`

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_parser.py`

**Line:** 174

**Comment:**

```suggestion
        ow, oc, _ic = configuration.tilesize_workgroup()
```

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_constraints.py`

**Line:** 251

**Comment:**

Use proper casing and punctuation.

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_parser_test.py`

**Line:** 130

**Comment:**

This formatting looks weird -- I'd expect the `0, 0, 0` to fit on the same line since `16, 0, 0` is longer and fits

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 121

**Comment:**

If these helpers are useful enough, we could also add them to the iree python bindings. I think these helpers are also fine, but I'd rather see them as free functions instead of methods in `Configuration`. They don't need to know about `subgroup_size` and workgroup_size` after all -- the only input should the the lowering config attr.

---

### Comment by bangtianliu

**File:** `tuner/tuner/common.py`

**Line:** 121

**Comment:**

Good point, I will send another IREE PR to support this. 

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 167

**Comment:**

IMO we can assert instead -- this is internal logic

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 165

**Comment:**

I'd make a single assignment to `lowering_config[key]` just after the `match` and introduce a new local variable `promoted_value = value` that gets modified inside the match

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 126

**Comment:**

There shouldn't be methods of `Configuration`. We can have them as free functions instead.

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 65

**Comment:**

Can we print the attributes directly? `config.lowering_config.attributes["workgroup"]`

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 128

**Comment:**

```suggestion
def get_workgroup_tile_sizes(config: Configuration) -> list[int]:
```

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 135

**Comment:**

```suggestion
def get_reduction_tile_sizes(config: Configuration) -> list[int]:
```

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 185

**Comment:**

This is pretty obvious, I don't think we need this comment but could use an empty line

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 168

**Comment:**

Don't check the type in the assert -- if the previous if doesn't match, `assert False`

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 46

**Comment:**

Could we extract workgroup and tile sizes from the `configuration` directly and remove these two function arguments?

---

### Comment by bangtianliu

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 46

**Comment:**

yeah, good catch!

---

### Comment by bangtianliu

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 46

**Comment:**

Oh, actually no. The tile sizes from the configuration are not used directly https://github.com/nod-ai/shark-ai/blob/main/tuner/tuner/dispatch_parser.py#L41, so these two function arguments cannot be removed. 

---

### Comment by bangtianliu

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 65

**Comment:**

Similar to the reply below, the tile sizes are generated by functions that vary based on the operators, as shown here: https://github.com/nod-ai/shark-ai/blob/main/tuner/tuner/dispatch_parser.py#L23toL40.

---

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 46

**Comment:**

This definitely should be removed eventually, but we don't have to put everything in the same PR. Could you add a TODO so that we don't forget about this?

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_constraints.py`

**Line:** 257

**Comment:**

Why aren't we using the new `get_lowering_config` helper here?

---

### Comment by bangtianliu

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 46

**Comment:**

Sure, will do. 

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_constraints.py`

**Line:** 225

**Comment:**

We shouldn't be creating a tuner context here. If it's needed, it should be passed as a function argument. This is likely to break object comparisons if we accidentally end up with two contexts at once.

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_constraints.py`

**Line:** 223

**Comment:**

use the int type from the context

---

## PR #626: [tuner]: retire data class GPUPipelineOptions, use iree_gpu.PipelineOptionsAttr. 

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/626
**State:** MERGED

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 137

**Comment:**

You can compare `pipeline_options` with a freshly constructed attribute with all default values

---

## PR #605: [tuner]: use iree_gpu.MMAIntrinsic and iree_gpu.MMAAttr 

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/605
**State:** MERGED

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen_test.py`

**Line:** 49

**Comment:**

Use the enum instead of passing a string. Also everywhere else

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 93

**Comment:**

you can do `mma_intrinsic.mma`

---

### Comment by kuhar

**File:** `tuner/tuner/common_test.py`

**Line:** 8

**Comment:**

Good catch!

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_constraints.py`

**Line:** 33

**Comment:**

Let's try to avoid repeated CAPI calls here -- we can hoist mnk into a local variable so that we only call this once

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_constraints.py`

**Line:** 162

**Comment:**

Let's try to avoid text-based processing here and instead enumerate all available intrinsics and select the enum that matches these parameters

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_constraints.py`

**Line:** 38

**Comment:**

Instead of a for loop, can we make it a local variable? Double for loop with just one iteration is confusing to me -- it took me a good minute to figure out what we are trying to do here.

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_constraints.py`

**Line:** 153

**Comment:**

Here we can construct mma directly from the intrinsic without creating an mmaintrinsicattr first

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_constraints.py`

**Line:** 42

**Comment:**

I'm still confused by this loop structure. Why do we need two `for`s?

---

### Comment by bangtianliu

**File:** `tuner/tuner/dispatch_constraints.py`

**Line:** 42

**Comment:**

oh, the code should be updated, sorry.

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_constraints.py`

**Line:** 39

**Comment:**

```suggestion
                intrinsic_m == m,
                intrinsic_n == n,
                intrinsic_k == k,
            )
            for m, n, k in mnk_shapes
```

---

## PR #586: [tuner]: use python binding to select mma intrinsics

**URL:** https://github.com/nod-ai/amd-shark-ai/pull/586
**State:** MERGED

### Comment by kuhar

**File:** `tuner/tuner/candidate_gen.py`

**Line:** 541

**Comment:**

This assertion message doesn't match the code. I'd say something like `Expected one executable variant op`. Saying `one op` only is misleading.

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 137

**Comment:**

Why is this optional? I think we can make it a required argument.

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_constraints.py`

**Line:** 25

**Comment:**

Also here

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_constraints.py`

**Line:** 142

**Comment:**

also here

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 139

**Comment:**

This is not the best variable name. Consider changing it to something like `available_mma_intrinsics`

---

### Comment by kuhar

**File:** `tuner/tuner/common.py`

**Line:** 150

**Comment:**

I don't think we need to check that `available_mma_intrinsics` is non-empty?

---

### Comment by kuhar

**File:** `tuner/tuner/common_test.py`

**Line:** 113

**Comment:**

We should make sure the new logic is covered by tests. Please add/modify existing tests to pass in the list of available intrinsics.

---

### Comment by kuhar

**File:** `tuner/tuner/dispatch_constraints_test.py`

**Line:** 42

**Comment:**

Doesn't this change the reason for no solutions? Now it can't produce any because there are no MFMAs to choose from.

---


---

**Total PRs with comments:** 58
**Total comments:** 629
