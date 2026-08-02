# Chronos-2 injection source audit

The installed Chronos implementation is the authority for this audit. `predict_df` converts every non-ID/non-timestamp/non-target context column into a past covariate. `validate_and_prepare_single_dict_task` stacks target followed by each past covariate, producing one row per variable and 336 columns along time. `Chronos2Model._prepare_patched_context` calls `self.instance_norm(context)` before patch construction. `InstanceNorm.forward` computes `nanmean` and RMS scale with `dim=-1`, so normalization is independently applied to every item/variable row along its 336-step time dimension. A constant repeated covariate becomes `(x - loc) / scale` before patching. Mathematically this is zero, but this audit records the actual float32 reduction result rather than assuming exact cancellation; nonzero roundoff can be amplified when the computed scale is also tiny.

## Actual installed paths and hashes

```json
{
  "python": "3.11.15 (main, Mar 11 2026, 17:20:07) [GCC 14.3.0]",
  "platform": "Linux-6.8.0-85-generic-x86_64-with-glibc2.35",
  "torch": "2.11.0+cu128",
  "cuda": "12.8",
  "chronos_package_path": "/home/zhaopp/miniconda3/envs/torch/lib/python3.11/site-packages/chronos",
  "chronos_package_version": "2.2.2",
  "chronos_model_path": "/home/zhaopp/.cache/huggingface/hub/models--amazon--chronos-2/snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c",
  "chronos_model_config_sha256": "ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031",
  "instance_norm_eps": 1e-05,
  "instance_norm_use_arcsinh": true,
  "source_records": {
    "predict_df": {
      "source_file": "/home/zhaopp/miniconda3/envs/torch/lib/python3.11/site-packages/chronos/chronos2/pipeline.py",
      "functions": "Chronos2Pipeline.predict_df",
      "source_sha256": "4ea9776692c24fda7946755aced31187d09ba45cca0dd10227fca01e509e42e9"
    },
    "dataframe_conversion": {
      "source_file": "/home/zhaopp/miniconda3/envs/torch/lib/python3.11/site-packages/chronos/df_utils.py",
      "functions": "convert_df_input_to_list_of_dicts_input",
      "source_sha256": "8bfa57ebd5d2dcb397d720295abf70e4ceb493b6f589785befcff80e7b99936f"
    },
    "covariate_tensor_conversion": {
      "source_file": "/home/zhaopp/miniconda3/envs/torch/lib/python3.11/site-packages/chronos/chronos2/dataset.py",
      "functions": "validate_and_prepare_single_dict_task",
      "source_sha256": "a719c8810a9f253940764256a5ab2f6eb9fbe00683948d0c65c4cd1a98ad14b1"
    },
    "batch_construction": {
      "source_file": "/home/zhaopp/miniconda3/envs/torch/lib/python3.11/site-packages/chronos/chronos2/dataset.py",
      "functions": "Chronos2Dataset._build_batch",
      "source_sha256": "a719c8810a9f253940764256a5ab2f6eb9fbe00683948d0c65c4cd1a98ad14b1"
    },
    "patch_construction": {
      "source_file": "/home/zhaopp/miniconda3/envs/torch/lib/python3.11/site-packages/chronos/chronos2/model.py",
      "functions": "Chronos2Model._prepare_patched_context",
      "source_sha256": "907ddd4f4ccc597f9369a50f526243643366e434193e11bdd6f54cea87d9f001"
    },
    "instance_normalization": {
      "source_file": "/home/zhaopp/miniconda3/envs/torch/lib/python3.11/site-packages/chronos/chronos_bolt.py",
      "functions": "InstanceNorm.forward",
      "source_sha256": "bd40058cb9d48170338c92b0d0f0e7584c7a1956ab2c7b41e2750e35c7be2528"
    }
  }
}
```

## Installed `InstanceNorm.forward`

```python
    def forward(
        self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
            scale = torch.where(scale == 0, self.eps, scale)
        else:
            loc, scale = loc_scale

        scaled_x = (x - loc) / scale

        if self.use_arcsinh:
            scaled_x = torch.arcsinh(scaled_x)

        return scaled_x.to(orig_dtype), (loc, scale)
```
