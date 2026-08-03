import torch

from stage4_adapter import UniCATokenAdapter, freeze_backbones, trainable_parameter_names


def test_zero_alpha_is_exact_identity_and_tokens_are_not_pooled():
    torch.manual_seed(1)
    adapter = UniCATokenAdapter(fusion_dim=64, chronos_dim=48, heads=4)
    chronos = torch.randn(3, 5, 48)
    fusion = torch.randn(3, 24, 64)
    output = adapter(chronos, fusion)
    torch.testing.assert_close(output, chronos, rtol=0, atol=0)
    assert adapter.alpha.item() == 0.0
    assert adapter.fusion_projection.in_features == 64


def test_only_adapter_remains_trainable():
    chronos = torch.nn.Linear(3, 4)
    fusionsf = torch.nn.Linear(5, 6)
    adapter = UniCATokenAdapter(fusion_dim=8, chronos_dim=24, heads=4)
    freeze_backbones(chronos, fusionsf)
    assert not any(p.requires_grad for p in chronos.parameters())
    assert not any(p.requires_grad for p in fusionsf.parameters())
    names = trainable_parameter_names(adapter)
    assert "alpha" in names
    assert "fusion_projection.weight" in names
    assert names


def test_token_order_can_affect_nonzero_adapter_output():
    torch.manual_seed(2)
    adapter = UniCATokenAdapter(fusion_dim=8, chronos_dim=24, heads=4)
    adapter.alpha.data.fill_(1.0)
    chronos = torch.randn(2, 4, 24)
    fusion = torch.randn(2, 7, 8)
    # Window shuffling changes K/V content for a query window.
    assert not torch.equal(adapter(chronos, fusion), adapter(chronos, fusion.flip(0)))
