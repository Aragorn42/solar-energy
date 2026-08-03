import torch

from stage4_adapter import CoRACorrelationAdapter, MissingAwareCoRAAdapter, UniCATokenAdapter, freeze_backbones, trainable_parameter_names


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


def test_cora_alpha_beta_zero_is_exact_identity():
    adapter = CoRACorrelationAdapter(fusion_dim=8, chronos_dim=24, heads=4, global_hidden=6)
    chronos = torch.randn(2, 5, 24)
    fusion = torch.randn(2, 7, 8)
    torch.testing.assert_close(adapter(chronos, fusion), chronos, rtol=0, atol=0)
    assert adapter.alpha.item() == adapter.beta.item() == 0.0


def test_missing_aware_gate_zero_is_hidden_identity_and_one_is_cora():
    torch.manual_seed(3)
    adapter = MissingAwareCoRAAdapter(fusion_dim=8, chronos_dim=24, gate_feature_dim=4, heads=4, global_hidden=6)
    adapter.gate_mlp[-1].bias.data.fill_(-100.0)
    hidden, fusion, features = torch.randn(2, 5, 24), torch.randn(2, 7, 8), torch.randn(2, 4)
    gated = adapter(hidden, fusion, features)
    torch.testing.assert_close(gated, hidden, rtol=0, atol=1e-5)
    adapter.gate_mlp[-1].bias.data.fill_(100.0)
    torch.testing.assert_close(adapter(hidden, fusion, features), adapter.cora(hidden, fusion), rtol=0, atol=1e-5)


def test_missing_aware_gate_is_window_specific_and_cora_frozen():
    torch.manual_seed(4)
    adapter = MissingAwareCoRAAdapter(fusion_dim=8, chronos_dim=24, gate_feature_dim=4, heads=4, global_hidden=6)
    adapter.gate_mlp[0].weight.data.normal_()
    adapter.gate_mlp[-1].weight.data.normal_()
    features = torch.tensor([[1., 0., 1., 0.], [0., 1., 1., 0.]])
    assert not torch.equal(adapter.gate(features)[0], adapter.gate(features)[1])
    adapter.freeze_cora()
    assert not any(p.requires_grad for p in adapter.cora.parameters())
    assert any(p.requires_grad for p in adapter.gate_mlp.parameters())
