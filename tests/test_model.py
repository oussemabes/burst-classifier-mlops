import torch

from burst_classifier.model import BurstCNN


def test_forward_shape():
    model = BurstCNN(num_classes=3)
    out = model(torch.randn(4, 1, 64, 32))
    assert out.shape == (4, 3)


def test_variable_time_lengths():
    model = BurstCNN(num_classes=3)
    for T in [10, 32, 100]:
        assert model(torch.randn(2, 1, 64, T)).shape == (2, 3)


def test_parameter_count_edge_safe():
    model = BurstCNN(num_classes=3)
    assert model.count_parameters() < 500_000, "Model too large for edge deployment"


def test_predict_proba_sums_to_one():
    model = BurstCNN(num_classes=3)
    model.eval()
    probs = model.predict_proba(torch.randn(8, 1, 64, 32))
    assert probs.shape == (8, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5)


def test_save_load_roundtrip(tmp_path):
    model = BurstCNN(num_classes=3)
    model.eval()
    path = tmp_path / "model.pt"
    model.save(str(path), metadata={"test": True})
    loaded = BurstCNN.load(str(path))
    x = torch.randn(1, 1, 64, 32)
    with torch.no_grad():
        torch.testing.assert_close(model(x), loaded(x))
