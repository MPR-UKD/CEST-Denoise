import pytest
from torch.utils.data import DataLoader

from DeepDenoise.src.dataloader import CESTDataModule


@pytest.fixture
def data_module():
    return CESTDataModule(
        dir="test_data",
        distribution=[0.7, 0.2, 0.1],
        batch_size=2,
        transform=None,
        noise_std=0.1,
    )


def test_data_module(data_module):
    assert isinstance(data_module.train_dataloader(), DataLoader)
    assert isinstance(data_module.val_dataloader(), DataLoader)
    assert isinstance(data_module.test_dataloader(), DataLoader)


def test_train_data(data_module):
    train_dataloader = data_module.train_dataloader()
    assert len(train_dataloader.dataset) == 5
    for data in train_dataloader:
        assert "ground_truth" in data
        assert "noisy" in data


def test_val_data(data_module):
    val_dataloader = data_module.val_dataloader()
    assert len(val_dataloader.dataset) == 2
    for data in val_dataloader:
        assert "ground_truth" in data
        assert "noisy" in data


def test_test_data(data_module):
    test_dataloader = data_module.test_dataloader()
    assert len(test_dataloader.dataset) == 1
    for data in test_dataloader:
        assert "ground_truth" in data
        assert "noisy" in data


if __name__ == "__main__":
    pytest.main()
