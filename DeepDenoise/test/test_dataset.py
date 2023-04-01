import pytest
from DeepDenoise.src.dataset import CESTDataset
from pathlib import Path


def test_dataset_len():
    dataset_train = CESTDataset(
        root_dir=Path("test_data"),
        mode="train",
    )
    dataset_val = CESTDataset(
        root_dir=Path("test_data"),
        mode="val",
    )
    dataset_test = CESTDataset(
        root_dir=Path("test_data"),
        mode="test",
    )
    assert dataset_train.__len__() == 7
    assert dataset_val.__len__() == 2
    assert dataset_test.__len__() == 1


def test_dataset_keys():
    dataset_train = CESTDataset(
        root_dir=Path("test_data"),
        mode="train",
    )
    data = dataset_train.__getitem__(0)
    assert "ground_truth" in data.keys()
    assert "noisy" in data.keys()


def test_dataset_shape():
    dataset_train = CESTDataset(
        root_dir=Path("test_data"),
        mode="train",
    )
    data = dataset_train.__getitem__(0)
    assert data["ground_truth"].size()[0] == 42
    assert data["ground_truth"].size()[1] == 128
    assert data["ground_truth"].size()[2] == 128


if __name__ == "__main__":
    pytest.main()
