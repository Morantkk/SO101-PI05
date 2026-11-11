"""Resume must preserve rank-local ordering without decoding completed batches."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/train"))
from data_loader import ResumableDistributedSampler


@pytest.mark.parametrize("world,rank", [(1, 0), (2, 0), (2, 1)])
def test_resume_reads_only_remaining_samples_and_resets_next_epoch(world, rank):
    class Dataset(torch.utils.data.Dataset):
        def __init__(self):
            self.reads = []

        def __len__(self):
            return 41

        def __getitem__(self, index):
            self.reads.append(index)
            return index

    data = Dataset()
    kwargs = dict(num_replicas=world, rank=rank, seed=42, shuffle=True, drop_last=True)
    reference = torch.utils.data.DistributedSampler(data, **kwargs)
    reference.set_epoch(2)
    original = list(reference)
    sampler = ResumableDistributedSampler(data, **kwargs)
    sampler.set_epoch(2, start_index=12)
    loader = torch.utils.data.DataLoader(data, batch_size=4, sampler=sampler, drop_last=True)
    result = torch.cat(list(loader)).tolist()
    expected = original[12:len(original) // 4 * 4]
    assert result == expected
    assert data.reads == expected
    assert len(loader) == len(expected) // 4

    sampler.set_epoch(3)
    reference.set_epoch(3)
    assert list(sampler) == list(reference)
    assert len(sampler) == len(reference)


def test_resume_rejects_invalid_offset():
    sampler = ResumableDistributedSampler(range(10), num_replicas=1, rank=0)
    with pytest.raises(ValueError):
        sampler.set_epoch(0, start_index=11)
    with pytest.raises(ValueError):
        sampler.set_epoch(0, start_index=-1)
