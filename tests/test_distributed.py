import pytest
from src.section_3_scaling.distributed import train_distributed

def test_distributed():
    # Run distributed training with 2 processes
    world_size = 2
    try:
        torch.multiprocessing.spawn(
            train_distributed,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
        assert True, "Distributed training completed successfully"
    except Exception as e:
        assert False, f"Distributed training failed: {str(e)}"
