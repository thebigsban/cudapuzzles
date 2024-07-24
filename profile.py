import time
import torch


def time_cuda_function(f, input):
    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)

    for _ in range(5):
        f(input)
    start.record()
    f(input)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

