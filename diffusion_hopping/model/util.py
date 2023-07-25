import torch
import torch.nn as nn
import torch_scatter


def centered_batch(x, batch, mask=None, dim_size=None):
    if mask is None:
        mean = torch_scatter.scatter_mean(x, batch, dim=0)
    else:
        mean = torch_scatter.scatter_mean(
            x[mask], batch[mask], dim=0, dim_size=dim_size
        )
    return x - mean[batch]


def skip_computation_on_oom(return_value=None, error_message=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if error_message is not None:
                        print(error_message)
                    return return_value
                else:
                    raise e

        return wrapper

    return decorator
