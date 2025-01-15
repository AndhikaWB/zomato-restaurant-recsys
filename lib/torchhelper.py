import json
import polars as pl

from torch import nn


class OutputHook:
    """Get the immediate output of a layer (module) in the model.

    After initializing, call this class with the input tensor (just like
    how you call the original model). The output will be the layer
    output.

    Args:
        model (nn.Module): The PyTorch model.
        module_name (str): Layer (module) name to be hooked. The
            layer must have a `forward` function.
        eval_mode (bool, optional): Set model to eval mode first.
            Defaults to True.
        to_numpy (bool, optional): Convert output to numpy array.
            Defaults to False.

    Raises:
        KeyError: If the module does not exist on the model.
    """

    def __init__(
        self,
        model: nn.Module,
        module_name: str,
        eval_mode: bool = True,
        to_numpy: bool = False,
    ):
        self.model = model
        self.to_numpy = to_numpy

        if eval_mode:
            self.model.eval()

        self.module = self.__get_module(module_name)
        if not self.module:
            raise KeyError(f'Module named "{module_name}" is not found!')

    def __get_module(self, module_name) -> nn.Module | None:
        modules = dict()

        for name, module in self.model.named_modules():
            modules[name] = module

        return modules.get(module_name, None)

    def __call__(self, *args, **kwargs):
        outputs = []

        # Hook should always return nothing (None)
        # Anything else will modify the output
        def hook(module, input, output):
            if self.to_numpy:
                outputs.append(output.numpy(force=True))
            else:
                outputs.append(output.detach())

        # Register forward hook then call the model
        # Handler will always be removed again, even on error
        handler = self.module.register_forward_hook(hook)
        try:
            self.model(*args, **kwargs)
        finally:
            handler.remove()

        if len(outputs) == 1:
            return outputs[0]
        return outputs


class SimpleLogger:
    """Logger to track value changes (e.g. can be called on each epoch).

    Usages:
        `logger.log('acc', value)`: Log the current value of acc. Call
            it multiple times to log a new value every time.
        `logger.get('acc')`: Return all previously logged acc data.
        `logger.clear('acc')`: Clear all acc data.
        `logger.mean('acc')`: Return the mean of all acc data.
        `logger.mean('acc', result_to = 'avg_acc')`: Save the mean
            result to avg_acc, then clear the original acc data.
    """

    def __init__(self):
        self.__history = dict()

    def get(self, key: str = None) -> list:
        """Get the history of something (by passing a key).

        If no key is passed, the whole history will be returned.
        """
        if key:
            return self.__history[key]
        return self.__history

    def log(self, key: str, value):
        """Log the current value of something (e.g. accuracy).

        To add a new value, just call this function again. Get all
        previously logged values by calling the `get` function.
        """
        if key in self.__history:
            self.__history[key].append(value)
        else:
            self.__history[key] = [value]

    def clear(self, key: str = None):
        """Clear a key and its values from history.

        If no key is passed, it will clear everything.
        """
        if key:
            self.__history.pop(key, None)
        else:
            self.__history = dict()

    def to_json(self, path):
        """Save the logger history to a JSON file."""
        with open(path, 'w') as file:
            json.dump(self.__history, file, indent=1)

    def from_json(self, path):
        """Load the logger history from a JSON file.

        Note that the input will not be sanitized first.
        """
        with open(path, 'r') as file:
            self.__history = json.load(file)

    def len(self, key: str):
        return len(self.__history[key])

    def max(self, key: str):
        return max(self.__history[key])

    def min(self, key: str):
        return min(self.__history[key])

    def first(self, key: str):
        return self.__history[key][0]

    def last(self, key: str):
        return self.__history[key][-1]

    def sum(self, key: str, result_to: str = None):
        """Compute the sum of something (e.g. acc).

        Can also save the result to another key (e.g. avg_acc). The
        original key (acc) will be cleared for convenience.
        """
        total = sum(self.__history[key])

        if result_to:
            self.clear(key)
            self.log(result_to, total)

        return total

    def mean(self, key: str, result_to: str = None):
        """Compute the mean of something (e.g. acc).

        Can also save the result to another key (e.g. avg_acc). The
        original key (acc) will be cleared for convenience.
        """
        mean = sum(self.__history[key]) / len(self.__history[key])

        if result_to:
            self.clear(key)
            self.log(result_to, mean)

        return mean


def parameters(model: nn.Module):
    """Print PyTorch model parameters and the related layers (modules).

    Args:
        model (nn.Module): The PyTorch model.
    """
    modules = {name: module for name, module in model.named_modules()}
    layers = {'param': [], 'layer': [], 'param_shape': [], 'param_type': []}

    for name, param in model.named_parameters():
        module_name = name.rsplit('.', 1)[0]
        module_name = modules[module_name].__class__.__name__

        layers['param'].append(name)
        layers['layer'].append(module_name)
        layers['param_shape'].append(list(param.shape))
        layers['param_type'].append(param.dtype)

    with pl.Config() as cfg:
        cfg.set_tbl_hide_dataframe_shape(True)
        cfg.set_tbl_hide_column_data_types(True)
        cfg.set_tbl_formatting('UTF8_FULL_CONDENSED')

        print(pl.DataFrame(layers))
