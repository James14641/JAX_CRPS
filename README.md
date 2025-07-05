# JAX_CRPS
Autodifferentiable implementation of the Continous Rank Probability Score CRPS in JAX.

## 📦 Installation
To install this package locally in editable mode:

```bash
pip install JAX-CRPS==0.1.0
```

You can then import the package in Python:

from JAX_CRPS.crps import jax_crps

### Example usage
```
import jax.numpy as jnp
observation = jnp.array([1.0, 2.0, 3.0])  # shape (3,)
forecast = jnp.array([
    [0.8, 1.1, 1.0, 1.2, 0.9],  # forecasts for location 1
    [1.8, 2.2, 2.0, 1.9, 2.1],  # location 2
    [2.9, 3.1, 3.0, 3.2, 2.8],  # location 3
]).T  # shape (3, 5) → transpose to (D=3, E=5)
# Note: jax_crps expects forecast shape (..., D, E)
crps_values = jax_crps(observation, forecast, ensemble_axis=-1)
print("CRPS at each location:", crps_values)
crps_mean_value = jax_crps_mean(observation, forecast)
print(crps_mean_value)
```

### dependencies
jax
https://docs.jax.dev/en/latest/quickstart.html