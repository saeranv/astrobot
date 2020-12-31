import forecast
import numpy as np


def test_draw_samples():
    """Test draw_samples"""

    mu = np.ones(10) * 5
    x = forecast.draw_samples(mu)
