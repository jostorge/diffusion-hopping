import unittest

import torch

from diffusion_hopping.model.diffusion.schedules import (
    CosineBetaSchedule,
    LinearBetaSchedule,
)


class TestSchedules(unittest.TestCase):
    def test_schedule_correct(self):
        T = 500
        schedule = LinearBetaSchedule(T)

        self.assertEqual(schedule.alpha[0], 0.9999)
        self.assertEqual(schedule.alpha[T - 1], 0.9800)
        self.assertAlmostEqual(schedule.alpha[T // 2].item(), 0.98993)
        self.assertEqual(schedule.beta[0], 0.0001)
        self.assertEqual(schedule.beta[T - 1], 0.02)

    def test_cosine_schedule_clipped(self):
        schedule = CosineBetaSchedule(500, beta_min=0.0001, beta_max=0.02)
        self.assertTrue(schedule.alpha.min() > 0)
        self.assertTrue(schedule.alpha.max() < 1)

    def test_cosine_schedule_beta(self):
        schedule = CosineBetaSchedule(500, beta_min=0.0001, beta_max=0.02)
        self.assertTrue(torch.allclose(schedule.beta, 1.0 - schedule.alpha, atol=1e-5))

    def test_schedule_length1(self):
        T = 500
        schedule = LinearBetaSchedule(T)
        self.assertEqual(len(schedule.alpha), T)
        self.assertEqual(len(schedule.beta), T)
        self.assertEqual(len(schedule.alpha_bar), T)
        self.assertEqual(len(schedule.sqrt_alpha), T)
        self.assertEqual(len(schedule.sqrt_recip_alpha), T)
        self.assertEqual(len(schedule.sqrt_alpha_bar), T)
        self.assertEqual(len(schedule.sqrt_one_minus_alpha_bar), T)
        self.assertEqual(len(schedule.posterior_variance), T)

    def test_schedule_length2(self):
        T = 250
        schedule = LinearBetaSchedule(T)
        self.assertEqual(len(schedule.alpha), T)
        self.assertEqual(len(schedule.beta), T)
        self.assertEqual(len(schedule.alpha_bar), T)
        self.assertEqual(len(schedule.sqrt_alpha), T)
        self.assertEqual(len(schedule.sqrt_recip_alpha), T)
        self.assertEqual(len(schedule.sqrt_alpha_bar), T)
        self.assertEqual(len(schedule.sqrt_one_minus_alpha_bar), T)
        self.assertEqual(len(schedule.posterior_variance), T)
