#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch
from opacus import PrivacyEngine
from opacus.schedulers import ExponentialNoise, LambdaNoise, StepNoise
from opacus.utils.adaptive_clipping.adaptive_clipping_utils import (
    PrivacyEngineAdaptiveClipping,
)
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class NoiseSchedulerTest(unittest.TestCase):
    def setUp(self):
        n_data, dim = 100, 10
        data = torch.randn(n_data, dim)
        model = nn.Linear(10, 10)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        data_loader = DataLoader(TensorDataset(data), batch_size=10)
        self.engine = PrivacyEngine()

        self.module, self.optimizer, self.data_loader = self.engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

    def test_exponential_scheduler(self):
        gamma = 0.99
        scheduler = ExponentialNoise(self.optimizer, gamma=gamma)

        self.assertEqual(self.optimizer.noise_multiplier, 1.0)
        scheduler.step()
        self.assertEqual(self.optimizer.noise_multiplier, gamma)

    def test_step_scheduler(self):
        gamma = 0.1
        step_size = 2
        scheduler = StepNoise(self.optimizer, step_size=step_size, gamma=gamma)

        self.assertEqual(self.optimizer.noise_multiplier, 1.0)
        scheduler.step()
        self.assertEqual(self.optimizer.noise_multiplier, 1.0)
        scheduler.step()
        self.assertEqual(self.optimizer.noise_multiplier, gamma)
        scheduler.step()
        self.assertEqual(self.optimizer.noise_multiplier, gamma)
        scheduler.step()
        self.assertEqual(self.optimizer.noise_multiplier, gamma**2)

    def test_lambda_scheduler(self):
        def noise_lambda(epoch):
            return 1 - epoch / 10

        scheduler = LambdaNoise(self.optimizer, noise_lambda=noise_lambda)

        self.assertEqual(self.optimizer.noise_multiplier, 1.0)
        scheduler.step()
        self.assertEqual(self.optimizer.noise_multiplier, noise_lambda(1))

    @unittest.expectedFailure
    def test_adaptive_ghost_clipping(self):
        # the criterion copies the noise multiplier when it's built, but the
        # accountant charges the scheduled optimizer.noise_multiplier
        n_data, dim = 32, 10
        data = torch.randn(n_data, dim)
        labels = torch.randint(0, 10, (n_data,))
        model = nn.Linear(10, 10)
        optimizer = optim.SGD(model.parameters(), lr=0.0)
        data_loader = DataLoader(TensorDataset(data, labels), batch_size=n_data)

        module, optimizer, criterion, _ = PrivacyEngineAdaptiveClipping().make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            criterion=nn.CrossEntropyLoss(reduction="mean"),
            noise_multiplier=0.5,
            max_grad_norm=1.0,
            poisson_sampling=False,
            grad_sample_mode="ghost",
        )
        scheduler = ExponentialNoise(optimizer, gamma=2.0)
        scheduler.step()

        optimizer.zero_grad()
        loss = criterion(module(data), labels)
        loss.backward()
        optimizer.step()

        # Theorem 1 of https://arxiv.org/pdf/1905.03871, where the criterion uses
        # batch_size / 20 as the std of the unclipped-count noise
        sigma = optimizer.noise_multiplier
        sigma_b = n_data / 20.0
        expected = (sigma**-2 - (2.0 * sigma_b) ** -2) ** -0.5
        self.assertAlmostEqual(optimizer._adjusted_noise_multiplier, expected, places=5)
