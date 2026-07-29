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
import torch.nn as nn
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


class AccountingConsistencyTest(unittest.TestCase):
    """
    The privacy guarantee is only valid if the sample rate charged to the
    accountant is the same rate the data loader's Poisson sampler actually
    executes.

    These rates are stored in the accountant and the batch sampler.
    Deviations can cause bugs in epsilon reporting.
    Issue 813 caught over reporting. Patch 814 could under reported.
    """

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.data_size = 128
        self.batch_size = 32
        self.dimension = 4

        x = torch.randn(self.data_size, self.dimension)
        y = torch.randint(low=0, high=2, size=(self.data_size,))

        self.dataset = TensorDataset(x, y)

    def _make_private(self, privacy_engine, **kwargs):
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size)
        model = nn.Linear(self.dimension, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        return privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            max_grad_norm=1.0,
            **kwargs,
        )

    def _train_one_epoch(self, model, optimizer, data_loader):
        criterion = nn.CrossEntropyLoss()
        steps = 0
        for x, y in data_loader:
            self.assertGreater(x.shape[0], 0, "Empty batch: reseed the test")
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            steps += 1
        return steps

    def test_history_rate_matches_executed_rate(self) -> None:
        privacy_engine = PrivacyEngine()
        model, optimizer, dp_loader = self._make_private(
            privacy_engine, noise_multiplier=1.0
        )
        self._train_one_epoch(model, optimizer, dp_loader)

        executed_rate = dp_loader.batch_sampler.sample_rate
        charged_rates = {
            sample_rate for _, sample_rate, _ in privacy_engine.accountant.history
        }
        self.assertEqual(charged_rates, {executed_rate})
        self.assertEqual(dp_loader.sample_rate, executed_rate)

    def test_charged_rate_matches_observed_inclusion_frequency(self) -> None:
        privacy_engine = PrivacyEngine()
        model, optimizer, dp_loader = self._make_private(
            privacy_engine, noise_multiplier=1.0
        )
        self._train_one_epoch(model, optimizer, dp_loader)
        charged_rate = privacy_engine.accountant.history[-1][1]

        n_epochs = 25
        drawn = sum(x.shape[0] for _ in range(n_epochs) for x, _ in dp_loader)
        observed_rate = drawn / (n_epochs * self.data_size * len(dp_loader))
        self.assertAlmostEqual(observed_rate, charged_rate, delta=0.15 * charged_rate)

    def test_partial_coverage_sampler_rejected_or_consistent(self) -> None:
        sampler = WeightedRandomSampler(
            weights=torch.ones(self.data_size), num_samples=8, replacement=True
        )
        data_loader = DataLoader(self.dataset, batch_size=4, sampler=sampler)
        model = nn.Linear(self.dimension, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        privacy_engine = PrivacyEngine()
        try:
            model, optimizer, dp_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
            )
        except ValueError:
            return

        self._train_one_epoch(model, optimizer, dp_loader)
        executed_rate = dp_loader.batch_sampler.sample_rate
        for _, charged_rate, _ in privacy_engine.accountant.history:
            self.assertEqual(charged_rate, executed_rate)

    def test_partial_coverage_sampler_epsilon_not_under_reported(self) -> None:
        """Catch under reporting if rate is set to batch_size / len(dataset)"""
        noise_multiplier = 1.0
        delta = 1e-5
        sampler = WeightedRandomSampler(
            weights=torch.ones(self.data_size), num_samples=8, replacement=True
        )
        data_loader = DataLoader(self.dataset, batch_size=4, sampler=sampler)
        model = nn.Linear(self.dimension, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        privacy_engine = PrivacyEngine()
        try:
            model, optimizer, dp_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=1.0,
            )
        except ValueError:
            return

        steps = self._train_one_epoch(model, optimizer, dp_loader)
        reported_epsilon = privacy_engine.get_epsilon(delta=delta)

        executed_rate = dp_loader.batch_sampler.sample_rate
        shadow_accountant = type(privacy_engine.accountant)()
        for _ in range(steps):
            shadow_accountant.step(
                noise_multiplier=noise_multiplier, sample_rate=executed_rate
            )
        spent_epsilon = shadow_accountant.get_epsilon(delta=delta)
        self.assertGreaterEqual(reported_epsilon, 0.99 * spent_epsilon)

    def test_epsilon_matches_target_after_training(self) -> None:
        target_epsilon = 5.0
        target_delta = 1e-5
        privacy_engine = PrivacyEngine()
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size)
        model = nn.Linear(self.dimension, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model, optimizer, dp_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            epochs=1,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=1.0,
        )
        steps = self._train_one_epoch(model, optimizer, dp_loader)

        self.assertEqual(steps, len(dp_loader))
        epsilon = privacy_engine.get_epsilon(delta=target_delta)
        self.assertLessEqual(epsilon, target_epsilon)
        self.assertGreater(epsilon, 0.5 * target_epsilon)
