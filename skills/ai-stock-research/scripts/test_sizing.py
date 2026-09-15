"""Offline checks for research sizing; no provider or broker calls."""
import unittest
from pydantic import ValidationError
from plugin_review import CandidateSizing


class SizingTests(unittest.TestCase):
    def make(self, **changes):
        values = dict(target_weight=0.25, portfolio_loss_budget=0.05,
                      downside_fraction=0.20, basis="Explicit scenario and budget",
                      evidence_refs=["selection"])
        return CandidateSizing(**(values | changes))

    def test_budget_supported_target(self):
        self.assertEqual(self.make().target_weight, 0.25)

    def test_over_budget_rejected(self):
        with self.assertRaises(ValidationError):
            self.make(target_weight=0.50)

    def test_missing_budget_cannot_inherit_weight(self):
        with self.assertRaises(ValidationError):
            self.make(portfolio_loss_budget=None)

    def test_unknown_is_not_zero(self):
        self.assertIsNone(self.make(target_weight=None, portfolio_loss_budget=None,
                                    downside_fraction=None).target_weight)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValidationError):
            self.make(downside_fraction=float("nan"))


if __name__ == "__main__":
    unittest.main()
