# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.
from typing import Union

import narwhals as nw

# import pandas as pd

_GROUP_ID = "group_id"
_EVENT = "event"
_LABEL = "label"
_LOSS = "loss"
_PREDICTION = "pred"
_ALL = "all"
_SIGN = "sign"


class Moment:
    """Generic moment.

    Our implementations of the reductions approach to fairness
    :footcite:p:`agarwal2018reductions` make use
    of :class:`Moment` objects to describe both the optimization objective
    and the fairness constraints
    imposed on the solution. This is an abstract class for all such objects.

    Read more in the :ref:`User Guide <reductions>`.
    """

    def __init__(self):
        self.data_loaded = False

    def load_data(
        self, X, y: nw.Series, *, sensitive_features: Union[nw.Series, None] = None
    ):
        """Load a set of data for use by this object.

        Parameters
        ----------
        X : array
            The feature array
        y : :class:`pandas.Series`
            The label vector
        sensitive_features : :class:`pandas.Series`
            The sensitive feature vector (default None)
        """
        assert self.data_loaded is False, "data can be loaded only once"

        if sensitive_features is not None:
            # TODO: remove this
            sensitive_features = nw.from_native(sensitive_features, series_only=True)
            assert isinstance(sensitive_features, nw.Series)
        self.X = nw.from_native(X, strict=False)
        self._y = nw.from_native(y, strict=False, allow_series=True)
        self.tags: nw.DataFrame = self._y.alias(_LABEL).to_frame()
        print(self.tags.columns)
        if sensitive_features is not None:
            self.tags = self.tags.with_columns(sensitive_features.alias(_GROUP_ID))
        self.data_loaded = True
        self._gamma_descr = None

    def _to_native_everything(self):
        # TODO: remove
        self.X = nw.to_native(self.X, strict=False)
        self._y = nw.to_native(self._y, strict=False)
        self.tags = nw.to_native(self.tags, strict=False)

    @property
    def total_samples(self):
        """Return the number of samples in the data."""
        return self.X.shape[0]

    @property
    def _y_as_series(self):
        """Return the y array as a :class:`~pandas.Series`."""
        return self._y

    def gamma(self, predictor):  # noqa: D102
        """Calculate the degree to which constraints are currently violated by the predictor."""
        raise NotImplementedError()

    def bound(self):  # noqa: D102
        """Return vector of fairness bound constraint the length of gamma."""
        raise NotImplementedError()

    def project_lambda(self, lambda_vec):  # noqa: D102
        """Return the projected lambda values."""
        raise NotImplementedError()

    def signed_weights(self, lambda_vec):  # noqa: D102
        """Return the signed weights."""
        raise NotImplementedError()

    def _moment_type(self):
        """Return the moment type, e.g., ClassificationMoment vs LossMoment."""
        return NotImplementedError()


# Ensure that Moment shows up in correct place in documentation
# when it is used as a base class
Moment.__module__ = "fairlearn.reductions"


class ClassificationMoment(Moment):
    """Moment that can be expressed as weighted classification error."""

    def _moment_type(self):
        return ClassificationMoment


# Ensure that ClassificationMoment shows up in correct place in documentation
# when it is used as a base class
ClassificationMoment.__module__ = "fairlearn.reductions"


class LossMoment(Moment):
    """Moment that can be expressed as weighted loss."""

    def __init__(self, loss):
        super().__init__()
        self.reduction_loss = loss

    def _moment_type(self):
        return LossMoment


# Ensure that LossMoment shows up in correct place in documentation
# when it is used as a base class
LossMoment.__module__ = "fairlearn.reductions"
