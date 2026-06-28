"""Preprocessing kernel — resample / N4 / z-score / grid-fit primitives, dataset-agnostic.
Callers inject the dataset adapter's loader; core never imports the data layer.
"""
