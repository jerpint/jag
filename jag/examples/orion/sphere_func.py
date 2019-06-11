#!/usr/bin/env python
import argparse
from orion.client import report_results


def sphere_func_2d(x, y):
    return x*x + y*y


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-x', help='the x coordinate', type=float)
    parser.add_argument('-y', help='the y coordinate', type=float)
    args = parser.parse_args()
    loss = sphere_func_2d(args.x, args.y)
    report_results([dict(
        name='test_error_rate',
        type='objective',
        value=loss)])
