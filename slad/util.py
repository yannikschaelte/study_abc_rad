"""Utility functions."""

import argparse


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    host = args.host
    port = args.port

    return host, port
