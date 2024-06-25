# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

from concurrent import futures
import contextlib
import datetime
import logging
import math
import multiprocessing
import socket
import sys
import time
import os
from typing import Any, List


import torch
import clip

import cv2

import requests

from PIL import Image

import numpy as np

import string
import random

import jwt

import grpc
import ml_server_pb2
import ml_server_pb2_grpc
import consts

_LOGGER = logging.getLogger(__name__)

_ONE_DAY = datetime.timedelta(days=1)
_PROCESS_COUNT = multiprocessing.cpu_count()
_THREAD_CONCURRENCY = 1


_AUTH_HEADER_KEY = "authorization"

_PUBLIC_KEY = consts.ML_SERVER_JWT_PUBLIC_KEY
_JWT_PAYLOAD = {
    "sub": "yral-ml-server",
    "company": "gobazzinga",
}

_RANDOM_NUM = random.randint(1, 1000000)


class SignatureValidationInterceptor(grpc.ServerInterceptor):
    def __init__(self):
        def abort(ignored_request, context):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid signature")

        self._abort_handler = grpc.unary_unary_rpc_method_handler(abort)

    def intercept_service(self, continuation, handler_call_details):
        metadata_dict = dict(handler_call_details.invocation_metadata)
        token = metadata_dict[_AUTH_HEADER_KEY].split()[1]
        payload = jwt.decode(
            token,
            _PUBLIC_KEY,
            algorithms=["EdDSA"],
        )

        if payload == _JWT_PAYLOAD:
            return continuation(handler_call_details)
        else:
            print(f"Received payload: {payload}")
            return self._abort_handler


class MLServer(ml_server_pb2_grpc.MLServerServicer):
    def predict(self, request, context):

        # wait for 5 secs
        time.sleep(5)

        # _LOGGER.info(multiprocessing.current_process())

        return ml_server_pb2.VideoEmbedResponse(result=[os.getpid(), _RANDOM_NUM])


def _wait_forever(server):
    try:
        while True:
            time.sleep(_ONE_DAY.total_seconds())
    except KeyboardInterrupt:
        server.stop(None)


def _run_server(bind_address):
    """Start a server in a subprocess."""
    _LOGGER.info("Starting new server.")
    options = [
        ("grpc.max_send_message_length", -1),
        ("grpc.max_receive_message_length", -1),
        ("grpc.so_reuseport", 1),
        ("grpc.use_local_subchannel_pool", 1),
    ]

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=_THREAD_CONCURRENCY,
        ),
        interceptors=(SignatureValidationInterceptor(),),
        options=options,
    )
    ml_server_pb2_grpc.add_MLServerServicer_to_server(MLServer(), server)
    server.add_insecure_port(bind_address)
    server.start()
    _wait_forever(server)


@contextlib.contextmanager
def _reserve_port():
    """Find and reserve a port for all subprocesses to use"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 0:
        raise RuntimeError("Failed to set SO_REUSEPORT.")
    sock.bind(("0.0.0.0", 50051))
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def main():
    # _LOGGER.info("Using spawn instead of forkserver.")
    multiprocessing.set_start_method("spawn", force=True)
    with _reserve_port() as port:
        bind_address = f"0.0.0.0:{port}"
        _LOGGER.info("Binding to '%s'", bind_address)
        sys.stdout.flush()
        workers = []
        for _ in range(_PROCESS_COUNT):
            # NOTE: It is imperative that the worker subprocesses be forked before
            # any gRPC servers start up. See
            # https://github.com/grpc/grpc/issues/16001 for more details.
            worker = multiprocessing.Process(target=_run_server, args=(bind_address,))
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join()


if __name__ == "__main__":
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[PID %(process)d] %(message)s")
    handler.setFormatter(formatter)
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    main()
