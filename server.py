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
_PROCESS_COUNT = 1  # multiprocessing.cpu_count() * 2
_THREAD_CONCURRENCY = 1


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)


_AUTH_HEADER_KEY = "authorization"

_PUBLIC_KEY = consts.ML_SERVER_JWT_PUBLIC_KEY
_JWT_PAYLOAD = {
    "sub": "yral-ml-server",
    "company": "gobazzinga",
}


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
            print(f"Received payload : {payload}")
            return self._abort_handler


def get_frames(video_path, num_frames=200):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("total_frames : ", total_frames)
    interval = max(total_frames // num_frames, 1)
    frames = []

    for i in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def download_video(uid, filepath):
    video_link = f"https://customer-2p3jflss4r4hmpnz.cloudflarestream.com/{uid}/downloads/default.mp4"
    downloadFile(video_link, filepath)


def downloadFile(downloadlink, filePath):
    with requests.get(downloadlink, stream=True) as r:
        r.raise_for_status()

        print("Downloading to", os.path.abspath(filePath))
        with open(filePath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
        f.close()


class MLServer(ml_server_pb2_grpc.MLServerServicer):
    def predict(self, request, context):
        # return ml_server_pb2.VideoEmbedResponse(result=[1.0])
        video_id = request.video_id
        filename = f"{video_id}.mp4"
        random_str = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=7)
        )
        file_path = f"/tmp/{random_str}_{filename}"
        download_video(video_id, file_path)
        frames = get_frames(file_path, 100)
        print("Number of frames extracted:", len(frames), end="\n\n")
        frame_images = [Image.fromarray(frame) for frame in frames]
        # create List[Tensor] using preprocess
        preprocess_list: List = [preprocess(frame) for frame in frame_images]
        # Preprocess all frames in the batch and move them to the device
        preprocessed_images = torch.stack(preprocess_list).to(device)

        # Pass the batch of preprocessed frames through the model to obtain embeddings
        with torch.no_grad():
            image_features_batch = model.encode_image(preprocessed_images)

        # Convert the embeddings to a numpy array
        embeddings_array = image_features_batch.cpu().numpy()

        # Calculate the average of all embeddings
        average_embedding = np.mean(embeddings_array, axis=0)
        res = average_embedding.tolist()

        # print(average_embedding)
        # print("average_embedding shape:", average_embedding.shape, res)

        # delete the video file
        os.remove(file_path)

        return ml_server_pb2.VideoEmbedResponse(result=average_embedding)


def _wait_forever(server):
    try:
        while True:
            time.sleep(_ONE_DAY.total_seconds())
    except KeyboardInterrupt:
        server.stop(None)


def _run_server(bind_address):
    """Start a server in a subprocess."""
    _LOGGER.info("Starting new server.")
    options = (("grpc.so_reuseport", 1),)

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
    """Find and reserve a port for all subprocesses to use."""
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 0:
        raise RuntimeError("Failed to set SO_REUSEPORT.")
    sock.bind(("", 0))
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def main():
    _LOGGER.info("Using spawn instead of forkserver.")
    multiprocessing.set_start_method("spawn", force=True)
    bind_address = "0.0.0.0:50051"
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
