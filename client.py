from __future__ import print_function

import logging
import os
import grpc
import ml_server_pb2
import ml_server_pb2_grpc
import contextlib


# def read_certificate(file_path):
#     with open(file_path, "rb") as f:
#         return f.read()


# def _load_credential_from_file(filepath):
#     with open(filepath, "rb") as f:
#         return f.read()


# # ROOT_CERTIFICATE = read_certificate(certifi.where())
# ROOT_CERTIFICATE = _load_credential_from_file(
#     "/Users/komalsai/Library/Application Support/mkcert/rootCA.pem"
# )


@contextlib.contextmanager
def create_client_channel(addr):
    # Call credential object will be invoked for every single RPC
    call_credentials = grpc.access_token_call_credentials(
        "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ5cmFsLW1sLXNlcnZlciIsImNvbXBhbnkiOiJnb2JhenppbmdhIn0.ObgL1-x_e2yEheD3Xj-nyh6NS2881f2qFyTMA-8ai0XW3bJjMkfqMpZBokubJv2RKhsUGoRdfwukZNobm2xdCA"
    )
    # Channel credential will be valid for the entire channel
    channel_credential = grpc.ssl_channel_credentials()
    # Combining channel credentials and call credentials together
    composite_credentials = grpc.composite_channel_credentials(
        channel_credential,
        call_credentials,
    )
    channel = grpc.secure_channel(addr, channel_credential)
    yield channel


def run():
    print("Will try to greet world ...")
    # with grpc.insecure_channel("localhost:50051") as channel:
    #     stub = prime_pb2_grpc.PrimeCheckerStub(channel)
    #     response = stub.check(prime_pb2.PrimeCandidate(candidate=11))

    with create_client_channel("yral-ml-server-test.fly.dev:443") as channel:
        stub = ml_server_pb2_grpc.MLServerStub(channel)
        response = stub.predict(
            ml_server_pb2.VideoEmbedRequest(video_id="d0ada0daa01a419caa1ed2b954531a1e")
        )

    print("Greeter client received: ", response.result)


if __name__ == "__main__":
    logging.basicConfig()
    run()
