import argparse
import os
import pickle
import struct
import sys
import traceback
from pathlib import Path
from typing import Any, BinaryIO

_MESSAGE_HEADER = struct.Struct('!Q')


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError('Input stream closed unexpectedly.')
        chunks.extend(chunk)
    return bytes(chunks)


def _read_message(stream: BinaryIO) -> Any | None:
    header = stream.read(_MESSAGE_HEADER.size)
    if not header:
        return None
    if len(header) < _MESSAGE_HEADER.size:
        raise EOFError('Received a truncated message header.')
    (size,) = _MESSAGE_HEADER.unpack(header)
    return pickle.loads(_read_exact(stream, size))


def _write_message(stream: BinaryIO, payload: Any) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_MESSAGE_HEADER.pack(len(data)))
    stream.write(data)
    stream.flush()


def _reserve_stdout_for_protocol() -> BinaryIO:
    protocol_stream = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    return protocol_stream


def _build_predictor(
    repo_path: Path,
    model_path: Path,
    inference_config_path: Path,
    device: str,
) -> Any:
    os.chdir(repo_path)
    repo = str(repo_path)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    import torch
    from inference.predictor import LimiXPredictor

    return LimiXPredictor(
        device=torch.device(device),
        model_path=str(model_path),
        inference_config=str(inference_config_path),
    )


def _serve(protocol_stream: BinaryIO, args: argparse.Namespace) -> int:
    predictor = _build_predictor(
        repo_path=args.repo_path,
        model_path=args.model_path,
        inference_config_path=args.inference_config,
        device=args.device,
    )
    X_train = None
    y_train = None

    _write_message(protocol_stream, {'status': 'ready'})

    while True:
        request = _read_message(sys.stdin.buffer)
        if request is None:
            return 0

        command = request.get('command')
        payload = request.get('payload')
        try:
            if command == 'fit':
                X_train, y_train = payload
                response = {'status': 'ok'}
            elif command == 'predict':
                if X_train is None or y_train is None:
                    raise RuntimeError('Worker received predict before fit.')
                y_score = predictor.predict(X_train, y_train, payload)
                response = {'status': 'ok', 'result': y_score}
            elif command == 'shutdown':
                _write_message(protocol_stream, {'status': 'ok'})
                return 0
            else:
                raise ValueError(f'Unsupported command: {command!r}')
        except Exception as error:
            response = {
                'status': 'error',
                'error': str(error),
                'traceback': traceback.format_exc(),
            }

        _write_message(protocol_stream, response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-path', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--inference-config', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    protocol_stream = _reserve_stdout_for_protocol()
    try:
        return _serve(protocol_stream, args)
    except Exception as error:
        _write_message(
            protocol_stream,
            {
                'status': 'error',
                'error': str(error),
                'traceback': traceback.format_exc(),
            },
        )
        return 1
    finally:
        protocol_stream.close()


if __name__ == '__main__':
    raise SystemExit(main())
