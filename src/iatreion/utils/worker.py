import atexit
import pickle
import struct
import subprocess
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

_MESSAGE_HEADER = struct.Struct('!Q')


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError('Worker closed the stream unexpectedly.')
        chunks.extend(chunk)
    return bytes(chunks)


def read_message(stream: BinaryIO) -> Any:
    header = stream.read(_MESSAGE_HEADER.size)
    if not header:
        raise EOFError('Worker closed the stream unexpectedly.')
    if len(header) < _MESSAGE_HEADER.size:
        raise EOFError('Worker returned a truncated message header.')
    (size,) = _MESSAGE_HEADER.unpack(header)
    return pickle.loads(_read_exact(stream, size))


def write_message(stream: BinaryIO, payload: Any) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_MESSAGE_HEADER.pack(len(data)))
    stream.write(data)
    stream.flush()


class _StderrTail:
    def __init__(self, stream: BinaryIO, *, max_lines: int = 200) -> None:
        self._stream = stream
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._thread = threading.Thread(
            target=self._drain,
            name='worker-stderr-tail',
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def dump(self) -> str:
        return '\n'.join(self._lines)

    def _drain(self) -> None:
        for raw_line in iter(self._stream.readline, b''):
            line = raw_line.decode('utf-8', errors='replace').rstrip()
            self._lines.append(line)


class SubprocessWorker:
    def __init__(
        self,
        argv: Sequence[str | Path],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        name: str = 'Worker',
        ready_status: str | None = None,
        shutdown_request: Any = None,
        shutdown_status: str = 'ok',
    ) -> None:
        self._argv = [str(arg) for arg in argv]
        self._cwd = None if cwd is None else str(cwd)
        self._env = None if env is None else dict(env)
        self._name = name
        self._ready_status = ready_status
        self._shutdown_request = shutdown_request
        self._shutdown_status = shutdown_status
        self._process: subprocess.Popen[bytes] | None = None
        self._stdin: BinaryIO | None = None
        self._stdout: BinaryIO | None = None
        self._stderr: BinaryIO | None = None
        self._stderr_tail: _StderrTail | None = None
        atexit.register(self.close)

    def ensure_started(self) -> bool:
        process = self._process
        if process is not None and process.poll() is None:
            return False
        self.close()
        try:
            self._start()
        except Exception:
            self.close()
            raise
        return True

    def request(self, payload: Any) -> Any:
        if self._stdin is None or self._stdout is None:
            raise RuntimeError(f'{self._name} is not running.')
        try:
            write_message(self._stdin, payload)
            return read_message(self._stdout)
        except (BrokenPipeError, EOFError, OSError, pickle.UnpicklingError) as error:
            raise RuntimeError(self.format_stream_error()) from error

    def close(self) -> None:
        process = self._process
        if process is None:
            return

        try:
            if (
                process.poll() is None
                and self._shutdown_request is not None
                and self._stdin is not None
            ):
                try:
                    write_message(self._stdin, self._shutdown_request)
                    response = read_message(self._stdout)
                    status = (
                        response.get('status') if isinstance(response, dict) else None
                    )
                    if status != self._shutdown_status:
                        raise RuntimeError(self.format_response_error(response))
                except Exception:
                    process.kill()

            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        finally:
            if self._stderr_tail is not None:
                self._stderr_tail.join(timeout=1)
            if self._stdin is not None:
                self._stdin.close()
            if self._stdout is not None:
                self._stdout.close()
            if self._stderr is not None:
                self._stderr.close()

            self._process = None
            self._stdin = None
            self._stdout = None
            self._stderr = None
            self._stderr_tail = None

    def format_response_error(self, response: Any) -> str:
        details = [f'{self._name} request failed.']
        if isinstance(response, dict):
            error = response.get('error')
            trace = response.get('traceback')
            if error:
                details.append(str(error))
            if trace:
                details.append(str(trace))
        else:
            details.append(f'Unexpected response: {response!r}')

        stderr_tail = self.stderr_tail
        if stderr_tail:
            details.append(f'Worker stderr:\n{stderr_tail}')
        return '\n'.join(details)

    def format_stream_error(self) -> str:
        details = [f'{self._name} exited unexpectedly.']
        process = self._process
        if process is not None and process.poll() is not None:
            details.append(f'Exit code: {process.returncode}')
        stderr_tail = self.stderr_tail
        if stderr_tail:
            details.append(f'Worker stderr:\n{stderr_tail}')
        return '\n'.join(details)

    @property
    def stderr_tail(self) -> str:
        if self._stderr_tail is None:
            return ''
        return self._stderr_tail.dump()

    def _start(self) -> None:
        process = subprocess.Popen(
            self._argv,
            cwd=self._cwd,
            env=self._env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            process.wait()
            raise RuntimeError(f'Failed to attach to {self._name} streams.')

        self._process = process
        self._stdin = process.stdin
        self._stdout = process.stdout
        self._stderr = process.stderr
        self._stderr_tail = _StderrTail(process.stderr)
        self._stderr_tail.start()

        if self._ready_status is None:
            return
        try:
            response = read_message(process.stdout)
        except (EOFError, OSError, pickle.UnpicklingError) as error:
            raise RuntimeError(self.format_stream_error()) from error

        status = response.get('status') if isinstance(response, dict) else None
        if status != self._ready_status:
            raise RuntimeError(self.format_response_error(response))
